"""K-fold обучение нейросетей из nn_solution.py с лучшим конфигом из nn_config.json.

Параметры main:
  epoch_coefs = (1.0, 1.1, ..., 2.0) — коэффициенты >= 1.0 для чекпоинтов по эпохам.

Эпохи для чекпоинтов считаются от ЛУЧШЕЙ НА ТЕКУЩИЙ МОМЕНТ эпохи (best_epoch из json не используется):
все k сетей обучаются синхронно (эпоха за эпохой), после каждой эпохи считается CV ROC AUC по OOF всех
фолдов. Пока максимум не обновляется — чекпоинты стоят на round(coef * best_epoch); при новом максимуме
best_epoch обновляется и чекпоинты пересчитываются (см. cv_checkpoints.py). На чекпоинте считаются
предсказания теста всеми k сетями и запоминаются веса. Обучение заканчивается, когда сохранены все
чекпоинты для итоговой best_epoch и пройден горизонт
max(best_epoch * max(max(epoch_coefs), 1.5), best_epoch + min_patience).

На каждом фолде: Preprocessing (дефолт) -> JointDistributionFilter(train, test) -> NNFeatures -> сеть;
ранней остановки нет, ReduceLROnPlateau по val loss с lr_scheduler_patience из конфига.

Что сохраняется (папки лежат на уровне preprocessing.py):
  models/nn_ep{эпоха}_fold{i}.pt          — веса сети на чекпоинте (torch.save(model.state_dict()))
  models/nn_fold{i}_pipeline.joblib       — обученные Preprocessing, JDF и состояние NNFeatures фолда
  models/nn_manifest.json                 — чекпоинты, их CV AUC, конфиг сети (читает predict.py)
  oof_predictions/nn_oof_ep{эпоха}_cv{auc}.csv   — OOF трейна (RefId, IsBadBuy, nn_prediction)
  oof_predictions/nn_test_ep{эпоха}_cv{auc}.csv  — предсказания теста, усреднённые по фолдам
А в папке скрипта (nn_submission_preds): cv_curve.png, cv_history.csv, summary.csv.
"""
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from cv_checkpoints import MODELS_DIR, OOF_DIR, history_frame, plot_cv_curve, run_adaptive

try:
    from . import nn_tuning as nn  # импорт как часть пакета: import nn_solution.nn_oof_submission
except ImportError:
    import nn_tuning as nn  # запуск скриптом из папки, где лежит nn_solution.py

from preprocessing import JointDistributionFilter

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "nn_config.json"
OUT_DIR = BASE_DIR / "nn_submission_preds"  # отчёты: график, история, summary

K_FOLDS = 5
EPOCH_COEFS = (1.0, )
MAX_EPOCHS = 300   # предохранитель
MIN_PATIENCE = 10  # минимум эпох после лучшей до остановки
METADATA_KEYS = {
    "best_tuning_auc",
    "best_epoch",
    "n_trials",
    "max_epochs",
    "early_stop_patience",
    "lr_scheduler_patience",
}


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    model_cfg = {k: v for k, v in cfg.items() if k not in METADATA_KEYS}
    scheduler_patience = int(cfg.get("lr_scheduler_patience", nn.LR_SCHEDULER_PATIENCE))
    return model_cfg, scheduler_patience, cfg.get("best_epoch")


def check_versions():
    """Падает сразу и понятно, если рядом лежат старые версии nn_solution.py / cv_checkpoints.py."""
    problems = []
    if "prep" not in getattr(nn.FoldData, "__dataclass_fields__", {}):
        problems.append("nn_solution.py устарел: у FoldData нет поля prep (и prepare_fold не передаёт prep=prep)")
    params = inspect.signature(run_adaptive).parameters
    for name in ("min_patience", "extra_snapshot"):
        if name not in params:
            problems.append(f"cv_checkpoints.py устарел: у run_adaptive нет параметра {name}")
    if problems:
        raise RuntimeError("; ".join(problems) + ". Обнови файлы и перезапусти ядро Jupyter.")


class FoldTrainer:
    """Один фолд: признаки, сеть, оптимизатор; обучается по эпохе за вызов."""

    def __init__(self, model_cfg, fold_idx, X, y, X_test, scheduler_patience):
        self.va_idx = fold_idx[1]

        fold = nn.prepare_fold(X, y, X_test, fold_idx)  # Preprocessing (fit на train-части фолда)
        jdf = JointDistributionFilter(fold.X_tr, fold.X_te)
        tr_f, va_f, te_f = (jdf.transform(d) for d in (fold.X_tr, fold.X_va, fold.X_te))

        builder = nn.NNFeatures().fit(tr_f)
        train_spec = builder.transform(tr_f)
        va_spec = builder.transform(va_f)
        test_spec = builder.transform(te_f)

        # всё, что нужно для предсказания на сырых данных; NNFeatures сохраняем как dict состояния,
        # чтобы файл не зависел от того, под каким именем импортирован модуль nn_solution
        self.pipeline = {
            "prep": fold.prep,
            "jdf": jdf,
            "builder_state": dict(builder.__dict__),
        }

        train_batch, train_target = nn.to_device(train_spec, fold.y_tr)
        self.va_batch, self.va_target = nn.to_device(va_spec, fold.y_va)
        self.test_batch = nn.to_device(test_spec)
        self.y_va = fold.y_va
        self.n_features = (train_spec["text_dim"], len(train_spec["embed_cols"]),
                           train_spec["ohe_dim"], train_spec["cont_dim"])

        torch.manual_seed(nn.RANDOM_STATE)
        self.model = nn.build_model(model_cfg, train_spec)
        self.cat_cols = list(self.model.cat_cols)

        n = train_batch["text"].shape[0]
        cat_tensor = (
            torch.stack([train_batch["cat_ids"][c] for c in self.cat_cols], dim=1)
            if self.cat_cols
            else torch.zeros((n, 0), dtype=torch.long, device=nn.DEVICE)
        )
        dataset = TensorDataset(
            train_batch["text"], cat_tensor, train_batch["ohe"], train_batch["cont"], train_target
        )
        self.n_train = len(dataset)
        gen = torch.Generator()
        gen.manual_seed(nn.RANDOM_STATE)  # свой генератор: порядок батчей не зависит от других фолдов
        self.loader = DataLoader(
            dataset, batch_size=int(model_cfg["batch_size"]), shuffle=True, drop_last=True, generator=gen
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=model_cfg["lr"], weight_decay=model_cfg.get("weight_decay", 0.0)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=scheduler_patience
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.val_proba = None

    def train_epoch(self):
        self.model.train()
        for text, cat_ids, ohe, cont, yb in self.loader:
            self.optimizer.zero_grad()
            batch = {
                "text": text,
                "cat_ids": {c: cat_ids[:, i] for i, c in enumerate(self.cat_cols)},
                "ohe": ohe,
                "cont": cont,
            }
            loss = self.loss_fn(self.model(batch), yb)
            loss.backward()
            self.optimizer.step()
        val_loss, self.val_proba = nn.evaluate(self.model, self.va_batch, self.va_target)
        self.scheduler.step(val_loss)

    def test_proba(self):
        return nn.evaluate(self.model, self.test_batch, None)[1]

    def state_dict_cpu(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


def main(epoch_coefs=EPOCH_COEFS, k_folds=K_FOLDS, max_epochs=MAX_EPOCHS, min_patience=MIN_PATIENCE,
         config_path=CONFIG_PATH, out_dir=OUT_DIR, models_dir=MODELS_DIR, oof_dir=OOF_DIR):
    check_versions()
    if min(epoch_coefs) < 1.0:
        raise ValueError("epoch_coefs должны быть >= 1.0 (чекпоинты считаются от лучшей эпохи)")

    model_cfg, scheduler_patience, tuning_best_epoch = load_config(config_path)
    print(f"config loaded from {config_path}")
    print(f"model config: {model_cfg}")
    print(f"lr_scheduler_patience={scheduler_patience}, best_epoch в конфиге (справочно): {tuning_best_epoch}")

    torch.manual_seed(nn.RANDOM_STATE)
    np.random.seed(nn.RANDOM_STATE)

    X, y, X_test, test_ref_id = nn.load_data()
    train_ref_id = pd.read_csv(nn.TRAIN_PATH)["RefId"]
    print(f"train: {X.shape}, test: {X_test.shape}, device: {nn.DEVICE}")

    out_dir, models_dir, oof_dir = Path(out_dir), Path(models_dir), Path(oof_dir)
    for d in (out_dir, models_dir, oof_dir):
        d.mkdir(parents=True, exist_ok=True)

    folds = list(
        StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=nn.RANDOM_STATE).split(X, y)
    )

    trainers = []
    for i, fold_idx in enumerate(folds):
        trainers.append(FoldTrainer(model_cfg, fold_idx, X, y, X_test, scheduler_patience))
        text_dim, emb_cat, ohe_dim, cont_dim = trainers[-1].n_features
        print(f"  fold {i + 1}/{len(folds)} подготовлен: text={text_dim} emb_cat={emb_cat} "
              f"ohe={ohe_dim} cont={cont_dim}")

    n_train, n_test = len(X), len(X_test)

    def advance(t_from, t_to):
        for _ in range(t_from, t_to):
            for tr in trainers:
                tr.train_epoch()

    def evaluate(t):
        oof = np.zeros(n_train)
        fold_aucs = []
        for tr in trainers:
            oof[tr.va_idx] = tr.val_proba
            fold_aucs.append(roc_auc_score(tr.y_va, tr.val_proba))
        return oof, fold_aucs

    def predict_test(t):  # веса меняются с эпохами, поэтому считаем прямо на чекпоинте
        out = np.zeros(n_test)
        for tr in trainers:
            out += tr.test_proba() / len(trainers)
        return out

    def snapshot_weights(t):  # веса всех фолдов на чекпоинте (устаревшие чекпоинты отбрасываются)
        return [tr.state_dict_cpu() for tr in trainers]

    tracker = run_adaptive(
        epoch_coefs, advance, evaluate, predict_test, y, eval_every=1, max_iter=max_epochs,
        log_every=1, label="epoch", lazy_test=False, min_patience=min_patience,
        extra_snapshot=snapshot_weights,
    )

    # ---------------- сохранение моделей ----------------
    pipeline_files = []
    for i, tr in enumerate(trainers):
        name = f"nn_fold{i}_pipeline.joblib"
        joblib.dump(tr.pipeline, models_dir / name, compress=3)
        pipeline_files.append(name)

    weight_files = {}
    for t, snap in sorted(tracker.saved.items()):
        weight_files[t] = []
        for i, state in enumerate(snap["extra"]):
            name = f"nn_ep{t}_fold{i}.pt"
            torch.save(state, models_dir / name)  # torch.save(model.state_dict())
            weight_files[t].append(name)
    print(f"веса и пайплайны сохранены в {models_dir}")

    # ---------------- отчёты и предсказания ----------------
    history_frame(tracker).to_csv(out_dir / "cv_history.csv", index=False)
    plot_cv_curve(
        tracker, out_dir / "cv_curve.png",
        title=f"NN: CV ROC AUC vs число эпох (макс. {tracker.best_auc:.5f} @ {tracker.best_iter})",
        xlabel="эпоха",
    )

    for t, snap in sorted(tracker.saved.items()):
        tag = f"ep{t}_cv{snap['cv_auc']:.5f}"
        pd.DataFrame(
            {"RefId": train_ref_id.values, "IsBadBuy": snap["oof"]}
        ).to_csv(oof_dir / f"nn_oof_{tag}.csv", index=False)
        pd.DataFrame(
            {"RefId": test_ref_id.values, "IsBadBuy": snap["test"]}
        ).to_csv(oof_dir / f"nn_test_{tag}.csv", index=False)

    rows, checkpoints = [], []
    for c, t in tracker.targets.items():
        if t not in tracker.saved:
            continue
        snap = tracker.saved[t]
        checkpoints.append(
            {"coef": c, "epoch": int(t), "cv_auc": float(snap["cv_auc"]), "files": weight_files[t]}
        )
        rows.append(
            {
                "coef": c,
                "epochs": t,
                "best_epoch_cv": tracker.best_iter,
                "best_cv_auc": tracker.best_auc,
                "oof_auc": snap["cv_auc"],
                "fold_auc_mean": float(np.mean(snap["fold_aucs"])),
                "fold_auc_std": float(np.std(snap["fold_aucs"])),
                "fold_aucs": [round(a, 5) for a in snap["fold_aucs"]],
            }
        )

    best = max(checkpoints, key=lambda c: c["cv_auc"])
    manifest = {
        "family": "nn",
        "k_folds": k_folds,
        "model_cfg": model_cfg,
        "best_epoch": int(tracker.best_iter),
        "pipelines": pipeline_files,
        "checkpoints": checkpoints,
        "default": {"epoch": best["epoch"], "cv_auc": best["cv_auc"]},
    }
    with open(models_dir / "nn_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"манифест: {models_dir / 'nn_manifest.json'}, по умолчанию {manifest['default']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"OOF/test предсказания: {oof_dir}; отчёты: {out_dir}")
    return summary


if __name__ == "__main__":
    main()