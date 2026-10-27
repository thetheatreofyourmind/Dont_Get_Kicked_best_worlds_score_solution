"""K-fold OOF и предсказания теста для лучших конфигов CatBoost из catboost_best_params.json.

Читает json в прежнем формате ([{"rank", "roc_auc", "trial_number", "params"}, ...]) — повторный
тюнинг не нужен.

Параметры main:
  model_ranks = (0, 1, 2, 3, 4) — какие конфиги из json брать (индекс с нуля: 0 — лучший);
  est_coefs   = (1.0, 1.1, ..., 2.0) — коэффициенты >= 1.0 для чекпоинтов по числу деревьев.

Число деревьев для чекпоинтов считается от ЛУЧШЕГО НА ТЕКУЩИЙ МОМЕНТ числа деревьев (по CV ROC AUC),
а не от iterations из json (см. cv_checkpoints.py). У CatBoost нет пошагового обучения «по одному дереву»
синхронно по фолдам, поэтому схема такая: каждый фолд обучается один раз на бюджет
iterations * budget_coef деревьев (по умолчанию budget_coef = max(est_coefs)), затем по готовым моделям
считается кривая CV AUC (staged_predict_proba с шагом eval_every) и те же адаптивные чекпоинты:
пока максимум не обновляется — чекпоинты на round(coef * best_iter), при новом максимуме пересчитываются.
Если лучшая итерация оказалась настолько поздней, что горизонт best_iter * max(est_coefs) выходит за
бюджет, скрипт предупредит о недостающих чекпоинтах — тогда увеличь budget_coef.

Что сохраняется (папки лежат на уровне preprocessing.py):
  models/catboost_m{rank}_fold{i}.cbm            — модель фолда (CatBoostClassifier.save_model)
  models/catboost_m{rank}_fold{i}_pipeline.joblib — обученные Preprocessing, JDF, список колонок и cat-колонок
  models/catboost_manifest.json                  — файлы, чекпоинты и их CV AUC (читает predict.py)
  oof_predictions/catboost_oof_m{rank}_est{деревья}_cv{auc}.csv   — OOF трейна (RefId, IsBadBuy, catboost_prediction)
  oof_predictions/catboost_test_m{rank}_est{деревья}_cv{auc}.csv  — предсказания теста, усреднённые по фолдам
А в папке скрипта (catboost_oof_preds): cv_curve_m{rank}.png, cv_history_m{rank}.csv, summary.csv.
"""
import gc
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cv_checkpoints import MODELS_DIR, OOF_DIR, history_frame, plot_cv_curve, run_adaptive

try:
    from . import catboost_tuning as ct  # импорт как часть пакета: import catboost_solution.catboost_training
except ImportError:
    import catboost_tuning as ct  # запуск скриптом из папки, где лежит catboost_tuning.py

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "catboost_oof_preds"  # отчёты: график, история, summary

K_FOLDS = 5
MODEL_RANKS = (0, )
EST_COEFS = (1.0,)
EVAL_EVERY = 20     # шаг сетки (в деревьях), на которой считается CV AUC
MIN_PATIENCE = 200  # минимум деревьев после лучшей итерации до остановки


def check_versions():
    params = inspect.signature(run_adaptive).parameters
    missing = [n for n in ("min_patience", "lazy_test") if n not in params]
    if missing:
        raise RuntimeError(
            f" у run_adaptive нет параметров {missing}. "
        )


def staged_predictions(model, pool, eval_every):
    """Предсказания по первым eval_every, 2*eval_every, ... деревьям за один проход."""
    n = model.tree_count_
    out = {}
    for i, p in enumerate(
        model.staged_predict_proba(pool, ntree_start=0, ntree_end=n, eval_period=eval_every, thread_count=-1)
    ):
        out[min((i + 1) * eval_every, n)] = np.asarray(p)[:, 1].astype(np.float32)
    return out


def train_folds(item, X, y, X_test, folds, est_coefs, eval_every, budget_coef):
    """Обучает модель на каждом фолде на бюджет деревьев; возвращает список словарей по фолдам."""
    params = ct.clean_params(item["params"])
    iterations = int(params.pop("iterations", 1000))
    n_trees = max(1, int(round(iterations * (budget_coef or max(est_coefs)))))

    base_params = dict(params)
    base_params.update(
        loss_function="Logloss", eval_metric="AUC", verbose=False,
        allow_writing_files=False, random_seed=ct.RANDOM_STATE, task_type="CPU",
    )
    for key in ("od_type", "od_wait"):  # без early stopping: число деревьев выбирается по CV
        base_params.pop(key, None)

    print(f"  iterations в json={iterations}, обучаем {n_trees} деревьев на каждом фолде")
    folds_data = []
    for k, fold_idx in enumerate(folds):
        fold = ct.prepare_fold(X, y, X_test, fold_idx)
        tr, va, te, info = ct.make_frames(fold)
        cat_cols = info["cat_cols"]

        tr_pool = Pool(tr, fold.y_tr, cat_features=cat_cols)
        va_pool = Pool(va, cat_features=cat_cols)
        te_pool = Pool(te, cat_features=cat_cols)

        model = CatBoostClassifier(iterations=n_trees, **base_params)
        model.fit(tr_pool, verbose=False)

        folds_data.append(
            {
                "model": model,
                "va_pool": va_pool,
                "te_pool": te_pool,
                "va_idx": fold_idx[1],
                "y_va": fold.y_va,
                "staged": staged_predictions(model, va_pool, eval_every),
                "pipeline": {"prep": fold.prep, "jdf": info["jdf"],
                             "columns": info["columns"], "cat_cols": cat_cols},
            }
        )
        print(f"  fold {k + 1}/{len(folds)} обучен: признаков {tr.shape[1]}, cat {len(cat_cols)}, "
              f"деревьев {model.tree_count_}")
        del fold, tr, va, te, tr_pool
        gc.collect()
    return folds_data, n_trees


def save_models(folds_data, rank, models_dir):
    """Модели — встроенным CatBoostClassifier.save_model, объекты предобработки — joblib."""
    files = []
    for i, f in enumerate(folds_data):
        model_file = f"catboost_m{rank}_fold{i}.cbm"
        pipe_file = f"catboost_m{rank}_fold{i}_pipeline.joblib"
        f["model"].save_model(str(models_dir / model_file))
        joblib.dump(f["pipeline"], models_dir / pipe_file, compress=3)
        files.append({"model": model_file, "pipeline": pipe_file})
    return files


def run_model(item, rank, X, y, X_test, folds, est_coefs, eval_every, min_patience, budget_coef,
              models_dir):
    folds_data, n_trees = train_folds(item, X, y, X_test, folds, est_coefs, eval_every, budget_coef)
    n_train, n_test = len(X), len(X_test)

    def advance(t_from, t_to):  # модели уже обучены; run_adaptive только «проигрывает» кривую CV AUC
        pass

    def val_proba(f, t):
        if t not in f["staged"]:  # чекпоинт вне сетки eval_every — считаем по запросу
            f["staged"][t] = f["model"].predict_proba(f["va_pool"], ntree_end=t)[:, 1]
        return f["staged"][t]

    def evaluate(t):
        oof = np.zeros(n_train)
        fold_aucs = []
        for f in folds_data:
            p = val_proba(f, t)
            oof[f["va_idx"]] = p
            fold_aucs.append(roc_auc_score(f["y_va"], p))
        return oof, fold_aucs

    def predict_test(t):
        out = np.zeros(n_test)
        for f in folds_data:
            out += f["model"].predict_proba(f["te_pool"], ntree_end=t)[:, 1] / len(folds_data)
        return out

    tracker = run_adaptive(
        est_coefs, advance, evaluate, predict_test, y, eval_every=eval_every, max_iter=n_trees,
        log_every=10, label="trees", lazy_test=True, min_patience=min_patience,
    )
    files = save_models(folds_data, rank, models_dir)
    print(f"  модели и пайплайны сохранены в {models_dir}")
    del folds_data
    gc.collect()
    return tracker, files, n_trees


def main(model_ranks=MODEL_RANKS, est_coefs=EST_COEFS, k_folds=K_FOLDS, eval_every=EVAL_EVERY,
         min_patience=MIN_PATIENCE, budget_coef=None, params_path=None,
         out_dir=OUT_DIR, models_dir=MODELS_DIR, oof_dir=OOF_DIR):
    check_versions()
    if min(est_coefs) < 1.0:
        raise ValueError("est_coefs должны быть >= 1.0 (чекпоинты считаются от лучшей итерации)")

    params_path = ct.find_params_file(params_path)
    with open(params_path, encoding="utf-8") as f:
        top = json.load(f)
    bad = [r for r in model_ranks if not 0 <= r < len(top)]
    if bad:
        raise IndexError(f"в {params_path} только {len(top)} конфигов, запрошены индексы {bad}")
    print(f"конфиги из {params_path}")

    np.random.seed(ct.RANDOM_STATE)
    X, y, X_test, test_ref_id = ct.load_data()
    train_ref_id = pd.read_csv(ct.TRAIN_PATH)["RefId"]
    print(f"train: {X.shape}, test: {X_test.shape}")

    folds = list(
        StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=ct.RANDOM_STATE).split(X, y)
    )
    out_dir, models_dir, oof_dir = Path(out_dir), Path(models_dir), Path(oof_dir)
    for d in (out_dir, models_dir, oof_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows, manifest_models = [], []
    for r in model_ranks:
        item = top[r]
        print(f"=== model {r}: trial {item.get('trial_number')}, tuning AUC={item.get('roc_auc', float('nan')):.5f} ===")
        tracker, files, n_trees = run_model(item, r, X, y, X_test, folds, est_coefs, eval_every,
                                            min_patience, budget_coef, models_dir)

        history_frame(tracker).to_csv(out_dir / f"cv_history_m{r}.csv", index=False)
        plot_cv_curve(
            tracker, out_dir / f"cv_curve_m{r}.png",
            title=f"CatBoost model {r}: CV ROC AUC vs число деревьев (макс. {tracker.best_auc:.5f} @ {tracker.best_iter})",
            xlabel="число деревьев",
        )

        for t, snap in sorted(tracker.saved.items()):
            tag = f"m{r}_est{t}_cv{snap['cv_auc']:.5f}"
            pd.DataFrame(
                {"RefId": train_ref_id.values, "IsBadBuy": snap["oof"]}
            ).to_csv(oof_dir / f"catboost_oof_{tag}.csv", index=False)
            pd.DataFrame(
                {"RefId": test_ref_id.values, "IsBadBuy": snap["test"]}
            ).to_csv(oof_dir / f"catboost_test_{tag}.csv", index=False)

        checkpoints = []
        for c, t in tracker.targets.items():
            if t not in tracker.saved:
                continue
            snap = tracker.saved[t]
            checkpoints.append({"coef": c, "n_iter": int(t), "cv_auc": float(snap["cv_auc"])})
            rows.append(
                {
                    "model_rank": r,
                    "trial_number": item.get("trial_number"),
                    "tuning_roc_auc": item.get("roc_auc"),
                    "iterations_json": item["params"].get("iterations"),
                    "best_iter_cv": tracker.best_iter,
                    "best_cv_auc": tracker.best_auc,
                    "coef": c,
                    "n_estimators": t,
                    "oof_auc": snap["cv_auc"],
                    "fold_auc_mean": float(np.mean(snap["fold_aucs"])),
                    "fold_auc_std": float(np.std(snap["fold_aucs"])),
                    "fold_aucs": [round(a, 5) for a in snap["fold_aucs"]],
                }
            )
        manifest_models.append(
            {
                "rank": int(r),
                "trial_number": item.get("trial_number"),
                "best_iter": int(tracker.best_iter),
                "best_cv_auc": float(tracker.best_auc),
                "n_trees": int(n_trees),  # столько деревьев в сохранённых моделях
                "checkpoints": checkpoints,
                "folds": files,
            }
        )

    # чекпоинт по умолчанию для predict.py — с лучшим CV AUC среди всех сохранённых
    best = max(
        ((m["rank"], c) for m in manifest_models for c in m["checkpoints"]),
        key=lambda rc: rc[1]["cv_auc"],
    )
    manifest = {
        "family": "catboost",
        "k_folds": k_folds,
        "models": manifest_models,
        "default": {"rank": best[0], "n_iter": best[1]["n_iter"], "cv_auc": best[1]["cv_auc"]},
    }
    with open(models_dir / "catboost_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"манифест: {models_dir / 'catboost_manifest.json'}, по умолчанию {manifest['default']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"OOF/test предсказания: {oof_dir}; отчёты: {out_dir}")
    return summary


if __name__ == "__main__":
    main()