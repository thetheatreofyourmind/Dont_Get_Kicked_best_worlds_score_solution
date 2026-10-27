"""K-fold OOF и предсказания теста для лучших конфигов XGBoost из xgb_top_params.json.

Параметры main:
  model_ranks = (0, 2, 3)  — какие конфиги из top_params брать (0 — лучший, 2 — третий, 3 — четвёртый);
  est_coefs   = (1.0, 1.1, ..., 2.0) — коэффициенты >= 1.0 для чекпоинтов по числу деревьев.

Число деревьев для чекпоинтов считается не из json, а от ЛУЧШЕГО НА ТЕКУЩИЙ МОМЕНТ числа деревьев:
все k фолдов обучаются синхронно (деревья добавляются по одному), каждые eval_every деревьев считается
CV ROC AUC по OOF всех фолдов. Пока максимум не обновляется — чекпоинты стоят на round(coef * best_iter);
при новом максимуме best_iter обновляется и чекпоинты пересчитываются (см. cv_checkpoints.py).
Обучение заканчивается, когда сохранены все чекпоинты для итоговой best_iter и пройден горизонт
max(best_iter * max(max(est_coefs), 1.5), best_iter + min_patience).

Для каждого конфига пайплайн свой: Preprocessing (его date_days / text_min_count_train) -> JDF -> FID -> XGB.
Фолды те же, что в nn_solution / lgbm_solution (StratifiedKFold, 5, shuffle, random_state=42).

Что сохраняется (папки лежат на уровне preprocessing.py):
  models/xgb_m{rank}_fold{i}.ubj             — бустер фолда (Booster.save_model, все обученные деревья)
  models/xgb_m{rank}_fold{i}_pipeline.joblib — обученные Preprocessing, JDF, FID и список колонок фолда
  models/xgb_manifest.json                   — какие файлы, какие чекпоинты и их CV AUC (читает predict.py)
  oof_predictions/xgb_oof_m{rank}_est{деревья}_cv{auc}.csv   — OOF трейна (RefId, IsBadBuy, xgb_prediction)
  oof_predictions/xgb_test_m{rank}_est{деревья}_cv{auc}.csv  — предсказания теста, усреднённые по фолдам
А в папке скрипта (xgb_oof_preds): cv_curve_m{rank}.png, cv_history_m{rank}.csv, summary.csv.
"""
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cv_checkpoints import MODELS_DIR, OOF_DIR, history_frame, plot_cv_curve, run_adaptive

try:
    from . import xgb_tuning as xs  # импорт как часть пакета: import xgb_solution.xgb_oof_submission
except ImportError:
    import xgb_tuning as xs  # запуск скриптом из папки, где лежит xgb_solution.py

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "xgb_oof_preds"  # отчёты: график, история, summary

K_FOLDS = 5
MODEL_RANKS = (0, )
EST_COEFS = (1.0, )
EVAL_EVERY = 10     # шаг сетки (в деревьях), на которой считается CV AUC
MAX_ITERS = 10000   # предохранитель на число деревьев
MIN_PATIENCE = 200  # минимум деревьев после лучшей итерации до остановки


def native_params(xgb_params):
    """Параметры из json (sklearn-имена совпадают с алиасами нативного API) + служебные."""
    params = dict(xgb_params)
    params.update(tree_method="hist", eval_metric="auc", seed=xs.RANDOM_STATE)
    return params


def build_fold_models(item, X, y, X_test, folds, importance_path):
    """Для каждого фолда: Preprocessing -> JDF -> FID -> DMatrix и пустой Booster."""
    params = native_params(item["xgb"])
    models = []
    for i, fold_idx in enumerate(folds):
        fold = xs.prepare_fold(X, y, X_test, fold_idx, item["preprocessing"])
        tr, va, te, objs = xs.make_frames(
            fold, item["jdf"], item["fid"], importance_path, return_objects=True
        )
        dtr = xgb.DMatrix(tr, label=fold.y_tr, enable_categorical=True)
        models.append(
            {
                "va_idx": fold_idx[1],
                "y_va": fold.y_va,
                "dtr": dtr,
                "dva": xgb.DMatrix(va, enable_categorical=True),
                "dte": xgb.DMatrix(te, enable_categorical=True),
                "booster": xgb.Booster(params, [dtr]),
                "n_features": tr.shape[1],
                # всё, что нужно для предсказания на сырых данных
                "pipeline": {
                    "prep": fold.prep,
                    "jdf": objs["jdf"],
                    "fid": objs["fid"],
                    "columns": list(tr.columns),
                },
            }
        )
        del fold, tr, va, te
        gc.collect()
        print(f"  fold {i + 1}/{len(folds)} подготовлен, признаков: {models[-1]['n_features']}")
    return models


def save_models(models, rank, models_dir):
    """Бустеры — встроенным Booster.save_model, объекты предобработки — joblib."""
    files = []
    for i, m in enumerate(models):
        model_file = f"xgb_m{rank}_fold{i}.ubj"
        pipe_file = f"xgb_m{rank}_fold{i}_pipeline.joblib"
        m["booster"].save_model(str(models_dir / model_file))
        joblib.dump(m["pipeline"], models_dir / pipe_file, compress=3)
        files.append({"model": model_file, "pipeline": pipe_file})
    return files


def run_model(item, rank, X, y, X_test, folds, est_coefs, eval_every, max_iter, importance_path,
              min_patience, models_dir):
    models = build_fold_models(item, X, y, X_test, folds, importance_path)
    n_train, n_test = len(X), len(X_test)

    def advance(t_from, t_to):
        for m in models:
            for it in range(t_from, t_to):
                m["booster"].update(m["dtr"], it)

    def evaluate(t):
        oof = np.zeros(n_train)
        fold_aucs = []
        for m in models:
            p = m["booster"].predict(m["dva"], iteration_range=(0, t))
            oof[m["va_idx"]] = p
            fold_aucs.append(roc_auc_score(m["y_va"], p))
        return oof, fold_aucs

    def predict_test(t):
        out = np.zeros(n_test)
        for m in models:
            out += m["booster"].predict(m["dte"], iteration_range=(0, t)) / len(models)
        return out

    # бустер хранит все деревья, поэтому предсказания теста считаем один раз в конце (lazy_test)
    tracker = run_adaptive(
        est_coefs, advance, evaluate, predict_test, y, eval_every=eval_every, max_iter=max_iter,
        log_every=10, label="trees", lazy_test=True, min_patience=min_patience,
    )
    files = save_models(models, rank, models_dir)
    print(f"  модели и пайплайны сохранены в {models_dir}")
    del models
    gc.collect()
    return tracker, files


def main(model_ranks=MODEL_RANKS, est_coefs=EST_COEFS, k_folds=K_FOLDS, eval_every=EVAL_EVERY,
         max_iter=MAX_ITERS, min_patience=MIN_PATIENCE, params_path=xs.PARAMS_PATH,
         out_dir=OUT_DIR, models_dir=MODELS_DIR, oof_dir=OOF_DIR):
    with open(params_path, encoding="utf-8") as f:
        payload = json.load(f)
    top = payload["top_params"]
    bad = [r for r in model_ranks if not 0 <= r < len(top)]
    if bad:
        raise IndexError(f"в {params_path} только {len(top)} конфигов, запрошены индексы {bad}")
    if min(est_coefs) < 1.0:
        raise ValueError("est_coefs должны быть >= 1.0 (чекпоинты считаются от лучшей итерации)")
    importance_path = payload.get("importance_path", xs.IMPORTANCE_PATH)

    np.random.seed(xs.RANDOM_STATE)
    X, y, X_test, test_ref_id = xs.load_data()
    train_ref_id = pd.read_csv(xs.TRAIN_PATH)["RefId"]
    print(f"train: {X.shape}, test: {X_test.shape}")

    folds = list(
        StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=xs.RANDOM_STATE).split(X, y)
    )
    out_dir, models_dir, oof_dir = Path(out_dir), Path(models_dir), Path(oof_dir)
    for d in (out_dir, models_dir, oof_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows, manifest_models = [], []
    for r in model_ranks:
        item = top[r]
        print(f"=== model rank {r}: outer {item['outer_trial']}, tuning val_auc={item['val_auc']:.5f}, "
              f"prep={item['preprocessing']} ===")
        tracker, files = run_model(item, r, X, y, X_test, folds, est_coefs, eval_every, max_iter,
                                   importance_path, min_patience, models_dir)

        history_frame(tracker).to_csv(out_dir / f"cv_history_m{r}.csv", index=False)
        plot_cv_curve(
            tracker, out_dir / f"cv_curve_m{r}.png",
            title=f"XGB model {r}: CV ROC AUC vs число деревьев (макс. {tracker.best_auc:.5f} @ {tracker.best_iter})",
            xlabel="число деревьев",
        )

        for t, snap in sorted(tracker.saved.items()):
            tag = f"m{r}_est{t}_cv{snap['cv_auc']:.5f}"
            pd.DataFrame(
                {"RefId": train_ref_id.values,  "IsBadBuy": snap["oof"]}
            ).to_csv(oof_dir / f"xgb_oof_{tag}.csv", index=False)
            pd.DataFrame(
                {"RefId": test_ref_id.values, "IsBadBuy": snap["test"]}
            ).to_csv(oof_dir / f"xgb_test_{tag}.csv", index=False)

        checkpoints = []
        for c, t in tracker.targets.items():
            if t not in tracker.saved:
                continue
            snap = tracker.saved[t]
            checkpoints.append({"coef": c, "n_iter": int(t), "cv_auc": float(snap["cv_auc"])})
            rows.append(
                {
                    "model_rank": r,
                    "outer_trial": item["outer_trial"],
                    "tuning_val_auc": item["val_auc"],
                    "tuning_best_estimators": item["best_estimators"],
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
                "outer_trial": int(item["outer_trial"]),
                "best_iter": int(tracker.best_iter),
                "best_cv_auc": float(tracker.best_auc),
                "n_trees": int(tracker.history[-1][0]),  # столько деревьев в сохранённых бустерах
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
        "family": "xgb",
        "k_folds": k_folds,
        "models": manifest_models,
        "default": {"rank": best[0], "n_iter": best[1]["n_iter"], "cv_auc": best[1]["cv_auc"]},
    }
    with open(models_dir / "xgb_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"манифест: {models_dir / 'xgb_manifest.json'}, по умолчанию {manifest['default']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"OOF/test предсказания: {oof_dir}; отчёты: {out_dir}")
    return summary


if __name__ == "__main__":
    main()