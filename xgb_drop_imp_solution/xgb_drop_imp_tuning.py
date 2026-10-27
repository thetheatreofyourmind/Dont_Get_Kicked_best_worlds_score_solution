"""XGBoost для dont_get_kicked: вложенный Optuna-тюнинг пайплайна.

Пайплайн:  Preprocessing(add_text_binary_matrix=True) -> JointDistributionFilter -> FeatureImportanceDropper
           -> XGBClassifier (enable_categorical, категории — pandas category после JDF).
Текст: просто бинарная матрица токенов text__* (редкие токены режет JDF по th_cat_*).

Вложенный тюнинг (оценка — ROC AUC на валидационной части фолда TUNE_FOLD, она же для early stopping):
  внешний trial  — параметры Preprocessing: date_days (3..30), text_min_count_train (30..500, log);
                   Preprocessing фитится на train-части фолда один раз на внешний trial;
  внутренние trials (INNER_TRIALS на каждый внешний) — параметры JDF (dp, th_num_*, th_cat_*),
                   FID (n_worst 150..250) и самого XGBoost.
  Значение внешнего trial = AUC лучшего внутреннего trial.
В xgb_top_params.json сохраняются TOP_K лучших внешних trials: параметры Preprocessing и лучшего
внутреннего trial (jdf, fid, xgb), val_auc и best_estimators (число деревьев при early stopping).
Файл перезаписывается после каждого внешнего trial.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from preprocessing import FeatureImportanceDropper, JointDistributionFilter, Preprocessing

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dont_get_kicked" / "datasets"
TRAIN_PATH = DATA_DIR / "training.csv"
TEST_PATH = DATA_DIR / "test.csv"
IMPORTANCE_PATH = BASE_DIR.parent / "importance_df.csv"  # = D:/DS_Proj/importance_df.csv (дефолт FID)
PARAMS_PATH = BASE_DIR / "xgb_drop_imp_top_params.json"

TARGET = "IsBadBuy"
RANDOM_STATE = 42
N_SPLITS = 5
TUNE_FOLD = 0
OUTER_TRIALS = 1
INNER_TRIALS = 50
TOP_K = 5
MAX_ESTIMATORS = 4000
EARLY_STOP_ROUNDS = 100
DROP_TOP_N_FEATURES = 5

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# Данные
# --------------------------------------------------------------------------- #
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train["PurchDate"] = pd.to_datetime(train["PurchDate"])
    test["PurchDate"] = pd.to_datetime(test["PurchDate"])
    y = train[TARGET]
    X = train.drop(columns=[TARGET, "RefId"])
    X_test = test.drop(columns=["RefId"])
    return X, y, X_test, test["RefId"]


@dataclass
class FoldData:
    X_tr: pd.DataFrame
    X_va: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: np.ndarray
    y_va: np.ndarray
    prep: object = None  # обученный Preprocessing фолда (нужен для сохранения)


def prepare_fold(X, y, X_test, fold_idx, prep_params):
    """Preprocessing с заданными date_days / text_min_count_train, fit на train-части фолда."""
    tr_idx, va_idx = fold_idx
    prep = Preprocessing(weekday_type = 'category', add_text_binary_matrix=True, **prep_params)
    prep.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    return FoldData(
        X_tr=prep.transform(X.iloc[tr_idx]),
        X_va=prep.transform(X.iloc[va_idx]),
        X_te=prep.transform(X_test),
        y_tr=y.iloc[tr_idx].to_numpy(),
        y_va=y.iloc[va_idx].to_numpy(),
        prep=prep,
    )


def drop_empty_categoricals(*frames):
    """Удаляет категориальные колонки без единой категории (после JDF все значения стали NaN).

    XGBoost (QuantileDMatrix) падает на такой колонке: "cannot call `vectorize` on size 0 inputs".
    Признак без категорий не несёт информации, поэтому удаление ничего не меняет. Набор категорий у JDF
    один и тот же для train/val/test, так что колонки удаляются согласованно.
    """
    ref = frames[0]
    empty = [
        c for c in ref.columns
        if isinstance(ref[c].dtype, pd.CategoricalDtype) and len(ref[c].cat.categories) == 0
    ]
    if not empty:
        return frames
    return tuple(None if f is None else f.drop(columns=empty) for f in frames)


def apply_pipeline(X_proc, jdf, fid):
    """Применение уже обученных JDF и FID к выходу Preprocessing (общий путь для обучения и predict.py)."""
    (X,) = drop_empty_categoricals(jdf.transform(X_proc))
    return fid.transform(X)


def make_frames(fold, jdf_params, fid_params, importance_path=IMPORTANCE_PATH, include_test=True,
                return_objects=False):
    """JDF (профили train+test) -> FID -> (train, val, test) для XGBoost.

    return_objects=True — четвёртым элементом вернуть обученные {"jdf": ..., "fid": ...}.
    """
    jdf = JointDistributionFilter(fold.X_tr, fold.X_te, **jdf_params)
    tr, va = jdf.transform(fold.X_tr), jdf.transform(fold.X_va)
    te = jdf.transform(fold.X_te) if include_test else None
    tr, va, te = drop_empty_categoricals(tr, va, te)

    fid = FeatureImportanceDropper(n_best = DROP_TOP_N_FEATURES,importance_path=str(importance_path), **fid_params).fit(tr)
    tr, va = fid.transform(tr), fid.transform(va)
    te = fid.transform(te) if include_test else None
    if return_objects:
        return tr, va, te, {"jdf": jdf, "fid": fid}
    return tr, va, te


def fit_xgb(xgb_params, n_estimators, X_tr, y_tr, X_va=None, y_va=None, early_stopping_rounds=None):
    kwargs = dict(
        n_estimators=n_estimators,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        **xgb_params,
    )
    if X_va is None:
        model = xgb.XGBClassifier(**kwargs)
        model.fit(X_tr, y_tr, verbose=False)
    else:
        if early_stopping_rounds:
            kwargs["early_stopping_rounds"] = early_stopping_rounds
        model = xgb.XGBClassifier(**kwargs)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def best_estimators_of(model, default):
    """Число деревьев на лучшей итерации (best_iteration у XGBoost нумеруется с 0)."""
    bi = getattr(model, "best_iteration", None)
    return int(bi) + 1 if bi is not None else int(default)


# --------------------------------------------------------------------------- #
# Optuna: внутренний и внешний уровни
# --------------------------------------------------------------------------- #
def sample_inner_params(trial):
    jdf_params = {
        "dp": trial.suggest_float("dp", 0.01, 0.05),
        "th_num_train": trial.suggest_float("th_num_train", 1e-4, 1e-1, log=True),
        "th_num_test": trial.suggest_float("th_num_test", 1e-4, 1e-1, log=True),
        "th_cat_train": trial.suggest_float("th_cat_train", 1e-3, 3e-2, log=True),
        "th_cat_test": trial.suggest_float("th_cat_test", 1e-3, 3e-2, log=True),
    }
    fid_params = {"n_worst": trial.suggest_int("n_worst", 150, 250)}
    xgb_params = {
        "objective": "binary:logistic",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    return jdf_params, fid_params, xgb_params


def run_inner_study(fold, n_trials, seed, label=""):
    """Тюнинг JDF + FID + XGB на подготовленном фолде. Возвращает лучший внутренний trial (dict)."""

    def objective(trial):
        start = time.time()
        jdf_params, fid_params, xgb_params = sample_inner_params(trial)
        tr, va, _ = make_frames(fold, jdf_params, fid_params, include_test=False)
        model = fit_xgb(xgb_params, MAX_ESTIMATORS, tr, fold.y_tr, va, fold.y_va, EARLY_STOP_ROUNDS)
        best_est = best_estimators_of(model, MAX_ESTIMATORS)
        auc = roc_auc_score(fold.y_va, model.predict_proba(va, iteration_range=(0, best_est))[:, 1])
        trial.set_user_attr("jdf", jdf_params)
        trial.set_user_attr("fid", fid_params)
        trial.set_user_attr("xgb", xgb_params)
        trial.set_user_attr("best_estimators", best_est)
        if trial.number % 10 == 0:
            print(f"    {label}inner {trial.number}: AUC={auc:.5f} trees={best_est} "
                  f"features={tr.shape[1]} {time.time() - start:.1f}s")
        return auc

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    best = max(study.trials, key=lambda t: t.value)
    return {
        "inner_trial": best.number,
        "val_auc": float(best.value),
        "best_estimators": int(best.user_attrs["best_estimators"]),
        "jdf": best.user_attrs["jdf"],
        "fid": best.user_attrs["fid"],
        "xgb": best.user_attrs["xgb"],
    }


def save_top_params(study, path, top_k, inner_trials):
    done = [t for t in study.trials if t.value is not None]
    done.sort(key=lambda t: (-t.value, t.number))
    top = []
    for r, t in enumerate(done[:top_k]):
        inner = t.user_attrs["best_inner"]
        top.append(
            {
                "rank": r,
                "outer_trial": t.number,
                "val_auc": inner["val_auc"],
                "best_estimators": inner["best_estimators"],
                "preprocessing": t.user_attrs["prep"],
                "inner_trial": inner["inner_trial"],
                "jdf": inner["jdf"],
                "fid": inner["fid"],
                "xgb": inner["xgb"],
            }
        )
    payload = {
        "tuning_fold": TUNE_FOLD,
        "outer_trials_done": len(done),
        "inner_trials": inner_trials,
        "max_estimators": MAX_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOP_ROUNDS,
        "importance_path": str(IMPORTANCE_PATH),
        "top_params": top,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def tune_nested(X, y, X_test, fold_idx, outer_trials, inner_trials, top_k, path=PARAMS_PATH):
    print(f"nested tuning: {outer_trials} outer x {inner_trials} inner trials, fold {TUNE_FOLD}")

    def outer_objective(trial):
        start = time.time()
        prep_params = {
            "date_days": trial.suggest_int("date_days", 3, 30),
            "text_min_count_train": trial.suggest_int("text_min_count_train", 30, 500, log=True),
        }
        print(f"outer {trial.number}: {prep_params}")
        fold = prepare_fold(X, y, X_test, fold_idx, prep_params)
        best_inner = run_inner_study(
            fold, inner_trials, RANDOM_STATE + trial.number, label=f"[outer {trial.number}] "
        )
        trial.set_user_attr("prep", prep_params)
        trial.set_user_attr("best_inner", best_inner)
        print(f"outer {trial.number}: best inner AUC={best_inner['val_auc']:.5f}, "
              f"trees={best_inner['best_estimators']} | {time.time() - start:.0f}s")
        return best_inner["val_auc"]

    def save_callback(study, trial):
        save_top_params(study, path, top_k, inner_trials)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(outer_objective, n_trials=outer_trials, callbacks=[save_callback])
    return study


def main(outer_trials=OUTER_TRIALS, inner_trials=INNER_TRIALS, top_k=TOP_K):
    if not Path(IMPORTANCE_PATH).exists():
        raise FileNotFoundError(f"не найден файл важностей признаков для FID: {IMPORTANCE_PATH}")

    np.random.seed(RANDOM_STATE)
    X, y, X_test, _ = load_data()
    print(f"train: {X.shape}, test: {X_test.shape}, target ratio: {y.mean():.4f}")

    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    )
    study = tune_nested(X, y, X_test, folds[TUNE_FOLD], outer_trials, inner_trials, top_k)
    payload = save_top_params(study, PARAMS_PATH, top_k, inner_trials)

    print(f"top-{len(payload['top_params'])} saved to {PARAMS_PATH}")
    for item in payload["top_params"]:
        print(f"#{item['rank']}: outer {item['outer_trial']}, AUC={item['val_auc']:.5f}, "
              f"trees={item['best_estimators']}, prep={item['preprocessing']}")
    return payload


if __name__ == "__main__":
    main()