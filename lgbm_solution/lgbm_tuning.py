"""LightGBM для dont_get_kicked: текстовые признаки из обученной нейросети + Optuna-тюнинг.

Схема:
  1. Текст. Данные для нейросети готовятся как в nn_solution: Preprocessing(add_text_binary_matrix=True)
     -> JointDistributionFilter (дефолтные параметры). Матрица токенов text__* (в порядке токенов,
     на которых обучена сеть) прогоняется через блок эмбеддингов и первое линейное преобразование
     сети из nn_solution/token_importance/nn_full_model.pt:
         text -> Linear(T -> text_emb_dim, без bias) [/ число токенов] -> Linear(-> text_out_dim)
     Результат — text_out_dim признаков nn_text_0 ... nn_text_{d-1} на каждый объект.
     По умолчанию берётся выход Linear до активации (apply_activation=False).
  2. Остальное. Preprocessing с дефолтными параметрами (add_text_binary_matrix=False), fit на train-части
     фолда -> JointDistributionFilter (параметры тюнятся) -> + nn_text_* признаки.
     Категориальные колонки (dtype category) идут в LightGBM как категориальные, NaN — как есть.
  3. Тюнинг: Optuna на одном фолде (StratifiedKFold, фолд TUNE_FOLD), val-часть фолда используется и для
     early stopping, и для метрики (ROC AUC). Тюнятся параметры JDF и параметры LightGBM.
  4. В lgbm_top_params.json сохраняются TOP_K лучших триалов (по val AUC) с параметрами JDF и LGBM
     и best_estimators (число деревьев на момент early stopping).

ВАЖНО: нейросеть обучена на всём трейне, поэтому её эмбеддинги токенов уже "видели" метки валидационных
частей фолдов — CV-метрики LightGBM с такими признаками оптимистичны.
"""
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from preprocessing import JointDistributionFilter, Preprocessing

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dont_get_kicked" / "datasets"
TRAIN_PATH = DATA_DIR / "training.csv"
TEST_PATH = DATA_DIR / "test.csv"
NN_MODEL_PATH = BASE_DIR.parent / "nn_solution" / "token_importance" / "nn_full_model.pt"
PARAMS_PATH = BASE_DIR / "lgbm_top_params.json"

TARGET = "IsBadBuy"
TEXT_PREFIX = "text__"
RANDOM_STATE = 42
N_SPLITS = 5
TUNE_FOLD = 0
OPTUNA_TRIALS = 40
TOP_K = 5
MAX_ESTIMATORS = 5000
EARLY_STOP_ROUNDS = 100

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


# --------------------------------------------------------------------------- #
# Текстовые признаки из нейросети
# --------------------------------------------------------------------------- #
def _gelu(x):
    from scipy.special import erf

    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


NP_ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "gelu": _gelu,
    "tanh": np.tanh,
    "selu": lambda x: 1.0507009873554805
    * np.where(x > 0, x, 1.6732632423543772 * (np.exp(np.minimum(x, 0.0)) - 1.0)),
    "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
    "elu": lambda x: np.where(x > 0, x, np.exp(np.minimum(x, 0.0)) - 1.0),
}


def load_nn_text_encoder(path=NN_MODEL_PATH):
    """Достаёт из nn_full_model.pt веса блока текста: токены, text_emb, первый Linear text_out."""
    import torch

    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["state_dict"]
    cfg = ckpt["model_cfg"]
    enc = {
        "tokens": list(ckpt["tokens"]),
        "W_emb": sd["text_emb.weight"].float().numpy(),  # (text_emb_dim, T)
        "W_out": sd["text_out.0.weight"].float().numpy(),  # (text_out_dim, text_emb_dim)
        "b_out": sd["text_out.0.bias"].float().numpy(),
        "normalize": bool(cfg["normalize_text"]),
        "activation": cfg["activation"],
    }
    if enc["W_emb"].shape[1] != len(enc["tokens"]):
        raise RuntimeError("число токенов в чекпоинте не совпадает с размером text_emb")
    return enc


def encode_text(text_matrix, enc, apply_activation=False):
    """text_matrix (n x T, 0/1) -> (n x text_out_dim): эмбеддинг токенов + первый Linear."""
    m = np.asarray(text_matrix, dtype=np.float32)
    emb = m @ enc["W_emb"].T
    if enc["normalize"]:
        emb = emb / np.clip(m.sum(axis=1, keepdims=True), 1.0, None)
    out = emb @ enc["W_out"].T + enc["b_out"]
    if apply_activation:
        out = NP_ACTIVATIONS[enc["activation"]](out)
    return out.astype(np.float32)


def fit_nn_text_pipeline(X, y, X_test, apply_activation=False, enc=None):
    """Обучение предобработки текста как для нейросети: Preprocessing + JDF + веса блока текста сети.

    Возвращает dict {"prep", "jdf", "enc", "apply_activation"} — сохраняется через joblib.
    """
    enc = enc or load_nn_text_encoder()
    prep = Preprocessing(add_text_binary_matrix=True)
    prep.fit(X, y)
    jdf = JointDistributionFilter(prep.transform(X), prep.transform(X_test))
    return {"prep": prep, "jdf": jdf, "enc": enc, "apply_activation": apply_activation}


def transform_nn_text(X, pipe):
    """Сырые данные -> DataFrame nn_text_* (индекс как у X) обученным пайплайном текста."""
    enc = pipe["enc"]
    X_f = pipe["jdf"].transform(pipe["prep"].transform(X))
    cols = [f"{TEXT_PREFIX}{t}" for t in enc["tokens"]]
    missing = [c for c in cols if c not in X_f.columns]
    if missing:
        raise RuntimeError(
            f"в данных нет {len(missing)} токенов, на которых обучена сеть (например {missing[:5]}): "
            "предобработка отличается от использованной в nn_solution"
        )
    names = [f"nn_text_{i}" for i in range(enc["W_out"].shape[0])]
    return pd.DataFrame(
        encode_text(X_f[cols].to_numpy(), enc, pipe["apply_activation"]), columns=names, index=X.index
    )


def build_nn_text_features(X, y, X_test, apply_activation=False, enc=None, return_pipeline=False):
    """Preprocessing + JDF как для нейросети -> матрица text__* -> эмбеддинги сети.

    Возвращает (text_train, text_test): DataFrame nn_text_* с индексами X и X_test;
    при return_pipeline=True третьим элементом — обученный пайплайн текста (для сохранения).
    """
    pipe = fit_nn_text_pipeline(X, y, X_test, apply_activation, enc)
    text_train = transform_nn_text(X, pipe)
    text_test = transform_nn_text(X_test, pipe)
    if return_pipeline:
        return text_train, text_test, pipe
    return text_train, text_test


# --------------------------------------------------------------------------- #
# Фолд: Preprocessing (дефолт) один раз, JDF — в каждом триале
# --------------------------------------------------------------------------- #
@dataclass
class FoldData:
    X_tr: pd.DataFrame
    X_va: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: np.ndarray
    y_va: np.ndarray
    prep: object = None  # обученный Preprocessing фолда (нужен для сохранения)


def prepare_fold(X, y, X_test, fold_idx):
    tr_idx, va_idx = fold_idx
    prep = Preprocessing()  # параметры по умолчанию, add_text_binary_matrix=False
    prep.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    return FoldData(
        X_tr=prep.transform(X.iloc[tr_idx]),
        X_va=prep.transform(X.iloc[va_idx]),
        X_te=prep.transform(X_test),
        y_tr=y.iloc[tr_idx].to_numpy(),
        y_va=y.iloc[va_idx].to_numpy(),
        prep=prep,
    )


def _attach_text(frame, text_df):
    out = pd.concat([frame, text_df.loc[frame.index]], axis=1)
    # LightGBM не любит спецсимволы в именах признаков (например, "/")
    out.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)) for c in out.columns]
    if not out.columns.is_unique:
        raise ValueError("после очистки имена признаков не уникальны")
    return out


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


def make_frames(fold, jdf_params, text_train, text_test, include_test=True, return_objects=False):
    """JDF (train+test профили) + nn_text_* признаки -> (train, val, test) для LightGBM."""
    jdf = JointDistributionFilter(fold.X_tr, fold.X_te, **jdf_params)
    tr, va, te = jdf.transform(fold.X_tr), jdf.transform(fold.X_va), (
        jdf.transform(fold.X_te) if include_test else None
    )
    tr, va, te = drop_empty_categoricals(tr, va, te)
    tr, va = _attach_text(tr, text_train), _attach_text(va, text_train)
    te = _attach_text(te, text_test) if include_test else None
    if return_objects:
        return tr, va, te, {"jdf": jdf}
    return tr, va, te


def apply_pipeline(X_proc, jdf, text_df):
    """Применение обученного JDF к выходу Preprocessing + nn_text_* (общий путь для обучения и predict.py)."""
    (X,) = drop_empty_categoricals(jdf.transform(X_proc))
    return _attach_text(X, text_df)


def fit_lgbm(lgbm_params, n_estimators, X_tr, y_tr, X_va=None, y_va=None, early_stopping_rounds=None):
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, random_state=RANDOM_STATE, verbose=-1, **lgbm_params
    )
    if X_va is None:
        model.fit(X_tr, y_tr)
    else:
        callbacks = []
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
    return model


# --------------------------------------------------------------------------- #
# Optuna
# --------------------------------------------------------------------------- #
def sample_params(trial):
    jdf_params = {
        "dp": trial.suggest_float("dp", 0.01, 0.05),
        "th_num_train": trial.suggest_float("th_num_train", 1e-4, 1e-1, log=True),
        "th_num_test": trial.suggest_float("th_num_test", 1e-4, 1e-1, log=True),
        "th_cat_train": trial.suggest_float("th_cat_train", 1e-3, 3e-2, log=True),
        "th_cat_test": trial.suggest_float("th_cat_test", 1e-3, 3e-2, log=True),
    }
    lgbm_params = {
        "objective": "binary",
        "metric": "auc",
        "subsample_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
        "cat_l2": trial.suggest_float("cat_l2", 1.0, 100.0, log=True),
        "min_data_per_group": trial.suggest_int("min_data_per_group", 20, 300, log=True),
    }
    return jdf_params, lgbm_params


def tune_on_single_fold(fold, text_train, text_test, n_trials):
    print(f"tuning {n_trials} trials on fold (train={len(fold.X_tr)}, val={len(fold.X_va)})...")

    def objective(trial):
        start = time.time()
        jdf_params, lgbm_params = sample_params(trial)
        tr, va, _ = make_frames(fold, jdf_params, text_train, text_test, include_test=False)
        model = fit_lgbm(
            lgbm_params, MAX_ESTIMATORS, tr, fold.y_tr, va, fold.y_va, EARLY_STOP_ROUNDS
        )
        auc = roc_auc_score(fold.y_va, model.predict_proba(va)[:, 1])
        best_estimators = int(model.best_iteration_ or MAX_ESTIMATORS)
        trial.set_user_attr("jdf", jdf_params)
        trial.set_user_attr("lgbm", lgbm_params)
        trial.set_user_attr("best_estimators", best_estimators)
        print(
            f"trial {trial.number}: AUC={auc:.5f} | trees={best_estimators} | "
            f"features={tr.shape[1]} | {time.time() - start:.1f}s"
        )
        return auc

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def save_top_params(study, path, top_k, apply_activation):
    done = [t for t in study.trials if t.value is not None]
    done.sort(key=lambda t: (-t.value, t.number))
    top = [
        {
            "rank": r,
            "trial": t.number,
            "val_auc": t.value,
            "best_estimators": t.user_attrs["best_estimators"],
            "jdf": t.user_attrs["jdf"],
            "lgbm": t.user_attrs["lgbm"],
        }
        for r, t in enumerate(done[:top_k])
    ]
    payload = {
        "tuning_fold": TUNE_FOLD,
        "n_trials": len(study.trials),
        "max_estimators": MAX_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOP_ROUNDS,
        "nn_text": {"model_path": str(NN_MODEL_PATH), "apply_activation": apply_activation},
        "top_params": top,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"top-{len(top)} params saved to {path}")
    return payload


def main(n_trials=OPTUNA_TRIALS, top_k=TOP_K, apply_activation=False):
    np.random.seed(RANDOM_STATE)
    X, y, X_test, _ = load_data()
    print(f"train: {X.shape}, test: {X_test.shape}, target ratio: {y.mean():.4f}")

    print("text features from NN embeddings...")
    text_train, text_test = build_nn_text_features(X, y, X_test, apply_activation)
    print(f"nn_text features: {text_train.shape[1]}")

    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    )
    fold = prepare_fold(X, y, X_test, folds[TUNE_FOLD])

    study = tune_on_single_fold(fold, text_train, text_test, n_trials)
    payload = save_top_params(study, PARAMS_PATH, top_k, apply_activation)
    for item in payload["top_params"]:
        print(f"#{item['rank']}: trial {item['trial']}, AUC={item['val_auc']:.5f}, "
              f"trees={item['best_estimators']}")
    return payload


if __name__ == "__main__":
    main()