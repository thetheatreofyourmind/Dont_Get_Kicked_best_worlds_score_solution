"""CatBoost для dont_get_kicked: Optuna-тюнинг + общие функции подготовки данных.

Пайплайн:  Preprocessing(**PREP_PARAMS) -> JointDistributionFilter(**JDF_PARAMS) -> CatBoostClassifier.
Категориальные колонки (dtype category после JDF) отдаются в CatBoost строками, NaN -> '__nan__'.
Параметры Preprocessing и JDF фиксированы (как в исходном ноутбуке) и не тюнятся.

Тюнинг: Optuna на валидационной части фолда TUNE_FOLD (StratifiedKFold, 5, shuffle, random_state=42),
метрика — ROC AUC; та же валидация используется для early stopping (od_wait=100), как в исходном коде.
Пространство поиска, TPESampler(seed=42, multivariate=True) и MedianPruner — без изменений.

В catboost_best_params.json сохраняется топ-5 в прежнем формате:
    [{"rank": 1, "roc_auc": ..., "trial_number": ..., "params": {...}}, ...]
Поэтому уже сохранённый json подходит для catboost_training.py без повторного тюнинга.

Отличия от ноутбука: Preprocessing обучается только на train-части фолда (раньше fit шёл по всему
трейну, включая метки валидации, из-за чего AUC был завышен), а валидация — фолд 0 из того же
StratifiedKFold, что и у остальных моделей.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from preprocessing import JointDistributionFilter, Preprocessing

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dont_get_kicked" / "datasets"
TRAIN_PATH = DATA_DIR / "training.csv"
TEST_PATH = DATA_DIR / "test.csv"
PARAMS_PATH = BASE_DIR / "catboost_best_params.json"
PARAMS_FILE_NAMES = ("catboost_best_params.json", )

TARGET = "IsBadBuy"
RANDOM_STATE = 42
N_SPLITS = 5
TUNE_FOLD = 0
OPTUNA_TRIALS = 100
TOP_K = 5

# параметры предобработки — как в исходном ноутбуке (в тюнинге не участвуют)
PREP_PARAMS = {"date_days": 7, "weekday_type": "category"}
JDF_PARAMS = {"th_cat_train": 0.002, "th_cat_test": 0.002}
NAN_TOKEN = "__nan__"


# --------------------------------------------------------------------------- #
# Данные и общие функции (используются catboost_training.py и predict.py)
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


def find_params_file(path=None):
    """Путь к json с топ-параметрами: явный путь или один из привычных файлов в папке скрипта."""
    if path is not None:
        return Path(path)
    for name in PARAMS_FILE_NAMES:
        if (BASE_DIR / name).exists():
            return BASE_DIR / name
    return PARAMS_PATH


@dataclass
class FoldData:
    X_tr: pd.DataFrame
    X_va: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: np.ndarray
    y_va: np.ndarray
    prep: object = None  # обученный Preprocessing фолда (нужен для сохранения)


def prepare_fold(X, y, X_test, fold_idx):
    """Preprocessing(**PREP_PARAMS), fit на train-части фолда."""
    tr_idx, va_idx = fold_idx
    prep = Preprocessing(**PREP_PARAMS)
    prep.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    return FoldData(
        X_tr=prep.transform(X.iloc[tr_idx]),
        X_va=prep.transform(X.iloc[va_idx]),
        X_te=prep.transform(X_test),
        y_tr=y.iloc[tr_idx].to_numpy(),
        y_va=y.iloc[va_idx].to_numpy(),
        prep=prep,
    )


def to_catboost_frame(X, cat_cols):
    """Категориальные колонки -> строки, NaN -> '__nan__' (CatBoost не принимает NaN в cat-признаках)."""
    out = X.copy()
    for c in cat_cols:
        s = out[c]
        # явный object: в новых pandas astype(str) даёт строковый dtype, а CatBoost надёжнее принимает object
        out[c] = s.astype(object).where(s.notna(), NAN_TOKEN).astype(str).astype(object)
    return out


def apply_pipeline(X_proc, jdf, cat_cols, columns=None):
    """Применение обученного JDF к выходу Preprocessing + приведение cat-колонок к строкам."""
    X = to_catboost_frame(jdf.transform(X_proc), cat_cols)
    return X if columns is None else X[columns]


def make_frames(fold):
    """JDF (профили train+test) -> (train, val, test) для CatBoost и info для сохранения.

    info = {"jdf": обученный JDF, "cat_cols": список категориальных колонок, "columns": порядок колонок}.
    """
    jdf = JointDistributionFilter(fold.X_tr, fold.X_te, **JDF_PARAMS)
    tr, va, te = (jdf.transform(d) for d in (fold.X_tr, fold.X_va, fold.X_te))
    cat_cols = [c for c in tr.columns if isinstance(tr[c].dtype, pd.CategoricalDtype)]
    tr, va, te = (to_catboost_frame(d, cat_cols) for d in (tr, va, te))
    return tr, va, te, {"jdf": jdf, "cat_cols": cat_cols, "columns": list(tr.columns)}


def clean_params(params):
    """Совместимость с набором параметров из json: subsample имеет смысл только при Bernoulli."""
    p = dict(params)
    if "subsample" in p and p.get("bootstrap_type", "Bernoulli") != "Bernoulli":
        p.pop("subsample", None)
    return p


# --------------------------------------------------------------------------- #
# Optuna
# --------------------------------------------------------------------------- #
def make_objective(train_pool, valid_pool, y_valid):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 500, 4000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 1e-2, log=True),
            "depth": trial.suggest_int("depth", 4, 12),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "rsm": trial.suggest_float("rsm", 0.3, 1.0),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 50),
            "model_size_reg": trial.suggest_float("model_size_reg", 0.0, 10.0),
            "od_type": "Iter",
            "od_wait": 100,
            "eval_metric": "AUC",
            "loss_function": "Logloss",
            "task_type": "CPU",
            "verbose": False,
            "allow_writing_files": False,
            "random_seed": RANDOM_STATE,
        }
        try:
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            preds = model.predict_proba(valid_pool)[:, 1]
            return roc_auc_score(y_valid, preds)
        except Exception as e:
            print(f"[Trial {trial.number}] failed: {e}")
            raise optuna.TrialPruned()

    return objective


class ProgressCallback:
    """Прогресс по каждому триалу: значение, лучший результат, время; при улучшении — параметры."""

    def __init__(self):
        self.start_time = time.time()
        self.best_value = -np.inf

    def __call__(self, study, trial):
        elapsed = time.time() - self.start_time
        total = len(study.trials)
        value_str = "   N/A  " if trial.value is None else f"{trial.value:.5f}"

        marker = "  "
        if trial.value is not None and trial.value > self.best_value:
            self.best_value = trial.value
            marker = "★ "

        try:
            best_overall = study.best_value
        except ValueError:
            best_overall = float("nan")

        print(
            f"[{marker}Trial {trial.number:>3}/{total}] "
            f"state={trial.state.name:<8} AUC={value_str} best={best_overall:.5f} elapsed={elapsed:6.1f}s"
        )
        if marker == "★ ":
            print(f"      new best params: {trial.params}")


def save_top_params(study, path, top_k=TOP_K):
    """Топ-k триалов в прежнем формате json: [{"rank", "roc_auc", "trial_number", "params"}]."""
    top_trials = sorted(
        [t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True
    )[:top_k]
    top = [
        {"rank": i + 1, "roc_auc": t.value, "trial_number": t.number, "params": t.params}
        for i, t in enumerate(top_trials)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(top, f, indent=2, ensure_ascii=False)
    return top


def main(n_trials=OPTUNA_TRIALS, top_k=TOP_K, params_path=PARAMS_PATH):
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # свой лог через ProgressCallback

    X, y, X_test, _ = load_data()
    print(f"train: {X.shape}, test: {X_test.shape}, target ratio: {y.mean():.4f}")

    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    )
    fold = prepare_fold(X, y, X_test, folds[TUNE_FOLD])
    tr, va, _, info = make_frames(fold)
    cat_cols = info["cat_cols"]
    print(f"признаков: {tr.shape[1]}, категориальных: {len(cat_cols)}")

    train_pool = Pool(tr, fold.y_tr, cat_features=cat_cols)
    valid_pool = Pool(va, fold.y_va, cat_features=cat_cols)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        study_name="catboost_tuning",
    )
    study.optimize(
        make_objective(train_pool, valid_pool, fold.y_va),
        n_trials=n_trials,
        callbacks=[ProgressCallback()],
        show_progress_bar=False,
    )

    top = save_top_params(study, params_path, top_k)
    print(f"\n=== TOP-{len(top)} (сохранено в {params_path}) ===")
    for row in top:
        print(f"#{row['rank']} roc_auc={row['roc_auc']:.5f} trial={row['trial_number']}")
    return top


if __name__ == "__main__":
    main()