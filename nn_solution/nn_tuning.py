"""Нейросеть (PyTorch) для dont_get_kicked + Optuna-тюнинг гиперпараметров.

Данные готовит пайплайн из preprocessing.py:
  Preprocessing(add_text_binary_matrix=True)  ->  JointDistributionFilter

Разбор колонок после пайплайна (по DataFrame, без списков вручную):
  * 'text__*'            — бинарные токены (мульти-хот). Эмбеддинг текста = X_text @ W
                           (эквивалентно сумме эмбеддингов токенов), опционально / число токенов.
  * dtype 'category'     — категориальные. None/NaN и неизвестные категории -> отдельный уровень.
                           >4 уровней -> эмбеддинг, <=4 -> OneHot.
  * все остальные        — непрерывные. NaN -> среднее по train, затем StandardScaler.

Архитектура:
  1. text -> Linear(T -> text_emb_dim, без bias) [/ n_tokens] -> Linear(->text_out_dim) + act + dropout
  2. категории-эмбеддинги -> Linear(->cat_out_dim) + act + dropout
  3. concat [text, cat, OHE, continuous]
  4. MLP1 (hidden1), concat [MLP1, вход] -> MLP2 (hidden2) -> логит, BCEWithLogitsLoss.

Тюнинг: Optuna на одном фолде (K=5 StratifiedKFold) — только гиперпараметры сети.
Preprocessing и JointDistributionFilter работают с дефолтными параметрами и применяются
к фолду один раз (кроме add_text_binary_matrix=True, он нужен для токенов text__*).
Early stopping по val ROC AUC (patience 10), ReduceLROnPlateau по val loss (patience LR_SCHEDULER_PATIENCE, пока выключен).
"""
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import JointDistributionFilter, Preprocessing

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dont_get_kicked" / "datasets"
TRAIN_PATH = DATA_DIR / "training.csv"
TEST_PATH = DATA_DIR / "test.csv"
CONFIG_PATH = BASE_DIR / "nn_config.json"
CURVE_PATH = BASE_DIR / "nn_learning_curve.png"

RANDOM_STATE = 42
N_SPLITS = 5
OPTUNA_TRIALS = 5
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 100  # > MAX_EPOCHS: ReduceLROnPlateau фактически выключен (пока)
TARGET = "IsBadBuy"

TEXT_PREFIX = "text__"
EMBED_MIN_LEVELS = 4       # уровней (с учётом уровня для None) > 4 -> эмбеддинг, иначе OneHot

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# Фичи
# --------------------------------------------------------------------------- #
class NNFeatures:
    """Превращает DataFrame после Preprocessing + JointDistributionFilter в тензорные блоки."""

    def fit(self, X: pd.DataFrame):
        cols = list(X.columns)
        self.text_cols = [c for c in cols if str(c).startswith(TEXT_PREFIX)]
        self.cat_cols = [c for c in cols if isinstance(X[c].dtype, pd.CategoricalDtype)]
        special = set(self.text_cols) | set(self.cat_cols)
        self.cont_cols = [c for c in cols if c not in special]

        # категории: уровни из train + один дополнительный уровень (последний индекс) для None/NaN/неизвестных
        self.cat_levels = {c: list(X[c].cat.categories) for c in self.cat_cols}
        self.cat_card = {c: len(self.cat_levels[c]) + 1 for c in self.cat_cols}
        self.embed_cols = [c for c in self.cat_cols if self.cat_card[c] > EMBED_MIN_LEVELS]
        self.ohe_cols = [c for c in self.cat_cols if self.cat_card[c] <= EMBED_MIN_LEVELS]

        # непрерывные: NaN -> среднее train, затем стандартизация
        cont = self._cont_frame(X)
        self.cont_means = cont.mean().fillna(0.0)
        self.scaler = StandardScaler().fit(cont.fillna(self.cont_means).to_numpy(dtype=np.float64))
        return self

    def _cont_frame(self, X):
        if not self.cont_cols:
            return pd.DataFrame(index=X.index)
        cont = X[self.cont_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        return cont.replace([np.inf, -np.inf], np.nan)

    def _cat_codes(self, X, col):
        levels = self.cat_levels[col]
        codes = pd.Categorical(X[col].astype(object), categories=levels).codes.astype(np.int64)
        return np.where(codes < 0, len(levels), codes)  # NaN/неизвестные -> отдельный уровень

    def transform(self, X: pd.DataFrame):
        n = len(X)
        if self.text_cols:
            text = (
                X[self.text_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            )
        else:
            text = np.zeros((n, 0), dtype=np.float32)

        cat_ids = {col: self._cat_codes(X, col) for col in self.embed_cols}

        ohe_blocks = []
        for col in self.ohe_cols:
            codes = self._cat_codes(X, col)
            block = np.zeros((n, self.cat_card[col]), dtype=np.float32)
            block[np.arange(n), codes] = 1.0
            ohe_blocks.append(block)
        ohe = np.hstack(ohe_blocks) if ohe_blocks else np.zeros((n, 0), dtype=np.float32)

        cont = self._cont_frame(X).fillna(self.cont_means).to_numpy(dtype=np.float64)
        cont = (
            self.scaler.transform(cont).astype(np.float32)
            if cont.shape[1]
            else np.zeros((n, 0), dtype=np.float32)
        )

        return {
            "text": text,
            "cat_ids": cat_ids,
            "ohe": ohe,
            "cont": cont,
            "text_dim": text.shape[1],
            "embed_cols": list(self.embed_cols),
            "embed_cards": [self.cat_card[c] for c in self.embed_cols],
            "ohe_dim": ohe.shape[1],
            "cont_dim": cont.shape[1],
        }


# --------------------------------------------------------------------------- #
# Данные фолда: Preprocessing и JointDistributionFilter с дефолтными параметрами, один раз на фолд
# --------------------------------------------------------------------------- #
@dataclass
class FoldData:
    X_tr: pd.DataFrame
    X_va: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: np.ndarray
    y_va: np.ndarray
    prep: object = None  # обученный Preprocessing фолда (нужен для сохранения)


def prepare_fold(X, y, X_test, fold):
    tr_idx, va_idx = fold
    prep = Preprocessing(weekday_type = 'category', add_text_binary_matrix=True)
    prep.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    return FoldData(
        X_tr=prep.transform(X.iloc[tr_idx]),
        X_va=prep.transform(X.iloc[va_idx]),
        X_te=prep.transform(X_test),
        y_tr=y.iloc[tr_idx].to_numpy(),
        y_va=y.iloc[va_idx].to_numpy(),
        prep=prep,
    )


def build_features(fold: FoldData):
    jdf = JointDistributionFilter(fold.X_tr, fold.X_te)
    tr = jdf.transform(fold.X_tr)
    va = jdf.transform(fold.X_va)
    builder = NNFeatures().fit(tr)
    return builder.transform(tr), builder.transform(va), builder


# --------------------------------------------------------------------------- #
# Модель
# --------------------------------------------------------------------------- #
def build_mlp(in_dim, hidden, activation, dropout):
    layers = []
    prev = in_dim
    for width in hidden:
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(prev, width))
        layers.append(nn.BatchNorm1d(width))
        layers.append(ACTIVATIONS[activation]())
        prev = width
    return nn.Sequential(*layers)


class NNModel(nn.Module):
    def __init__(self, config, text_dim, embed_cards, ohe_dim, cont_dim, embed_cols):
        super().__init__()
        self.config = config
        self.cat_cols = list(embed_cols)
        activation = config["activation"]
        dropout = float(config["dropout"])
        text_emb_dim = int(config["text_emb_dim"])
        cat_emb_dim = int(config["cat_emb_dim"])
        text_out_dim = int(config.get("text_out_dim", 8))
        cat_out_dim = int(config.get("cat_out_dim", 2))

        # текст: мульти-хот токенов @ W == сумма эмбеддингов токенов
        self.text_in = text_dim
        if text_dim:
            self.text_emb = nn.Linear(text_dim, text_emb_dim, bias=False)
            self.text_out = nn.Sequential(
                nn.Linear(text_emb_dim, text_out_dim), ACTIVATIONS[activation](), nn.Dropout(dropout)
            )
        else:
            text_out_dim = 0

        # категории
        if embed_cards:
            self.cat_embs = nn.ModuleList([nn.Embedding(card, cat_emb_dim) for card in embed_cards])
            self.cat_out = nn.Sequential(
                nn.Linear(cat_emb_dim * len(embed_cards), cat_out_dim),
                ACTIVATIONS[activation](),
                nn.Dropout(dropout),
            )
        else:
            self.cat_embs = nn.ModuleList()
            cat_out_dim = 0

        self.ohe_dim = ohe_dim
        self.cont_dim = cont_dim
        in_dim = text_out_dim + cat_out_dim + ohe_dim + cont_dim
        self.mlp1 = build_mlp(in_dim, config["hidden1"], activation, dropout)
        hidden1_out = config["hidden1"][-1]
        self.mlp2 = build_mlp(hidden1_out + in_dim, config["hidden2"], activation, dropout)
        self.out = nn.Linear(config["hidden2"][-1], 1)

    def forward(self, batch):
        parts = []
        if self.text_in:
            text = batch["text"]
            t = self.text_emb(text)
            if self.config["normalize_text"]:
                t = t / text.sum(dim=1, keepdim=True).clamp(min=1.0)
            parts.append(self.text_out(t))

        if len(self.cat_embs):
            cat_flat = torch.cat(
                [emb(batch["cat_ids"][col]) for emb, col in zip(self.cat_embs, self.cat_cols)], dim=1
            )
            parts.append(self.cat_out(cat_flat))

        if self.ohe_dim:
            parts.append(batch["ohe"])
        if self.cont_dim:
            parts.append(batch["cont"])
        x6 = torch.cat(parts, dim=1)

        h = self.mlp1(x6)
        x7 = torch.cat([h, x6], dim=1)
        return self.out(self.mlp2(x7)).squeeze(-1)


def build_model(config, spec):
    model = NNModel(
        config,
        text_dim=spec["text_dim"],
        embed_cards=spec["embed_cards"],
        ohe_dim=spec["ohe_dim"],
        cont_dim=spec["cont_dim"],
        embed_cols=spec["embed_cols"],
    )
    return model.to(DEVICE)


def to_device(spec, y=None):
    batch = {
        "text": torch.as_tensor(spec["text"], dtype=torch.float32, device=DEVICE),
        "cat_ids": {
            col: torch.as_tensor(ids, dtype=torch.long, device=DEVICE)
            for col, ids in spec["cat_ids"].items()
        },
        "ohe": torch.as_tensor(spec["ohe"], dtype=torch.float32, device=DEVICE),
        "cont": torch.as_tensor(spec["cont"], dtype=torch.float32, device=DEVICE),
    }
    if y is None:
        return batch
    return batch, torch.as_tensor(np.asarray(y), dtype=torch.float32, device=DEVICE)


def evaluate(model, batch, target=None):
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        loss = (
            nn.functional.binary_cross_entropy_with_logits(logits, target).item()
            if target is not None
            else None
        )
        proba = torch.sigmoid(logits).cpu().numpy()
    return loss, proba


def train_nn(model, train_batch, train_target, val_batch, val_target, config,
             max_epochs=MAX_EPOCHS, patience=EARLY_STOP_PATIENCE,
             scheduler_patience=LR_SCHEDULER_PATIENCE, verbose=False, log_every=10,
             early_stopping=True, epoch_callback=None):
    """Обучение сети.

    early_stopping=True  — остановка по val ROC AUC (patience), в конце загружаются веса лучшей эпохи.
    early_stopping=False — всегда ровно max_epochs эпох, веса не откатываются (нужно для обучения
                           на фиксированное число эпох с остановками для предсказаний).
    epoch_callback(epoch, model) — вызывается после каждой эпохи (после валидации); epoch с 1.
                           Модель при вызове в eval-режиме.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.0)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
    )
    loss_fn = nn.BCEWithLogitsLoss()
    cat_cols = list(model.cat_cols)
    n_train = train_batch["text"].shape[0]
    cat_tensor = (
        torch.stack([train_batch["cat_ids"][c] for c in cat_cols], dim=1)
        if cat_cols
        else torch.zeros((n_train, 0), dtype=torch.long, device=DEVICE)
    )
    dataset = TensorDataset(
        train_batch["text"],
        cat_tensor,
        train_batch["ohe"],
        train_batch["cont"],
        train_target,
    )
    loader = DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=True, drop_last=True
    )

    history = {
        "train_loss": [],
        "train_eval_loss": [],
        "val_loss": [],
        "val_auc": [],
        "lr": [],
    }
    best_val_auc = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for text, cat_ids, ohe, cont, yb in loader:
            optimizer.zero_grad()
            batch = {
                "text": text,
                "cat_ids": {c: cat_ids[:, i] for i, c in enumerate(cat_cols)},
                "ohe": ohe,
                "cont": cont,
            }
            logits = model(batch)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
        train_loss = running / len(dataset)

        train_eval_loss, _ = evaluate(model, train_batch, train_target)
        val_loss, val_proba = evaluate(model, val_batch, val_target)
        val_auc = roc_auc_score(val_target.cpu().numpy(), val_proba)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_eval_loss"].append(train_eval_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(
                f"    epoch {epoch:4d} | train_loss={train_loss:.5f} | "
                f"train_eval_loss={train_eval_loss:.5f} | "
                f"val_loss={val_loss:.5f} | val_auc={val_auc:.5f} | lr={current_lr:.2e}"
            )

        if epoch_callback is not None:
            epoch_callback(epoch, model)

        if val_auc > best_val_auc + 1e-6:
            best_val_auc = val_auc
            if early_stopping:
                best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if early_stopping and bad_epochs >= patience:
                if verbose:
                    print(f"    early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    if early_stopping:
        model.load_state_dict(best_state)
    return model, history, best_epoch


# --------------------------------------------------------------------------- #
# Optuna
# --------------------------------------------------------------------------- #
def sample_config(trial):
    return {
        "text_emb_dim": trial.suggest_categorical("text_emb_dim", [64, 128, 256]),
        "cat_emb_dim": trial.suggest_categorical("cat_emb_dim", [32, 64, 128]),
        "normalize_text": trial.suggest_categorical("normalize_text", [True, False]),
        "activation": trial.suggest_categorical("activation", list(ACTIVATIONS)),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "text_out_dim": trial.suggest_categorical("text_out_dim", [8, 16, 32, 64]),
        "cat_out_dim": trial.suggest_categorical("cat_out_dim", [4, 8, 16]),
        "hidden1": trial.suggest_categorical("hidden1", [[256, 128], [128, 64]]),
        "hidden2": trial.suggest_categorical("hidden2", [[256, 128], [128, 64]]),
        "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }


def tune_on_single_fold(fold: FoldData, n_trials):
    print(f"tuning {n_trials} trials on fold (train={len(fold.X_tr)}, val={len(fold.X_va)})...")
    # признаки одинаковы для всех триалов — считаем один раз
    train_spec, va_spec, _ = build_features(fold)
    train_batch, train_target = to_device(train_spec, fold.y_tr)
    va_batch, va_target = to_device(va_spec, fold.y_va)
    print(
        f"features: text={train_spec['text_dim']} emb_cat={len(train_spec['embed_cols'])} "
        f"ohe={train_spec['ohe_dim']} cont={train_spec['cont_dim']}"
    )

    def objective(trial):
        start = time.time()
        config = sample_config(trial)
        trial.set_user_attr("config", config)
        torch.manual_seed(RANDOM_STATE)

        model = build_model(config, train_spec)
        model, _, _ = train_nn(model, train_batch, train_target, va_batch, va_target, config)

        _, proba = evaluate(model, va_batch, va_target)
        auc = roc_auc_score(fold.y_va, proba)
        print(f"trial {trial.number}: AUC={auc:.5f} | {time.time() - start:.1f}s | {config}")
        return auc

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def plot_loss_curves(history, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train loss (batches, dropout on)", color="C2")
    ax.plot(epochs, history["train_eval_loss"], label="train loss (eval mode)", color="C0")
    ax.plot(epochs, history["val_loss"], label="validation loss", color="C1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title("Neural network training vs validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"learning curve saved to {path}")


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train["PurchDate"] = pd.to_datetime(train["PurchDate"])
    test["PurchDate"] = pd.to_datetime(test["PurchDate"])
    y = train[TARGET]
    X = train.drop(columns=[TARGET, "RefId"])
    X_test = test.drop(columns=["RefId"])
    return X, y, X_test, test["RefId"]


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    X, y, X_test, _ = load_data()
    print(f"train: {X.shape}, test: {X_test.shape}, target ratio: {y.mean():.4f}")
    print(f"device: {DEVICE}")

    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    )
    fold = prepare_fold(X, y, X_test, folds[0])

    study = tune_on_single_fold(fold, OPTUNA_TRIALS)
    best_config = study.best_trial.user_attrs["config"]
    print(f"best tuning AUC (fold 0 holdout): {study.best_value:.5f}")
    print(f"best params: {best_config}")

    train_spec, va_spec, _ = build_features(fold)
    train_batch, train_target = to_device(train_spec, fold.y_tr)
    va_batch, va_target = to_device(va_spec, fold.y_va)

    torch.manual_seed(RANDOM_STATE)
    model = build_model(best_config, train_spec)
    print("training best model on fold 0...")
    model, history, best_epoch = train_nn(
        model, train_batch, train_target, va_batch, va_target, best_config, verbose=True
    )
    val_loss, proba = evaluate(model, va_batch, va_target)
    print(
        f"fold 0 best model: epoch={best_epoch}, val_loss={val_loss:.5f}, "
        f"AUC={roc_auc_score(fold.y_va, proba):.5f}"
    )
    plot_loss_curves(history, CURVE_PATH)

    config_out = dict(best_config)
    config_out["best_tuning_auc"] = study.best_value
    config_out["best_epoch"] = best_epoch
    config_out["n_trials"] = OPTUNA_TRIALS
    config_out["max_epochs"] = MAX_EPOCHS
    config_out["early_stop_patience"] = EARLY_STOP_PATIENCE
    config_out["lr_scheduler_patience"] = LR_SCHEDULER_PATIENCE
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"config saved to {CONFIG_PATH}")


if __name__ == "__main__":
    main()