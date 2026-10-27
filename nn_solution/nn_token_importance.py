"""Важность текстовых токенов нейросети + UMAP их эмбеддингов.

Шаги:
  1. Читает лучший конфиг из nn_config.json (сохраняется nn_solution.py): гиперпараметры и best_epoch.
  2. Строит признаки на ВСЁМ трейне (Preprocessing -> JointDistributionFilter, как в фолдах)
     и обучает сеть на всём датасете без валидации ровно best_epoch * epoch_coef эпох.
  3. Важность токена: берётся вся таблица трейна, колонка токена text__X = 1, все остальные
     text__* = 0, остальные признаки (категории, непрерывные) не меняются; таблица прогоняется
     через сеть, предсказанные вероятности IsBadBuy усредняются по строкам.
     Дополнительно считается baseline (все text__* = 0) и importance_delta = importance - baseline.
  4. Частота токена: доля (и число) строк трейна, где токен встречается.
  5. Эмбеддинг токена: столбец весов text_emb (Linear T -> text_emb_dim) для этого токена.
     Эмбеддинги проецируются в 2D через UMAP; цвет точки — важность, размер — частота
     (размер ~ частоте, поэтому частые токены сильно крупнее редких).

None в категориальных колонках Preprocessing кодирует строкой 'init_None', которая в тексте
даёт токены 'init' и 'none' — они присутствуют в таблице и на графике как обычные токены.

Результаты в подпапке nn_solution/token_importance:
  token_importance.csv    — token, count, freq, importance, importance_delta, umap_x, umap_y
  token_embeddings.csv    — token + эмбеддинг (emb_0 ... emb_{d-1})
  token_umap.png          — scatterplot
  nn_full_model.pt        — веса обученной сети, конфиг, список токенов
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import Normalize
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


import nn_solution.nn_tuning as nn  # импорт как часть пакета: import nn_solution.nn_token_importance

from preprocessing.preprocessing import JointDistributionFilter, Preprocessing


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "nn_config.json"
OUT_DIR = BASE_DIR / "token_importance"

TEXT_PREFIX = nn.TEXT_PREFIX
METADATA_KEYS = {
    "best_tuning_auc",
    "best_epoch",
    "n_trials",
    "max_epochs",
    "early_stop_patience",
    "lr_scheduler_patience",
}


# --------------------------------------------------------------------------- #
# Конфиг, признаки, обучение на всём датасете
# --------------------------------------------------------------------------- #
def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    if "best_epoch" not in cfg:
        raise KeyError("в nn_config.json нет best_epoch — перезапусти nn_solution.py")
    model_cfg = {k: v for k, v in cfg.items() if k not in METADATA_KEYS}
    return model_cfg, int(cfg["best_epoch"])


def build_full_features(X, y, X_test):
    """Preprocessing -> JointDistributionFilter (train+test профили) -> NNFeatures на всём трейне."""
    prep = Preprocessing(add_text_binary_matrix=True)
    prep.fit(X, y)
    X_p = prep.transform(X)
    X_te_p = prep.transform(X_test)

    jdf = JointDistributionFilter(X_p, X_te_p)
    X_f = jdf.transform(X_p)

    builder = nn.NNFeatures().fit(X_f)
    spec = builder.transform(X_f)
    return spec, builder


def fit_full(model_cfg, spec, y, epochs, log_every=5):
    """Обучение на всех данных без валидации: ровно `epochs` эпох, постоянный LR."""
    torch.manual_seed(nn.RANDOM_STATE)
    model = nn.build_model(model_cfg, spec)
    batch, target = nn.to_device(spec, y)

    cat_cols = list(model.cat_cols)
    n = batch["text"].shape[0]
    cat_tensor = (
        torch.stack([batch["cat_ids"][c] for c in cat_cols], dim=1)
        if cat_cols
        else torch.zeros((n, 0), dtype=torch.long, device=nn.DEVICE)
    )
    dataset = TensorDataset(batch["text"], cat_tensor, batch["ohe"], batch["cont"], target)
    loader = DataLoader(
        dataset, batch_size=int(model_cfg["batch_size"]), shuffle=True, drop_last=True
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_cfg["lr"], weight_decay=model_cfg.get("weight_decay", 0.0)
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for text, cat_ids, ohe, cont, yb in loader:
            optimizer.zero_grad()
            b = {
                "text": text,
                "cat_ids": {c: cat_ids[:, i] for i, c in enumerate(cat_cols)},
                "ohe": ohe,
                "cont": cont,
            }
            loss = loss_fn(model(b), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs} | train_loss={running / len(dataset):.5f}")

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(batch)).cpu().numpy()
    print(f"in-sample AUC (на обучающих данных, только sanity check): "
          f"{roc_auc_score(np.asarray(y), proba):.5f}")
    return model, batch


# --------------------------------------------------------------------------- #
# Важность токенов
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_token_importance(model, batch, text_dim, chunk=20000):
    """Для каждого токена j: text = one-hot(j), остальное без изменений, средняя вероятность.

    Возвращает (importance[text_dim], baseline), baseline — все text__* = 0.
    """
    model.eval()
    n = batch["cont"].shape[0]
    sums = torch.zeros(text_dim + 1, dtype=torch.float64)  # последняя ячейка — baseline

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        text = torch.zeros((stop - start, text_dim), device=nn.DEVICE)
        sub = {
            "text": text,
            "ohe": batch["ohe"][start:stop],
            "cont": batch["cont"][start:stop],
            "cat_ids": {c: v[start:stop] for c, v in batch["cat_ids"].items()},
        }
        sums[text_dim] += torch.sigmoid(model(sub)).sum().double().cpu()
        for j in range(text_dim):
            text[:, j] = 1.0
            sums[j] += torch.sigmoid(model(sub)).sum().double().cpu()
            text[:, j] = 0.0
        print(f"  importance: строки {stop}/{n}")

    means = (sums / n).numpy()
    return means[:text_dim], float(means[text_dim])


# --------------------------------------------------------------------------- #
# UMAP и график
# --------------------------------------------------------------------------- #
def project_umap(embeddings, n_neighbors=15, min_dist=0.1, metric="cosine", seed=None):
    import umap  # pip install umap-learn

    n = len(embeddings)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(n_neighbors, n - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=nn.RANDOM_STATE if seed is None else seed,
    )
    return reducer.fit_transform(embeddings)


def plot_token_umap(df, path, top_freq_labels=25, top_importance_labels=5,
                    s_min=6.0, s_max=1800.0, title=None):
    """Scatter: цвет — importance, размер — freq (площадь ~ частоте, частые токены крупные)."""
    freq = df["freq"].to_numpy(dtype=float)
    imp = df["importance"].to_numpy(dtype=float)
    f_norm = freq / freq.max()

    def size_of(f):
        return s_min + (s_max - s_min) * (np.asarray(f, dtype=float) / freq.max())

    sizes = size_of(freq)

    cmap = plt.get_cmap("viridis")
    vmin, vmax = np.percentile(imp, [2, 98])
    if vmax <= vmin:
        vmin, vmax = imp.min(), imp.max() + 1e-12
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cmap(norm(imp))
    rgba[:, 3] = 0.35 + 0.6 * np.sqrt(f_norm)  # редкие токены ещё и бледнее

    # крупные точки рисуем первыми, мелкие поверх — чтобы они не терялись под большими
    order = np.argsort(-freq)
    x = df["umap_x"].to_numpy()
    y = df["umap_y"].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.scatter(
        x[order], y[order], s=sizes[order], c=rgba[order],
        edgecolors="black", linewidths=(0.2 + 0.8 * f_norm[order]),
    )

    # подписи: самые частые + самые важные/неважные токены
    label_idx = set(np.argsort(-freq)[:top_freq_labels].tolist())
    label_idx |= set(np.argsort(-imp)[:top_importance_labels].tolist())
    label_idx |= set(np.argsort(imp)[:top_importance_labels].tolist())
    for i in label_idx:
        ax.annotate(df["token"].iloc[i], (x[i], y[i]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.85)
    cbar.set_label("Важность токена (средняя предсказанная вероятность IsBadBuy)")

    ref = np.unique(np.quantile(freq, [0.0, 0.5, 0.9, 1.0]))
    handles = [ax.scatter([], [], s=size_of(f), c="lightgray", edgecolors="black",
                          linewidths=0.2 + 0.8 * f / freq.max(), label=f"{f:.2%}") for f in ref]
    ax.legend(handles=handles, title="Частота токена", loc="upper left",
              labelspacing=1.6, borderpad=1.0, framealpha=0.9)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title or "Эмбеддинги текстовых токенов (UMAP): цвет — важность, размер — частота")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"график сохранён: {path}")
    return fig


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(epoch_coef=1.0, umap_neighbors=15, umap_min_dist=0.1, umap_metric="cosine",
         top_freq_labels=25, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg, best_epoch = load_config(CONFIG_PATH)
    epochs = max(1, int(round(best_epoch * epoch_coef)))
    print(f"config: {CONFIG_PATH}\nbest_epoch={best_epoch}, epoch_coef={epoch_coef} -> epochs={epochs}")
    print(f"model config: {model_cfg}")

    np.random.seed(nn.RANDOM_STATE)
    X, y, X_test, _ = nn.load_data()
    y = y.to_numpy()
    print(f"train: {X.shape}, test: {X_test.shape}, device: {nn.DEVICE}")

    # --- 1) признаки на всём трейне и обучение без валидации ---
    spec, builder = build_full_features(X, pd.Series(y, index=X.index), X_test)
    text_dim = spec["text_dim"]
    if text_dim == 0:
        raise RuntimeError("после JointDistributionFilter не осталось колонок text__*")
    tokens = [c[len(TEXT_PREFIX):] for c in builder.text_cols]
    print(f"токенов: {text_dim}; text={text_dim} emb_cat={len(spec['embed_cols'])} "
          f"ohe={spec['ohe_dim']} cont={spec['cont_dim']}")

    model, batch = fit_full(model_cfg, spec, y, epochs)
    torch.save(
        {"state_dict": model.state_dict(), "model_cfg": model_cfg, "epochs": epochs, "tokens": tokens},
        out_dir / "nn_full_model.pt",
    )

    # --- 2) важность и частота токенов ---
    importance, baseline = compute_token_importance(model, batch, text_dim)
    counts = spec["text"].sum(axis=0).astype(int)
    df = pd.DataFrame(
        {
            "token": tokens,
            "count": counts,
            "freq": counts / len(y),
            "importance": importance,
            "importance_delta": importance - baseline,
        }
    )
    print(f"baseline (все text__* = 0): {baseline:.5f}")
    df.sort_values("importance", ascending=False).to_csv(
        out_dir / "token_importance.csv", index=False
    )

    # --- 3) эмбеддинги токенов: столбцы весов text_emb (Linear: out x in) ---
    emb = model.text_emb.weight.detach().cpu().numpy().T  # (text_dim, text_emb_dim)
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    emb_df.insert(0, "token", tokens)
    emb_df.to_csv(out_dir / "token_embeddings.csv", index=False)

    # --- 4) UMAP + scatterplot ---
    try:
        xy = project_umap(emb, umap_neighbors, umap_min_dist, umap_metric)
    except ImportError:
        print("UMAP не установлен: pip install umap-learn. CSV с важностью и эмбеддингами уже "
              "сохранены, повтори построение графика после установки.")
        return df
    df["umap_x"], df["umap_y"] = xy[:, 0], xy[:, 1]
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "token_importance.csv", index=False)
    plot_token_umap(df, out_dir / "token_umap.png", top_freq_labels=top_freq_labels)

    print(f"\nтоп-10 по важности:\n{df.head(10).to_string(index=False)}")
    print(f"\nнижние 10 по важности:\n{df.tail(10).to_string(index=False)}")
    print(f"\nрезультаты сохранены в {out_dir}")
    return df


if __name__ == "__main__":
    main()