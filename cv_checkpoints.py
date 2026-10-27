"""Адаптивные чекпоинты по числу итераций (деревьев / эпох) для k-fold OOF.

Идея. Все k моделей обучаются синхронно (в lockstep): после каждой остановки считается CV ROC AUC по
OOF-предсказаниям всех фолдов. Пока максимум CV AUC не обновляется, чекпоинты ставятся на итерации
round(coef * текущая_лучшая_итерация); коэффициенты >= 1.0, поэтому все чекпоинты не раньше лучшей итерации.
Как только CV AUC превысил максимум, лучшая итерация обновляется, старые чекпоинты отбрасываются
и ставятся новые. Обучение заканчивается, когда сохранены все чекпоинты для итоговой лучшей итерации и
пройден горизонт max(ceil(best_iter * max(max(coefs), min_horizon)), best_iter + min_patience).
min_patience — абсолютный минимум терпения: без него при маленькой лучшей итерации (например, лучшая
эпоха 1 и коэффициент 1.5 дают горизонт 2) обучение остановилось бы на первом же шумном провале.

Модуль кладётся рядом с preprocessing.py (корень проекта).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import roc_auc_score

# папки результатов на уровне preprocessing.py (корень проекта)
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"          # веса моделей, объекты предобработки (joblib), манифесты
OOF_DIR = ROOT_DIR / "oof_predictions"    # OOF и тестовые предсказания всех моделей


class CheckpointTracker:
    def __init__(self, coefs, min_horizon: float = 1.5, min_patience: int = 0):
        coefs = tuple(float(c) for c in coefs)
        if not coefs:
            raise ValueError("пустой кортеж коэффициентов")
        if min(coefs) < 1.0:
            raise ValueError(
                f"все коэффициенты должны быть >= 1.0 (получено {min(coefs)}): "
                "чекпоинты считаются от лучшей на текущий момент итерации и не могут быть раньше неё"
            )
        self.coefs = coefs
        self.min_horizon = float(min_horizon)
        self.min_patience = int(min_patience)
        self.history = []        # [(итерация, cv_auc)]
        self.best_auc = -np.inf
        self.best_iter = 0
        self.targets = {}        # coef -> итерация чекпоинта (для текущей лучшей итерации)
        self.saved = {}          # итерация -> payload (OOF, test, ...)

    def observe(self, t: int, auc: float) -> bool:
        """Записывает CV AUC на итерации t. True, если это новый максимум."""
        self.history.append((int(t), float(auc)))
        if auc > self.best_auc:
            self.best_auc, self.best_iter = float(auc), int(t)
            self.targets = {c: max(int(t), int(round(c * t))) for c in self.coefs}
            self.saved = {}  # прежние чекпоинты относились к прежней лучшей итерации
            return True
        return False

    def pending(self, t: int):
        """Ещё не сохранённые чекпоинты строго после итерации t (по возрастанию)."""
        return sorted({x for x in self.targets.values() if x > t and x not in self.saved})

    def needs_snapshot(self, t: int) -> bool:
        return t in self.targets.values() and t not in self.saved

    def store(self, t: int, payload: dict) -> None:
        self.saved[int(t)] = payload

    def horizon(self) -> int:
        by_coef = int(math.ceil(self.best_iter * max(max(self.coefs), self.min_horizon)))
        return max(by_coef, self.best_iter + self.min_patience)

    def finished(self, t: int) -> bool:
        return (
            bool(self.targets)
            and all(x in self.saved for x in self.targets.values())
            and t >= self.horizon()
        )

    def missing_targets(self):
        return {c: x for c, x in self.targets.items() if x not in self.saved}


def run_adaptive(coefs, advance, evaluate, predict_test, y, eval_every=1, max_iter=10000,
                 min_horizon=1.5, log_every=1, label="iter", lazy_test=False, min_patience=0,
                 extra_snapshot=None):
    """Синхронное обучение k моделей с адаптивными чекпоинтами.

    advance(t_from, t_to)  — довести все модели от t_from до t_to итераций;
    evaluate(t)            — (oof, fold_aucs): OOF-вектор по всем объектам трейна на итерации t
                             и список AUC по фолдам;
    predict_test(t)        — усреднённое по фолдам предсказание теста на итерации t;
    y                      — метки трейна (для CV AUC по всему OOF);
    min_patience           — минимум итераций после лучшей, прежде чем можно остановиться;
    extra_snapshot(t)      — необязательный: что ещё запомнить на чекпоинте (кладётся в payload["extra"],
                             отбрасывается вместе с устаревшим чекпоинтом; нужно для весов нейросети);
    eval_every             — шаг сетки, на которой считается CV AUC (чекпоинты добавляются сверх сетки);
    lazy_test              — True: предсказания теста считаются один раз в конце только для итоговых
                             чекпоинтов (годится для бустинга: модель хранит все деревья и умеет
                             предсказывать по первым t деревьям). False: сразу на чекпоинте (нужно для
                             нейросети: веса меняются с эпохами).
    Возвращает CheckpointTracker (history, best_iter, targets, saved).
    """
    y = np.asarray(y)
    tracker = CheckpointTracker(coefs, min_horizon, min_patience)
    t, n_eval = 0, 0

    while True:
        nxt = (t // eval_every + 1) * eval_every
        pend = tracker.pending(t)
        if pend:
            nxt = min(nxt, pend[0])
        nxt = min(nxt, max_iter)

        advance(t, nxt)
        t = nxt
        oof, fold_aucs = evaluate(t)
        auc = float(roc_auc_score(y, oof))
        new_best = tracker.observe(t, auc)
        n_eval += 1

        if tracker.needs_snapshot(t):
            payload = {"oof": np.array(oof, copy=True),
                       "test": None if lazy_test else predict_test(t),
                       "fold_aucs": list(fold_aucs), "cv_auc": auc}
            if extra_snapshot is not None:
                payload["extra"] = extra_snapshot(t)
            tracker.store(t, payload)
            if not new_best:  # на новом максимуме чекпоинт coef=1.0 перезаписывается постоянно — не шумим
                print(f"  [checkpoint] {label}={t}: CV AUC={auc:.5f} "
                      f"(best {tracker.best_auc:.5f} @ {tracker.best_iter})")
        if n_eval % log_every == 0:
            mark = " *new best*" if new_best else ""
            print(f"  {label}={t}: CV AUC={auc:.5f}, best={tracker.best_auc:.5f} @ {tracker.best_iter}, "
                  f"horizon={tracker.horizon()}{mark}")

        if tracker.finished(t):
            break
        if t >= max_iter:
            miss = tracker.missing_targets()
            print(f"  [warning] достигнут max_iter={max_iter}, не сохранены чекпоинты: {miss}")
            break

    if lazy_test:
        for it in sorted(tracker.saved):
            tracker.saved[it]["test"] = predict_test(it)
    return tracker


def history_frame(tracker) -> pd.DataFrame:
    return pd.DataFrame(tracker.history, columns=["iteration", "cv_roc_auc"])


def plot_cv_curve(tracker, path, title="CV ROC AUC vs iterations", xlabel="iteration"):
    """График CV ROC AUC от числа итераций: максимум и сохранённые чекпоинты отмечены."""
    import matplotlib.pyplot as plt

    it = np.array([h[0] for h in tracker.history])
    auc = np.array([h[1] for h in tracker.history])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(it, auc, lw=1.6, color="tab:blue", label="CV ROC AUC (OOF по всем фолдам)")

    lo = max(auc.min(), tracker.best_auc - 0.05)
    hi = tracker.best_auc + 0.003
    ax.set_ylim(lo, hi)

    ax.scatter([tracker.best_iter], [tracker.best_auc], marker="*", s=260, c="red", zorder=6,
               label=f"максимум {tracker.best_auc:.5f} @ {tracker.best_iter}")

    coef_of = {}
    for c, x in tracker.targets.items():
        coef_of.setdefault(x, []).append(c)
    for x, cs in sorted(coef_of.items()):
        if x not in tracker.saved:
            continue
        ax.axvline(x, ls="--", lw=0.8, color="gray", alpha=0.6)
        y_x = tracker.saved[x]["cv_auc"]
        ax.scatter([x], [y_x], s=36, c="tab:orange", zorder=5)
        ax.annotate("×" + "/".join(f"{c:g}" for c in cs), (x, y_x), xytext=(3, 8),
                    textcoords="offset points", fontsize=8, rotation=90, va="bottom")

    ax.scatter([], [], s=36, c="tab:orange", label="чекпоинты (coef × лучшая итерация)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CV ROC AUC")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"график сохранён: {path}")