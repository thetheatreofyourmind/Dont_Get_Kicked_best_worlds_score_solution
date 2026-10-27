"""Preprocessing для dont_get_kick (на основе xgb_solution) + JointDistributionFilter.

Preprocessing — признаки (price_factor, text_factor, Date_factor, REL_*, флаги is_trash,
опционально бинарная матрица токенов text__*).
JointDistributionFilter — двусторонняя фильтрация редких значений/категорий/токенов по
профилям train и test + генератор графиков распределений (.distributions).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

MMR_COLS = [
    "MMRAcquisitionAuctionAveragePrice",
    "MMRAcquisitionAuctionCleanPrice",
    "MMRAcquisitionRetailAveragePrice",
    "MMRAcquisitonRetailCleanPrice",
    "MMRCurrentAuctionAveragePrice",
    "MMRCurrentAuctionCleanPrice",
    "MMRCurrentRetailAveragePrice",
    "MMRCurrentRetailCleanPrice",
]
NUMER_COLS = MMR_COLS + ["WarrantyCost", "VehBCost"]
CAT_COLS = [
    "Auction",
    "BYRNO",
    "VNZIP1",
    "VNST",
    "VehYear",
    "Make",
    "Model",
    "Trim",
    "SubModel",
    "WheelType",
    "Size",
    "Color",
    "PRIMEUNIT",
]
TEXT_COLS = ["Make", "Model", "Trim", "SubModel"]

DEFAULT_JDF_NUM_COLS = [
    "WarrantyCost",
    "VehBCost",
    "REL_DRIFT_RET_CLEAN",
    "REL_DRIFT_RET_AVG",
    "REL_DRIFT_AUC_AVG",
    "REL_WarrantyCost",
    "REL_VehBCost",
    "VehOdo",
    "VEHODO/VEHICLEAGE",
]
DEFAULT_JDF_CAT_COLS = list(CAT_COLS)

TOKEN_RE = re.compile(r"[a-z0-9]+")
PURCH_DATE_ORIGIN = pd.Timestamp("2009-01-05")

BLUE, ORANGE = "steelblue", "orange"


class FeatureImportanceDropper(BaseEstimator, TransformerMixin):

    def __init__(self, importance_path: str = 'D:/DS_Proj/importance_df.csv', n_worst: int = 170, n_best=None):
        self.importance_path = importance_path
        self.n_worst = n_worst
        self.n_best = n_best

    def fit(self, X: pd.DataFrame, y=None):
        imp = pd.read_csv(self.importance_path)
        imp = imp.loc[:, ["Feature", "Importance"]]
        imp = imp.sort_values("Importance", ascending=True).reset_index(drop=True)
        self.features_to_drop_ = imp["Feature"].iloc[: self.n_worst].tolist() + \
            (imp["Feature"].iloc[-self.n_best:].tolist() if self.n_best else [])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.features_to_drop_ if c in X.columns]
        return X.drop(columns=cols)


class Preprocessing(BaseEstimator, TransformerMixin):
    """Базовый препроцессинг для dont_get_kick.

    Параметры
    ---------
    mmr_cols : list of str
        Колонки цен MMR (по умолчанию все 8 MMR-признаков).
    numer_cols : list of str
        Все числовые колонки (mmr_cols + WarrantyCost + VehBCost).
    cat_cols : list of str
        Категориальные колонки.
    text_cols : list of str
        Текстовые колонки (Make, Model, Trim, SubModel).
    price_trash_threshold : float
        Порог для значений ценовых колонок (ниже — мусор).
    cramer_v_min : float
        Минимальный Cramer's V для флагов is_trash (коррелированные удаляются).
    price_pipeline_degree : int
        Степень PolynomialFeatures для price_pipeline.
    price_pipeline_alpha : float
        Зарезервирован (в price_pipeline используется LogisticRegression, C=100).
    text_min_count_train : int или float
        Минимальный count (int) или доля (float) токена в трейне.
    text_logreg_C : float
        C для логистической регрессии text_factor.
    add_text_binary_matrix : bool
        Добавлять ли бинарную матрицу токенов (колонки text__*) в датасет.
    date_pipeline_degree : int
        Степень PolynomialFeatures для Date_factor.
    date_pipeline_alpha : float
        alpha для Ridge в Date_factor.
    date_days : int
        Дни в одном отрезке для date_features.
    min_flag_count : int
        Минимальное количество вхождений флага is_trash в трейне.
    """

    def __init__(
        self,
        mmr_cols: Optional[list] = None,
        numer_cols: Optional[list] = None,
        cat_cols: Optional[list] = None,
        text_cols: Optional[list] = None,
        price_trash_threshold: float = 100.0,
        cramer_v_min: float = 0.5,
        price_pipeline_degree: int = 3,
        price_pipeline_alpha: float = 0.007,
        text_min_count_train: int = 100,
        text_logreg_C: float = 65.0,
        add_text_binary_matrix: bool = False,
        date_pipeline_degree: int = 13,
        date_pipeline_alpha: float = 0.007,
        date_days: int = 14,
        min_flag_count: int = 500,
        weekday_type = 'category',
    ):
        self.mmr_cols = list(mmr_cols) if mmr_cols is not None else list(MMR_COLS)
        self.numer_cols = list(numer_cols) if numer_cols is not None else list(NUMER_COLS)
        self.cat_cols = list(cat_cols) if cat_cols is not None else list(CAT_COLS)
        self.text_cols = list(text_cols) if text_cols is not None else list(TEXT_COLS)
        self.price_trash_threshold = price_trash_threshold
        self.cramer_v_min = cramer_v_min
        self.price_pipeline_degree = price_pipeline_degree
        self.price_pipeline_alpha = price_pipeline_alpha
        self.text_min_count_train = text_min_count_train
        self.text_logreg_C = text_logreg_C
        self.add_text_binary_matrix = add_text_binary_matrix
        self.date_pipeline_degree = date_pipeline_degree
        self.date_pipeline_alpha = date_pipeline_alpha
        self.date_days = date_days
        self.min_flag_count = min_flag_count
        self.weekday_type = weekday_type
        # Сохраняемые после fit атрибуты
        self.numer_cols_means = {}
        self.text_vocab = set()
        self.text_model = None
        self.text_vector_names = []
        self.price_pipeline = None
        self.date_ppl = None
        self.trash_flag_cols = []

    # ------------------------------------------------------------------ #
    # Вспомогательные утилиты
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cramers_v(x, y):
        """Cramer's V для двух бинарных/категориальных векторов."""
        x = np.asarray(x)
        y = np.asarray(y)
        confusion = pd.crosstab(x, y)
        try:
            from scipy.stats import chi2_contingency

            chi2, _, _, _ = chi2_contingency(confusion.values)
        except Exception:
            chi2 = 0.0
        n = confusion.values.sum()
        phi2 = chi2 / n
        r, k = confusion.shape
        phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min((kcorr - 1), (rcorr - 1))
        if denom <= 0:
            return 0.0
        return np.sqrt(phi2corr / denom)

    @staticmethod
    def _build_text(X):
        return (
            X["Make"].fillna("").astype(str)
            + " "
            + X["Model"].fillna("").astype(str)
            + " "
            + X["Trim"].fillna("").astype(str)
            + " "
            + X["SubModel"].fillna("").astype(str)
        ).str.lower()

    @classmethod
    def _token_lists(cls, X):
        """Уникальные токены текста для каждой строки (с сохранением порядка)."""
        text = cls._build_text(X)
        return text.map(lambda s: list(dict.fromkeys(TOKEN_RE.findall(s)))).tolist()

    def _token_matrix(self, token_lists):
        """Бинарная матрица (n_rows x n_vocab, uint8) по словарю text_vector_names."""
        index = {t: j for j, t in enumerate(self.text_vector_names)}
        matrix = np.zeros((len(token_lists), len(index)), dtype=np.uint8)
        for i, toks in enumerate(token_lists):
            for t in toks:
                j = index.get(t)
                if j is not None:
                    matrix[i, j] = 1
        return matrix

    def _trash_mask(self, s):
        return (s < self.price_trash_threshold) | s.isna()

    def _clean_numeric(self, X):
        """Копия числовых колонок: мусор/NaN заменены средним по трейну."""
        X_num = X[self.numer_cols].copy().astype(float)
        for col in self.numer_cols:
            X_num.loc[self._trash_mask(X_num[col]), col] = self.numer_cols_means[col]
        return X_num

    def _date_frame(self, X):
        return pd.DataFrame(
            {"PurchDate": (X["PurchDate"] - PURCH_DATE_ORIGIN) // pd.Timedelta(days=self.date_days)}
        )

    def _fill_cat_none(self, X):
        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna("init_None").astype(str)
        return X

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        X = X.copy()
        X["PurchDate"] = pd.to_datetime(X["PurchDate"])
        y_arr = y.values.astype(float) if hasattr(y, "values") else np.asarray(y, dtype=float)

        # --- 1) Чистка ценовых колонок: средние по "нормальным" значениям ---
        self.numer_cols_means = {}
        for col in self.numer_cols:
            self.numer_cols_means[col] = X.loc[~self._trash_mask(X[col]), col].mean()

        # --- 2) Флаги is_trash + отбор по Cramer's V и частоте ---
        trash_flags = {
            f"{col}_is_trash": self._trash_mask(X[col]).astype(int) for col in self.numer_cols
        }
        flag_cols = list(trash_flags.keys())
        keep_flags = list(flag_cols)
        for i, c1 in enumerate(flag_cols):
            for c2 in flag_cols[i + 1:]:
                v = self._cramers_v(trash_flags[c1], trash_flags[c2])
                if v > self.cramer_v_min:
                    if trash_flags[c1].mean() < trash_flags[c2].mean():
                        if c1 in keep_flags:
                            keep_flags.remove(c1)
                    else:
                        if c2 in keep_flags:
                            keep_flags.remove(c2)
        for c in list(keep_flags):
            if trash_flags[c].sum() < self.min_flag_count:
                keep_flags.remove(c)
        self.trash_flag_cols = keep_flags

        # --- 3) price_pipeline ---
        X_num = self._clean_numeric(X)
        self.price_pipeline = Pipeline(
            [
                ("SS1", StandardScaler()),
                ("PT", PolynomialFeatures(self.price_pipeline_degree, include_bias=False)),
                ("SS2", StandardScaler()),
                ("LR", LogisticRegression(C=100, max_iter=10000)),
            ]
        ).fit(X_num, y_arr)

        # --- 4) Date_factor ---
        self.date_ppl = Pipeline(
            [
                ("ss1", StandardScaler()),
                ("pt", PolynomialFeatures(degree=self.date_pipeline_degree, include_bias=False)),
                ("ss2", StandardScaler()),
                ("lr", Ridge(alpha=self.date_pipeline_alpha)),
            ]
        ).fit(self._date_frame(X), y_arr)

        # --- 5) Текст: словарь токенов + LogReg text_factor ---
        # None в категориальных колонках -> 'init_None' (как и в transform)
        Xp = self._fill_cat_none(X.copy())
        token_lists = self._token_lists(Xp)
        token_counts = Counter(t for toks in token_lists for t in toks)
        thr = self.text_min_count_train
        if isinstance(thr, float):
            thr = thr * len(Xp)
        self.text_vocab = {t for t, c in token_counts.items() if c >= thr}
        self.text_vector_names = sorted(self.text_vocab)

        if self.text_vector_names:
            token_matrix = self._token_matrix(token_lists)
            self.text_model = LogisticRegression(C=self.text_logreg_C, max_iter=5000)
            self.text_model.fit(token_matrix, y_arr)
        else:
            self.text_model = None

        return self

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #
    def transform(self, X):
        X = X.copy()
        X["PurchDate"] = pd.to_datetime(X["PurchDate"])
        X = self._fill_cat_none(X)

        X_trans = self._clean_numeric(X)
        divisor = X_trans["MMRAcquisitionAuctionCleanPrice"].copy()
        X_trans["price_factor"] = self.price_pipeline.predict_proba(X_trans[self.numer_cols])[:, 1]

        # Категориальные колонки (None уже заменён на 'init_None')
        for col in self.cat_cols:
            if col in X.columns:
                X_trans[col] = X[col]

        # Бинарные флаги is_trash для ценовых колонок
        for col_base in self.numer_cols:
            flag_name = f"{col_base}_is_trash"
            if flag_name in self.trash_flag_cols:
                X_trans[flag_name] = self._trash_mask(X[col_base]).astype(int)

        # Токены текста: один раз для OHE-матрицы и text_factor
        token_matrix = None
        if self.text_vector_names:
            token_matrix = self._token_matrix(self._token_lists(X))

        # OHE-матрица текста (если флаг)
        if self.add_text_binary_matrix and token_matrix is not None:
            ohe_df = pd.DataFrame(
                token_matrix,
                columns=[f"text__{t}" for t in self.text_vector_names],
                index=X.index,
            )
            X_trans = pd.concat([X_trans, ohe_df], axis=1)

        # Feature engineering ценовых фичей
        X_trans["REL_DRIFT_RET_CLEAN"] = (
            X_trans["MMRCurrentRetailCleanPrice"] - X_trans["MMRAcquisitonRetailCleanPrice"]
        ) / X_trans["MMRAcquisitonRetailCleanPrice"]
        X_trans["REL_DRIFT_RET_AVG"] = (
            X_trans["MMRCurrentRetailAveragePrice"] - X_trans["MMRAcquisitionRetailAveragePrice"]
        ) / X_trans["MMRAcquisitionRetailAveragePrice"]
        X_trans["REL_DRIFT_AUC_AVG"] = (
            X_trans["MMRCurrentAuctionAveragePrice"] - X_trans["MMRAcquisitionAuctionAveragePrice"]
        ) / X_trans["MMRAcquisitionAuctionAveragePrice"]

        X_trans.drop(columns=self.numer_cols[: len(MMR_COLS)], inplace=True, errors="ignore")

        rel_cols = ["WarrantyCost", "VehBCost"]
        X_trans[["REL_" + col for col in rel_cols]] = (
            X_trans[rel_cols].values / divisor.values.reshape(-1, 1)
        )

        # Фичи VehOdo / VehicleAge
        X_trans[["VehOdo", "VehicleAge"]] = X[["VehOdo", "VehicleAge"]]
        X_trans["VEHODO/VEHICLEAGE"] = X["VehOdo"] / (X["VehicleAge"] + 1)

        # Текстовый фактор (LogReg на бинарной матрице токенов)
        if self.text_model is not None and token_matrix is not None:
            X_trans["text_factor"] = self.text_model.predict_proba(token_matrix)[:, 1]

        # Дата
        X_trans["Date_factor"] = self.date_ppl.predict(self._date_frame(X))
        X_trans["date_days_count"] = (
            (X["PurchDate"] - PURCH_DATE_ORIGIN) // pd.Timedelta(days=self.date_days)
        ).astype(self.weekday_type)

        for col in self.cat_cols:
            if col in X_trans.columns:
                X_trans[col] = X_trans[col].astype("category")

        return X_trans

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)


# ====================================================================== #
# Визуализация распределений (используется JointDistributionFilter.distributions)
# ====================================================================== #
def _is_categorical(s: pd.Series) -> bool:
    """Категориальная колонка: category, object, string и т. п. (не числа и не даты)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return False
    return isinstance(s.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(s)


def _numeric_view(s: pd.Series):
    """Числовое представление колонки или None, если его нет.

    * числа -> float; даты -> дни от 2000-01-01;
    * category/object, ВСЕ непустые значения которых являются числами (например, VehYear
      как category) -> числа; иначе None (min/max у неупорядоченной категории не определён).
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return (s - pd.Timestamp("2000-01-01")).dt.days.astype(float)
    if isinstance(s.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s.astype(object), errors="coerce")
        ok = v.notna().any() and v.notna().sum() == s.notna().sum()
        return v if ok else None
    return s.astype(float)


def _stats_text(tc: pd.Series, te: pd.Series) -> str:
    """min/max по train и test: абсолютные и как доля от max|x| по обеим выборкам."""
    ranges = {}
    for name, s in (("Train", tc), ("Test", te)):
        v = _numeric_view(s)
        v = v.dropna() if v is not None else None
        ranges[name] = (float(v.min()), float(v.max())) if v is not None and len(v) else None

    vals = [abs(x) for r in ranges.values() if r for x in r]
    scale = max(vals) if vals and max(vals) > 0 else 1.0

    lines = []
    for name in ("Train", "Test"):
        r = ranges[name]
        if r is None:
            lines.append(f"{name}: min=N/A, max=N/A")
        else:
            lo, hi = r
            lines.append(f"{name}: min={lo:.4g}, max={hi:.4g}  ({lo / scale:.1%}, {hi / scale:.1%})")
    return "\n".join(lines)


def _draw_null_panel(ax, col: str, train_null: float, test_null: float, stats_text: str) -> None:
    """Доля None/NaN (столбцы) и под графиком — min/max."""
    shares = [train_null, test_null]
    y_max = max(shares) * 1.25 + 0.01
    ax.bar(["Train", "Test"], shares, color=[BLUE, ORANGE])
    ax.set_ylim(0, y_max)
    for i, v in enumerate(shares):
        ax.text(i, v + y_max * 0.02, f"{v:.4%}", ha="center", fontweight="bold")
    ax.set_title(f"{col} — None/NaN Share + Min/Max")
    ax.grid(True, alpha=0.3, axis="y")
    ax.text(
        0.5, -0.12, stats_text, transform=ax.transAxes, ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8),
    )


def _draw_cat_bars(ax, cats, train_counts: dict, test_counts: dict, title: str) -> None:
    x = np.arange(len(cats))
    width = 0.35
    ax.bar(x - width / 2, [train_counts.get(c, 0) for c in cats], width, label="Train", color=BLUE, alpha=0.8)
    ax.bar(x + width / 2, [test_counts.get(c, 0) for c in cats], width, label="Test", color=ORANGE, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c)[:25] for c in cats], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


class JointDistributionFilter(TransformerMixin):
    """Двусторонняя фильтрация признаков по профилям распределения трейна/теста.

    Числовые признаки:
      * «тяжёлые» значения (доля среди непустых >= dp: нули, медиана-заполнитель,
        фиксированные уровни) оцениваются как категории по доле;
      * остальная часть оценивается квантильной плотностью. Плотность безразмерная:
        для равномерного распределения она ≈ 1, поэтому один порог подходит разным колонкам;
        ширина квантильного интервала ограничена снизу долей диапазона (min_dx_frac),
        так что плотность не может «взрываться» на повторяющихся значениях.
    Категориальные признаки: оценка по доле категории.
    Бинарные признаки 'text__*': считается частота единиц в train и test (если он передан);
    колонка удаляется в transform, если частота ниже th_cat_train (train) или th_cat_test (test).

    transform заменяет на NaN значения (категориальные колонки возвращаются с dtype category), не
    проходящие пороги на трейне и, если передан X_test, на тесте.

    Метод distributions(...) — генератор графиков распределений train vs test.

    Класс намеренно не наследуется от BaseEstimator: X_train/X_test в конструкторе не
    сохраняются как атрибуты, и get_params/repr у BaseEstimator из-за этого падают.
    """

    def __init__(
        self,
        X_train,
        X_test=None,
        num_cols=None,
        cat_cols=None,
        dp: float = 0.02,
        th_num_train: float = 0.001,
        th_num_test: float = 0.001,
        th_cat_train: float = 0.005,
        th_cat_test: float = 0.005,
        min_dx_frac: float = 1e-3,
        bin_prefix: str = "text__",
        verbose: bool = False,
    ):
        self.dp = dp
        self.th_num_train = th_num_train
        self.th_num_test = th_num_test
        self.th_cat_train = th_cat_train
        self.th_cat_test = th_cat_test
        self.min_dx_frac = min_dx_frac
        self.bin_prefix = bin_prefix
        self.verbose = verbose

        # 1) сначала колонки (нужны в _build_profiles)
        self.num_cols = list(num_cols) if num_cols is not None else list(DEFAULT_JDF_NUM_COLS)
        self.cat_cols = list(cat_cols) if cat_cols is not None else list(DEFAULT_JDF_CAT_COLS)

        # 2) потом профили и частоты бинарных колонок
        self.train_profiles = {}
        self.test_profiles = {}
        self.bin_train_freq = {}
        self.bin_test_freq = {}
        self.bin_cols_to_drop_ = []

        self._build_profiles(X_train, is_test=False)
        if X_test is not None:
            self._build_profiles(X_test, is_test=True)

        self._build_bin_freqs(X_train, X_test)

    # ------------------------------------------------------------------ профили
    def _num_profile(self, s):
        x = pd.to_numeric(s, errors="coerce").dropna()
        n = len(x)
        if n == 0:
            return None

        # 1) тяжёлые значения: доля >= dp от числа непустых
        shares = x.value_counts(normalize=True)
        heavy = shares[shares >= self.dp]
        heavy_shares = heavy.to_dict()

        # 2) непрерывная часть без тяжёлых значений
        rest = x[~x.isin(heavy.index)].values
        profile = {"heavy": heavy_shares, "cx": None, "log_cy": None}
        if len(rest) < 2:
            return profile

        rng = rest.max() - rest.min()
        if rng <= 0:
            return profile

        n_bins = int(round(1 / self.dp))
        edges = np.quantile(rest, np.linspace(0, 1, n_bins + 1))
        dx = np.maximum(np.diff(edges), self.min_dx_frac * rng)   # пол на ширину интервала
        dens = (1.0 / n_bins) / dx * rng                          # безразмерная плотность, ≈1 для равномерной

        centers = (edges[:-1] + edges[1:]) / 2.0
        # на границах берём плотность соседнего интервала: min/max самой выборки проходят фильтр
        cx = np.concatenate([[edges[0]], centers, [edges[-1]]])
        cy = np.concatenate([[dens[0]], dens, [dens[-1]]])
        profile["cx"], profile["log_cy"] = cx, np.log(cy)
        return profile

    def _build_profiles(self, df, is_test: bool = False):
        profiles = {}
        for col in self.num_cols:
            if col in df.columns:
                p = self._num_profile(df[col])
                if p is not None:
                    profiles[col] = p
        for col in self.cat_cols:
            if col in df.columns:
                profiles[col] = df[col].value_counts(normalize=True).to_dict()
        if is_test:
            self.test_profiles = profiles
        else:
            self.train_profiles = profiles

    def _build_bin_freqs(self, X_train, X_test=None):
        """Частоты бинарных колонок text__* в трейне/тесте и список колонок на удаление."""
        bin_cols = [c for c in X_train.columns if str(c).startswith(self.bin_prefix)]

        self.bin_train_freq = {
            c: float(pd.to_numeric(X_train[c], errors="coerce").fillna(0).mean())
            for c in bin_cols
        }
        self.bin_test_freq = {}
        if X_test is not None:
            self.bin_test_freq = {
                c: float(pd.to_numeric(X_test[c], errors="coerce").fillna(0).mean())
                for c in bin_cols
                if c in X_test.columns
            }

        drop = []
        for c in bin_cols:
            if self.bin_train_freq[c] < self.th_cat_train:
                drop.append(c)
            elif X_test is not None and self.bin_test_freq.get(c, 0.0) < self.th_cat_test:
                # колонки нет в тесте или она редкая в тесте -> дропаем
                drop.append(c)
        self.bin_cols_to_drop_ = drop

        if self.verbose:
            print(f"бинарные {self.bin_prefix}*: дропается {len(drop)} из {len(bin_cols)}")

    # ------------------------------------------------------------------ маски
    @staticmethod
    def _num_pass_mask(x, profile, th_num, th_heavy):
        """True для значений, прошедших порог плотности или доли тяжёлого значения."""
        if profile["cx"] is None:
            mask = np.zeros(len(x), dtype=bool)
        else:
            log_d = np.interp(x, profile["cx"], profile["log_cy"], left=-np.inf, right=-np.inf)
            mask = np.exp(log_d) >= th_num          # NaN-значения дают False (и так остаются NaN)

        if profile["heavy"]:
            share = pd.Series(x).map(profile["heavy"]).values   # NaN, если значение не тяжёлое
            is_heavy = ~np.isnan(share)
            mask = np.where(is_heavy, np.nan_to_num(share) >= th_heavy, mask)
        return mask

    @staticmethod
    def _cat_share(values, profile):
        # Приводим к object: у категориальной Series map/fillna(0.0) падает,
        # потому что 0.0 не входит в список категорий.
        s = pd.Series(np.asarray(values, dtype=object))
        return s.map(profile).astype(float).fillna(0.0).values

    def _surviving_categories(self, col):
        """Категории, прошедшие пороги по train (и по test, если он передан)."""
        cats = []
        test_profile = self.test_profiles.get(col)
        for k, share in self.train_profiles[col].items():
            if share < self.th_cat_train or share <= 0:
                continue
            if test_profile is not None and test_profile.get(k, 0.0) < self.th_cat_test:
                continue
            cats.append(k)
        return cats

    # ------------------------------------------------------------------ sklearn API
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        for col in self.num_cols:
            if col not in X_out.columns or col not in self.train_profiles:
                continue
            x = pd.to_numeric(X_out[col], errors="coerce").astype(float).values
            final = self._num_pass_mask(
                x, self.train_profiles[col], self.th_num_train, self.th_cat_train
            )
            if col in self.test_profiles:
                final &= self._num_pass_mask(
                    x, self.test_profiles[col], self.th_num_test, self.th_cat_test
                )
            was_nan = np.isnan(x)
            out = x.copy()
            out[~final] = np.nan
            X_out[col] = out
            if self.verbose:
                print(f"{col}: замаскировано {(~final & ~was_nan).mean():.2%} значений")

        for col in self.cat_cols:
            if col not in X_out.columns or col not in self.train_profiles:
                continue
            vals = X_out[col].values
            final = self._cat_share(vals, self.train_profiles[col]) >= self.th_cat_train
            if col in self.test_profiles:
                final &= self._cat_share(vals, self.test_profiles[col]) >= self.th_cat_test
            # Набор категорий ОДИН И ТОТ ЖЕ для train и test: только значения, которые вообще
            # могут пройти фильтр. Всё остальное (в том числе «чужие» категории из dtype
            # входной колонки) становится NaN. Иначе XGBoost падает с
            # "Found a category not in the training set".
            cats = self._surviving_categories(col)
            X_out[col] = (
                pd.Series(np.asarray(X_out[col].values, dtype=object), index=X_out.index)
                .where(final)
                .astype(pd.CategoricalDtype(categories=cats))
            )
            if self.verbose:
                print(f"{col}: замаскировано {(~final).mean():.2%} значений")

        # бинарные text__* колонки с низкой частотой в train/test
        drop_cols = [c for c in self.bin_cols_to_drop_ if c in X_out.columns]
        if drop_cols:
            X_out = X_out.drop(columns=drop_cols)

        return X_out

    # ------------------------------------------------------------------ визуализация
    def distributions(self, X_train, X_test, columns=None, apply_filter=True, max_bars=30):
        """Генератор графиков распределений train vs test по колонкам.

        Параметры
        ---------
        X_train, X_test : DataFrame
            Выход Preprocessing (до фильтра, если apply_filter=True).
        columns : list, optional
            Какие колонки рисовать (по умолчанию все колонки X_train, присутствующие и в X_test).
        apply_filter : bool
            Пропустить X_train/X_test через self.transform перед построением графиков
            (т.е. показать данные после JDF). False — рисовать как есть.
        max_bars : int
            Максимум категорий на первом графике.

        Yields
        ------
        (fig, axes) — фигура с 3 subplot'ами:
            axes[0] — гистограмма частот train vs test (числовые: overlay, категории: grouped bar),
            axes[1] — доля None/NaN + min/max (абсолютные и как доля от max|x|),
            axes[2] — box plot (числовые) или bottom-10 самых редких категорий.

        Пример:
            g = jdf.distributions(train_prep, test_prep)
            fig, axes = next(g)
        """
        import matplotlib.pyplot as plt

        if X_test is None:
            raise ValueError("distributions: нужен X_test для сравнения с трейном")
        if apply_filter:
            X_train = self.transform(X_train)
            X_test = self.transform(X_test)

        cols = list(X_train.columns) if columns is None else list(columns)
        cols = [c for c in cols if c in X_train.columns and c in X_test.columns]

        for col in cols:
            tc = X_train[col]
            te = X_test[col]

            train_null = float(tc.isna().mean())
            test_null = float(te.isna().mean())
            stats_text = _stats_text(tc, te)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
            fig.suptitle(f"Column: {col}", fontsize=14, fontweight="bold")

            if _is_categorical(tc):
                # Только реально встречающиеся категории: после JDF в dtype category остаются
                # и «выбитые» категории с нулевой частотой, их на графиках быть не должно.
                tcnt = {k: int(v) for k, v in tc.dropna().value_counts().items() if v > 0}
                ecnt = {k: int(v) for k, v in te.dropna().value_counts().items() if v > 0}
                all_cats = sorted(set(tcnt) | set(ecnt), key=str)

                # 1) Частоты: самые частые категории
                by_total = sorted(all_cats, key=lambda c: -(tcnt.get(c, 0) + ecnt.get(c, 0)))
                shown = by_total[:max_bars]
                suffix = f" (top {len(shown)} of {len(all_cats)})" if len(all_cats) > max_bars else ""
                _draw_cat_bars(axes[0], shown, tcnt, ecnt, f"{col} — Frequency (Train vs Test){suffix}")

                # 2) Доля None/NaN + min/max
                _draw_null_panel(axes[1], col, train_null, test_null, stats_text)

                # 3) Bottom-10: самые редкие из реально встречающихся
                rarest = sorted(all_cats, key=lambda c: (tcnt.get(c, 0), ecnt.get(c, 0), str(c)))[:10]
                _draw_cat_bars(axes[2], rarest, tcnt, ecnt,
                               f"{col} — Bottom 10 Rarest Categories (by Train Count)")

            else:
                tn = _numeric_view(tc).dropna()
                en = _numeric_view(te).dropna()

                # 1) Гистограмма
                axes[0].hist(tn, bins=50, alpha=0.5, label="Train", color=BLUE, density=True)
                axes[0].hist(en, bins=50, alpha=0.5, label="Test", color=ORANGE, density=True)
                axes[0].set_xlabel(col)
                axes[0].set_ylabel("Density")
                axes[0].set_title(f"{col} — Histogram (Train vs Test)")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                # 2) Доля None/NaN + min/max
                _draw_null_panel(axes[1], col, train_null, test_null, stats_text)

                # 3) Box plot
                if len(tn) > 0 and len(en) > 0:
                    bp = axes[2].boxplot([tn, en], patch_artist=True)
                    for patch, color in zip(bp["boxes"], (BLUE, ORANGE)):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)
                    axes[2].set_xticks([1, 2])
                    axes[2].set_xticklabels(["Train", "Test"])
                    axes[2].grid(True, alpha=0.3, axis="y")
                else:
                    axes[2].text(0.5, 0.5, "No valid data for box plot",
                                 ha="center", va="center", transform=axes[2].transAxes)
                axes[2].set_title(f"{col} — Box Plot (Train vs Test)")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            yield fig, axes
            plt.close(fig)  # закрываем предыдущую фигуру, когда запрошена следующая