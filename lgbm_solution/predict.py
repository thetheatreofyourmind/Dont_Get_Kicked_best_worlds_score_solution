"""Предсказание LightGBM на сырых тестовых данных (модели из lgbm_oof_submission.py).

Загружает из папки models (рядом с preprocessing.py):
  lgbm_manifest.json                        — какие файлы и чекпоинты есть, чекпоинт по умолчанию;
  lgbm_text_pipeline.joblib                 — пайплайн текста: Preprocessing + JDF + веса блока текста сети;
  lgbm_m{rank}_fold{i}.txt                  — бустеры фолдов;
  lgbm_m{rank}_fold{i}_pipeline.joblib      — обученные Preprocessing, JDF и колонки каждого фолда.
Предсказание = среднее по фолдам вероятностей, посчитанных по первым n_iter деревьям каждого бустера.
По умолчанию берётся модель и число деревьев с лучшим CV ROC AUC среди сохранённых.

Из консоли:
    python predict.py path/to/raw_test.csv [-o predictions.csv] [--model-rank 0] [--n-iter 1200] [-verbose]
Из Python:
    from lgbm_solution.predict import predict, LGBMPredictor
    df = predict("path/to/raw_test.csv", verbose=True)   # DataFrame: RefId, IsBadBuy
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # корень проекта: здесь preprocessing.py, cv_checkpoints.py, папка models
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import preprocessing  # noqa: E402,F401  (классы нужны для распаковки joblib-объектов)
from cv_checkpoints import MODELS_DIR  # noqa: E402

MANIFEST = "lgbm_manifest.json"
_SIBLING = {}


def _load_sibling(stem):
    """Загружает lgbm_solution.py из этой же папки по пути к файлу (не зависит от имени пакета)."""
    if stem not in _SIBLING:
        spec = importlib.util.spec_from_file_location(f"_sibling_{stem}", HERE / f"lgbm_tuning.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SIBLING[stem] = module
    return _SIBLING[stem]


def _read_raw(raw):
    if isinstance(raw, (str, Path)):
        return pd.read_csv(raw)
    return raw.copy()


class LGBMPredictor:
    def __init__(self, models_dir=MODELS_DIR, model_rank=None, n_iter=None, verbose=False):
        self.verbose = verbose
        models_dir = Path(models_dir)
        manifest = json.loads((models_dir / MANIFEST).read_text(encoding="utf-8"))
        self.rank, self.n_iter = self._select(manifest, model_rank, n_iter)
        entry = next(m for m in manifest["models"] if m["rank"] == self.rank)

        self._log(f"загрузка моделей: rank={self.rank}, деревьев={self.n_iter}, фолдов={len(entry['folds'])}")
        self.text_pipe = joblib.load(models_dir / manifest["text_pipeline"])
        self.folds = []
        for i, f in enumerate(entry["folds"]):
            booster = lgb.Booster(model_file=str(models_dir / f["model"]))
            pipeline = joblib.load(models_dir / f["pipeline"])
            self.folds.append({"booster": booster, "pipe": pipeline})
            self._log(f"  fold {i + 1}: {f['model']}")

    @staticmethod
    def _select(manifest, model_rank, n_iter):
        by_rank = {m["rank"]: m for m in manifest["models"]}
        rank = manifest["default"]["rank"] if model_rank is None else model_rank
        if rank not in by_rank:
            raise ValueError(f"модели с rank={rank} нет в манифесте, доступны: {sorted(by_rank)}")
        entry = by_rank[rank]
        if n_iter is None:
            if model_rank is None:
                n_iter = manifest["default"]["n_iter"]
            else:  # лучший по CV AUC сохранённый чекпоинт этой модели
                n_iter = max(entry["checkpoints"], key=lambda c: c["cv_auc"])["n_iter"]
        if not 1 <= n_iter <= entry["n_trees"]:
            raise ValueError(f"n_iter={n_iter} вне диапазона 1..{entry['n_trees']} для rank={rank}")
        return rank, int(n_iter)

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def predict_proba(self, raw):
        """Вероятности IsBadBuy (np.ndarray) для сырых данных (DataFrame или путь к csv)."""
        ls = _load_sibling("lgbm_solution")
        raw = _read_raw(raw)
        X = raw.drop(columns=[c for c in ("RefId", "IsBadBuy") if c in raw.columns])
        n, out, start = len(self.folds), np.zeros(len(X)), time.time()
        self._log(f"предсказание для {len(X)} строк")

        text_df = ls.transform_nn_text(X, self.text_pipe)  # эмбеддинги текста из блока нейросети
        self._log(f"  текстовые признаки: {text_df.shape[1]} ({time.time() - start:.1f} c)")

        for i, f in enumerate(self.folds):
            pipe = f["pipe"]
            X_p = pipe["prep"].transform(X)
            X_f = ls.apply_pipeline(X_p, pipe["jdf"], text_df)
            missing = [c for c in pipe["columns"] if c not in X_f.columns]
            if missing:
                raise KeyError(f"после предобработки нет колонок {missing[:5]}...: данные не совпадают с обучающими")
            out += f["booster"].predict(X_f[pipe["columns"]], num_iteration=self.n_iter) / n
            self._log(f"  fold {i + 1}/{n} готов, признаков {X_f.shape[1]} ({time.time() - start:.1f} c)")
        return out

    def predict(self, raw):
        """DataFrame с колонками RefId и IsBadBuy (вероятность)."""
        raw = _read_raw(raw)
        if "RefId" not in raw.columns:
            raise ValueError("в сырых данных нет колонки RefId")
        return pd.DataFrame({"RefId": raw["RefId"].values, "IsBadBuy": self.predict_proba(raw)})


def predict(raw, model_rank=None, n_iter=None, models_dir=MODELS_DIR, verbose=False):
    """Функция-обёртка: сырые данные (DataFrame или путь к csv) -> DataFrame[RefId, IsBadBuy]."""
    return LGBMPredictor(models_dir, model_rank, n_iter, verbose).predict(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Предсказание LightGBM на сырых тестовых данных")
    parser.add_argument("data", help="путь к сырым тестовым данным (csv в формате test.csv)")
    parser.add_argument("-o", "--output", default=None,
                        help="куда сохранить предсказания (по умолчанию рядом с данными)")
    parser.add_argument("--model-rank", type=int, default=None, help="какая модель из манифеста (по умолчанию лучшая)")
    parser.add_argument("--n-iter", type=int, default=None, help="сколько деревьев использовать")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="папка с моделями")
    parser.add_argument("-verbose", "--verbose", action="store_true", help="показывать ход предсказания")
    args = parser.parse_args(argv)

    result = predict(args.data, args.model_rank, args.n_iter, args.models_dir, args.verbose)
    out = Path(args.output) if args.output else Path(args.data).with_name(Path(args.data).stem + "_lgbm_predictions.csv")
    result.to_csv(out, index=False)
    print(f"сохранено: {out} ({len(result)} строк)")


if __name__ == "__main__":
    main()