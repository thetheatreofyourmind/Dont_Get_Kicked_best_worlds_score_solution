"""Предсказание нейросети на сырых тестовых данных (модели из nn_oof_submission.py).

Загружает из папки models (рядом с preprocessing.py):
  nn_manifest.json                  — чекпоинты (эпохи) и их CV AUC, конфиг сети, чекпоинт по умолчанию;
  nn_fold{i}_pipeline.joblib        — обученные Preprocessing, JDF и состояние NNFeatures каждого фолда;
  nn_ep{эпоха}_fold{i}.pt           — веса сетей (state_dict) на выбранной эпохе.
Предсказание = среднее по фолдам вероятностей. По умолчанию берётся эпоха с лучшим CV ROC AUC
среди сохранённых; веса сохранены только для чекпоинтов, поэтому другую эпоху взять нельзя.

Из консоли:
    python predict.py path/to/raw_test.csv [-o predictions.csv] [--epoch 21] [-verbose]
Из Python:
    from nn_solution.predict import predict, NNPredictor
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import preprocessing  # noqa: E402,F401  (классы нужны для распаковки joblib-объектов)
from cv_checkpoints import MODELS_DIR  # noqa: E402

MANIFEST = "nn_manifest.json"
_SIBLING = {}


def _load_sibling(stem):
    """Загружает nn_solution.py из этой же папки по пути к файлу (не зависит от имени пакета)."""
    if stem not in _SIBLING:
        spec = importlib.util.spec_from_file_location(f"_sibling_{stem}", HERE / f"nn_tuning.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SIBLING[stem] = module
    return _SIBLING[stem]


def _read_raw(raw):
    if isinstance(raw, (str, Path)):
        return pd.read_csv(raw)
    return raw.copy()


class NNPredictor:
    def __init__(self, models_dir=MODELS_DIR, epoch=None, verbose=False):
        self.verbose = verbose
        models_dir = Path(models_dir)
        nn = _load_sibling("nn_solution")
        manifest = json.loads((models_dir / MANIFEST).read_text(encoding="utf-8"))

        by_epoch = {c["epoch"]: c for c in manifest["checkpoints"]}
        self.epoch = manifest["default"]["epoch"] if epoch is None else epoch
        if self.epoch not in by_epoch:
            raise ValueError(f"веса для эпохи {self.epoch} не сохранены, доступны: {sorted(by_epoch)}")
        entry = by_epoch[self.epoch]
        self.cfg = manifest["model_cfg"]

        self._log(f"загрузка моделей: эпоха={self.epoch}, фолдов={len(manifest['pipelines'])}, "
                  f"устройство={nn.DEVICE}")
        self.folds = []
        for i, pipe_file in enumerate(manifest["pipelines"]):
            pipeline = joblib.load(models_dir / pipe_file)
            builder = nn.NNFeatures()
            builder.__dict__.update(pipeline["builder_state"])
            state = torch.load(models_dir / entry["files"][i], map_location=nn.DEVICE)
            self.folds.append({"pipe": pipeline, "builder": builder, "state": state, "model": None})
            self._log(f"  fold {i + 1}: {entry['files'][i]}")

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def predict_proba(self, raw):
        """Вероятности IsBadBuy (np.ndarray) для сырых данных (DataFrame или путь к csv)."""
        nn = _load_sibling("nn_solution")
        raw = _read_raw(raw)
        X = raw.drop(columns=[c for c in ("RefId", "IsBadBuy") if c in raw.columns])
        n, out, start = len(self.folds), np.zeros(len(X)), time.time()
        self._log(f"предсказание для {len(X)} строк")
        for i, f in enumerate(self.folds):
            pipe = f["pipe"]
            X_f = pipe["jdf"].transform(pipe["prep"].transform(X))
            spec = f["builder"].transform(X_f)
            batch = nn.to_device(spec)
            if f["model"] is None:  # размеры блоков определяет builder, поэтому модель строим по spec
                model = nn.build_model(self.cfg, spec)
                model.load_state_dict(f["state"])
                f["model"] = model
            _, proba = nn.evaluate(f["model"], batch, None)
            out += proba / n
            self._log(f"  fold {i + 1}/{n} готов, признаков text={spec['text_dim']} "
                      f"cat={len(spec['embed_cols'])} ohe={spec['ohe_dim']} cont={spec['cont_dim']} "
                      f"({time.time() - start:.1f} c)")
        return out

    def predict(self, raw):
        """DataFrame с колонками RefId и IsBadBuy (вероятность)."""
        raw = _read_raw(raw)
        if "RefId" not in raw.columns:
            raise ValueError("в сырых данных нет колонки RefId")
        return pd.DataFrame({"RefId": raw["RefId"].values, "IsBadBuy": self.predict_proba(raw)})


def predict(raw, epoch=None, models_dir=MODELS_DIR, verbose=False):
    """Функция-обёртка: сырые данные (DataFrame или путь к csv) -> DataFrame[RefId, IsBadBuy]."""
    return NNPredictor(models_dir, epoch, verbose).predict(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Предсказание нейросети на сырых тестовых данных")
    parser.add_argument("data", help="путь к сырым тестовым данным (csv в формате test.csv)")
    parser.add_argument("-o", "--output", default=None,
                        help="куда сохранить предсказания (по умолчанию рядом с данными)")
    parser.add_argument("--epoch", type=int, default=None, help="эпоха чекпоинта (по умолчанию лучшая по CV AUC)")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="папка с моделями")
    parser.add_argument("-verbose", "--verbose", action="store_true", help="показывать ход предсказания")
    args = parser.parse_args(argv)

    result = predict(args.data, args.epoch, args.models_dir, args.verbose)
    out = Path(args.output) if args.output else Path(args.data).with_name(Path(args.data).stem + "_nn_predictions.csv")
    result.to_csv(out, index=False)
    print(f"сохранено: {out} ({len(result)} строк)")


if __name__ == "__main__":
    main()