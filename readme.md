# Don't Get Kicked — end-to-end ML pipelines (XGBoost · LightGBM · CatBoost · Neural network)

**Languages / Языки:** [English](#english) · [Русский](#русский)

---

<a name="english"></a>
# English

## 1. Overview

Solution for the Kaggle competition **Don't Get Kicked!**: predict whether a car bought at an auction is a "kick" (a bad buy, `IsBadBuy`). The repository contains four model families, a shared preprocessing layer, tools to tune / train / save / reuse everything, and standalone Kaggle notebooks.

### Results

| Solution | Score | Link |
|---|---|---|
| Previous record holder: *kotrying* | **0.26892** | [Win the past competition🏆 \<Don't Get Kicked!\>](https://www.kaggle.com/code/kotrying/win-the-past-competition-don-t-get-kicked) |
| **My project** (best submission, private score) | **0.27085** | [💎🥇Top score Don'tGetKicked solution🥇💎](https://www.kaggle.com/code/vladnoff/top-score-don-tgetkicked-solution) |

Difference: **+0.00193** over the previous record.


## 2. Key ideas

1. The XGBoost, LGBM, CatBoost, and NeuralNetwork (PyTorch) models are being trained, as well as 2 more: XGBoost trained on features without the 5 most important ones from the original XGBoost and XGBoost trained on a dataset that is transformed through JDF from the most uncertain predictions. About JDF just below. The obtained basic models produce predictions (or prediction logs), a linear combination of which we optimize through Optoona maximizing the competition metric: ROC_AUC. 
2. Since we ideally need weakly correlated OOF predictions for the effectiveness of ensemble methods and blending in particular, I decided to do different, even slightly strange preprocessing for the models. For example, for the LGBM model for processing text tokens, I used NN-trained embeddings: individually, the idea is not at all mandatory, but for blending it gives an increase in metrics. Learn more with diagrams (literally pictures!) Preprocessing can be found in the corresponding kaggle notebooks in the folders. But there are still common blocks, only some presets of some features for each model change. 
3. The general block of Preprocessing. We highlight categorical, numerical features. A special feature of the dataset will be the presence of categories such as 'SubModel' that can carry information about the vehicle configuration: for example, 'QUAD CAB 4.7L SLT' where individual tokens can carry information. In terms of models, I processed them as just a binary matrix of the presence of a token in a sample feature, in NN I trained embeddings for each such token, and at the same time, cutbust processed them in the 3rd way through its Ordered-Encoding. It was also possible to establish patterns with the seasonality of the increase in sales of damaged cars. And for the 'MMR*' block of price features, I decided to train a logregression on a 3-degree polynomial in order to capture some dependencies like 'currentPrice - Acquisition' and their nonlinear amplifications. (After all, the resolving tree does not look well for such patterns). All this is described in more detail in EDA.ipynb
4. The second common preprocessing unit is JointDistributionFilter (JDF). In fact, some cheap analogue of the DANN architecture. Its purpose is to compare the distributions of features in the training and test samples, and we replace everything that does not fit both ways with None or 'Other' (depending on the model). The point is that during the training of the classifier, we learn only those patterns that are really useful to us on the test. Learn more about JDF in EDA.ipynb, as well as the outcome of its impact on the metric on the test. 
5. The third FeatureImportanceDropper block simply removes a number of the most unimportant hacks (for the case when the binary text matrix is submitted without processing). It is also used to remove the 5 most important features in XGB_drop_important_solution. 
6. The implementation of the above classes allows you to tune their hyperparameters along with the hyperparameters of the classification algorithm itself. Thus, the best number of neurons in the hidden NN layer and the bin size for the number of days in preprocessing can be adjusted simultaneously through Optuna. The tuning was done via .py scripts in each folder with library versions from requirements.txt . The results of the predictions on the test and OOF are in a separate oof_predictions folder. It is on them that the blending model is trained, which brought the record. Kaggle laptops have different versions from mine, so the results are slightly different and the hyperparameters in laptops are suboptimal (they are made more for an example to show how the pipeline works). 
7. Validation of the parameters for speed was carried out on one fold. The final oof predictions were obtained from 5 models: each validates the number of trees or epochs on its fold and is trained on the rest.

### Results

![alt text](image-1.png)
*Screenshot of my model submissions (best one: private score 0.27085).*
## 3. Project structure

```
project_root/
├── preprocessing.py          # Preprocessing, JointDistributionFilter (+ plots), FeatureImportanceDropper
├── cv_checkpoints.py         # adaptive checkpoints driver, CV-curve plot, paths of models/ and oof_predictions/
├── importance_df.csv         # feature-importance table (Feature, Importance) used by FeatureImportanceDropper
├── requirements.txt          # pinned library versions
├── dont_get_kicked/datasets/ # training.csv, test.csv (competition data)
│
├── xgb_solution/
│   ├── xgb_solution.py           # nested Optuna tuning -> xgb_top_params.json (top-5 configs + best_estimators)
│   ├── xgb_oof_submission.py     # k-fold training, adaptive checkpoints, saves models + OOF/test predictions
│   └── predict.py                # inference on raw test data
├── lgbm_solution/                # same layout; text features come from the NN embedding block
├── catboost_solution/
│   ├── catboost_tuning.py        # Optuna tuning -> catboost_best_params.json
│   ├── catboost_training.py      # k-fold training, adaptive checkpoints, saving
│   └── predict.py
├── nn_solution/
│   ├── nn_solution.py            # Optuna tuning -> nn_config.json (+ NNFeatures / model code)
│   ├── nn_oof_submission.py      # k-fold training with adaptive epoch checkpoints, saving
│   ├── nn_token_importance.py    # full-data NN -> token importance, UMAP of token embeddings, nn_full_model.pt
│   └── predict.py
│
├── models/                   # saved models + fitted preprocessing + manifests (created by the training scripts)
├── oof_predictions/          # OOF and test predictions of every saved checkpoint
└── kaggle_notebooks/         # standalone notebooks (no local imports) + eda.ipynb
```

File names follow the pattern *tuning script* `<model>_solution.py` (`catboost_tuning.py` for CatBoost) and *training script* `<model>_oof_submission.py` (`catboost_training.py` for CatBoost); rename freely — the scripts only import their sibling `*_solution`/`*_tuning` module.

## 5. How models are saved

Everything lives in two folders next to `preprocessing.py` (paths are defined in `cv_checkpoints.py`).

**`models/`**

| Family | Files |
|---|---|
| XGBoost | `xgb_m{rank}_fold{i}.ubj` (`Booster.save_model`), `xgb_m{rank}_fold{i}_pipeline.joblib` (fitted `Preprocessing`, JDF, `FeatureImportanceDropper`, column order), `xgb_manifest.json` |
| LightGBM | `lgbm_m{rank}_fold{i}.txt` (`Booster.save_model`), `lgbm_m{rank}_fold{i}_pipeline.joblib`, `lgbm_text_pipeline.joblib` (text pipeline + NN weights), `lgbm_manifest.json` |
| CatBoost | `catboost_m{rank}_fold{i}.cbm` (`save_model`), `catboost_m{rank}_fold{i}_pipeline.joblib`, `catboost_manifest.json` |
| Neural network | `nn_ep{epoch}_fold{i}.pt` (`torch.save(model.state_dict())`), `nn_fold{i}_pipeline.joblib` (fitted `Preprocessing`, JDF, `NNFeatures` state), `nn_manifest.json` |

`rank` is the index of the config in the tuning JSON (0 = best). Boosters keep all trained trees; predictions use the first `n_iter` trees. The **manifest** lists the files, every saved checkpoint with its CV ROC AUC, and the default checkpoint (the best CV AUC) used by `predict.py`.

**`oof_predictions/`**: `{family}_oof_m{rank}_est{iterations}_cv{auc}.csv` (`RefId, IsBadBuy, <family>_prediction`) and `{family}_test_m{rank}_est{iterations}_cv{auc}.csv` (`RefId, IsBadBuy`, averaged over folds); for the NN `nn_oof_ep{epoch}_cv{auc}.csv` / `nn_test_ep{epoch}_cv{auc}.csv`. Reports (CV curve PNG, CV history CSV, `summary.csv`) are written to the script's own folder (e.g. `xgb_solution/xgb_oof_preds/`).

## 6. How to use the pipelines

### Installation
```bash
pip install -r requirements.txt
```
The versions used by the author locally: pandas 2.3.3, numpy 1.26.4, scikit-learn 1.8.0, xgboost 2.1.4, lightgbm 4.6.0, catboost 1.2.5 (plus scipy, joblib, optuna, torch, matplotlib, umap-learn — see `requirements.txt`). Different versions can shift CV scores in the third decimal.

### A. Predict with the saved models (no training)
Prerequisite: the `models/` folder (created by the training scripts, or copied from a release). Input: a **raw** test CSV in the competition format (`RefId`, `PurchDate`, ...). Output: a CSV with `RefId` and `IsBadBuy` (probability).

```bash
python xgb_solution/predict.py      path/to/test.csv -o xgb_pred.csv -verbose
python lgbm_solution/predict.py     path/to/test.csv -o lgbm_pred.csv
python catboost_solution/predict.py path/to/test.csv -o catboost_pred.csv
python nn_solution/predict.py       path/to/test.csv -o nn_pred.csv
```
Options: `--model-rank R` and `--n-iter N` (boosters), `--epoch E` (NN), `--models-dir DIR`, `-verbose`. By default the checkpoint with the best CV AUC from the manifest is used. From Python:

```python
from xgb_solution.predict import predict
df = predict("path/to/test.csv", verbose=True)   # DataFrame with RefId, IsBadBuy
```

### B. Retrain from scratch
1. Put `training.csv` and `test.csv` into `dont_get_kicked/datasets/`.
2. **Neural network:** `nn_solution/nn_solution.py` (tuning → `nn_config.json`) → `nn_oof_submission.py` (k-fold, saves models) → `nn_token_importance.py` (full-data network → `nn_full_model.pt`, token importance CSV, UMAP plot).
3. **XGBoost:** needs `importance_df.csv` → `xgb_solution.py` (tuning → `xgb_top_params.json`) → `xgb_oof_submission.py`.
4. **LightGBM:** needs `nn_full_model.pt` from step 2 → `lgbm_solution.py` → `lgbm_oof_submission.py`.
5. **CatBoost:** `catboost_tuning.py` → `catboost_training.py`.
6. Use `predict.py` of each folder (section A) or blend the files from `oof_predictions/`.

Training scripts take `model_ranks` (which tuned configs to use) and `est_coefs` (coefficients ≥ 1 for the checkpoints), e.g. `main(model_ranks=(0, 2, 3), est_coefs=(1.0, 1.2, 1.5))`.

### C. Standalone Kaggle notebooks (`kaggle_notebooks/`)
`kaggle_xgb`, `kaggle_lgbm`, `kaggle_catboost`, `kaggle_nn` reproduce a model end-to-end with no local imports: paste the tuned hyper-parameters from the JSON into the *Config* cell, set `N_ITER` / `N_EPOCHS`, run all cells, get `submission.csv` (no model objects are saved). XGBoost needs `importance_df.csv` and LightGBM needs `nn_full_model.pt` attached as Kaggle datasets. Extra notebooks: `kaggle_xgb_no_jdf` (ablation without JDF, 100 Optuna trials), `kaggle_xgb_own_features` (a fresh feature set + Optuna) and `eda.ipynb` (motivates the preprocessing: date factor, price/text factors, feature importance, interactive JDF plots).

## 7. Reproducibility notes and limitations

- Folds are fixed (`StratifiedKFold(5, shuffle=True, random_state=42)`); library versions and thread counts still change CV in the third decimal (local vs Kaggle runs differed by about 0.003 ROC AUC).
- Stacked factors (`price_factor`, `text_factor`, `Date_factor`) are fitted inside each fold's training part, but on the training rows themselves they are in-sample predictions.
- LightGBM's text features come from a network trained on the whole train set, so its CV score is optimistic; the saved test predictions are unaffected by that.
- Hyper-parameters are tuned on one fold with early stopping on the same validation part, which is optimistic; the final number of trees / epochs is therefore re-selected on the full CV curve.
- CatBoost cannot be trained one tree at a time in lockstep, so its folds are trained once on a fixed budget (`iterations × max(est_coefs)`) and the same adaptive checkpoints are replayed on the finished models.

## 8. Links

- Previous record holder: [Win the past competition🏆 \<Don't Get Kicked!\>](https://www.kaggle.com/code/kotrying/win-the-past-competition-don-t-get-kicked) — score 0.26892
- This solution: [💎🥇Top score Don'tGetKicked solution🥇💎](https://www.kaggle.com/code/vladnoff/top-score-don-tgetkicked-solution) — private score 0.27085

---

<a name="русский"></a>
# Русский

## 1. О проекте

Решение соревнования Kaggle **Don't Get Kicked!**: нужно предсказать, окажется ли автомобиль, купленный на аукционе, «киком» (неудачной покупкой, `IsBadBuy`). В репозитории четыре семейства моделей, общий слой предобработки, инструменты для тюнинга, обучения, сохранения и повторного использования, а также самодостаточные ноутбуки для Kaggle.

### Результаты

| Решение | Score | Ссылка |
|---|---|---|
| Предыдущий рекордсмен: *kotrying* | **0.26892** | [Win the past competition🏆 \<Don't Get Kicked!\>](https://www.kaggle.com/code/kotrying/win-the-past-competition-don-t-get-kicked) |
| **Мой проект** (лучший сабмит, private score) | **0.27085** | [💎🥇Top score Don'tGetKicked solution🥇💎](https://www.kaggle.com/code/vladnoff/top-score-don-tgetkicked-solution) |

Разница с предыдущим рекордом: **+0.00193**.


## 2. Ключевые идеи

1. Обучаются модели XGBoost, LGBM, CatBoost, NeuralNetwork (PyTorch), а также еще 2: XGBoost обученный на признаках без 5 самых важных из оригинального XGboost и XGBoost обученный на датасете, который трансформирован через JDF от самых неуверенных предсказаний. О JDF чуть ниже. Полученные базовые модели выдают предсказания (или логиты предсказаний), линейную комбинацию которых мы оптимизируем через Optuna максимизируя метрику соревнования : ROC_AUC. 
2. Поскольку для эффективности ансамблевых методов и блендинга в частности нам нужны, в идеале, слабо коррелирующие OOF предсказания, я решил сделать разный, даже немного странный preprocessing для моделей. Так например для LGBM модели для обработки текстовых токенов я воспользовался предобученными с помощью NN ембедингами: в отдельности идея вовсе не обязательная, но для блендинга дает прирост метрики. Подробнее со схемами(буквально картинками!) предобработки можно ознакомиться в соответсвующих kaggle-ноутбуках в папках. Но общие блоки все-таки есть, меняются лишь некоторые предсатвления некоторых признаков для каждой модели. 
3. Общий блок предобработки Preprocessing. Выделяем категорильаные, числовые признаки. Особенностью датасета будет наличие категорий, типа 'SubModel' которые могут нести в себе информацию о комплектации автомобиля: например 'QUAD CAB 4.7L SLT' где отдельные токены могут нести в себе информацию. В части моделей я обработал их как просто бинарную матрицу наличия токена в признаке сэмпла, в NN я обучал эмбеддинги для каждого такого токена, а катбуст в то же время обработал их 3-м способом через свой Ordered-Encoding. Также удлаось установить закономерности с сезонностью увеличения продаж битых автомобилей. А для блока ценовых признаков 'MMR*' я решил обучить логрегрессию на полиноме 3 степени, чтобы уловить некоторые зависимости типа 'CurrentPrice - AcquisitonPrice' и их нелинейные усиления. (Все таки разрешающее дерево плохо ищет такие закономерности). Подробнее обо всем этом рассказано в EDA.ipynb
4. Второй общий блок предобработки JointDistributionFilter (JDF). По сути, некоторый дешевый аналог DANN архитектуры. Его цель сравнить распределения признаков на обучающей и тестовой выборках, и все что не подходит и туда и туда мы заменяем на None или 'Other'(зависит от модели). Смысл в том чтобы во время обучения классификатора мы выучивали только те закономерности которые нам реально пригодятся на тесте. Подробнее про JDF в EDA.ipynb а также про итог его воздействия на метрику на тесте. 
5. Третий блок FeatureImportanceDropper просто удаляет некоторое количество самых неважных призинаков (для случая когда текстовая бинарная матрица подается без обработки). Также используется для удаления 5 самых важных признаков в XGB_drop_important_solution. 
6. Реализация вышеописанных классов позволяет тюнить их гиперпараметры наряду с гиперпараметрами самого алгоритма классификации. Так одновременно через Optuna может настраиваться и лучшее число нейронов в скрытом слое NN и размер бина для количества дней в preprocessing. Тюнинг производился через .py скрипты в каждой папке с версиями библиотек из requirements.txt. Результаты предсказаний на тесте и OOF лежат в отдельной папке oof_predictions. Именно на них обучается модель блендинга, которая и принесла рекорд. В kaggle ноутбуках версии отличаются от моих, поэтому результаты немного отличаются и гиперпараметры в ноутбуках неоптимальны (они сделаны скорее для примера,чтобы показать как устроен пайплайн). 
7. Валидация параметров для быстроты проводилась на одном фолде. Итоговые oof-предсказания получены от 5 моделей: каждая валидирует число деревьев или эпох на своем фолде и обучается на остальных. 

### Результаты лучших моделей

![alt text](image.png)
*Скриншот моих сабмитов (лучший: private score 0.27085).*

## 3. Структура проекта

```
project_root/
├── preprocessing.py          # Preprocessing, JointDistributionFilter (+ графики), FeatureImportanceDropper
├── cv_checkpoints.py         # адаптивные чекпоинты, график CV-кривой, пути папок models/ и oof_predictions/
├── importance_df.csv         # таблица важности признаков (Feature, Importance) для FeatureImportanceDropper
├── requirements.txt          # версии библиотек
├── dont_get_kicked/datasets/ # training.csv, test.csv (данные соревнования)
│
├── xgb_solution/
│   ├── xgb_solution.py           # вложенный тюнинг Optuna -> xgb_top_params.json (топ-5 конфигов + best_estimators)
│   ├── xgb_oof_submission.py     # k-fold обучение, адаптивные чекпоинты, сохранение моделей и OOF/test
│   └── predict.py                # предсказание на сырых данных
├── lgbm_solution/                # то же; текстовые признаки берутся из блока эмбеддингов нейросети
├── catboost_solution/
│   ├── catboost_tuning.py        # тюнинг Optuna -> catboost_best_params.json
│   ├── catboost_training.py      # k-fold обучение, адаптивные чекпоинты, сохранение
│   └── predict.py
├── nn_solution/
│   ├── nn_solution.py            # тюнинг Optuna -> nn_config.json (+ код NNFeatures / модели)
│   ├── nn_oof_submission.py      # k-fold обучение с адаптивными чекпоинтами по эпохам, сохранение
│   ├── nn_token_importance.py    # сеть на всех данных -> важность токенов, UMAP эмбеддингов, nn_full_model.pt
│   └── predict.py
│
├── models/                   # сохранённые модели, обученная предобработка, манифесты (создаётся скриптами обучения)
├── oof_predictions/          # OOF и тестовые предсказания каждого сохранённого чекпоинта
└── kaggle_notebooks/         # самодостаточные ноутбуки (без локальных импортов) + eda.ipynb
```

Имена файлов: скрипт тюнинга `<model>_solution.py` (для CatBoost `catboost_tuning.py`), скрипт обучения `<model>_oof_submission.py` (для CatBoost `catboost_training.py`); можно переименовывать — скрипты импортируют только соседний модуль `*_solution` / `*_tuning`.


## 5. Как сохраняются модели

Всё лежит в двух папках рядом с `preprocessing.py` (пути заданы в `cv_checkpoints.py`).

**`models/`**

| Семейство | Файлы |
|---|---|
| XGBoost | `xgb_m{rank}_fold{i}.ubj` (`Booster.save_model`), `xgb_m{rank}_fold{i}_pipeline.joblib` (обученные `Preprocessing`, JDF, `FeatureImportanceDropper`, порядок колонок), `xgb_manifest.json` |
| LightGBM | `lgbm_m{rank}_fold{i}.txt` (`Booster.save_model`), `lgbm_m{rank}_fold{i}_pipeline.joblib`, `lgbm_text_pipeline.joblib` (пайплайн текста + веса NN), `lgbm_manifest.json` |
| CatBoost | `catboost_m{rank}_fold{i}.cbm` (`save_model`), `catboost_m{rank}_fold{i}_pipeline.joblib`, `catboost_manifest.json` |
| Нейросеть | `nn_ep{epoch}_fold{i}.pt` (`torch.save(model.state_dict())`), `nn_fold{i}_pipeline.joblib` (обученные `Preprocessing`, JDF, состояние `NNFeatures`), `nn_manifest.json` |

`rank` — индекс конфига в json после тюнинга (0 — лучший). Бустеры хранят все обученные деревья, предсказание использует первые `n_iter` деревьев. **Манифест** перечисляет файлы, все сохранённые чекпоинты с их CV ROC AUC и чекпоинт по умолчанию (лучший CV AUC), который берёт `predict.py`.

**`oof_predictions/`**: `{family}_oof_m{rank}_est{iterations}_cv{auc}.csv` (`RefId, IsBadBuy, <family>_prediction`) и `{family}_test_m{rank}_est{iterations}_cv{auc}.csv` (`RefId, IsBadBuy`, среднее по фолдам); для нейросети `nn_oof_ep{epoch}_cv{auc}.csv` / `nn_test_ep{epoch}_cv{auc}.csv`. Отчёты (график CV-кривой PNG, история CV CSV, `summary.csv`) пишутся в папку самого скрипта (например, `xgb_solution/xgb_oof_preds/`).

## 6. Как пользоваться пайплайнами

### Установка
```bash
pip install -r requirements.txt
```
Версии автора (локально): pandas 2.3.3, numpy 1.26.4, scikit-learn 1.8.0, xgboost 2.1.4, lightgbm 4.6.0, catboost 1.2.5 (плюс scipy, joblib, optuna, torch, matplotlib, umap-learn — см. `requirements.txt`). Другие версии могут сдвигать CV в третьем знаке.

### A. Предсказание сохранёнными моделями (без обучения)
Нужна папка `models/` (создаётся скриптами обучения или берётся из релиза). Вход: **сырой** тестовый CSV в формате соревнования (`RefId`, `PurchDate`, ...). Выход: CSV с `RefId` и `IsBadBuy` (вероятность).

```bash
python xgb_solution/predict.py      path/to/test.csv -o xgb_pred.csv -verbose
python lgbm_solution/predict.py     path/to/test.csv -o lgbm_pred.csv
python catboost_solution/predict.py path/to/test.csv -o catboost_pred.csv
python nn_solution/predict.py       path/to/test.csv -o nn_pred.csv
```
Опции: `--model-rank R` и `--n-iter N` (бустинги), `--epoch E` (NN), `--models-dir DIR`, `-verbose`. По умолчанию берётся чекпоинт с лучшим CV AUC из манифеста. Из Python:

```python
from xgb_solution.predict import predict
df = predict("path/to/test.csv", verbose=True)   # DataFrame с RefId, IsBadBuy
```

### B. Полное переобучение
1. Положите `training.csv` и `test.csv` в `dont_get_kicked/datasets/`.
2. **Нейросеть:** `nn_solution/nn_solution.py` (тюнинг → `nn_config.json`) → `nn_oof_submission.py` (k-fold, сохраняет модели) → `nn_token_importance.py` (сеть на всех данных → `nn_full_model.pt`, CSV важности токенов, график UMAP).
3. **XGBoost:** нужен `importance_df.csv` → `xgb_solution.py` (тюнинг → `xgb_top_params.json`) → `xgb_oof_submission.py`.
4. **LightGBM:** нужен `nn_full_model.pt` из шага 2 → `lgbm_solution.py` → `lgbm_oof_submission.py`.
5. **CatBoost:** `catboost_tuning.py` → `catboost_training.py`.
6. Используйте `predict.py` из каждой папки (раздел A) или смешивайте файлы из `oof_predictions/`.

Скрипты обучения принимают `model_ranks` (какие конфиги после тюнинга брать) и `est_coefs` (коэффициенты ≥ 1 для чекпоинтов), например `main(model_ranks=(0, 2, 3), est_coefs=(1.0, 1.2, 1.5))`.

### C. Самодостаточные ноутбуки Kaggle (`kaggle_notebooks/`)
`kaggle_xgb`, `kaggle_lgbm`, `kaggle_catboost`, `kaggle_nn` воспроизводят модель от начала до конца без локальных импортов: вставьте подобранные гиперпараметры из JSON в ячейку *Config*, задайте `N_ITER` / `N_EPOCHS`, запустите все ячейки — получится `submission.csv` (объекты моделей не сохраняются). Для XGBoost нужен `importance_df.csv`, для LightGBM — `nn_full_model.pt`, оба как Kaggle-датасеты. Дополнительно: `kaggle_xgb_no_jdf` (абляция без JDF, 100 триалов Optuna), `kaggle_xgb_own_features` (новый набор признаков + Optuna) и `eda.ipynb` (обоснование предобработки: date factor, price/text factor, важность признаков, интерактивные графики JDF).

## 7. Воспроизводимость и ограничения

- Фолды зафиксированы (`StratifiedKFold(5, shuffle=True, random_state=42)`); версии библиотек и число потоков всё равно меняют CV в третьем знаке (локальный и Kaggle-запуск расходились примерно на 0.003 ROC AUC).
- Стэкинг-факторы (`price_factor`, `text_factor`, `Date_factor`) обучаются внутри обучающей части каждого фолда, но на самих обучающих строках это in-sample предсказания.
- Текстовые признаки LightGBM получены из сети, обученной на всём трейне, поэтому его CV-оценка оптимистична; сохранённые тестовые предсказания это не затрагивает.
- Гиперпараметры подбираются на одном фолде с early stopping на той же валидации, что оптимистично; поэтому итоговое число деревьев / эпох заново выбирается по полной CV-кривой.
- CatBoost нельзя обучать по одному дереву синхронно, поэтому его фолды обучаются один раз на фиксированный бюджет (`iterations × max(est_coefs)`), а те же адаптивные чекпоинты «проигрываются» на готовых моделях.

## 8. Ссылки

- Предыдущий рекордсмен: [Win the past competition🏆 \<Don't Get Kicked!\>](https://www.kaggle.com/code/kotrying/win-the-past-competition-don-t-get-kicked) — score 0.26892
- Это решение: [Highest score "Don't Get Kicked" solution](https://www.kaggle.com/code/vladnoff/highest-score-don-t-get-kicked-solution) — private score 0.27085