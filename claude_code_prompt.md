# ЗАДАЧА: Хакатон по прогнозированию временных рядов (логистика)

## КОНТЕКСТ
Ты — ML-инженер, участвующий в хакатоне. Задача: прогнозирование объёмов отгрузок
со складов маркетплейса. Нужно побить baseline организаторов (Ridge, score ~0.32)
с помощью ансамблевого подхода.

---

## ДАННЫЕ

**Train:** `train_solo_track.parquet`
Колонки:
- `route_id` — уникальный ID маршрута (1000 маршрутов)
- `timestamp` — момент времени, шаг 30 минут
- `status_1, status_2, status_3` — товары на ТЕКУЩЕМ складе за последние 30 мин
- `status_4, status_5, status_6` — товары на ПРЕДЫДУЩЕМ складе за последние 30 мин
- `target_1h` — ЦЕЛЕВАЯ ПЕРЕМЕННАЯ: объём отгрузок за последний час

**Test:** `test_solo_track.parquet`
Колонки: только `id`, `route_id`, `timestamp` — статусов НЕТ!

**Submission:** CSV с колонками `id` и `y_pred`

---

## МЕТРИКА

```python
class WapePlusRbias:
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        wape  = np.abs(y_pred - y_true).sum() / y_true.sum()
        rbias = np.abs(y_pred.sum() / y_true.sum() - 1)
        return wape + rbias
```

**ВАЖНО:** Метрика = WAPE + |Relative Bias|. Меньше — лучше.
- WAPE оптимизируется через MAE/Tweedie loss
- |Relative Bias| требует отдельной post-processing коррекции

---

## ПЛАН РЕАЛИЗАЦИИ (выполнять последовательно)

### ШАГ 1 — EDA и базовый анализ
```
- Загрузи train и test, выведи shape, dtypes, пропуски
- Покажи временной диапазон train и test
- Покажи распределение target_1h (гистограмма + статистики)
- Проверь: есть ли нули? Есть ли выбросы (99-й перцентиль)?
- Выведи пример нескольких маршрутов (plot временного ряда)
```

### ШАГ 2 — Feature Engineering (без status-фичей!)
Создать следующие признаки (доступны в тесте, нет leakage):

```python
# A) Лаговые фичи таргета
# Для горизонта H шагов вперёд использовать ТОЛЬКО лаги >= H
# Рекомендуемые лаги (в 30-мин шагах): [2, 3, 4, 6, 8, 12, 24, 48, 48*7]
# Все лаги считать внутри groupby('route_id')

# B) Rolling статистики (сдвинутые на min_lag шагов)
# windows: [6, 12, 24, 48] → mean, std, median

# C) Временные признаки из timestamp
# hour (0-23), minute (0/30), dayofweek (0-6), is_weekend
# slot = hour*2 + (minute>=30)  # 48 слотов в сутки

# D) Маршрутные агрегации (глобальные, без leakage)
# route_mean, route_std, route_median (по всему train)
# route_hour_mean = среднее по (route_id, hour)
# route_dow_mean  = среднее по (route_id, dayofweek)

# E) Target encoding route_id через 5-fold CV (избегаем leakage)
```

### ШАГ 3 — Time-based Split для валидации
```python
# Использовать последние 4 непересекающихся окна по 8 шагов (= 4 часа)
# НЕ использовать random split — только временной!
# Метрику считать на каждом окне отдельно + итоговую агрегацию
```

### ШАГ 4 — Ансамбль моделей (DRFAM-подход, M5 winner)

Обучить 6 моделей LightGBM:

| # | Стратегия | Уровень агрегации | Loss |
|---|---|---|---|
| M1 | Direct (отдельная на каждый из 8 шагов) | Global | tweedie |
| M2 | Direct | Global | mae |
| M3 | Recursive (одна модель, horizon-as-feature) | Global | tweedie |
| M4 | Direct | Per-route group (кластеры по route_mean) | mae |
| M5 | Snapshot (как baseline) | Global | tweedie |
| M6 | Direct | Global | huber |

```python
# Параметры LightGBM (стартовые):
params = {
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}
# Early stopping на 100 раундах по кастомной WAPE-метрике
```

**Direct стратегия (M1, M2, M4, M6):**
```python
# Для шага h = 1..8 создать отдельный target: shift(-h) внутри route_id
# Обучить отдельную модель per horizon
# ВАЖНО: для шага h использовать только лаги >= h (иначе leakage!)
```

**Recursive стратегия (M3):**
```python
# Добавить фичу 'horizon' (1..8) к датасету
# Одна модель предсказывает всё
# При инференсе: предсказать шаг 1, добавить как лаг, предсказать шаг 2, ...
```

### ШАГ 5 — Ансамблирование
```python
# Шаг 5a: простое среднее всех 6 моделей (baseline ансамбля)

# Шаг 5b: оптимизация весов через scipy.optimize на OOF predictions
from scipy.optimize import minimize

def wape_loss(weights, preds_matrix, actuals):
    ensemble = preds_matrix @ weights
    return np.abs(ensemble - actuals).sum() / actuals.sum()

# constraints: weights >= 0, sum(weights) == 1
# Запустить 100 случайных стартов (np.random.dirichlet)
# Взять лучший результат
```

### ШАГ 6 — Bias Correction (ОБЯЗАТЕЛЬНО!)
```python
# После получения финальных предсказаний ансамбля на валидации:

# Вариант A: глобальная мультипликативная коррекция
ratio = val_actual.sum() / val_ensemble_pred.sum()
test_pred_corrected = test_pred * ratio

# Вариант B: per-route коррекция (если хватает данных)
for route_id in routes:
    mask_val  = val['route_id'] == route_id
    mask_test = test['route_id'] == route_id
    r = val_actual[mask_val].sum() / val_pred[mask_val].sum()
    test_pred[mask_test] *= r

# Измерить rbias до и после коррекции, выбрать лучший вариант
```

### ШАГ 7 — Финальная пост-обработка
```python
# 1. Клиппинг отрицательных значений: pred = np.maximum(pred, 0)
# 2. Seed averaging: обучить каждую модель с seeds [42, 123, 456],
#    усреднить предсказания внутри каждого типа модели
# 3. Isotonic calibration (опционально):
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(val_pred_sorted, val_actual_sorted)
test_calibrated = ir.predict(test_pred)
```

### ШАГ 8 — Формирование submission
```python
# submission = pd.DataFrame({'id': test_df['id'], 'y_pred': final_pred})
# Проверить: assert submission['id'].isna().sum() == 0
# Проверить: assert (submission['y_pred'] >= 0).all()
# Сохранить: submission.to_csv('submission_solo.csv', index=False)
```

---

## КРИТЕРИИ УСПЕХА

| Шаг | Ожидаемый score (WAPE + |Bias|) |
|---|---|
| Ridge baseline (организаторы) | ~0.32 |
| LightGBM MAE + лаги | ~0.25–0.27 |
| DRFAM (6 моделей) + OOF weights | ~0.20–0.23 |
| + Bias correction | ~0.18–0.21 |
| + Seed averaging + calibration | ~0.16–0.19 |

---

## ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ

```bash
pip install lightgbm scikit-learn pandas numpy scipy pyarrow
```

**Структура файлов на выходе:**
```
submission_solo.csv          ← финальный сабмит
val_metrics_report.txt       ← метрики по каждой модели и ансамблю
feature_importance.png       ← топ-30 фичей по важности
```

---

## ВАЖНЫЕ ОГРАНИЧЕНИЯ

1. **Никаких status-фичей в тесте** → не использовать status_1..6 как основные фичи модели
2. **Temporal leakage** → при создании лагов для шага h использовать ТОЛЬКО лаги >= h
3. **Random split запрещён** → только time-based validation
4. **Клиппинг** → предсказания не могут быть отрицательными
5. **Сабмит** → строго две колонки: `id` и `y_pred`

---

## ДОПОЛНИТЕЛЬНЫЕ ИДЕИ (если останется время)

- N-HiTS через `pip install neuralforecast` как 7-й компонент ансамбля
- Кластеризация маршрутов по профилю (kmeans на route_hour_avg) → отдельные модели per кластер
- Fourier-фичи для сезонности: sin/cos с периодами 48 (сутки) и 48*7 (неделя)
- Wavelet decomposition таргета для выделения трендовой компоненты
