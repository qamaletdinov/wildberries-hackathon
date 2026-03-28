# Ансамбль для мультишагового прогнозирования отгрузок: мировые практики

**Лучший способ побить baseline в 0.32 (Ridge + snapshot) — это ансамбль из LightGBM-моделей с разными стратегиями мультишагового прогнозирования (Direct + Recursive), обученных на разных уровнях агрегации маршрутов, с Tweedie/MAE loss и пост-обработкой для коррекции bias.** Именно такой подход — DRFAM (Direct and Recursive Forecast Averaging Method) — занял 1-е место на M5 Competition, крупнейшем соревновании по прогнозированию спроса в истории Kaggle. Все топ-50 участников M5 использовали LightGBM. Нейросети (N-HiTS, TFT) полезны как компонент ансамбля, но градиентный бустинг с хорошим feature engineering остаётся «золотым стандартом» для табличных данных с временными рядами. Ниже — конкретные техники, код и рекомендации, собранные из решений топ-1/топ-3 Kaggle, arXiv и production-систем (DoorDash, Amazon).

---

## A. Feature engineering без реальных фичей в тесте

Ключевая проблема задачи: в тесте есть только `route_id` + `timestamp`, а `status_1..6` недоступны. Это классический сценарий «future covariates unavailable». Топ-решения Kaggle решают это четырьмя стратегиями, от простых к сложным.

**Стратегия 1 — «Убрать status-фичи полностью»** (baseline). Обучить модель только на тех фичах, которые доступны в тесте: лаги таргета, временные признаки, агрегации по маршрутам. Это исключает train/test mismatch и часто работает лучше, чем попытки «достроить» status-фичи.

**Стратегия 2 — «Snapshot последнего состояния»** (подход организаторов). Если на момент предсказания `t` status-фичи известны, используем их как «замороженный снимок» для всех будущих шагов. Это работает, но quality degrades с ростом горизонта, поскольку snapshot устаревает.

**Стратегия 3 — «Two-stage forecasting»**. Сначала прогнозируем `status_1..6` на будущие шаги вспомогательными моделями, затем подставляем прогнозы в основную модель. Этот подход использовался в Corporación Favorita (5-е место, WaveNet + Seq2Seq). Однако ошибки первого этапа каскадируются, поэтому рекомендуется сглаживать прогнозы status-фичей через rolling windows.

**Стратегия 4 — «Гибрид: лаги таргета + временные + маршрутные»** (рекомендуемая). Это наиболее робастный подход, подтверждённый M5 Competition. Конкретный набор признаков:

```python
# === КРИТИЧЕСКОЕ ПРАВИЛО: для Direct-модели на горизонт H шагов ===
# доступны ТОЛЬКО лаги >= H (иначе data leakage)
H = 2  # например, 2 шага по 30 мин = 1 час

# 1. Лаги таргета (30-мин интервалы)
safe_lags = [H, H+1, H+2, H+4, H+6, H+12, H+24, H+48, H+48*7]
for lag in safe_lags:
    df[f'target_lag_{lag}'] = df.groupby('route_id')['target_1h'].shift(lag)

# 2. Rolling статистики (сдвинуты на H)
for window in [6, 12, 24, 48]:  # 3ч, 6ч, 12ч, 24ч
    shifted = df.groupby('route_id')['target_1h'].shift(H)
    df[f'roll_mean_{window}'] = shifted.rolling(window).mean()
    df[f'roll_std_{window}'] = shifted.rolling(window).std()

# 3. Временные признаки
df['hour'] = df['timestamp'].dt.hour
df['dow'] = df['timestamp'].dt.dayofweek
df['slot'] = df['hour'] * 2 + (df['timestamp'].dt.minute >= 30).astype(int)
df['is_weekend'] = (df['dow'] >= 5).astype(int)

# Для нейросетей/линейных моделей — циклическое кодирование:
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
# Для LightGBM — raw integers работают не хуже (деревья сами находят сплиты)

# 4. Маршрутные агрегации
route_stats = df.groupby('route_id')['target_1h'].agg(['mean','std','median'])
df = df.merge(route_stats, on='route_id', suffixes=('','_route'))
# Профиль маршрута по часам:
df['route_hour_avg'] = df.groupby(['route_id','hour'])['target_1h'].transform('mean')
df['route_dow_avg'] = df.groupby(['route_id','dow'])['target_1h'].transform('mean')

# 5. Target encoding для route_id (K-Fold чтобы не было leakage)
from sklearn.model_selection import KFold
enc = np.zeros(len(df))
for tr_idx, val_idx in KFold(5, shuffle=True, random_state=42).split(df):
    means = df.iloc[tr_idx].groupby('route_id')['target_1h'].mean()
    enc[val_idx] = df.iloc[val_idx]['route_id'].map(means)
df['route_target_enc'] = enc
```

**Приём из M5 (1-е место)**: для каждого события/особого периода создать фичу «расстояние до события» в диапазоне от -15 до +15 шагов. Это моделирует предпиковые и постпиковые эффекты спроса, что особенно важно для логистики с еженедельной сезонностью.

---

## B. LightGBM побеждает, но ансамбль с нейросетями добавляет 1–3%

Данные из всех крупных соревнований однозначны: **LightGBM с хорошим feature engineering — лучшая single-model стратегия** для задач прогнозирования спроса/отгрузок. Но картина нюансирована.

| Соревнование (год) | 1-е место | Модель |
|---|---|---|
| M5 Accuracy (2020) | YeonJun Im | 6× LightGBM (DRFAM) |
| Corporación Favorita (2018) | — | LightGBM + Neural Net ensemble |
| Rossmann (2015) | — | XGBoost (500 моделей) |
| Walmart (2014) | — | Среднее 6 статистических моделей |

**Когда LightGBM/XGBoost выигрывают**: табличные данные с ручным feature engineering; малые и средние датасеты; когда нужна скорость обучения; данные с перемежающимся спросом (intermittent demand). Победитель M5 подчёркивал: «LightGBM работает хорошо из коробки, без предобработки данных, нативно поддерживает Tweedie/Poisson loss».

**Когда нейросети добавляют ценность**: при большом количестве связанных временных рядов (>1000); при длинных горизонтах прогноза (Direct multi-output без error accumulation); когда нет времени на feature engineering (end-to-end обучение); для probabilistic forecasting (DeepAR, TFT).

**Конкретные рекомендации по нейросетям для данной задачи:**

- **N-HiTS** — лучший стартовый вариант для нейросетевого компонента ансамбля. Прямой multi-output, быстрое обучение, не требует future covariates. Работает с иерархической интерполяцией на разных временных масштабах — идеально для 30-минутных данных с дневной/недельной сезонностью.
- **TFT** (Temporal Fusion Transformer) — лучший выбор если есть «known future inputs» (например, плановые акции, праздники). Нативно разделяет static covariates, known future inputs и past-only inputs. Однако требует больше данных и времени на обучение.
- **PatchTST** — эффективен для длинных горизонтов за счёт patch-tokenization (уменьшение квадратичной сложности attention). Reduction на **21% MSE** по сравнению с предыдущими трансформерами в бенчмарках.

Статья «Are Transformers Effective for Time Series Forecasting?» (AAAI 2023) показала, что простые линейные модели (DLinear) могут бить старые трансформеры на **20–50%**. Однако PatchTST (ICLR 2023) развенчал этот вывод, продемонстрировав, что правильно спроектированные трансформеры всё же эффективны. **Практический консенсус**: нейросети не заменяют LightGBM, но дополняют его в ансамбле.

Все перечисленные нейросети доступны в библиотеке **NeuralForecast (Nixtla)** с единым sklearn-like API.

---

## C. Построение ансамбля под WAPE-метрику

**WAPE = Σ|yᵢ − ŷᵢ| / Σ|yᵢ|** — это MAE, нормализованная на общий объём. Оптимизация WAPE эквивалентна оптимизации MAE, поскольку знаменатель — константа для фиксированного test set. Это означает, что **оптимальный прогноз для WAPE — это медиана условного распределения** (Rob Hyndman, 2025). Отсюда выбор loss = MAE (L1) для обучения.

**Архитектура ансамбля уровня победителей M5 (DRFAM)**:

```
Уровень 1: Стратегии прогноза
├── Recursive LightGBM × 3 пула (per-route, per-route-group, global)
├── Direct LightGBM × 3 пула (per-route, per-route-group, global)
└── (Опционально) N-HiTS / TFT

Уровень 2: Простое среднее всех моделей
(или оптимизация весов через scipy.optimize)
```

**Ключевое правило из M5**: простое арифметическое среднее 6 разнородных LightGBM-моделей (3 пула × 2 стратегии) заняло 1-е место. Сложный стекинг не потребовался. Разнообразие обеспечивается через: (a) разные стратегии multi-step — recursive vs direct; (b) разные уровни агрегации — per-route vs per-group vs global; (c) разные loss-функции — Tweedie vs MAE vs Huber.

**Оптимизация весов ансамбля под WAPE:**

```python
from scipy.optimize import minimize

def wape_ensemble(weights, predictions_matrix, actuals):
    """predictions_matrix: shape (n_samples, n_models)"""
    y_ens = predictions_matrix @ weights
    return np.sum(np.abs(y_ens - actuals)) / np.sum(np.abs(actuals))

def find_optimal_weights(oof_preds, actuals, n_models, n_trials=100):
    best_w, best_score = None, np.inf
    for _ in range(n_trials):
        w0 = np.random.dirichlet(np.ones(n_models))
        res = minimize(wape_ensemble, w0, args=(oof_preds, actuals),
                       method='SLSQP',
                       bounds=[(0,1)]*n_models,
                       constraints=[{'type':'eq','fun': lambda w: w.sum()-1}])
        if res.fun < best_score:
            best_score, best_w = res.fun, res.x
    return best_w

# ВАЖНО: использовать out-of-fold predictions, не in-sample!
```

**DoorDash ELITE Framework** (production): temporal stacking с разными base learners (Prophet, LightGBM, статистические модели). Стекинг-слой обучается на out-of-fold предсказаниях с temporal cross-validation. Даёт **~10% улучшение** над лучшей single-model.

**Мета-исследование 2025 года** (33 ансамблевых метода, 50 датасетов) подтвердило: multi-level stacking (base models → stackers → final selector) consistently лучше простого среднего на **~5%**, но простое среднее — самый робастный fallback.

---

## D. Борьба с Relative Bias: четыре уровня защиты

**Relative Bias = Σ(ŷᵢ − yᵢ) / Σ|yᵢ|** — это систематическое отклонение прогноза вверх или вниз. В комбинированной метрике WAPE + |Relative Bias| bias штрафуется дополнительно, поэтому нейтральный bias (≈0) критически важен.

**Уровень 1: Правильный выбор loss-функции.** Tweedie loss (M5 стандарт) избегает bias, который возникает при log-трансформации таргета. Amazon обнаружил, что обратное преобразование `exp(log_pred)` систематически занижает прогноз (неравенство Йенсена). Tweedie работает с исходными значениями и моделирует right-skewed распределение напрямую:

```python
model = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.5,  # тюнить 1.0–2.0
    metric='mae'  # для мониторинга WAPE-эквивалента
)
```

**Уровень 2: Ансамбль моделей с разнонаправленным bias.** Модели с MAE loss тяготеют к медиане (underforecast для skewed data), модели с MSE — к среднему (overforecast для outliers). Их среднее часто имеет bias ≈ 0.

**Уровень 3: Пост-обработка — мультипликативная коррекция bias.** Измеряем bias на валидации, корректируем тест:

```python
# Per-route, per-horizon коррекция
for route in routes:
    for h in horizons:
        mask = (val['route_id'] == route) & (val['horizon'] == h)
        ratio = val_actual[mask].sum() / val_pred[mask].sum()
        test_pred[test_mask] *= ratio

# Или глобальная коррекция:
global_ratio = val_actual.sum() / val_pred.sum()
test_pred *= global_ratio
```

**Уровень 4: Asymmetric loss для fine-tuning направления bias.**

```python
def asymmetric_mae_objective(preds, train_data):
    labels = train_data.get_label()
    residual = preds - labels
    alpha = 0.52  # > 0.5 штрафует overforecast сильнее
    grad = np.where(residual > 0, alpha, -(1-alpha))
    hess = np.ones_like(grad) * 0.01  # маленький hessian для стабильности
    return grad, hess
```

**Rectify Strategy** (Taieb & Hyndman, 2012) — теоретически оптимальный метод. Строим recursive forecast, затем обучаем Direct-модели на остатках (bias) для каждого горизонта. Итоговый прогноз = recursive + rectification. Доказано, что этот подход **асимптотически несмещённый** и «всегда не хуже лучшего из recursive и direct».

---

## E. Конкретные трюки из топ-решений Kaggle

**Трюк 1 — DRFAM (M5, 1-е место).** Простое среднее Direct и Recursive прогнозов. Direct не имеет error accumulation, Recursive лучше использует короткие лаги. Среднее берёт лучшее от обоих. Реализация: обучить 2 набора моделей (по 3 пула каждый), усреднить все 6.

**Трюк 2 — Partial pooling (M5, 1-е место).** Вместо одной глобальной или N отдельных моделей — обучение на промежуточных уровнях группировки. Для задачи с маршрутами: per-route, per-region, global. Каждый уровень добавляет разнообразие и робастность.

**Трюк 3 — Tweedie loss + LightGBM (M5, стандарт).** Для данных с нулями и правым хвостом (типично для отгрузок) Tweedie работает лучше MSE и MAE. Parameter `tweedie_variance_power` тюнится в диапазоне **1.0–2.0** (1 = Poisson, 2 = Gamma).

**Трюк 4 — Horizon-as-feature (Favorita, multiple teams).** Вместо H отдельных моделей — одна модель с дополнительной фичей `forecast_horizon`. Экономит время, работает сравнимо с Direct для малых H:

```python
# Создаём датасет с horizon как фичей
rows = []
for h in range(1, H+1):
    temp = df.copy()
    temp['horizon'] = h
    temp['target'] = df.groupby('route_id')['target_1h'].shift(-h)
    rows.append(temp)
train_expanded = pd.concat(rows)
model.fit(train_expanded[features + ['horizon']], train_expanded['target'])
```

**Трюк 5 — Rolling-smoothed recursive (M5, 4-е место).** При recursive forecasting не использовать raw point predictions как лаги. Вместо этого — сглаживать через rolling window ≥7 шагов. Это снижает чувствительность к шумным предсказаниям на предыдущих шагах.

**Трюк 6 — Множественный seed averaging (Favorita, 5-е место).** Обучить каждую модель 3–5 раз с разными random seeds, усреднить предсказания. Снижает variance без дополнительного feature engineering. Каждая отдельная модель давала top-1% результат.

**Трюк 7 — Event proximity features (M5, Rossmann).** Для каждого особого события (конец месяца, праздник, акция) создать фичу `days_to_event` в диапазоне [-15, +15]. В задаче с 30-мин шагами — `steps_to_event`. Моделирует предпиковые/постпиковые паттерны.

**Трюк 8 — Cross-validation: 4 последних окна (M5, 1-е место).** Валидация на последних 4 непересекающихся временных окнах длиной в горизонт прогноза. Измерять и mean, и std метрики — если std высок, модель нестабильна и нужна регуляризация.

**Трюк 9 — Custom WAPE eval metric с early stopping:**

```python
def wape_eval(y_true, y_pred):
    return 'wape', np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)), False

model = lgb.LGBMRegressor(objective='mae', n_estimators=3000)
model.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          eval_metric=wape_eval,
          callbacks=[lgb.early_stopping(100)])
```

**Трюк 10 — Isotonic regression для bias calibration (production).** Финальная пост-обработка: обучить IsotonicRegression на (predictions_val → actuals_val), применить к test. Это нелинейная коррекция bias, которая исправляет systematic over/underforecast в разных диапазонах:

```python
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(val_pred, val_actual)
test_pred_corrected = ir.predict(test_pred)
```

---

## Рекомендуемый план действий для хакатона

Приоритеты выстроены по ожидаемому вкладу в улучшение score:

1. **Базовый LightGBM с MAE loss** на лагах + временных фичах (без status) → ожидаемо побьёт Ridge baseline
2. **Direct стратегия** — отдельная модель на каждый горизонт → устраняет error accumulation
3. **Добавить Recursive стратегию** → усреднить с Direct (DRFAM) → снижает variance
4. **Partial pooling** — модели на 2–3 уровнях агрегации маршрутов → diversity в ансамбле
5. **Tweedie loss** как альтернатива MAE → лучше для skewed данных
6. **Seed averaging** (3–5 seeds на модель) → бесплатное снижение variance
7. **Bias correction** на валидации → мультипликативная коррекция per-route per-horizon
8. **N-HiTS как нейросетевой компонент** → добавить в ансамбль для дополнительного diversity
9. **Оптимизация весов ансамбля** через scipy.optimize на OOF predictions
10. **Isotonic calibration** как финальная пост-обработка

Ключевые репозитории: [M5 Winning Methods](https://github.com/Mcompetitions/M5-methods), [Favorita 5th place](https://github.com/LenzDu/Kaggle-Competition-Favorita), [NeuralForecast](https://github.com/Nixtla/neuralforecast), [Microsoft Forecasting Best Practices](https://github.com/microsoft/forecasting).

## Заключение

Три инсайта выделяются из анализа всех источников. Во-первых, **разнообразие важнее сложности**: победитель M5 использовал простое среднее 6 вариаций LightGBM — никакого стекинга, никаких нейросетей — и обошёл все сложные ансамбли. Разнообразие достигается через комбинацию стратегий (Direct + Recursive) и уровней агрегации, а не через экзотические архитектуры. Во-вторых, **для задач с WAPE + |Relative Bias| метрикой критически важна двухэтапная оптимизация**: сначала минимизировать MAE (что минимизирует WAPE), затем применить bias correction (что минимизирует |Relative Bias|). Эти два компонента метрики оптимизируются разными механизмами и не должны смешиваться в одном loss. В-третьих, **отсутствие status-фичей в тесте — не проблема, а возможность**: лаговые фичи таргета + временные признаки + маршрутные агрегации часто дают результат сравнимый или лучший, чем модели с real-time features, потому что исключают train/test distribution mismatch.