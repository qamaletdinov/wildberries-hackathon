# Wildberries Hackathon: Shipment Forecasting

Прогнозирование объёмов отгрузок со складов маркетплейса.
Ансамбль из 6 LightGBM-моделей (DRFAM-подход, победитель M5 Competition).

## Быстрый старт

### 1. Установить uv (если нет)

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Клонировать репо и синхронизировать окружение

```bash
git clone https://github.com/qamaletdinov/wildberries-hackathon.git
cd wildberries-hackathon
uv sync
```

`uv sync` автоматически:
- создаст `.venv`
- установит нужную версию Python (если нет)
- установит все зависимости из `uv.lock` с точными версиями

### 3. Положить данные

Скопировать файлы данных в корень проекта:
```
train_solo_track.parquet
test_solo_track.parquet
```

### 4. Запуск

```bash
# EDA — быстрый анализ данных (< 1 мин)
uv run python eda.py

# Полный пайплайн — обучение, валидация, сабмит
uv run python solution.py
```

`uv run` автоматически активирует виртуальное окружение.

## Выходные файлы

| Файл | Описание |
|---|---|
| `submission_solo.csv` | Финальный сабмит (id, y_pred) |
| `val_metrics_report.txt` | Метрики по каждой модели и ансамблю |
| `feature_importance.png` | Топ-30 фичей по важности |
| `eda_target_hist.png` | Распределение target |
| `eda_routes_ts.png` | Временные ряды маршрутов |
| `eda_hourly_profile.png` | Средний профиль по часам |

## Архитектура ансамбля

| # | Стратегия | Уровень | Loss |
|---|---|---|---|
| M1 | Direct (8 моделей per horizon) | Global | Tweedie |
| M2 | Direct | Global | MAE |
| M3 | Recursive (horizon-as-feature) | Global | Tweedie |
| M4 | Direct | Per-route cluster | MAE |
| M5 | Snapshot (status features) | Global | Tweedie |
| M6 | Direct | Global | Huber |

Каждая модель обучается с 3 seed-ами (seed averaging).
Веса ансамбля оптимизируются через `scipy.optimize` на OOF-предсказаниях.
Финальный прогноз корректируется на bias (мультипликативная коррекция).

## Полезные команды uv

```bash
uv sync              # синхронизировать окружение из uv.lock
uv lock              # пересоздать lock-файл (после изменения pyproject.toml)
uv add <package>     # добавить зависимость
uv run python ...    # запустить скрипт в окружении
```
