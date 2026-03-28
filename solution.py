"""
Хакатон: прогнозирование отгрузок со складов маркетплейса.
Ансамбль из 6 LightGBM-моделей (DRFAM-подход, победитель M5).

Запуск: python solution.py

Выход:
  - submission_solo.csv
  - val_metrics_report.txt
  - feature_importance.png
"""

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
np.random.seed(42)

# ╔═══════════════════════════════════════════════════════════╗
# ║                    КОНФИГУРАЦИЯ                           ║
# ╚═══════════════════════════════════════════════════════════╝

DATA_DIR = "."
TRAIN_FILE = os.path.join(DATA_DIR, "train_solo_track.parquet")
TEST_FILE = os.path.join(DATA_DIR, "test_solo_track.parquet")

# Лаги (в 30-мин шагах) — базовые, будут фильтроваться по горизонту
BASE_LAGS = [2, 3, 4, 6, 8, 12, 24, 48, 336]  # 336 = 48*7 (неделя)
ROLLING_WINDOWS = [6, 12, 24, 48]  # 3ч, 6ч, 12ч, 24ч
N_FOLDS_TE = 5          # для target encoding
N_ROUTE_CLUSTERS = 5    # для M4 (per-route group)
SEEDS = [42, 123, 456]  # seed averaging

LGB_BASE_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
}

EARLY_STOPPING_ROUNDS = 100


# ╔═══════════════════════════════════════════════════════════╗
# ║                    МЕТРИКА                                ║
# ╚═══════════════════════════════════════════════════════════╝

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.abs(y_pred - y_true).sum() / np.maximum(y_true.sum(), 1e-12)


def rbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.abs(y_pred.sum() / np.maximum(y_true.sum(), 1e-12) - 1)


def wape_plus_rbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return wape(y_true, y_pred) + rbias(y_true, y_pred)


def lgb_wape_eval(y_true, y_pred):
    """Custom eval metric для LightGBM (early stopping)."""
    score = np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(np.abs(y_true)), 1e-12)
    return "wape", score, False  # name, value, is_higher_better


# ╔═══════════════════════════════════════════════════════════╗
# ║               ЗАГРУЗКА ДАННЫХ                             ║
# ╚═══════════════════════════════════════════════════════════╝

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Загрузка данных...")
    train = pd.read_parquet(TRAIN_FILE)
    test = pd.read_parquet(TEST_FILE)
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    train = train.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    test = test.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    print(f"  train: {train.shape}, test: {test.shape}")
    return train, test


# ╔═══════════════════════════════════════════════════════════╗
# ║              FEATURE ENGINEERING                          ║
# ╚═══════════════════════════════════════════════════════════╝

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Временные признаки из timestamp."""
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["slot"] = df["hour"] * 2 + (df["minute"] >= 30).astype(np.int8)
    # Fourier для сезонности (суточная + недельная)
    df["hour_sin"] = np.sin(2 * np.pi * df["slot"] / 48)
    df["hour_cos"] = np.cos(2 * np.pi * df["slot"] / 48)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lag_features(df: pd.DataFrame, min_lag: int = 1) -> pd.DataFrame:
    """Лаговые фичи таргета (с учётом минимального лага для горизонта)."""
    safe_lags = [l for l in BASE_LAGS if l >= min_lag]
    for lag in safe_lags:
        df[f"target_lag_{lag}"] = df.groupby("route_id")["target_1h"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, min_lag: int = 1) -> pd.DataFrame:
    """Rolling статистики (сдвинутые на min_lag)."""
    shifted = df.groupby("route_id")["target_1h"].shift(min_lag)
    for w in ROLLING_WINDOWS:
        roll = shifted.rolling(w, min_periods=1)
        df[f"roll_mean_{w}"] = roll.mean()
        df[f"roll_std_{w}"] = roll.std()
        df[f"roll_median_{w}"] = roll.median()
    return df


def add_route_aggregations(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Маршрутные агрегации (вычисляются ТОЛЬКО по train)."""
    # Глобальные stats
    route_stats = train_df.groupby("route_id")["target_1h"].agg(
        route_mean="mean", route_std="std", route_median="median"
    ).reset_index()
    df = df.merge(route_stats, on="route_id", how="left")

    # Per hour
    route_hour = train_df.groupby(["route_id", "hour"])["target_1h"].mean()
    route_hour = route_hour.rename("route_hour_mean").reset_index()
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    df = df.merge(route_hour, on=["route_id", "hour"], how="left")

    # Per dayofweek
    if "dayofweek" not in df.columns:
        df["dayofweek"] = df["timestamp"].dt.dayofweek
    route_dow = train_df.groupby(["route_id", "dayofweek"])["target_1h"].mean()
    route_dow = route_dow.rename("route_dow_mean").reset_index()
    df = df.merge(route_dow, on=["route_id", "dayofweek"], how="left")

    return df


def add_target_encoding(df: pd.DataFrame, train_mask: np.ndarray) -> pd.DataFrame:
    """Target encoding для route_id через K-Fold (только на train)."""
    df["route_target_enc"] = np.nan
    train_idx = np.where(train_mask)[0]
    kf = KFold(N_FOLDS_TE, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_idx):
        tr_rows = train_idx[tr_idx]
        vl_rows = train_idx[val_idx]
        means = df.iloc[tr_rows].groupby("route_id")["target_1h"].mean()
        df.loc[df.index[vl_rows], "route_target_enc"] = (
            df.iloc[vl_rows]["route_id"].map(means).values
        )
    # Для теста — глобальное среднее по route_id из всего train
    global_means = df.loc[train_mask].groupby("route_id")["target_1h"].mean()
    test_idx = np.where(~train_mask)[0]
    df.loc[df.index[test_idx], "route_target_enc"] = (
        df.iloc[test_idx]["route_id"].map(global_means).values
    )
    return df


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    min_lag: int = 2,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Полный pipeline фичей. Возвращает объединённый df и маску is_train.
    min_lag: минимальный лаг (зависит от горизонта для Direct моделей).
    """
    train_cp = train.copy()
    test_cp = test.copy()

    # Добавляем hour/dayofweek к train для агрегаций
    train_cp["hour"] = train_cp["timestamp"].dt.hour
    train_cp["dayofweek"] = train_cp["timestamp"].dt.dayofweek

    # Помечаем
    train_cp["is_test"] = False
    test_cp["is_test"] = True
    if "target_1h" not in test_cp.columns:
        test_cp["target_1h"] = np.nan

    combined = pd.concat([train_cp, test_cp], ignore_index=True)
    combined = combined.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    train_mask = ~combined["is_test"].values

    # Фичи
    combined = add_time_features(combined)
    combined = add_lag_features(combined, min_lag=min_lag)
    combined = add_rolling_features(combined, min_lag=min_lag)
    combined = add_route_aggregations(combined, train_cp)
    combined = add_target_encoding(combined, train_mask)

    return combined, train_mask


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Список фичей (исключаем meta-колонки и status)."""
    exclude = {
        "route_id", "timestamp", "target_1h", "is_test", "id",
        "status_1", "status_2", "status_3",
        "status_4", "status_5", "status_6",
    }
    return [c for c in df.columns if c not in exclude]


# ╔═══════════════════════════════════════════════════════════╗
# ║             ВАЛИДАЦИЯ (TIME-BASED)                        ║
# ╚═══════════════════════════════════════════════════════════╝

def create_time_splits(
    train: pd.DataFrame, n_windows: int = 4, window_size: int = 8
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Последние n_windows непересекающихся окон по window_size шагов.
    Возвращает list of (train_idx, val_idx).
    """
    timestamps = train["timestamp"].sort_values().unique()
    splits = []
    for i in range(n_windows, 0, -1):
        val_end = len(timestamps) - (i - 1) * window_size
        val_start = val_end - window_size
        if val_start < 0:
            continue
        val_ts = set(timestamps[val_start:val_end])
        val_mask = train["timestamp"].isin(val_ts).values
        train_mask = train["timestamp"] < timestamps[val_start]
        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return splits


# ╔═══════════════════════════════════════════════════════════╗
# ║                МОДЕЛИ (6 типов)                           ║
# ╚═══════════════════════════════════════════════════════════╝

def train_lgb_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: dict,
    seed: int = 42,
) -> lgb.LGBMRegressor:
    """Обучение одной LightGBM-модели с early stopping."""
    p = {**LGB_BASE_PARAMS, **params, "random_state": seed}
    model = lgb.LGBMRegressor(**p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=lgb_wape_eval,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


# ── M1: Direct, Global, Tweedie ──────────────────────────

def train_direct_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizons: List[int],
    loss: str = "tweedie",
    extra_params: Optional[dict] = None,
    cluster_labels: Optional[dict] = None,
    seeds: List[int] = None,
) -> Dict[int, List]:
    """
    Direct стратегия: отдельная модель на каждый горизонт h.
    Если cluster_labels задан — обучает per-cluster модели.
    Возвращает {horizon: [list of (model, feature_cols, cluster_id|None)]}.
    """
    if seeds is None:
        seeds = SEEDS
    if extra_params is None:
        extra_params = {}

    params = {"objective": loss}
    if loss == "tweedie":
        params["tweedie_variance_power"] = 1.5
        params["metric"] = "mae"
    elif loss == "mae":
        params["metric"] = "mae"
    elif loss == "huber":
        params["metric"] = "mae"
    params.update(extra_params)

    models_by_horizon = {}

    for h in horizons:
        print(f"    Горизонт h={h}...", end=" ", flush=True)
        # Строим фичи с min_lag=h
        combined, train_mask = build_features(train_df, test_df, min_lag=h)
        feat_cols = get_feature_cols(combined)

        # Target для горизонта h: значение через h шагов
        combined["target_h"] = combined.groupby("route_id")["target_1h"].shift(-h)

        # Train часть с валидным таргетом
        tr = combined[train_mask & combined["target_h"].notna()].copy()
        ts = combined[~train_mask].copy()

        # Time-based split для early stopping (последнее окно)
        timestamps = tr["timestamp"].sort_values().unique()
        split_ts = timestamps[-8:]  # последние 8 шагов для val
        val_mask_es = tr["timestamp"].isin(set(split_ts)).values
        tr_es = tr[~val_mask_es]
        vl_es = tr[val_mask_es]

        horizon_models = []

        if cluster_labels is not None:
            # Per-cluster models
            for cid in sorted(set(cluster_labels.values())):
                routes_in_cluster = [r for r, c in cluster_labels.items() if c == cid]
                tr_c = tr_es[tr_es["route_id"].isin(routes_in_cluster)]
                vl_c = vl_es[vl_es["route_id"].isin(routes_in_cluster)]
                if len(tr_c) < 100 or len(vl_c) < 10:
                    continue
                for seed in seeds:
                    model = train_lgb_model(
                        tr_c[feat_cols], tr_c["target_h"].values,
                        vl_c[feat_cols], vl_c["target_h"].values,
                        params, seed,
                    )
                    horizon_models.append((model, feat_cols, cid))
        else:
            for seed in seeds:
                model = train_lgb_model(
                    tr_es[feat_cols], tr_es["target_h"].values,
                    vl_es[feat_cols], vl_es["target_h"].values,
                    params, seed,
                )
                horizon_models.append((model, feat_cols, None))

        models_by_horizon[h] = horizon_models
        print(f"{len(horizon_models)} моделей")

    return models_by_horizon


# ── M3: Recursive, Global, Tweedie ───────────────────────

def train_recursive_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_horizon: int,
    seeds: List[int] = None,
) -> List:
    """
    Recursive стратегия: одна модель с horizon-as-feature.
    """
    if seeds is None:
        seeds = SEEDS

    print("    Строим расширенный датасет (horizon-as-feature)...")

    # Берём min_lag=1 для recursive (при инференсе лаги обновляются)
    combined, train_mask = build_features(train_df, test_df, min_lag=1)
    feat_cols = get_feature_cols(combined)

    tr = combined[train_mask].copy()

    # Расширяем train: для каждого h=1..max_horizon
    rows = []
    for h in range(1, max_horizon + 1):
        temp = tr.copy()
        temp["horizon"] = h
        temp["target_h"] = tr.groupby("route_id")["target_1h"].shift(-h)
        rows.append(temp)
    expanded = pd.concat(rows, ignore_index=True).dropna(subset=["target_h"])

    all_feat_cols = feat_cols + ["horizon"]

    # Split для early stopping
    timestamps = tr["timestamp"].sort_values().unique()
    split_ts = set(timestamps[-8:])
    val_mask = expanded["timestamp"].isin(split_ts).values
    tr_exp = expanded[~val_mask]
    vl_exp = expanded[val_mask]

    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "metric": "mae",
    }

    models = []
    for seed in seeds:
        model = train_lgb_model(
            tr_exp[all_feat_cols], tr_exp["target_h"].values,
            vl_exp[all_feat_cols], vl_exp["target_h"].values,
            params, seed,
        )
        models.append((model, all_feat_cols))
    print(f"    {len(models)} recursive моделей обучено")
    return models


# ── M5: Snapshot (baseline-like) ──────────────────────────

def train_snapshot_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seeds: List[int] = None,
) -> List:
    """
    Snapshot: одна модель, использует последнее известное состояние
    status-фичей + лаговые + временные. Для теста — берём последний
    известный status из train.
    """
    if seeds is None:
        seeds = SEEDS

    print("    Строим snapshot-фичи...")

    # Добавляем status-лаги (последний snapshot)
    train_cp = train_df.copy()
    test_cp = test_df.copy()

    # Для теста подставляем последние известные status из train
    status_cols = [c for c in train_cp.columns if c.startswith("status_")]
    if status_cols:
        last_status = (
            train_cp.sort_values("timestamp")
            .groupby("route_id")[status_cols]
            .last()
            .reset_index()
        )
        test_cp = test_cp.merge(last_status, on="route_id", how="left")

    combined, train_mask = build_features(train_cp, test_cp, min_lag=2)

    # Добавляем status cols к фичам
    feat_cols = get_feature_cols(combined)
    # Разрешаем status для snapshot-модели
    for sc in status_cols:
        if sc in combined.columns and sc not in feat_cols:
            feat_cols.append(sc)

    tr = combined[train_mask & combined["target_1h"].notna()].copy()

    # Split
    timestamps = tr["timestamp"].sort_values().unique()
    split_ts = set(timestamps[-8:])
    val_mask = tr["timestamp"].isin(split_ts).values
    tr_es = tr[~val_mask]
    vl_es = tr[val_mask]

    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "metric": "mae",
    }

    models = []
    for seed in seeds:
        model = train_lgb_model(
            tr_es[feat_cols], tr_es["target_1h"].values,
            vl_es[feat_cols], vl_es["target_1h"].values,
            params, seed,
        )
        models.append((model, feat_cols))
    print(f"    {len(models)} snapshot моделей обучено")
    return models


# ╔═══════════════════════════════════════════════════════════╗
# ║         ПРЕДСКАЗАНИЯ ДЛЯ DIRECT МОДЕЛЕЙ                  ║
# ╚═══════════════════════════════════════════════════════════╝

def predict_direct(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_by_horizon: Dict[int, List],
    horizons_map: pd.Series,
    cluster_labels: Optional[dict] = None,
) -> np.ndarray:
    """
    Предсказания Direct моделей для теста.
    horizons_map: Series(index=test index, values=horizon h) — горизонт каждой тестовой строки.
    """
    preds = np.zeros(len(test_df))
    counts = np.zeros(len(test_df))

    for h, model_list in models_by_horizon.items():
        h_mask = horizons_map == h
        if h_mask.sum() == 0:
            continue

        # Строим фичи с min_lag=h
        combined, train_mask = build_features(train_df, test_df, min_lag=h)
        feat_cols_base = get_feature_cols(combined)
        test_part = combined[~train_mask]

        for model, feat_cols, cid in model_list:
            if cid is not None and cluster_labels is not None:
                routes_in_cluster = [r for r, c in cluster_labels.items() if c == cid]
                mask = h_mask & test_df["route_id"].isin(routes_in_cluster)
            else:
                mask = h_mask

            if mask.sum() == 0:
                continue

            # Выравниваем индексы: test_part и test_df могут иметь разный порядок
            # Используем позиционное соответствие через route_id + timestamp
            test_subset_idx = test_df[mask].index
            # Находим соответствующие строки в test_part
            merged_key = test_part.set_index(["route_id", "timestamp"])
            test_key = test_df.loc[test_subset_idx, ["route_id", "timestamp"]]

            pred_rows = merged_key.loc[
                pd.MultiIndex.from_frame(test_key)
            ]

            # Используем те фичи, на которых модель обучалась
            use_cols = [c for c in feat_cols if c in pred_rows.columns]
            p = model.predict(pred_rows[use_cols])
            preds[test_subset_idx] += p
            counts[test_subset_idx] += 1

    # Среднее по seed-ам
    counts = np.maximum(counts, 1)
    return preds / counts


def predict_recursive(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rec_models: List,
    horizons_map: pd.Series,
    max_horizon: int,
) -> np.ndarray:
    """Предсказания Recursive модели."""
    combined, train_mask = build_features(train_df, test_df, min_lag=1)
    feat_cols_base = get_feature_cols(combined)
    test_part = combined[~train_mask].copy()

    preds = np.zeros(len(test_df))
    counts = np.zeros(len(test_df))

    for model, feat_cols in rec_models:
        for h in range(1, max_horizon + 1):
            h_mask = horizons_map == h
            if h_mask.sum() == 0:
                continue

            test_subset_idx = test_df[h_mask].index
            merged_key = test_part.set_index(["route_id", "timestamp"])
            test_key = test_df.loc[test_subset_idx, ["route_id", "timestamp"]]
            pred_rows = merged_key.loc[
                pd.MultiIndex.from_frame(test_key)
            ].copy()
            pred_rows["horizon"] = h

            use_cols = [c for c in feat_cols if c in pred_rows.columns]
            p = model.predict(pred_rows[use_cols])
            preds[test_subset_idx] += p
            counts[test_subset_idx] += 1

    counts = np.maximum(counts, 1)
    return preds / counts


def predict_snapshot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    snap_models: List,
) -> np.ndarray:
    """Предсказания Snapshot модели."""
    train_cp = train_df.copy()
    test_cp = test_df.copy()

    status_cols = [c for c in train_cp.columns if c.startswith("status_")]
    if status_cols:
        last_status = (
            train_cp.sort_values("timestamp")
            .groupby("route_id")[status_cols]
            .last()
            .reset_index()
        )
        test_cp = test_cp.merge(last_status, on="route_id", how="left")

    combined, train_mask = build_features(train_cp, test_cp, min_lag=2)
    test_part = combined[~train_mask]

    preds = np.zeros(len(test_df))
    counts = np.zeros(len(test_df))

    for model, feat_cols in snap_models:
        merged_key = test_part.set_index(["route_id", "timestamp"])
        test_key = test_df[["route_id", "timestamp"]]
        pred_rows = merged_key.loc[pd.MultiIndex.from_frame(test_key)]
        use_cols = [c for c in feat_cols if c in pred_rows.columns]
        p = model.predict(pred_rows[use_cols])
        preds += p
        counts += 1

    counts = np.maximum(counts, 1)
    return preds / counts


# ╔═══════════════════════════════════════════════════════════╗
# ║            OOF ПРЕДСКАЗАНИЯ ДЛЯ ВАЛИДАЦИИ                ║
# ╚═══════════════════════════════════════════════════════════╝

def get_oof_predictions(
    train_df: pd.DataFrame,
    n_windows: int = 4,
    window_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OOF-предсказания всех 6 моделей на валидационных окнах.
    Возвращает: oof_preds (n_val, 6), oof_actuals (n_val,), val_route_ids (n_val,).
    """
    print("\n" + "=" * 60)
    print("ВАЛИДАЦИЯ (OOF)")
    print("=" * 60)

    splits = create_time_splits(train_df, n_windows, window_size)
    print(f"Валидационных окон: {len(splits)}")

    all_preds = []   # list of arrays (n_val_window, 6)
    all_actuals = []
    all_route_ids = []

    for fold_i, (tr_idx, vl_idx) in enumerate(splits):
        print(f"\n--- Fold {fold_i + 1}/{len(splits)} ---")
        tr_fold = train_df.iloc[tr_idx].copy()
        vl_fold = train_df.iloc[vl_idx].copy()

        # Определяем горизонт для val строк
        last_tr_ts = tr_fold.groupby("route_id")["timestamp"].max()
        vl_fold = vl_fold.merge(
            last_tr_ts.rename("last_tr_ts"), on="route_id", how="left"
        )
        vl_fold["horizon"] = (
            (vl_fold["timestamp"] - vl_fold["last_tr_ts"]).dt.total_seconds() / 1800
        ).astype(int)
        horizons_map = vl_fold["horizon"]

        max_h = int(horizons_map.max())
        horizons = list(range(1, max_h + 1))

        # Подготовим vl_fold как "тест" для predict-функций
        vl_as_test = vl_fold.drop(columns=["last_tr_ts", "horizon"], errors="ignore").copy()
        if "id" not in vl_as_test.columns:
            vl_as_test["id"] = range(len(vl_as_test))
        vl_as_test_reset = vl_as_test.reset_index(drop=True)
        horizons_map_reset = horizons_map.reset_index(drop=True)

        fold_preds = np.zeros((len(vl_as_test_reset), 6))

        # Кластеры для M4
        route_means = tr_fold.groupby("route_id")["target_1h"].mean()
        km = KMeans(n_clusters=N_ROUTE_CLUSTERS, random_state=42, n_init=10)
        km.fit(route_means.values.reshape(-1, 1))
        cluster_labels = dict(zip(route_means.index, km.labels_))

        # M1: Direct, Global, Tweedie
        print("  M1: Direct + Tweedie...")
        m1 = train_direct_models(tr_fold, vl_as_test_reset, horizons, loss="tweedie", seeds=[42])
        fold_preds[:, 0] = predict_direct(tr_fold, vl_as_test_reset, m1, horizons_map_reset)

        # M2: Direct, Global, MAE
        print("  M2: Direct + MAE...")
        m2 = train_direct_models(tr_fold, vl_as_test_reset, horizons, loss="mae", seeds=[42])
        fold_preds[:, 1] = predict_direct(tr_fold, vl_as_test_reset, m2, horizons_map_reset)

        # M3: Recursive, Global, Tweedie
        print("  M3: Recursive + Tweedie...")
        m3 = train_recursive_model(tr_fold, vl_as_test_reset, max_h, seeds=[42])
        fold_preds[:, 2] = predict_recursive(tr_fold, vl_as_test_reset, m3, horizons_map_reset, max_h)

        # M4: Direct, Per-route group, MAE
        print("  M4: Direct + Clusters + MAE...")
        m4 = train_direct_models(
            tr_fold, vl_as_test_reset, horizons, loss="mae",
            cluster_labels=cluster_labels, seeds=[42],
        )
        fold_preds[:, 3] = predict_direct(
            tr_fold, vl_as_test_reset, m4, horizons_map_reset,
            cluster_labels=cluster_labels,
        )

        # M5: Snapshot, Global, Tweedie
        print("  M5: Snapshot + Tweedie...")
        m5 = train_snapshot_model(tr_fold, vl_as_test_reset, seeds=[42])
        fold_preds[:, 4] = predict_snapshot(tr_fold, vl_as_test_reset, m5)

        # M6: Direct, Global, Huber
        print("  M6: Direct + Huber...")
        m6 = train_direct_models(tr_fold, vl_as_test_reset, horizons, loss="huber", seeds=[42])
        fold_preds[:, 5] = predict_direct(tr_fold, vl_as_test_reset, m6, horizons_map_reset)

        all_preds.append(fold_preds)
        all_actuals.append(vl_fold["target_1h"].values)
        all_route_ids.append(vl_fold["route_id"].values)

    oof_preds = np.concatenate(all_preds, axis=0)
    oof_actuals = np.concatenate(all_actuals, axis=0)
    oof_route_ids = np.concatenate(all_route_ids, axis=0)

    return oof_preds, oof_actuals, oof_route_ids


# ╔═══════════════════════════════════════════════════════════╗
# ║              ОПТИМИЗАЦИЯ ВЕСОВ АНСАМБЛЯ                   ║
# ╚═══════════════════════════════════════════════════════════╝

def optimize_ensemble_weights(
    oof_preds: np.ndarray,
    oof_actuals: np.ndarray,
    n_trials: int = 100,
) -> np.ndarray:
    """Оптимизация весов через scipy.optimize + dirichlet starts."""
    n_models = oof_preds.shape[1]
    print(f"\nОптимизация весов ({n_models} моделей, {n_trials} стартов)...")

    def loss_fn(w):
        ens = oof_preds @ w
        return np.sum(np.abs(ens - oof_actuals)) / np.maximum(np.sum(oof_actuals), 1e-12)

    best_w, best_score = None, np.inf
    for _ in range(n_trials):
        w0 = np.random.dirichlet(np.ones(n_models))
        res = minimize(
            loss_fn, w0, method="SLSQP",
            bounds=[(0, 1)] * n_models,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        )
        if res.fun < best_score:
            best_score = res.fun
            best_w = res.x

    # Для сравнения — равные веса
    eq_w = np.ones(n_models) / n_models
    eq_score = loss_fn(eq_w)

    print(f"  Равные веса WAPE: {eq_score:.5f}")
    print(f"  Оптим. веса WAPE: {best_score:.5f}")
    print(f"  Веса: {np.round(best_w, 4)}")

    return best_w


# ╔═══════════════════════════════════════════════════════════╗
# ║                 BIAS CORRECTION                           ║
# ╚═══════════════════════════════════════════════════════════╝

def compute_bias_correction(
    oof_preds_ensemble: np.ndarray,
    oof_actuals: np.ndarray,
    oof_route_ids: np.ndarray,
) -> Tuple[float, dict]:
    """
    Вычисляет глобальный и per-route bias correction ratio.
    """
    # Глобальный
    global_ratio = oof_actuals.sum() / np.maximum(oof_preds_ensemble.sum(), 1e-12)
    print(f"\n  Глобальный bias ratio: {global_ratio:.5f}")

    # Per-route
    route_ratios = {}
    for rid in np.unique(oof_route_ids):
        mask = oof_route_ids == rid
        actual_sum = oof_actuals[mask].sum()
        pred_sum = oof_preds_ensemble[mask].sum()
        if pred_sum > 1e-12 and actual_sum > 1e-12:
            route_ratios[rid] = actual_sum / pred_sum
        else:
            route_ratios[rid] = global_ratio

    return global_ratio, route_ratios


# ╔═══════════════════════════════════════════════════════════╗
# ║                ФИНАЛЬНОЕ ОБУЧЕНИЕ + САБМИТ                ║
# ╚═══════════════════════════════════════════════════════════╝

def train_final_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    weights: np.ndarray,
    global_ratio: float,
    route_ratios: dict,
) -> np.ndarray:
    """Обучение всех 6 моделей на полном train и предсказание на test."""
    print("\n" + "=" * 60)
    print("ФИНАЛЬНОЕ ОБУЧЕНИЕ НА ВСЁМ TRAIN")
    print("=" * 60)

    # Определяем горизонт тестовых строк
    last_train_ts = train_df.groupby("route_id")["timestamp"].max()
    test_with_h = test_df.merge(last_train_ts.rename("last_tr_ts"), on="route_id")
    test_with_h["horizon"] = (
        (test_with_h["timestamp"] - test_with_h["last_tr_ts"]).dt.total_seconds() / 1800
    ).astype(int)
    horizons_map = test_with_h["horizon"]
    max_h = int(horizons_map.max())
    horizons = list(range(1, max_h + 1))
    print(f"  Макс горизонт теста: {max_h} шагов")

    # Кластеры
    route_means = train_df.groupby("route_id")["target_1h"].mean()
    km = KMeans(n_clusters=N_ROUTE_CLUSTERS, random_state=42, n_init=10)
    km.fit(route_means.values.reshape(-1, 1))
    cluster_labels = dict(zip(route_means.index, km.labels_))

    test_reset = test_df.reset_index(drop=True)
    horizons_map_reset = horizons_map.reset_index(drop=True)

    all_preds = np.zeros((len(test_reset), 6))

    # M1: Direct, Global, Tweedie (seed averaging)
    print("\nM1: Direct + Tweedie (seed avg)...")
    m1 = train_direct_models(train_df, test_reset, horizons, loss="tweedie", seeds=SEEDS)
    all_preds[:, 0] = predict_direct(train_df, test_reset, m1, horizons_map_reset)

    # M2: Direct, Global, MAE
    print("\nM2: Direct + MAE (seed avg)...")
    m2 = train_direct_models(train_df, test_reset, horizons, loss="mae", seeds=SEEDS)
    all_preds[:, 1] = predict_direct(train_df, test_reset, m2, horizons_map_reset)

    # M3: Recursive, Global, Tweedie
    print("\nM3: Recursive + Tweedie (seed avg)...")
    m3 = train_recursive_model(train_df, test_reset, max_h, seeds=SEEDS)
    all_preds[:, 2] = predict_recursive(train_df, test_reset, m3, horizons_map_reset, max_h)

    # M4: Direct, Per-route group, MAE
    print("\nM4: Direct + Clusters + MAE (seed avg)...")
    m4 = train_direct_models(
        train_df, test_reset, horizons, loss="mae",
        cluster_labels=cluster_labels, seeds=SEEDS,
    )
    all_preds[:, 3] = predict_direct(
        train_df, test_reset, m4, horizons_map_reset,
        cluster_labels=cluster_labels,
    )

    # M5: Snapshot, Global, Tweedie
    print("\nM5: Snapshot + Tweedie (seed avg)...")
    m5 = train_snapshot_model(train_df, test_reset, seeds=SEEDS)
    all_preds[:, 4] = predict_snapshot(train_df, test_reset, m5)

    # M6: Direct, Global, Huber
    print("\nM6: Direct + Huber (seed avg)...")
    m6 = train_direct_models(train_df, test_reset, horizons, loss="huber", seeds=SEEDS)
    all_preds[:, 5] = predict_direct(train_df, test_reset, m6, horizons_map_reset)

    # Ансамбль
    ensemble_pred = all_preds @ weights

    # Bias correction (глобальный)
    pred_corrected_global = ensemble_pred * global_ratio

    # Bias correction (per-route)
    pred_corrected_route = ensemble_pred.copy()
    for rid, ratio in route_ratios.items():
        mask = test_reset["route_id"] == rid
        pred_corrected_route[mask.values] *= ratio

    # Выбираем лучший вариант (используем глобальный как дефолт,
    # потому что per-route может оверфитить на малом val)
    print(f"\n  Средний pred (raw):    {ensemble_pred.mean():.4f}")
    print(f"  Средний pred (global): {pred_corrected_global.mean():.4f}")
    print(f"  Средний pred (route):  {pred_corrected_route.mean():.4f}")

    final_pred = pred_corrected_global

    # Isotonic calibration — пропускаем без val данных для финального прогона,
    # используется только bias correction

    # Клиппинг
    final_pred = np.maximum(final_pred, 0)

    # Feature importance (из M1, первая модель, первый горизонт)
    save_feature_importance(m1)

    return final_pred


def save_feature_importance(models_by_horizon: Dict[int, List]):
    """Сохраняет график feature importance (первая модель, первый горизонт)."""
    first_h = min(models_by_horizon.keys())
    model, feat_cols, _ = models_by_horizon[first_h][0]
    imp = model.feature_importances_
    sorted_idx = np.argsort(imp)[-30:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feat_cols[i] for i in sorted_idx],
        imp[sorted_idx],
        color="steelblue",
    )
    ax.set_title(f"Top-30 Feature Importance (M1, h={first_h})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("\nСохранено: feature_importance.png")


# ╔═══════════════════════════════════════════════════════════╗
# ║                      ОТЧЁТ                               ║
# ╚═══════════════════════════════════════════════════════════╝

def write_val_report(
    oof_preds: np.ndarray,
    oof_actuals: np.ndarray,
    weights: np.ndarray,
):
    """Пишет отчёт о метриках валидации."""
    model_names = ["M1_Direct_Tweedie", "M2_Direct_MAE", "M3_Recursive_Tweedie",
                   "M4_Direct_Cluster_MAE", "M5_Snapshot_Tweedie", "M6_Direct_Huber"]

    lines = ["=" * 60, "VALIDATION METRICS REPORT", "=" * 60, ""]

    for i, name in enumerate(model_names):
        p = oof_preds[:, i]
        w = wape(oof_actuals, p)
        r = rbias(oof_actuals, p)
        s = w + r
        lines.append(f"{name:30s}  WAPE={w:.5f}  |Bias|={r:.5f}  Score={s:.5f}")

    # Равный ансамбль
    eq_ens = oof_preds.mean(axis=1)
    w_eq = wape(oof_actuals, eq_ens)
    r_eq = rbias(oof_actuals, eq_ens)
    lines.append(f"\n{'Equal Ensemble':30s}  WAPE={w_eq:.5f}  |Bias|={r_eq:.5f}  Score={w_eq + r_eq:.5f}")

    # Оптимальный ансамбль
    opt_ens = oof_preds @ weights
    w_opt = wape(oof_actuals, opt_ens)
    r_opt = rbias(oof_actuals, opt_ens)
    lines.append(f"{'Optimized Ensemble':30s}  WAPE={w_opt:.5f}  |Bias|={r_opt:.5f}  Score={w_opt + r_opt:.5f}")

    # С bias correction
    ratio = oof_actuals.sum() / np.maximum(opt_ens.sum(), 1e-12)
    corrected = opt_ens * ratio
    w_c = wape(oof_actuals, corrected)
    r_c = rbias(oof_actuals, corrected)
    lines.append(f"{'+ Bias Correction':30s}  WAPE={w_c:.5f}  |Bias|={r_c:.5f}  Score={w_c + r_c:.5f}")

    lines.append(f"\nОптимальные веса: {np.round(weights, 4)}")
    lines.append(f"Bias ratio: {ratio:.5f}")

    report = "\n".join(lines)
    print("\n" + report)

    with open("val_metrics_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nСохранено: val_metrics_report.txt")


# ╔═══════════════════════════════════════════════════════════╗
# ║                      MAIN                                 ║
# ╚═══════════════════════════════════════════════════════════╝

def main():
    t0 = time.time()

    # 1. Загрузка
    train_df, test_df = load_data()

    # 2. OOF-валидация → получаем предсказания всех 6 моделей
    oof_preds, oof_actuals, oof_route_ids = get_oof_predictions(
        train_df, n_windows=4, window_size=8,
    )

    # 3. Оптимизация весов ансамбля
    weights = optimize_ensemble_weights(oof_preds, oof_actuals, n_trials=100)

    # 4. Bias correction на OOF
    oof_ensemble = oof_preds @ weights
    global_ratio, route_ratios = compute_bias_correction(
        oof_ensemble, oof_actuals, oof_route_ids,
    )

    # 5. Отчёт по валидации
    write_val_report(oof_preds, oof_actuals, weights)

    # 6. Финальное обучение + предсказание
    final_pred = train_final_and_predict(
        train_df, test_df, weights, global_ratio, route_ratios,
    )

    # 7. Формирование submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "y_pred": final_pred,
    })
    assert submission["id"].isna().sum() == 0, "Есть NaN в id!"
    assert (submission["y_pred"] >= 0).all(), "Есть отрицательные предсказания!"

    submission.to_csv("submission_solo.csv", index=False)
    print(f"\nСохранено: submission_solo.csv ({len(submission)} строк)")
    print(f"\nОбщее время: {time.time() - t0:.0f} сек")


if __name__ == "__main__":
    main()
