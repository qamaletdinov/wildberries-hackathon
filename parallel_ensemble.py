"""Leakage-safe parallel ensemble for the Wildberries forecasting task.

The script deliberately leaves the historical ``solution.py`` untouched.  It
implements the common, test-like evaluation contract for seasonal baselines,
LightGBM, XGBoost, CatBoost and a causal temporal convolutional network (TCN).

Examples
--------
Fast end-to-end verification on 100 routes and one fold::

    python parallel_ensemble.py --smoke

Larger CPU/GPU backtest::

    python parallel_ensemble.py --folds 4 --lookback-days 42 \
        --origin-stride 4 --models seasonal,lightgbm,xgboost,catboost,tcn

The forecast unit is ``route_id x future timestamp``.  At each validation
origin the models see only rows at or before that origin and forecast the next
eight 30-minute slots, exactly as in the supplied test set.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TARGET = "target_1h"
STATUS_COLUMNS = tuple(f"status_{i}" for i in range(1, 7))
HORIZON = 8
STEPS_PER_DAY = 48
STEPS_PER_WEEK = 336
SEED = 42


@dataclass(frozen=True)
class RunConfig:
    data_dir: Path
    output_dir: Path
    folds: int = 4
    fold_spacing: int = STEPS_PER_DAY
    lookback_days: int = 42
    origin_stride: int = 4
    max_routes: int | None = None
    models: tuple[str, ...] = ("seasonal", "lightgbm", "xgboost", "catboost", "tcn")
    max_rounds: int = 700
    early_stopping_rounds: int = 60
    catboost_loss: str = "RMSE"
    use_gpu: bool = True
    tcn_sequence_length: int = STEPS_PER_WEEK
    tcn_architecture: str = "two_stream"
    tcn_channels: int = 64
    tcn_epochs: int = 12
    tcn_batch_size: int = 256
    tcn_patience: int = 3


@dataclass(frozen=True)
class PanelData:
    route_ids: np.ndarray
    timestamps: pd.DatetimeIndex
    target: np.ndarray
    statuses: np.ndarray
    future_timestamps: pd.DatetimeIndex
    test_ids: np.ndarray


@dataclass(frozen=True)
class Fold:
    fold: int
    origin_idx: int
    origin_timestamp: pd.Timestamp
    validation_timestamps: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class MatrixBundle:
    X: np.ndarray
    y: np.ndarray | None
    origins: np.ndarray
    feature_names: tuple[str, ...]
    route_codes: np.ndarray
    horizons: np.ndarray


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = max(float(np.abs(y_true).sum()), 1e-12)
    return float(np.abs(y_pred - y_true).sum() / denominator)


def relative_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = max(float(np.abs(y_true).sum()), 1e-12)
    return float(abs(float(y_pred.sum() - y_true.sum())) / denominator)


def competition_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return wape(y_true, y_pred) + relative_bias(y_true, y_pred)


def metric_record(
    fold: int | str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int | str]:
    return {
        "fold": fold,
        "model": model,
        "n": int(y_true.size),
        "wape": wape(y_true, y_pred),
        "relative_bias": relative_bias(y_true, y_pred),
        "score": competition_score(y_true, y_pred),
        "prediction_mean": float(y_pred.mean()),
        "target_mean": float(y_true.mean()),
    }


def _assert_regular_grid(train: pd.DataFrame, route_ids: np.ndarray) -> pd.DatetimeIndex:
    counts = train.groupby("timestamp", sort=True)["route_id"].nunique()
    if counts.nunique() != 1 or int(counts.iloc[0]) != len(route_ids):
        raise ValueError("Train is not a complete timestamp x route grid")
    timestamps = pd.DatetimeIndex(counts.index)
    delta = timestamps.to_series().diff().dropna()
    if not (delta == pd.Timedelta(minutes=30)).all():
        raise ValueError("Train timestamps are not a regular 30-minute grid")
    duplicated = train.duplicated(["timestamp", "route_id"]).sum()
    if duplicated:
        raise ValueError(f"Found {duplicated} duplicate timestamp-route keys")
    return timestamps


def load_panel(data_dir: Path, max_routes: int | None = None) -> PanelData:
    train_path = data_dir / "train_solo_track.parquet"
    test_path = data_dir / "test_solo_track.parquet"
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    required_train = {"route_id", "timestamp", TARGET, *STATUS_COLUMNS}
    required_test = {"id", "route_id", "timestamp"}
    if missing := required_train.difference(train.columns):
        raise ValueError(f"Missing train columns: {sorted(missing)}")
    if missing := required_test.difference(test.columns):
        raise ValueError(f"Missing test columns: {sorted(missing)}")

    train["timestamp"] = pd.to_datetime(train["timestamp"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    route_ids = np.sort(train["route_id"].unique())
    if max_routes is not None:
        route_ids = route_ids[:max_routes]
        train = train[train["route_id"].isin(route_ids)].copy()
        test = test[test["route_id"].isin(route_ids)].copy()

    timestamps = _assert_regular_grid(train, route_ids)
    ordered = train.sort_values(["timestamp", "route_id"], kind="stable")
    ordered_route_grid = ordered["route_id"].to_numpy().reshape(len(timestamps), -1)
    expected_route_grid = np.broadcast_to(route_ids, ordered_route_grid.shape)
    if not np.array_equal(ordered_route_grid, expected_route_grid):
        raise ValueError("Route order differs between timestamps")

    target = ordered[TARGET].to_numpy(dtype=np.float32).reshape(len(timestamps), -1)
    statuses = ordered[list(STATUS_COLUMNS)].to_numpy(dtype=np.float32)
    statuses = statuses.reshape(len(timestamps), len(route_ids), len(STATUS_COLUMNS))

    future_timestamps = pd.DatetimeIndex(np.sort(test["timestamp"].unique()))
    if len(future_timestamps) != HORIZON:
        raise ValueError(f"Expected {HORIZON} test timestamps, got {len(future_timestamps)}")
    expected_future = pd.date_range(timestamps[-1] + pd.Timedelta(minutes=30), periods=HORIZON, freq="30min")
    if not future_timestamps.equals(expected_future):
        raise ValueError("Test timestamps are not the eight steps immediately after train")

    test_ordered = test.sort_values(["timestamp", "route_id"], kind="stable")
    expected_rows = len(route_ids) * HORIZON
    if len(test_ordered) != expected_rows:
        raise ValueError(f"Expected {expected_rows} test rows after route filtering, got {len(test_ordered)}")

    return PanelData(
        route_ids=route_ids,
        timestamps=timestamps,
        target=target,
        statuses=statuses,
        future_timestamps=future_timestamps,
        test_ids=test_ordered["id"].to_numpy(),
    )


def make_folds(panel: PanelData, n_folds: int, spacing: int) -> list[Fold]:
    folds: list[Fold] = []
    # The real forecast is issued at the last train time (10:30) for
    # 11:00--14:30.  Backtest origins use the same time of day, so daily
    # seasonality does not make validation artificially easier or harder.
    deployment_time = panel.timestamps[-1].time()
    candidates = np.flatnonzero(
        np.asarray([timestamp.time() == deployment_time for timestamp in panel.timestamps])
        & (np.arange(len(panel.timestamps)) + HORIZON < len(panel.timestamps))
    )
    if not len(candidates):
        raise ValueError("No historical timestamp can mimic the deployment forecast origin")
    last_origin = int(candidates[-1])
    for reverse_position in range(n_folds - 1, -1, -1):
        origin = last_origin - reverse_position * spacing
        if panel.timestamps[origin].time() != deployment_time:
            raise ValueError("Fold spacing does not preserve the deployment time of day")
        validation = panel.timestamps[origin + 1 : origin + HORIZON + 1]
        if len(validation) != HORIZON:
            raise ValueError("A validation fold does not contain eight steps")
        folds.append(
            Fold(
                fold=len(folds),
                origin_idx=origin,
                origin_timestamp=panel.timestamps[origin],
                validation_timestamps=tuple(validation),
            )
        )
    return folds


def _cyclic_features(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    slot = timestamps.hour.to_numpy() * 2 + (timestamps.minute.to_numpy() >= 30)
    day_of_week = timestamps.dayofweek.to_numpy()
    return {
        "future_hour_sin": np.sin(2 * np.pi * slot / STEPS_PER_DAY),
        "future_hour_cos": np.cos(2 * np.pi * slot / STEPS_PER_DAY),
        "future_dow_sin": np.sin(2 * np.pi * day_of_week / 7),
        "future_dow_cos": np.cos(2 * np.pi * day_of_week / 7),
        "future_is_weekend": (day_of_week >= 5).astype(np.float32),
    }


def _candidate_origins(
    max_origin: int,
    lookback_steps: int,
    stride: int,
    minimum_history: int = STEPS_PER_WEEK + HORIZON,
) -> np.ndarray:
    start = max(minimum_history, max_origin - lookback_steps + 1)
    origins = np.arange(start, max_origin + 1, stride, dtype=np.int32)
    if len(origins) < 4:
        raise ValueError("Too few training origins; increase lookback or reduce stride")
    return origins


def _feature_chunk(
    panel: PanelData,
    origin: int,
    future_timestamps: pd.DatetimeIndex,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Create N_routes x H rows using information available at ``origin`` only."""
    y = panel.target
    status = panel.statuses
    n_routes = y.shape[1]
    horizons = np.arange(1, HORIZON + 1, dtype=np.float32)
    route_code = np.arange(n_routes, dtype=np.float32)
    route_grid = np.tile(route_code, HORIZON)
    horizon_grid = np.repeat(horizons, n_routes)

    columns: list[np.ndarray] = [route_grid, horizon_grid]
    names = ["route_code", "horizon"]

    cyclic = _cyclic_features(future_timestamps)
    for name, values in cyclic.items():
        columns.append(np.repeat(values.astype(np.float32), n_routes))
        names.append(name)

    # Recent values relative to the forecast origin.  These never reach into
    # the forecast horizon, even when the training label is later in time.
    for lag in (0, 1, 2, 3, 5, 7, 11, 23, 47, 95, 335):
        columns.append(np.tile(y[origin - lag], HORIZON))
        names.append(f"origin_target_lag_{lag}")

    # Same-slot seasonal values relative to each target timestamp.  Because
    # every seasonal lag is > H=8, all values are known at the origin.
    target_indices = origin + np.arange(1, HORIZON + 1)
    for lag in (STEPS_PER_DAY, 2 * STEPS_PER_DAY, STEPS_PER_WEEK):
        values = y[target_indices - lag].reshape(-1)
        columns.append(values)
        names.append(f"target_time_lag_{lag}")

    for window in (6, 12, 24, 48, 96, 336):
        history = y[origin - window + 1 : origin + 1]
        for statistic, values in (
            ("mean", history.mean(axis=0)),
            ("std", history.std(axis=0)),
            ("min", history.min(axis=0)),
            ("max", history.max(axis=0)),
        ):
            columns.append(np.tile(values, HORIZON))
            names.append(f"history_{statistic}_{window}")

    columns.extend(
        [
            np.tile(y[origin] - y[origin - 1], HORIZON),
            np.tile(y[origin] - y[origin - 47 : origin + 1].mean(axis=0), HORIZON),
            np.tile(y[origin - 5 : origin + 1].mean(axis=0) - y[origin - 47 : origin + 1].mean(axis=0), HORIZON),
        ]
    )
    names.extend(["last_difference", "last_minus_daily_mean", "short_minus_daily_mean"])

    # Only status snapshots at/before the forecast origin are used.  This
    # gives train/test feature parity even though test has no future statuses.
    for status_idx, status_name in enumerate(STATUS_COLUMNS):
        columns.append(np.tile(status[origin, :, status_idx], HORIZON))
        names.append(f"origin_{status_name}")
        columns.append(np.tile(status[origin, :, status_idx] - status[origin - 1, :, status_idx], HORIZON))
        names.append(f"origin_{status_name}_delta_1")
        columns.append(np.tile(status[origin, :, status_idx] - status[origin - STEPS_PER_DAY, :, status_idx], HORIZON))
        names.append(f"origin_{status_name}_delta_48")

    X = np.column_stack(columns).astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite feature at origin {origin}")
    return X, tuple(names)


def build_matrix(
    panel: PanelData,
    origins: Sequence[int],
    include_labels: bool,
    deployment_future: bool = False,
) -> MatrixBundle:
    chunks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    origin_labels: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    n_routes = len(panel.route_ids)
    for origin in origins:
        if deployment_future:
            future_timestamps = panel.future_timestamps
        else:
            future_timestamps = panel.timestamps[origin + 1 : origin + HORIZON + 1]
        X_chunk, current_names = _feature_chunk(panel, int(origin), future_timestamps)
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise AssertionError("Feature order changed between origins")
        chunks.append(X_chunk)
        origin_labels.append(np.full(n_routes * HORIZON, origin, dtype=np.int32))
        if include_labels:
            labels.append(panel.target[origin + 1 : origin + HORIZON + 1].reshape(-1))

    X = np.concatenate(chunks, axis=0)
    y = np.concatenate(labels).astype(np.float32, copy=False) if labels else None
    origin_array = np.concatenate(origin_labels)
    route_codes = X[:, 0].astype(np.int32)
    horizons = X[:, 1].astype(np.int8)
    if y is not None and len(y) != len(X):
        raise AssertionError("Feature/label row mismatch")
    return MatrixBundle(X, y, origin_array, feature_names or (), route_codes, horizons)


def seasonal_predictions(panel: PanelData, origin: int) -> dict[str, np.ndarray]:
    target_indices = origin + np.arange(1, HORIZON + 1)
    daily = panel.target[target_indices - STEPS_PER_DAY].reshape(-1)
    weekly = panel.target[target_indices - STEPS_PER_WEEK].reshape(-1)
    two_day = panel.target[target_indices - 2 * STEPS_PER_DAY].reshape(-1)
    robust_blend = np.median(np.stack([daily, two_day, weekly]), axis=0)
    return {
        "seasonal_daily": daily,
        "seasonal_weekly": weekly,
        "seasonal_median": robust_blend,
    }


class TCNWindowDataset:
    """Lazy origin-route windows; no future value is materialized as input."""

    def __init__(
        self,
        panel: PanelData,
        origins: Sequence[int],
        sequence_length: int,
    ) -> None:
        import torch

        self.panel = panel
        self.origins = np.asarray(origins, dtype=np.int32)
        self.sequence_length = sequence_length
        self.n_routes = len(panel.route_ids)
        if self.origins.min() - sequence_length + 1 < 0:
            raise ValueError("TCN origin does not have a complete input window")
        if self.origins.max() + HORIZON >= len(panel.timestamps):
            raise ValueError("TCN training/validation origin does not have complete labels")
        # Keeps Dataset compatible with torch without importing it at module load.
        self._tensor = torch.from_numpy

    def __len__(self) -> int:
        return len(self.origins) * self.n_routes

    def __getitem__(self, index: int):
        origin_position, route = divmod(index, self.n_routes)
        origin = int(self.origins[origin_position])
        start = origin - self.sequence_length + 1

        target_history = self.panel.target[start : origin + 1, route].astype(np.float32, copy=False)
        status_history = self.panel.statuses[start : origin + 1, route].astype(np.float32, copy=False)
        target_scale = np.float32(np.abs(target_history).mean() + 1.0)
        status_scales = np.abs(status_history).mean(axis=0).astype(np.float32) + 1.0
        sequence = np.column_stack(
            [target_history / target_scale, status_history / status_scales]
        ).astype(np.float32, copy=False)

        scales = np.concatenate(([target_scale], status_scales)).astype(np.float32)
        log_scales = np.log1p(scales).astype(np.float32) / np.float32(15.0)
        future_timestamps = self.panel.timestamps[origin + 1 : origin + HORIZON + 1]
        cyclic = _cyclic_features(future_timestamps)
        future_time = np.column_stack(list(cyclic.values())).astype(np.float32)
        target = (
            self.panel.target[origin + 1 : origin + HORIZON + 1, route] / target_scale
        ).astype(np.float32)

        return (
            self._tensor(sequence.T.copy()),
            np.int64(route),
            self._tensor(log_scales),
            self._tensor(future_time),
            self._tensor(target),
            np.float32(target_scale),
        )


def _make_tcn_model(n_routes: int, channels: int, architecture: str):
    import torch
    from torch import nn
    from torch.nn import functional

    class CausalResidualBlock(nn.Module):
        def __init__(self, width: int, dilation: int) -> None:
            super().__init__()
            self.padding = 2 * dilation
            self.conv1 = nn.Conv1d(width, width, kernel_size=3, dilation=dilation)
            self.conv2 = nn.Conv1d(width, width, kernel_size=3, dilation=dilation)
            self.norm1 = nn.GroupNorm(8, width)
            self.norm2 = nn.GroupNorm(8, width)
            self.dropout = nn.Dropout(0.10)

        def _causal_conv(self, values, convolution):
            return convolution(functional.pad(values, (self.padding, 0)))

        def forward(self, values):
            residual = values
            values = self._causal_conv(values, self.conv1)
            values = self.dropout(functional.gelu(self.norm1(values)))
            values = self._causal_conv(values, self.conv2)
            values = self.dropout(functional.gelu(self.norm2(values)))
            return functional.gelu(values + residual)

    def causal_stack() -> nn.Sequential:
        return nn.Sequential(
            *(CausalResidualBlock(channels, dilation) for dilation in (1, 2, 4, 8, 16, 32, 64))
        )

    class DirectForecastHead(nn.Module):
        def __init__(self, temporal_context_size: int) -> None:
            super().__init__()
            embedding_size = 16
            self.route_embedding = nn.Embedding(n_routes, embedding_size)
            context_size = temporal_context_size + embedding_size + 1 + len(STATUS_COLUMNS)
            future_size = 5
            self.head = nn.Sequential(
                nn.Linear(context_size + future_size, 128),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        def forecast(self, temporal_context, route, log_scales, future_time):
            context = torch.cat(
                [temporal_context, self.route_embedding(route), log_scales], dim=1
            )
            context = context[:, None, :].expand(-1, HORIZON, -1)
            decoded = self.head(torch.cat([context, future_time], dim=2)).squeeze(-1)
            return functional.softplus(decoded)

    class SingleStreamGlobalTCN(DirectForecastHead):
        def __init__(self) -> None:
            super().__init__(temporal_context_size=channels)
            self.input_projection = nn.Conv1d(1 + len(STATUS_COLUMNS), channels, kernel_size=1)
            self.blocks = causal_stack()

        def forward(self, sequence, route, log_scales, future_time):
            encoded = self.input_projection(sequence)
            encoded = self.blocks(encoded)[:, :, -1]
            return self.forecast(encoded, route, log_scales, future_time)

    class TwoStreamGlobalTCN(DirectForecastHead):
        """Separate temporal filters for demand and operational statuses."""

        def __init__(self) -> None:
            # Keep both original embeddings and a learned interaction.  This
            # lets the head ignore an unhelpful status branch without erasing
            # target-only information.
            super().__init__(temporal_context_size=3 * channels)
            self.target_projection = nn.Conv1d(1, channels, kernel_size=1)
            self.status_projection = nn.Conv1d(len(STATUS_COLUMNS), channels, kernel_size=1)
            self.target_blocks = causal_stack()
            self.status_blocks = causal_stack()
            self.fusion = nn.Sequential(
                nn.Linear(2 * channels, 2 * channels),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(2 * channels, channels),
                nn.GELU(),
            )

        def forward(self, sequence, route, log_scales, future_time):
            target_context = self.target_blocks(self.target_projection(sequence[:, :1]))[:, :, -1]
            status_context = self.status_blocks(self.status_projection(sequence[:, 1:]))[:, :, -1]
            interaction = self.fusion(torch.cat([target_context, status_context], dim=1))
            temporal_context = torch.cat([target_context, status_context, interaction], dim=1)
            return self.forecast(temporal_context, route, log_scales, future_time)

    if architecture == "single":
        return SingleStreamGlobalTCN()
    if architecture == "two_stream":
        return TwoStreamGlobalTCN()
    raise ValueError(f"Unknown TCN architecture: {architecture}")


def train_tcn(
    panel: PanelData,
    training_origins: np.ndarray,
    validation_origin: int,
    config: RunConfig,
) -> np.ndarray:
    """Train a global causal TCN and return horizon-major validation predictions."""
    import torch
    from torch.utils.data import DataLoader

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
    if config.use_gpu and device.type != "cuda":
        raise RuntimeError("TCN GPU was requested but CUDA is unavailable")
    torch.set_float32_matmul_precision("high")

    n_inner = max(2, math.ceil(len(training_origins) * 0.1))
    if n_inner >= len(training_origins):
        n_inner = 1
    fit_origins = training_origins[:-n_inner]
    inner_origins = training_origins[-n_inner:]
    fit_dataset = TCNWindowDataset(panel, fit_origins, config.tcn_sequence_length)
    inner_dataset = TCNWindowDataset(panel, inner_origins, config.tcn_sequence_length)
    outer_dataset = TCNWindowDataset(panel, [validation_origin], config.tcn_sequence_length)
    loader_arguments = {
        "batch_size": config.tcn_batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    fit_loader = DataLoader(fit_dataset, shuffle=True, **loader_arguments)
    inner_loader = DataLoader(inner_dataset, shuffle=False, **loader_arguments)
    outer_loader = DataLoader(outer_dataset, shuffle=False, **loader_arguments)

    model = _make_tcn_model(
        len(panel.route_ids), config.tcn_channels, config.tcn_architecture
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.tcn_epochs)
    global_mean = float(panel.target[: int(training_origins[-1]) + 1].mean())
    best_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    def autocast_context():
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def predict_loader(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        predicted_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        with torch.inference_mode():
            for sequence, route, log_scales, future_time, target, target_scale in loader:
                sequence = sequence.to(device, non_blocking=True)
                route = route.to(device, non_blocking=True)
                log_scales = log_scales.to(device, non_blocking=True)
                future_time = future_time.to(device, non_blocking=True)
                with autocast_context():
                    prediction = model(sequence, route, log_scales, future_time)
                scale = target_scale.numpy()[:, None]
                predicted_batches.append(prediction.float().cpu().numpy() * scale)
                target_batches.append(target.numpy() * scale)
        return np.concatenate(target_batches), np.concatenate(predicted_batches)

    for epoch in range(config.tcn_epochs):
        model.train()
        epoch_loss = 0.0
        examples = 0
        for sequence, route, log_scales, future_time, target, target_scale in fit_loader:
            sequence = sequence.to(device, non_blocking=True)
            route = route.to(device, non_blocking=True)
            log_scales = log_scales.to(device, non_blocking=True)
            future_time = future_time.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_scale = target_scale.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                prediction = model(sequence, route, log_scales, future_time)
                raw_absolute_error = torch.abs(prediction - target) * target_scale[:, None]
                loss = raw_absolute_error.mean() / max(global_mean, 1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_examples = len(route)
            epoch_loss += float(loss.detach()) * batch_examples
            examples += batch_examples
        scheduler.step()

        inner_target, inner_prediction = predict_loader(inner_loader)
        inner_score = competition_score(inner_target.reshape(-1), inner_prediction.reshape(-1))
        print(
            f"    TCN epoch {epoch + 1:02d}: train_loss={epoch_loss / max(examples, 1):.5f}, "
            f"inner_score={inner_score:.5f}",
            flush=True,
        )
        if inner_score < best_score - 1e-5:
            best_score = inner_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.tcn_patience:
                break

    model.load_state_dict(best_state)
    _, outer_prediction = predict_loader(outer_loader)
    # Dataset order is route-major while the submission/backtest order is
    # horizon-major, matching panel.target[time, route].reshape(-1).
    return np.maximum(outer_prediction.T.reshape(-1), 0).astype(np.float64)


def _inner_train_validation(bundle: MatrixBundle) -> tuple[np.ndarray, np.ndarray]:
    unique_origins = np.unique(bundle.origins)
    n_inner = max(2, math.ceil(len(unique_origins) * 0.1))
    if n_inner >= len(unique_origins):
        n_inner = 1
    boundary = unique_origins[-n_inner]
    inner_val = bundle.origins >= boundary
    inner_train = ~inner_val
    if not inner_train.any() or not inner_val.any():
        raise ValueError("Could not construct chronological inner validation")
    return inner_train, inner_val


def train_lightgbm(
    train: MatrixBundle,
    validation: MatrixBundle,
    config: RunConfig,
) -> np.ndarray:
    import lightgbm as lgb

    train_mask, inner_mask = _inner_train_validation(train)
    model = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=config.max_rounds,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=SEED,
        verbosity=-1,
    )
    model.fit(
        train.X[train_mask],
        train.y[train_mask],
        eval_X=train.X[inner_mask],
        eval_y=train.y[inner_mask],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
        categorical_feature=[0, 1],
    )
    return np.maximum(model.predict(validation.X), 0).astype(np.float64)


def train_xgboost(
    train: MatrixBundle,
    validation: MatrixBundle,
    config: RunConfig,
) -> np.ndarray:
    from xgboost import XGBRegressor

    train_mask, inner_mask = _inner_train_validation(train)
    model = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=config.max_rounds,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda" if config.use_gpu else "cpu",
        random_state=SEED,
        n_jobs=-1,
        eval_metric="mae",
        early_stopping_rounds=config.early_stopping_rounds,
    )
    model.fit(
        train.X[train_mask],
        train.y[train_mask],
        eval_set=[(train.X[inner_mask], train.y[inner_mask])],
        verbose=False,
    )
    return np.maximum(model.predict(validation.X), 0).astype(np.float64)


def train_catboost(
    train: MatrixBundle,
    validation: MatrixBundle,
    config: RunConfig,
) -> np.ndarray:
    from catboost import CatBoostRegressor, FeaturesData, Pool

    train_mask, inner_mask = _inner_train_validation(train)

    def pool_from_rows(row_mask: np.ndarray | slice, labels: np.ndarray | None = None) -> Pool:
        values = train.X[row_mask] if labels is not None else validation.X
        numeric = np.ascontiguousarray(values[:, 2:], dtype=np.float32)
        route_levels = np.asarray(
            [str(index).encode("ascii") for index in range(int(values[:, 0].max()) + 1)],
            dtype=object,
        )
        horizon_levels = np.asarray(
            [str(index).encode("ascii") for index in range(HORIZON + 1)],
            dtype=object,
        )
        categorical = np.column_stack(
            [route_levels[values[:, 0].astype(np.int32)], horizon_levels[values[:, 1].astype(np.int32)]]
        ).astype(object, copy=False)
        features = FeaturesData(
            num_feature_data=numeric,
            cat_feature_data=categorical,
            num_feature_names=list(train.feature_names[2:]),
            cat_feature_names=["route_code", "horizon"],
        )
        return Pool(features, label=labels)

    fit_pool = pool_from_rows(train_mask, train.y[train_mask])
    inner_pool = pool_from_rows(inner_mask, train.y[inner_mask])
    validation_pool = pool_from_rows(slice(None), None)
    model = CatBoostRegressor(
        loss_function=config.catboost_loss,
        eval_metric=config.catboost_loss,
        iterations=config.max_rounds,
        learning_rate=0.07,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        task_type="GPU" if config.use_gpu else "CPU",
        devices="0" if config.use_gpu else None,
        allow_writing_files=False,
        verbose=False,
        od_type="Iter",
        od_wait=config.early_stopping_rounds,
    )
    model.fit(
        fit_pool,
        eval_set=inner_pool,
        use_best_model=True,
        verbose=False,
    )
    return np.maximum(model.predict(validation_pool), 0).astype(np.float64)


MODEL_TRAINERS = {
    "lightgbm": train_lightgbm,
    "xgboost": train_xgboost,
    "catboost": train_catboost,
}


def optimize_weights(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    n_models = predictions.shape[1]
    initial = np.full(n_models, 1.0 / n_models)

    def objective(weights: np.ndarray) -> float:
        blended = predictions @ weights
        # Small regularizer prevents unstable all-or-nothing weights when
        # several experts are effectively tied on a small set of folds.
        regularizer = 1e-4 * float(((weights - initial) ** 2).sum())
        return competition_score(y_true, blended) + regularizer

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Weight optimization failed: {result.message}")
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def cross_fitted_ensemble(
    fold_ids: np.ndarray,
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray | None, dict[int, np.ndarray]]:
    unique_folds = np.unique(fold_ids)
    if len(unique_folds) < 2:
        return None, {}
    blended = np.empty_like(y_true, dtype=np.float64)
    weights_by_fold: dict[int, np.ndarray] = {}
    for fold in unique_folds:
        fit_mask = fold_ids != fold
        score_mask = fold_ids == fold
        weights = optimize_weights(y_true[fit_mask], predictions[fit_mask])
        blended[score_mask] = predictions[score_mask] @ weights
        weights_by_fold[int(fold)] = weights
    return blended, weights_by_fold


def _plot_results(metrics: pd.DataFrame, prediction_frame: pd.DataFrame, output_dir: Path) -> None:
    fold_metrics = metrics[metrics["fold"] != "all"].copy()
    if not fold_metrics.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        order = fold_metrics.groupby("model")["score"].mean().sort_values().index
        data = [fold_metrics.loc[fold_metrics["model"] == model, "score"].to_numpy() for model in order]
        ax.boxplot(data, tick_labels=order, showmeans=True)
        ax.set_ylabel("WAPE + absolute relative bias (lower is better)")
        ax.set_title("Test-like temporal backtest by model")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "01_model_scores.png", dpi=160)
        plt.close(fig)

    model_columns = [c for c in prediction_frame.columns if c.startswith("pred_")]
    if len(model_columns) >= 2:
        errors = prediction_frame[model_columns].sub(prediction_frame["y_true"], axis=0)
        corr = errors.corr()
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        labels = [c.removeprefix("pred_") for c in corr.columns]
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        for row in range(len(labels)):
            for col in range(len(labels)):
                ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("Correlation of validation errors\n(lower correlation adds ensemble diversity)")
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(output_dir / "02_error_correlations.png", dpi=160)
        plt.close(fig)


def run_backtest(config: RunConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    panel = load_panel(config.data_dir, config.max_routes)
    folds = make_folds(panel, config.folds, config.fold_spacing)
    metrics: list[dict[str, float | int | str]] = []
    fold_frames: list[pd.DataFrame] = []

    for fold in folds:
        print(f"Fold {fold.fold}: origin={fold.origin_timestamp}, routes={len(panel.route_ids)}", flush=True)
        y_true = panel.target[fold.origin_idx + 1 : fold.origin_idx + HORIZON + 1].reshape(-1).astype(np.float64)
        frame = pd.DataFrame(
            {
                "fold": fold.fold,
                "origin_timestamp": fold.origin_timestamp,
                "timestamp": np.repeat(pd.DatetimeIndex(fold.validation_timestamps), len(panel.route_ids)),
                "route_id": np.tile(panel.route_ids, HORIZON),
                "horizon": np.repeat(np.arange(1, HORIZON + 1), len(panel.route_ids)),
                "y_true": y_true,
            }
        )

        if "seasonal" in config.models:
            for model_name, prediction in seasonal_predictions(panel, fold.origin_idx).items():
                frame[f"pred_{model_name}"] = prediction
                metrics.append(metric_record(fold.fold, model_name, y_true, prediction))

        tree_models = [name for name in config.models if name in MODEL_TRAINERS]
        if tree_models or "tcn" in config.models:
            max_training_origin = fold.origin_idx - HORIZON
            origins = _candidate_origins(
                max_training_origin,
                config.lookback_days * STEPS_PER_DAY,
                config.origin_stride,
            )
            print(
                f"  training origins={len(origins)}, inner chronology preserved",
                flush=True,
            )
        if tree_models:
            train_bundle = build_matrix(panel, origins, include_labels=True)
            validation_bundle = build_matrix(panel, [fold.origin_idx], include_labels=True)
            print(f"  tree matrix={train_bundle.X.shape}", flush=True)
            for model_name in tree_models:
                model_started = time.perf_counter()
                prediction = MODEL_TRAINERS[model_name](train_bundle, validation_bundle, config)
                frame[f"pred_{model_name}"] = prediction
                metrics.append(metric_record(fold.fold, model_name, y_true, prediction))
                print(
                    f"  {model_name}: score={competition_score(y_true, prediction):.5f}, "
                    f"seconds={time.perf_counter() - model_started:.1f}",
                    flush=True,
                )
        if "tcn" in config.models:
            model_started = time.perf_counter()
            prediction = train_tcn(panel, origins, fold.origin_idx, config)
            frame["pred_tcn"] = prediction
            metrics.append(metric_record(fold.fold, "tcn", y_true, prediction))
            print(
                f"  tcn: score={competition_score(y_true, prediction):.5f}, "
                f"seconds={time.perf_counter() - model_started:.1f}",
                flush=True,
            )
        fold_frames.append(frame)

    prediction_frame = pd.concat(fold_frames, ignore_index=True)
    model_columns = [c for c in prediction_frame.columns if c.startswith("pred_")]
    model_names = [c.removeprefix("pred_") for c in model_columns]
    y_all = prediction_frame["y_true"].to_numpy()
    prediction_matrix = prediction_frame[model_columns].to_numpy()
    fold_ids = prediction_frame["fold"].to_numpy()

    full_weights = optimize_weights(y_all, prediction_matrix)
    prediction_frame["pred_ensemble_oof_fit"] = prediction_matrix @ full_weights
    metrics.append(metric_record("all", "ensemble_oof_fit", y_all, prediction_frame["pred_ensemble_oof_fit"].to_numpy()))

    crossfit_prediction, crossfit_weights = cross_fitted_ensemble(fold_ids, y_all, prediction_matrix)
    if crossfit_prediction is not None:
        prediction_frame["pred_ensemble_crossfit"] = crossfit_prediction
        metrics.append(metric_record("all", "ensemble_crossfit", y_all, crossfit_prediction))

    for model_name, column in zip(model_names, model_columns, strict=True):
        metrics.append(metric_record("all", model_name, y_all, prediction_frame[column].to_numpy()))

    metrics_frame = pd.DataFrame(metrics)
    weights_frame = pd.DataFrame({"model": model_names, "full_oof_weight": full_weights})
    for fold, weights in crossfit_weights.items():
        weights_frame[f"leave_fold_{fold}_out_weight"] = weights

    prediction_frame.to_parquet(config.output_dir / "oof_predictions.parquet", index=False)
    metrics_frame.to_csv(config.output_dir / "metrics.csv", index=False)
    weights_frame.to_csv(config.output_dir / "ensemble_weights.csv", index=False)
    prediction_frame[model_columns].sub(prediction_frame["y_true"], axis=0).corr().to_csv(
        config.output_dir / "error_correlations.csv"
    )
    _plot_results(metrics_frame, prediction_frame, config.output_dir)

    manifest = {
        "config": {**asdict(config), "data_dir": str(config.data_dir), "output_dir": str(config.output_dir)},
        "data": {
            "routes": len(panel.route_ids),
            "train_timestamps": len(panel.timestamps),
            "train_start": str(panel.timestamps[0]),
            "train_end": str(panel.timestamps[-1]),
            "forecast_horizon": HORIZON,
        },
        "folds": [
            {
                "fold": fold.fold,
                "origin": str(fold.origin_timestamp),
                "validation_start": str(fold.validation_timestamps[0]),
                "validation_end": str(fold.validation_timestamps[-1]),
            }
            for fold in folds
        ],
        "full_oof_weights": dict(zip(model_names, full_weights.tolist(), strict=True)),
        "crossfit_available": crossfit_prediction is not None,
        "elapsed_seconds": time.perf_counter() - started,
        "notes": [
            "Every validation fold forecasts the next eight half-hour slots for every route.",
            "Feature rows use only target/status history at or before the forecast origin.",
            "Early stopping uses an inner chronological tail, never the outer validation window.",
            "The all-OOF ensemble is weight-fitted on the same OOF rows and is optimistic.",
            "Use ensemble_crossfit for the less biased ensemble estimate when folds >= 2.",
        ],
    }
    (config.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nAll-fold metrics:")
    print(metrics_frame[metrics_frame["fold"] == "all"].sort_values("score").to_string(index=False))
    print(f"Artifacts: {config.output_dir}")


def parse_args(argv: Iterable[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ml_parallel_ensemble"))
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--fold-spacing", type=int, default=STEPS_PER_DAY)
    parser.add_argument("--lookback-days", type=int, default=42)
    parser.add_argument("--origin-stride", type=int, default=4)
    parser.add_argument("--max-routes", type=int)
    parser.add_argument("--models", default="seasonal,lightgbm,xgboost,catboost,tcn")
    parser.add_argument("--max-rounds", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--catboost-loss", choices=("MAE", "RMSE"), default="RMSE")
    parser.add_argument("--tcn-sequence-length", type=int, default=STEPS_PER_WEEK)
    parser.add_argument(
        "--tcn-architecture", choices=("single", "two_stream"), default="two_stream"
    )
    parser.add_argument("--tcn-channels", type=int, default=64)
    parser.add_argument("--tcn-epochs", type=int, default=12)
    parser.add_argument("--tcn-batch-size", type=int, default=256)
    parser.add_argument("--tcn-patience", type=int, default=3)
    parser.add_argument("--cpu", action="store_true", help="Disable GPU for XGBoost/CatBoost")
    parser.add_argument("--smoke", action="store_true", help="Small but real end-to-end run")
    args = parser.parse_args(argv)

    if args.smoke:
        args.output_dir = Path("artifacts/ml_parallel_ensemble_smoke")
        args.folds = 1
        args.lookback_days = 10
        args.origin_stride = 12
        args.max_routes = 100
        args.max_rounds = 40
        args.early_stopping_rounds = 8
        args.tcn_channels = 32
        args.tcn_epochs = 3
        args.tcn_batch_size = 128
        args.tcn_patience = 2

    models = tuple(part.strip() for part in args.models.split(",") if part.strip())
    unknown = set(models).difference({"seasonal", "tcn", *MODEL_TRAINERS})
    if unknown:
        parser.error(f"Unknown models: {sorted(unknown)}")
    if args.folds < 1:
        parser.error("--folds must be at least 1")
    if args.tcn_channels % 8:
        parser.error("--tcn-channels must be divisible by 8 for GroupNorm")

    return RunConfig(
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        folds=args.folds,
        fold_spacing=args.fold_spacing,
        lookback_days=args.lookback_days,
        origin_stride=args.origin_stride,
        max_routes=args.max_routes,
        models=models,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        catboost_loss=args.catboost_loss,
        use_gpu=not args.cpu,
        tcn_sequence_length=args.tcn_sequence_length,
        tcn_architecture=args.tcn_architecture,
        tcn_channels=args.tcn_channels,
        tcn_epochs=args.tcn_epochs,
        tcn_batch_size=args.tcn_batch_size,
        tcn_patience=args.tcn_patience,
    )


if __name__ == "__main__":
    run_backtest(parse_args())
