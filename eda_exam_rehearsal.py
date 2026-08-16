"""Reproducible raw-data audit for the Wildberries forecasting task.

This module is deliberately separate from the notebook.  Every table and plot
shown in ``eda_exam_rehearsal.ipynb`` is computed here from the parquet files.
The raw data is never modified in place.  The first-stage audit does not drop,
clip, impute, resample, or otherwise clean observations.

Run from the repository root:

    uv run python eda_exam_rehearsal.py
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from scipy.signal import periodogram
from scipy.stats import spearmanr, wasserstein_distance

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / "train_solo_track.parquet"
TEST_PATH = PROJECT_DIR / "test_solo_track.parquet"
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "eda_exam"
TABLE_DIR = ARTIFACT_DIR / "tables"
PLOT_DIR = ARTIFACT_DIR / "plots"

KEY_COLS = ["route_id", "timestamp"]
TARGET_COL = "target_1h"
STATUS_COLS = [f"status_{index}" for index in range(1, 7)]
QUANTILES = [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.figsize": (12, 5),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


@dataclass(frozen=True)
class DataBundle:
    """Raw train/test tables loaded from disk."""

    train: pd.DataFrame
    test: pd.DataFrame


def ensure_artifact_dirs() -> None:
    """Create directories used only for generated audit artifacts."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(paths: Iterable[Path] = (TRAIN_PATH, TEST_PATH)) -> pd.DataFrame:
    """Return reproducibility metadata without loading whole tables."""

    records: list[dict[str, object]] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        records.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "size_mib": path.stat().st_size / 2**20,
                "rows": metadata.num_rows,
                "columns": metadata.num_columns,
                "row_groups": metadata.num_row_groups,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(records)


def load_data() -> DataBundle:
    """Load raw parquet data and validate the minimum structural contract."""

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    expected_train = {*KEY_COLS, *STATUS_COLS, TARGET_COL}
    expected_test = {"id", *KEY_COLS}
    missing_train = expected_train.difference(train.columns)
    missing_test = expected_test.difference(test.columns)
    if missing_train or missing_test:
        raise ValueError(
            f"Schema mismatch: missing_train={sorted(missing_train)}, "
            f"missing_test={sorted(missing_test)}"
        )
    if not pd.api.types.is_datetime64_any_dtype(train["timestamp"]):
        train = train.assign(timestamp=pd.to_datetime(train["timestamp"], errors="raise"))
    if not pd.api.types.is_datetime64_any_dtype(test["timestamp"]):
        test = test.assign(timestamp=pd.to_datetime(test["timestamp"], errors="raise"))

    return DataBundle(train=train, test=test)


def table_overview(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Column-level audit: types, missingness, cardinality, and basic ranges."""

    rows: list[dict[str, object]] = []
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        row: dict[str, object] = {
            "table": table_name,
            "column": column,
            "dtype": str(series.dtype),
            "rows": len(series),
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "unique_including_na": int(series.nunique(dropna=False)),
        }
        if pd.api.types.is_numeric_dtype(series):
            row.update(
                {
                    "zero_count": int((series == 0).sum()),
                    "negative_count": int((series < 0).sum()),
                    "min": non_null.min() if len(non_null) else np.nan,
                    "median": non_null.median() if len(non_null) else np.nan,
                    "max": non_null.max() if len(non_null) else np.nan,
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            row.update(
                {
                    "min": non_null.min() if len(non_null) else pd.NaT,
                    "max": non_null.max() if len(non_null) else pd.NaT,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def key_integrity_report(bundle: DataBundle) -> pd.DataFrame:
    """Check exact duplicates, candidate keys, IDs, and route overlap."""

    train, test = bundle.train, bundle.test
    train_duplicate_key_mask = train.duplicated(KEY_COLS, keep=False)
    test_duplicate_key_mask = test.duplicated(KEY_COLS, keep=False)
    train_routes = set(train["route_id"].unique())
    test_routes = set(test["route_id"].unique())

    checks = [
        ("train_rows", len(train)),
        ("test_rows", len(test)),
        ("train_exact_duplicate_rows", int(train.duplicated(keep=False).sum())),
        ("test_exact_duplicate_rows", int(test.duplicated(keep=False).sum())),
        ("train_duplicate_key_rows", int(train_duplicate_key_mask.sum())),
        ("test_duplicate_key_rows", int(test_duplicate_key_mask.sum())),
        ("train_duplicate_key_values", int(train.loc[train_duplicate_key_mask, KEY_COLS].drop_duplicates().shape[0])),
        ("test_duplicate_key_values", int(test.loc[test_duplicate_key_mask, KEY_COLS].drop_duplicates().shape[0])),
        ("test_duplicate_id_rows", int(test.duplicated(["id"], keep=False).sum())),
        ("train_route_count", len(train_routes)),
        ("test_route_count", len(test_routes)),
        ("routes_in_both", len(train_routes & test_routes)),
        ("test_only_routes", len(test_routes - train_routes)),
        ("train_only_routes", len(train_routes - test_routes)),
    ]
    return pd.DataFrame(checks, columns=["check", "value"])


def numeric_distribution_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Exact quantiles and shape statistics for numeric columns."""

    records: list[dict[str, object]] = []
    for column in columns:
        series = df[column].dropna()
        quantile_values = series.quantile(QUANTILES)
        record: dict[str, object] = {
            "column": column,
            "count": int(series.size),
            "missing": int(df[column].isna().sum()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "skew": float(series.skew()),
            "kurtosis": float(series.kurt()),
            "zeros": int((series == 0).sum()),
            "zero_pct": float((series == 0).mean() * 100),
            "negatives": int((series < 0).sum()),
        }
        for quantile, value in quantile_values.items():
            label = f"p{quantile * 100:g}".replace(".", "_")
            record[label] = float(value)
        records.append(record)
    return pd.DataFrame(records)


def target_diagnostics(train: pd.DataFrame) -> pd.DataFrame:
    """Target diagnostics without changing or trimming the target."""

    target = train[TARGET_COL]
    q1, q3 = target.quantile([0.25, 0.75])
    iqr = q3 - q1
    tukey_upper = q3 + 1.5 * iqr
    total = float(target.sum())
    sorted_values = np.sort(target.to_numpy())
    top_one_start = int(np.floor(0.99 * len(sorted_values)))
    top_one_share = float(sorted_values[top_one_start:].sum() / total) if total else np.nan

    return pd.DataFrame(
        {
            "metric": [
                "row_count",
                "total_target",
                "mean",
                "median",
                "std",
                "minimum",
                "maximum",
                "zero_rows",
                "zero_pct",
                "negative_rows",
                "tukey_upper_reference",
                "rows_above_tukey_reference",
                "top_1pct_rows_share_of_total_target",
            ],
            "value": [
                len(target),
                total,
                float(target.mean()),
                float(target.median()),
                float(target.std()),
                float(target.min()),
                float(target.max()),
                int((target == 0).sum()),
                float((target == 0).mean() * 100),
                int((target < 0).sum()),
                float(tukey_upper),
                int((target > tukey_upper).sum()),
                top_one_share,
            ],
        }
    )


def infer_modal_step_minutes(train: pd.DataFrame) -> float:
    """Infer the dominant positive interval from observations within routes."""

    ordered = train[KEY_COLS].sort_values(KEY_COLS)
    deltas = ordered.groupby("route_id", sort=False)["timestamp"].diff()
    positive_minutes = deltas[deltas > pd.Timedelta(0)].dt.total_seconds().div(60)
    if positive_minutes.empty:
        raise ValueError("Cannot infer a positive time interval")
    return float(positive_minutes.mode().iloc[0])


def time_grid_audit(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit ordering, interval regularity, and missing slots per route."""

    unique_keys = train[KEY_COLS].drop_duplicates().sort_values(KEY_COLS)
    modal_step_minutes = infer_modal_step_minutes(train)
    modal_step = pd.Timedelta(minutes=modal_step_minutes)

    deltas = unique_keys.groupby("route_id", sort=False)["timestamp"].diff()
    delta_counts = (
        deltas.dropna()
        .value_counts()
        .rename_axis("delta")
        .reset_index(name="transition_count")
        .sort_values(["transition_count", "delta"], ascending=[False, True])
        .reset_index(drop=True)
    )
    delta_counts["delta_minutes"] = delta_counts["delta"].dt.total_seconds() / 60

    route_grid = unique_keys.groupby("route_id")["timestamp"].agg(
        first_timestamp="min",
        last_timestamp="max",
        observed_unique_slots="size",
    )
    span_steps = (route_grid["last_timestamp"] - route_grid["first_timestamp"]) / modal_step
    route_grid["span_is_integer_steps"] = np.isclose(span_steps, np.round(span_steps))
    route_grid["expected_slots_between_own_bounds"] = np.round(span_steps).astype("int64") + 1
    route_grid["missing_slots_between_own_bounds"] = (
        route_grid["expected_slots_between_own_bounds"] - route_grid["observed_unique_slots"]
    )
    route_grid["coverage_pct_between_own_bounds"] = (
        route_grid["observed_unique_slots"]
        / route_grid["expected_slots_between_own_bounds"]
        * 100
    )
    route_grid = route_grid.reset_index()

    overview = pd.DataFrame(
        {
            "metric": [
                "modal_positive_step_minutes",
                "train_min_timestamp",
                "train_max_timestamp",
                "unique_routes",
                "unique_route_timestamp_keys",
                "non_modal_positive_transitions",
                "routes_with_missing_slots_between_own_bounds",
                "total_missing_slots_between_own_bounds",
                "routes_with_non_integer_span",
            ],
            "value": [
                modal_step_minutes,
                train["timestamp"].min(),
                train["timestamp"].max(),
                train["route_id"].nunique(),
                len(unique_keys),
                int((deltas.notna() & (deltas != modal_step)).sum()),
                int((route_grid["missing_slots_between_own_bounds"] > 0).sum()),
                int(route_grid["missing_slots_between_own_bounds"].sum()),
                int((~route_grid["span_is_integer_steps"]).sum()),
            ],
        }
    )
    return overview, delta_counts, route_grid


def test_horizon_audit(bundle: DataBundle, step_minutes: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify how test timestamps relate to each route's latest train row."""

    train, test = bundle.train, bundle.test
    last_train = train.groupby("route_id", as_index=False)["timestamp"].max().rename(
        columns={"timestamp": "last_train_timestamp"}
    )
    merged = test.merge(last_train, on="route_id", how="left", validate="many_to_one")
    merged["minutes_after_last_train"] = (
        merged["timestamp"] - merged["last_train_timestamp"]
    ).dt.total_seconds() / 60
    merged["horizon_steps"] = merged["minutes_after_last_train"] / step_minutes
    merged["horizon_is_integer"] = np.isclose(
        merged["horizon_steps"], np.round(merged["horizon_steps"]), equal_nan=False
    )

    rows_per_route = test.groupby("route_id").size()
    horizon_counts = (
        merged.groupby("horizon_steps", dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("horizon_steps")
    )
    overview = pd.DataFrame(
        {
            "metric": [
                "test_min_timestamp",
                "test_max_timestamp",
                "test_rows",
                "test_routes",
                "rows_without_train_route",
                "minimum_rows_per_route",
                "maximum_rows_per_route",
                "unique_rows_per_route_values",
                "minimum_horizon_steps",
                "maximum_horizon_steps",
                "non_integer_horizon_rows",
                "test_duplicate_ids",
                "test_duplicate_route_timestamp_rows",
            ],
            "value": [
                test["timestamp"].min(),
                test["timestamp"].max(),
                len(test),
                test["route_id"].nunique(),
                int(merged["last_train_timestamp"].isna().sum()),
                int(rows_per_route.min()),
                int(rows_per_route.max()),
                sorted(rows_per_route.unique().tolist()),
                float(merged["horizon_steps"].min()),
                float(merged["horizon_steps"].max()),
                int((~merged["horizon_is_integer"]).sum()),
                int(test.duplicated("id", keep=False).sum()),
                int(test.duplicated(KEY_COLS, keep=False).sum()),
            ],
        }
    )
    return overview, horizon_counts


def route_statistics(train: pd.DataFrame) -> pd.DataFrame:
    """Describe heterogeneous target volume and availability by route."""

    route = train.groupby("route_id").agg(
        rows=(TARGET_COL, "size"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        target_sum=(TARGET_COL, "sum"),
        target_mean=(TARGET_COL, "mean"),
        target_median=(TARGET_COL, "median"),
        target_std=(TARGET_COL, "std"),
        target_max=(TARGET_COL, "max"),
        zero_target_rows=(TARGET_COL, lambda values: int((values == 0).sum())),
    )
    route["zero_target_pct"] = route["zero_target_rows"] / route["rows"] * 100
    route["target_share"] = route["target_sum"] / route["target_sum"].sum()
    return route.reset_index()


def route_distribution_summary(routes: pd.DataFrame) -> pd.DataFrame:
    """Quantiles across routes; each route has equal weight in this table."""

    columns = [
        "rows",
        "target_sum",
        "target_mean",
        "target_median",
        "target_std",
        "target_max",
        "zero_target_pct",
        "target_share",
    ]
    summary = routes[columns].quantile(QUANTILES).T
    summary.columns = [f"p{q * 100:g}".replace(".", "_") for q in QUANTILES]
    return summary.reset_index(names="route_metric")


def temporal_profiles(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate target by calendar dimensions without resampling raw rows."""

    time = train["timestamp"]
    frame = train[[TARGET_COL]].copy()
    frame["date"] = time.dt.floor("D")
    frame["hour"] = time.dt.hour
    frame["minute"] = time.dt.minute
    frame["slot_of_day"] = frame["hour"] * 2 + (frame["minute"] >= 30).astype(int)
    frame["day_of_week"] = time.dt.dayofweek
    frame["day_name"] = time.dt.day_name()

    daily = frame.groupby("date").agg(
        rows=(TARGET_COL, "size"),
        observed_slots=("slot_of_day", "nunique"),
        total=(TARGET_COL, "sum"),
        mean=(TARGET_COL, "mean"),
        median=(TARGET_COL, "median"),
    ).reset_index()
    daily["slot_coverage_pct"] = daily["observed_slots"] / 48 * 100
    daily["complete_48_slot_day"] = daily["observed_slots"] == 48
    slot = frame.groupby("slot_of_day")[TARGET_COL].agg(rows="size", total="sum", mean="mean", median="median").reset_index()
    weekday = frame.groupby(["day_of_week", "day_name"])[TARGET_COL].agg(
        rows="size", total="sum", mean="mean", median="median"
    ).reset_index().sort_values("day_of_week")
    return {"daily": daily, "slot": slot, "weekday": weekday}


def correlation_report(train: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlations as descriptive associations, not causal effects."""

    columns = [*STATUS_COLS, TARGET_COL]
    corr = train[columns].corr(method="pearson")
    return corr


def target_autocorrelation_report(train: pd.DataFrame) -> pd.DataFrame:
    """Within-route target autocorrelation at predeclared operational lags."""

    ordered = train[["route_id", "timestamp", TARGET_COL]].sort_values(KEY_COLS)
    grouped = ordered.groupby("route_id", sort=False)[TARGET_COL]
    records: list[dict[str, float | int]] = []
    for lag in [1, 2, 3, 4, 6, 12, 24, 48, 96, 336]:
        lagged = grouped.shift(lag)
        valid = ordered[TARGET_COL].notna() & lagged.notna()
        records.append(
            {
                "lag_steps": lag,
                "pair_count": int(valid.sum()),
                "pearson_correlation": float(ordered.loc[valid, TARGET_COL].corr(lagged.loc[valid])),
            }
        )
    return pd.DataFrame(records)


def zero_pattern_audit(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Locate zeros by route/time and measure consecutive zero runs."""

    ordered = train[["route_id", "timestamp", TARGET_COL]].sort_values(KEY_COLS).reset_index(drop=True)
    ordered["is_zero"] = ordered[TARGET_COL].eq(0)
    ordered["date"] = ordered["timestamp"].dt.floor("D")
    ordered["slot_of_day"] = ordered["timestamp"].dt.hour * 2 + (
        ordered["timestamp"].dt.minute >= 30
    ).astype(int)

    route_zero = ordered.groupby("route_id").agg(
        rows=(TARGET_COL, "size"),
        zero_rows=("is_zero", "sum"),
        target_mean=(TARGET_COL, "mean"),
        target_median=(TARGET_COL, "median"),
    ).reset_index()
    route_zero["zero_pct"] = route_zero["zero_rows"] / route_zero["rows"] * 100

    zero_by_slot = ordered.groupby("slot_of_day").agg(
        rows=(TARGET_COL, "size"), zero_rows=("is_zero", "sum")
    ).reset_index()
    zero_by_slot["zero_pct"] = zero_by_slot["zero_rows"] / zero_by_slot["rows"] * 100

    zero_by_date = ordered.groupby("date").agg(
        rows=(TARGET_COL, "size"), zero_rows=("is_zero", "sum")
    ).reset_index()
    zero_by_date["zero_pct"] = zero_by_date["zero_rows"] / zero_by_date["rows"] * 100

    run_id = (~ordered["is_zero"]).groupby(ordered["route_id"]).cumsum()
    zero_rows = ordered.loc[ordered["is_zero"], ["route_id", "timestamp"]].copy()
    zero_rows["run_id"] = run_id.loc[ordered["is_zero"]].to_numpy()
    zero_runs = zero_rows.groupby(["route_id", "run_id"]).agg(
        run_start=("timestamp", "min"),
        run_end=("timestamp", "max"),
        run_steps=("timestamp", "size"),
    ).reset_index()
    zero_runs["run_hours"] = zero_runs["run_steps"] * infer_modal_step_minutes(train) / 60
    zero_runs = zero_runs.sort_values(["run_steps", "route_id"], ascending=[False, True])

    overview = pd.DataFrame(
        {
            "metric": [
                "zero_rows",
                "zero_pct",
                "routes_with_at_least_one_zero",
                "routes_without_zeros",
                "median_route_zero_pct",
                "maximum_route_zero_pct",
                "zero_run_count",
                "median_zero_run_steps",
                "maximum_zero_run_steps",
            ],
            "value": [
                int(ordered["is_zero"].sum()),
                float(ordered["is_zero"].mean() * 100),
                int((route_zero["zero_rows"] > 0).sum()),
                int((route_zero["zero_rows"] == 0).sum()),
                float(route_zero["zero_pct"].median()),
                float(route_zero["zero_pct"].max()),
                len(zero_runs),
                float(zero_runs["run_steps"].median()) if len(zero_runs) else 0,
                int(zero_runs["run_steps"].max()) if len(zero_runs) else 0,
            ],
        }
    )
    return overview, route_zero, zero_by_slot, zero_by_date, zero_runs


def tail_event_audit(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe the upper tail without labeling it as error or removing it."""

    target = train[TARGET_COL]
    q1, q3 = target.quantile([0.25, 0.75])
    thresholds = {
        "p99": float(target.quantile(0.99)),
        "p99_9": float(target.quantile(0.999)),
        "tukey_upper": float(q3 + 1.5 * (q3 - q1)),
    }
    records = []
    for name, threshold in thresholds.items():
        mask = target > threshold
        records.append(
            {
                "threshold_name": name,
                "threshold": threshold,
                "rows_strictly_above": int(mask.sum()),
                "row_pct": float(mask.mean() * 100),
                "target_share": float(target[mask].sum() / target.sum()),
                "routes_represented": int(train.loc[mask, "route_id"].nunique()),
                "timestamps_represented": int(train.loc[mask, "timestamp"].nunique()),
            }
        )
    overview = pd.DataFrame(records)

    p999_mask = target > thresholds["p99_9"]
    tail = train.loc[p999_mask, ["route_id", "timestamp", *STATUS_COLS, TARGET_COL]].copy()
    route_median = train.groupby("route_id")[TARGET_COL].median()
    tail["route_median"] = tail["route_id"].map(route_median)
    tail["multiple_of_route_median"] = tail[TARGET_COL] / tail["route_median"].replace(0, np.nan)
    top_events = tail.sort_values(TARGET_COL, ascending=False).head(50).reset_index(drop=True)

    tail_by_route = tail.groupby("route_id").agg(
        tail_rows=(TARGET_COL, "size"),
        tail_target_sum=(TARGET_COL, "sum"),
        tail_target_max=(TARGET_COL, "max"),
    ).reset_index().sort_values(["tail_rows", "tail_target_sum"], ascending=False)
    tail_by_date = tail.assign(date=tail["timestamp"].dt.floor("D")).groupby("date").agg(
        tail_rows=(TARGET_COL, "size"),
        tail_target_sum=(TARGET_COL, "sum"),
    ).reset_index()
    return overview, top_events, tail_by_route, tail_by_date


def status_lead_lag_report(train: pd.DataFrame, max_horizon: int = 8) -> pd.DataFrame:
    """Measure raw and route-centered status associations with future target."""

    ordered = train[["route_id", "timestamp", *STATUS_COLS, TARGET_COL]].sort_values(KEY_COLS)
    grouped_target = ordered.groupby("route_id", sort=False)[TARGET_COL]
    route_target_mean = ordered.groupby("route_id")[TARGET_COL].transform("mean")
    centered_status = ordered[STATUS_COLS] - ordered.groupby("route_id")[STATUS_COLS].transform("mean")
    records: list[dict[str, object]] = []
    for horizon in range(max_horizon + 1):
        future_target = grouped_target.shift(-horizon)
        centered_future = future_target - route_target_mean
        valid_target = future_target.notna()
        for status in STATUS_COLS:
            valid = valid_target & ordered[status].notna()
            records.append(
                {
                    "horizon_steps": horizon,
                    "horizon_hours": horizon * 0.5,
                    "status": status,
                    "pair_count": int(valid.sum()),
                    "raw_pearson": float(ordered.loc[valid, status].corr(future_target.loc[valid])),
                    "route_centered_pearson": float(
                        centered_status.loc[valid, status].corr(centered_future.loc[valid])
                    ),
                }
            )
    return pd.DataFrame(records)


def temporal_stability_audit(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare equal chronological quarters and weekly aggregate behavior."""

    frame = train[["route_id", "timestamp", TARGET_COL]].copy()
    min_ts, max_ts = frame["timestamp"].min(), frame["timestamp"].max()
    boundaries = pd.date_range(min_ts, max_ts, periods=5)
    labels = ["Q1_earliest", "Q2", "Q3", "Q4_latest"]
    frame["time_quarter"] = pd.cut(
        frame["timestamp"], bins=boundaries, labels=labels, include_lowest=True
    )
    quarter_summary = frame.groupby("time_quarter", observed=True)[TARGET_COL].agg(
        rows="size",
        mean="mean",
        median="median",
        std="std",
        minimum="min",
        maximum="max",
    ).reset_index()
    quarter_summary["zero_pct"] = frame.groupby("time_quarter", observed=True)[TARGET_COL].apply(
        lambda values: (values == 0).mean() * 100
    ).to_numpy()
    quarter_summary["p99"] = frame.groupby("time_quarter", observed=True)[TARGET_COL].quantile(0.99).to_numpy()

    earliest = frame.loc[frame["time_quarter"] == labels[0], TARGET_COL].to_numpy()
    latest = frame.loc[frame["time_quarter"] == labels[-1], TARGET_COL].to_numpy()
    route_quarter = frame.groupby(["route_id", "time_quarter"], observed=True)[TARGET_COL].mean().unstack()
    valid_routes = route_quarter[[labels[0], labels[-1]]].dropna()
    rank_corr = spearmanr(valid_routes[labels[0]], valid_routes[labels[-1]])
    overview = pd.DataFrame(
        {
            "metric": [
                "history_start",
                "history_end",
                "latest_vs_earliest_mean_ratio",
                "latest_vs_earliest_median_ratio",
                "wasserstein_distance_earliest_latest",
                "wasserstein_normalized_by_global_std",
                "route_mean_rank_spearman_earliest_latest",
                "route_mean_rank_spearman_pvalue",
            ],
            "value": [
                min_ts,
                max_ts,
                float(quarter_summary.iloc[-1]["mean"] / quarter_summary.iloc[0]["mean"]),
                float(quarter_summary.iloc[-1]["median"] / quarter_summary.iloc[0]["median"]),
                float(wasserstein_distance(earliest, latest)),
                float(wasserstein_distance(earliest, latest) / frame[TARGET_COL].std()),
                float(rank_corr.statistic),
                float(rank_corr.pvalue),
            ],
        }
    )

    frame["week_start"] = frame["timestamp"].dt.to_period("W-SUN").dt.start_time
    weekly = frame.groupby("week_start")[TARGET_COL].agg(
        rows="size", total="sum", mean="mean", median="median", std="std"
    ).reset_index()
    weekly["zero_pct"] = frame.groupby("week_start")[TARGET_COL].apply(
        lambda values: (values == 0).mean() * 100
    ).to_numpy()
    weekly["p99"] = frame.groupby("week_start")[TARGET_COL].quantile(0.99).to_numpy()
    return overview, quarter_summary, route_quarter.reset_index(), weekly


def aggregate_periodogram(train: pd.DataFrame) -> pd.DataFrame:
    """Periodogram of aggregate target after removing a linear trend."""

    aggregate = train.groupby("timestamp")[TARGET_COL].sum().sort_index()
    values = aggregate.to_numpy(dtype=float)
    positions = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(positions, values, deg=1)
    detrended = values - (slope * positions + intercept)
    frequencies, powers = periodogram(detrended)
    positive = frequencies > 0
    frequencies = frequencies[positive]
    powers = powers[positive]
    periods_hours = 0.5 / frequencies
    spectrum = pd.DataFrame(
        {
            "frequency_cycles_per_step": frequencies,
            "period_steps": 1 / frequencies,
            "period_hours": periods_hours,
            "power": powers,
        }
    )
    spectrum["power_share"] = spectrum["power"] / spectrum["power"].sum()
    return spectrum.sort_values("power", ascending=False).reset_index(drop=True)


def route_segment_audit(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe two alternative route segmentations without selecting either."""

    routes = route_statistics(train)
    routes["coefficient_of_variation"] = routes["target_std"] / routes["target_mean"].replace(0, np.nan)
    routes["volume_quintile"] = pd.qcut(
        routes["target_mean"], q=5, labels=["V1_low", "V2", "V3", "V4", "V5_high"]
    )
    routes["variability_quintile"] = pd.qcut(
        routes["coefficient_of_variation"],
        q=5,
        labels=["C1_stable", "C2", "C3", "C4", "C5_variable"],
    )
    segment_summary = routes.groupby(
        ["volume_quintile", "variability_quintile"], observed=True
    ).agg(
        route_count=("route_id", "size"),
        median_target_mean=("target_mean", "median"),
        median_cv=("coefficient_of_variation", "median"),
        median_zero_pct=("zero_target_pct", "median"),
        total_target_share=("target_share", "sum"),
    ).reset_index()
    return routes, segment_summary


def save_figure(fig: plt.Figure, filename: str) -> Path:
    ensure_artifact_dirs()
    path = PLOT_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_missingness(bundle: DataBundle) -> tuple[plt.Figure, Path]:
    frames = []
    for table_name, frame in [("train", bundle.train), ("test", bundle.test)]:
        frames.append(
            pd.DataFrame(
                {
                    "table": table_name,
                    "column": frame.columns,
                    "missing_pct": frame.isna().mean().to_numpy() * 100,
                }
            )
        )
    missing = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=missing, x="column", y="missing_pct", hue="table", ax=ax)
    ax.set(title="Missing values in raw tables", xlabel="Column", ylabel="Missing rows, %")
    ax.tick_params(axis="x", rotation=45)
    path = save_figure(fig, "01_missingness.png")
    return fig, path


def plot_target_distribution(train: pd.DataFrame) -> tuple[plt.Figure, Path]:
    target = train[TARGET_COL].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(target, bins=120, color="steelblue", alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set(title="Raw target (full range)", xlabel=TARGET_COL, ylabel="Rows, log scale")

    axes[1].hist(np.log1p(target), bins=120, color="darkorange", alpha=0.85)
    axes[1].set(title="log1p view (raw rows retained)", xlabel=f"log1p({TARGET_COL})", ylabel="Rows")

    quantile_grid = np.linspace(0, 1, 1001)
    quantile_values = np.quantile(target, quantile_grid)
    axes[2].plot(quantile_grid * 100, quantile_values, color="purple")
    axes[2].set_yscale("symlog", linthresh=1)
    axes[2].set(title="Exact empirical quantile curve", xlabel="Percentile", ylabel=TARGET_COL)

    fig.suptitle(f"Target distribution; n={len(target):,}; no clipping or row removal")
    fig.tight_layout()
    path = save_figure(fig, "02_target_distribution.png")
    return fig, path


def plot_status_distributions(train: pd.DataFrame) -> tuple[plt.Figure, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for column, ax in zip(STATUS_COLS, axes.flat, strict=True):
        values = train[column].to_numpy()
        ax.hist(np.log1p(values), bins=100, alpha=0.85)
        ax.set(title=column, xlabel=f"log1p({column})", ylabel="Rows")
    fig.suptitle(f"Status distributions; n={len(train):,}; log1p is display-only")
    fig.tight_layout()
    path = save_figure(fig, "03_status_distributions.png")
    return fig, path


def plot_route_heterogeneity(routes: pd.DataFrame) -> tuple[plt.Figure, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(np.log1p(routes["target_sum"]), bins=60)
    axes[0].set(title="Total target per route", xlabel="log1p(target sum)", ylabel="Routes")
    axes[1].hist(np.log1p(routes["target_mean"]), bins=60)
    axes[1].set(title="Mean target per route", xlabel="log1p(target mean)", ylabel="Routes")
    axes[2].hist(routes["zero_target_pct"], bins=60)
    axes[2].set(title="Zero frequency per route", xlabel="Zero target rows, %", ylabel="Routes")
    fig.suptitle(f"Route heterogeneity; each of {len(routes):,} routes has equal plot weight")
    fig.tight_layout()
    path = save_figure(fig, "04_route_heterogeneity.png")
    return fig, path


def plot_temporal_profiles(profiles: dict[str, pd.DataFrame]) -> tuple[plt.Figure, Path]:
    daily, slot, weekday = profiles["daily"], profiles["slot"], profiles["weekday"]
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    axes[0].plot(daily["date"], daily["total"], linewidth=1, color="steelblue")
    incomplete = daily[~daily["complete_48_slot_day"]]
    axes[0].scatter(
        incomplete["date"],
        incomplete["total"],
        color="crimson",
        marker="X",
        s=80,
        label="Incomplete calendar day",
        zorder=3,
    )
    for row in incomplete.itertuples(index=False):
        axes[0].annotate(
            f"{row.observed_slots}/48 slots",
            (row.date, row.total),
            xytext=(5, 5),
            textcoords="offset points",
            color="crimson",
        )
    if len(incomplete):
        axes[0].legend()
    axes[0].set(
        title="Total target by calendar day (partial days retained and marked)",
        xlabel="Date",
        ylabel="Target sum",
    )
    axes[1].plot(slot["slot_of_day"], slot["mean"], marker="o", markersize=3)
    axes[1].set(title="Mean target by 30-minute slot", xlabel="Slot (0=00:00)", ylabel="Mean target")
    axes[2].bar(weekday["day_name"], weekday["mean"])
    axes[2].set(title="Mean target by weekday", xlabel="Weekday", ylabel="Mean target")
    axes[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = save_figure(fig, "05_temporal_profiles.png")
    return fig, path


def plot_correlations(correlation: pd.DataFrame) -> tuple[plt.Figure, Path]:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Pearson associations on raw train rows (not causal effects)")
    fig.tight_layout()
    path = save_figure(fig, "06_raw_pearson_correlations.png")
    return fig, path


def plot_test_horizons(horizon_counts: pd.DataFrame) -> tuple[plt.Figure, Path]:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=horizon_counts, x="horizon_steps", y="rows", color="steelblue", ax=ax)
    ax.set(title="Test rows by forecast horizon", xlabel="Steps after last train timestamp", ylabel="Rows")
    fig.tight_layout()
    path = save_figure(fig, "07_test_horizons.png")
    return fig, path


def plot_route_archetypes(train: pd.DataFrame, routes: pd.DataFrame) -> tuple[plt.Figure, Path, pd.DataFrame]:
    """Plot deterministic volume archetypes selected by route-sum ranks."""

    ordered_routes = routes.sort_values(["target_sum", "route_id"]).reset_index(drop=True)
    requested_quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    positions = [round(q * (len(ordered_routes) - 1)) for q in requested_quantiles]
    selected = ordered_routes.iloc[positions].copy()
    selected["selection_quantile"] = requested_quantiles

    global_max = train["timestamp"].max()
    display_start = global_max - pd.Timedelta(days=14)
    subset = train[
        train["route_id"].isin(selected["route_id"])
        & (train["timestamp"] >= display_start)
    ]

    fig, axes = plt.subplots(len(selected), 1, figsize=(18, 14), sharex=True)
    for ax, row in zip(axes, selected.itertuples(index=False), strict=True):
        route_rows = subset[subset["route_id"] == row.route_id].sort_values("timestamp")
        ax.plot(route_rows["timestamp"], route_rows[TARGET_COL], linewidth=0.7)
        ax.set_ylabel(TARGET_COL)
        ax.set_title(
            f"route_id={row.route_id}; target_sum rank quantile={row.selection_quantile:.2f}; "
            f"full-history sum={row.target_sum:,.0f}"
        )
    axes[-1].set_xlabel("Timestamp; fixed display window = final 14 calendar days")
    fig.suptitle("Deterministically selected route-volume archetypes")
    fig.tight_layout()
    path = save_figure(fig, "08_route_archetypes_last_14_days.png")
    return fig, path, selected


def plot_zero_patterns(
    route_zero: pd.DataFrame, zero_by_slot: pd.DataFrame, zero_by_date: pd.DataFrame
) -> tuple[plt.Figure, Path]:
    fig, axes = plt.subplots(3, 1, figsize=(16, 13))
    axes[0].hist(route_zero["zero_pct"], bins=60)
    axes[0].set(title="Zero-target frequency across routes", xlabel="Zero rows per route, %", ylabel="Routes")
    axes[1].plot(zero_by_slot["slot_of_day"], zero_by_slot["zero_pct"], marker="o", markersize=3)
    axes[1].set(title="Zero-target frequency by slot", xlabel="30-minute slot", ylabel="Zero rows, %")
    axes[2].plot(zero_by_date["date"], zero_by_date["zero_pct"])
    axes[2].set(title="Zero-target frequency by date", xlabel="Date", ylabel="Zero rows, %")
    fig.tight_layout()
    path = save_figure(fig, "09_zero_patterns.png")
    return fig, path


def plot_tail_patterns(
    overview: pd.DataFrame, tail_by_route: pd.DataFrame, tail_by_date: pd.DataFrame
) -> tuple[plt.Figure, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    axes[0].bar(overview["threshold_name"], overview["row_pct"])
    axes[0].set(title="Rows above reference thresholds", xlabel="Reference", ylabel="Rows, %")
    axes[1].hist(tail_by_route["tail_rows"], bins=50)
    axes[1].set(title="p99.9 tail rows per represented route", xlabel="Tail rows", ylabel="Routes")
    axes[2].plot(tail_by_date["date"], tail_by_date["tail_rows"])
    axes[2].set(title="p99.9 tail rows by date", xlabel="Date", ylabel="Tail rows")
    fig.tight_layout()
    path = save_figure(fig, "10_upper_tail_patterns.png")
    return fig, path


def plot_status_lead_lag(report: pd.DataFrame) -> tuple[plt.Figure, Path]:
    raw = report.pivot(index="status", columns="horizon_steps", values="raw_pearson")
    centered = report.pivot(index="status", columns="horizon_steps", values="route_centered_pearson")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.heatmap(raw, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=axes[0])
    axes[0].set_title("Raw correlation: status(t) vs target(t+h)")
    sns.heatmap(centered, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=axes[1])
    axes[1].set_title("Route-centered correlation: status(t) vs target(t+h)")
    fig.tight_layout()
    path = save_figure(fig, "11_status_lead_lag.png")
    return fig, path


def plot_temporal_stability(
    weekly: pd.DataFrame, spectrum: pd.DataFrame
) -> tuple[plt.Figure, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axes[0, 0].plot(weekly["week_start"], weekly["mean"], marker="o")
    axes[0, 0].set(title="Weekly target mean", xlabel="Week start", ylabel="Mean")
    axes[0, 1].plot(weekly["week_start"], weekly["zero_pct"], marker="o")
    axes[0, 1].set(title="Weekly zero frequency", xlabel="Week start", ylabel="Zero rows, %")
    axes[1, 0].plot(weekly["week_start"], weekly["p99"], marker="o")
    axes[1, 0].set(title="Weekly target p99", xlabel="Week start", ylabel="p99")
    visible = spectrum[(spectrum["period_hours"] >= 1) & (spectrum["period_hours"] <= 500)].sort_values("period_hours")
    axes[1, 1].plot(visible["period_hours"], visible["power_share"])
    axes[1, 1].set_xscale("log")
    axes[1, 1].set(title="Aggregate-target periodogram", xlabel="Period, hours (log scale)", ylabel="Power share")
    fig.tight_layout()
    path = save_figure(fig, "12_temporal_stability_and_spectrum.png")
    return fig, path


def plot_route_segments(segment_summary: pd.DataFrame) -> tuple[plt.Figure, Path]:
    count_matrix = segment_summary.pivot(
        index="variability_quintile", columns="volume_quintile", values="route_count"
    )
    share_matrix = segment_summary.pivot(
        index="variability_quintile", columns="volume_quintile", values="total_target_share"
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(count_matrix, annot=True, fmt=".0f", cmap="Blues", ax=axes[0])
    axes[0].set_title("Route count: volume × variability quintiles")
    sns.heatmap(share_matrix, annot=True, fmt=".3f", cmap="Greens", ax=axes[1])
    axes[1].set_title("Total target share: volume × variability quintiles")
    fig.tight_layout()
    path = save_figure(fig, "13_route_segmentations.png")
    return fig, path


def validation_assertions(
    bundle: DataBundle,
    key_report: pd.DataFrame,
    time_overview: pd.DataFrame,
    horizon_overview: pd.DataFrame,
) -> pd.DataFrame:
    """Machine-checkable structural assertions with explicit pass/fail output."""

    key_values = key_report.set_index("check")["value"]
    time_values = time_overview.set_index("metric")["value"]
    horizon_values = horizon_overview.set_index("metric")["value"]
    checks = [
        ("train_has_rows", len(bundle.train) > 0),
        ("test_has_rows", len(bundle.test) > 0),
        ("train_route_timestamp_is_unique", key_values["train_duplicate_key_rows"] == 0),
        ("test_route_timestamp_is_unique", key_values["test_duplicate_key_rows"] == 0),
        ("test_id_is_unique", key_values["test_duplicate_id_rows"] == 0),
        ("all_test_routes_exist_in_train", key_values["test_only_routes"] == 0),
        ("time_step_is_positive", float(time_values["modal_positive_step_minutes"]) > 0),
        ("test_horizons_are_integer_steps", horizon_values["non_integer_horizon_rows"] == 0),
        ("target_has_no_missing", bundle.train[TARGET_COL].isna().sum() == 0),
    ]
    report = pd.DataFrame(checks, columns=["assertion", "passed"])
    report["status"] = np.where(report["passed"], "PASS", "REVIEW")
    return report


def export_table(frame: pd.DataFrame, filename: str, index: bool = False) -> Path:
    ensure_artifact_dirs()
    path = TABLE_DIR / filename
    frame.to_csv(path, index=index)
    return path


def run_raw_audit(bundle: DataBundle | None = None) -> dict[str, object]:
    """Compute and export the complete first-stage raw audit."""

    ensure_artifact_dirs()
    bundle = bundle or load_data()
    manifest = file_manifest()
    train_overview = table_overview(bundle.train, "train")
    test_overview = table_overview(bundle.test, "test")
    key_report = key_integrity_report(bundle)
    distributions = numeric_distribution_report(bundle.train, [*STATUS_COLS, TARGET_COL])
    target_report = target_diagnostics(bundle.train)
    time_overview, delta_counts, route_grid = time_grid_audit(bundle.train)
    step_minutes = float(
        time_overview.set_index("metric").loc["modal_positive_step_minutes", "value"]
    )
    horizon_overview, horizon_counts = test_horizon_audit(bundle, step_minutes)
    routes = route_statistics(bundle.train)
    route_summary = route_distribution_summary(routes)
    profiles = temporal_profiles(bundle.train)
    correlations = correlation_report(bundle.train)
    autocorrelations = target_autocorrelation_report(bundle.train)
    assertions = validation_assertions(bundle, key_report, time_overview, horizon_overview)

    tables: dict[str, pd.DataFrame] = {
        "manifest": manifest,
        "train_overview": train_overview,
        "test_overview": test_overview,
        "key_report": key_report,
        "numeric_distributions": distributions,
        "target_report": target_report,
        "time_overview": time_overview,
        "delta_counts": delta_counts,
        "route_grid": route_grid,
        "horizon_overview": horizon_overview,
        "horizon_counts": horizon_counts,
        "routes": routes,
        "route_summary": route_summary,
        "daily_profile": profiles["daily"],
        "slot_profile": profiles["slot"],
        "weekday_profile": profiles["weekday"],
        "correlations": correlations,
        "autocorrelations": autocorrelations,
        "assertions": assertions,
    }
    for name, frame in tables.items():
        export_table(frame, f"{name}.csv", index=name == "correlations")

    figures = {}
    figures["missingness"] = plot_missingness(bundle)
    figures["target_distribution"] = plot_target_distribution(bundle.train)
    figures["status_distributions"] = plot_status_distributions(bundle.train)
    figures["route_heterogeneity"] = plot_route_heterogeneity(routes)
    figures["temporal_profiles"] = plot_temporal_profiles(profiles)
    figures["correlations"] = plot_correlations(correlations)
    figures["test_horizons"] = plot_test_horizons(horizon_counts)
    figures["route_archetypes"] = plot_route_archetypes(bundle.train, routes)

    metadata = {
        "raw_data_mutated": False,
        "rows_dropped": 0,
        "values_clipped": 0,
        "values_imputed": 0,
        "tables_exported": sorted(tables),
        "plots_exported": sorted(path.name for path in PLOT_DIR.glob("*.png")),
    }
    (ARTIFACT_DIR / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"bundle": bundle, "tables": tables, "figures": figures, "metadata": metadata}


def run_deep_audit(bundle: DataBundle | None = None) -> dict[str, object]:
    """Compute deeper diagnostics while leaving all modeling choices open."""

    ensure_artifact_dirs()
    bundle = bundle or load_data()
    train = bundle.train

    zero_overview, route_zero, zero_by_slot, zero_by_date, zero_runs = zero_pattern_audit(train)
    tail_overview, top_tail_events, tail_by_route, tail_by_date = tail_event_audit(train)
    lead_lag = status_lead_lag_report(train)
    stability_overview, quarter_summary, route_quarter, weekly = temporal_stability_audit(train)
    spectrum = aggregate_periodogram(train)
    segmented_routes, segment_summary = route_segment_audit(train)

    tables: dict[str, pd.DataFrame] = {
        "zero_overview": zero_overview,
        "route_zero": route_zero,
        "zero_by_slot": zero_by_slot,
        "zero_by_date": zero_by_date,
        "zero_runs": zero_runs,
        "tail_overview": tail_overview,
        "top_tail_events": top_tail_events,
        "tail_by_route": tail_by_route,
        "tail_by_date": tail_by_date,
        "status_lead_lag": lead_lag,
        "stability_overview": stability_overview,
        "quarter_summary": quarter_summary,
        "route_quarter": route_quarter,
        "weekly_stability": weekly,
        "periodogram": spectrum,
        "segmented_routes": segmented_routes,
        "segment_summary": segment_summary,
    }
    for name, frame in tables.items():
        export_table(frame, f"deep_{name}.csv")

    figures = {
        "zero_patterns": plot_zero_patterns(route_zero, zero_by_slot, zero_by_date),
        "tail_patterns": plot_tail_patterns(tail_overview, tail_by_route, tail_by_date),
        "status_lead_lag": plot_status_lead_lag(lead_lag),
        "temporal_stability": plot_temporal_stability(weekly, spectrum),
        "route_segments": plot_route_segments(segment_summary),
    }
    metadata = {
        "raw_data_mutated": False,
        "rows_dropped": 0,
        "values_clipped": 0,
        "values_imputed": 0,
        "model_trained": False,
        "tables_exported": sorted(tables),
        "plots_exported": sorted(path.name for path in PLOT_DIR.glob("*.png")),
    }
    (ARTIFACT_DIR / "deep_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"bundle": bundle, "tables": tables, "figures": figures, "metadata": metadata}


def _print_frame(title: str, frame: pd.DataFrame, max_rows: int = 30) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    print(frame.head(max_rows).to_string(index=False))


def main() -> None:
    result = run_raw_audit()
    tables = result["tables"]
    for name in [
        "manifest",
        "train_overview",
        "test_overview",
        "key_report",
        "target_report",
        "time_overview",
        "horizon_overview",
        "route_summary",
        "autocorrelations",
        "assertions",
    ]:
        _print_frame(name, tables[name])
    print(f"\nArtifacts: {ARTIFACT_DIR}")
    print(json.dumps(result["metadata"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
