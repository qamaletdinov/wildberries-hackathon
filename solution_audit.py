"""Executable audit of two historical Wildberries forecasting solutions.

The audit imports feature functions but never trains models or overwrites files in
the historical repositories.  Small synthetic time series expose leakage and
train/test parity problems through observable behavior.  Submission files are
read only for structural and scale diagnostics; no leaderboard labels are used.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CURRENT_REPO = Path(__file__).resolve().parent
WB_REPO = Path(
    os.environ.get("WB_COMPARISON_REPO", CURRENT_REPO.parent.parent / "wb")
).resolve()
ARTIFACT_DIR = CURRENT_REPO / "artifacts" / "solution_audit"
TABLE_DIR = ARTIFACT_DIR / "tables"
PLOT_DIR = ARTIFACT_DIR / "plots"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> pd.DataFrame:
    paths = [
        CURRENT_REPO / "solution.py",
        WB_REPO / "src" / "features.py",
        WB_REPO / "src" / "train_improved.py",
        WB_REPO / "src" / "predict_improved.py",
        WB_REPO / "src" / "train_advanced.py",
        WB_REPO / "src" / "predict_advanced.py",
        WB_REPO / "src" / "train_per_route.py",
        WB_REPO / "src" / "predict_per_route.py",
    ]
    return pd.DataFrame(
        [
            {
                "file": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else np.nan,
                "sha256": sha256(path) if path.exists() else None,
            }
            for path in paths
        ]
    )


def _import_solution_modules() -> tuple[Any, Any, Any]:
    """Import guarded modules from both repositories without running main()."""

    current_path = str(CURRENT_REPO)
    wb_path = str(WB_REPO)
    for path in [current_path, wb_path]:
        if path not in sys.path:
            sys.path.insert(0, path)
    current_solution = importlib.import_module("solution")
    wb_features = importlib.import_module("src.features")
    wb_improved = importlib.import_module("src.train_improved")
    return current_solution, wb_features, wb_improved


def _source_location(function: Any) -> tuple[str, int]:
    lines, start = inspect.getsourcelines(function)
    del lines
    return inspect.getsourcefile(function) or "unknown", start


def _finding(
    solution: str,
    finding_id: str,
    severity: str,
    category: str,
    check: str,
    observed: object,
    expected: object,
    risk: str,
    function: Any,
    passed: bool,
) -> dict[str, object]:
    path, line = _source_location(function)
    return {
        "solution": solution,
        "finding_id": finding_id,
        "severity": severity,
        "category": category,
        "check": check,
        "observed": str(observed),
        "expected": str(expected),
        "risk": risk,
        "file": path,
        "line": line,
        "passed": bool(passed),
        "status": "PASS" if passed else "FAIL",
    }


def _synthetic_status_frame(targets: list[float], route_ids: list[int] | None = None) -> pd.DataFrame:
    size = len(targets)
    route_ids = route_ids or [0] * size
    timestamps = []
    counters: dict[int, int] = {}
    origin = pd.Timestamp("2025-01-01")
    for route_id in route_ids:
        offset = counters.get(route_id, 0)
        timestamps.append(origin + pd.Timedelta(minutes=30 * offset))
        counters[route_id] = offset + 1
    frame = pd.DataFrame(
        {
            "route_id": route_ids,
            "timestamp": timestamps,
            "target_1h": targets,
        }
    )
    for index in range(1, 7):
        frame[f"status_{index}"] = np.arange(size) + index
    return frame


def audit_wb_solution(wb_features: Any, wb_improved: Any) -> tuple[list[dict[str, object]], pd.DataFrame]:
    findings: list[dict[str, object]] = []

    # Dynamic proof 1: rolling feature at row t contains target at row t.
    frame = _synthetic_status_frame([10.0, 20.0, 30.0, 40.0])
    rolling = wb_features.create_rolling_features(frame.copy())
    observed_first = rolling.loc[0, "target_1h_rolling_mean_2"]
    findings.append(
        _finding(
            "wb_second_solution",
            "WB-ROLLING-CURRENT-TARGET",
            "P0",
            "target_leakage",
            "Does first rolling feature contain the current label?",
            observed_first,
            "NaN/absent until past target exists",
            "Validation can become unrealistically strong because X(t) contains y(t).",
            wb_features.create_rolling_features,
            passed=pd.isna(observed_first),
        )
    )

    # Dynamic proof 2: future validation target changes aggregates on earlier rows.
    full = _synthetic_status_frame([1.0, 1.0, 100.0])
    full_aggregated = wb_improved.add_route_aggregates(full.copy())
    train_only_aggregated = wb_improved.add_route_aggregates(full.iloc[:2].copy())
    observed_full_mean = float(full_aggregated.loc[0, "target_1h_mean"])
    expected_train_mean = float(train_only_aggregated.loc[0, "target_1h_mean"])
    findings.append(
        _finding(
            "wb_second_solution",
            "WB-FULL-HISTORY-ROUTE-AGG",
            "P0",
            "temporal_leakage",
            "Does a future target change the route feature on an earlier row?",
            observed_full_mean,
            expected_train_mean,
            "Computing aggregates before temporal split leaks validation labels into features.",
            wb_improved.add_route_aggregates,
            passed=np.isclose(observed_full_mean, expected_train_mean),
        )
    )

    # Dynamic proof 3: train and test feature sets differ when test has no labels/status.
    train_features = wb_features.engineer_features(frame.copy(), is_train=True)
    train_features = wb_improved.add_route_aggregates(train_features, is_train=True)
    train_features = wb_improved.add_hour_seasonality(train_features, is_train=True)
    test_raw = pd.DataFrame(
        {
            "id": [0, 1],
            "route_id": [0, 0],
            "timestamp": [pd.Timestamp("2025-01-01 02:00"), pd.Timestamp("2025-01-01 02:30")],
        }
    )
    test_features = wb_features.engineer_features(test_raw.copy(), is_train=False)
    test_features = wb_improved.add_route_aggregates(test_features, is_train=False)
    test_features = wb_improved.add_hour_seasonality(test_features, is_train=False)
    model_features = wb_features.get_feature_cols(train_features)
    missing_in_test = sorted(set(model_features).difference(test_features.columns))
    findings.append(
        _finding(
            "wb_second_solution",
            "WB-TRAIN-TEST-FEATURE-PARITY",
            "P0",
            "training_serving_skew",
            "Are all model features produced by the test pipeline?",
            f"{len(missing_in_test)} missing: {missing_in_test}",
            "0 missing features",
            "Inference fills absent trained features with zero, changing scale and semantics.",
            wb_features.engineer_features,
            passed=len(missing_in_test) == 0,
        )
    )

    demonstration = pd.DataFrame(
        {
            "check": [
                "WB rolling mean at first row",
                "WB early-row route mean using full data",
                "WB early-row route mean using train-only data",
                "WB missing trained features in test pipeline",
            ],
            "value": [observed_first, observed_full_mean, expected_train_mean, len(missing_in_test)],
        }
    )
    return findings, demonstration


def audit_current_solution(current_solution: Any) -> tuple[list[dict[str, object]], pd.DataFrame]:
    findings: list[dict[str, object]] = []

    # Dynamic proof 1: rolling is applied to a globally indexed shifted series,
    # so the first rows of a new route can inherit values from the previous route.
    frame = _synthetic_status_frame(
        [10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0],
        route_ids=[0, 0, 0, 0, 1, 1, 1, 1],
    ).sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    rolled = current_solution.add_rolling_features(frame.copy(), min_lag=1)
    route_one_first = rolled.index[rolled["route_id"].eq(1)][0]
    observed_boundary = rolled.loc[route_one_first, "roll_mean_6"]
    findings.append(
        _finding(
            "current_hackathon_solution",
            "CURRENT-CROSS-ROUTE-ROLLING",
            "P0",
            "entity_contamination",
            "Does the first row of route 1 inherit rolling target values from route 0?",
            observed_boundary,
            "NaN at a route boundary with min_lag=1",
            "Features for one route can contain target history from a different route.",
            current_solution.add_rolling_features,
            passed=pd.isna(observed_boundary),
        )
    )

    # Dynamic proof 2: shuffled target encoding for the earliest row changes when
    # only future targets are perturbed.
    base = pd.DataFrame(
        {
            "route_id": [0] * 20,
            "timestamp": pd.date_range("2025-01-01", periods=20, freq="30min"),
            "target_1h": np.arange(1, 21, dtype=float),
        }
    )
    changed = base.copy()
    changed.loc[1:, "target_1h"] += 10_000
    mask = np.ones(len(base), dtype=bool)
    encoded_base = current_solution.add_target_encoding(base.copy(), mask)
    encoded_changed = current_solution.add_target_encoding(changed.copy(), mask)
    base_first = float(encoded_base.loc[0, "route_target_enc"])
    changed_first = float(encoded_changed.loc[0, "route_target_enc"])
    findings.append(
        _finding(
            "current_hackathon_solution",
            "CURRENT-RANDOM-TARGET-ENCODING",
            "P1",
            "temporal_leakage",
            "Does changing only future labels alter the earliest row's encoding?",
            f"before={base_first:.3f}; after={changed_first:.3f}",
            "earliest-row feature unchanged by future labels",
            "Random K-fold target encoding violates chronological feature availability.",
            current_solution.add_target_encoding,
            passed=np.isclose(base_first, changed_first),
        )
    )

    # Dynamic proof 3: aggregate features on earlier rows change if future labels
    # are included in the train_df argument.
    full = _synthetic_status_frame([1.0, 1.0, 100.0])
    full["hour"] = full["timestamp"].dt.hour
    full["dayofweek"] = full["timestamp"].dt.dayofweek
    early = full.iloc[:2].copy()
    with_future = current_solution.add_route_aggregations(early.copy(), full.copy())
    without_future = current_solution.add_route_aggregations(early.copy(), early.copy())
    observed_full_mean = float(with_future.loc[0, "route_mean"])
    expected_past_mean = float(without_future.loc[0, "route_mean"])
    findings.append(
        _finding(
            "current_hackathon_solution",
            "CURRENT-FULL-HISTORY-ROUTE-AGG",
            "P1",
            "temporal_leakage",
            "Does a future target alter an aggregate feature on earlier rows?",
            observed_full_mean,
            expected_past_mean,
            "Safe only when train_df is strictly limited to the past of evaluated rows.",
            current_solution.add_route_aggregations,
            passed=np.isclose(observed_full_mean, expected_past_mean),
        )
    )

    demonstration = pd.DataFrame(
        {
            "check": [
                "Current solution route-boundary roll_mean_6",
                "Current earliest target encoding before future perturbation",
                "Current earliest target encoding after future perturbation",
                "Current early-row route mean using future",
                "Current early-row route mean using past only",
            ],
            "value": [
                observed_boundary,
                base_first,
                changed_first,
                observed_full_mean,
                expected_past_mean,
            ],
        }
    )
    return findings, demonstration


def submission_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit output structure and scale without access to hidden labels."""

    candidates = [("current_hackathon_solution", CURRENT_REPO / "submission_solo.csv")]
    candidates.extend(("wb_second_solution", path) for path in sorted(WB_REPO.glob("submission*.csv")))
    summaries: list[dict[str, object]] = []
    predictions: dict[str, pd.Series] = {}
    for solution_name, path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        prediction_candidates = [
            column
            for column in frame.columns
            if column != "id" and pd.api.types.is_numeric_dtype(frame[column])
        ]
        preferred = [column for column in ["y_pred", "forecast"] if column in prediction_candidates]
        if not preferred and len(prediction_candidates) != 1:
            summaries.append(
                {
                    "solution": solution_name,
                    "file": path.name,
                    "rows": len(frame),
                    "prediction_column": None,
                    "status": f"REVIEW columns={list(frame.columns)}",
                }
            )
            continue
        prediction_column = preferred[0] if preferred else prediction_candidates[0]
        values = frame[prediction_column]
        key = f"{solution_name}:{path.name}"
        predictions[key] = values.reset_index(drop=True)
        summaries.append(
            {
                "solution": solution_name,
                "file": path.name,
                "rows": len(frame),
                "prediction_column": prediction_column,
                "id_unique": bool(frame["id"].is_unique) if "id" in frame else False,
                "missing_predictions": int(values.isna().sum()),
                "negative_predictions": int((values < 0).sum()),
                "minimum": float(values.min()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "maximum": float(values.max()),
                "std": float(values.std()),
                "status": "OK",
            }
        )
    summary = pd.DataFrame(summaries)

    equal_length = {name: series for name, series in predictions.items() if len(series) == 8000}
    correlation = pd.DataFrame(equal_length).corr() if equal_length else pd.DataFrame()
    return summary, correlation


def plot_findings(findings: pd.DataFrame) -> Path:
    ensure_dirs()
    counts = findings.groupby(["solution", "severity", "status"]).size().reset_index(name="checks")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=counts, x="solution", y="checks", hue="severity", ax=ax)
    ax.set(title="Executable audit findings by solution", xlabel="Solution", ylabel="Checks")
    fig.tight_layout()
    path = PLOT_DIR / "01_executable_findings.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_submission_scales(summary: pd.DataFrame) -> Path:
    ensure_dirs()
    valid = summary[summary["status"].eq("OK")].copy()
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.scatterplot(
        data=valid,
        x="mean",
        y="std",
        hue="solution",
        size="maximum",
        sizes=(30, 300),
        alpha=0.75,
        ax=ax,
    )
    for row in valid.itertuples(index=False):
        if row.file in {
            "submission_solo.csv",
            "submission_v2.csv",
            "submission_v3_lgb_xgb.csv",
            "submission_v4_final.csv",
            "submission_per_route.csv",
        }:
            ax.annotate(row.file, (row.mean, row.std), xytext=(4, 4), textcoords="offset points")
    ax.set(title="Saved submission scales (no hidden labels)", xlabel="Prediction mean", ylabel="Prediction std")
    fig.tight_layout()
    path = PLOT_DIR / "02_submission_scales.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def run_solution_audit() -> dict[str, object]:
    if not WB_REPO.is_dir():
        raise FileNotFoundError(
            "The optional historical comparison repository was not found. "
            "Set WB_COMPARISON_REPO to its local path before running this audit."
        )
    ensure_dirs()
    current_solution, wb_features, wb_improved = _import_solution_modules()
    wb_findings, wb_demo = audit_wb_solution(wb_features, wb_improved)
    current_findings, current_demo = audit_current_solution(current_solution)
    findings = pd.DataFrame([*wb_findings, *current_findings])
    submission_summary, submission_correlation = submission_inventory()
    manifest = source_manifest()
    tables = {
        "source_manifest": manifest,
        "findings": findings,
        "wb_demonstration": wb_demo,
        "current_demonstration": current_demo,
        "submission_summary": submission_summary,
        "submission_correlation": submission_correlation,
    }
    for name, frame in tables.items():
        frame.to_csv(TABLE_DIR / f"{name}.csv", index=name == "submission_correlation")
    plots = {
        "findings": plot_findings(findings),
        "submission_scales": plot_submission_scales(submission_summary),
    }
    metadata = {
        "historical_source_files_modified": False,
        "models_trained": False,
        "hidden_labels_used": False,
        "failed_checks": int((~findings["passed"]).sum()),
        "passed_checks": int(findings["passed"].sum()),
    }
    (ARTIFACT_DIR / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"tables": tables, "plots": plots, "metadata": metadata}


def main() -> None:
    result = run_solution_audit()
    print(result["tables"]["findings"].to_string(index=False))
    print("\nSubmission summary")
    print(result["tables"]["submission_summary"].to_string(index=False))
    print(f"\nArtifacts: {ARTIFACT_DIR}")
    print(json.dumps(result["metadata"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
