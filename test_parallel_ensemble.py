from dataclasses import replace

import numpy as np
import pandas as pd

from parallel_ensemble import (
    HORIZON,
    PanelData,
    TCNWindowDataset,
    _make_tcn_model,
    build_matrix,
    make_folds,
)


def synthetic_panel(n_steps: int = 500, n_routes: int = 3) -> PanelData:
    timestamps = pd.date_range("2025-01-01 00:00", periods=n_steps, freq="30min")
    route_ids = np.arange(100, 100 + n_routes)
    target = np.arange(n_steps * n_routes, dtype=np.float32).reshape(n_steps, n_routes)
    statuses = np.stack([target + offset for offset in range(6)], axis=-1)
    future = pd.date_range(timestamps[-1] + pd.Timedelta(minutes=30), periods=HORIZON, freq="30min")
    return PanelData(
        route_ids=route_ids,
        timestamps=timestamps,
        target=target,
        statuses=statuses,
        future_timestamps=future,
        test_ids=np.arange(n_routes * HORIZON),
    )


def test_features_do_not_depend_on_future_target_or_status() -> None:
    panel = synthetic_panel()
    origin = 400
    original = build_matrix(panel, [origin], include_labels=True)

    changed_target = panel.target.copy()
    changed_statuses = panel.statuses.copy()
    changed_target[origin + 1 :] += 1_000_000
    changed_statuses[origin + 1 :] += 1_000_000
    changed = build_matrix(
        replace(panel, target=changed_target, statuses=changed_statuses),
        [origin],
        include_labels=True,
    )

    np.testing.assert_array_equal(original.X, changed.X)
    assert not np.array_equal(original.y, changed.y)


def test_one_route_change_does_not_change_other_routes_features() -> None:
    panel = synthetic_panel()
    origin = 400
    original = build_matrix(panel, [origin], include_labels=False)

    changed_target = panel.target.copy()
    changed_statuses = panel.statuses.copy()
    changed_target[: origin + 1, 0] += 123_456
    changed_statuses[: origin + 1, 0] += 123_456
    changed = build_matrix(
        replace(panel, target=changed_target, statuses=changed_statuses),
        [origin],
        include_labels=False,
    )

    unchanged_rows = original.route_codes != 0
    np.testing.assert_array_equal(original.X[unchanged_rows], changed.X[unchanged_rows])
    assert not np.array_equal(original.X[~unchanged_rows], changed.X[~unchanged_rows])


def test_backtest_folds_match_deployment_time_of_day() -> None:
    panel = synthetic_panel(n_steps=1_000)
    folds = make_folds(panel, n_folds=4, spacing=48)
    deployment_time = panel.timestamps[-1].time()

    assert all(fold.origin_timestamp.time() == deployment_time for fold in folds)
    assert all(len(fold.validation_timestamps) == HORIZON for fold in folds)
    assert all(
        fold.validation_timestamps[0] == fold.origin_timestamp + pd.Timedelta(minutes=30)
        for fold in folds
    )


def test_feature_and_label_order_is_horizon_then_route() -> None:
    panel = synthetic_panel()
    origin = 400
    bundle = build_matrix(panel, [origin], include_labels=True)
    n_routes = len(panel.route_ids)

    np.testing.assert_array_equal(bundle.horizons, np.repeat(np.arange(1, HORIZON + 1), n_routes))
    np.testing.assert_array_equal(bundle.route_codes, np.tile(np.arange(n_routes), HORIZON))
    np.testing.assert_array_equal(
        bundle.y,
        panel.target[origin + 1 : origin + HORIZON + 1].reshape(-1),
    )


def test_tcn_window_is_causal_and_has_direct_multi_horizon_target() -> None:
    panel = synthetic_panel()
    origin = 400
    dataset = TCNWindowDataset(panel, [origin], sequence_length=336)
    sequence, route, log_scales, future_time, target, target_scale = dataset[0]

    changed_target = panel.target.copy()
    changed_statuses = panel.statuses.copy()
    changed_target[origin + 1 :] += 1_000_000
    changed_statuses[origin + 1 :] += 1_000_000
    changed_dataset = TCNWindowDataset(
        replace(panel, target=changed_target, statuses=changed_statuses),
        [origin],
        sequence_length=336,
    )
    changed_sequence, _, changed_scales, changed_time, changed_target_tensor, changed_scale = changed_dataset[0]

    np.testing.assert_array_equal(sequence.numpy(), changed_sequence.numpy())
    np.testing.assert_array_equal(log_scales.numpy(), changed_scales.numpy())
    np.testing.assert_array_equal(future_time.numpy(), changed_time.numpy())
    assert float(target_scale) == float(changed_scale)
    assert route == 0
    assert sequence.shape == (7, 336)
    assert future_time.shape == (HORIZON, 5)
    assert target.shape == (HORIZON,)
    assert not np.array_equal(target.numpy(), changed_target_tensor.numpy())


def test_both_tcn_architectures_return_eight_nonnegative_values() -> None:
    import torch

    batch = 4
    sequence = torch.rand(batch, 7, 336)
    route = torch.arange(batch)
    log_scales = torch.rand(batch, 7)
    future_time = torch.rand(batch, HORIZON, 5)

    for architecture in ("single", "two_stream"):
        model = _make_tcn_model(n_routes=10, channels=16, architecture=architecture)
        prediction = model(sequence, route, log_scales, future_time)
        assert prediction.shape == (batch, HORIZON)
        assert torch.isfinite(prediction).all()
        assert (prediction >= 0).all()
