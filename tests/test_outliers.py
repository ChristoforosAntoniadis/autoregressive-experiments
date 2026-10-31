"""
Tests for outlier detection and handling in autoregressive_experiments.outliers, and its use in
the forecasts (model_forecasts), the XGBoost loss (ml_forecasting) and the test sensitivity table
(forecast_evaluation).
"""

import numpy as np
import pandas as pd
import pytest

from autoregressive_experiments.forecast_evaluation import flagged_months, sensitivity_rows
from autoregressive_experiments.ml_forecasting import XGBOOST, build_model, lag_features
from autoregressive_experiments.model_forecasts import NAIVE, SARIMA, VAR_MODEL, series_path
from autoregressive_experiments.outliers import (
    HALF_WIDTH,
    THRESHOLD,
    clip_levels,
    hampel,
    pulse_dummies,
)
from autoregressive_experiments.tuning import origin_paths

# Alternating small changes with one spike in the middle: 13 changes, so the middle one's
# window (HALF_WIDTH 6 on each side) is the whole series.
CHANGES = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 5.0, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
LEVELS = np.concatenate([[10.0], 10.0 + np.cumsum(CHANGES)])
# Bounds of the spike's window: six 0.1s, six 0.2s and 5.0 have median 0.2; the absolute
# deviations (six 0.1s, six 0s, 4.8) have median 0.1; so 0.2 +/- 3 x 1.4826 x 0.1.
SPREAD = 3 * 1.4826 * 0.1


def test_settings_are_the_ones_the_hand_computations_assume() -> None:
    """The expected values below use a half-width of 6 months and a threshold of 3."""
    assert (HALF_WIDTH, THRESHOLD) == (6, 3.0)


def test_hampel_flags_only_the_spike_by_hand() -> None:
    """Only the 5.0 change is outside its bounds, 0.2 - 0.445 and 0.2 + 0.445; the small changes
    sit inside theirs (each window has median 0.1 or 0.2 and spread of the same order)."""
    lower, upper, flags = hampel(CHANGES)

    assert flags.tolist() == [index == 6 for index in range(13)]
    assert (lower[6], upper[6]) == pytest.approx((0.2 - SPREAD, 0.2 + SPREAD))


def test_a_constant_window_flags_nothing() -> None:
    """Equal changes have no spread, so nothing can be called unusual."""
    assert not hampel(np.full(10, 0.3))[2].any()


def test_clipping_caps_the_spike_and_keeps_the_last_level() -> None:
    """The 5.0 change becomes its upper bound, 0.2 + 0.445; the last level is unchanged, and
    every level before the spike moves up by the amount clipped off, 5.0 - 0.645."""
    clipped = clip_levels(LEVELS)

    assert clipped[-1] == LEVELS[-1]
    assert np.diff(clipped)[6] == pytest.approx(0.2 + SPREAD)
    assert clipped[:7] - LEVELS[:7] == pytest.approx(np.full(7, 5.0 - 0.2 - SPREAD))
    assert np.array_equal(clipped[7:], LEVELS[7:])
    assert np.array_equal(clip_levels(np.column_stack([LEVELS, LEVELS]))[:, 1], clipped)


def test_pulse_dummies_by_hand() -> None:
    """One flagged change (the 7th): a single column with its 1 in row 6; fitted from the 3rd
    change on, row 4; fitted from the 8th on, no flagged change is left."""
    full = pulse_dummies(CHANGES)
    later = pulse_dummies(CHANGES, start=2)

    assert full is not None and full.shape == (13, 1) and np.flatnonzero(full[:, 0]) == [6]
    assert later is not None and later.shape == (11, 1) and np.flatnonzero(later[:, 0]) == [4]
    assert pulse_dummies(CHANGES, start=7) is None


def spiky_history() -> np.ndarray:
    """
    Two random walks with a drift and one large jump in month 40.

    Returns
    -------
    np.ndarray
        Shape (80, 2), trillions; seed 0.
    """
    steps = np.random.default_rng(0).normal(0.05, 0.1, size=(80, 2))
    steps[40] += 3.0
    return 5 + np.cumsum(steps, axis=0)


def test_each_outlier_handling_changes_the_sarima_forecast() -> None:
    """With the jump in its history, SARIMA forecasts differently when the jump is clipped or
    given a pulse regressor; every forecast is finite."""
    history = spiky_history()
    orders = {"p": 1, "q": 0, "P": 0, "Q": 0, "trend": "c"}

    paths = {
        mode: series_path(SARIMA, orders | {"outliers": mode}, history, np.array([]), 0, 6)
        for mode in ("none", "clip", "dummies")
    }

    assert all(np.isfinite(path).all() for path in paths.values())
    assert not np.allclose(paths["none"], paths["clip"])
    assert not np.allclose(paths["none"], paths["dummies"])


def test_clipping_uses_only_the_history_up_to_each_origin() -> None:
    """A clipped VAR's forecasts from months 50 and 60 don't change when every later month
    changes, and naive never clips."""
    history = spiky_history()
    changed = history.copy()
    changed[61:] += 50.0
    settings = {"lags": 1, "trend": "c", "outliers": "clip"}
    dates = np.arange(80)

    before = origin_paths(VAR_MODEL, settings, history, dates, [50, 60], 79, [0, 1])
    after = origin_paths(VAR_MODEL, settings, changed, dates, [50, 60], 79, [0, 1])
    naive = origin_paths(NAIVE, {}, history, dates, [60], 79, [0])

    for column in (0, 1):
        for first, second in zip(before[column], after[column]):
            assert np.array_equal(first, second)
    assert naive[0][0] == pytest.approx(np.full(12, history[60, 0]))


def test_xgboost_trains_with_the_huber_loss() -> None:
    """A loss of "huber" gives XGBoost its pseudo-Huber objective; the fit still learns a
    straight rule well (R^2 above 0.8)."""
    rng = np.random.default_rng(0)
    changes = np.zeros((200, 2))
    for t in range(1, 200):
        changes[t] = [0.5 * changes[t - 1, 0] + rng.normal(), 0.8 * changes[t - 1, 0]]
    features, target = lag_features(changes, 1), changes[1:, 1]
    settings = {"lags": 1, "n_estimators": 200, "max_depth": 3, "loss": "huber"}

    model = build_model(XGBOOST, settings).fit(features, target)

    assert model.regressor_[-1].get_params()["objective"] == "reg:pseudohubererror"
    assert 1 - np.var(target - model.predict(features)) / np.var(target) > 0.8


def test_sensitivity_drops_forecasts_touching_a_flagged_month_by_hand() -> None:
    """Four one-month forecasts from months 0-3; month 2 is flagged, so the forecasts from month 1
    (target 2) and month 2 (origin 2) are dropped. Left: origins 0 and 3, naive errors 1 and 1,
    the model's 0.5 and 0.5: MAE 0.5 vs 1, ratio 0.5, n = 2."""
    base = pd.DataFrame({"origin_index": range(4), "target_index": range(1, 5), "horizon": 1})
    tests = {
        NAIVE: base.assign(error=[1.0, 1.0, 1.0, 1.0]),
        "model": base.assign(error=[0.5, 4.0, -9.0, -0.5]),
    }
    flags = np.array([False, False, True, False, False])

    rows = {r["model"]: r for r in sensitivity_rows(tests, flags) if r["horizon"] == 1}

    assert rows["model"]["n"] == 2
    assert rows["model"]["MAE"] == pytest.approx(0.5)
    assert rows["model"]["MAE_vs_naive"] == pytest.approx(0.5)
    assert rows[NAIVE]["MAE_vs_naive"] == pytest.approx(1.0)


def test_flagged_months_line_up_with_the_levels() -> None:
    """The spike is the change into month 7 (the 7th change), so month 7 is flagged and the
    first month, which has no change, never is."""
    flags = flagged_months(LEVELS)

    assert flags.tolist() == [index == 7 for index in range(14)]
