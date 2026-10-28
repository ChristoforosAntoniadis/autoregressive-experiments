"""
Tests for the lag-feature ML family in autoregressive_experiments.ml_forecasting.
"""

import numpy as np
import pytest

from autoregressive_experiments.ml_forecasting import (
    BOOSTING,
    LASSO,
    RIDGE,
    SETTINGS,
    fit_model,
    lag_features,
    ml_forecaster,
    recursive_forecast,
)


class LinearRule:
    """A stand-in model whose prediction is a fixed linear rule, so a forecast can be checked
    by hand."""

    def __init__(self, weights: list[float]) -> None:
        """
        Remember the weights.

        Parameters
        ----------
        weights : list of float
            One weight per feature column, in lag_features order.
        """
        self.weights = np.array(weights)

    def predict(self, rows: np.ndarray) -> np.ndarray:
        """
        Apply the rule.

        Parameters
        ----------
        rows : np.ndarray
            Feature rows, shape (n, len(weights)).

        Returns
        -------
        np.ndarray
            rows @ weights, shape (n,).
        """
        return rows @ self.weights


def simulated_system(months: int) -> np.ndarray:
    """
    Levels of two series whose changes follow a known linear rule with small noise.

    Parameters
    ----------
    months : int
        Number of changes to simulate.

    Returns
    -------
    np.ndarray
        Levels, shape (months + 1, 2), starting at 0.

    Notes
    -----
    x_t = 0.5 x_(t-1) + noise; y_t = 0.8 x_(t-1) + noise, with noise SD 0.1 (seed 0): y
    depends on last month's x, so a model of y needs both series' lags.
    """
    rng = np.random.default_rng(0)
    changes = np.zeros((months, 2))
    for t in range(1, months):
        changes[t, 0] = 0.5 * changes[t - 1, 0] + rng.normal()
        changes[t, 1] = 0.8 * changes[t - 1, 0] + 0.1 * rng.normal()
    return np.vstack([np.zeros(2), changes.cumsum(axis=0)])


def test_lag_features_by_hand() -> None:
    """Changes (1, 10), (2, 20), (3, 30), (4, 40) with 2 lags: months 2 and 3 get rows, each
    the first series' last two changes (most recent first), then the second series'."""
    changes = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

    rows = lag_features(changes, 2)

    assert rows.tolist() == [[2.0, 1.0, 20.0, 10.0], [3.0, 2.0, 30.0, 20.0]]


def test_recursive_forecast_feeds_each_month_back_by_hand() -> None:
    """One lag; x' = 0.5 x and y' = x (each series' next change from last month's x).

    From x = 2, y = 7: month 1 gives x = 1, y = 2; month 2 gives x = 0.5, y = 1, which only
    works if month 1's forecasts were fed back as month 2's inputs.
    """
    # Feature columns with one lag: [x lag 1, y lag 1].
    models = [LinearRule([0.5, 0.0]), LinearRule([1.0, 0.0])]

    path = recursive_forecast(models, np.array([[9.0, 9.0], [2.0, 7.0]]), 1, 2)

    assert path.tolist() == [[1.0, 2.0], [0.5, 1.0]]


def test_ridge_recovers_the_simulated_rule() -> None:
    """On 400 simulated months with one lag, Ridge's one-month predictions follow the true
    rule (y' = 0.8 x, x' = 0.5 x) within the noise."""
    changes = np.diff(simulated_system(400), axis=0)
    features = lag_features(changes, 1)

    fitted = [fit_model(RIDGE, features, changes[1:, column]) for column in range(2)]

    probe = np.array([[1.0, 0.0], [-2.0, 0.0]])
    assert fitted[0].predict(probe) == pytest.approx([0.5, -1.0], abs=0.15)
    assert fitted[1].predict(probe) == pytest.approx([0.8, -1.6], abs=0.05)


def test_boosting_is_reproducible_and_stops_before_the_cap() -> None:
    """Two fits on the same data give identical predictions (fixed seed, time-ordered
    holdout), and early stopping picks fewer trees than the cap."""
    changes = np.diff(simulated_system(200), axis=0)
    features = lag_features(changes, 3)

    first = fit_model(BOOSTING, features, changes[3:, 1])
    second = fit_model(BOOSTING, features, changes[3:, 1])

    assert np.array_equal(first.predict(features), second.predict(features))
    assert 1 <= first.n_iter_ < SETTINGS.boosting_max_iterations


def test_ml_forecaster_ignores_everything_after_the_origin() -> None:
    """Changing every month after the origin leaves the forecast unchanged, and a history that
    doesn't match the harness's levels trips the leakage guard."""
    levels = simulated_system(120)
    origin = 90
    forecaster = ml_forecaster(LASSO, 3, levels, 1, {})

    before = forecaster(levels[: origin + 1, 1], 6)["forecast"]
    changed = levels.copy()
    changed[origin + 1 :] += 100.0
    after = ml_forecaster(LASSO, 3, changed, 1, {})(changed[: origin + 1, 1], 6)["forecast"]

    assert np.array_equal(before, after)
    with pytest.raises(AssertionError):
        forecaster(levels[: origin + 1, 0], 6)
