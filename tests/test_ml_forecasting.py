"""
Tests for the lag-feature ML family in autoregressive_experiments.ml_forecasting.
"""

import numpy as np
import pytest

from autoregressive_experiments.ml_forecasting import (
    FOREST,
    LASSO,
    RIDGE,
    XGBOOST,
    build_model,
    fit_model,
    lag_columns,
    lag_features,
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


def test_lag_columns_pick_fewer_lags_out_of_a_wider_matrix() -> None:
    """With 3 lags per series the columns are x1 x2 x3 y1 y2 y3; 2 lags are columns 0, 1, 3, 4,
    which on the same months equal the 2-lag matrix (it starts a month earlier, so one row
    later lines up)."""
    changes = np.arange(20.0).reshape(10, 2)

    columns = lag_columns(2, 3, 2)

    assert columns == [0, 1, 3, 4]
    assert np.array_equal(lag_features(changes, 3)[:, columns], lag_features(changes, 2)[1:])


def test_recursive_forecast_feeds_each_month_back_by_hand() -> None:
    """One lag; x' = 0.5 x and y' = x (each series' next change from last month's x).

    From x = 2, y = 7: month 1 gives x = 1, y = 2; month 2 gives x = 0.5, y = 1, which only
    works if month 1's forecasts were fed back as month 2's inputs.
    """
    # Feature columns with one lag: [x lag 1, y lag 1].
    models = [(LinearRule([0.5, 0.0]), 1), (LinearRule([1.0, 0.0]), 1)]

    path = recursive_forecast(models, np.array([[9.0, 9.0], [2.0, 7.0]]), 2)

    assert path.tolist() == [[1.0, 2.0], [0.5, 1.0]]


def test_recursive_forecast_lets_each_series_use_its_own_lags() -> None:
    """x' = x1 + 0.5 x2 (2 lags: columns x1 x2 y1 y2), y' = y1 (1 lag: columns x1 y1).

    From changes (1, 10), (2, 20): x' = 2 + 0.5 x 1 = 2.5, y' = 20; then x'' = 2.5 + 0.5 x 2
    = 3.5, y'' = 20.
    """
    models = [(LinearRule([1.0, 0.5, 0.0, 0.0]), 2), (LinearRule([0.0, 1.0]), 1)]

    path = recursive_forecast(models, np.array([[1.0, 10.0], [2.0, 20.0]]), 2)

    assert path.tolist() == [[2.5, 20.0], [3.5, 20.0]]


class Overflowing:
    """A stand-in model whose prediction fails as a scaler does on an infinite value."""

    def __init__(self, message: str) -> None:
        """
        Remember the error to raise.

        Parameters
        ----------
        message : str
            The ValueError's message.
        """
        self.message = message

    def predict(self, row: np.ndarray) -> np.ndarray:
        """
        Fail with the stored message.

        Parameters
        ----------
        row : np.ndarray
            Ignored.

        Returns
        -------
        np.ndarray
            Never; always raises.

        Raises
        ------
        ValueError
            Always, with the stored message.
        """
        raise ValueError(self.message)


def test_an_overflowing_recursion_gives_no_forecast_but_other_errors_still_raise() -> None:
    """A prediction that overflows to infinity makes the whole path NaN (no usable forecast);
    any other ValueError still stops the run."""
    changes = np.array([[1.0, 1.0], [2.0, 2.0]])
    overflow = Overflowing("Input contains infinity or a value too large for dtype('float64').")

    path = recursive_forecast([(LinearRule([1.0, 0.0]), 1), (overflow, 1)], changes, 3)

    assert path.shape == (3, 2) and np.isnan(path).all()
    with pytest.raises(ValueError, match="shapes"):
        recursive_forecast([(Overflowing("shapes do not match"), 1)], changes, 3)


def test_ridge_recovers_the_simulated_rule() -> None:
    """On 400 simulated months with one lag and no penalty, Ridge's one-month predictions follow
    the true rule (y' = 0.8 x, x' = 0.5 x) within the noise, in the data's own units although
    the fit standardizes features and target."""
    changes = np.diff(simulated_system(400), axis=0)
    settings = {"lags": 1, "alpha": 0.0}

    fitted = [fit_model(RIDGE, settings, changes, column) for column in range(2)]

    probe = np.array([[1.0, 0.0], [-2.0, 0.0]])
    assert fitted[0].predict(probe) == pytest.approx([0.5, -1.0], abs=0.15)
    assert fitted[1].predict(probe) == pytest.approx([0.8, -1.6], abs=0.05)


def test_lasso_without_a_penalty_is_least_squares() -> None:
    """Lasso with alpha 0 is fitted as a linear regression, so it matches Ridge with alpha 0:
    both are plain least squares."""
    changes = np.diff(simulated_system(120), axis=0)
    features, target = lag_features(changes, 2), changes[2:, 1]

    lasso = build_model(LASSO, {"lags": 2, "alpha": 0.0, "max_iter": 10}).fit(features, target)
    ridge = build_model(RIDGE, {"lags": 2, "alpha": 0.0}).fit(features, target)

    assert lasso.predict(features) == pytest.approx(ridge.predict(features), abs=1e-9)


def test_forest_and_xgboost_are_reproducible() -> None:
    """Two fits on the same data give identical predictions (fixed seed)."""
    changes = np.diff(simulated_system(150), axis=0)
    features, target = lag_features(changes, 3), changes[3:, 1]
    forest = {"lags": 3, "n_estimators": 50, "max_features": 0.5, "max_samples": 0.7}
    boosting = {"lags": 3, "n_estimators": 50, "subsample": 0.7, "colsample_bytree": 0.5}

    for model, settings in ((FOREST, forest), (XGBOOST, boosting)):
        first = build_model(model, settings).fit(features, target).predict(features)
        second = build_model(model, settings).fit(features, target).predict(features)
        assert np.array_equal(first, second)
