"""
Tests for the forecast scores (autoregressive_experiments.metrics), the models' forecasts
(autoregressive_experiments.model_forecasts) and the champions' test
(autoregressive_experiments.forecast_evaluation).
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from autoregressive_experiments.features import lagged_exog
from autoregressive_experiments.forecast_evaluation import add_bands, forecast_rows, score_model
from autoregressive_experiments.metrics import (
    accuracy,
    diebold_mariano,
    empirical_band,
    interval_scores,
    mase_scale,
    pinball_loss,
)
from autoregressive_experiments.model_forecasts import (
    COMBINATION,
    EXPONENTIAL,
    NAIVE,
    SARIMAX,
    SMA,
    VAR_MODEL,
    joint_paths,
    sarima_name,
    series_path,
)

LEVELS = np.array([1.0, 2.0, 4.0, 7.0])  # monthly changes 1, 2, 3
NO_DATES = np.array([])


def smoothing(**chosen: Any) -> dict[str, Any]:
    """
    Exponential-smoothing settings in the tuned form: no trend, no season, unless chosen.

    Parameters
    ----------
    **chosen : Any
        The settings that differ from single exponential smoothing, e.g. trend="add".

    Returns
    -------
    dict of str to Any
        All five settings the tuning chooses.
    """
    single = {
        "trend": None,
        "damped_trend": False,
        "seasonal": None,
        "use_boxcox": False,
        "initialization_method": "estimated",
    }
    return single | chosen


def one_series(levels: np.ndarray) -> np.ndarray:
    """
    A history of one series in the two-column layout the forecasts take.

    Parameters
    ----------
    levels : np.ndarray
        Levels, shape (months,).

    Returns
    -------
    np.ndarray
        Shape (months, 2): the levels, then a copy as the second series.
    """
    return np.column_stack([levels, levels])


def test_naive_and_moving_average_by_hand() -> None:
    """Naive repeats the last level, 7; SMA(2) forecasts the mean of 4 and 7, 5.5, every month."""
    history = one_series(LEVELS)

    assert series_path(NAIVE, {}, history, NO_DATES, 0, 2).tolist() == [7.0, 7.0]
    assert series_path(SMA, {"window": 2}, history, NO_DATES, 0, 3).tolist() == [5.5] * 3


def test_exponential_smoothing_forms_behave_as_their_definitions() -> None:
    """On a straight line 10, 11, ..., 69: single smoothing forecasts a flat path; Holt carries
    the slope of 1 forward; damped Holt still rises, by less each month; Holt-Winters on a line
    plus a 12-month pattern repeats that pattern."""
    line = one_series(np.arange(10.0, 70.0))
    pattern = np.tile([0.0, 3, -2, 1, 0, -1, 2, 0, -3, 1, 0, -1], 8)
    seasonal = one_series(np.arange(96.0) + pattern)

    flat = series_path(EXPONENTIAL, smoothing(), line, NO_DATES, 0, 3)
    holt = series_path(EXPONENTIAL, smoothing(trend="add"), line, NO_DATES, 0, 3)
    damped = series_path(
        EXPONENTIAL, smoothing(trend="add", damped_trend=True), line, NO_DATES, 0, 12
    )
    winters = series_path(
        EXPONENTIAL, smoothing(trend="add", seasonal="add"), seasonal, NO_DATES, 0, 12
    )

    assert flat == pytest.approx(np.full(3, flat[0]))
    # The line ends at 69, so a slope of 1 gives 70, 71, 72.
    assert holt == pytest.approx([70.0, 71.0, 72.0], abs=1e-3)
    steps = np.diff(np.concatenate([[69.0], damped]))
    assert (steps > 0).all() and (np.diff(steps) <= 1e-9).all()
    assert winters == pytest.approx(np.arange(96.0, 108.0) + pattern[:12], abs=0.05)


def test_the_average_weighs_sarima_and_naive_by_hand() -> None:
    """Weight 0 is naive itself; weight 0.25 is a quarter of the way from naive to the SARIMA
    (weight 1)."""
    history = one_series(np.cumsum(np.random.default_rng(0).normal(0.1, 1.0, 80)))
    sarima = [{"p": 1, "q": 0, "P": 0, "Q": 0, "trend": "c"}] * 2
    weights = (0.0, 0.25, 1.0)

    naive, quarter, full = (
        series_path(COMBINATION, {"weight": w}, history, NO_DATES, 0, 3, sarima) for w in weights
    )

    assert naive == pytest.approx(np.full(3, history[-1, 0]))
    assert quarter == pytest.approx(0.25 * full + 0.75 * naive)


def test_sarima_name_by_hand() -> None:
    """Orders (2, 2) and seasonal (1, 0) with d = 1, D = 0 and a 12-month season; no drift is
    named, a drift is the default and isn't."""
    orders = {"p": 2, "q": 2, "P": 1, "Q": 0}

    assert sarima_name(orders | {"trend": "n"}) == "SARIMA(2,1,2)x(1,0,0,12) no drift"
    assert sarima_name(orders | {"trend": "c"}) == "SARIMA(2,1,2)x(1,0,0,12)"


def test_forecasts_ignore_everything_after_the_origin() -> None:
    """A SARIMAX fitted on the months up to an origin gives the same forecasts however the later
    months change (the regressors for later months come from forecasts); a VAR forecasts both
    series."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    y = np.concatenate([[0.0], 0.8 * x[:-1]]) + 0.1 * rng.normal(size=80)
    levels = np.column_stack([x.cumsum(), y.cumsum()])
    changed = levels.copy()
    changed[61:] += 100.0
    sarima = [{"p": 1, "q": 0, "P": 0, "Q": 0, "trend": "c"}] * 2

    before = series_path(SARIMAX, {"lags": 3}, levels[:61], NO_DATES, 1, 6, sarima)
    after = series_path(SARIMAX, {"lags": 3}, changed[:61], NO_DATES, 1, 6, sarima)
    system = joint_paths(VAR_MODEL, {"lags": 2, "trend": "c"}, levels[:61], 4)

    assert np.array_equal(before, after)
    assert system.shape == (4, 2)


def test_lagged_exog_by_hand() -> None:
    """Changes 1..5 with 2 lags: months 2, 3 and 4 get rows (most recent change first); months
    0 and 1 have no full history and get none."""
    matrix = lagged_exog(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2)

    assert matrix.tolist() == [[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]]


def test_mase_scale_is_the_mean_absolute_monthly_change() -> None:
    """|1| + |2| + |3| over 3 months = 2."""
    assert mase_scale(LEVELS) == pytest.approx(2.0)


def test_accuracy_metrics_on_two_hand_computed_errors() -> None:
    """Actual 2 and 4, forecast 1 and 7: errors 1 and -3. MAE (1 + 3) / 2 = 2; MSE (1 + 9) / 2
    = 5; RMSE sqrt(5); MAPE 100 * (1/2 + 3/4) / 2 = 62.5; MASE with scale 0.5 is 2 / 0.5 = 4."""
    scores = accuracy(np.array([2.0, 4.0]), np.array([1.0, 7.0]), scale=0.5)

    assert scores == pytest.approx(
        {"MAE": 2.0, "MSE": 5.0, "RMSE": np.sqrt(5.0), "MAPE": 62.5, "MASE": 4.0}
    )


def test_diebold_mariano_one_step_by_hand() -> None:
    """Loss differences 1, 2, 3, 4: mean 2.5, variance (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25,
    so DM = 2.5 / sqrt(1.25 / 4) = 4.472. The small-sample factor sqrt((4 + 1 - 2) / 4) = 0.866
    gives 3.873; positive, because the first model has the larger loss."""
    statistic, p_value = diebold_mariano(np.array([1.0, 2, 3, 4]), np.zeros(4), horizon=1)

    assert statistic == pytest.approx(3.873, abs=1e-3)
    assert 0.02 < p_value < 0.05  # t with 3 degrees of freedom


def test_diebold_mariano_two_step_falls_back_when_the_long_run_variance_turns_negative() -> None:
    """Differences 1, 3, 1, 3: deviations -1, 1, -1, 1; lag-0 autocovariance 1, lag-1
    autocovariance -3/4, so 1 + 2(-0.75) = -0.5 < 0 and the lag-0 variance 1 is used:
    DM = 2 / sqrt(1 / 4) = 4, times sqrt((4 + 1 - 4 + 2 / 4) / 4) = 0.612, so 2.449."""
    statistic, _ = diebold_mariano(np.array([1.0, 3, 1, 3]), np.zeros(4), horizon=2)

    assert statistic == pytest.approx(2.449, abs=1e-3)


def test_diebold_mariano_finds_no_difference_between_identical_losses() -> None:
    """Identical losses give all-zero differences with zero variance: statistic 0, p-value 1."""
    losses = np.array([0.5, 1.0, 0.2])
    assert diebold_mariano(losses, losses, horizon=1) == (0.0, 1.0)


def test_empirical_band_by_hand() -> None:
    """Errors 0, 1, ..., 40: numpy's linear quantiles put the 2.5% quantile at 0.025 x 40 = 1,
    the 10% at 4, the median at 20, the 90% at 36 and the 97.5% at 39; no errors give NaN."""
    assert empirical_band(np.arange(41.0)).tolist() == pytest.approx([1, 4, 20, 36, 39])
    assert np.isnan(empirical_band(np.array([]))).all()


def test_pinball_loss_by_hand() -> None:
    """Actual 0 against quantile forecasts -2, -1, 0, 1, 2: the losses are 0.025 * 2, 0.1 * 1, 0,
    0.1 * 1 and 0.025 * 2, so the mean over 5 quantiles is 0.3 / 5 = 0.06."""
    quantiles = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    assert pinball_loss(np.array([0.0]), quantiles) == pytest.approx(0.06)


def test_interval_scores_by_hand() -> None:
    """Both forecasts have quantiles -2, -1, 0, 1, 2. Actual 0 is inside both intervals, actual
    5 inside neither: coverage 0.5 each; widths 2 (80%) and 4 (95%); 0 < 2 counts as below the
    97.5% quantile, 5 doesn't: 0.5; nothing is below the median."""
    quantiles = np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], (2, 1))

    scores = interval_scores(np.array([0.0, 5.0]), quantiles)

    assert scores["coverage_80"] == scores["coverage_95"] == 0.5
    assert (scores["width_80"], scores["width_95"]) == (2.0, 4.0)
    assert scores["below_97.5"] == 0.5
    assert scores["below_50"] == 0.0


def test_forecast_rows_by_hand() -> None:
    """Origins 1 and 2 forecasting 2 and 1 months of levels 0..4: three rows, each with its target
    month and error (actual - forecast)."""
    actual = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    dates = pd.date_range("2020-01-31", periods=5, freq="ME").to_numpy()

    rows = forecast_rows([np.array([1.5, 3.5]), np.array([2.0])], [1, 2], actual, dates)

    assert rows["target_index"].tolist() == [2, 3, 3]
    assert rows["error"].tolist() == [0.5, -0.5, 1.0]


def test_bands_use_only_errors_already_known_at_the_origin() -> None:
    """One-month forecasts from origins 0..5 with errors 0, 1, ..., 5. The band at origin 3 may
    use only targets up to month 3, the errors 0, 1 and 2 (median 1): its median is the forecast
    + 1, whatever later errors are."""
    rows = pd.DataFrame(
        {
            "origin_index": range(6),
            "target_index": range(1, 7),
            "horizon": 1,
            "forecast": 10.0,
            "error": np.arange(6.0),
        }
    )

    banded = add_bands(rows, first_origin=3)

    assert banded["origin_index"].tolist() == [3, 4, 5]
    assert banded.loc[banded["origin_index"] == 3, "q50"].item() == pytest.approx(11.0)


def test_score_model_tests_against_naive_but_not_naive_against_itself() -> None:
    """Scored against itself, naive gets no Diebold-Mariano test (NaN); a different model gets
    one, and its band coverage lies between 0 and 1."""
    rng = np.random.default_rng(0)
    actual = rng.normal(size=480)
    naive = pd.DataFrame(
        {
            "origin_index": np.repeat(np.arange(40), 12),
            "horizon": np.tile(np.arange(1, 13), 40),
            "actual": actual,
            "forecast": 0.0,
            "error": actual,
        }
    )
    naive[["q2.5", "q10", "q50", "q90", "q97.5"]] = np.tile([-2.0, -1.3, 0.0, 1.3, 2.0], (480, 1))
    model = naive.assign(forecast=0.3, error=actual - 0.3)

    own = score_model(naive, naive, scale=1.0)
    other = score_model(model, naive, scale=1.0)

    assert np.isnan(own[0]["DM_absolute_p"])
    assert np.isfinite(other[0]["DM_absolute_p"])
    assert 0 <= other[0]["coverage_80"] <= 1
