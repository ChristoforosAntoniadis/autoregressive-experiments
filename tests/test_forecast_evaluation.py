"""
Tests for the out-of-sample forecast evaluation (step U6) in
autoregressive_experiments.forecast_evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from autoregressive_experiments.forecast_evaluation import (
    EXPONENTIAL_SMOOTHING,
    SYSTEMS,
    accuracy,
    combine_forecasts,
    convergence_failures,
    diebold_mariano,
    drift_forecast,
    exponential_smoothing_forecaster,
    interval_scores,
    lagged_exog,
    level_variances,
    mase_scale,
    moving_average_forecaster,
    naive_forecast,
    pinball_loss,
    rolling_forecasts,
    sarimax_forecaster,
    seasonal_naive_forecast,
    standardized_quantiles,
    system_forecast,
    system_forecaster,
    system_level_variances,
)

LEVELS = np.array([1.0, 2.0, 4.0, 7.0])  # monthly changes 1, 2, 3


def test_baselines_on_a_hand_made_series() -> None:
    """Naive repeats 7; drift adds the average change (7 - 1) / 3 = 2 per month; seasonal naive
    with a 2-month period repeats the last two changes, 2 then 3."""
    # Drift: 7 + 2 = 9, 9 + 2 = 11. Seasonal naive: 7 + 2 = 9, 9 + 3 = 12.
    assert naive_forecast(LEVELS, 2)["forecast"].tolist() == [7.0, 7.0]
    assert drift_forecast(LEVELS, 2)["forecast"].tolist() == [9.0, 11.0]
    assert seasonal_naive_forecast(LEVELS, 2, period=2)["forecast"].tolist() == [9.0, 12.0]


def test_moving_average_forecasts_the_mean_of_the_last_levels_at_every_horizon() -> None:
    """Last two levels 4 and 7: SMA(2) forecasts 5.5 for every month ahead."""
    # moving_average_forecaster(2) builds the forecaster; calling it forecasts 3 months.
    assert moving_average_forecaster(2)(LEVELS, 3)["forecast"].tolist() == [5.5, 5.5, 5.5]


def test_exponential_smoothing_variants_behave_as_their_definitions() -> None:
    """On a straight line 10, 11, ..., 69: single smoothing forecasts a flat path; Holt carries
    the slope of 1 forward; damped Holt still rises, by less each month; Holt-Winters on a line
    plus a 12-month pattern repeats that pattern."""
    line = np.arange(10.0, 70.0)
    single = exponential_smoothing_forecaster(EXPONENTIAL_SMOOTHING["single exp. smoothing"])
    holt = exponential_smoothing_forecaster(EXPONENTIAL_SMOOTHING["double exp. smoothing (Holt)"])
    damped = exponential_smoothing_forecaster(
        EXPONENTIAL_SMOOTHING["double exp. smoothing, damped"]
    )
    winters = exponential_smoothing_forecaster(
        EXPONENTIAL_SMOOTHING["triple exp. smoothing (Holt-Winters)"]
    )

    flat = single(line, 3)["forecast"]
    assert flat == pytest.approx(np.full(3, flat[0]))
    # The line ends at 69, so a slope of 1 gives 70, 71, 72.
    assert holt(line, 3)["forecast"] == pytest.approx([70.0, 71.0, 72.0], abs=1e-3)
    # Monthly steps from the last level: all positive and never growing (damping shrinks them).
    steps = np.diff(np.concatenate([[line[-1]], damped(line, 12)["forecast"]]))
    assert (steps > 0).all() and (np.diff(steps) <= 1e-9).all()
    # Line 0, 1, ..., 95 plus a 12-month pattern (8 years); the next year is 96..107 plus the
    # same pattern.
    pattern = np.tile([0.0, 3, -2, 1, 0, -1, 2, 0, -3, 1, 0, -1], 8)
    seasonal = np.arange(96.0) + pattern
    assert winters(seasonal, 12)["forecast"] == pytest.approx(
        np.arange(96.0, 108.0) + pattern[:12], abs=0.05
    )


def test_combine_forecasts_averages_two_models_at_the_same_origin_and_horizon() -> None:
    """Forecasts 2 and 4 average to 3; 1 and 5 to 3; a model's extra row has no partner."""
    # Pairs: origin 1 horizon 1 (a 2, b 4) and origin 1 horizon 2 (a 1, b 5); model a's
    # origin-2 row (forecast 8) has no b row, so it is dropped.
    rows = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b", "a"],
            "origin": [1, 1, 1, 1, 2],
            "horizon": [1, 2, 1, 2, 1],
            "target": [2, 3, 2, 3, 3],
            "actual": [9.0, 9, 9, 9, 9],
            "forecast": [2.0, 1, 4, 5, 8],
        }
    )

    combined = combine_forecasts(rows, "a", "b", "a + b")

    assert combined["forecast"].tolist() == [3.0, 3.0]
    assert combined["model"].unique().tolist() == ["a + b"]


def test_naive_spread_is_a_random_walk() -> None:
    """Changes 1, 2, 3 have standard deviation 1, so the spread is 1, then sqrt(2)."""
    assert naive_forecast(LEVELS, 2)["sd_constant"] == pytest.approx([1.0, np.sqrt(2)])


def test_level_variances_sum_the_error_weights() -> None:
    """psi 1, 0.5 give summed weights 1, 1.5. With unit shocks: h = 1 has 1^2 = 1; h = 2 has
    1.5^2 (the first shock, two months on) + 1^2 (the second) = 3.25. A random walk (psi 1, 0,
    0) with shock variance 2 grows by 2 a month: 2, 4, 6."""
    assert level_variances(np.array([1.0, 0.5]), np.ones(2)) == pytest.approx([1.0, 3.25])
    assert level_variances(np.array([1.0, 0, 0]), np.full(3, 2.0)) == pytest.approx([2, 4, 6])


def test_level_variances_weight_each_month_by_its_own_shock_variance() -> None:
    """psi 1, 0.5 with shock variances 1 then 4: h = 2 has 1.5^2 * 1 + 1^2 * 4 = 6.25."""
    assert level_variances(np.array([1.0, 0.5]), np.array([1.0, 4.0]))[1] == pytest.approx(6.25)


def test_system_level_variances_reduce_to_the_univariate_case() -> None:
    """One series with psi 1, 0.5 and shock variance 1: 1, then 1.5^2 + 1 = 3.25, as for
    level_variances."""
    # psi has shape [month, series, shock]: two months of 1 x 1 error-weight matrices.
    psi = np.array([[[1.0]], [[0.5]]])

    assert system_level_variances(psi, np.array([[1.0]]))[:, 0] == pytest.approx([1.0, 3.25])


def test_system_level_variances_of_white_noise_differences_grow_by_the_shock_variances() -> None:
    """No dynamics (psi = I, then zeros): each level's variance is h times its own shock
    variance, 2 and 1; the shocks' covariance of 0.5 doesn't enter the diagonals."""
    psi = np.stack([np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))])
    sigma = np.array([[2.0, 0.5], [0.5, 1.0]])

    assert system_level_variances(psi, sigma) == pytest.approx(np.array([[2, 1], [4, 2], [6, 3]]))


def two_random_walks(count: int = 150) -> np.ndarray:
    """Two independent standard-normal random walks, as the levels of a two-series system.

    Parameters
    ----------
    count : int, default 150
        Number of months.

    Returns
    -------
    numpy.ndarray
        Shape (count, 2), one column per series, seeded with 0.
    """
    return np.cumsum(np.random.default_rng(0).normal(size=(count, 2)), axis=0)


def test_system_forecaster_refuses_a_history_that_doesnt_match_the_months_handed_over() -> None:
    """The leakage guard: the target's history must be exactly the system's first months."""
    all_levels = two_random_walks()
    # SYSTEMS[0] is the VAR(5); the forecaster is for series 0.
    forecaster = system_forecaster(SYSTEMS[0], all_levels, target=0, cache={})

    with pytest.raises(AssertionError):
        forecaster(all_levels[:100, 1], 3)  # the other series' history


def test_both_series_share_one_system_fit_per_origin() -> None:
    """Forecasting each series from the same origin fits the system once; each forecaster
    returns its own series of that one fit."""
    all_levels = two_random_walks()
    # Keyed by (system, months of history, steps); holds the (mean, spread) of one fit.
    cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    first = system_forecaster(SYSTEMS[0], all_levels, target=0, cache=cache)
    second = system_forecaster(SYSTEMS[0], all_levels, target=1, cache=cache)

    output_first = first(all_levels[:100, 0], 3)
    output_second = second(all_levels[:100, 1], 3)
    mean, spread = system_forecast(SYSTEMS[0], all_levels[:100], 3)

    assert len(cache) == 1
    assert output_first["forecast"] == pytest.approx(mean[:, 0])
    assert output_second["sd_constant"] == pytest.approx(spread[:, 1])


def test_standardized_quantiles_for_normal_and_unit_variance_student_t() -> None:
    """Normal 97.5%: 1.960. Student-t with 5 degrees of freedom: 2.571 rescaled by
    sqrt(3 / 5) to unit variance, 1.991. Both are symmetric, with the median at 0."""
    # nu = inf selects the normal. A t with nu degrees of freedom has variance nu / (nu - 2).
    normal = standardized_quantiles(np.inf)
    student = standardized_quantiles(5.0)

    # The quantiles are 2.5, 10, 50, 90 and 97.5%: index -1 is 97.5%, index 2 the median.
    assert normal[-1] == pytest.approx(1.960, abs=1e-3)
    assert student[-1] == pytest.approx(1.991, abs=1e-3)
    assert normal[2] == student[2] == 0
    assert normal[0] == pytest.approx(-normal[-1])


def test_pinball_loss_by_hand() -> None:
    """Actual 0 against quantile forecasts -2, -1, 0, 1, 2: the losses are 0.025 * 2, 0.1 * 1, 0,
    0.1 * 1 and 0.025 * 2, so the mean over 5 quantiles is 0.3 / 5 = 0.06."""
    # Pinball loss: q * (actual - forecast) above the quantile, (1 - q) * (forecast - actual)
    # below it; for the 97.5% quantile at 2 that is (1 - 0.975) * 2 = 0.025 * 2.
    quantiles = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    assert pinball_loss(np.array([0.0]), quantiles) == pytest.approx(0.06)


def test_interval_scores_by_hand() -> None:
    """Both forecasts have quantiles -2, -1, 0, 1, 2. Actual 0 is inside both intervals, actual
    5 inside neither: coverage 0.5 each; widths 2 (80%) and 4 (95%); 0 < 2 counts as below the
    97.5% quantile, 5 doesn't: 0.5; nothing is below the median."""
    # Columns are the 2.5 / 10 / 50 / 90 / 97.5% quantiles: the 80% interval is [-1, 1] and
    # the 95% interval [-2, 2]. Actual 0 equals the median, which isn't "below" it.
    quantiles = np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], (2, 1))

    scores = interval_scores(np.array([0.0, 5.0]), quantiles)

    assert scores["coverage_80"] == scores["coverage_95"] == 0.5
    assert (scores["width_80"], scores["width_95"]) == (2.0, 4.0)
    assert scores["below_97.5"] == 0.5
    assert scores["below_50"] == 0.0


def test_convergence_failures_list_each_failed_origin_once() -> None:
    """Two horizons per origin; the 2020-02 refit failed, so it is listed once."""
    origins = pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"])
    forecasts = pd.DataFrame(
        {"model": "m", "origin": origins, "converged": [True, True, False, False]}
    )

    assert convergence_failures(forecasts) == {"m": ["2020-02"]}


def test_seasonal_naive_refuses_to_forecast_past_one_period() -> None:
    """With a 2-month period there are only 2 past changes to repeat, so 3 steps are refused."""
    with pytest.raises(ValueError, match="at most 2"):
        seasonal_naive_forecast(LEVELS, 3, period=2)


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
    # Harvey-Leybourne-Newbold factor: sqrt((n + 1 - 2h + h(h - 1) / n) / n), n = 4, h = 1.
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


def test_rolling_forecasts_only_use_data_up_to_each_origin() -> None:
    """Levels 0, 1, 2, ... make leakage visible: a naive forecast from origin k must be exactly
    k, and its target k + h, so every error equals the horizon."""
    levels = np.arange(20, dtype=float)
    dates = pd.Series(pd.date_range("2020-01-31", periods=20, freq="ME"))

    forecasts = rolling_forecasts(dates, levels, 10, 15, {"naive": naive_forecast})

    # Turn each origin date back into its index k, which is also its level.
    origins = forecasts["origin"].map(lambda date: dates[dates == date].index[0])
    assert (forecasts["forecast"] == origins).all()
    assert (forecasts["actual"] - forecasts["forecast"] == forecasts["horizon"]).all()
    # Origins 10 to 14; targets never pass index 15, so origin 14 forecasts 1 step only.
    assert forecasts["origin"].nunique() == 5
    assert forecasts["target"].max() == dates[15]
    assert len(forecasts) == 5 + 4 + 3 + 2 + 1


def test_lagged_exog_by_hand() -> None:
    """Changes 1..5 with 2 lags: months 2, 3 and 4 get rows (most recent change first); months
    0 and 1 have no full history and get none."""
    matrix = lagged_exog(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2)

    # Month 2 sees changes 2 and 1, month 3 sees 3 and 2, month 4 sees 4 and 3; never its own.
    assert matrix.tolist() == [[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]]


def test_sarimax_forecaster_ignores_everything_after_the_origin() -> None:
    """Changing every month after the origin, in both series, leaves the forecast unchanged:
    the regressors for later months come from the other series' forecast, not its future.

    A history that doesn't match the harness's levels trips the leakage guard.
    """
    rng = np.random.default_rng(0)
    # The second series follows the first's last change (x drives y), so the regressor matters.
    x = rng.normal(size=80)
    y = np.concatenate([[0.0], 0.8 * x[:-1]]) + 0.1 * rng.normal(size=80)
    all_levels = np.column_stack([x.cumsum(), y.cumsum()])
    ar1 = {"p": 1, "q": 0, "P": 0, "Q": 0}
    origin = 60

    forecaster = sarimax_forecaster(ar1, ar1, all_levels, 1, 3, {})
    before = forecaster(all_levels[: origin + 1, 1], 6)["forecast"]
    changed = all_levels.copy()
    changed[origin + 1 :] += 100.0
    after = sarimax_forecaster(ar1, ar1, changed, 1, 3, {})(changed[: origin + 1, 1], 6)

    assert np.array_equal(before, after["forecast"])
    with pytest.raises(AssertionError):
        forecaster(all_levels[: origin + 1, 0], 6)
