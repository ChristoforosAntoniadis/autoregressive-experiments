"""
Tests for the ready-made family (Prophet) in autoregressive_experiments.prophet_forecasting and
its tuning in autoregressive_experiments.tuning.
"""

import numpy as np
import pandas as pd

from autoregressive_experiments.prophet_forecasting import prophet_forecast, prophet_forecaster
from autoregressive_experiments.tuning import prophet_one_step

SETTINGS = {
    "growth": "linear",
    "changepoint_prior_scale": 0.05,
    "changepoint_range": 0.8,
    "seasonality_mode": "additive",
    "seasonality_prior_scale": 1.0,
    "yearly_seasonality": True,
}


def monthly_series(months: int) -> tuple[np.ndarray, np.ndarray]:
    """
    A trending, seasonal, positive monthly series with month-end dates.

    Parameters
    ----------
    months : int
        Number of months.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        The dates (month ends from 2005-01) and the levels: a line plus a yearly sine plus
        noise (seed 0), all above 1.
    """
    dates = pd.date_range("2005-01-31", periods=months, freq="ME").to_numpy()
    time = np.arange(months)
    noise = np.random.default_rng(0).normal(0, 0.02, months)
    return dates, 2 + 0.01 * time + 0.1 * np.sin(2 * np.pi * time / 12) + noise


def test_prophet_ignores_later_months_and_repeats_exactly() -> None:
    """Changing every month after the origin leaves the forecast and its spread unchanged, two
    runs agree exactly (seeded samples), and the spread is positive at every horizon."""
    dates, levels = monthly_series(100)
    origin = 80
    forecaster = prophet_forecaster(SETTINGS, dates, 200)

    before = forecaster(levels[: origin + 1], 6)
    again = forecaster(levels[: origin + 1], 6)
    changed = levels.copy()
    changed[origin + 1 :] += 50.0
    after = prophet_forecaster(SETTINGS, dates, 200)(changed[: origin + 1], 6)

    for key in ("forecast", "sd_constant"):
        assert np.array_equal(before[key], again[key])
        assert np.array_equal(before[key], after[key])
    assert np.all(before["sd_constant"] > 0)


def test_prophet_fold_errors_equal_fitting_each_month_on_earlier_months() -> None:
    """Each tuning error is the actual level minus Prophet's one-month forecast from the months
    before it, the same whether the refits run in parallel or one by one."""
    dates, levels = monthly_series(64)

    errors = prophet_one_step(pd.Series(dates), levels, SETTINGS, [(60, 62)])

    expected = [
        levels[month + 1]
        - prophet_forecast(dates[: month + 1], levels[: month + 1], SETTINGS, 1)[0][0]
        for month in (60, 61)
    ]
    assert errors.tolist() == expected
