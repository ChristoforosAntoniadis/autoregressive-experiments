"""
Tests for Prophet in autoregressive_experiments.prophet_forecasting, as the econometric family
uses it through autoregressive_experiments.model_forecasts.
"""

import numpy as np
import pandas as pd

from autoregressive_experiments.model_forecasts import series_path
from autoregressive_experiments.prophet_forecasting import PROPHET, prophet_forecast

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


def test_prophet_repeats_exactly_and_follows_the_trend() -> None:
    """Two fits on the same 80 months agree exactly. The forecasts for months 80 and 91 differ by
    the trend, 11 x 0.01 = 0.11, plus the sine's change from month 80 (phase 8 of 12,
    sin = -0.866) to month 91 (phase 7, sin = -0.5): 0.1 x 0.366 = 0.037, so about 0.147."""
    dates, levels = monthly_series(80)

    first = prophet_forecast(dates, levels, SETTINGS, 12)
    second = prophet_forecast(dates, levels, SETTINGS, 12)

    assert np.array_equal(first, second)
    assert abs(first[11] - first[0] - 0.147) < 0.03


def test_prophet_through_series_path_uses_only_the_history_it_is_given() -> None:
    """series_path hands Prophet the dates of the months it forecasts from, so the history up to
    month 60 gives the same forecast whatever the dates array holds after it."""
    dates, levels = monthly_series(100)
    history = np.column_stack([levels, levels])[:61]

    path = series_path(PROPHET, SETTINGS, history, dates, 0, 3)

    assert np.array_equal(path, prophet_forecast(dates[:61], levels[:61], SETTINGS, 3))
