"""
Tests for picking significant ACF / PACF lags in autoregressive_experiments.diagnostics.
"""

import numpy as np

from autoregressive_experiments.diagnostics import significant_lags


def test_significant_lags_keeps_intervals_that_exclude_zero() -> None:
    """Row 0 (lag 0) is skipped; lags 1 and 3 lie wholly above or below zero."""
    # One row per lag, [lower, upper] bound of the confidence interval, as statsmodels returns.
    intervals = np.array(
        [
            [1.0, 1.0],  # lag 0: always 1, never reported
            [0.2, 0.6],  # lag 1: above zero
            [-0.1, 0.3],  # lag 2: contains zero
            [-0.5, -0.1],  # lag 3: below zero
            [0.0, 0.4],  # lag 4: touches zero, not significant
        ]
    )

    assert significant_lags(intervals) == [1, 3]


def test_significant_lags_is_empty_when_every_interval_contains_zero() -> None:
    """No lag is significant if every interval straddles zero."""
    intervals = np.array([[1.0, 1.0], [-0.2, 0.2], [-0.1, 0.1]])

    assert significant_lags(intervals) == []
