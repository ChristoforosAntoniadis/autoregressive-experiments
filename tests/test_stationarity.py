"""
Tests for the ADF / KPSS stationarity check in autoregressive_experiments.diagnostics.
"""

import numpy as np
import pandas as pd
import pytest

from autoregressive_experiments.diagnostics import (
    VALUE_COLUMNS,
    assess_stationarity,
    stationarity_report,
)


@pytest.mark.parametrize(
    ("adf_p_value", "kpss_p_value", "expected"),
    [
        (0.01, 0.10, "Evidence supports stationarity (tests agree)."),
        (0.40, 0.01, "Evidence supports non-stationarity (tests agree)."),
        (0.01, 0.01, "Tests conflict; treat the stationarity result as inconclusive."),
        (0.40, 0.10, "Tests conflict; treat the stationarity result as inconclusive."),
        # p = 0.05 is not below the 5% level for ADF, and does not reject KPSS's null.
        (0.05, 0.05, "Tests conflict; treat the stationarity result as inconclusive."),
    ],
)
def test_assess_stationarity_combines_both_tests(
    adf_p_value: float, kpss_p_value: float, expected: str
) -> None:
    """ADF must reject and KPSS must not reject for a 'stationary' verdict, and vice versa.

    ADF's null is a unit root (a small p-value means stationary); KPSS's null is stationarity
    (a small p-value means non-stationary). The verdict is only firm when both agree.
    """
    assert assess_stationarity(adf_p_value, kpss_p_value) == expected


def test_stationarity_report_separates_white_noise_from_random_walk() -> None:
    """White noise is reported as stationary and its cumulative sum (a random walk) as not."""
    noise = np.random.default_rng(0).normal(size=300)
    dataframe = pd.DataFrame({VALUE_COLUMNS[0]: noise, VALUE_COLUMNS[1]: noise.cumsum()})

    # One assessment line per column, in VALUE_COLUMNS order.
    assert assessments(stationarity_report(dataframe, "test data", regression="c")) == [
        "  Assessment: Evidence supports stationarity (tests agree).",
        "  Assessment: Evidence supports non-stationarity (tests agree).",
    ]


def test_trend_regression_treats_a_noisy_linear_trend_as_trend_stationary() -> None:
    """A trend plus noise has no unit root: with "ct" it is stationary around its trend."""
    noise = np.random.default_rng(0).normal(size=300)
    trend = 0.05 * np.arange(300) + noise
    dataframe = pd.DataFrame({VALUE_COLUMNS[0]: trend, VALUE_COLUMNS[1]: noise.cumsum()})

    report = stationarity_report(dataframe, "test data", regression="ct")

    assert "Both tests use a constant and a linear trend" in report
    assert assessments(report) == [
        "  Assessment: Evidence supports stationarity (tests agree).",
        "  Assessment: Evidence supports non-stationarity (tests agree).",
    ]


def assessments(report: str) -> list[str]:
    """The verdict lines of a stationarity report.

    Parameters
    ----------
    report : str
        The text returned by stationarity_report.

    Returns
    -------
    list of str
        Every line that contains "Assessment", in report order (one per series).
    """
    return [line for line in report.splitlines() if "Assessment" in line]
