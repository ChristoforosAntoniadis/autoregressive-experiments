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
def test_assess_stationarity_combines_both_tests(adf_p_value, kpss_p_value, expected):
    """ADF must reject and KPSS must not reject for a 'stationary' verdict, and vice versa."""
    assert assess_stationarity(adf_p_value, kpss_p_value) == expected


def test_stationarity_report_separates_white_noise_from_random_walk():
    """White noise is reported as stationary and its cumulative sum (a random walk) as not."""
    noise = np.random.default_rng(0).normal(size=300)
    dataframe = pd.DataFrame({VALUE_COLUMNS[0]: noise, VALUE_COLUMNS[1]: noise.cumsum()})

    assessments = [
        line for line in stationarity_report(dataframe).splitlines() if "Assessment" in line
    ]

    assert assessments == [
        "  Assessment: Evidence supports stationarity (tests agree).",
        "  Assessment: Evidence supports non-stationarity (tests agree).",
    ]
