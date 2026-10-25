"""
Tests for Granger causality, impulse responses and the variance decomposition (V7, V8) in
autoregressive_experiments.var_analysis.
"""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

from autoregressive_experiments.var_analysis import granger_tests, variance_decomposition


def x_drives_y(count: int = 400) -> pd.DataFrame:
    """y depends on last month's x; x ignores y."""
    rng = np.random.default_rng(0)
    x, y = np.zeros(count), np.zeros(count)
    for t in range(1, count):
        x[t] = 0.5 * x[t - 1] + rng.normal()
        y[t] = 0.3 * y[t - 1] + 0.6 * x[t - 1] + rng.normal()
    return pd.DataFrame({"x": x, "y": y})


def test_granger_finds_x_driving_y_and_not_the_reverse():
    """Both the VAR's Wald test and the VECM's F-test see x -> y; neither sees y -> x."""
    differences = x_drives_y()
    levels = differences.cumsum()
    systems = {
        "VAR(5)": VAR(differences).fit(2, trend="c"),
        "VECM(5)": VECM(levels, k_ar_diff=2, coint_rank=1, deterministic="co").fit(),
    }

    table = granger_tests(systems, ["x", "y"]).set_index(["model", "causing"])

    assert table.loc[("VAR(5)", "x"), "p_value"] < 0.001
    assert table.loc[("VAR(5)", "y"), "p_value"] > 0.05
    assert table.loc[("VECM(5)", "x"), "p_value"] < 0.001


def test_variance_decomposition_by_hand():
    """First-month responses [[1, 0], [0.5, 1]]: series 0 is all its own shock; series 1 gets
    0.5^2 = 0.25 from shock 0 and 1 from its own, so 0.25 / 1.25 = 20% and 80%. Shares sum to 1."""
    responses = np.array([[[1.0, 0.0], [0.5, 1.0]]])

    shares = variance_decomposition(responses)

    assert shares[0, 0] == pytest.approx([1.0, 0.0])
    assert shares[0, 1] == pytest.approx([0.2, 0.8])
    assert shares.sum(axis=2) == pytest.approx(np.ones((1, 2)))


def test_variance_decomposition_matches_statsmodels_on_a_var():
    """On the VAR's own (differences) responses, the shares equal statsmodels' fevd()."""
    fitted = VAR(x_drives_y()).fit(2, trend="c")
    periods = 12

    shares = variance_decomposition(fitted.irf(periods).orth_irfs)
    expected = fitted.fevd(periods).decomp  # [variable, month, shock]

    assert shares[:periods] == pytest.approx(np.transpose(expected, (1, 0, 2)))


def test_cumulative_responses_of_a_hand_built_var1():
    """VAR(1) with A = [[0.5, 0], [0.2, 0.5]] and identity shock covariance: the response at
    month s is A^s, so the cumulative level response is I + A + A^2 + ..."""
    coefficients = np.array([[0.5, 0.0], [0.2, 0.5]])
    rng = np.random.default_rng(0)
    values = np.zeros((20000, 2))
    for t in range(1, 20000):
        values[t] = coefficients @ values[t - 1] + rng.normal(size=2)
    fitted = VAR(pd.DataFrame(values, columns=["a", "b"])).fit(1, trend="n")

    cumulative = fitted.irf(3).orth_cum_effects
    expected = sum(np.linalg.matrix_power(coefficients, s) for s in range(4))

    assert cumulative[3] == pytest.approx(expected, abs=0.05)
