"""
Tests for the asymmetric GARCH step (U5c) in autoregressive_experiments.asymmetric_garch, and
for the EGARCH / GJR-GARCH variance paths it adds to volatility_models.
"""

import numpy as np
import pandas as pd
import pytest

from autoregressive_experiments.asymmetric_garch import (
    EGARCH_T,
    GJR_T,
    asymmetric_check,
    persistence,
)
from autoregressive_experiments.volatility_models import garch_variance_path


def simulate_gjr(count: int, gamma: float) -> np.ndarray:
    """GJR-GARCH(1,1): variance_t = 0.1 + (0.05 + gamma [e_(t-1) < 0]) e_(t-1)^2 + 0.6
    variance_(t-1), normal shocks."""
    shocks = np.random.default_rng(0).normal(size=count)
    values = np.zeros(count)
    variance = 0.1 / (1 - 0.05 - gamma / 2 - 0.6)
    for t in range(count):
        if t:
            previous = values[t - 1]
            variance = 0.1 + (0.05 + gamma * (previous < 0)) * previous**2 + 0.6 * variance
        values[t] = np.sqrt(variance) * shocks[t]
    return values


def test_persistence_by_hand():
    """GJR: alpha + gamma / 2 + beta = 0.1 + 0.2 / 2 + 0.6 = 0.8; EGARCH: beta = 0.9."""
    gjr = pd.Series({"alpha[1]": 0.1, "gamma[1]": 0.2, "beta[1]": 0.6})
    egarch = pd.Series({"alpha[1]": 0.1, "gamma[1]": 0.2, "beta[1]": 0.9})

    assert persistence(GJR_T, gjr) == pytest.approx(0.8)
    assert persistence(EGARCH_T, egarch) == pytest.approx(0.9)


def test_gjr_finds_simulated_asymmetry_and_beats_symmetric_garch():
    """With falls raising variance by 0.3 more, GJR's gamma is positive and significant, and
    BIC prefers it to the symmetric GARCH."""
    result = asymmetric_check(simulate_gjr(2000, gamma=0.3))

    assert result[GJR_T]["gamma"] > 0
    assert result[GJR_T]["gamma_p"] < 0.01
    assert result[GJR_T]["delta_bic"] < 0


def test_egarch_variance_path_is_reproducible_and_in_the_units_it_was_given():
    """EGARCH's multi-step variances are simulated with a fixed seed: two runs match exactly,
    and multiplying the residuals by 10 multiplies them by about 100 (simulation noise aside)."""
    values = simulate_gjr(600, gamma=0.3) * 0.05

    first, nu = garch_variance_path(values, EGARCH_T, 6)
    second, _ = garch_variance_path(values, EGARCH_T, 6)
    larger, _ = garch_variance_path(10 * values, EGARCH_T, 6)

    assert np.array_equal(first, second)
    assert np.isfinite(nu)
    assert larger == pytest.approx(100 * first, rel=0.02)
