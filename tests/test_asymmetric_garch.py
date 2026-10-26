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
    variance_(t-1), normal shocks.

    Parameters
    ----------
    count : int
        Number of draws to simulate.
    gamma : float
        Extra ARCH weight on negative shocks (the asymmetry); 0 gives a symmetric GARCH(1,1).

    Returns
    -------
    numpy.ndarray
        The simulated series, shape (count,), seeded with 0 so every run is the same.

    Notes
    -----
    The first draw starts at the unconditional variance 0.1 / (1 - 0.05 - gamma / 2 - 0.6):
    with symmetric normal shocks, half of them are negative, so gamma counts half.
    """
    shocks = np.random.default_rng(0).normal(size=count)
    values = np.zeros(count)
    variance = 0.1 / (1 - 0.05 - gamma / 2 - 0.6)
    for t in range(count):
        if t:
            previous = values[t - 1]
            variance = 0.1 + (0.05 + gamma * (previous < 0)) * previous**2 + 0.6 * variance
        values[t] = np.sqrt(variance) * shocks[t]
    return values


def test_persistence_by_hand() -> None:
    """GJR: alpha + gamma / 2 + beta = 0.1 + 0.2 / 2 + 0.6 = 0.8; EGARCH: beta = 0.9."""
    gjr = pd.Series({"alpha[1]": 0.1, "gamma[1]": 0.2, "beta[1]": 0.6})
    egarch = pd.Series({"alpha[1]": 0.1, "gamma[1]": 0.2, "beta[1]": 0.9})

    assert persistence(GJR_T, gjr) == pytest.approx(0.8)
    assert persistence(EGARCH_T, egarch) == pytest.approx(0.9)


def test_gjr_finds_simulated_asymmetry_and_beats_symmetric_garch() -> None:
    """With falls raising variance by 0.3 more, GJR's gamma is positive and significant, and
    BIC prefers it to the symmetric GARCH."""
    # Only the fit and BIC are tested here, so the bootstrap checks are skipped.
    result = asymmetric_check(simulate_gjr(2000, gamma=0.3), replications=0)

    assert result[GJR_T]["gamma"] > 0
    assert result[GJR_T]["gamma_p"] < 0.01
    # delta_bic is GJR's BIC minus the symmetric GARCH's: negative means GJR is preferred.
    assert result[GJR_T]["delta_bic"] < 0


def test_egarch_variance_path_is_reproducible_and_in_the_units_it_was_given() -> None:
    """EGARCH's multi-step variances are simulated with a fixed seed: two runs match exactly,
    and multiplying the residuals by 10 multiplies them by about 100 (simulation noise aside)."""
    values = simulate_gjr(600, gamma=0.3) * 0.05

    first, nu = garch_variance_path(values, EGARCH_T, 6)
    second, _ = garch_variance_path(values, EGARCH_T, 6)
    larger, _ = garch_variance_path(10 * values, EGARCH_T, 6)

    assert np.array_equal(first, second)
    assert np.isfinite(nu)
    # Variance scales with the square of the data: 10^2 = 100.
    assert larger == pytest.approx(100 * first, rel=0.02)
