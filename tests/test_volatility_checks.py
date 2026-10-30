"""
Tests for the volatility checks in autoregressive_experiments.volatility_checks.
"""

import numpy as np

from autoregressive_experiments.volatility_checks import (
    standardize_by_regime,
    variance_break,
    volatility_check,
)


def break_series() -> np.ndarray:
    """200 independent draws: standard deviation 1 for the first 100, then 5.

    Returns
    -------
    numpy.ndarray
        Shape (200,), seeded with 0: one variance break at index 100 and no clustering.
    """
    noise = np.random.default_rng(0).normal(size=200)
    return np.concatenate([noise[:100], 5 * noise[100:]])


def garch_series() -> np.ndarray:
    """1,000 draws of GARCH(1,1): variance_t = 0.1 + 0.3 e_(t-1)^2 + 0.6 variance_(t-1).

    Returns
    -------
    numpy.ndarray
        Shape (1000,), normal shocks seeded with 0.

    Notes
    -----
    The first draw starts at the unconditional variance 0.1 / (1 - 0.3 - 0.6) = 1.
    """
    shocks = np.random.default_rng(0).normal(size=1000)
    values = np.zeros(1000)
    variance = 0.1 / (1 - 0.3 - 0.6)
    for t in range(1000):
        if t:
            variance = 0.1 + 0.3 * values[t - 1] ** 2 + 0.6 * variance
        values[t] = np.sqrt(variance) * shocks[t]
    return values


def test_standardize_by_regime_divides_each_regime_by_its_own_root_mean_square() -> None:
    """[1, -1] has root mean square 1 and [2, -2] has 2, so both become [1, -1]."""
    # The second argument is the break index: the second regime starts at position 2.
    assert standardize_by_regime(np.array([1.0, -1.0, 2.0, -2.0]), 2).tolist() == [
        1.0,
        -1.0,
        1.0,
        -1.0,
    ]


def test_variance_break_finds_where_the_standard_deviation_jumps() -> None:
    """The jump from 1 to 5 is at index 100; the search lands within a few months of it."""
    assert abs(variance_break(break_series()) - 100) <= 3


def test_independent_noise_has_no_arch_effects() -> None:
    """Independent normal noise has constant variance: ARCH-LM rejects at no lag ("NO ARCH")."""
    assert volatility_check(np.random.default_rng(0).normal(size=300))["verdict"] == "NO ARCH"


def test_a_single_variance_jump_is_explained_by_the_break() -> None:
    """As is, the jump looks like ARCH effects; per-regime scaling removes them."""
    result = volatility_check(break_series())

    # raw_p: ARCH-LM p-values on the series as given, one per tested lag.
    assert (result["raw_p"] < 0.05).any()
    assert result["verdict"] == "VARIANCE BREAK"


def test_garch_clustering_survives_removing_a_break() -> None:
    """GARCH volatility moves throughout, so scaling two regimes can't remove it."""
    assert volatility_check(garch_series())["verdict"] == "GARCH NEEDED"
