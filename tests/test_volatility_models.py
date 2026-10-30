"""
Tests for the GARCH(1,1) step in autoregressive_experiments.volatility_models.
"""

import numpy as np
import pytest

from autoregressive_experiments.volatility_models import (
    CONSTANT,
    GARCH_NORMAL,
    GARCH_T,
    GARCH_T_BREAK,
    bootstrap_checks,
    bootstrap_p_values,
    fit_candidate,
    fit_variance_models,
    garch_check,
    garch_variance_path,
    half_life,
    original_units_loglikelihood,
)


def simulate_garch(count: int) -> np.ndarray:
    """GARCH(1,1): variance_t = 0.1 + 0.3 e_(t-1)^2 + 0.6 variance_(t-1), normal shocks.

    Parameters
    ----------
    count : int
        Number of draws to simulate.

    Returns
    -------
    numpy.ndarray
        The simulated series, shape (count,), seeded with 0 so every run is the same.

    Notes
    -----
    The first draw starts at the unconditional variance 0.1 / (1 - 0.3 - 0.6) = 1. Persistence
    is alpha + beta = 0.9, so a volatility shock halves in about 6.6 months.
    """
    shocks = np.random.default_rng(0).normal(size=count)
    values = np.zeros(count)
    variance = 0.1 / (1 - 0.3 - 0.6)
    for t in range(count):
        if t:
            variance = 0.1 + 0.3 * values[t - 1] ** 2 + 0.6 * variance
        values[t] = np.sqrt(variance) * shocks[t]
    return values


def test_half_life_is_the_months_for_a_shock_to_halve() -> None:
    """0.5 ** 1 = 0.5, so 1 month; 0.9 ** h = 0.5 gives h = log(0.5) / log(0.9) = 6.58."""
    assert half_life(0.5) == pytest.approx(1.0)
    assert half_life(0.9) == pytest.approx(6.579, abs=1e-3)
    # Persistence 1 (IGARCH): a shock never fades.
    assert half_life(1.0) == np.inf


def test_loglikelihood_is_in_the_units_arch_was_given_despite_its_rescaling() -> None:
    """Multiplying data by 10 divides its density by 10 per point: the log-likelihoods of y and
    10y differ by exactly n log 10 = 300 log 10, whatever internal scale arch chose for each."""
    y = np.random.default_rng(0).normal(size=300) * 0.05

    small = original_units_loglikelihood(fit_variance_models(y)[CONSTANT])
    large = original_units_loglikelihood(fit_variance_models(10 * y)[CONSTANT])

    assert small - large == pytest.approx(300 * np.log(10))


def test_break_adjusted_garch_wins_when_clustering_sits_on_top_of_a_variance_jump() -> None:
    """GARCH(1,1) data whose second half is scaled by 5: a plain GARCH has to explain the jump
    with persistence, the break-adjusted one only has to explain the clustering, so BIC
    prefers it and its persistence stays below 1.

    The verdict isn't asserted: on a single draw of a true GARCH, the six bootstrap checks
    reject at least once about 9% of the time by design (measured: 200 simulated series).
    """
    values = simulate_garch(600)
    values[300:] *= 5

    result = garch_check(values, replications=0)

    assert result["best"] == GARCH_T_BREAK
    assert result["persistence"] < 1


def test_garch_variance_path_is_in_the_units_it_was_given() -> None:
    """Multiplying the residuals by 10 multiplies every forecast variance by 100, whatever
    internal scale arch chose; the Student-t degrees of freedom don't change."""
    values = simulate_garch(600) * 0.05

    small, nu_small = garch_variance_path(values, GARCH_T, 6)
    large, nu_large = garch_variance_path(10 * values, GARCH_T, 6)

    assert large == pytest.approx(100 * small, rel=1e-3)
    assert nu_large == pytest.approx(nu_small, rel=1e-3)


def test_break_adjusted_variance_path_forecasts_on_the_latest_regime_scale() -> None:
    """With the second half scaled by 5, the break-adjusted forecast variances sit on the
    volatile regime's scale: far above the calm half's average squared residual."""
    values = simulate_garch(600)
    values[300:] *= 5

    variances, _ = garch_variance_path(values, GARCH_T_BREAK, 3)

    # The volatile half's variance is about 5^2 = 25 times the calm half's; 5 times is a
    # loose bar that a forecast on the calm scale (about 1) can't clear.
    assert (variances > 5 * np.mean(values[:300] ** 2)).all()


def test_garch_fit_recovers_the_simulated_parameters() -> None:
    """With 3,000 draws the estimates land close to alpha 0.3 and beta 0.6."""
    params = fit_variance_models(simulate_garch(3000))[GARCH_NORMAL].params

    assert params["alpha[1]"] == pytest.approx(0.3, abs=0.07)
    assert params["beta[1]"] == pytest.approx(0.6, abs=0.07)


def test_bic_prefers_garch_for_clustered_data_and_a_constant_variance_for_noise() -> None:
    """BIC prefers a stationary GARCH for clustered data; for independent noise GARCH's two
    extra parameters don't pay off, which decides the verdict whatever the checks say.

    The clustered case's verdict isn't asserted: a single draw of a true GARCH fails one of
    the six bootstrap checks about 9% of the time by design (measured: 200 simulated series).
    """
    clustered = garch_check(simulate_garch(1000), replications=0)
    assert clustered["best"] != CONSTANT
    assert clustered["persistence"] < 1
    noise = np.random.default_rng(0).normal(size=300)
    # The verdict for a constant variance doesn't depend on the checks: a few replications do.
    assert garch_check(noise, replications=9)["verdict"] == "CONSTANT VARIANCE"


def test_bootstrap_p_values_by_hand() -> None:
    """Observed statistics 2 and 5 against 3 simulated rows (1, 6), (3, 4), (2, 1).

    Column 1: 3 and 2 are >= 2, so p = (1 + 2) / (3 + 1) = 0.75. Column 2: only 6 is >= 5,
    so p = (1 + 1) / 4 = 0.5. Ties count as at least as extreme.
    """
    simulated = np.array([[1.0, 6.0], [3.0, 4.0], [2.0, 1.0]])

    p_values = bootstrap_p_values(np.array([2.0, 5.0]), simulated)

    assert p_values == pytest.approx([0.75, 0.5])


def test_bootstrap_checks_are_reproducible_and_on_the_bootstrap_grid() -> None:
    """The seed is fixed, so two runs give identical p-values; with 19 replications every
    p-value is a multiple of 1 / 20, between 1 / 20 and 1."""
    values = simulate_garch(300)
    candidate = fit_candidate(values, GARCH_T)

    first = bootstrap_checks(candidate, GARCH_T, replications=19)
    second = bootstrap_checks(candidate, GARCH_T, replications=19)

    for key in ("arch_p_curve", "arch_p", "squared_ljung_box_p"):
        assert np.array_equal(first[key], second[key])
    p_values = np.concatenate([first["arch_p_curve"], first["squared_ljung_box_p"]])
    assert p_values * 20 == pytest.approx(np.round(p_values * 20))
    assert ((p_values >= 1 / 20) & (p_values <= 1)).all()
