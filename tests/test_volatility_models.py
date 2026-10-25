"""
Tests for the GARCH(1,1) step (U5b) in autoregressive_experiments.volatility_models.
"""

import numpy as np
import pytest

from autoregressive_experiments.volatility_models import (
    CONSTANT,
    GARCH_NORMAL,
    GARCH_T,
    GARCH_T_BREAK,
    fit_variance_models,
    garch_check,
    garch_variance_path,
    half_life,
    original_units_loglikelihood,
)


def simulate_garch(count: int) -> np.ndarray:
    """GARCH(1,1): variance_t = 0.1 + 0.3 e_(t-1)^2 + 0.6 variance_(t-1), normal shocks."""
    shocks = np.random.default_rng(0).normal(size=count)
    values = np.zeros(count)
    variance = 0.1 / (1 - 0.3 - 0.6)
    for t in range(count):
        if t:
            variance = 0.1 + 0.3 * values[t - 1] ** 2 + 0.6 * variance
        values[t] = np.sqrt(variance) * shocks[t]
    return values


def test_half_life_is_the_months_for_a_shock_to_halve():
    """0.5 ** 1 = 0.5, so 1 month; 0.9 ** h = 0.5 gives h = log(0.5) / log(0.9) = 6.58."""
    assert half_life(0.5) == pytest.approx(1.0)
    assert half_life(0.9) == pytest.approx(6.579, abs=1e-3)
    assert half_life(1.0) == np.inf


def test_loglikelihood_is_in_the_units_arch_was_given_despite_its_rescaling():
    """Multiplying data by 10 divides its density by 10 per point: the log-likelihoods of y and
    10y differ by exactly n log 10 = 300 log 10, whatever internal scale arch chose for each."""
    y = np.random.default_rng(0).normal(size=300) * 0.05

    small = original_units_loglikelihood(fit_variance_models(y)[CONSTANT])
    large = original_units_loglikelihood(fit_variance_models(10 * y)[CONSTANT])

    assert small - large == pytest.approx(300 * np.log(10))


def test_break_adjusted_garch_wins_when_clustering_sits_on_top_of_a_variance_jump():
    """GARCH(1,1) data whose second half is scaled by 5: a plain GARCH has to explain the jump
    with persistence, the break-adjusted one only has to explain the clustering."""
    values = simulate_garch(600)
    values[300:] *= 5

    result = garch_check(values)

    assert result["best"] == GARCH_T_BREAK
    assert result["verdict"] == "CARRY GARCH FORWARD"


def test_garch_variance_path_is_in_the_units_it_was_given():
    """Multiplying the residuals by 10 multiplies every forecast variance by 100, whatever
    internal scale arch chose; the Student-t degrees of freedom don't change."""
    values = simulate_garch(600) * 0.05

    small, nu_small = garch_variance_path(values, GARCH_T, 6)
    large, nu_large = garch_variance_path(10 * values, GARCH_T, 6)

    assert large == pytest.approx(100 * small, rel=1e-3)
    assert nu_large == pytest.approx(nu_small, rel=1e-3)


def test_break_adjusted_variance_path_forecasts_on_the_latest_regime_scale():
    """With the second half scaled by 5, the break-adjusted forecast variances sit on the
    volatile regime's scale: far above the calm half's average squared residual."""
    values = simulate_garch(600)
    values[300:] *= 5

    variances, _ = garch_variance_path(values, GARCH_T_BREAK, 3)

    assert (variances > 5 * np.mean(values[:300] ** 2)).all()


def test_garch_fit_recovers_the_simulated_parameters():
    """With 3,000 draws the estimates land close to alpha 0.3 and beta 0.6."""
    params = fit_variance_models(simulate_garch(3000))[GARCH_NORMAL].params

    assert params["alpha[1]"] == pytest.approx(0.3, abs=0.07)
    assert params["beta[1]"] == pytest.approx(0.6, abs=0.07)


def test_garch_data_carries_garch_forward_and_noise_keeps_a_constant_variance():
    """BIC prefers GARCH for clustered data, which leaves no ARCH effects once standardized;
    for independent noise GARCH's two extra parameters don't pay off."""
    assert garch_check(simulate_garch(1000))["verdict"] == "CARRY GARCH FORWARD"
    noise = np.random.default_rng(0).normal(size=300)
    assert garch_check(noise)["verdict"] == "CONSTANT VARIANCE"
