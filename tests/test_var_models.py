"""
Tests for VAR / VECM estimation and checks in autoregressive_experiments.var_models.
"""

import numpy as np
import pandas as pd
import pytest

from autoregressive_experiments.var_models import (
    check_candidate,
    companion_roots,
    fit_candidates,
    is_stable,
)


def test_companion_roots_of_a_diagonal_var1_are_its_diagonal() -> None:
    """A = diag(0.5, 0.2): each series decays on its own, so the roots are 0.5 and 0.2."""
    # Coefficients have shape p x k x k (one k x k matrix A_i per lag): here p = 1, k = 2.
    roots = companion_roots(np.array([[[0.5, 0.0], [0.0, 0.2]]]))

    assert np.abs(roots) == pytest.approx([0.5, 0.2])


def test_companion_roots_of_an_ar2_by_hand() -> None:
    """y_t = 0.5 y_(t-1) + 0.3 y_(t-2): roots of x^2 - 0.5x - 0.3 = 0 are
    (0.5 +/- sqrt(0.25 + 1.2)) / 2 = 0.852 and -0.352."""
    # A single series with two lags: 1 x 1 matrices for lag 1 (0.5) and lag 2 (0.3).
    roots = companion_roots(np.array([[[0.5]], [[0.3]]]))

    assert roots.real == pytest.approx([0.852, -0.352], abs=1e-3)


@pytest.mark.parametrize(
    ("moduli", "unit_roots", "expected"),
    [
        ([0.9, 0.3], 0, True),  # all inside
        ([1.02, 0.3], 0, False),  # one outside: explosive
        ([1.0, 0.4, 0.2], 1, True),  # a VECM's single unit root, the rest inside
        ([1.0, 1.0, 0.2], 1, False),  # two unit roots where one is expected
        ([0.9, 0.4], 1, False),  # the expected unit root is missing
    ],
)
def test_is_stable_counts_unit_roots_and_requires_the_rest_inside(
    moduli: list[float], unit_roots: int, expected: bool
) -> None:
    """Stable means exactly `unit_roots` moduli at 1 and every other modulus below 1.

    A VAR on differences expects no unit root; a rank-1 VECM of two series expects one (their
    shared stochastic trend).
    """
    assert is_stable(np.array(moduli), unit_roots) == expected


def simulated_levels(count: int = 400) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A cointegrated pair (y = 2x + noise, x a random walk), as levels and differences.

    Parameters
    ----------
    count : int, default 400
        Number of months of levels to simulate.

    Returns
    -------
    levels : pandas.DataFrame
        Columns "a" (x) and "b" (2x + noise) on a month-end date index from 2000-01, seeded
        with 0.
    differences : pandas.DataFrame
        The first differences of levels (count - 1 rows; the first month is dropped).
    """
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=count))
    dates = pd.date_range("2000-01-31", periods=count, freq="ME")
    levels = pd.DataFrame({"a": x, "b": 2 * x + rng.normal(size=count)}, index=dates)
    return levels, levels.diff().dropna()


def test_every_candidate_is_fitted_and_the_vecm_finds_the_true_relation() -> None:
    """Six candidates; the VECM's long-run relation is a - b / 2 and it carries one unit root."""
    levels, differences = simulated_levels()

    candidates = fit_candidates(levels, differences)

    assert list(candidates) == ["VAR(1)", "VAR(3)", "VAR(5)", "VECM(1)", "VECM(3)", "VECM(5)"]
    # beta is only identified up to scale, so compare the ratio: b's weight over a's is -1/2.
    beta = candidates["VECM(1)"]["fitted"].beta.ravel()
    assert beta[1] / beta[0] == pytest.approx(-0.5, abs=0.02)
    assert is_stable(np.abs(candidates["VECM(1)"]["roots"]), unit_roots=1)


def test_portmanteau_flags_a_var1_fitted_to_data_with_a_strong_second_lag() -> None:
    """Differences with a strong lag-2 effect: VAR(1) leaves autocorrelation behind, VAR(3)
    doesn't."""
    # Each series is its own AR(2) with only a lag-2 term: x_t = 0.6 x_(t-2) + e_t.
    rng = np.random.default_rng(0)
    values = np.zeros((500, 2))
    for t in range(2, 500):
        values[t] = 0.6 * values[t - 2] + rng.normal(size=2)
    differences = pd.DataFrame(values, columns=["a", "b"])
    levels = differences.cumsum()

    candidates = fit_candidates(levels, differences)

    # "white" is True when the Portmanteau test passes at every lag checked.
    assert not check_candidate(candidates["VAR(1)"])["white"]
    assert check_candidate(candidates["VAR(3)"])["white"]


def test_check_candidate_fails_when_no_test_lag_can_be_tested() -> None:
    """A model with 24 lags leaves no testable lag up to PLOT_LAGS = 24, so there's no
    evidence of white residuals: "white" must be False, not vacuously True."""
    # fitted is never used: with no testable lag, test_whiteness is never called.
    candidate = {"fitted": None, "model_lags": 24, "roots": np.array([0.5]), "unit_roots": 0}

    result = check_candidate(candidate)

    assert result["portmanteau"] == {}
    assert not result["white"]
    assert result["verdict"] == "FAIL"
