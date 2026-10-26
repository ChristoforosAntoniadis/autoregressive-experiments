"""
Tests for the VAR lag selection and Johansen cointegration steps (V2, V3) in
autoregressive_experiments.var_diagnostics.
"""

import numpy as np
import pytest

from autoregressive_experiments.var_diagnostics import (
    johansen_rank,
    johansen_test,
    select_var_lags,
)


@pytest.mark.parametrize(
    ("statistics", "critical", "expected"),
    [
        ([20.0, 1.0], [15.5, 3.8], 1),  # rank 0 rejected, rank <= 1 not: rank 1
        ([10.0, 1.0], [15.5, 3.8], 0),  # rank 0 not rejected
        ([20.0, 5.0], [15.5, 3.8], 2),  # both rejected: full rank
        ([15.5, 1.0], [15.5, 3.8], 1),  # a statistic equal to its critical value rejects
    ],
)
def test_johansen_rank_is_the_first_null_not_rejected(
    statistics: list[float], critical: list[float], expected: int
) -> None:
    """The rank is the first r whose null "rank <= r" is not rejected (statistic below its
    critical value); if every null is rejected, the rank is full (2 for two series).

    The i-th statistic and critical value test the null "rank <= i".
    """
    assert johansen_rank(np.array(statistics), np.array(critical)) == expected


def test_johansen_finds_one_relation_for_a_cointegrated_pair() -> None:
    """y = 2x + noise with x a random walk: one cointegrating relation, x - y / 2."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=300))
    levels = np.column_stack([x, 2 * x + rng.normal(size=300)])

    result = johansen_test(levels, lagged_differences=1)

    assert (result["trace_rank"], result["max_eigen_rank"]) == (1, 1)
    # The vector is normalized to 1 on x, so its y coefficient is -1/2.
    assert result["vector"][1] == pytest.approx(-0.5, abs=0.02)


def test_johansen_finds_no_relation_between_independent_random_walks() -> None:
    """Two independent random walks share no long-run relation: both tests give rank 0."""
    rng = np.random.default_rng(1)
    levels = np.cumsum(rng.normal(size=(300, 2)), axis=0)

    result = johansen_test(levels, lagged_differences=1)

    assert (result["trace_rank"], result["max_eigen_rank"]) == (0, 0)


def test_bic_recovers_the_lag_order_of_a_simulated_var1() -> None:
    """Differences from a VAR(1) with strong cross-effects: BIC picks 1 lag, and every
    criterion is reported for 0 to 10 lags."""
    rng = np.random.default_rng(0)
    coefficients = np.array([[0.5, 0.3], [0.2, 0.4]])
    values = np.zeros((400, 2))
    for t in range(1, 400):
        values[t] = coefficients @ values[t - 1] + rng.normal(size=2)

    selected, criteria = select_var_lags(values)

    assert selected["bic"] == 1
    assert list(criteria.index) == list(range(11))
