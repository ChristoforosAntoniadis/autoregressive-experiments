"""
Tests for the SARIMA candidate grid and the AICc finite-sample correction in
autoregressive_experiments.diagnostics.
"""

import numpy as np
import pytest

from autoregressive_experiments.diagnostics import aicc, nonseasonal_orders


def test_nonseasonal_orders_keep_every_p_q_with_p_plus_d_plus_q_at_most_5():
    """With d = 1, p + q <= 4: the 15 pairs below, from (0, 0) up to (4, 0) and (0, 4)."""
    expected = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2),
        (3, 0), (3, 1),
        (4, 0),
    ]  # fmt: skip

    assert nonseasonal_orders() == expected


def test_aicc_adds_the_small_sample_penalty():
    """AICc = AIC + 2k(k+1) / (n - k - 1): with AIC 10, k 2, n 10 that is 10 + 12/7."""
    assert aicc(10.0, 2, 10) == pytest.approx(10 + 12 / 7)


def test_aicc_is_nan_when_parameters_use_up_the_sample():
    """With n - k - 1 <= 0 the correction is undefined (k 3, n 4 gives a zero denominator)."""
    assert np.isnan(aicc(10.0, 3, 4))
