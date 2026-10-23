"""
Tests for the AICc finite-sample correction in autoregressive_experiments.diagnostics.
"""

import numpy as np
import pytest

from autoregressive_experiments.diagnostics import aicc


def test_aicc_adds_the_small_sample_penalty():
    """AICc = AIC + 2k(k+1) / (n - k - 1): with AIC 10, k 2, n 10 that is 10 + 12/7."""
    assert aicc(10.0, 2, 10) == pytest.approx(10 + 12 / 7)


def test_aicc_is_nan_when_parameters_use_up_the_sample():
    """With n - k - 1 <= 0 the correction is undefined (k 3, n 4 gives a zero denominator)."""
    assert np.isnan(aicc(10.0, 3, 4))
