"""
Tests for the residual checks (step U4) in autoregressive_experiments.residual_checks.
"""

import numpy as np
import pandas as pd

from autoregressive_experiments.residual_checks import passes, recommended_orders, residual_tests


def candidate_table() -> pd.DataFrame:
    columns = ["series", "p", "d", "q", "P", "Q", "converged", "AIC", "AICc", "BIC"]
    rows = [
        ("a", 2, 1, 2, 0, 0, True, -100.0, -99.0, -90.0),
        ("a", 0, 1, 4, 0, 0, True, -99.5, -98.0, -89.0),
        ("a", 0, 1, 1, 0, 0, True, -95.0, -95.0, -93.0),
        ("a", 1, 1, 1, 0, 0, False, -200.0, -200.0, -200.0),  # did not converge: ignored
        ("b", 1, 1, 0, 1, 1, True, -50.0, -50.0, -40.0),
        ("b", 0, 1, 0, 0, 0, True, -47.0, -47.0, -45.0),
    ]
    return pd.DataFrame(rows, columns=columns)


def test_recommended_orders_take_each_criterion_winner_and_a_tied_aic_runner_up():
    """Series a: (2,1,2) wins AIC and AICc, (0,1,1) wins BIC, (0,1,4) is 0.5 AIC behind (a tie).
    Series b: the AIC runner-up is 3 AIC behind, more than the margin of 2, so it isn't added
    for being a runner-up; it still enters as the BIC winner."""
    picks = [
        (m["series"], (m["p"], m["d"], m["q"], m["P"], m["Q"]), m["reason"])
        for m in recommended_orders(candidate_table())
    ]

    assert picks == [
        ("a", (2, 1, 2, 0, 0), "AIC, AICc"),
        ("a", (0, 1, 1, 0, 0), "BIC"),
        ("a", (0, 1, 4, 0, 0), "AIC tie (+0.500)"),
        ("b", (1, 1, 0, 1, 1), "AIC, AICc"),
        ("b", (0, 1, 0, 0, 0), "BIC"),
    ]


def test_passes_ignores_untestable_lags_and_fails_below_five_percent():
    """NaN (too few degrees of freedom) is skipped; any p-value under 0.05 fails the model."""
    tests = pd.DataFrame(
        {"lag": [6, 12, 24], "ljung_box_p": [np.nan, 0.20, 0.06], "breusch_godfrey_p": [0.5] * 3}
    )

    assert passes(tests)
    assert not passes(tests.assign(breusch_godfrey_p=[0.5, 0.049, 0.5]))


def test_white_noise_passes_and_an_ar1_series_fails():
    """Independent noise has no autocorrelation; x_t = 0.8 x_(t-1) + e_t has plenty."""
    noise = np.random.default_rng(0).normal(size=300)
    ar1 = np.zeros(300)
    for t in range(1, 300):
        ar1[t] = 0.8 * ar1[t - 1] + noise[t]

    assert passes(residual_tests(pd.Series(noise), arma_terms=0))
    ar1_tests = residual_tests(pd.Series(ar1), arma_terms=0)
    assert (ar1_tests[["ljung_box_p", "breusch_godfrey_p"]] < 1e-6).all().all()


def test_ljung_box_loses_one_degree_of_freedom_per_arma_term():
    """With 6 ARMA terms, lag 6 has 0 degrees of freedom (n/a), and the same statistic at
    lag 12 is judged on 6 instead of 12 degrees of freedom, so its p-value is smaller."""
    noise = pd.Series(np.random.default_rng(0).normal(size=300))

    plain = residual_tests(noise, arma_terms=0)
    corrected = residual_tests(noise, arma_terms=6)

    assert np.isnan(corrected["ljung_box_p"].iloc[0])
    assert corrected["ljung_box_p"].iloc[1] < plain["ljung_box_p"].iloc[1]
    # Breusch-Godfrey has no ARMA correction, so it is unchanged.
    assert corrected["breusch_godfrey_p"].tolist() == plain["breusch_godfrey_p"].tolist()
