"""
Tests for the time-aware random search in autoregressive_experiments.tuning.
"""

import numpy as np
import pytest
from scipy import stats
from sklearn.base import clone

from autoregressive_experiments import tuning
from autoregressive_experiments.forecast_evaluation import EXPONENTIAL, SARIMA, SMA, VAR_MODEL
from autoregressive_experiments.ml_forecasting import (
    MAX_LAG,
    RIDGE,
    build_model,
    lag_columns,
    lag_features,
)
from autoregressive_experiments.nn_forecasting import PYTORCH_MLP
from autoregressive_experiments.tuning import (
    CANDIDATES,
    SPACES,
    TunableModel,
    arima_one_step,
    constant_quantiles,
    draw_candidates,
    ml_search,
    optuna_search,
    smoothing_one_step,
    system_one_step,
    time_series_folds,
    volatility_quantiles,
)


def random_walks(months: int, seed: int = 0) -> np.ndarray:
    """
    Two independent random walks with a small drift, as positive levels.

    Parameters
    ----------
    months : int
        Number of levels.
    seed : int, default=0
        Seed of the steps.

    Returns
    -------
    np.ndarray
        Shape (months, 2), starting near 10, so every level stays positive.
    """
    steps = np.random.default_rng(seed).normal(0.05, 0.3, size=(months, 2))
    return 10 + np.cumsum(steps, axis=0)


def test_folds_over_the_training_changes_by_hand() -> None:
    """166 training changes, 5 folds, at least 60 training months: each fold tests
    (166 - 60) // 5 = 21 months, so the first trains on 166 - 5 x 21 = 61 and the last ends
    at 166."""
    assert time_series_folds(166) == [(61, 82), (82, 103), (103, 124), (124, 145), (145, 166)]


def test_draws_are_reproducible_and_small_spaces_are_tried_in_full() -> None:
    """SARIMA's space keeps the 15 (p, q) pairs with p + 1 + q <= 5, times 2 x 2 x 2 for P, Q
    and the trend: 120 valid settings, of which CANDIDATES are drawn, the same ones each time.
    Exponential smoothing has 3 x 3 x 2 x 2 = 36 settings without damping, plus 24 damped
    ones with a trend: 60, at most CANDIDATES, so all are tried. SMA's 23 windows are all
    tried too."""
    first, size = draw_candidates(SARIMA, SPACES[SARIMA])
    second, _ = draw_candidates(SARIMA, SPACES[SARIMA])

    assert size == 120 and len(first) == CANDIDATES and first == second
    assert all(setting["p"] + setting["q"] <= 4 for setting in first)
    smoothing, smoothing_size = draw_candidates(EXPONENTIAL, SPACES[EXPONENTIAL])
    assert smoothing_size == 60 and len(smoothing) == 60
    assert not any(s["damped_trend"] and s["trend"] is None for s in smoothing)
    assert len(draw_candidates(SMA, SPACES[SMA])[0]) == 23


def test_moving_average_errors_by_hand() -> None:
    """Levels 0..9; a fold testing changes 5 and 6 forecasts levels 6 and 7 from the mean of
    the two levels before each: (4 + 5) / 2 = 4.5 and 5.5, so both errors are 1.5."""
    levels = np.arange(10.0)

    errors = smoothing_one_step(levels, SMA, {"window": 2}, [(5, 7)])

    assert errors.tolist() == [1.5, 1.5]


def test_sarima_fold_errors_ignore_everything_after_the_fold() -> None:
    """Changing every month after the first fold leaves that fold's errors unchanged: each fold
    is fitted on earlier months and run through its own months only."""
    changes = np.diff(random_walks(71)[:, 0])
    setting = {"p": 1, "q": 0, "P": 0, "Q": 0, "trend": "c"}
    folds = [(40, 50), (50, 60)]

    before, residuals, _ = arima_one_step(changes, setting, folds)
    changed = changes.copy()
    changed[50:] += 5.0
    after, _, _ = arima_one_step(changed, setting, folds)

    assert np.array_equal(before[:10], after[:10])
    assert not np.array_equal(before[10:], after[10:])
    # Per fold: the in-sample residuals and the fold's 10 one-month errors.
    assert [len(run_on) for _, run_on in residuals] == [10, 10]


def test_tunable_model_survives_cloning_and_takes_new_settings() -> None:
    """RandomizedSearchCV clones the estimator and sets each drawn setting on the copy: the copy
    keeps the model and settings, and set_params replaces only what it is given."""
    original = TunableModel(RIDGE, lags=2, alpha=1.0)

    copy = clone(original).set_params(alpha=10.0)

    assert original.get_params() == {"model": RIDGE, "lags": 2, "alpha": 1.0}
    assert copy.get_params() == {"model": RIDGE, "lags": 2, "alpha": 10.0}


def test_randomized_search_scores_equal_fitting_each_fold_on_earlier_months() -> None:
    """Every setting's score from RandomizedSearchCV equals the MAE of fitting it on the months
    before each fold and predicting the fold's months by hand: the search uses the
    time-ordered folds, not its default ones."""
    changes = np.diff(random_walks(81), axis=0)
    folds = [(50, 60), (60, 70), (70, 80)]

    best, scored, size = ml_search(changes, RIDGE, 0, folds)

    features, target = lag_features(changes, MAX_LAG), changes[MAX_LAG:, 0]
    for entry in scored[:3]:
        setting = entry["settings"]
        columns = lag_columns(setting["lags"], MAX_LAG, 2)
        errors = []
        for start, end in folds:
            first, last = start - MAX_LAG, end - MAX_LAG
            fitted = build_model(RIDGE, setting).fit(features[:first, columns], target[:first])
            errors.append(target[first:last] - fitted.predict(features[first:last, columns]))
        assert entry["score"] == pytest.approx(np.mean(np.abs(np.concatenate(errors))))
    # Ridge's space: 8 lag counts x 8 penalties.
    assert size == 64 and len(scored) == CANDIDATES
    assert best == min(scored, key=lambda row: row["score"])["settings"]


def test_optuna_search_is_seeded_and_scores_every_trial_on_the_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three Optuna trials of the PyTorch network: two searches choose the same settings and
    scores (seeded TPE, deterministic fits), the best is the lowest-scoring trial, and its score
    is the MAE of fitting it on each fold's earlier months, as for every other model."""
    monkeypatch.setattr(tuning, "CANDIDATES", 3)
    changes = np.diff(random_walks(81), axis=0)
    folds = [(50, 60), (60, 70), (70, 80)]

    best, scored, size = optuna_search(changes, PYTORCH_MLP, 0, folds)
    again, scored_again, _ = optuna_search(changes, PYTORCH_MLP, 0, folds)

    assert (best, scored) == (again, scored_again) and len(scored) == 3 and size == 0
    winner = min(scored, key=lambda row: row["score"])
    assert best == winner["settings"]
    features, target = lag_features(changes, MAX_LAG), changes[MAX_LAG:, 0]
    columns = lag_columns(best["lags"], MAX_LAG, 2)
    errors = []
    for start, end in folds:
        first, last = start - MAX_LAG, end - MAX_LAG
        fitted = build_model(PYTORCH_MLP, best).fit(features[:first, columns], target[:first])
        errors.append(target[first:last] - fitted.predict(features[first:last, columns]))
    assert winner["score"] == pytest.approx(np.mean(np.abs(np.concatenate(errors))), rel=1e-6)


def test_constant_quantiles_by_hand() -> None:
    """Residuals 3 and -4 have a root mean square of sqrt((9 + 16) / 2) = 3.536, so the 97.5%
    quantile is 1.960 x 3.536 = 6.930 for every fold month."""
    quantiles = constant_quantiles(np.array([3.0, -4.0]), np.zeros(3))

    assert quantiles.shape == (3, 5)
    assert quantiles[:, -1] == pytest.approx(stats.norm.ppf(0.975) * np.sqrt(12.5))


def test_volatility_quantiles_are_in_the_units_they_were_given() -> None:
    """Multiplying the residuals by 10 multiplies the quantiles by 10 (arch rescales inside);
    the median of a symmetric distribution is 0."""
    residuals = np.random.default_rng(1).standard_t(6, size=150) * 0.05
    setting = {"p": 1, "q": 1, "dist": "t", "break_adjusted": False}

    small = volatility_quantiles("GARCH", setting, residuals[:120], residuals[120:])
    large = volatility_quantiles("GARCH", setting, 10 * residuals[:120], 10 * residuals[120:])

    assert large == pytest.approx(10 * small, rel=1e-3)
    assert small[:, 2] == pytest.approx(0.0, abs=1e-12)


def test_systems_share_their_fits_between_the_series() -> None:
    """Scoring both series of a VAR re-uses one fit per month: 10 fold months give 10 cached
    forecasts, not 20."""
    levels = random_walks(71)
    cache: dict[tuple[str, str, int], np.ndarray] = {}
    setting = {"lags": 2, "trend": "c"}

    for column in (0, 1):
        system_one_step(levels, VAR_MODEL, setting, column, [(60, 70)], cache)

    assert len(cache) == 10
