"""
Tests for the tuning and champion selection on the validation split in
autoregressive_experiments.tuning.
"""

import json

import numpy as np
import pytest

from autoregressive_experiments.model_forecasts import EXPONENTIAL, NAIVE, SARIMA, SMA
from autoregressive_experiments.tuning import (
    choose_champions,
    decode,
    grid_size,
    horizon_metrics,
    mean_mase,
    origin_paths,
    search_space,
    usable,
    validation_origins,
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


def test_validation_origins_by_hand() -> None:
    """Training ends at month 166 and validation at 222: origins 166, 170, ..., 218, every 4
    months, all before the last validation month: (218 - 166) / 4 + 1 = 14 folds."""
    origins = validation_origins(166, 222)

    assert origins[0] == 166 and origins[-1] == 218 and len(origins) == 14
    assert set(np.diff(origins)) == {4}


def test_linked_settings_become_one_choice_of_valid_combinations() -> None:
    """SARIMA's p and q under p + 1 + q <= 5 give the 15 pairs with p + q <= 4; exponential
    smoothing's trend and damping give 5 forms (no trend undamped only, additive and
    multiplicative each undamped or damped); decoding unpacks them again."""
    sarima = search_space(SARIMA)
    smoothing = search_space(EXPONENTIAL)

    assert len(sarima["p+q"]) == 15
    assert all(sum(json.loads(choice).values()) <= 4 for choice in sarima["p+q"])
    assert len(smoothing["trend+damped_trend"]) == 5
    assert decode({"p+q": '{"p": 2, "q": 1}', "trend": "n"}) == {"p": 2, "q": 1, "trend": "n"}


def test_grid_sizes_by_hand() -> None:
    """SMA tries 23 windows x 2 outlier handlings = 46; exponential smoothing 5 forms x 3
    seasons x 2 Box-Cox x 2 initializations x 2 outlier handlings = 120; a space with a
    continuous range has no grid."""
    assert grid_size(search_space(SMA)) == 46
    assert grid_size(search_space(EXPONENTIAL)) == 120
    assert grid_size({"rate": {"low": 0.1, "high": 1.0, "log": True}}) is None


def test_horizon_metrics_and_their_mean_by_hand() -> None:
    """Levels 0..20 with a scale of 1; from origins 0 and 10 a path that is 1 too low at every
    step: MAE 1 at every horizon, so MASE 1 and its mean 1; only horizons the paths reach are
    scored."""
    actual = np.arange(21.0)
    paths = [actual[1:13] - 1, actual[11:21] - 1]

    metrics = horizon_metrics(paths, [0, 10], actual, scale=1.0)

    assert metrics[1]["MAE"] == pytest.approx(1.0) and metrics[1]["n"] == 2
    # The second origin forecasts 10 months, so only the first reaches horizon 12.
    assert metrics[12]["n"] == 1
    assert mean_mase(metrics) == pytest.approx(1.0)


def scored(value: float) -> dict[int, dict[str, float]]:
    """
    Metrics with the same MASE at every horizon.

    Parameters
    ----------
    value : float
        The MASE.

    Returns
    -------
    dict of int to dict of str to float
        Horizon -> {"MASE": value}.
    """
    return {horizon: {"MASE": value} for horizon in (1, 3, 6, 9, 12)}


def test_unusable_scores_rank_last_and_the_lowest_score_is_champion() -> None:
    """NaN ranks as infinity; within each family the lowest mean MASE wins: SMA (0.9) among the
    econometric models at 1.0, KNN (0.5) among ML at 2.0 with Lasso unusable, GRU (1.1) among
    the networks at 1.5."""
    assert usable(np.nan) == np.inf and usable(0.5) == 0.5
    econometric = (
        "SARIMA",
        "SARIMAX",
        "SARIMA + naive average",
        "VAR",
        "VECM",
        "SMA",
        "exponential smoothing",
        "Prophet",
    )
    ml = ("Ridge", "Lasso", "random forest", "XGBoost", "KNN")
    networks = (
        "MLP",
        "NumPy A1",
        "NumPy A2",
        "TensorFlow MLP",
        "PyTorch MLP",
        "LSTM",
        "GRU",
        "TCN",
        "N-BEATS",
        "DLinear",
    )
    by_model = {model: scored(1.0) for model in econometric}
    by_model |= {model: scored(2.0) for model in ml} | {model: scored(1.5) for model in networks}
    by_model |= {"SMA": scored(0.9), "Lasso": scored(np.nan), "KNN": scored(0.5)}
    by_model |= {"GRU": scored(1.1)}

    champions = choose_champions({0: by_model})

    assert champions[0] == {
        "econometric": "SMA",
        "machine learning": "KNN",
        "neural network": "GRU",
    }


def test_origin_paths_ignore_later_months_and_match_one_fit() -> None:
    """Forecasts run in parallel from each origin equal a direct fit on the months up to it, and
    changing the months after an origin leaves its forecasts unchanged."""
    levels = random_walks(60)
    dates = np.arange(60)
    changed = levels.copy()
    changed[41:] += 5.0

    paths = origin_paths(SMA, {"window": 3}, levels, dates, [30, 40], 59, [0])
    after = origin_paths(SMA, {"window": 3}, changed, dates, [30, 40], 59, [0])
    naive = origin_paths(NAIVE, {}, levels, dates, [40], 59, [0, 1])

    assert paths[0][0] == pytest.approx(np.full(12, levels[28:31, 0].mean()))
    assert np.array_equal(paths[0][1], after[0][1])
    assert naive[1][0] == pytest.approx(np.full(12, levels[40, 1]))
