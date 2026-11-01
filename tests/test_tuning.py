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
    failed_metrics,
    grid_size,
    horizon_metrics,
    origin_paths,
    score,
    score_error,
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
    # 5 trend forms x 3 seasons x 2 Box-Cox = 30 on the levels; on the logarithm only the
    # additive forms without Box-Cox, 3 x 2 x 1 = 6.
    assert len(smoothing["trend+damped_trend+seasonal+use_boxcox+transform"]) == 36
    assert decode({"p+q": '{"p": 2, "q": 1}', "trend": "n"}) == {"p": 2, "q": 1, "trend": "n"}


def test_grid_sizes_by_hand() -> None:
    """SMA tries 23 windows x 2 outlier handlings x 2 transforms = 92; exponential smoothing 36
    valid linked combinations x 2 initializations x 2 outlier handlings = 144; a space with a
    continuous range has no grid."""
    assert grid_size(search_space(SMA)) == 92
    assert grid_size(search_space(EXPONENTIAL)) == 144
    assert grid_size({"rate": {"low": 0.1, "high": 1.0, "log": True}}) is None


def test_horizon_metrics_and_the_score_by_hand() -> None:
    """Levels 0..20 with a scale of 1; from origins 0 and 10 a path that is 1 too low at every
    step: MAE 1 at every horizon, so MASE 1; only horizons the paths reach are scored. Against a
    reference 2 too low everywhere the score is 0.5 at every horizon, so 0.5 overall."""
    actual = np.arange(21.0)
    paths = [actual[1:13] - 1, actual[11:21] - 1]
    reference = [actual[1:13] - 2, actual[11:21] - 2]

    metrics = horizon_metrics(paths, [0, 10], actual, scale=1.0)
    naive = horizon_metrics(reference, [0, 10], actual, scale=1.0)

    assert metrics[1]["MAE"] == pytest.approx(1.0) and metrics[1]["n"] == 2
    # The second origin forecasts 10 months, so only the first reaches horizon 12.
    assert metrics[12]["n"] == 1
    assert score(metrics, naive) == pytest.approx(0.5)
    assert score(metrics, metrics) == pytest.approx(1.0)


def test_a_failed_setting_scores_nan_and_ranks_last() -> None:
    """A fit that raised leaves NaN metrics with nothing scored, which usable() ranks last."""
    metrics = failed_metrics()

    assert metrics[1]["n"] == 0 and np.isnan(metrics[12]["MAE"])
    assert usable(score(metrics, metrics)) == np.inf


def test_score_error_is_zero_when_every_origin_agrees() -> None:
    """A model whose error is half naive's at every origin and step scores 0.5 in every resample
    of the origins, so its bootstrap standard error is 0."""
    actual = np.arange(40.0)
    origins = [10, 15, 20, 25]
    naive = {0: [actual[o + 1 : o + 13] - 2 for o in origins]}
    model = {0: [actual[o + 1 : o + 13] - 1 for o in origins]}

    error = score_error(model, naive, origins, actual[:, None], [1.0], [0])

    assert error == pytest.approx(0.0)


def test_unusable_scores_rank_last_and_the_lowest_score_is_champion_with_its_ties() -> None:
    """NaN ranks as infinity; within each family the lowest score wins: SMA (0.9) among the
    econometric models at 1.0, KNN (0.5) among ML at 2.0 with Lasso unusable, GRU (1.1) among
    the networks at 1.5. Models within one standard error of the champion are tied with it: SMA's
    standard error 0.1 reaches the 1.0 models, KNN's 0.1 reaches nothing, GRU's 0.5 reaches the
    1.5 models but not the unusable one."""
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
    by_model = {model: 1.0 for model in econometric}
    by_model |= {model: 2.0 for model in ml} | {model: 1.5 for model in networks}
    by_model |= {"SMA": 0.9, "Lasso": np.nan, "KNN": 0.5, "GRU": 1.1}
    errors = {model: 0.0 for model in by_model} | {"SMA": 0.1, "KNN": 0.1, "GRU": 0.5}

    champions = choose_champions({0: by_model}, {0: errors})

    assert {f: r["model"] for f, r in champions[0].items()} == {
        "econometric": "SMA",
        "machine learning": "KNN",
        "neural network": "GRU",
    }
    assert set(champions[0]["econometric"]["tied_with"]) == set(econometric) - {"SMA"}
    assert champions[0]["machine learning"]["tied_with"] == []
    assert set(champions[0]["neural network"]["tied_with"]) == set(networks) - {"GRU"}


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
