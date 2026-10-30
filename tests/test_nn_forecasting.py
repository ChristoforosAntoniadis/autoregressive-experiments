"""
Tests for the neural-network family in autoregressive_experiments.nn_forecasting.
"""

from itertools import pairwise

import numpy as np
import pytest

from autoregressive_experiments.ml_forecasting import build_model, lag_features
from autoregressive_experiments.modeling.utilities.network import Cache, NumpyNetwork
from autoregressive_experiments.modeling.utilities.optimizers import Optimizer, newton_schulz
from autoregressive_experiments.nn_forecasting import (
    DLINEAR,
    GRU,
    LSTM,
    MLP,
    NBEATS,
    NUMPY_A1,
    NUMPY_A2,
    NUMPY_ARCHITECTURES,
    PYTORCH_MLP,
    TCN,
    TENSORFLOW_MLP,
    build_network,
)


def simulated_changes(months: int) -> np.ndarray:
    """
    Two series' monthly changes with a known rule: y follows last month's x.

    Parameters
    ----------
    months : int
        Number of changes.

    Returns
    -------
    np.ndarray
        Shape (months, 2): x_t = 0.5 x_(t-1) + noise, y_t = 0.8 x_(t-1) + 0.1 noise (seed 0).
    """
    rng = np.random.default_rng(0)
    changes = np.zeros((months, 2))
    for t in range(1, months):
        changes[t, 0] = 0.5 * changes[t - 1, 0] + rng.normal()
        changes[t, 1] = 0.8 * changes[t - 1, 0] + 0.1 * rng.normal()
    return changes


@pytest.mark.parametrize(
    ("architecture", "use_bias", "dropout"),
    [(NUMPY_A1, False, 0.0), (NUMPY_A2, False, 0.0), (NUMPY_A1, True, 0.3), (NUMPY_A2, True, 0.3)],
)
def test_backpropagation_matches_numerical_gradients(
    architecture: str, use_bias: bool, dropout: float
) -> None:
    """For both architectures, with and without biases and dropout, every parameter's
    backpropagated gradient of the mean squared error equals the central finite difference
    (step 1e-6) to within 1e-6 relative error. The dropout mask is held fixed by re-seeding its
    generator for every forward pass."""
    rng = np.random.default_rng(1)
    features, target = rng.normal(size=(7, 3)), rng.normal(size=(7, 1))
    network = NumpyNetwork(
        architecture=NUMPY_ARCHITECTURES[architecture],
        hidden_units=4,
        use_bias=use_bias,
        dropout=dropout,
    )
    # A1 has one hidden layer: input -> 4 -> 1; A2 has three: input -> 4 -> 4 -> 4 -> 1.
    hidden = 1 if architecture == NUMPY_A1 else 3
    sizes = [3, *([4] * hidden), 1]
    network.weights_ = [rng.normal(0, 0.5, (a, b)) for a, b in pairwise(sizes)]
    network.biases_ = [rng.normal(0, 0.5, (1, b)) for b in sizes[1:]] if use_bias else []

    def run() -> tuple[np.ndarray, Cache]:
        """
        One forward pass with the same dropout mask every time.

        Returns
        -------
        tuple of (np.ndarray, Cache)
            The network's predictions and its cache.
        """
        return network.forward(features, np.random.RandomState(5))

    prediction, cache = run()
    weight_gradients, bias_gradients = network.backward(target, prediction, cache)

    for layer, gradient in zip(
        [*network.weights_, *network.biases_], [*weight_gradients, *bias_gradients]
    ):
        numerical = np.zeros_like(layer)
        for index in np.ndindex(layer.shape):
            original = layer[index]
            layer[index] = original + 1e-6
            above = float(np.mean((target - run()[0]) ** 2))
            layer[index] = original - 1e-6
            below = float(np.mean((target - run()[0]) ** 2))
            layer[index] = original
            numerical[index] = (above - below) / 2e-6
        assert gradient == pytest.approx(numerical, rel=1e-6, abs=1e-9)


def test_one_optimizer_step_by_hand() -> None:
    """Weight 1, gradient 0.5, learning rate 0.1, first step:
    - SGD: 1 - 0.1 x 0.5 = 0.95
    - momentum: v = 0.1 x 0.5 = 0.05, so 1 - 0.1 x 0.05 = 0.995
    - AdaBelief: m = 0.05, s = 0.001 x (0.5 - 0.05)^2 = 0.0002025; bias-corrected 0.5 and
      0.2025, so 1 - 0.1 x 0.5 / 0.45 = 0.888889, also Muon's step for a bias"""
    expected = {"sgd": 0.95, "momentum": 0.995, "adabelief": 1 - 0.1 * 0.5 / 0.45}
    # Muon hands biases to AdaBelief, as the original networks do.
    bias = np.array([[1.0]])
    Optimizer("muon", 0.1).step((0, "bias"), bias, np.array([[0.5]]))
    assert bias[0, 0] == pytest.approx(expected["adabelief"], rel=1e-6)
    for name, value in expected.items():
        weights = np.array([[1.0]])
        Optimizer(name, 0.1).step((0, "weights"), weights, np.array([[0.5]]))
        assert weights[0, 0] == pytest.approx(value, rel=1e-6)


def test_newton_schulz_pushes_singular_values_towards_one() -> None:
    """A random 8 x 4 matrix whose singular values span about 1,000 to 1 comes out with them
    within a factor of 3 of each other (five quintic iterations land near 1, not on it), in the
    same shape."""
    matrix = np.random.default_rng(2).normal(size=(8, 4)) * np.array([10.0, 1.0, 0.1, 0.01])

    result = newton_schulz(matrix)

    before = np.linalg.svd(matrix, compute_uv=False)
    after = np.linalg.svd(result, compute_uv=False)
    assert result.shape == (8, 4)
    assert before[0] / before[-1] > 300 and after[0] / after[-1] < 3


@pytest.mark.parametrize("architecture", [NUMPY_A1, NUMPY_A2])
def test_the_numpy_networks_learn_the_rule_and_are_reproducible(architecture: str) -> None:
    """Through the shared lag-feature pipeline (scaling included), each network explains most
    of y's variance from last month's x (R^2 above 0.8 on its own rows), and two fits give
    identical predictions (fixed seed)."""
    changes = simulated_changes(300)
    features, target = lag_features(changes, 1), changes[1:, 1]
    settings = {
        "lags": 1,
        "hidden_units": 8,
        "optimizer": "adabelief",
        "learning_rate": 0.001,
        "batch_size": 16,
    }

    first = build_model(architecture, settings).fit(features, target).predict(features)
    second = build_model(architecture, settings).fit(features, target).predict(features)

    assert np.array_equal(first, second)
    assert 1 - np.var(target - first) / np.var(target) > 0.8


@pytest.mark.parametrize("model", [TENSORFLOW_MLP, PYTORCH_MLP])
def test_the_framework_networks_learn_the_rule_and_are_reproducible(model: str) -> None:
    """Through the shared lag-feature pipeline, the Keras and PyTorch networks (A2, Adam, some
    dropout and weight decay) explain most of y's variance from last month's x (R^2 above 0.8)
    and give identical predictions on refitting (fixed seed, deterministic operations)."""
    changes = simulated_changes(300)
    features, target = lag_features(changes, 1), changes[1:, 1]
    settings = {
        "lags": 1,
        "architecture": "A2",
        "hidden_units": 8,
        "optimizer": "adam",
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "dropout": 0.1,
        "batch_size": 16,
    }

    first = build_model(model, settings).fit(features, target).predict(features)
    second = build_model(model, settings).fit(features, target).predict(features)

    assert np.array_equal(first, second)
    assert 1 - np.var(target - first) / np.var(target) > 0.8


def test_lag_rows_become_sequences_in_time_order() -> None:
    """A row x1 x2 x3 y1 y2 y3 (each series' last three changes, most recent first) becomes the
    sequence (x3, y3), (x2, y2), (x1, y1): oldest month first, one column per series."""
    import torch

    from autoregressive_experiments.modeling.utilities.sequence_networks import to_sequence

    row = torch.tensor([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]])

    assert to_sequence(row).tolist() == [[[3.0, 30.0], [2.0, 20.0], [1.0, 10.0]]]


@pytest.mark.parametrize(
    ("model", "extra"),
    [
        (LSTM, {"layers": 2, "dropout": 0.1}),
        (GRU, {"layers": 1, "dropout": 0.1}),
        (TCN, {"layers": 2, "kernel_size": 3, "dropout": 0.1}),
        (NBEATS, {"layers": 2, "blocks": 2, "dropout": 0.1}),
        (DLINEAR, {"kernel_size": 3}),
    ],
)
def test_the_sequence_networks_learn_the_rule_and_are_reproducible(
    model: str, extra: dict[str, float]
) -> None:
    """Through the shared lag-feature pipeline, each sequence network explains most of y's
    variance from last month's x within a six-month window (R^2 above 0.8) and gives identical
    predictions on refitting (fixed seed, deterministic operations)."""
    changes = simulated_changes(300)
    features, target = lag_features(changes, 6), changes[6:, 1]
    settings = {
        "lags": 6,
        "hidden_units": 16,
        "optimizer": "adam",
        "learning_rate": 0.01,
        "use_bias": True,
        **extra,
    }

    first = build_model(model, settings).fit(features, target).predict(features)
    second = build_model(model, settings).fit(features, target).predict(features)

    assert np.array_equal(first, second)
    assert 1 - np.var(target - first) / np.var(target) > 0.8


def test_the_mlp_is_reproducible_through_the_lag_feature_pipeline() -> None:
    """Built by the shared pipeline (scaling included), two fits of the MLP on the same data
    give identical predictions (fixed seed)."""
    changes = np.random.default_rng(0).normal(size=(150, 2))
    features, target = lag_features(changes, 3), changes[3:, 1]
    settings = {"lags": 3, "hidden_layer_sizes": [8], "solver": "lbfgs", "max_iter": 500}

    first = build_model(MLP, settings).fit(features, target).predict(features)
    second = build_model(MLP, settings).fit(features, target).predict(features)

    assert np.array_equal(first, second)


def test_an_unknown_network_or_optimizer_is_refused() -> None:
    """Only the networks in NN_MODELS and the four original optimizers can be built."""
    with pytest.raises(ValueError, match="unknown neural network"):
        build_network("transformer", {})
    with pytest.raises(ValueError, match="unknown optimizer"):
        Optimizer("adam", 0.1)
