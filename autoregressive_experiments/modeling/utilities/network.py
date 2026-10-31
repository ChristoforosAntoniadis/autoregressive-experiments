"""
The hand-written networks A1 and A2 as a scikit-learn regressor: forward pass, backpropagation
of the mean squared error (or the Huber loss), mini-batch training and early stopping, ported
from the neural-network-numpy project's regression networks.

- A1: one hidden layer of sigmoid units, then one linear output unit
- A2: three hidden layers of ReLU units, then one linear output unit
- weights drawn from N(0, 0.1^2); biases, when used, start at 0; mean squared error, or the
  Huber loss (Huber 1964), squared within HUBER_DELTA of the target and absolute beyond
- dropout (inverted, Srivastava et al. 2014) after every hidden layer when the rate is above 0

Input: feature rows and targets, standardized by ml_forecasting's pipeline.

Output: a fitted estimator; nothing is written to disk.
"""

from itertools import pairwise
from typing import NamedTuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.configs.configurables import RANDOM_SEED
from autoregressive_experiments.modeling.utilities.activations import activate, activation_gradient
from autoregressive_experiments.modeling.utilities.optimizers import Optimizer

# The hand-written architectures: name -> (hidden layers, their activation). The output layer
# is always one linear unit.
ARCHITECTURES = {"A1": (1, "sigmoid"), "A2": (3, "relu")}
# The initial weights' standard deviation, the original networks' fixed scale.
INIT_SD = 0.1
# Where the Huber loss turns from squared to absolute; its reasoning is in the config.
HUBER_DELTA = HYPERPARAMETERS.outliers.huber_delta


def loss_value(loss: str, error: np.ndarray) -> float:
    """
    The mean loss of a batch's errors.

    Parameters
    ----------
    loss : str
        "mse" or "huber".
    error : np.ndarray
        Prediction minus target, any shape.

    Returns
    -------
    float
        Mean squared error, or the mean Huber loss: e^2 / 2 within HUBER_DELTA, else
        HUBER_DELTA x (|e| - HUBER_DELTA / 2).
    """
    if loss == "mse":
        return float(np.mean(error**2))
    size = np.abs(error)
    return float(
        np.mean(
            np.where(size <= HUBER_DELTA, error**2 / 2, HUBER_DELTA * (size - HUBER_DELTA / 2))
        )
    )


def loss_gradient(loss: str, error: np.ndarray) -> np.ndarray:
    """
    The derivative of loss_value() with respect to each prediction.

    Parameters
    ----------
    loss : str
        "mse" or "huber".
    error : np.ndarray
        Prediction minus target, shape (rows, 1).

    Returns
    -------
    np.ndarray
        Same shape: 2 x e / rows for the mean squared error, the error clipped to
        +/- HUBER_DELTA / rows for the Huber loss.
    """
    if loss == "mse":
        return 2 / len(error) * error
    return np.clip(error, -HUBER_DELTA, HUBER_DELTA) / len(error)


class Cache(NamedTuple):
    """
    What one forward pass keeps for backpropagation.

    Attributes
    ----------
    inputs : list of np.ndarray
        Each layer's input (after the layer below's dropout), shape (rows, fan_in).
    pre : list of np.ndarray
        Each hidden layer's pre-activations, shape (rows, units).
    post : list of np.ndarray
        Each hidden layer's activations before dropout, same shape.
    masks : list of np.ndarray or None
        Each hidden layer's dropout mask (0 or 1 / (1 - rate)), or None without dropout.
    """

    inputs: list[np.ndarray]
    pre: list[np.ndarray]
    post: list[np.ndarray]
    masks: list[np.ndarray | None]


class NumpyNetwork(RegressorMixin, BaseEstimator):
    """
    A1 or A2 as a scikit-learn regressor: hand-written forward and backward passes, mini-batch
    training and early stopping.

    Parameters
    ----------
    architecture : str, default="A1"
        A key of ARCHITECTURES: "A1" or "A2".
    hidden_units : int, default=8
        Units in each hidden layer.
    optimizer : str, default="adabelief"
        One of OPTIMIZERS.
    learning_rate : float, default=0.001
        The optimizer's step size.
    use_bias : bool, default=False
        Whether every layer has a bias (the original A1 and A2 have none).
    dropout : float, default=0.0
        Dropout rate after every hidden layer (0: no dropout).
    batch_size : int, default=16
        Rows per mini-batch.
    epochs : int, default=100
        The most passes over the training rows.
    patience : int, default=10
        Epochs without a holdout improvement before training stops.
    min_delta : float, default=0.0001
        The least holdout-loss decrease that counts as an improvement.
    min_epochs : int, default=20
        Epochs before early stopping may act.
    validation_share : float, default=0.25
        Share of the rows, the last ones in time order, held out to decide when to stop.
    loss : str, default="mse"
        "mse" (the original networks' loss) or "huber", for training and early stopping.

    Notes
    -----
    Explicit keyword parameters, so scikit-learn's search and cloning work on it as on any
    estimator. Training follows the original networks' trainer exactly; only the holdout is
    the end of the window instead of a separate validation split.
    """

    def __init__(
        self,
        architecture: str = "A1",
        hidden_units: int = 8,
        optimizer: str = "adabelief",
        learning_rate: float = 0.001,
        use_bias: bool = False,
        dropout: float = 0.0,
        batch_size: int = 16,
        epochs: int = 100,
        patience: int = 10,
        min_delta: float = 0.0001,
        min_epochs: int = 20,
        validation_share: float = 0.25,
        loss: str = "mse",
    ) -> None:
        self.architecture = architecture
        self.hidden_units = hidden_units
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.validation_share = validation_share
        self.loss = loss

    def forward(
        self, features: np.ndarray, dropout_rng: np.random.RandomState | None = None
    ) -> tuple[np.ndarray, Cache]:
        """
        Run the network on a batch.

        Parameters
        ----------
        features : np.ndarray
            Input rows, shape (rows, inputs).
        dropout_rng : np.random.RandomState or None, default=None
            Draws the dropout masks in training; None runs without dropout (evaluation).

        Returns
        -------
        tuple of (np.ndarray, Cache)
            The predictions, shape (rows, 1), and what backward() needs.

        Notes
        -----
        Per layer: Z = X W (+ b); hidden layers then apply the activation and, in training,
        multiply by a dropout mask (random >= rate) / (1 - rate), so the expected activation is
        unchanged and evaluation needs no rescaling.
        """
        _, activation = ARCHITECTURES[self.architecture]
        cache = Cache([], [], [], [])
        output = features
        for index, weights in enumerate(self.weights_):
            cache.inputs.append(output)
            output = output @ weights
            if self.use_bias:
                output = output + self.biases_[index]
            if index < len(self.weights_) - 1:
                cache.pre.append(output)
                output = activate(activation, output)
                cache.post.append(output)
                mask = None
                if dropout_rng is not None and self.dropout > 0:
                    keep = dropout_rng.random_sample(output.shape) >= self.dropout
                    mask = keep / (1 - self.dropout)
                    output = output * mask
                cache.masks.append(mask)
        return output, cache

    def backward(
        self, target: np.ndarray, prediction: np.ndarray, cache: Cache
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Backpropagate the loss to every weight matrix and bias.

        Parameters
        ----------
        target : np.ndarray
            The batch's targets, shape (rows, 1).
        prediction : np.ndarray
            The network's predictions, same shape.
        cache : Cache
            From the forward() that made `prediction`.

        Returns
        -------
        tuple of (list of np.ndarray, list of np.ndarray)
            d loss / d weights and d loss / d bias per layer, in layer order (the bias list is
            empty without biases).

        Notes
        -----
        Processing:
        1. d loss / d prediction from loss_gradient().
        2. From the last layer back: d loss / d W = input' @ grad and d loss / d b = the sum of
           grad over the rows; the gradient for the layer below is grad @ W', multiplied by
           that layer's dropout mask (the values it dropped get none) and its activation
           derivative.
        """
        _, activation = ARCHITECTURES[self.architecture]
        gradient = loss_gradient(self.loss, prediction - target)
        weight_gradients, bias_gradients = [], []
        for index in range(len(self.weights_) - 1, -1, -1):
            weight_gradients.append(cache.inputs[index].T @ gradient)
            if self.use_bias:
                bias_gradients.append(gradient.sum(axis=0, keepdims=True))
            if index > 0:
                gradient = gradient @ self.weights_[index].T
                mask = cache.masks[index - 1]
                if mask is not None:
                    gradient = gradient * mask
                gradient = gradient * activation_gradient(
                    activation, cache.pre[index - 1], cache.post[index - 1]
                )
        return weight_gradients[::-1], bias_gradients[::-1]

    def fit(self, features: np.ndarray, target: np.ndarray) -> "NumpyNetwork":
        """
        Train on time-ordered rows, stopping early on the last ones.

        Parameters
        ----------
        features : np.ndarray
            Training rows in time order, shape (rows, inputs).
        target : np.ndarray
            Their targets, shape (rows,).

        Returns
        -------
        NumpyNetwork
            Itself, with the best parameters found (weights_, biases_) and the epochs run
            (epochs_run_).

        Notes
        -----
        Processing:
        1. Hold out the last validation_share of the rows (at least one), in time order.
        2. Draw the weights layer by layer from N(0, INIT_SD^2), seeded with RANDOM_SEED;
           biases start at zero, so turning them on doesn't change any drawn weight.
        3. Each epoch: shuffle the training rows (a second generator, same seed), and for every
           mini-batch run forward (dropout masks from a third generator, seed + 1, as in the
           original), backward and one optimizer step per parameter; then score the holdout
           without dropout.
        4. Stop when the holdout loss hasn't improved by min_delta for `patience` epochs, once
           min_epochs have passed, or when a loss stops being finite; keep the parameters of
           the best holdout epoch.

        Shuffling only reorders the training rows within the window, so no later month enters
        the training batches; the holdout is always the latest months.
        """
        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float).reshape(-1, 1)
        split = len(features) - max(1, round(len(features) * self.validation_share))
        train_x, train_y = features[:split], target[:split]
        hold_x, hold_y = features[split:], target[split:]
        hidden, _ = ARCHITECTURES[self.architecture]
        sizes = [features.shape[1], *([self.hidden_units] * hidden), 1]
        weights_rng = np.random.RandomState(RANDOM_SEED)
        self.weights_ = [
            weights_rng.normal(0.0, INIT_SD, size=(fan_in, fan_out))
            for fan_in, fan_out in pairwise(sizes)
        ]
        self.biases_ = [np.zeros((1, fan_out)) for fan_out in sizes[1:]] if self.use_bias else []
        shuffle_rng = np.random.RandomState(RANDOM_SEED)
        dropout_rng = np.random.RandomState(RANDOM_SEED + 1)
        optimizer = Optimizer(self.optimizer, self.learning_rate)
        best_loss, best, waited = np.inf, None, 0
        self.epochs_run_ = 0
        for epoch in range(1, self.epochs + 1):
            order = shuffle_rng.permutation(len(train_x))
            batch_losses = []
            for start in range(0, len(order), self.batch_size):
                rows = order[start : start + self.batch_size]
                prediction, cache = self.forward(train_x[rows], dropout_rng)
                batch_losses.append(loss_value(self.loss, prediction - train_y[rows]))
                weight_gradients, bias_gradients = self.backward(train_y[rows], prediction, cache)
                for index, gradient in enumerate(weight_gradients):
                    optimizer.step((index, "weights"), self.weights_[index], gradient)
                for index, gradient in enumerate(bias_gradients):
                    optimizer.step((index, "bias"), self.biases_[index], gradient)
            hold_loss = loss_value(self.loss, self.forward(hold_x)[0] - hold_y)
            self.epochs_run_ = epoch
            if not (np.isfinite(np.mean(batch_losses)) and np.isfinite(hold_loss)):
                break
            if hold_loss < best_loss - self.min_delta:
                best_loss, waited = hold_loss, 0
                best = (
                    [layer.copy() for layer in self.weights_],
                    [layer.copy() for layer in self.biases_],
                )
            elif epoch <= self.min_epochs:
                waited = 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        if best is not None:
            self.weights_, self.biases_ = best
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the targets of new rows (no dropout).

        Parameters
        ----------
        features : np.ndarray
            Input rows, shape (rows, inputs).

        Returns
        -------
        np.ndarray
            Predictions, shape (rows,).
        """
        return self.forward(np.asarray(features, dtype=float))[0].ravel()
