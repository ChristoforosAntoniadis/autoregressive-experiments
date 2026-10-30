"""
The PyTorch networks as one scikit-learn regressor: the multilayer perceptron, built like the
neural-network-numpy project's PyTorch networks (the layers of A1 or A2, biases optional, and
PyTorch's own defaults for initialization and optimizers), and the sequence networks of
sequence_networks (LSTM, GRU, TCN, N-BEATS, DLinear), all trained by the same mini-batch loop
with early stopping.

- A1: one hidden layer of sigmoid units, then one linear output unit
- A2: three hidden layers of ReLU units, then one linear output unit
- dropout after every hidden layer when the rate is above 0; biases on every layer, or none
- optimizers: SGD, momentum SGD (0.9), Adam or Muon, with optional weight decay
- early stopping on the last rows of each training window, in time order

PyTorch is imported when a network is first fitted, so modules that only list the networks
don't pay its start-up time.

Input: feature rows and targets, standardized by ml_forecasting's pipeline.

Output: a fitted estimator; nothing is written to disk.
"""

import copy
import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from autoregressive_experiments.configs.configurables import RANDOM_SEED

# The hidden layers of each architecture: name -> (hidden layers, their activation). The output
# layer is always one linear unit.
ARCHITECTURES = {"A1": (1, "sigmoid"), "A2": (3, "relu")}


def torch_optimizer(
    name: str, parameters: list[Any], learning_rate: float, weight_decay: float
) -> Any:
    """
    One of PyTorch's optimizers, as the original PyTorch networks build them.

    Parameters
    ----------
    name : str
        "sgd", "momentum", "adam" or "muon".
    parameters : list of torch.nn.Parameter
        The network's parameters. torch.optim.Muon accepts only 2-D weight matrices, so with
        Muon the biases get Adam at the same learning rate, as in the original networks.
    learning_rate : float
        Step size.
    weight_decay : float
        Weight decay (0 for none).

    Returns
    -------
    Any
        A torch.optim.Optimizer.

    Raises
    ------
    ValueError
        If `name` isn't one of the four.
    """
    import torch

    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(
            parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "muon":
        matrices = [p for p in parameters if p.ndim == 2]
        others = [p for p in parameters if p.ndim != 2]
        muon = torch.optim.Muon(matrices, lr=learning_rate, weight_decay=weight_decay)
        return MuonWithAdam(
            muon,
            torch.optim.Adam(others, lr=learning_rate, weight_decay=weight_decay)
            if others
            else None,
        )
    raise ValueError(f"unknown PyTorch optimizer {name!r}")


class MuonWithAdam:
    """
    Muon for the weight matrices and Adam for the biases, stepped together.

    Parameters
    ----------
    muon : torch.optim.Muon
        The optimizer of the 2-D weight matrices.
    adam : torch.optim.Adam or None
        The optimizer of every other parameter (the biases), or None without biases.
    """

    def __init__(self, muon: Any, adam: Any) -> None:
        self.muon = muon
        self.adam = adam

    def zero_grad(self) -> None:
        """
        Clear both optimizers' gradients.

        Returns
        -------
        None
        """
        self.muon.zero_grad()
        if self.adam is not None:
            self.adam.zero_grad()

    def step(self) -> None:
        """
        Step both optimizers.

        Returns
        -------
        None
        """
        self.muon.step()
        if self.adam is not None:
            self.adam.step()


class TorchNetwork(RegressorMixin, BaseEstimator):
    """
    A PyTorch network (A1, A2 or a sequence network), trained by a mini-batch loop with early
    stopping.

    Parameters
    ----------
    architecture : str, default="A1"
        A key of ARCHITECTURES, or one of sequence_networks.SEQUENCE_ARCHITECTURES.
    hidden_units : int, default=8
        Units in each hidden layer (recurrent units, TCN channels, N-BEATS layer width).
    layers : int, default=1
        Sequence networks only: recurrent layers, TCN blocks, or N-BEATS layers per block.
    kernel_size : int, default=3
        TCN: convolution kernel; DLinear: moving-average length.
    blocks : int, default=2
        N-BEATS: blocks in the chain.
    optimizer : str, default="adam"
        "sgd", "momentum", "adam" or "muon".
    learning_rate : float, default=0.001
        The optimizer's step size.
    use_bias : bool, default=False
        Whether every layer has a bias (the original A1 and A2 have none).
    weight_decay : float, default=0.0
        Weight decay.
    dropout : float, default=0.0
        Dropout rate after every hidden layer (0: no dropout layers).
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
    """

    def __init__(
        self,
        architecture: str = "A1",
        hidden_units: int = 8,
        layers: int = 1,
        kernel_size: int = 3,
        blocks: int = 2,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        use_bias: bool = False,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        batch_size: int = 16,
        epochs: int = 100,
        patience: int = 10,
        min_delta: float = 0.0001,
        min_epochs: int = 20,
        validation_share: float = 0.25,
    ) -> None:
        self.architecture = architecture
        self.hidden_units = hidden_units
        self.layers = layers
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.validation_share = validation_share

    def build(self, inputs: int) -> Any:
        """
        The untrained network.

        Parameters
        ----------
        inputs : int
            Width of the feature rows.

        Returns
        -------
        torch.nn.Module
            A1 / A2 as an nn.Sequential, or a sequence network; maps (batch, inputs) to
            (batch, 1).
        """
        from torch import nn

        from autoregressive_experiments.modeling.utilities.sequence_networks import (
            build_sequence_module,
        )

        if self.architecture not in ARCHITECTURES:
            return build_sequence_module(self.architecture, inputs, self)
        hidden, activation = ARCHITECTURES[self.architecture]
        modules: list[Any] = []
        width = inputs
        for _ in range(hidden):
            modules.append(nn.Linear(width, self.hidden_units, bias=self.use_bias))
            modules.append(nn.Sigmoid() if activation == "sigmoid" else nn.ReLU())
            if self.dropout > 0:
                modules.append(nn.Dropout(self.dropout))
            width = self.hidden_units
        modules.append(nn.Linear(width, 1, bias=self.use_bias))
        return nn.Sequential(*modules)

    def fit(self, features: np.ndarray, target: np.ndarray) -> "TorchNetwork":
        """
        Build the network and train it on time-ordered rows, stopping early on the last ones.

        Parameters
        ----------
        features : np.ndarray
            Training rows in time order, shape (rows, inputs).
        target : np.ndarray
            Their targets, shape (rows,).

        Returns
        -------
        TorchNetwork
            Itself, with the fitted module (model_) holding the best holdout weights.

        Notes
        -----
        Processing:
        1. Use deterministic algorithms on one thread, and seed the weights (torch.manual_seed)
           and the shuffles (a generator) with RANDOM_SEED.
        2. Hold out the last validation_share of the rows (at least one), in time order.
        3. Build the network: for A1 / A2, Linear -> activation (-> dropout) per hidden layer,
           then a one-unit Linear, all with or without biases; otherwise the sequence network
           (sequence_networks.build_sequence_module).
        4. Each epoch: shuffle the training rows, step the optimizer on each mini-batch's mean
           squared error, then score the holdout. Stop when it hasn't improved by min_delta
           for `patience` epochs after min_epochs, or when a loss stops being finite; keep the
           best holdout epoch's weights.
        """
        import torch
        from torch import nn

        torch.use_deterministic_algorithms(True)
        # One thread: these networks are tiny, so splitting each operation across threads costs
        # more than it saves (a TCN fit 5x faster, an LSTM 25%), and a fixed thread count keeps
        # every rerun identical to the last bit.
        torch.set_num_threads(1)
        torch.manual_seed(RANDOM_SEED)
        rows = torch.as_tensor(np.asarray(features, dtype="float32"))
        values = torch.as_tensor(np.asarray(target, dtype="float32").reshape(-1, 1))
        split = len(rows) - max(1, round(len(rows) * self.validation_share))
        train_x, train_y, hold_x, hold_y = (
            rows[:split],
            values[:split],
            rows[split:],
            values[split:],
        )
        model = self.build(rows.shape[1])
        optimizer = torch_optimizer(
            self.optimizer, list(model.parameters()), self.learning_rate, self.weight_decay
        )
        loss_fn = nn.MSELoss()
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        best_loss, best_state, waited = math.inf, None, 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            order = torch.randperm(len(train_x), generator=generator)
            batch_losses = []
            for start in range(0, len(train_x), self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(train_x[batch]), train_y[batch])
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                hold_loss = loss_fn(model(hold_x), hold_y).item()
            if not (math.isfinite(float(np.mean(batch_losses))) and math.isfinite(hold_loss)):
                break
            if hold_loss < best_loss - self.min_delta:
                best_loss, waited = hold_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            elif epoch > self.min_epochs:
                waited += 1
                if waited >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the targets of new rows.

        Parameters
        ----------
        features : np.ndarray
            Input rows, shape (rows, inputs).

        Returns
        -------
        np.ndarray
            Predictions, shape (rows,), float64.
        """
        import torch

        with torch.no_grad():
            output = self.model_(torch.as_tensor(np.asarray(features, dtype="float32")))
        return output.numpy().astype(float).ravel()
