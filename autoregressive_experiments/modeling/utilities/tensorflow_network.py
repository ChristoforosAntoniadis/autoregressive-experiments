"""
The TensorFlow (Keras) multilayer perceptron as a scikit-learn regressor, built like the
neural-network-numpy project's Keras networks: the layers of A1 or A2 (biases optional) and
Keras's own defaults (Glorot-uniform weights, Keras's optimizers), trained by a compiled
TensorFlow loop (tf.function) with the same mini-batches and early stopping as the PyTorch
network.

- A1: one hidden layer of sigmoid units, then one linear output unit
- A2: three hidden layers of ReLU units, then one linear output unit
- dropout after every hidden layer when the rate is above 0; biases on every layer, or none
- optimizers: SGD, momentum SGD (0.9), Adam or Muon, with optional weight decay
- early stopping on the last rows of each training window, in time order

TensorFlow is imported when a network is first fitted, so modules that only list the networks
don't pay its start-up time.

Input: feature rows and targets, standardized by ml_forecasting's pipeline.

Output: a fitted estimator; nothing is written to disk.
"""

import os
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from autoregressive_experiments.configs.configurables import RANDOM_SEED
from autoregressive_experiments.modeling.utilities.network import HUBER_DELTA

# The hidden layers of each architecture: name -> (hidden layers, their activation). The output
# layer is always one linear unit.
ARCHITECTURES = {"A1": (1, "sigmoid"), "A2": (3, "relu")}


def keras_optimizer(name: str, learning_rate: float, weight_decay: float) -> Any:
    """
    One of Keras's optimizers, as the original Keras networks build them.

    Parameters
    ----------
    name : str
        "sgd", "momentum", "adam" or "muon".
    learning_rate : float
        Step size.
    weight_decay : float
        Decoupled weight decay (0 for none); Muon also applies it to the parameters it hands to
        its internal Adam.

    Returns
    -------
    Any
        A keras.optimizers.Optimizer.

    Raises
    ------
    ValueError
        If `name` isn't one of the four.
    """
    import keras

    decay = {"weight_decay": weight_decay} if weight_decay > 0 else {}
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, **decay)
    if name == "momentum":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, **decay)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate, **decay)
    if name == "muon":
        if decay:
            decay["adam_weight_decay"] = weight_decay
        return keras.optimizers.Muon(learning_rate=learning_rate, **decay)
    raise ValueError(f"unknown Keras optimizer {name!r}")


class KerasNetwork(RegressorMixin, BaseEstimator):
    """
    A1 or A2 in Keras, trained with Keras's fit() and early stopping.

    Parameters
    ----------
    architecture : str, default="A1"
        A key of ARCHITECTURES.
    hidden_units : int, default=8
        Units in each hidden layer.
    optimizer : str, default="adam"
        "sgd", "momentum", "adam" or "muon".
    learning_rate : float, default=0.001
        The optimizer's step size.
    use_bias : bool, default=False
        Whether every layer has a bias (the original A1 and A2 have none).
    weight_decay : float, default=0.0
        Decoupled weight decay.
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
        Epochs before early stopping may act (Keras's start_from_epoch).
    validation_share : float, default=0.25
        Share of the rows, the last ones in time order, held out to decide when to stop.
    loss : str, default="mse"
        "mse" or "huber" (Keras's Huber with delta HUBER_DELTA), for training and early
        stopping.
    """

    def __init__(
        self,
        architecture: str = "A1",
        hidden_units: int = 8,
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
        loss: str = "mse",
    ) -> None:
        self.architecture = architecture
        self.hidden_units = hidden_units
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
        self.loss = loss

    def fit(self, features: np.ndarray, target: np.ndarray) -> "KerasNetwork":
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
        KerasNetwork
            Itself, with the fitted Keras model (model_) holding the best holdout weights.

        Notes
        -----
        Processing:
        1. Make TensorFlow's operations deterministic and seed Python, NumPy and TensorFlow
           (weights, dropout) with RANDOM_SEED (keras.utils.set_random_seed).
        2. Hold out the last validation_share of the rows (at least one), in time order.
        3. Build Dense -> activation (-> dropout) per hidden layer, then a one-unit linear
           Dense, all with or without biases.
        4. Compile one training step (forward, the loss, gradients, the Keras
           optimizer's update) and the holdout loss with tf.function, each traced once for
           any batch size.
        5. Each epoch: shuffle the training rows (a NumPy generator, same seed), run the step on
           each mini-batch, then score the holdout. Stop when it hasn't improved by min_delta
           for `patience` epochs after min_epochs, or when a loss stops being finite; keep the
           best holdout epoch's weights.

        Why a compiled loop instead of model.fit(): fit() adds a fixed cost to every epoch
        (data adapters, callbacks, a separate evaluation pass) that dominates on 40 to 200
        rows; the loop does the same training, as the PyTorch network's does, several times
        faster.
        """
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import keras
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()
        keras.utils.set_random_seed(RANDOM_SEED)
        rows = np.asarray(features, dtype="float32")
        values = np.asarray(target, dtype="float32").reshape(-1, 1)
        split = len(rows) - max(1, round(len(rows) * self.validation_share))
        train_x, train_y = rows[:split], values[:split]
        hold_x, hold_y = tf.constant(rows[split:]), tf.constant(values[split:])
        hidden, activation = ARCHITECTURES[self.architecture]
        inputs = keras.Input(shape=(rows.shape[1],))
        layer = inputs
        for _ in range(hidden):
            layer = keras.layers.Dense(self.hidden_units, use_bias=self.use_bias)(layer)
            layer = keras.layers.Activation(activation)(layer)
            if self.dropout > 0:
                layer = keras.layers.Dropout(self.dropout)(layer)
        model = keras.Model(inputs, keras.layers.Dense(1, use_bias=self.use_bias)(layer))
        optimizer = keras_optimizer(self.optimizer, self.learning_rate, self.weight_decay)
        optimizer.build(model.trainable_variables)
        loss_fn = (
            keras.losses.MeanSquaredError()
            if self.loss == "mse"
            else keras.losses.Huber(delta=HUBER_DELTA)
        )
        signature = [
            tf.TensorSpec([None, rows.shape[1]], tf.float32),
            tf.TensorSpec([None, 1], tf.float32),
        ]

        @tf.function(input_signature=signature)
        def train_step(batch_x: Any, batch_y: Any) -> Any:
            """
            One optimizer step on one mini-batch (dropout on).

            Parameters
            ----------
            batch_x : tf.Tensor
                Feature rows, shape (rows, inputs).
            batch_y : tf.Tensor
                Their targets, shape (rows, 1).

            Returns
            -------
            tf.Tensor
                The batch's loss before the step.
            """
            with tf.GradientTape() as tape:
                loss = loss_fn(batch_y, model(batch_x, training=True))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function(input_signature=signature)
        def holdout_loss(batch_x: Any, batch_y: Any) -> Any:
            """
            The loss without dropout.

            Parameters
            ----------
            batch_x : tf.Tensor
                Feature rows, shape (rows, inputs).
            batch_y : tf.Tensor
                Their targets, shape (rows, 1).

            Returns
            -------
            tf.Tensor
                The loss.
            """
            return loss_fn(batch_y, model(batch_x, training=False))

        shuffle_rng = np.random.RandomState(RANDOM_SEED)
        best_loss, best_weights, waited = np.inf, None, 0
        for epoch in range(1, self.epochs + 1):
            order = shuffle_rng.permutation(len(train_x))
            batch_losses = [
                float(
                    train_step(
                        train_x[order[start : start + self.batch_size]],
                        train_y[order[start : start + self.batch_size]],
                    )
                )
                for start in range(0, len(order), self.batch_size)
            ]
            hold = float(holdout_loss(hold_x, hold_y))
            if not (np.isfinite(np.mean(batch_losses)) and np.isfinite(hold)):
                break
            if hold < best_loss - self.min_delta:
                best_loss, waited = hold, 0
                best_weights = model.get_weights()
            elif epoch > self.min_epochs:
                waited += 1
                if waited >= self.patience:
                    break
        if best_weights is not None:
            model.set_weights(best_weights)
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
        rows = np.asarray(features, dtype="float32")
        return np.asarray(self.model_(rows, training=False)).astype(float).ravel()
