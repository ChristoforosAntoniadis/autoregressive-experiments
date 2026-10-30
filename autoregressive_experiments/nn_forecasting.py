"""
The neural-network family, each predicting next month's change of one series from the last k
monthly changes of both series: scikit-learn's multilayer perceptron (MLPRegressor), two
networks written by hand in NumPy (A1 and A2), and a TensorFlow (Keras) and a PyTorch
multilayer perceptron, each A1 or A2 by tuning, built with the framework's own defaults.

A1 and A2 are the regression networks of the neural-network-numpy project, ported here (the
regression part only): the same layers, activations, loss, optimizers, weight initialization,
mini-batch training and early stopping. Only the hidden-layer width and the batch size differ,
and both are tuned, since this data has 2 to 16 inputs and 40 to 200 training rows per fit.

- A1: one hidden layer of sigmoid units, then one linear output unit
- A2: three hidden layers of ReLU units, then one linear output unit
- no biases; weights drawn from a normal distribution with SD 0.1
- loss: mean squared error
- optimizers: SGD, momentum SGD, AdaBelief or Muon, with their published defaults
- training: shuffled mini-batches for up to `epochs` epochs; early stopping on a time-ordered
  holdout of the last rows of each training window, keeping the best weights

Every network uses the same lag-feature pipeline as the ML family (``ml_forecasting``): the same
features, standardized (features and target) inside every fit, refitted at every forecast origin
and forecasting several months ahead recursively. Only the networks are defined here;
ml_forecasting's build_model wraps them in that pipeline. The hand-written networks' building
blocks (activations, optimizers, the network and its training) are in
``modeling/utilities``.

Every setting, the lag count included, is chosen per series by the time-aware search of
``tuning`` on the training split (random draws, or Optuna's TPE sampler for the TensorFlow and
PyTorch networks), from the search spaces in the ``nn`` section of
``configs/hyperparameters.json``, and then kept fixed.

Input: none; the networks are built by ``ml_forecasting`` and evaluated by
``forecast_evaluation``.

Output: network builders; nothing is written to disk.
"""

from collections.abc import Mapping
from typing import Any

from sklearn.neural_network import MLPRegressor

from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.configs.configurables import RANDOM_SEED
from autoregressive_experiments.modeling.utilities.network import NumpyNetwork
from autoregressive_experiments.modeling.utilities.tensorflow_network import KerasNetwork
from autoregressive_experiments.modeling.utilities.torch_network import TorchNetwork

# Settings; each value's reasoning is in configs/hyperparameters.json.
SETTINGS = HYPERPARAMETERS.nn
MLP, NUMPY_A1, NUMPY_A2 = "MLP", "NumPy A1", "NumPy A2"
TENSORFLOW_MLP, PYTORCH_MLP = "TensorFlow MLP", "PyTorch MLP"
NN_MODELS = (MLP, NUMPY_A1, NUMPY_A2, TENSORFLOW_MLP, PYTORCH_MLP)
# The networks tuned by Optuna (their spaces include numeric ranges); the others by
# scikit-learn's RandomizedSearchCV.
OPTUNA_MODELS = (TENSORFLOW_MLP, PYTORCH_MLP)
# Model -> its search space: setting name -> the values tried.
NN_SPACES: dict[str, Mapping[str, Any]] = {
    MLP: SETTINGS.mlp,
    NUMPY_A1: SETTINGS.numpy_a1,
    NUMPY_A2: SETTINGS.numpy_a2,
    TENSORFLOW_MLP: SETTINGS.tensorflow_mlp,
    PYTORCH_MLP: SETTINGS.pytorch_mlp,
}
# The hand-written networks: model name -> their architecture in modeling/utilities/network.py.
NUMPY_ARCHITECTURES = {NUMPY_A1: "A1", NUMPY_A2: "A2"}


def build_network(model: str, options: Mapping[str, Any]) -> Any:
    """
    An unfitted network with the given settings.

    Parameters
    ----------
    model : str
        One of NN_MODELS.
    options : Mapping of str to Any
        Its keyword arguments from its search space, without "lags".

    Returns
    -------
    Any
        The unfitted estimator; ml_forecasting.build_model adds the scaling.

    Raises
    ------
    ValueError
        If `model` isn't one of NN_MODELS.
    """
    if model == MLP:
        return MLPRegressor(**options, random_state=RANDOM_SEED)
    if model in NUMPY_ARCHITECTURES:
        return NumpyNetwork(architecture=NUMPY_ARCHITECTURES[model], **options)
    if model == TENSORFLOW_MLP:
        return KerasNetwork(**options)
    if model == PYTORCH_MLP:
        return TorchNetwork(**options)
    raise ValueError(f"unknown neural network {model!r}")
