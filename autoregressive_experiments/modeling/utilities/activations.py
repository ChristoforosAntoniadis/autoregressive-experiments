"""
Hidden-layer activations of the hand-written networks, and their derivatives for
backpropagation: the logistic sigmoid (A1) and ReLU (A2). The output layer is linear, so it
needs neither.

Input: pre-activations from network.NumpyNetwork.

Output: arrays; nothing is written to disk.
"""

import numpy as np


def activate(name: str, values: np.ndarray) -> np.ndarray:
    """
    Apply a hidden layer's activation.

    Parameters
    ----------
    name : str
        "sigmoid" or "relu".
    values : np.ndarray
        Pre-activations, shape (rows, units).

    Returns
    -------
    np.ndarray
        The activations, same shape.

    Notes
    -----
    The sigmoid's input is clipped to [-500, 500] first, as in the original network, so exp()
    can't overflow.
    """
    if name == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(values, -500, 500)))
    return np.maximum(0, values)


def activation_gradient(name: str, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """
    The derivative of a hidden layer's activation, element by element.

    Parameters
    ----------
    name : str
        "sigmoid" or "relu".
    pre : np.ndarray
        The layer's pre-activations, shape (rows, units).
    post : np.ndarray
        Its activations, same shape.

    Returns
    -------
    np.ndarray
        d activation / d pre-activation, same shape: s (1 - s) for the sigmoid, 1 where the
        pre-activation is positive and 0 elsewhere for ReLU.
    """
    if name == "sigmoid":
        return post * (1 - post)
    return (pre > 0).astype(float)
