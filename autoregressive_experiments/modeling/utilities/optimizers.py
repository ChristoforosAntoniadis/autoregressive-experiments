"""
The hand-written networks' optimizers: SGD, momentum SGD, AdaBelief and Muon, with the constants
the original networks used.

Input: weight matrices and their gradients from network.NumpyNetwork.

Output: weights updated in place; nothing is written to disk.
"""

import numpy as np

OPTIMIZERS = ("sgd", "momentum", "adabelief", "muon")
# The optimizers' own constants, as the original networks set them: momentum's averaging
# weight; AdaBelief's (Zhuang et al. 2020) moment decays and epsilon; Muon's momentum, Nesterov
# look-ahead and Newton-Schulz steps (Jordan et al. 2024).
MOMENTUM_BETA = 0.9
ADABELIEF_BETAS, ADABELIEF_EPSILON = (0.9, 0.999), 1e-8
MUON_MOMENTUM, MUON_NESTEROV, MUON_STEPS = 0.95, True, 5


def newton_schulz(
    matrix: np.ndarray, steps: int = MUON_STEPS, epsilon: float = 1e-7
) -> np.ndarray:
    """
    Approximately orthogonalize a matrix with Newton-Schulz iterations (Muon's core step).

    Parameters
    ----------
    matrix : np.ndarray
        A gradient-like matrix, shape (fan_in, fan_out).
    steps : int, default=MUON_STEPS
        Iterations.
    epsilon : float, default=1e-7
        Floor of the norm the matrix is first divided by, so a zero matrix stays zero.

    Returns
    -------
    np.ndarray
        Same shape: roughly the nearest semi-orthogonal matrix (its singular values pushed
        towards 1).

    Notes
    -----
    Processing:
    1. Work on the wide orientation (transpose a tall matrix), so X X' is the small Gram matrix.
    2. Scale to unit Frobenius norm, so every singular value is at most 1.
    3. Repeat X <- a X + (b G + c G^2) X with G = X X' and the quintic coefficients
       (3.4445, -4.775, 2.0315) of Jordan et al. (2024).
    """
    a, b, c = 3.4445, -4.775, 2.0315
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.T if transposed else matrix
    x = x / max(float(np.linalg.norm(x)), epsilon)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if transposed else x


class Optimizer:
    """
    One of the original networks' optimizers, updating each parameter in place.

    Parameters
    ----------
    name : str
        One of OPTIMIZERS.
    learning_rate : float
        Step size.

    Notes
    -----
    State (velocities, moments, step counts) is kept per parameter, by its key: (layer index,
    "weights" or "bias"). Muon only orthogonalizes weight matrices; biases get AdaBelief with
    the same learning rate, as in the original networks.
    """

    def __init__(self, name: str, learning_rate: float) -> None:
        if name not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {name!r}")
        self.name = name
        self.learning_rate = learning_rate
        self.first: dict[tuple[int, str], np.ndarray] = {}
        self.second: dict[tuple[int, str], np.ndarray] = {}
        self.steps: dict[tuple[int, str], int] = {}

    def adabelief(self, key: tuple[int, str], param: np.ndarray, gradient: np.ndarray) -> None:
        """
        One AdaBelief step on one parameter, in place.

        Parameters
        ----------
        key : tuple of (int, str)
            The parameter's (layer index, "weights" or "bias"), which keys its state.
        param : np.ndarray
            The parameter; changed in place.
        gradient : np.ndarray
            d loss / d parameter, same shape.

        Returns
        -------
        None
            The parameter is updated in place.

        Notes
        -----
        m <- b1 m + (1 - b1) g; s <- b2 s + (1 - b2) (g - m)^2; with bias-corrected m_hat and
        s_hat, param <- param - lr m_hat / (sqrt(s_hat) + eps) (Zhuang et al. 2020).
        """
        beta1, beta2 = ADABELIEF_BETAS
        first = self.first.setdefault(key, np.zeros_like(param))
        second = self.second.setdefault(key, np.zeros_like(param))
        self.steps[key] = count = self.steps.get(key, 0) + 1
        first[:] = beta1 * first + (1 - beta1) * gradient
        second[:] = beta2 * second + (1 - beta2) * (gradient - first) ** 2
        corrected_first = first / (1 - beta1**count)
        corrected_second = second / (1 - beta2**count)
        param -= (
            self.learning_rate * corrected_first / (np.sqrt(corrected_second) + ADABELIEF_EPSILON)
        )

    def step(self, key: tuple[int, str], param: np.ndarray, gradient: np.ndarray) -> None:
        """
        Update one parameter in place.

        Parameters
        ----------
        key : tuple of (int, str)
            The parameter's (layer index, "weights" or "bias"), which keys its state.
        param : np.ndarray
            A weight matrix, shape (fan_in, fan_out), or a bias, shape (1, fan_out); changed in
            place.
        gradient : np.ndarray
            d loss / d parameter, same shape.

        Returns
        -------
        None
            The parameter is updated in place.

        Notes
        -----
        - sgd: p <- p - lr g
        - momentum: v <- beta v + (1 - beta) g; p <- p - lr v
        - adabelief: see adabelief()
        - muon, weights: buffer <- mu buffer + (1 - mu) g; Nesterov direction
          (1 - mu) g + mu buffer; w <- w - lr sqrt(max(1, fan_out / fan_in))
          newton_schulz(direction); biases: AdaBelief
        """
        rate = self.learning_rate
        if self.name == "sgd":
            param -= rate * gradient
        elif self.name == "momentum":
            velocity = self.first.setdefault(key, np.zeros_like(param))
            velocity[:] = MOMENTUM_BETA * velocity + (1 - MOMENTUM_BETA) * gradient
            param -= rate * velocity
        elif self.name == "adabelief" or key[1] != "weights":
            self.adabelief(key, param, gradient)
        else:
            buffer = self.first.setdefault(key, np.zeros_like(param))
            buffer[:] = MUON_MOMENTUM * buffer + (1 - MUON_MOMENTUM) * gradient
            direction = (
                (1 - MUON_MOMENTUM) * gradient + MUON_MOMENTUM * buffer
                if MUON_NESTEROV
                else buffer
            )
            fan_in, fan_out = param.shape
            param -= rate * np.sqrt(max(1.0, fan_out / fan_in)) * newton_schulz(direction)
