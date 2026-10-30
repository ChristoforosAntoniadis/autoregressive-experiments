"""
Forecast scores shared by the validation (tuning) and the test (forecast_evaluation): point
accuracy (MAE, MSE, RMSE, MAPE, MASE), the Diebold-Mariano test, and the quantile scores of the
uncertainty bands.

Input: actual and forecast levels.

Output: scores; nothing is written to disk.
"""

import numpy as np
from scipy import stats

from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS

# Settings; each value's reasoning is in configs/hyperparameters.json.
# Forecast horizons that are scored, in months ahead; every model forecasts up to the longest.
HORIZONS = HYPERPARAMETERS.forecast_evaluation.horizons
MAX_HORIZON = max(HORIZONS)
# Ordered low to high: interval_scores unpacks them as 95% low, 80% low, median, 80% high,
# 95% high.
QUANTILES = HYPERPARAMETERS.forecast_evaluation.quantiles
METRICS = ("MAE", "MSE", "RMSE", "MAPE", "MASE")


def mase_scale(train_levels: np.ndarray) -> float:
    """
    In-sample MAE of the one-step naive forecast: the mean absolute monthly change.

    Parameters
    ----------
    train_levels : np.ndarray
        The training split's levels of one series, shape (months,), trillions.

    Returns
    -------
    float
        The MASE denominator, trillions.

    Notes
    -----
    Taken from the training split only and kept fixed for validation and test, so MASE is
    comparable across splits and the scaling never uses out-of-sample data (Hyndman and
    Koehler 2006).
    """
    return float(np.mean(np.abs(np.diff(train_levels))))


def accuracy(actual: np.ndarray, forecast: np.ndarray, scale: float) -> dict[str, float]:
    """
    Point-accuracy metrics of a set of forecasts.

    Parameters
    ----------
    actual : np.ndarray
        Actual levels, shape (forecasts,), trillions; must be non-zero (for MAPE).
    forecast : np.ndarray
        Forecast levels, same shape, trillions.
    scale : float
        MASE denominator from mase_scale, trillions.

    Returns
    -------
    dict of str to float
        "MAE" and "RMSE" (trillions), "MSE" (trillions squared), "MAPE" (%), "MASE" (unitless;
        below 1 is better than the training split's one-month naive).
    """
    errors = np.asarray(actual) - np.asarray(forecast)
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors**2))
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(100 * np.mean(np.abs(errors / np.asarray(actual)))),
        "MASE": mae / scale,
    }


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> tuple[float, float]:
    """
    Diebold-Mariano test of equal expected loss, with the Harvey-Leybourne-Newbold small-sample
    correction. A negative statistic means model a has the lower loss. Two-sided p-value from a
    t distribution with n - 1 degrees of freedom.

    Parameters
    ----------
    loss_a : np.ndarray
        Loss of model a per forecast (e.g. absolute or squared error), shape (forecasts,),
        in forecast-origin order.
    loss_b : np.ndarray
        Loss of model b for the same targets, same shape and order.
    horizon : int
        Forecast horizon h in months; sets the number of autocovariance lags (h - 1).

    Returns
    -------
    tuple of (float, float)
        The corrected DM statistic and its two-sided p-value. (0.0, 1.0) if the loss
        differences have no variance (e.g. identical forecasts).

    Notes
    -----
    Processing:
    1. Loss differences d = loss_a - loss_b and their deviations from the mean.
    2. Autocovariances of d at lags 0..h-1 (divided by n, not n - lag).
    3. Long-run variance: lag 0 plus twice the others (rectangular kernel); fall back to lag 0
       if that isn't positive, and return "no difference" if even that is zero.
    4. DM = mean(d) / sqrt(long-run variance / n).
    5. HLN correction: multiply by sqrt((n + 1 - 2h + h(h - 1) / n) / n).
    6. Two-sided p-value from Student-t with n - 1 degrees of freedom.

    Why HLN and a t distribution: with 56 or fewer forecasts per horizon, the plain DM test
    over-rejects; the correction and the t reference make it closer to its nominal size
    (Harvey, Leybourne and Newbold 1997).
    """
    differences = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    count = len(differences)
    deviations = differences - differences.mean()
    # h-step forecasts from consecutive origins overlap, so their loss differences are
    # autocorrelated up to lag h - 1.
    autocovariances = [
        np.sum(deviations[lag:] * deviations[: count - lag]) / count for lag in range(horizon)
    ]
    long_run_variance = autocovariances[0] + 2 * sum(autocovariances[1:])
    # shortcut: the rectangular kernel can turn negative for h > 1; fall back to the lag-0
    # variance then, as common implementations do. Upgrade to a Bartlett kernel if this recurs.
    if long_run_variance <= 0:
        long_run_variance = autocovariances[0]
    if long_run_variance == 0:
        return 0.0, 1.0
    statistic = differences.mean() / np.sqrt(long_run_variance / count)
    statistic *= np.sqrt((count + 1 - 2 * horizon + horizon * (horizon - 1) / count) / count)
    return float(statistic), float(2 * stats.t.sf(abs(statistic), count - 1))


def empirical_band(errors: np.ndarray) -> np.ndarray:
    """
    The QUANTILES of past forecast errors: the offsets of an empirical prediction interval.

    Parameters
    ----------
    errors : np.ndarray
        Past out-of-sample errors (actual - forecast) at one horizon, shape (errors,), trillions.

    Returns
    -------
    np.ndarray
        One offset per QUANTILES level, shape (5,), trillions; NaN if there are no errors.

    Notes
    -----
    Adding these to a forecast gives its band: an empirical prediction interval (Williams and
    Goodman 1971; Lee and Scholtes 2014), the split-conformal construction of Vovk et al.
    (2005), valid for any model because it relies only on the model's own errors. numpy's
    default (linear) quantile interpolation is used.
    """
    if len(errors) == 0:
        return np.full(len(QUANTILES), np.nan)
    return np.quantile(np.asarray(errors, dtype=float), QUANTILES)


def pinball_loss(actual: np.ndarray, quantiles: np.ndarray) -> float:
    """
    Mean quantile loss over all forecasts and QUANTILES: under-forecasts cost q per unit,
    over-forecasts 1 - q. Lower is better; it rewards intervals that are calibrated and narrow.

    Parameters
    ----------
    actual : np.ndarray
        Actual levels, shape (forecasts,), trillions.
    quantiles : np.ndarray
        Quantile forecasts, shape (forecasts, 5), trillions, columns in QUANTILES order.

    Returns
    -------
    float
        Mean pinball loss, trillions.

    Notes
    -----
    With gap = actual - quantile, the loss is q * gap when the actual is above the quantile
    (gap > 0) and (q - 1) * gap when below; max() of the two picks the right branch, since the
    other one is negative (Koenker and Bassett 1978; Gneiting and Raftery 2007).
    """
    gaps = np.asarray(actual)[:, None] - quantiles
    levels = np.array(QUANTILES)
    return float(np.mean(np.maximum(levels * gaps, (levels - 1) * gaps)))


def interval_scores(actual: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    """
    Calibration and sharpness scores of a set of quantile forecasts.

    Parameters
    ----------
    actual : np.ndarray
        Actual levels, shape (forecasts,), trillions.
    quantiles : np.ndarray
        Quantile forecasts, shape (forecasts, 5), trillions, columns in QUANTILES order.

    Returns
    -------
    dict of str to float
        "coverage_80" / "coverage_95": share of actuals inside the 10-90% / 2.5-97.5% interval
        (targets 0.80 / 0.95); "width_80" / "width_95": mean interval width, trillions;
        "pinball": mean pinball loss, trillions; "below_<q>" for each quantile (e.g.
        "below_2.5"): share of actuals below it (target q / 100).
    """
    # Relies on QUANTILES' order: 2.5, 10, 50, 90, 97.5%.
    low_95, low_80, _, high_80, high_95 = quantiles.T
    scores = {
        "coverage_80": float(np.mean((actual >= low_80) & (actual <= high_80))),
        "coverage_95": float(np.mean((actual >= low_95) & (actual <= high_95))),
        "width_80": float(np.mean(high_80 - low_80)),
        "width_95": float(np.mean(high_95 - low_95)),
        "pinball": pinball_loss(actual, quantiles),
    }
    for level, column in zip(QUANTILES, quantiles.T):
        # :g drops trailing zeros, so the names are below_2.5, below_10, ..., below_97.5.
        scores[f"below_{100 * level:g}"] = float(np.mean(actual < column))
    return scores
