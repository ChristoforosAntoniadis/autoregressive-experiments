"""
Point forecasts of every model from a history of levels: the one place where each model is fitted
and forecasts, shared by the tuning on the validation split (tuning) and the champions' test
(forecast_evaluation).

Every function fits on the levels it is given only, up to and including the forecast origin, and
forecasts the next `steps` months' levels. The models fall into the three families the
champions are chosen from:

- econometric: SARIMA (on the first differences, so d = 1), SARIMAX (the SARIMA plus the other
  series' lagged changes), the SARIMA + naive weighted average, VAR (on the differences) and VECM
  (on the levels), a simple moving average, exponential smoothing and Prophet
- machine learning: the lag-feature models of ml_forecasting
- neural network: the networks of nn_forecasting

VAR, VECM and the lag-feature models forecast both series together (each series' forecasts feed
the other's), so they are tuned as one system with one setting; the others model one series.

Every model except the benchmarks and the SARIMA + naive average has two tuned settings that
every family shares: "transform" (none, or the model works on the logarithm of the levels and
its forecasts are converted back, so changes count relative to the level) and "outliers" (see
outliers: the history is fitted as it is, with flagged changes clipped, or, for SARIMA and
SARIMAX, with a pulse regressor per flagged change). The transform is applied first, so the
outliers are judged on the transformed changes. The average inherits its SARIMA's settings.

The benchmarks (BENCHMARKS, chosen in the config) are naive (the last level repeated), drift
(the last level plus the mean change of the history per step, the random walk with drift), the
seasonal naive (the last season's changes repeated) and the moving average (the last level plus
the mean of the last MOVING_AVERAGE_WINDOW changes per step); none is tuned. One of them, the
reference (REFERENCE), is what the validation score and MASE compare against.

Input: levels and dates handed over by tuning and forecast_evaluation.

Output: forecasts; nothing is written to disk.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.vecm import VECM

from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.diagnostics import (
    DIFFERENCING_ORDER,
    SEASONAL_PERIOD,
    fit_sarima,
)
from autoregressive_experiments.features import lagged_exog
from autoregressive_experiments.ml_forecasting import LAG_MODELS, ML_MODELS, ml_forecast
from autoregressive_experiments.nn_forecasting import NN_MODELS
from autoregressive_experiments.outliers import CLIP, DUMMIES, NONE, clip_levels, pulse_dummies
from autoregressive_experiments.prophet_forecasting import PROPHET, prophet_forecast
from autoregressive_experiments.var_models import VECM_RANK

NAIVE, DRIFT, SEASONAL_NAIVE = "naive", "drift", "seasonal naive"
MOVING_AVERAGE = "moving average"
# Settings; each value's reasoning is in the dataset's config file (configs/<dataset>.json).
# The benchmarks run beside the champions, and the one the scores are relative to.
BENCHMARKS = tuple(HYPERPARAMETERS.forecast_evaluation.benchmarks)
REFERENCE = HYPERPARAMETERS.tuning.score_reference
# Changes the moving average benchmark averages.
MOVING_AVERAGE_WINDOW = HYPERPARAMETERS.forecast_evaluation.moving_average_window
# The MASE scale is the reference's in-sample one-step MAE: the change itself for naive, the
# change minus the same position's change a season earlier for the seasonal naive
# (metrics.mase_scale).
MASE_LAG = SEASONAL_PERIOD if REFERENCE == SEASONAL_NAIVE else 1
SARIMA, SARIMAX, COMBINATION = "SARIMA", "SARIMAX", "SARIMA + naive average"
VAR_MODEL, VECM_MODEL = "VAR", "VECM"
SMA, EXPONENTIAL = "SMA", "exponential smoothing"
ECONOMETRIC = (SARIMA, SARIMAX, COMBINATION, VAR_MODEL, VECM_MODEL, SMA, EXPONENTIAL, PROPHET)
# Family -> its models; one champion per family and series is chosen on validation.
FAMILIES: dict[str, tuple[str, ...]] = {
    "econometric": ECONOMETRIC,
    "machine learning": ML_MODELS,
    "neural network": NN_MODELS,
}
# Models that forecast both series together, tuned as one system with one setting.
JOINT_MODELS = (VAR_MODEL, VECM_MODEL, *LAG_MODELS)
# The "transform" setting's values: the levels as they are, or their logarithm.
LOG = "log"


def split_transform(settings: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    A model's transform and its other settings.

    Parameters
    ----------
    settings : Mapping of str to Any
        Tuned settings, possibly with a "transform" entry.

    Returns
    -------
    tuple of (str, dict of str to Any)
        NONE or LOG (NONE when the entry is missing), and the settings without it.
    """
    options = dict(settings)
    return options.pop("transform", NONE), options


def transformed(levels: np.ndarray, transform: str) -> np.ndarray:
    """
    The levels a model is fitted on: as they are, or their natural logarithm.

    Parameters
    ----------
    levels : np.ndarray
        Levels, any shape, trillions; positive when `transform` is LOG.
    transform : str
        NONE or LOG.

    Returns
    -------
    np.ndarray
        The same shape.
    """
    return np.log(levels) if transform == LOG else levels


def restored(path: np.ndarray, transform: str) -> np.ndarray:
    """
    Forecasts back in trillions: the inverse of transformed().

    Parameters
    ----------
    path : np.ndarray
        Forecasts in the fitted units, any shape.
    transform : str
        NONE or LOG.

    Returns
    -------
    np.ndarray
        The same shape, trillions.
    """
    return np.exp(path) if transform == LOG else path


def split_outliers(settings: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    A model's outlier handling and its other settings.

    Parameters
    ----------
    settings : Mapping of str to Any
        Tuned settings, possibly with an "outliers" entry.

    Returns
    -------
    tuple of (str, dict of str to Any)
        NONE, CLIP or DUMMIES (NONE when the entry is missing), and the settings without it.
    """
    options = dict(settings)
    return options.pop("outliers", NONE), options


def sarima_name(settings: Mapping[str, Any]) -> str:
    """
    Report name of a SARIMA from its settings, e.g. "SARIMA(2,1,2)x(1,0,0,12)".

    Parameters
    ----------
    settings : Mapping of str to Any
        The orders "p", "q", "P", "Q" and the "trend" ("c" drift, "n" none).

    Returns
    -------
    str
        The orders in the usual notation (d fixed at 1, D at 0), with " no drift" appended when
        the model has no constant.
    """
    orders = f"({settings['p']},{DIFFERENCING_ORDER},{settings['q']})"
    seasonal = f"({settings['P']},0,{settings['Q']},{SEASONAL_PERIOD})"
    return f"SARIMA{orders}x{seasonal}" + ("" if settings["trend"] == "c" else " no drift")


def sarima_path(settings: Mapping[str, Any], levels: np.ndarray, steps: int) -> np.ndarray:
    """
    SARIMA on one series' first differences, its forecasts added back onto the last level.

    Parameters
    ----------
    settings : Mapping of str to Any
        Orders "p", "q", "P", "Q", "trend" and, optionally, "transform" and "outliers".
    levels : np.ndarray
        The series' levels up to the origin, shape (months,), trillions.
    steps : int
        Months to forecast.

    Returns
    -------
    np.ndarray
        Level forecasts, shape (steps,), trillions.

    Notes
    -----
    With "dummies", each flagged change gets a pulse regressor whose future values are zero, so
    the shock is absorbed by its own coefficient instead of the model's dynamics.
    """
    transform, _ = split_transform(settings)
    levels = transformed(levels, transform)
    mode, _ = split_outliers(settings)
    if mode == CLIP:
        levels = clip_levels(levels)
    changes = np.diff(levels)
    pulses = pulse_dummies(changes) if mode == DUMMIES else None
    fitted = fit_sarima(
        pd.Series(changes),
        settings["p"],
        settings["q"],
        settings["P"],
        settings["Q"],
        exog=pulses,
        trend=settings["trend"],
    )
    future = None if pulses is None else np.zeros((steps, pulses.shape[1]))
    path = levels[-1] + np.cumsum(np.asarray(fitted.forecast(steps, exog=future)))
    return np.asarray(restored(path, transform))


def sarimax_path(
    settings: Mapping[str, Any],
    sarima: Sequence[Mapping[str, Any]],
    history: np.ndarray,
    column: int,
    steps: int,
) -> np.ndarray:
    """
    The series' tuned SARIMA plus the other series' last `lags` monthly changes as regressors.

    Parameters
    ----------
    settings : Mapping of str to Any
        {"lags": k} and, optionally, "transform" and "outliers".
    sarima : Sequence of Mapping of str to Any
        Both series' tuned SARIMA settings, in column order.
    history : np.ndarray
        Both series' levels up to the origin, shape (months, 2), trillions.
    column : int
        The series forecast; the other column supplies the regressors.
    steps : int
        Months to forecast.

    Returns
    -------
    np.ndarray
        Level forecasts, shape (steps,), trillions.

    Notes
    -----
    Processing:
    1. Forecast the other series' changes with its own tuned SARIMA.
    2. Fit the SARIMAX on this series' changes from month `lags` on, with the lag matrix of the
       other series' changes as regressors.
    3. The forecast months' regressors are the other series' actual changes followed by its
       forecast ones; from h = 2 on they include forecasts, never actual future values. No
       same-month regressor, which would be leakage.

    With "clip", both series' histories are clipped before the fit; with "dummies", the own
    series' flagged changes get pulse regressors next to the lags, zero in the future. With
    "log", both series enter as logarithms, the other series' planned path included (its own
    SARIMA forecasts it in trillions, whatever that SARIMA's transform).
    """
    lags, other = settings["lags"], 1 - column
    transform, _ = split_transform(settings)
    mode, _ = split_outliers(settings)
    planned_levels = sarima_path(sarima[other], history[:, other], steps)
    history = transformed(history, transform)
    planned = np.diff(
        np.concatenate([[history[-1, other]], transformed(planned_levels, transform)])
    )
    fit_history = clip_levels(history) if mode == CLIP else history
    own_changes, other_changes = np.diff(fit_history[:, column]), np.diff(fit_history[:, other])
    exog = lagged_exog(other_changes, lags)
    future = lagged_exog(np.concatenate([other_changes, planned]), lags)[-steps:]
    pulses = pulse_dummies(own_changes, lags) if mode == DUMMIES else None
    if pulses is not None:
        exog = np.column_stack([exog, pulses])
        future = np.column_stack([future, np.zeros((steps, pulses.shape[1]))])
    own = sarima[column]
    fitted = fit_sarima(
        pd.Series(own_changes[lags:]),
        own["p"],
        own["q"],
        own["P"],
        own["Q"],
        exog=exog,
        trend=own["trend"],
    )
    path = history[-1, column] + np.cumsum(np.asarray(fitted.forecast(steps, exog=future)))
    return np.asarray(restored(path, transform))


def smoothing_path(
    model: str, settings: Mapping[str, Any], levels: np.ndarray, steps: int
) -> np.ndarray:
    """
    A simple moving average, or exponential smoothing in its tuned form, on one series' levels.

    Parameters
    ----------
    model : str
        SMA or EXPONENTIAL.
    settings : Mapping of str to Any
        {"window": k} for SMA; ExponentialSmoothing's keyword arguments otherwise.
    levels : np.ndarray
        The series' levels up to the origin, shape (months,), trillions.
    steps : int
        Months to forecast.

    Returns
    -------
    np.ndarray
        Level forecasts, shape (steps,), trillions.

    Notes
    -----
    The moving average forecasts the mean of the last `window` levels at every horizon. A
    seasonal exponential-smoothing model gets the project's season length, so it isn't
    configured twice; its smoothing weights are re-estimated at every fit.
    """
    if model == SMA:
        return np.full(steps, float(np.mean(levels[-settings["window"] :])))
    options = dict(settings)
    if options.get("seasonal"):
        options["seasonal_periods"] = SEASONAL_PERIOD
    return np.asarray(ExponentialSmoothing(levels, **options).fit().forecast(steps))


def system_path(
    model: str, settings: Mapping[str, Any], history: np.ndarray, steps: int
) -> np.ndarray:
    """
    A VAR on both series' differences or a VECM on their levels.

    Parameters
    ----------
    model : str
        VAR_MODEL or VECM_MODEL.
    settings : Mapping of str to Any
        "lags" and, for the VAR, "trend" ("c" or "ct"); for the VECM, "deterministic".
    history : np.ndarray
        Both series' levels up to the origin, shape (months, 2), trillions.
    steps : int
        Months to forecast.

    Returns
    -------
    np.ndarray
        Both series' level forecasts, shape (steps, 2), trillions.

    Notes
    -----
    The VAR is fitted by OLS and its forecast differences are added onto the last levels; the
    VECM has cointegration rank VECM_RANK and forecasts the levels directly.
    """
    lags = settings["lags"]
    if model == VAR_MODEL:
        differences = np.diff(history, axis=0)
        fitted = VAR(differences).fit(lags, trend=settings["trend"])
        return np.asarray(
            history[-1] + np.cumsum(fitted.forecast(differences[-lags:], steps), axis=0)
        )
    fitted = VECM(
        history, k_ar_diff=lags, coint_rank=VECM_RANK, deterministic=settings["deterministic"]
    ).fit()
    return np.asarray(fitted.predict(steps))


def joint_paths(
    model: str, settings: Mapping[str, Any], history: np.ndarray, steps: int
) -> np.ndarray:
    """
    Both series' forecasts from a model that forecasts them together.

    Parameters
    ----------
    model : str
        One of JOINT_MODELS.
    settings : Mapping of str to Any
        Its settings, the same for both series.
    history : np.ndarray
        Both series' levels up to the origin, shape (months, 2), trillions.
    steps : int
        Months to forecast.

    Returns
    -------
    np.ndarray
        Level forecasts, shape (steps, 2), trillions.

    Notes
    -----
    With "log" both series enter as logarithms; with "clip" the model is fitted on the clipped
    history, whose last levels are the actual ones, so the forecasts start from the real latest
    values.
    """
    transform, options = split_transform(settings)
    mode, options = split_outliers(options)
    history = transformed(history, transform)
    if mode == CLIP:
        history = clip_levels(history)
    if model in (VAR_MODEL, VECM_MODEL):
        path = system_path(model, options, history, steps)
    else:
        path = np.asarray(ml_forecast(model, [options, options], history, steps))
    return np.asarray(restored(path, transform))


def series_path(
    model: str,
    settings: Mapping[str, Any],
    history: np.ndarray,
    dates: np.ndarray,
    column: int,
    steps: int,
    sarima: Sequence[Mapping[str, Any]] = (),
) -> np.ndarray:
    """
    One series' forecasts from any model, naive included.

    Parameters
    ----------
    model : str
        One of BENCHMARKS, or any model of FAMILIES.
    settings : Mapping of str to Any
        Its settings (ignored by the benchmarks).
    history : np.ndarray
        Both series' levels up to the origin, shape (months, 2), trillions.
    dates : np.ndarray
        The months of `history` (month ends), for Prophet.
    column : int
        The series forecast.
    steps : int
        Months to forecast.
    sarima : Sequence of Mapping of str to Any, default=()
        Both series' tuned SARIMA settings; needed by SARIMAX and the SARIMA + naive average.

    Returns
    -------
    np.ndarray
        Level forecasts, shape (steps,), trillions.

    Raises
    ------
    ValueError
        If `model` is unknown.

    Notes
    -----
    Naive repeats the last level; drift adds the history's mean change for every step ahead
    (the random walk with drift, the usual second benchmark for a trending series); the seasonal
    naive adds the changes of the last SEASONAL_PERIOD periods again, in order, so each forecast
    period moves as the same position of the last season did (Hyndman and Athanasopoulos 2021,
    section 5.2, on the levels' changes); the moving average is the drift of the last
    MOVING_AVERAGE_WINDOW changes only, so the level keeps moving at the recent mean rate. The
    SARIMA + naive average weighs the series' tuned SARIMA by settings["weight"] and the naive
    forecast by the rest.
    """
    levels = history[:, column]
    if model == NAIVE:
        return np.full(steps, levels[-1])
    if model in (DRIFT, MOVING_AVERAGE):
        window = len(levels) if model == DRIFT else MOVING_AVERAGE_WINDOW
        mean_change = np.mean(np.diff(levels)[-window:])
        return np.asarray(levels[-1] + np.arange(1, steps + 1) * mean_change)
    if model == SEASONAL_NAIVE:
        season = np.diff(levels)[-SEASONAL_PERIOD:]
        return np.asarray(levels[-1] + np.cumsum(np.resize(season, steps)))
    if model == SARIMA:
        return sarima_path(settings, levels, steps)
    if model == SARIMAX:
        return sarimax_path(settings, sarima, history, column, steps)
    if model == COMBINATION:
        weight = settings["weight"]
        return np.asarray(
            weight * sarima_path(sarima[column], levels, steps) + (1 - weight) * levels[-1]
        )
    if model in (SMA, EXPONENTIAL, PROPHET):
        transform, options = split_transform(settings)
        mode, options = split_outliers(options)
        levels = transformed(levels, transform)
        if mode == CLIP:
            levels = clip_levels(levels)
        if model == PROPHET:
            path = prophet_forecast(dates[: len(levels)], levels, options, steps)
        else:
            path = smoothing_path(model, options, levels, steps)
        return np.asarray(restored(path, transform))
    if model in JOINT_MODELS:
        return joint_paths(model, settings, history, steps)[:, column]
    raise ValueError(f"unknown model {model!r}")


def family_of(model: str) -> str:
    """
    The family a model belongs to.

    Parameters
    ----------
    model : str
        Any model of FAMILIES.

    Returns
    -------
    str
        "econometric", "machine learning" or "neural network".

    Raises
    ------
    KeyError
        If the model is in no family.
    """
    for family, models in FAMILIES.items():
        if model in models:
            return family
    raise KeyError(model)
