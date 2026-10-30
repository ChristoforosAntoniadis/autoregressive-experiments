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

from autoregressive_experiments.diagnostics import (
    DIFFERENCING_ORDER,
    SEASONAL_PERIOD,
    fit_sarima,
)
from autoregressive_experiments.features import lagged_exog
from autoregressive_experiments.ml_forecasting import LAG_MODELS, ML_MODELS, ml_forecast
from autoregressive_experiments.nn_forecasting import NN_MODELS
from autoregressive_experiments.prophet_forecasting import PROPHET, prophet_forecast
from autoregressive_experiments.var_models import VECM_RANK

NAIVE = "naive"
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
        Orders "p", "q", "P", "Q" and "trend".
    levels : np.ndarray
        The series' levels up to the origin, shape (months,), trillions.
    steps : int
        Months to forecast.

    Returns
    -------
    np.ndarray
        Level forecasts, shape (steps,), trillions.
    """
    fitted = fit_sarima(
        pd.Series(np.diff(levels)),
        settings["p"],
        settings["q"],
        settings["P"],
        settings["Q"],
        trend=settings["trend"],
    )
    return np.asarray(levels[-1] + np.cumsum(np.asarray(fitted.forecast(steps))))


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
        {"lags": k}.
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
    """
    lags, other = settings["lags"], 1 - column
    own_changes, other_changes = np.diff(history[:, column]), np.diff(history[:, other])
    planned = np.diff(
        np.concatenate(
            [[history[-1, other]], sarima_path(sarima[other], history[:, other], steps)]
        )
    )
    own = sarima[column]
    fitted = fit_sarima(
        pd.Series(own_changes[lags:]),
        own["p"],
        own["q"],
        own["P"],
        own["Q"],
        exog=lagged_exog(other_changes, lags),
        trend=own["trend"],
    )
    future = lagged_exog(np.concatenate([other_changes, planned]), lags)[-steps:]
    return np.asarray(
        history[-1, column] + np.cumsum(np.asarray(fitted.forecast(steps, exog=future)))
    )


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
    """
    if model in (VAR_MODEL, VECM_MODEL):
        return system_path(model, settings, history, steps)
    return np.asarray(ml_forecast(model, [settings, settings], history, steps))


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
        NAIVE, or any model of FAMILIES.
    settings : Mapping of str to Any
        Its settings (ignored by naive).
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
    Naive repeats the last level. The SARIMA + naive average weighs the series' tuned SARIMA by
    settings["weight"] and the naive forecast by the rest.
    """
    levels = history[:, column]
    if model == NAIVE:
        return np.full(steps, levels[-1])
    if model == SARIMA:
        return sarima_path(settings, levels, steps)
    if model == SARIMAX:
        return sarimax_path(settings, sarima, history, column, steps)
    if model == COMBINATION:
        weight = settings["weight"]
        return np.asarray(
            weight * sarima_path(sarima[column], levels, steps) + (1 - weight) * levels[-1]
        )
    if model in (SMA, EXPONENTIAL):
        return smoothing_path(model, settings, levels, steps)
    if model == PROPHET:
        return prophet_forecast(dates[: len(levels)], levels, settings, steps)
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
