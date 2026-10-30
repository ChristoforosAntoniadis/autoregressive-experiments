"""
A ready-made model, Prophet (Taylor and Letham 2018), a decomposable model of trend, yearly
seasonality and noise, fitted by Stan.

Each series is modelled on its own, on its levels with their month-end dates, and refitted at
every forecast origin on the months up to it. The trend is piecewise linear (or flat) with
automatically placed changepoints; weekly and daily seasonality are switched off, since the data
is monthly. Point forecasts are Prophet's yhat.

Prophet belongs to the econometric family. Every setting is chosen per series by ``tuning`` on
the validation split (Optuna) from the search space in the ``ready_made`` section of
``configs/hyperparameters.json``. Its bands, like every model's, come from its own past errors
(forecast_evaluation).

Input: levels and dates handed over by ``model_forecasts``.

Output: forecasts; nothing is written to disk.
"""

from collections.abc import Mapping
import logging
from typing import Any

import numpy as np
import pandas as pd

from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS

PROPHET = "Prophet"
READY_MADE_MODELS = (PROPHET,)
# Settings; each value's reasoning is in configs/hyperparameters.json.
READY_MADE_SPACES: dict[str, Mapping[str, Any]] = {PROPHET: HYPERPARAMETERS.ready_made.prophet}
# Month-end dates, as in the data; the future months Prophet forecasts follow them.
FREQUENCY = "ME"


def prophet_model(settings: Mapping[str, Any]) -> Any:
    """
    An unfitted Prophet model for monthly data.

    Parameters
    ----------
    settings : Mapping of str to Any
        The tuned settings: growth, changepoint_prior_scale, changepoint_range,
        seasonality_mode, seasonality_prior_scale and yearly_seasonality.

    Returns
    -------
    prophet.Prophet
        The model, with weekly and daily seasonality off and cmdstanpy's logging quiet.
    """
    # Prophet reports at import that plotly, for its interactive plots, is missing; no plots are
    # drawn with it here.
    logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)
    from prophet import Prophet

    # After the import, which sets cmdstanpy's logger to INFO (two lines per fit otherwise).
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return Prophet(
        **settings,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    )


def prophet_forecast(
    dates: pd.Series | np.ndarray,
    levels: np.ndarray,
    settings: Mapping[str, Any],
    steps: int,
) -> np.ndarray:
    """
    Fit Prophet on one series' history and forecast its levels.

    Parameters
    ----------
    dates : pd.Series or np.ndarray
        The history's month-end dates, oldest first.
    levels : np.ndarray
        Its levels, shape (months,), trillions.
    settings : Mapping of str to Any
        See prophet_model.
    steps : int
        Months to forecast.

    Returns
    -------
    np.ndarray
        The level forecasts (yhat), shape (steps,), trillions.

    Notes
    -----
    Prophet's fit is deterministic (Stan's optimizer from a fixed start). Its own sampled
    intervals are off: every model's bands come from its past errors (forecast_evaluation).
    """
    history = pd.DataFrame({"ds": pd.to_datetime(np.asarray(dates)), "y": levels})
    model = prophet_model(settings).fit(history)
    future = model.make_future_dataframe(periods=steps, freq=FREQUENCY, include_history=False)
    return np.asarray(model.predict(future)["yhat"].to_numpy())
