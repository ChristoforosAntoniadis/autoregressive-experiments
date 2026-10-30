"""
The ready-made family: Prophet (Taylor and Letham 2018), a decomposable model of trend, yearly
seasonality and noise, fitted by Stan.

Each series is modelled on its own, on its levels with their month-end dates, and refitted at
every forecast origin on the months up to it. The trend is piecewise linear (or flat) with
automatically placed changepoints; weekly and daily seasonality are switched off, since the data
is monthly. Point forecasts are Prophet's yhat. For the intervals, Prophet's own uncertainty
samples (trend changes and noise, seeded with RANDOM_SEED) give the forecast's standard deviation
per horizon, which the evaluation turns into normal quantiles like the other models'.

Every setting is chosen per series by the time-aware search of ``tuning`` (Optuna) from the
search space in the ``ready_made`` section of ``configs/hyperparameters.json``.

Input: levels and dates handed over by the forecast-evaluation harness and the tuning.

Output: forecasts; nothing is written to disk.
"""

from collections.abc import Callable, Mapping
import logging
from typing import Any

import numpy as np
import pandas as pd

from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.configs.configurables import RANDOM_SEED

PROPHET = "Prophet"
READY_MADE_MODELS = (PROPHET,)
# Settings; each value's reasoning is in configs/hyperparameters.json.
READY_MADE_SPACES: dict[str, Mapping[str, Any]] = {PROPHET: HYPERPARAMETERS.ready_made.prophet}
# Month-end dates, as in the data; the future months Prophet forecasts follow them.
FREQUENCY = "ME"


def prophet_model(settings: Mapping[str, Any], samples: int) -> Any:
    """
    An unfitted Prophet model for monthly data.

    Parameters
    ----------
    settings : Mapping of str to Any
        The tuned settings: growth, changepoint_prior_scale, changepoint_range,
        seasonality_mode, seasonality_prior_scale and yearly_seasonality.
    samples : int
        Uncertainty samples to draw when predicting (0 for point forecasts only).

    Returns
    -------
    prophet.Prophet
        The model, with weekly and daily seasonality off and cmdstanpy's logging quiet.
    """
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # Prophet reports at import that plotly, for its interactive plots, is missing; no plots are
    # drawn with it here.
    logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)
    from prophet import Prophet

    return Prophet(
        **settings,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=samples,
    )


def prophet_forecast(
    dates: pd.Series | np.ndarray,
    levels: np.ndarray,
    settings: Mapping[str, Any],
    steps: int,
    samples: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
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
    samples : int, default=0
        Uncertainty samples for the forecast's standard deviation; 0 skips it.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray or None)
        The level forecasts (yhat), shape (steps,), trillions; and their standard deviation
        across the samples, shape (steps,), or None when samples is 0.

    Notes
    -----
    The samples come from NumPy's global generator, so it is seeded with RANDOM_SEED first;
    Prophet's fit itself is deterministic (Stan's optimizer from a fixed start).
    """
    history = pd.DataFrame({"ds": pd.to_datetime(np.asarray(dates)), "y": levels})
    model = prophet_model(settings, 0).fit(history)
    future = model.make_future_dataframe(periods=steps, freq=FREQUENCY, include_history=False)
    mean = model.predict(future)["yhat"].to_numpy()
    if not samples:
        return mean, None
    model.uncertainty_samples = samples
    np.random.seed(RANDOM_SEED)
    draws = model.predictive_samples(future)["yhat"]
    return mean, draws.std(axis=1, ddof=1)


def prophet_forecaster(
    settings: Mapping[str, Any], all_dates: np.ndarray, samples: int
) -> Callable[[np.ndarray, int], dict[str, np.ndarray]]:
    """
    A forecaster for the harness: Prophet refitted at every origin.

    Parameters
    ----------
    settings : Mapping of str to Any
        The series' tuned settings.
    all_dates : np.ndarray
        Month-end dates of the whole sample; only the months the harness hands over are used.
    samples : int
        Uncertainty samples per forecast (for the intervals).

    Returns
    -------
    callable
        (levels, steps) -> {"forecast": level path, "sd_constant": its standard deviation},
        both shape (steps,), trillions.
    """

    def forecast(levels: np.ndarray, steps: int) -> dict[str, np.ndarray]:
        """
        Fit on the levels so far and forecast `steps` months.

        Parameters
        ----------
        levels : np.ndarray
            Levels up to and including the origin, shape (months,), trillions; their length
            picks the matching dates.
        steps : int
            Months to forecast.

        Returns
        -------
        dict of str to np.ndarray
            "forecast" and "sd_constant", shape (steps,), trillions.
        """
        mean, spread = prophet_forecast(all_dates[: len(levels)], levels, settings, steps, samples)
        assert spread is not None
        return {"forecast": mean, "sd_constant": spread}

    return forecast
