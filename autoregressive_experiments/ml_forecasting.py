"""
The lag-feature ML family: Ridge, Lasso, random forest, XGBoost and k-nearest neighbours, each
predicting next month's change of one series from the last k monthly changes of both series.
The networks of ``nn_forecasting`` use the same lag-feature pipeline defined here.

Each model is refitted at every forecast origin on that origin's data only, one model per
series, and forecasts several months ahead recursively: each month's predicted changes of both
series become the inputs for the next month, as a VAR does (a nonlinear VAR, in effect). The
forecast changes are added onto the last level, as for SARIMA. The models work on changes, not
levels, because both series are I(1) and because trees and neighbours can't predict
outside the range they were trained on, which the test levels leave.

Every setting, the lag count k included, is chosen by ``tuning`` on the validation split
(Optuna), from the search spaces in the ``ml`` section of ``configs/hyperparameters.json``, and
then kept fixed; one setting serves both series, which these models forecast together.
Features and target are standardized inside every fit, from the rows that fit sees only, so the
penalties and tree thresholds in the search spaces are in standard-deviation units.

Input: levels handed over by ``model_forecasts`` (for tuning and the test).

Output: model builders and forecasts; nothing is written to disk.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.configs.configurables import RANDOM_SEED
from autoregressive_experiments.features import lagged_exog
from autoregressive_experiments.nn_forecasting import NN_MODELS, NN_SPACES, build_network

# Settings; each value's reasoning is in configs/hyperparameters.json.
SETTINGS = HYPERPARAMETERS.ml
RIDGE, LASSO, FOREST, XGBOOST, KNN = "Ridge", "Lasso", "random forest", "XGBoost", "KNN"
ML_MODELS = (RIDGE, LASSO, FOREST, XGBOOST, KNN)
# Every model on lag features: the ML family, then the neural networks of nn_forecasting.
LAG_MODELS = (*ML_MODELS, *NN_MODELS)
# Model -> its search space: setting name -> the values tried.
ML_SPACES: dict[str, Mapping[str, Any]] = {
    RIDGE: SETTINGS.ridge,
    LASSO: SETTINGS.lasso,
    FOREST: SETTINGS.random_forest,
    XGBOOST: SETTINGS.xgboost,
    KNN: SETTINGS.knn,
}


def max_lag(model: str) -> int:
    """
    The longest lag a model's search space tries.

    Parameters
    ----------
    model : str
        One of LAG_MODELS.

    Returns
    -------
    int
        The largest value of its "lags" setting; the tuning builds that model's features once
        with this many lags.

    Notes
    -----
    Per model, not shared: the sequence networks read windows of up to two years, and one
    shared maximum would cost every other model its first training rows.
    """
    return int(max({**ML_SPACES, **NN_SPACES}[model]["lags"]))


def lag_features(changes: np.ndarray, lags: int) -> np.ndarray:
    """
    The feature matrix: each row holds the last `lags` changes of every series before its month.

    Parameters
    ----------
    changes : np.ndarray
        Monthly changes of the series, shape (n, series), trillions, oldest first.
    lags : int
        How many past changes of each series a row holds.

    Returns
    -------
    np.ndarray
        Shape (n - lags, series x lags): row i is the first series' changes before month
        i + lags (most recent first), then the second series' the same way.

    Notes
    -----
    Built from features.lagged_exog per series, so a row never holds its own month's change:
    the target of row i is changes[i + lags].
    """
    return np.hstack([lagged_exog(changes[:, column], lags) for column in range(changes.shape[1])])


def lag_columns(lags: int, max_lag: int, series: int) -> list[int]:
    """
    The columns of a lag_features(changes, max_lag) matrix that hold each series' first `lags`
    lags, so one matrix serves every lag count.

    Parameters
    ----------
    lags : int
        Lags wanted per series, at most `max_lag`.
    max_lag : int
        The lag count the matrix was built with.
    series : int
        Number of series (column blocks of `max_lag` columns each).

    Returns
    -------
    list of int
        Column positions, series after series, most recent lag first: the same layout as
        lag_features(changes, lags) on the same rows.
    """
    return [block * max_lag + lag for block in range(series) for lag in range(lags)]


def build_model(model: str, settings: Mapping[str, Any]) -> TransformedTargetRegressor:
    """
    An unfitted lag-feature model with the given settings, standardizing its features and its
    target.

    Parameters
    ----------
    model : str
        One of LAG_MODELS: an ML model, or a network built by nn_forecasting.build_network.
    settings : Mapping of str to Any
        The model's keyword arguments from its search space, plus "lags" (ignored here).

    Returns
    -------
    TransformedTargetRegressor
        StandardScaler then the model, wrapped so the target is standardized for the fit and
        the predictions are returned in trillions.

    Raises
    ------
    ValueError
        If `model` isn't one of LAG_MODELS.

    Notes
    -----
    The scalers learn their means and SDs from the rows the model is fitted on, never from
    later months. Standardizing the target makes the penalties (Ridge, Lasso), the pruning
    thresholds (forest, XGBoost) and the regularization terms scale-free, so one search space
    serves both series. Lasso with a zero penalty is fitted as a plain linear regression:
    coordinate descent without a penalty converges poorly, and the solution is the same.
    """
    options = {key: value for key, value in settings.items() if key != "lags"}
    base: Any
    if model == RIDGE:
        base = Ridge(**options)
    elif model == LASSO:
        base = LinearRegression() if options["alpha"] == 0 else Lasso(**options)
    elif model == FOREST:
        # One thread: with several, the trees' predictions are summed in the order the threads
        # finish, which changes the last bit of a forecast from run to run.
        base = RandomForestRegressor(**options, random_state=RANDOM_SEED, n_jobs=1)
    elif model == XGBOOST:
        base = XGBRegressor(**options, random_state=RANDOM_SEED, verbosity=0)
    elif model == KNN:
        base = KNeighborsRegressor(**options)
    elif model in NN_MODELS:
        base = build_network(model, options)
    else:
        raise ValueError(f"unknown lag-feature model {model!r}")
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), base), transformer=StandardScaler()
    )


def fit_model(
    model: str, settings: Mapping[str, Any], changes: np.ndarray, column: int
) -> TransformedTargetRegressor:
    """
    Fit one ML model for one series on all the changes given.

    Parameters
    ----------
    model : str
        One of LAG_MODELS.
    settings : Mapping of str to Any
        Its tuned settings, including "lags".
    changes : np.ndarray
        Monthly changes of every series, shape (n, series), trillions, oldest first.
    column : int
        The series whose next change the model predicts.

    Returns
    -------
    TransformedTargetRegressor
        The fitted model; predict() takes lag_features rows with settings["lags"] lags.
    """
    lags = settings["lags"]
    return build_model(model, settings).fit(lag_features(changes, lags), changes[lags:, column])


def recursive_forecast(
    models: list[tuple[Any, int]], changes: np.ndarray, steps: int
) -> np.ndarray:
    """
    Forecast every series' changes `steps` months ahead, feeding each month's forecasts back.

    Parameters
    ----------
    models : list of tuple of (Any, int)
        One (fitted model, its lag count) per series, in the column order of `changes`; each
        predicts its series' next change from a lag_features row with its own lag count.
    changes : np.ndarray
        Monthly changes up to the origin, shape (n, series), trillions, oldest first.
    steps : int
        Months to forecast.

    Returns
    -------
    np.ndarray
        Forecast changes, shape (steps, series), trillions.

    Notes
    -----
    Processing, for each month ahead:
    1. For each model, take the last `lags` changes of every series, actual or already
       forecast, and lay them out as one lag_features row: each series' changes, most recent
       first.
    2. Predict every series' next change and append them to the history.

    The row layout matches lag_features exactly, so each model sees inputs arranged as in
    training; from the second month on they include forecast changes, never actual future ones.
    The series' models may use different lag counts, since each was tuned on its own series.
    """
    history = list(changes)
    path = []
    for _ in range(steps):
        step = []
        for fitted, lags in models:
            recent = np.array(history[-lags:])
            # Each series' last `lags` changes, most recent first, series after series.
            row = recent[::-1].T.reshape(1, -1)
            step.append(float(fitted.predict(row)[0]))
        history.append(np.array(step))
        path.append(step)
    return np.array(path)


def ml_forecast(
    model: str, settings: Sequence[Mapping[str, Any]], history: np.ndarray, steps: int
) -> np.ndarray:
    """
    Fit one model per series on the history and forecast both series' levels.

    Parameters
    ----------
    model : str
        One of LAG_MODELS.
    settings : Sequence of Mapping of str to Any
        Each series' tuned settings for this model, in column order.
    history : np.ndarray
        Levels of every series up to and including the origin, shape (months, series),
        trillions, oldest first.
    steps : int
        Months to forecast.

    Returns
    -------
    np.ndarray
        Forecast levels, shape (steps, series), trillions.

    Notes
    -----
    Processing:
    1. Difference the levels.
    2. Fit one model per series with that series' tuned settings.
    3. Forecast the changes recursively and add their running total to the last levels.
    """
    changes = np.diff(history, axis=0)
    models = [
        (fit_model(model, options, changes, column), options["lags"])
        for column, options in enumerate(settings)
    ]
    return history[-1] + np.cumsum(recursive_forecast(models, changes, steps), axis=0)
