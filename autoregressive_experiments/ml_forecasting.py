"""
The lag-feature ML family: Ridge, Lasso, random forest and histogram gradient boosting, each
predicting next month's change of one series from the last k monthly changes of both series.

Each model is refitted at every forecast origin on that origin's data only, one model per
series, and forecasts several months ahead recursively: each month's predicted changes of both
series become the inputs for the next month, as a VAR does (a nonlinear VAR, in effect). The
forecast changes are added onto the last level, as for SARIMA. The models work on changes, not
levels, because both series are I(1) (step U1) and because trees can't predict outside the range
they were trained on, which the test levels leave.

The lag counts are the VAR candidates (1, 3 and 5), so no new number is introduced; every other
setting, with its justification, is in the ``ml`` section of ``configs/hyperparameters.json``.

Input: levels handed over by the forecast-evaluation harness (``forecast_evaluation``), which
also selects, scores and reports these models.

Output: forecasters for that harness; nothing is written to disk.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.features import lagged_exog
from autoregressive_experiments.var_models import VAR_LAGS

# Settings; each value's justification is in configs/hyperparameters.json.
SETTINGS = HYPERPARAMETERS.ml
# The lag counts tried: the VAR candidates, so no new number is introduced.
ML_LAGS = VAR_LAGS
RIDGE, LASSO, FOREST, BOOSTING = "Ridge", "Lasso", "random forest", "gradient boosting"
ML_MODELS = (RIDGE, LASSO, FOREST, BOOSTING)
# (model, lags) -> the name the harness reports, e.g. "Ridge (3 lags)".
ML_NAMES = {
    (model, lags): f"{model} ({lags} lag{'' if lags == 1 else 's'})"
    for model in ML_MODELS
    for lags in ML_LAGS
}


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


def penalty_grid(model: str, features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    The penalties cross-validation chooses from: a log-spaced grid plus zero.

    Parameters
    ----------
    model : str
        RIDGE or LASSO.
    features : np.ndarray
        The window's training features, shape (rows, k).
    target : np.ndarray
        The window's training targets, shape (rows,).

    Returns
    -------
    np.ndarray
        penalty_points positive penalties, then 0 (plain least squares).

    Notes
    -----
    Lasso: the largest penalty is the smallest one that zeroes every coefficient,
    max |x_j' (y - mean y)| / rows on the standardized features (the value scikit-learn's own
    path starts from), and the grid runs down to penalty_span of it. Ridge: penalty_span x rows
    to rows / penalty_span, which with standardized features runs from almost no shrinkage to
    near-zero coefficients. Zero is always added, so "no penalty" is a choice cross-validation
    can make rather than an edge of the grid.
    """
    decades = np.log10(SETTINGS.penalty_span)
    if model == RIDGE:
        positive = len(target) * np.logspace(decades, -decades, SETTINGS.penalty_points)
    else:
        scaled = StandardScaler().fit_transform(features)
        largest = np.max(np.abs(scaled.T @ (target - target.mean()))) / len(target)
        positive = largest * np.logspace(0, decades, SETTINGS.penalty_points)
    return np.append(positive, 0.0)


def penalized_model(model: str, features: np.ndarray, target: np.ndarray) -> Pipeline:
    """
    Ridge or Lasso with standardized features and a penalty chosen by time-ordered
    cross-validation.

    Parameters
    ----------
    model : str
        RIDGE or LASSO.
    features : np.ndarray
        The window's training features, shape (rows, k); they set the penalty grid.
    target : np.ndarray
        The window's training targets, shape (rows,).

    Returns
    -------
    Pipeline
        StandardScaler then RidgeCV or LassoCV, unfitted.

    Notes
    -----
    The scaler learns the mean and SD from the training window it is fitted on, never from
    later months. TimeSeriesSplit keeps every validation fold after its training fold, so the
    penalty is chosen the way the whole evaluation works: on months after the fit. The grid is
    penalty_grid(): log-spaced penalties plus zero.
    """
    folds = TimeSeriesSplit(n_splits=SETTINGS.cv_folds)
    alphas = penalty_grid(model, features, target)
    if model == RIDGE:
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=folds))
    return make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=folds))


def boosting_model(features: np.ndarray, target: np.ndarray) -> HistGradientBoostingRegressor:
    """
    Histogram gradient boosting, its number of trees chosen on the end of the window.

    Parameters
    ----------
    features : np.ndarray
        Training features, shape (rows, k), oldest first.
    target : np.ndarray
        Training targets, shape (rows,), trillions.

    Returns
    -------
    HistGradientBoostingRegressor
        Fitted on all rows with the chosen number of trees (its n_iter_).

    Notes
    -----
    Processing:
    1. Fit on the first 1 - boosting_holdout_share of the rows with scikit-learn's early
       stopping, scored on the held-out last rows: training stops once boosting_patience trees
       in a row improve the holdout loss by less than boosting_tolerance.
    2. Take the number of trees with the best holdout loss within that run.
    3. Refit on all rows with exactly that many trees.

    The holdout is the end of the window, in time order, passed as X_val / y_val: scikit-learn's
    default draws a random holdout, which would let later months choose for earlier ones.
    """
    split = round(len(target) * (1 - SETTINGS.boosting_holdout_share))
    options: dict[str, Any] = {
        "learning_rate": SETTINGS.boosting_learning_rate,
        "random_state": SETTINGS.random_seed,
    }
    probe = HistGradientBoostingRegressor(
        max_iter=SETTINGS.boosting_max_iterations,
        early_stopping=True,
        n_iter_no_change=SETTINGS.boosting_patience,
        tol=SETTINGS.boosting_tolerance,
        **options,
    )
    probe.fit(features[:split], target[:split], X_val=features[split:], y_val=target[split:])
    # validation_score_[i] is the holdout score (higher is better) after i trees; [0] is before
    # any tree, so the index of the best score is the number of trees, at least 1.
    trees = max(int(np.argmax(probe.validation_score_)), 1)
    return HistGradientBoostingRegressor(max_iter=trees, early_stopping=False, **options).fit(
        features, target
    )


def fit_model(model: str, features: np.ndarray, target: np.ndarray) -> Any:
    """
    Fit one ML model on one series' training rows.

    Parameters
    ----------
    model : str
        One of ML_MODELS.
    features : np.ndarray
        Training features, shape (rows, k), oldest first.
    target : np.ndarray
        Training targets (the series' next-month changes), shape (rows,), trillions.

    Returns
    -------
    Any
        A fitted scikit-learn estimator (or pipeline) with predict().

    Raises
    ------
    ValueError
        If `model` isn't one of ML_MODELS.
    """
    if model in (RIDGE, LASSO):
        return penalized_model(model, features, target).fit(features, target)
    if model == FOREST:
        forest = RandomForestRegressor(
            n_estimators=SETTINGS.forest_trees, random_state=SETTINGS.random_seed, n_jobs=-1
        )
        return forest.fit(features, target)
    if model == BOOSTING:
        return boosting_model(features, target)
    raise ValueError(f"unknown ML model {model!r}")


def recursive_forecast(
    models: list[Any], changes: np.ndarray, lags: int, steps: int
) -> np.ndarray:
    """
    Forecast every series' changes `steps` months ahead, feeding each month's forecasts back.

    Parameters
    ----------
    models : list of Any
        One fitted model per series (column order of `changes`), each predicting its series'
        next change from lag_features.
    changes : np.ndarray
        Monthly changes up to the origin, shape (n, series), trillions, oldest first.
    lags : int
        The lag count the models were fitted with.
    steps : int
        Months to forecast.

    Returns
    -------
    np.ndarray
        Forecast changes, shape (steps, series), trillions.

    Notes
    -----
    Processing, for each month ahead:
    1. Take the last `lags` changes of every series, actual or already forecast.
    2. Lay them out as one lag_features row: each series' changes, most recent first.
    3. Predict every series' next change and append it to the history.

    The row layout matches lag_features exactly, so each model sees inputs arranged as in
    training; from the second month on they include forecast changes, never actual future ones.
    """
    history = [row for row in changes[-lags:]]
    path = []
    for _ in range(steps):
        recent = np.array(history[-lags:])
        # Each series' last `lags` changes, most recent first, series after series.
        row = recent[::-1].T.reshape(1, -1)
        step = np.array([float(model.predict(row)[0]) for model in models])
        history.append(step)
        path.append(step)
    return np.array(path)


def ml_forecast(model: str, lags: int, history: np.ndarray, steps: int) -> dict[str, np.ndarray]:
    """
    Fit one model per series on the history and forecast both series' levels.

    Parameters
    ----------
    model : str
        One of ML_MODELS.
    lags : int
        Lag count (one of ML_LAGS).
    history : np.ndarray
        Levels of every series up to and including the origin, shape (months, series),
        trillions, oldest first.
    steps : int
        Months to forecast.

    Returns
    -------
    dict of str to np.ndarray
        "levels": forecast levels, shape (steps, series), trillions.
        "trees": the number of boosting trees chosen per series, shape (series,); NaN for the
        other models.

    Notes
    -----
    Processing:
    1. Difference the levels and build the lag features; the targets are the changes from
       month `lags` on.
    2. Fit one model per series.
    3. Forecast the changes recursively and add their running total to the last levels.
    """
    changes = np.diff(history, axis=0)
    features = lag_features(changes, lags)
    models = [
        fit_model(model, features, changes[lags:, column]) for column in range(changes.shape[1])
    ]
    path = recursive_forecast(models, changes, lags, steps)
    trees = np.array([float(getattr(fitted, "n_iter_", np.nan)) for fitted in models])
    return {"levels": history[-1] + np.cumsum(path, axis=0), "trees": trees}


def ml_forecaster(
    model: str,
    lags: int,
    all_levels: np.ndarray,
    target: int,
    cache: dict[tuple[str, int, int, int], dict[str, np.ndarray]],
) -> Callable[[np.ndarray, int], dict[str, np.ndarray | float]]:
    """
    A forecaster for one series of one ML model; the fit at each origin is shared between the
    two series through the cache, as for the VAR systems.

    Parameters
    ----------
    model : str
        One of ML_MODELS.
    lags : int
        Lag count (one of ML_LAGS).
    all_levels : np.ndarray
        Levels of both series over the whole sample, shape (months, 2), trillions, columns in
        VALUE_COLUMNS order. Only the months the harness hands over are ever used.
    target : int
        Column of `all_levels` this forecaster scores.
    cache : dict of tuple of (str, int, int, int) to dict of str to np.ndarray
        Shared, mutated store of ml_forecast results, keyed by (model, lags, months of history,
        steps).

    Returns
    -------
    callable
        A forecaster for the harness: (levels, steps) -> {"forecast": level path, shape
        (steps,), trillions; "trees": the boosting trees chosen for this series, NaN for other
        models}.
    """

    def forecast(levels: np.ndarray, steps: int) -> dict[str, np.ndarray | float]:
        """
        Forecast this series from the models fitted on both series so far.

        Parameters
        ----------
        levels : np.ndarray
            This series' levels up to and including the origin, shape (months,), trillions;
            only its length is used to cut both series, after the leakage check.
        steps : int
            Months to forecast.

        Returns
        -------
        dict of str to np.ndarray or float
            "forecast": the level path, shape (steps,), trillions; "trees": see ml_forecaster.

        Raises
        ------
        AssertionError
            If the history cut from `all_levels` doesn't match the levels the harness passed.
        """
        history = all_levels[: len(levels)]
        # Leakage guard: both series end at the month the harness handed over.
        assert np.array_equal(history[:, target], levels)
        key = (model, lags, len(levels), steps)
        if key not in cache:
            cache[key] = ml_forecast(model, lags, history, steps)
        result = cache[key]
        return {"forecast": result["levels"][:, target], "trees": float(result["trees"][target])}

    return forecast
