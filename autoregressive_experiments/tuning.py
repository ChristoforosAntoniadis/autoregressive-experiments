"""
Time-aware random search: every model's settings, tuned on the training split only.

For each series and model, up to CANDIDATES settings are drawn at random, without repeats, from
the model's search space in ``configs/hyperparameters.json`` (smaller spaces are tried in full).
Each setting is scored by cross-validation in time order: FOLDS expanding folds of scikit-learn's
TimeSeriesSplit over the training split's monthly changes, the first fold training on at least
MIN_TRAIN_MONTHS months. Within a fold the model sees only the months before the fold, and is
scored on one-month-ahead forecasts of the fold's months, each made from the actual past. Point
models are scored by mean absolute error (the evaluation's one-month MASE ranks models the same
way, being MAE over a constant), volatility models by the mean pinball loss of their one-month
quantiles; the lowest score wins.

How each model forecasts a fold's months:

- SARIMA and SARIMAX: fitted once per fold, then run through the fold's months with their
  parameters fixed (statsmodels' extend), so each forecast uses the actual past
- GARCH, EGARCH and GJR-GARCH: fitted once per fold on the tuned SARIMA's in-sample residuals,
  then run through the fold's one-month errors with their parameters fixed (arch's fix)
- VAR, VECM, SMA and exponential smoothing: re-estimated at every month of the fold, exactly as
  the evaluation does (they are cheap to fit)
- the SARIMA + naive average: each weight applied to the tuned SARIMA's fold forecasts
- ML and neural networks: searched with scikit-learn's RandomizedSearchCV on the same folds, the
  settings run in parallel: fitted once per fold and predicting the fold's rows, whose features
  are actual past changes
- the TensorFlow and PyTorch networks: the same folds and score, but searched by Optuna's TPE
  sampler (CANDIDATES trials, one after another, seeded with RANDOM_SEED, no pruning), which
  chooses each trial's settings from the scores of the earlier ones
- Prophet: searched by Optuna the same way, refitted at every month of each fold, as the
  evaluation does (the refits run in parallel)

SARIMA is tuned first: its tuned orders feed the volatility models, SARIMAX and the average.

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` (the training split only).

Output:

- ``models/tuned_hyperparameters.json``: every model's tuned settings per series, read by the
  forecast evaluation
- ``reports/tuning.txt``: the folds, and per series and model the settings tried, the winner,
  its cross-validation score and naive's
- ``reports/tuning_scores.csv``: every setting tried and its score
- ``reports/figures/tuning.png``: per series, every setting's score relative to the reference
  (naive, or constant variance for the volatility models), the winner marked
"""

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

from arch import arch_model
from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import typer

from autoregressive_experiments.config import FIGURES_DIR, INTERIM_DATA_DIR, REPORTS_DIR
from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.configs.configurables import RANDOM_SEED
from autoregressive_experiments.diagnostics import (
    DIFFERENCING_ORDER,
    MAX_ORDER_SUM,
    SEASONAL_PERIOD,
    VALUE_COLUMNS,
    fit_sarima,
    repo_path,
)
from autoregressive_experiments.features import DATE_COLUMN, lagged_exog, split_boundaries
from autoregressive_experiments.forecast_evaluation import (
    EXPONENTIAL,
    QUANTILES,
    SARIMA,
    SARIMAX,
    SMA,
    TUNED_PATH,
    TUNING_REPORT_PATH,
    VAR_MODEL,
    VECM_MODEL,
    VOLATILITY_SPECS,
    pinball_loss,
    series_title,
    system_forecast,
    volatility_spec,
)
from autoregressive_experiments.ml_forecasting import (
    LAG_MODELS,
    ML_SPACES,
    RIDGE,
    build_model,
    lag_columns,
    lag_features,
    max_lag,
)
from autoregressive_experiments.nn_forecasting import NN_SPACES, OPTUNA_MODELS
from autoregressive_experiments.prophet_forecasting import (
    PROPHET,
    READY_MADE_SPACES,
    prophet_forecast,
)
from autoregressive_experiments.residual_checks import (
    MUTED_INK,
    PRIMARY_INK,
    SECONDARY_INK,
    SERIES_PALETTE,
    SURFACE,
    apply_chart_style,
)
from autoregressive_experiments.runlog import Progress, begin_phase, finish_run, start_run_log
from autoregressive_experiments.volatility_checks import regime_scales, variance_break

app = typer.Typer()

# Settings; each value's reasoning is in configs/hyperparameters.json.
SETTINGS = HYPERPARAMETERS.tuning
FOLDS = SETTINGS.folds
MIN_TRAIN_MONTHS = SETTINGS.min_train_months
CANDIDATES = SETTINGS.candidates
COMBINATION = "combination"
# Model key -> its search space, in tuning order: SARIMA first, since the volatility models,
# SARIMAX and the average build on its tuned orders. The volatility models share one space.
SPACES: dict[str, Mapping[str, Any]] = {
    SARIMA: SETTINGS.sarima,
    **{model: SETTINGS.variance for model in VOLATILITY_SPECS},
    SARIMAX: SETTINGS.sarimax,
    COMBINATION: SETTINGS.combination,
    VAR_MODEL: SETTINGS.var,
    VECM_MODEL: SETTINGS.vecm,
    SMA: SETTINGS.sma,
    EXPONENTIAL: SETTINGS.exponential_smoothing,
    **ML_SPACES,
    **NN_SPACES,
    **READY_MADE_SPACES,
}
# A cross-validation fold: (first month it tests, first month after it), as positions in the
# training split's monthly changes; the fold trains on every change before its first test month.
Fold = tuple[int, int]

SCORES_PATH = REPORTS_DIR / "tuning_scores.csv"
FIGURE_PATH = FIGURES_DIR / "tuning.png"


def time_series_folds(count: int) -> list[Fold]:
    """
    The expanding, time-ordered folds over `count` monthly changes.

    Parameters
    ----------
    count : int
        Monthly changes in the training split.

    Returns
    -------
    list of tuple of (int, int)
        FOLDS (start, end) pairs, oldest first: each fold tests changes[start:end] and trains on
        changes[:start]. The test blocks are adjacent, equally long and end at the last change.

    Notes
    -----
    scikit-learn's TimeSeriesSplit with test_size = (count - MIN_TRAIN_MONTHS) // FOLDS, so the
    first fold trains on at least MIN_TRAIN_MONTHS months (the rounding gives it the rest).
    """
    splitter = TimeSeriesSplit(n_splits=FOLDS, test_size=(count - MIN_TRAIN_MONTHS) // FOLDS)
    return [(int(test[0]), int(test[-1]) + 1) for _, test in splitter.split(np.arange(count))]


def valid_setting(model: str, setting: Mapping[str, Any]) -> bool:
    """
    Whether a combination of a search space's values is a model that exists.

    Parameters
    ----------
    model : str
        The model key (SARIMA, EXPONENTIAL, ...).
    setting : Mapping of str to Any
        One combination of its space's values.

    Returns
    -------
    bool
        False for SARIMA orders over the budget p + d + q <= MAX_ORDER_SUM, and for exponential
        smoothing with a damped trend but no trend; True otherwise.
    """
    if model == SARIMA:
        return bool(setting["p"] + DIFFERENCING_ORDER + setting["q"] <= MAX_ORDER_SUM)
    if model == EXPONENTIAL:
        return bool(setting["trend"] is not None or not setting["damped_trend"])
    return True


def draw_candidates(model: str, space: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """
    The settings to try: CANDIDATES drawn at random without repeats, or all of them when the
    space is that small.

    Parameters
    ----------
    model : str
        The model key, for valid_setting().
    space : Mapping of str to Any
        Setting name -> the list of values to try.

    Returns
    -------
    tuple of (list of dict of str to Any, int)
        The drawn settings, in draw order, and the number of valid settings in the space.

    Notes
    -----
    The draw is seeded with RANDOM_SEED afresh for every model, so each model's draw is
    reproducible on its own, whatever is tuned before it.
    """
    grid = [setting for setting in ParameterGrid(dict(space)) if valid_setting(model, setting)]
    generator = np.random.default_rng(RANDOM_SEED)
    picks = generator.choice(len(grid), size=min(CANDIDATES, len(grid)), replace=False)
    return [grid[int(index)] for index in picks], len(grid)


def arima_one_step(
    changes: np.ndarray,
    setting: Mapping[str, Any],
    folds: list[Fold],
    exog: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], int]:
    """
    One-month errors of a SARIMA (or SARIMAX) over every fold, parameters fixed within a fold.

    Parameters
    ----------
    changes : np.ndarray
        One series' monthly changes over the training split, shape (n,), trillions.
    setting : Mapping of str to Any
        SARIMA orders "p", "q", "P", "Q" and "trend".
    folds : list of tuple of (int, int)
        From time_series_folds(n).
    exog : np.ndarray or None, default=None
        SARIMAX regressors, shape (n - k, k): row i belongs to changes[i + k] (lagged_exog of
        the other series); the first k months then have no row and are left out of every fit.

    Returns
    -------
    tuple of (np.ndarray, list of tuple of (np.ndarray, np.ndarray), int)
        The one-month errors (actual - forecast) of every fold's months, concatenated, shape
        (tested months,), trillions; per fold, the fit's in-sample residuals (after the
        filter's burn-in) and its one-month errors on the fold, for the volatility models; and
        the number of fits whose optimizer didn't converge.

    Notes
    -----
    Processing, per fold:
    1. Fit on the changes before the fold (and their regressors).
    2. extend() the fit over the fold's months without re-estimating: its fitted values are
       the one-month forecasts, each from the actual months before it.
    """
    offset = 0 if exog is None else changes.shape[0] - exog.shape[0]
    errors, residuals, failed = [], [], 0
    for start, end in folds:
        fitted = fit_sarima(
            pd.Series(changes[offset:start]),
            setting["p"],
            setting["q"],
            setting["P"],
            setting["Q"],
            exog=None if exog is None else exog[: start - offset],
            trend=setting["trend"],
        )
        failed += int(not fitted.mle_retvals.get("converged", False))
        extended = fitted.extend(
            changes[start:end], exog=None if exog is None else exog[start - offset : end - offset]
        )
        fold_errors = changes[start:end] - np.asarray(extended.fittedvalues)
        errors.append(fold_errors)
        residuals.append(
            (np.asarray(fitted.resid)[fitted.loglikelihood_burn :], np.asarray(fold_errors))
        )
    return np.concatenate(errors), residuals, failed


def volatility_quantiles(
    model: str, setting: Mapping[str, Any], fit_on: np.ndarray, run_on: np.ndarray
) -> np.ndarray:
    """
    One-month QUANTILES of the shocks in `run_on` from a volatility model fitted on `fit_on`.

    Parameters
    ----------
    model : str
        "GARCH", "EGARCH" or "GJR-GARCH".
    setting : Mapping of str to Any
        Its "p", "q", "dist" and "break_adjusted".
    fit_on : np.ndarray
        The SARIMA's in-sample residuals before the fold, shape (n,), trillions.
    run_on : np.ndarray
        Its one-month errors in the fold, shape (m,), trillions.

    Returns
    -------
    np.ndarray
        Shape (m, 5), trillions: each fold month's quantiles, from the variance known one month
        ahead (the model's parameters fixed, its volatility driven by the actual past errors).

    Notes
    -----
    Processing:
    1. With the break adjustment, divide each regime of `fit_on` by its own root mean square
       (step U5's break, found on `fit_on` only) and the fold's errors by the last regime's.
    2. Fit the model on `fit_on` (zero mean, arch's rescaling).
    3. Run the fitted model over both series together with its parameters fixed; the fold
       months' conditional volatility is the one-month SD.
    4. Multiply by the fitted error distribution's standardized quantiles, and undo the
       rescaling and the regime scale.
    """
    spec = volatility_spec(model, setting)
    last_scale = 1.0
    if setting["break_adjusted"]:
        scales = regime_scales(fit_on, variance_break(fit_on))
        fit_on, last_scale = fit_on / scales, float(scales[-1])
        run_on = run_on / last_scale
    fitted = arch_model(fit_on, mean="Zero", rescale=True, **spec).fit(disp="off")
    both = np.concatenate([fit_on, run_on]) * fitted.scale
    fixed = arch_model(both, mean="Zero", rescale=False, **spec).fix(fitted.params.to_numpy())
    sd = np.asarray(fixed.conditional_volatility)[len(fit_on) :] / fitted.scale * last_scale
    distribution = fitted.model.distribution
    shape = fitted.params.to_numpy()[len(fitted.params) - distribution.num_params :]
    standardized = distribution.ppf(np.array(QUANTILES), shape if len(shape) else None)
    return sd[:, None] * np.asarray(standardized)[None, :]


def constant_quantiles(fit_on: np.ndarray, run_on: np.ndarray) -> np.ndarray:
    """
    The reference for the volatility models: normal quantiles with the residuals' constant SD.

    Parameters
    ----------
    fit_on : np.ndarray
        In-sample residuals before the fold, shape (n,), trillions.
    run_on : np.ndarray
        The fold's one-month errors, shape (m,), trillions.

    Returns
    -------
    np.ndarray
        Shape (m, 5), trillions: the same quantiles for every fold month, from the root mean
        square of `fit_on` (zero-mean residuals, as the volatility models assume).
    """
    sd = float(np.sqrt(np.mean(fit_on**2)))
    return np.tile(sd * stats.norm.ppf(QUANTILES), (len(run_on), 1))


def system_one_step(
    levels: np.ndarray,
    kind: str,
    setting: Mapping[str, Any],
    column: int,
    folds: list[Fold],
    cache: dict[tuple[str, str, int], np.ndarray],
) -> np.ndarray:
    """
    One-month errors of a VAR or VECM over every fold, re-estimated at every month.

    Parameters
    ----------
    levels : np.ndarray
        Both series' training levels, shape (n + 1, 2), trillions, VALUE_COLUMNS order.
    kind : str
        VAR_MODEL or VECM_MODEL.
    setting : Mapping of str to Any
        Its settings (see forecast_evaluation.system_forecast).
    column : int
        The series scored.
    folds : list of tuple of (int, int)
        Folds over the n monthly changes.
    cache : dict of tuple of (str, str, int) to np.ndarray
        Shared store of one-month forecasts of both series, keyed by (kind, settings as JSON,
        origin), so the second series reuses the first one's fits.

    Returns
    -------
    np.ndarray
        One-month errors of the series' level, every fold month, trillions.

    Notes
    -----
    Change t is the move into level t + 1, so its forecast comes from levels[: t + 1].
    """
    errors = []
    for start, end in folds:
        for month in range(start, end):
            key = (kind, json.dumps(setting, sort_keys=True), month)
            if key not in cache:
                cache[key] = system_forecast(kind, setting, levels[: month + 1], 1)[0][0]
            errors.append(levels[month + 1, column] - cache[key][column])
    return np.array(errors)


def smoothing_one_step(
    levels: np.ndarray, model: str, setting: Mapping[str, Any], folds: list[Fold]
) -> np.ndarray:
    """
    One-month errors of a moving average or exponential smoothing, refitted at every month.

    Parameters
    ----------
    levels : np.ndarray
        One series' training levels, shape (n + 1,), trillions.
    model : str
        SMA or EXPONENTIAL.
    setting : Mapping of str to Any
        {"window": k} for SMA; ExponentialSmoothing's keyword arguments otherwise.
    folds : list of tuple of (int, int)
        Folds over the n monthly changes.

    Returns
    -------
    np.ndarray
        One-month level errors, every fold month, trillions.
    """
    options = dict(setting)
    if model == EXPONENTIAL and setting["seasonal"]:
        options["seasonal_periods"] = SEASONAL_PERIOD
    errors = []
    for start, end in folds:
        for month in range(start, end):
            history = levels[: month + 1]
            if model == SMA:
                forecast = float(np.mean(history[-setting["window"] :]))
            else:
                forecast = float(ExponentialSmoothing(history, **options).fit().forecast(1)[0])
            errors.append(levels[month + 1] - forecast)
    return np.array(errors)


class TunableModel(RegressorMixin, BaseEstimator):
    """
    One ML model as a scikit-learn estimator whose settings, the lag count included, can be set
    by RandomizedSearchCV.

    Parameters
    ----------
    model : str, default=RIDGE
        One of LAG_MODELS.
    **settings : Any
        Its settings (search-space names), including "lags".

    Notes
    -----
    The features it receives hold max_lag(model) lags of both series; fit and predict keep the
    columns of settings["lags"] (lag_columns), so the lag count is searched like any other
    setting.
    The model itself comes from ml_forecasting.build_model, the same builder the evaluation
    uses, so Lasso with a zero penalty is a linear regression here too. get_params and
    set_params are written out because the settings differ by model and arrive as keywords.
    """

    def __init__(self, model: str = RIDGE, **settings: Any) -> None:
        self.model = model
        self.settings = settings

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        The model name and its settings, as scikit-learn's search and clone read them.

        Parameters
        ----------
        deep : bool, default=True
            Unused: there are no nested estimators to expand.

        Returns
        -------
        dict of str to Any
            {"model": ..., setting: value, ...}.
        """
        return {"model": self.model, **self.settings}

    def set_params(self, **params: Any) -> "TunableModel":
        """
        Change the model name or any settings.

        Parameters
        ----------
        **params : Any
            "model" and / or settings to replace.

        Returns
        -------
        TunableModel
            Itself, as scikit-learn expects.
        """
        self.model = params.pop("model", self.model)
        self.settings = {**self.settings, **params}
        return self

    def fit(self, features: np.ndarray, target: np.ndarray) -> "TunableModel":
        """
        Fit the model on the lag columns its settings ask for.

        Parameters
        ----------
        features : np.ndarray
            lag_features(changes, max_lag(model)) rows, shape (rows, 2 x max_lag(model)).
        target : np.ndarray
            The series' changes for those rows, shape (rows,), trillions.

        Returns
        -------
        TunableModel
            Itself, fitted.
        """
        columns = lag_columns(self.settings["lags"], max_lag(self.model), 2)
        self.fitted_ = build_model(self.model, self.settings).fit(features[:, columns], target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the series' changes for the given rows.

        Parameters
        ----------
        features : np.ndarray
            lag_features(changes, max_lag(model)) rows, shape (rows, 2 x max_lag(model)).

        Returns
        -------
        np.ndarray
            Predicted changes, shape (rows,), trillions.
        """
        columns = lag_columns(self.settings["lags"], max_lag(self.model), 2)
        return np.asarray(self.fitted_.predict(features[:, columns]))


def ml_search(
    changes: np.ndarray, model: str, column: int, folds: list[Fold]
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """
    Tune one ML model or network for one series with scikit-learn's RandomizedSearchCV.

    Parameters
    ----------
    changes : np.ndarray
        Both series' training changes, shape (n, 2), trillions.
    model : str
        One of LAG_MODELS.
    column : int
        The series predicted.
    folds : list of tuple of (int, int)
        Folds over the n monthly changes; their first test month must be at least the
        model's max_lag.

    Returns
    -------
    tuple of (dict of str to Any, list of dict of str to Any, int)
        As search(): the best setting, every drawn setting with its score (MAE, trillions) in
        draw order, and the number of settings in the space.

    Notes
    -----
    Processing:
    1. Build the features once with the model's max_lag lags, so every lag count is trained
       and scored on the same months.
    2. Turn the folds into row indices: row r's target is change r + max_lag, so a fold trains
       on the rows whose target comes before it and tests the rows inside it. Passing them
       explicitly matters: RandomizedSearchCV's default folds ignore time order.
    3. Draw CANDIDATES settings (seed RANDOM_SEED; all of them when the space is smaller), fit
       each on every fold in parallel, and score it by mean absolute error. The folds are equally
       long, so the mean of the fold MAEs is the MAE over all tested months, as for the other
       models. No final refit: the evaluation refits at every origin.
    """
    features, target, splits = lag_splits(changes, model, column, folds)
    space = dict(SPACES[model])
    logger.info(f"TUNE | {model}: {CANDIDATES} settings x {len(folds)} folds, in parallel")
    searcher = RandomizedSearchCV(
        TunableModel(model),
        param_distributions=space,
        n_iter=CANDIDATES,
        scoring="neg_mean_absolute_error",
        cv=splits,
        refit=False,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        error_score="raise",
    ).fit(features, target)
    results = searcher.cv_results_
    scored: list[dict[str, Any]] = [
        {"settings": dict(params), "score": -float(score)}
        for params, score in zip(results["params"], results["mean_test_score"])
    ]
    winner = min(scored, key=lambda row: row["score"])
    return dict(winner["settings"]), scored, len(ParameterGrid(space))


def lag_splits(
    changes: np.ndarray, model: str, column: int, folds: list[Fold]
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    The lag features, targets and fold row indices of one model's search.

    Parameters
    ----------
    changes : np.ndarray
        Both series' training changes, shape (n, 2), trillions.
    model : str
        One of LAG_MODELS; its max_lag sets how many lags the features hold.
    column : int
        The series predicted.
    folds : list of tuple of (int, int)
        Folds over the n monthly changes; their first test month must be at least the
        model's max_lag.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, list of tuple of (np.ndarray, np.ndarray))
        Features with max_lag lags, shape (n - max_lag, 2 x max_lag); the targets, shape
        (n - max_lag,); and per fold the training rows (targets before the fold) and the test
        rows (targets inside it).

    Notes
    -----
    Row r's target is change r + max_lag, so every lag count of the model is trained and
    scored on the same months, and every model is scored on the folds' months. Passing the
    folds explicitly matters: scikit-learn's default folds ignore time order.
    """
    longest = max_lag(model)
    features = lag_features(changes, longest)
    target = changes[longest:, column]
    rows = np.arange(len(target))
    splits = [
        (rows[: start - longest], rows[start - longest : end - longest]) for start, end in folds
    ]
    return features, target, splits


def run_optuna(
    label: str, model: str, score: Callable[[dict[str, Any]], float]
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """
    Search one model's space with Optuna's TPE sampler.

    Parameters
    ----------
    label : str
        Names the series in the progress lines.
    model : str
        The model key in SPACES.
    score : callable
        Setting -> its cross-validation score (lower is better).

    Returns
    -------
    tuple of (dict of str to Any, list of dict of str to Any, int)
        As search(): the best setting, every trial's setting and score in trial order, and 0
        for the space's size (it has continuous ranges).

    Notes
    -----
    Each trial suggests one value per setting: a choice from a list, or a float from a
    {"low", "high", "log"} range (on a log scale when "log" is true). CANDIDATES trials run one
    after another, so the seeded sampler's choices repeat exactly; no pruning, so every trial
    is scored on every fold.
    """
    import optuna

    space = SPACES[model]
    progress = Progress(f"TUNE | {label} | {model}", CANDIDATES)

    def objective(trial: Any) -> float:
        """
        The cross-validation score of the trial's settings.

        Parameters
        ----------
        trial : optuna.trial.Trial
            The trial suggesting the settings.

        Returns
        -------
        float
            score() of the suggested setting.
        """
        setting = {
            name: trial.suggest_float(name, spec["low"], spec["high"], log=spec["log"])
            if isinstance(spec, dict)
            else trial.suggest_categorical(name, spec)
            for name, spec in space.items()
        }
        return score(setting)

    def report(study: Any, trial: Any) -> None:
        """
        One progress line now and then, with the best score so far.

        Parameters
        ----------
        study : optuna.study.Study
            The running study.
        trial : optuna.trial.FrozenTrial
            The trial just finished.

        Returns
        -------
        None
        """
        progress.step(trial.number + 1, f"best so far {study.best_value:.5f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(objective, n_trials=CANDIDATES, callbacks=[report])
    # Every trial completes or raises, so each has a value.
    scored = [
        {"settings": dict(trial.params), "score": float(trial.value)}
        for trial in study.trials
        if trial.value is not None
    ]
    return dict(study.best_params), scored, 0


def optuna_search(
    changes: np.ndarray, model: str, column: int, folds: list[Fold]
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """
    Tune one network for one series with Optuna's TPE sampler.

    Parameters
    ----------
    changes : np.ndarray
        Both series' training changes, shape (n, 2), trillions.
    model : str
        One of OPTUNA_MODELS.
    column : int
        The series predicted.
    folds : list of tuple of (int, int)
        Folds over the n monthly changes.

    Returns
    -------
    tuple of (dict of str to Any, list of dict of str to Any, int)
        As run_optuna().

    Notes
    -----
    Each trial is scored like every other lag-feature model: fitted on each fold's earlier
    months and scored by the mean absolute error of the fold's months (cross_val_score on
    lag_splits' rows).
    """
    features, target, splits = lag_splits(changes, model, column, folds)

    def score(setting: dict[str, Any]) -> float:
        """
        Cross-validation MAE of one setting.

        Parameters
        ----------
        setting : dict of str to Any
            The trial's settings.

        Returns
        -------
        float
            Mean absolute one-month error over the fold months, trillions.
        """
        scores = cross_val_score(
            TunableModel(model, **setting),
            features,
            target,
            cv=splits,
            scoring="neg_mean_absolute_error",
            error_score="raise",
        )
        return -float(np.mean(scores))

    return run_optuna(series_title(VALUE_COLUMNS[column]), model, score)


def prophet_one_step(
    dates: pd.Series, levels: np.ndarray, setting: Mapping[str, Any], folds: list[Fold]
) -> np.ndarray:
    """
    One-month errors of Prophet over every fold, refitted at every month, in parallel.

    Parameters
    ----------
    dates : pd.Series
        The training split's month-end dates, one per level.
    levels : np.ndarray
        One series' training levels, shape (n + 1,), trillions.
    setting : Mapping of str to Any
        Prophet's settings.
    folds : list of tuple of (int, int)
        Folds over the n monthly changes.

    Returns
    -------
    np.ndarray
        One-month level errors, every fold month, trillions.

    Notes
    -----
    Change t is the move into level t + 1, so its forecast comes from the levels up to t, as
    for exponential smoothing. The refits are independent, so they run in parallel processes;
    each gives the same forecast wherever it runs.
    """
    from joblib import Parallel, delayed

    months = [month for start, end in folds for month in range(start, end)]
    forecasts = Parallel(n_jobs=-1)(
        delayed(prophet_forecast)(dates[: month + 1], levels[: month + 1], setting, 1)
        for month in months
    )
    return np.array([levels[month + 1] - mean[0] for month, (mean, _) in zip(months, forecasts)])


def search(
    label: str,
    model: str,
    score: Callable[..., float],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """
    Score every drawn setting of one model and return the best.

    Parameters
    ----------
    label : str
        Names the series in the progress lines.
    model : str
        The model key in SPACES.
    score : callable
        Setting -> its cross-validation score (lower is better).

    Returns
    -------
    tuple of (dict of str to Any, list of dict of str to Any, int)
        The best setting (the first drawn on a tie), every drawn setting with its score as
        {"settings": ..., "score": ...} in draw order, and the number of valid settings in the
        space.
    """
    candidates, size = draw_candidates(model, SPACES[model])
    progress = Progress(f"TUNE | {label} | {model}", len(candidates))
    scored: list[dict[str, Any]] = []
    for done, setting in enumerate(candidates, start=1):
        scored.append({"settings": setting, "score": float(score(setting))})
        best = min(row["score"] for row in scored)
        progress.step(done, f"best so far {best:.5f}")
    winner = min(scored, key=lambda row: row["score"])
    return dict(winner["settings"]), scored, size


def tune_series(
    levels: np.ndarray,
    dates: pd.Series,
    target: int,
    folds: list[Fold],
    system_cache: dict[tuple[str, str, int], np.ndarray],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """
    Tune every model for one series.

    Parameters
    ----------
    levels : np.ndarray
        Both series' training levels, shape (n + 1, 2), trillions, VALUE_COLUMNS order.
    dates : pd.Series
        Their month-end dates, for Prophet.
    target : int
        The series tuned (column of `levels`).
    folds : list of tuple of (int, int)
        From time_series_folds(n).
    system_cache : dict of tuple of (str, str, int) to np.ndarray
        Shared VAR / VECM forecasts (see system_one_step).

    Returns
    -------
    tuple of (dict of str to dict of str to Any, list of dict of str to Any)
        Model key -> tuned settings; and one summary per model: "model", "metric", "tried",
        "size", "best" (settings), "score", "reference" (naive's MAE, or constant variance's
        pinball loss), "failed" (SARIMA / SARIMAX fits that didn't converge) and "scores" (every
        setting's score).

    Notes
    -----
    Processing:
    1. SARIMA, scored by MAE; then the tuned SARIMA's fold residuals and errors.
    2. GARCH, EGARCH and GJR-GARCH on those, scored by pinball loss against constant variance.
    3. SARIMAX (the tuned orders plus the other series' lags) and the SARIMA + naive weight.
    4. VAR and VECM, SMA and exponential smoothing, scored by MAE.
    5. The ML models and neural networks, by RandomizedSearchCV (ml_search), or by Optuna
       for the TensorFlow and PyTorch networks (optuna_search), scored by MAE.
    6. Prophet, by Optuna, refitted at every fold month (prophet_one_step), scored by MAE.

    Naive's one-month forecast is no change, so its error is the change itself and its MAE the
    mean absolute change over the fold months: the reference every point model is shown
    against.
    """
    label = series_title(VALUE_COLUMNS[target])
    changes = np.diff(levels, axis=0)
    own, other = changes[:, target], changes[:, 1 - target]
    tested = np.concatenate([own[start:end] for start, end in folds])
    naive_mae = float(np.mean(np.abs(tested)))
    tuned: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    failures = {SARIMA: 0, SARIMAX: 0}

    def record(
        model: str,
        metric: str,
        reference: float,
        result: tuple[dict[str, Any], list[dict[str, Any]], int],
    ) -> None:
        """
        Store one model's search result: its winner and summary.

        Parameters
        ----------
        model : str
            The model key in SPACES.
        metric : str
            "MAE" or "pinball", for the report.
        reference : float
            The reference score the report and figure divide by.
        result : tuple of (dict of str to Any, list of dict of str to Any, int)
            What search() or ml_search() returned.

        Returns
        -------
        None
            Fills `tuned` and appends to `summaries`.
        """
        best, scored, size = result
        tuned[model] = best
        summaries.append(
            {
                "model": model,
                "metric": metric,
                "tried": len(scored),
                "size": size,
                "best": best,
                "score": min(row["score"] for row in scored),
                "reference": reference,
                "failed": failures.get(model, 0),
                "scores": scored,
            }
        )

    def arima_mae(setting: Mapping[str, Any], exog: np.ndarray | None = None) -> float:
        """
        Cross-validation MAE of a SARIMA, or of a SARIMAX when regressors are given.

        Parameters
        ----------
        setting : Mapping of str to Any
            SARIMA orders and trend.
        exog : np.ndarray or None, default=None
            The other series' lag matrix (SARIMAX) or None (SARIMA).

        Returns
        -------
        float
            Mean absolute one-month error over the fold months, trillions. Fits that didn't
            converge are added to `failures`.
        """
        errors, _, failed = arima_one_step(own, setting, folds, exog)
        failures[SARIMAX if exog is not None else SARIMA] += failed
        return float(np.mean(np.abs(errors)))

    record(SARIMA, "MAE", naive_mae, search(label, SARIMA, arima_mae))
    errors, residuals, _ = arima_one_step(own, tuned[SARIMA], folds)
    fold_errors = np.concatenate([run_on for _, run_on in residuals])
    reference = pinball_loss(
        fold_errors,
        np.vstack([constant_quantiles(fit_on, run_on) for fit_on, run_on in residuals]),
    )
    for model in VOLATILITY_SPECS:
        record(
            model,
            "pinball",
            reference,
            search(
                label,
                model,
                lambda setting, model=model: pinball_loss(
                    fold_errors,
                    np.vstack(
                        [
                            volatility_quantiles(model, setting, fit_on, run_on)
                            for fit_on, run_on in residuals
                        ]
                    ),
                ),
            ),
        )
    record(
        SARIMAX,
        "MAE",
        naive_mae,
        search(
            label,
            SARIMAX,
            lambda setting: arima_mae(tuned[SARIMA], lagged_exog(other, setting["lags"])),
        ),
    )
    # The average forecasts weight x SARIMA's change + (1 - weight) x naive's (none), so its
    # error is (1 - weight) x the change + weight x SARIMA's error.
    record(
        COMBINATION,
        "MAE",
        naive_mae,
        search(
            label,
            COMBINATION,
            lambda setting: float(
                np.mean(np.abs((1 - setting["weight"]) * tested + setting["weight"] * errors))
            ),
        ),
    )
    for kind in (VAR_MODEL, VECM_MODEL):
        record(
            kind,
            "MAE",
            naive_mae,
            search(
                label,
                kind,
                lambda setting, kind=kind: float(
                    np.mean(
                        np.abs(system_one_step(levels, kind, setting, target, folds, system_cache))
                    )
                ),
            ),
        )
    for model in (SMA, EXPONENTIAL):
        record(
            model,
            "MAE",
            naive_mae,
            search(
                label,
                model,
                lambda setting, model=model: float(
                    np.mean(np.abs(smoothing_one_step(levels[:, target], model, setting, folds)))
                ),
            ),
        )
    for model in LAG_MODELS:
        search_model = optuna_search if model in OPTUNA_MODELS else ml_search
        record(model, "MAE", naive_mae, search_model(changes, model, target, folds))
    record(
        PROPHET,
        "MAE",
        naive_mae,
        run_optuna(
            label,
            PROPHET,
            lambda setting: float(
                np.mean(np.abs(prophet_one_step(dates, levels[:, target], setting, folds)))
            ),
        ),
    )
    return tuned, summaries


def tuning_report(
    summaries: dict[str, list[dict[str, Any]]], folds: list[Fold], dates: pd.Series
) -> str:
    """
    The text report: the method, the folds, and per series and model the winner and its score.

    Parameters
    ----------
    summaries : dict of str to list of dict of str to Any
        Series -> tune_series' summaries.
    folds : list of tuple of (int, int)
        The folds, over the training changes.
    dates : pd.Series
        The training split's months (levels), so change t moves into dates[t + 1].

    Returns
    -------
    str
        The report text, lines joined with newlines.
    """
    lines = [
        "Hyperparameter tuning: time-aware random search on the training split",
        (
            f"Each model's settings are drawn at random from its search space ({CANDIDATES} "
            f"per model and series, or all when the space is smaller; seed {RANDOM_SEED}) and "
            f"scored by {FOLDS} expanding cross-validation folds in time order: each fold is "
            "fitted on the months before it only and forecasts each of its months one month ahead "
            "from the actual past. Point models: mean absolute error (trillions); volatility "
            "models: mean pinball loss of the one-month quantiles. The lowest wins and is "
            "fixed for the evaluation. Search spaces and their reasoning: "
            "autoregressive_experiments/configs/hyperparameters.json."
        ),
        "Folds (months tested; each trains on every month before):",
    ]
    for number, (start, end) in enumerate(folds, start=1):
        lines.append(
            f"  {number}: {dates.iloc[start + 1]:%Y-%m} to {dates.iloc[end]:%Y-%m} "
            f"({end - start} months; trains on {start} changes)"
        )
    lines.append("")
    for column, rows in summaries.items():
        lines.append(f"{column}")
        lines.append(
            f"  {'model':<24}{'tried':>12}{'metric':>9}{'score':>11}{'reference':>11}"
            f"{'ratio':>7}   tuned settings"
        )
        for row in rows:
            ratio = row["score"] / row["reference"]
            tried = f"{row['tried']} of {row['size']}" if row["size"] else f"{row['tried']} TPE"
            settings = ", ".join(f"{key}={value}" for key, value in row["best"].items())
            failed = f" ({row['failed']} fits didn't converge)" if row["failed"] else ""
            lines.append(
                f"  {row['model']:<24}{tried:>12}{row['metric']:>9}{row['score']:11.5f}"
                f"{row['reference']:11.5f}{ratio:7.3f}   {settings}{failed}"
            )
        lines.append("")
    lines.append(
        "reference: naive's MAE for point models (its one-month error is the change itself); "
        "constant-variance normal quantiles' pinball loss for the volatility models. ratio "
        "below 1: better than the reference in cross-validation."
    )
    lines.append(
        f"Every setting tried: {repo_path(SCORES_PATH)}; figure: {repo_path(FIGURE_PATH)}"
    )
    lines.append(f"Tuned settings: {repo_path(TUNED_PATH)}")
    return "\n".join(lines)


def tuning_figure(summaries: dict[str, list[dict[str, Any]]]) -> Figure:
    """
    Per series, every setting's cross-validation score relative to the reference, the winner
    marked.

    Parameters
    ----------
    summaries : dict of str to list of dict of str to Any
        Series -> tune_series' summaries.

    Returns
    -------
    Figure
        One panel per series; one row per model, its settings as hollow points and the winner
        filled, on a log scale around the reference line at 1. The caller saves and closes it.

    Notes
    -----
    The spread of a row shows how much the settings matter for that model; a winner left of 1
    beat naive (or constant variance) in cross-validation, which the evaluation then checks
    out of sample.
    """
    apply_chart_style()
    models = [row["model"] for row in next(iter(summaries.values()))]
    figure, axes = plt.subplots(
        1,
        len(summaries),
        figsize=(7.5 * len(summaries), 0.42 * len(models) + 2.2),
        squeeze=False,
        sharey=True,
        facecolor=SURFACE,
    )
    for axis, (column, rows) in zip(axes[0], summaries.items()):
        axis.set_facecolor(SURFACE)
        for position, row in enumerate(rows):
            ratios = np.array([entry["score"] for entry in row["scores"]]) / row["reference"]
            axis.scatter(
                ratios,
                np.full(len(ratios), position),
                s=22,
                facecolors="none",
                edgecolors=MUTED_INK,
                linewidths=1,
                label="a setting tried" if position == 0 else None,
            )
            axis.scatter(
                [row["score"] / row["reference"]],
                [position],
                s=60,
                color=SERIES_PALETTE[0],
                zorder=3,
                label="tuned (lowest score)" if position == 0 else None,
            )
        axis.axvline(1, color=PRIMARY_INK, linewidth=1.5, label="reference")
        axis.set_xscale("log")
        # Plain numbers (0.5, 2) instead of powers of ten on the log axis.
        plain = FuncFormatter(lambda value, _: f"{value:g}")
        axis.xaxis.set_major_formatter(plain)
        axis.xaxis.set_minor_formatter(plain)
        axis.set_yticks(range(len(models)), models)
        axis.set_xlabel("Cross-validation score / reference (log scale)", color=SECONDARY_INK)
        axis.set_title(
            f"{series_title(column)}: every setting tried, on the training folds\n"
            "left of 1: better than naive (volatility models: than constant variance)",
            loc="left",
            fontsize=10,
        )
        axis.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=3, fontsize=8, frameon=False
        )
    # Once only: the panels share the y axis, so inverting each would flip it back.
    axes[0][0].invert_yaxis()
    figure.tight_layout()
    return figure


@app.command()
def main(levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv") -> None:
    """
    Tune every model on the training split and write the tuned settings, report and figure.

    Parameters
    ----------
    levels_path : Path, default=INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv"
        Cleaned monthly levels of both series, trillions.

    Returns
    -------
    None
        Writes TUNED_PATH, TUNING_REPORT_PATH, SCORES_PATH and FIGURE_PATH, and prints the
        report.

    Notes
    -----
    Processing:
    1. Keep the training split's levels (the months before validation's first).
    2. Build the folds over its monthly changes.
    3. Tune every model for each series (tune_series).
    4. Write the tuned settings, every setting's score, the report and the figure.

    Validation and test months are never read: the split boundary is applied before anything
    else.
    """
    start_run_log("tune")
    begin_phase("Read the training split and build the folds")
    levels = pd.read_csv(levels_path, parse_dates=[DATE_COLUMN])
    levels = levels.sort_values(DATE_COLUMN).reset_index(drop=True)
    validation_start, _ = split_boundaries(levels)
    train = levels[levels[DATE_COLUMN] < validation_start].reset_index(drop=True)
    values = train[list(VALUE_COLUMNS)].to_numpy(dtype=float)
    folds = time_series_folds(len(values) - 1)
    system_cache: dict[tuple[str, str, int], np.ndarray] = {}
    tuned: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, list[dict[str, Any]]] = {}
    for target, column in enumerate(VALUE_COLUMNS):
        begin_phase(f"Tune every model for {series_title(column)}")
        tuned[column], summaries[column] = tune_series(
            values, train[DATE_COLUMN], target, folds, system_cache
        )

    begin_phase("Write the tuned settings, report and figure")
    TUNED_PATH.write_text(json.dumps(tuned, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "series": column,
                "model": row["model"],
                "metric": row["metric"],
                "settings": json.dumps(entry["settings"]),
                "score": entry["score"],
                "tuned": entry["settings"] == row["best"],
            }
            for column, rows in summaries.items()
            for row in rows
            for entry in row["scores"]
        ]
    ).to_csv(SCORES_PATH, index=False)
    report = tuning_report(summaries, folds, train[DATE_COLUMN])
    TUNING_REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    figure = tuning_figure(summaries)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"Tuned settings written to {TUNED_PATH}")
    finish_run()


if __name__ == "__main__":
    app()
