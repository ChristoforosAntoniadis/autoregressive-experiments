"""
Tuning and champion selection on the validation split: the middle part of the textbook three-way
split (Hastie, Tibshirani and Friedman 2009, section 7.2), in which training data fits each
model's parameters, validation chooses its hyperparameters and each family's champion, and the
test split (forecast_evaluation) measures the champions once.

Validation is rolling-origin with an expanding window (Tashman 2000): an origin every
VALIDATION_STEP periods through the validation split, at least 10 of them. At each origin a model
is fitted on every period from the start of the data up to the origin and forecasts up to
MAX_HORIZON ahead; only targets inside the validation split are scored, so test periods are
never seen. A setting's score is its MAE relative to the reference benchmark's (naive, or the
seasonal naive for seasonal data) at the same horizon, averaged over the forecast horizons: 0.8
means 20% more accurate than the reference on average, and every horizon
counts the same (a plain average of MASE would be set by the longest horizon, whose errors are
ten times the shortest's).

Each model's settings are searched by Optuna: spaces of at most CANDIDATES settings in full by
its grid sampler, larger ones by its seeded TPE sampler (Bergstra et al. 2011) with CANDIDATES
trials, run one after another so the choices repeat exactly. Settings that only exist together
(SARIMA's p and q under the order budget, exponential smoothing's damping and trend) are tried
as one choice of their valid combinations. A setting without a usable forecast, or whose fit
fails, ranks last. Univariate models are tuned per series; VAR, VECM and the lag-feature models
forecast both series together and are tuned once, on the two series' average score. Each
origin's fit runs in a separate process, in parallel; every fit gives the same forecast wherever
it runs.

Each tuned model's score gets a standard error from a bootstrap over the validation origins
(Efron and Tibshirani 1993), because a dozen origins decide it. Each family's champion per
series (econometric, machine learning, neural network; see model_forecasts) is its model with
the lowest score; the family's other models within one standard error of it are reported as
tied (the one-standard-error rule of Breiman et al. 1984; Hastie, Tibshirani and Friedman 2009,
section 7.10), so a win inside the noise is not read as a win.

Input: the cleaned levels (features.LEVELS_PATH; train and validation periods only).

Output:

- ``models/<dataset>/tuned_hyperparameters.json``: every model's tuned settings per series
- ``models/champions.json``: each series' champion per family, with its settings, validation
  score, standard error, tied models and MASE by horizon, read by forecast_evaluation
- ``reports/tuning.txt``: the origins, and per series every model's score, standard error and
  MASE by horizon, its settings, the champions and their ties
- ``reports/validation_metrics.csv``: every tuned model's and both benchmarks' metrics per
  series and horizon on validation, with the score and its standard error
- ``reports/tuning_scores.csv``: every setting tried and its score
- ``reports/<dataset>/figures/tuning.png``: per series, every setting's validation score, the tuned
  settings and the champions marked
"""

from collections.abc import Mapping, Sequence
import itertools
import json
from pathlib import Path
from typing import Any

from joblib import Parallel, cpu_count, delayed
from joblib.externals.loky import get_reusable_executor
from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np
import pandas as pd
import typer

from autoregressive_experiments.config import (
    DATASET,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)
from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.configs.configurables import RANDOM_SEED
from autoregressive_experiments.diagnostics import (
    DIFFERENCING_ORDER,
    MAX_ORDER_SUM,
    VALUE_COLUMNS,
    repo_path,
)
from autoregressive_experiments.features import (
    DATE_COLUMN,
    DATE_FORMAT,
    LEVELS_PATH,
    PERIOD,
    load_levels,
    series_title,
)
from autoregressive_experiments.metrics import (
    HORIZONS,
    MAX_HORIZON,
    METRICS,
    accuracy,
    mase_scale,
)
from autoregressive_experiments.ml_forecasting import ML_SPACES
from autoregressive_experiments.model_forecasts import (
    BENCHMARKS,
    COMBINATION,
    EXPONENTIAL,
    FAMILIES,
    JOINT_MODELS,
    LOG,
    MASE_LAG,
    REFERENCE,
    SARIMA,
    SARIMAX,
    SMA,
    VAR_MODEL,
    VECM_MODEL,
    family_of,
    joint_paths,
    series_path,
)
from autoregressive_experiments.modeling.utilities.torch_network import fit_device
from autoregressive_experiments.nn_forecasting import GPU_MODELS, NN_SPACES, SEQUENCE_MODELS
from autoregressive_experiments.prophet_forecasting import PROPHET, READY_MADE_SPACES
from autoregressive_experiments.residual_checks import (
    MUTED_INK,
    PRIMARY_INK,
    SECONDARY_INK,
    SERIES_PALETTE,
    SURFACE,
    apply_chart_style,
)
from autoregressive_experiments.runlog import Progress, begin_phase, finish_run, start_run_log

app = typer.Typer()

# Settings; each value's reasoning is in the dataset's config file (configs/<dataset>.json).
SETTINGS = HYPERPARAMETERS.tuning
VALIDATION_STEP = SETTINGS.validation_step
CANDIDATES = SETTINGS.candidates
BOOTSTRAP_SAMPLES = SETTINGS.bootstrap_samples
# Models within this many standard errors of a family's best score are reported as tied with it:
# the one-standard-error rule (Breiman et al. 1984).
TIE_STANDARD_ERRORS = 1.0
# Model -> its search space: setting name -> a list of values, or a {"low", "high", "log"} range.
SPACES: dict[str, Mapping[str, Any]] = {
    SARIMA: SETTINGS.sarima,
    SARIMAX: SETTINGS.sarimax,
    COMBINATION: SETTINGS.combination,
    VAR_MODEL: SETTINGS.var,
    VECM_MODEL: SETTINGS.vecm,
    SMA: SETTINGS.sma,
    EXPONENTIAL: SETTINGS.exponential_smoothing,
    **READY_MADE_SPACES,
    **ML_SPACES,
    **NN_SPACES,
}
# Settings that only exist together: tried as one choice of their valid combinations.
LINKED: dict[str, tuple[str, ...]] = {
    SARIMA: ("p", "q"),
    EXPONENTIAL: ("trend", "damped_trend", "seasonal", "use_boxcox", "transform"),
}
# The univariate models, in tuning order: SARIMA first, since SARIMAX and the average use it.
UNIVARIATE = (SARIMA, SARIMAX, COMBINATION, SMA, EXPONENTIAL, PROPHET)
# Processes fitting a GPU model's origins at once. The fits share one card, so beyond a few
# processes they only queue on it, and Windows refuses a CUDA context to each of many
# processes. Measured on the heaviest LSTM setting (60-day window, 32 units, 2 layers) over 23
# origins and both series on an RTX 3080 Ti: 2 workers 330 s, 4 workers 313 s, 8 workers 361 s,
# 16 workers refused ("device busy or unavailable"); 20 CPU workers 863 s.
GPU_JOBS = 4
# A model whose first settings all fail to fit is broken (a missing library, a GPU error), not
# unlucky: the run stops instead of logging sixty warnings and crowning a champion with no score.
FAILED_TRIALS_LIMIT = 3

TUNED_PATH = MODELS_DIR / "tuned_hyperparameters.json"
CHAMPIONS_PATH = MODELS_DIR / "champions.json"
REPORT_PATH = REPORTS_DIR / "tuning.txt"
VALIDATION_METRICS_PATH = REPORTS_DIR / "validation_metrics.csv"
SCORES_PATH = REPORTS_DIR / "tuning_scores.csv"
FIGURE_PATH = FIGURES_DIR / "tuning.png"
# The figure's axis ends here: a setting several times worse than naive is a failed setting,
# and a diverged network's score (thousands of times naive's) would squash the useful range.
MAX_SHOWN_RATIO = 3.0

# Per-horizon metrics of one series: horizon -> {"n", plus every name in metrics.METRICS}.
Metrics = dict[int, dict[str, float]]
# One series' forecasts: one level path per origin, in origin order.
Paths = list[np.ndarray]


def usable(score: float) -> float:
    """
    A score for ranking: itself when finite, infinity otherwise.

    Parameters
    ----------
    score : float
        A validation score; NaN or infinite when a setting gave no usable forecast (for example
        a multiplicative exponential-smoothing fit that broke down).

    Returns
    -------
    float
        `score`, or inf, so such a setting ranks last.

    Notes
    -----
    Every comparison with NaN is false, so min() over scores that include NaN can return the
    NaN one; ranking by usable() never does.
    """
    return score if np.isfinite(score) else np.inf


def validation_origins(train_end: int, validation_end: int) -> list[int]:
    """
    The validation origins: the last training month, then every VALIDATION_STEP months.

    Parameters
    ----------
    train_end : int
        Position of the last training month.
    validation_end : int
        Position of the last validation month.

    Returns
    -------
    list of int
        Origins train_end, train_end + VALIDATION_STEP, ..., all before validation_end, so each
        forecasts at least one validation month.
    """
    return list(range(train_end, validation_end, VALIDATION_STEP))


def valid_setting(model: str, setting: Mapping[str, Any]) -> bool:
    """
    Whether a combination of linked settings is a model that exists.

    Parameters
    ----------
    model : str
        A key of LINKED.
    setting : Mapping of str to Any
        Values of its linked settings.

    Returns
    -------
    bool
        False for SARIMA orders over the budget p + d + q <= MAX_ORDER_SUM; for exponential
        smoothing with a damped trend but no trend, or with the logarithm and a multiplicative
        trend or season or Box-Cox (those need positive data, and the logarithm of a level
        below 1 is negative); True otherwise.
    """
    if model == SARIMA:
        return bool(setting["p"] + DIFFERENCING_ORDER + setting["q"] <= MAX_ORDER_SUM)
    if model == EXPONENTIAL:
        damped_without_trend = setting["trend"] is None and setting["damped_trend"]
        positive_only = setting["trend"] == "mul" or setting["seasonal"] == "mul"
        positive_only = positive_only or bool(setting["use_boxcox"])
        return not damped_without_trend and not (setting["transform"] == LOG and positive_only)
    return True


def search_space(model: str) -> dict[str, Any]:
    """
    A model's space as Optuna parameters: its own, with linked settings merged into one choice.

    Parameters
    ----------
    model : str
        A key of SPACES.

    Returns
    -------
    dict of str to Any
        Parameter name -> a list of choices or a {"low", "high", "log"} range. Linked settings
        become one parameter named after them ("p+q"), whose choices are their valid
        combinations, encoded as JSON strings.
    """
    space = dict(SPACES[model])
    linked = LINKED.get(model, ())
    if not linked:
        return space
    combinations = [
        dict(zip(linked, values))
        for values in itertools.product(*(space.pop(name) for name in linked))
    ]
    valid = [json.dumps(c) for c in combinations if valid_setting(model, c)]
    return {"+".join(linked): valid, **space}


def decode(params: Mapping[str, Any]) -> dict[str, Any]:
    """
    A model's settings from Optuna parameters, unpacking merged linked settings.

    Parameters
    ----------
    params : Mapping of str to Any
        A trial's parameters.

    Returns
    -------
    dict of str to Any
        Setting name -> value.
    """
    settings: dict[str, Any] = {}
    for name, value in params.items():
        settings |= json.loads(value) if "+" in name else {name: value}
    return settings


def grid_size(space: Mapping[str, Any]) -> int | None:
    """
    The number of settings in a space made only of lists, or None if it has a range.

    Parameters
    ----------
    space : Mapping of str to Any
        From search_space().

    Returns
    -------
    int or None
        The product of the lists' lengths; None for a space with a continuous range.
    """
    if any(isinstance(choices, dict) for choices in space.values()):
        return None
    return int(np.prod([len(choices) for choices in space.values()]))


def workers(model: str) -> int:
    """
    How many processes fit a model's origins at once.

    Parameters
    ----------
    model : str
        One of BENCHMARKS, or any model of model_forecasts.FAMILIES.

    Returns
    -------
    int
        GPU_JOBS for a GPU model when a GPU is present; otherwise one process per physical
        core.

    Notes
    -----
    Physical cores, not logical ones: each fit runs on one thread, so a second process per core
    gains little, while every process holds its own copy of the libraries (with 20 processes on
    16 GB, free memory fell to 0.05 GB when the pool was re-created after a GPU model).
    """
    on_gpu = model in GPU_MODELS and fit_device(SEQUENCE_MODELS[model]) == "cuda"
    return GPU_JOBS if on_gpu else cpu_count(only_physical_cores=True)


def origin_paths(
    model: str,
    settings: Mapping[str, Any],
    levels: np.ndarray,
    dates: np.ndarray,
    origins: Sequence[int],
    last_target: int,
    columns: Sequence[int],
    sarima: Sequence[Mapping[str, Any]] = (),
) -> dict[int, list[np.ndarray]]:
    """
    A model's forecasts from every origin, fitted on the months up to each, in parallel.

    Parameters
    ----------
    model : str
        One of BENCHMARKS, or any model of model_forecasts.FAMILIES.
    settings : Mapping of str to Any
        Its settings.
    levels : np.ndarray
        Both series' levels, shape (months, 2), trillions; only months up to each origin are
        passed to the fit.
    dates : np.ndarray
        Their month-end dates.
    origins : Sequence of int
        Forecast origins (positions).
    last_target : int
        The last month that may be forecast: each origin forecasts min(MAX_HORIZON,
        last_target - origin) months.
    columns : Sequence of int
        The series to forecast.
    sarima : Sequence of Mapping of str to Any, default=()
        Both series' tuned SARIMA settings (for SARIMAX and the SARIMA + naive average).

    Returns
    -------
    dict of int to list of np.ndarray
        Column -> one level path per origin, in origin order.

    Notes
    -----
    A joint model forecasts both series in one fit per origin; a univariate one is fitted per
    series. The fits are independent, so they run in parallel processes (workers()).
    """
    steps = [min(MAX_HORIZON, last_target - origin) for origin in origins]
    if model in JOINT_MODELS:
        paths = Parallel(n_jobs=workers(model))(
            delayed(joint_paths)(model, settings, levels[: origin + 1], count)
            for origin, count in zip(origins, steps)
        )
        return {column: [path[:, column] for path in paths] for column in columns}
    jobs = [(column, origin, count) for column in columns for origin, count in zip(origins, steps)]
    paths = Parallel(n_jobs=workers(model))(
        delayed(series_path)(
            model, settings, levels[: origin + 1], dates[: origin + 1], column, count, sarima
        )
        for column, origin, count in jobs
    )
    result: dict[int, list[np.ndarray]] = {column: [] for column in columns}
    for (column, _, _), path in zip(jobs, paths):
        result[column].append(path)
    return result


def horizon_metrics(
    paths: Sequence[np.ndarray], origins: Sequence[int], actual: np.ndarray, scale: float
) -> Metrics:
    """
    MAE, MSE, RMSE, MAPE and MASE per horizon from one series' forecasts.

    Parameters
    ----------
    paths : Sequence of np.ndarray
        One level path per origin, each as long as that origin's forecast.
    origins : Sequence of int
        Their origins (positions).
    actual : np.ndarray
        The series' levels over the months the forecasts target, shape (months,), trillions.
    scale : float
        The MASE denominator: the training split's mean absolute monthly change.

    Returns
    -------
    dict of int to dict of str to float
        Horizon -> "n" (forecasts scored) and the five metrics, for every horizon in HORIZONS
        that some origin reaches.
    """
    metrics: Metrics = {}
    for horizon in HORIZONS:
        pairs = [
            (actual[origin + horizon], path[horizon - 1])
            for origin, path in zip(origins, paths)
            if len(path) >= horizon
        ]
        if pairs:
            truth, forecast = (np.array(values) for values in zip(*pairs))
            metrics[horizon] = {"n": float(len(pairs)), **accuracy(truth, forecast, scale)}
    return metrics


def score(metrics: Metrics, reference: Metrics) -> float:
    """
    The validation score: MAE relative to naive's, averaged over the horizons.

    Parameters
    ----------
    metrics : dict of int to dict of str to float
        A model's per-horizon metrics, from horizon_metrics().
    reference : dict of int to dict of str to float
        Naive's, on the same origins.

    Returns
    -------
    float
        The mean over the horizons both reach of the model's MAE / naive's MAE (NaN if a
        forecast was not finite); 1 is naive's own score.

    Notes
    -----
    Dividing horizon by horizon gives every horizon the same weight; the ratio also reads
    directly as the accuracy gained or lost against the benchmark.
    """
    horizons = [h for h in HORIZONS if h in metrics and h in reference]
    return float(np.mean([metrics[h]["MAE"] / reference[h]["MAE"] for h in horizons]))


def failed_metrics() -> Metrics:
    """
    The metrics of a setting whose fit failed: nothing scored, every metric NaN.

    Returns
    -------
    dict of int to dict of str to float
        Horizon -> {"n": 0, every METRICS name: NaN}, so the setting scores NaN and ranks
        last (usable()) while the tables keep their shape.
    """
    return {h: {"n": 0.0, **{name: np.nan for name in METRICS}} for h in HORIZONS}


def score_error(
    paths: Mapping[int, Paths],
    reference: Mapping[int, Paths],
    origins: Sequence[int],
    levels: np.ndarray,
    scales: Sequence[float],
    columns: Sequence[int],
) -> float:
    """
    The bootstrap standard error of a model's validation score.

    Parameters
    ----------
    paths : Mapping of int to list of np.ndarray
        The model's forecasts: column -> one level path per origin.
    reference : Mapping of int to list of np.ndarray
        Naive's, from the same origins.
    origins : Sequence of int
        The validation origins.
    levels : np.ndarray
        Both series' levels, shape (months, 2), trillions.
    scales : Sequence of float
        Each series' MASE denominator.
    columns : Sequence of int
        The series the model is scored on.

    Returns
    -------
    float
        The standard deviation of the score over BOOTSTRAP_SAMPLES resamples of the origins
        (drawn with replacement, seeded with RANDOM_SEED); NaN when the score itself is.

    Notes
    -----
    The origins are the sampling unit, because every forecast from one origin shares its fit
    and the months it saw; resampling single forecasts would understate the error (Efron and
    Tibshirani 1993, chapter 6). Model and naive are resampled together, so their ratio keeps
    the pairing.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        picked = rng.integers(0, len(origins), len(origins))
        chosen = [origins[i] for i in picked]
        per_series = []
        for column in columns:
            actual, scale = levels[:, column], scales[column]
            own = horizon_metrics([paths[column][i] for i in picked], chosen, actual, scale)
            naive = horizon_metrics([reference[column][i] for i in picked], chosen, actual, scale)
            per_series.append(score(own, naive))
        draws.append(float(np.mean(per_series)))
    return float(np.std(draws))


def tune_model(
    model: str,
    columns: Sequence[int],
    levels: np.ndarray,
    dates: np.ndarray,
    origins: Sequence[int],
    last_target: int,
    scales: Sequence[float],
    reference: Mapping[int, Paths],
    sarima: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """
    Search one model's settings on the validation origins with Optuna.

    Parameters
    ----------
    model : str
        A key of SPACES.
    columns : Sequence of int
        The series it is tuned on: one for a univariate model, both for a joint one (scored on
        their average).
    levels, dates, origins, last_target
        As in origin_paths(); last_target is the last validation month.
    scales : Sequence of float
        Each series' MASE denominator.
    reference : Mapping of int to list of np.ndarray
        Naive's forecasts from the same origins, column -> paths: the score's denominator.
    sarima : Sequence of Mapping of str to Any, default=()
        Both series' tuned SARIMA settings (for SARIMAX and the SARIMA + naive average).

    Returns
    -------
    dict of str to Any
        "settings" (the best), "score" (its validation score), "error" (column -> the
        bootstrap standard error of that series' own score), "metrics" (column -> its
        per-horizon metrics), "trials" (every trial's settings, score and per-series score) and
        "size" (settings in the space; None with a continuous range).

    Raises
    ------
    RuntimeError
        If the first FAILED_TRIALS_LIMIT settings all fail to fit.

    Notes
    -----
    Processing:
    1. Build the Optuna space (search_space()); use the grid sampler when it has at most
       CANDIDATES settings and no ranges, otherwise the TPE sampler seeded with RANDOM_SEED.
    2. Each trial forecasts from every origin and scores MAE relative to naive, averaged over
       the horizons and then over the series tuned on; an unusable score, or a fit that raises,
       counts as infinity, so one broken setting cannot stop a run of many hours. When the
       first FAILED_TRIALS_LIMIT settings all raise, the model itself is broken and the run
       stops.
    3. Keep the trial with the lowest score (the first one on a tie) and bootstrap each
       series' standard error from its own score, since the tie rule is applied per series.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = search_space(model)
    size = grid_size(space)
    grid = size is not None and size <= CANDIDATES
    sampler: Any = (
        optuna.samplers.GridSampler(dict(space), seed=RANDOM_SEED)
        if grid
        else optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    trials = size if grid and size is not None else CANDIDATES
    label = " + ".join(series_title(VALUE_COLUMNS[column]) for column in columns)
    progress = Progress(f"TUNE | {label} | {model}", trials)
    naive_metrics = {
        column: horizon_metrics(reference[column], origins, levels[:, column], scales[column])
        for column in columns
    }
    results: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        """
        The validation score of the trial's settings.

        Parameters
        ----------
        trial : optuna.trial.Trial
            The trial suggesting the settings.

        Returns
        -------
        float
            The mean over the series of their horizon-averaged MAE relative to naive; inf when
            unusable or when the fit failed.
        """
        params = {
            name: trial.suggest_float(name, choices["low"], choices["high"], log=choices["log"])
            if isinstance(choices, dict)
            else trial.suggest_categorical(name, choices)
            for name, choices in space.items()
        }
        settings = decode(params)
        failed = False
        try:
            paths = origin_paths(
                model, settings, levels, dates, origins, last_target, columns, sarima
            )
            metrics = {
                column: horizon_metrics(paths[column], origins, levels[:, column], scales[column])
                for column in columns
            }
        except Exception as error:  # noqa: BLE001  one failed setting must not end the run
            logger.warning(f"{model} {settings} failed and ranks last: {error!r}")
            paths, metrics, failed = {}, {column: failed_metrics() for column in columns}, True
        per_series = {column: score(metrics[column], naive_metrics[column]) for column in columns}
        value = usable(float(np.mean(list(per_series.values()))))
        results.append(
            {
                "settings": settings,
                "score": value,
                "per_series": per_series,
                "metrics": metrics,
                "paths": paths,
                "failed": failed,
            }
        )
        if len(results) >= FAILED_TRIALS_LIMIT and all(row["failed"] for row in results):
            raise RuntimeError(
                f"{model}: the first {len(results)} settings all failed to fit; stopping the run"
            )
        return value

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
        progress.step(trial.number + 1, f"best so far {study.best_value:.4f}")

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, callbacks=[report])
    best = min(results, key=lambda row: row["score"])
    errors = {
        column: score_error(best["paths"], reference, origins, levels, scales, [column])
        if np.isfinite(best["score"])
        else np.nan
        for column in columns
    }
    return {
        "settings": best["settings"],
        "score": best["score"],
        "error": errors,
        "metrics": best["metrics"],
        "trials": [
            {key: row[key] for key in ("settings", "score", "per_series")} for row in results
        ],
        "size": size,
    }


def choose_champions(
    scores: Mapping[int, Mapping[str, float]], errors: Mapping[int, Mapping[str, float]]
) -> dict[int, dict[str, dict[str, Any]]]:
    """
    Each series' champion per family, and the family's models tied with it.

    Parameters
    ----------
    scores : Mapping of int to Mapping of str to float
        Column -> model -> its tuned settings' validation score.
    errors : Mapping of int to Mapping of str to float
        Column -> model -> that score's standard error.

    Returns
    -------
    dict of int to dict of str to dict
        Column -> family -> {"model": the lowest score, "tied_with": the family's other
        models whose score is within TIE_STANDARD_ERRORS of the champion's standard error of
        it, in score order}. A family none of whose models has a finite score is left out,
        with a warning: a champion without a score would be fitted on test for nothing.

    Notes
    -----
    The one-standard-error rule: a difference smaller than the noise in the champion's own
    score is not evidence that the champion is better, so those models are named as ties
    rather than hidden behind a single winner.
    """
    champions: dict[int, dict[str, dict[str, Any]]] = {}
    for column, by_model in scores.items():
        champions[column] = {}
        for family, models in FAMILIES.items():
            ranked = sorted(
                (m for m in models if np.isfinite(by_model[m])), key=lambda m: by_model[m]
            )
            if not ranked:
                logger.warning(f"{VALUE_COLUMNS[column]}: no {family} model has a usable score")
                continue
            best = ranked[0]
            limit = by_model[best] + TIE_STANDARD_ERRORS * errors[column][best]
            tied = [m for m in ranked[1:] if usable(by_model[m]) <= limit]
            champions[column][family] = {"model": best, "tied_with": tied}
    return champions


def metrics_rows(
    metrics: Mapping[int, Mapping[str, Metrics]],
    scores: Mapping[int, Mapping[str, float]],
    errors: Mapping[int, Mapping[str, float]],
    champions: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> pd.DataFrame:
    """
    The validation metrics as a table.

    Parameters
    ----------
    metrics : Mapping of int to Mapping of str to dict
        Column -> model (benchmarks included) -> per-horizon metrics.
    scores : Mapping of int to Mapping of str to float
        Column -> model -> validation score (the benchmarks' too; naive's is 1).
    errors : Mapping of int to Mapping of str to float
        Column -> model -> the score's standard error (NaN for the benchmarks).
    champions : Mapping of int to Mapping of str to Mapping
        From choose_champions().

    Returns
    -------
    pd.DataFrame
        One row per series, model and horizon: "series", "family" ("benchmark" for naive and
        drift), "model", "champion", "horizon", "n", every METRICS name, "score" (the
        validation score) and "score_se" (its standard error).
    """
    rows = []
    for column, by_model in metrics.items():
        winners = {record["model"] for record in champions[column].values()}
        for model, per_horizon in by_model.items():
            family = "benchmark" if model in BENCHMARKS else family_of(model)
            for horizon, values in per_horizon.items():
                rows.append(
                    {
                        "series": VALUE_COLUMNS[column],
                        "family": family,
                        "model": model,
                        "champion": model in winners,
                        "horizon": horizon,
                        **values,
                        "score": scores[column][model],
                        "score_se": errors[column].get(model, np.nan),
                    }
                )
    return pd.DataFrame(rows)


def tuning_report(
    table: pd.DataFrame,
    tuned: Mapping[str, Mapping[str, Any]],
    champions: Mapping[int, Mapping[str, Mapping[str, Any]]],
    origins: Sequence[int],
    dates: pd.Series,
    validation_end: int,
) -> str:
    """
    The text report: the method, the origins, and per series every model's validation scores.

    Parameters
    ----------
    table : pd.DataFrame
        From metrics_rows().
    tuned : Mapping of str to Mapping of str to Any
        Series -> model -> tuned settings.
    champions : Mapping of int to Mapping of str to Mapping
        From choose_champions(), for the ties.
    origins : Sequence of int
        The validation origins.
    dates : pd.Series
        The dates of the whole sample.
    validation_end : int
        Position of the last validation period.

    Returns
    -------
    str
        The report text.
    """
    lines = [
        "Tuning and champion selection on the validation split",
        (
            f"{len(origins)} expanding-window origins, every {VALIDATION_STEP} {PERIOD}s from "
            f"{dates.iloc[origins[0]]:{DATE_FORMAT}} to {dates.iloc[origins[-1]]:{DATE_FORMAT}}: "
            f"each model is fitted on every {PERIOD} from {dates.iloc[0]:{DATE_FORMAT}} to the "
            f"origin and forecasts the validation {PERIOD}s up to "
            f"{dates.iloc[validation_end]:{DATE_FORMAT}}, at most "
            f"{MAX_HORIZON} ahead. Score: MAE relative to {REFERENCE}'s at the same horizon, "
            f"averaged over horizons {', '.join(map(str, HORIZONS))} (0.8: 20% more accurate than "
            f"{REFERENCE}; "
            "every horizon counts the same). se: the score's standard error from "
            f"{BOOTSTRAP_SAMPLES} bootstrap resamples of the origins. Settings searched by "
            f"Optuna (grid when the space has at most {CANDIDATES} settings, seeded TPE with "
            f"{CANDIDATES} trials otherwise); joint models (VAR, VECM, ML, networks) tuned once "
            "on both series. Champion per family and series: the lowest score; models within "
            f"{TIE_STANDARD_ERRORS:g} standard error of it are tied with it, not beaten. "
            "Search spaces and their reasoning: "
            f"autoregressive_experiments/configs/{DATASET}.json. The {', '.join(GPU_MODELS)} "
            f"were fitted on the {fit_device(SEQUENCE_MODELS[GPU_MODELS[0]])} (a GPU's results "
            "reproduce on a GPU, not bit for bit on a CPU)."
        ),
        "",
    ]
    for index, column in enumerate(VALUE_COLUMNS):
        rows = table[table["series"] == column]
        scores = rows.drop_duplicates("model").sort_values("score")
        lines.append(f"{column}  (score, its se, then MASE by horizon; * champion of its family)")
        header = "".join(f"{f'h={h}':>8}" for h in HORIZONS)
        lines.append(
            f"  {'model':<26}{'family':<18}{'score':>7}{'se':>7}{header}   tuned settings"
        )
        for row in scores.itertuples():
            per_horizon = rows[rows["model"] == row.model].set_index("horizon")["MASE"]
            cells = "".join(f"{per_horizon.get(h, np.nan):8.3f}" for h in HORIZONS)
            settings = ", ".join(f"{k}={v}" for k, v in tuned[column].get(row.model, {}).items())
            mark = "*" if row.champion else " "
            se = f"{row.score_se:7.3f}" if np.isfinite(row.score_se) else f"{'-':>7}"
            lines.append(
                f" {mark}{row.model:<26}{row.family:<18}{row.score:7.3f}{se}{cells}   {settings}"
            )
        for family in FAMILIES:
            record = champions[index].get(family)
            if record is None:
                lines.append(f"  {family} champion: none (no model gave a usable score)")
                continue
            tied = ", ".join(record["tied_with"]) or "none"
            lines.append(f"  {family} champion {record['model']}; tied with: {tied}")
        lines.append("")
    lines.append(
        f"All numbers: {repo_path(VALIDATION_METRICS_PATH)}; every setting tried: "
        f"{repo_path(SCORES_PATH)}; figure: {repo_path(FIGURE_PATH)}"
    )
    lines.append(
        f"Tuned settings: {repo_path(TUNED_PATH)}; champions: {repo_path(CHAMPIONS_PATH)}"
    )
    return "\n".join(lines)


def tuning_figure(
    trials: Mapping[str, Sequence[Mapping[str, Any]]], table: pd.DataFrame
) -> Figure:
    """
    Per series, every setting's validation score relative to naive's, the tuned one and the
    champions marked.

    Parameters
    ----------
    trials : Mapping of str to Sequence of Mapping of str to Any
        Model -> its trials (with "per_series": column -> score).
    table : pd.DataFrame
        From metrics_rows(): the tuned models' scores, naive's and the champion flags.

    Returns
    -------
    Figure
        One panel per series; one row per model (its family in the label), the settings tried
        as hollow points, the tuned one filled (palette slot 2 for the family champions), on a
        log scale around naive at 1 that ends at MAX_SHOWN_RATIO (settings beyond it shown as
        one arrow at the edge). The caller saves and closes it.
    """
    apply_chart_style()
    models = list(trials)
    figure, axes = plt.subplots(
        1,
        len(VALUE_COLUMNS),
        figsize=(8 * len(VALUE_COLUMNS), 0.4 * len(models) + 2.4),
        squeeze=False,
        sharey=True,
        facecolor=SURFACE,
    )
    families = table.drop_duplicates("model").set_index("model")["family"]
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    for column, axis in enumerate(axes[0]):
        name = VALUE_COLUMNS[column]
        rows = table[table["series"] == name].drop_duplicates("model").set_index("model")
        axis.set_facecolor(SURFACE)
        for position, model in enumerate(models):
            # The score is already the ratio to the reference, which sits at 1 by construction.
            scores = np.array(
                [trial["per_series"].get(column, np.nan) for trial in trials[model]], dtype=float
            )
            ratios = scores[np.isfinite(scores)]
            shown = ratios[ratios <= MAX_SHOWN_RATIO]
            axis.scatter(
                shown,
                np.full(len(shown), position),
                s=20,
                facecolors="none",
                edgecolors=MUTED_INK,
                linewidths=1,
            )
            beyond = int((ratios > MAX_SHOWN_RATIO).sum())
            if beyond:
                axis.scatter([MAX_SHOWN_RATIO], [position], marker=">", s=30, color=MUTED_INK)
            champion = bool(rows.loc[model, "champion"])
            axis.scatter(
                [min(rows.loc[model, "score"], MAX_SHOWN_RATIO)],
                [position],
                s=70 if champion else 45,
                color=SERIES_PALETTE[1] if champion else SERIES_PALETTE[0],
                zorder=3,
            )
        axis.axvline(1, color=PRIMARY_INK, linewidth=1.5)
        axis.set_xscale("log")
        axis.set_xlim(right=MAX_SHOWN_RATIO * 1.08)
        # Fixed ratios: the log locator leaves 1 to 2 unlabelled when the range is wide.
        low, high = axis.get_xlim()
        ticks = (0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 5, 7, 10, 20, 50)
        axis.set_xticks([r for r in ticks if low <= r <= high])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.xaxis.set_major_formatter(plain)
        axis.set_yticks(range(len(models)), [f"{m} ({families[m]})" for m in models])
        axis.set_xlabel(
            f"Validation score: MAE / {REFERENCE}'s, mean over horizons", color=SECONDARY_INK
        )
        axis.set_title(
            f"{series_title(name)}: every setting tried on the validation origins\n"
            f"left of the line at 1: more accurate than {REFERENCE}",
            loc="left",
            fontsize=10,
        )
    # One legend for both panels, from proxy markers.
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor=MUTED_INK,
            label="a setting tried",
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", color=SERIES_PALETTE[0], label="tuned (lowest score)"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=SERIES_PALETTE[1],
            label="tuned, family champion",
        ),
        plt.Line2D(
            [],
            [],
            marker=">",
            linestyle="",
            color=MUTED_INK,
            label=f"settings more than {MAX_SHOWN_RATIO:g}x worse than {REFERENCE} (off scale)",
        ),
        plt.Line2D([], [], color=PRIMARY_INK, label=REFERENCE),
    ]
    figure.legend(handles=handles, loc="lower center", ncols=5, fontsize=9, frameon=False)
    axes[0][0].invert_yaxis()
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


@app.command()
def main(levels_path: Path = LEVELS_PATH) -> None:
    """
    Tune every model on the validation origins, choose each family's champion, and write the
    settings, champions, report, tables and figure.

    Parameters
    ----------
    levels_path : Path, default=LEVELS_PATH
        Cleaned monthly levels of both series, trillions.

    Returns
    -------
    None
        Writes TUNED_PATH, CHAMPIONS_PATH, REPORT_PATH, VALIDATION_METRICS_PATH, SCORES_PATH and
        FIGURE_PATH, and prints the report.

    Notes
    -----
    Processing:
    1. Keep the train and validation periods; test periods are dropped before anything else.
    2. Build the validation origins and each series' MASE scale (on the training split), and
       score the benchmarks (among them the score's reference) on the origins.
    3. Tune SARIMA per series, then the other univariate models (SARIMAX and the average use the
       tuned SARIMA), then the joint models on both series.
    4. Stop joblib's workers, choose the champions and their ties, and write every output.
    """
    start_run_log("tune")
    begin_phase("Read the data and build the validation origins")
    data, splits = load_levels(levels_path)
    train_end = len(splits["train"]) - 1
    validation_end = train_end + len(splits["validation"])
    known = data.iloc[: validation_end + 1]
    levels = known[list(VALUE_COLUMNS)].to_numpy(dtype=float)
    dates = known[DATE_COLUMN].to_numpy()
    origins = validation_origins(train_end, validation_end)
    if levels.min() <= 0 and any(LOG in space.get("transform", ()) for space in SPACES.values()):
        raise ValueError(
            "a level is zero or negative, so no model can work on the logarithm: remove "
            '"log" from every transform list in configs/<dataset>.json'
        )
    scales = [
        mase_scale(splits["train"][column].to_numpy(dtype=float), MASE_LAG)
        for column in VALUE_COLUMNS
    ]
    columns = list(range(len(VALUE_COLUMNS)))
    tuned: dict[str, dict[str, Any]] = {name: {} for name in VALUE_COLUMNS}
    metrics: dict[int, dict[str, Metrics]] = {column: {} for column in columns}
    scores: dict[int, dict[str, float]] = {column: {} for column in columns}
    errors: dict[int, dict[str, float]] = {column: {} for column in columns}
    trials: dict[str, list[dict[str, Any]]] = {}
    benchmarks = {
        model: origin_paths(model, {}, levels, dates, origins, validation_end, columns)
        for model in BENCHMARKS
    }
    reference = benchmarks[REFERENCE]
    for column in columns:
        for model, paths in benchmarks.items():
            metrics[column][model] = horizon_metrics(
                paths[column], origins, levels[:, column], scales[column]
            )
        # Scored after every benchmark's metrics exist: the reference may be listed last.
        for model in benchmarks:
            scores[column][model] = score(metrics[column][model], metrics[column][REFERENCE])

    def store(model: str, result: Mapping[str, Any], tuned_columns: Sequence[int]) -> None:
        """
        Keep one search's winner, metrics, score and trials.

        Parameters
        ----------
        model : str
            The model tuned.
        result : Mapping of str to Any
            From tune_model().
        tuned_columns : Sequence of int
            The series it was tuned on.

        Returns
        -------
        None
        """
        for column in tuned_columns:
            tuned[VALUE_COLUMNS[column]][model] = result["settings"]
            metrics[column][model] = result["metrics"][column]
            scores[column][model] = score(result["metrics"][column], metrics[column][REFERENCE])
            errors[column][model] = result["error"][column]
        trials.setdefault(model, []).extend(result["trials"])

    begin_phase("Tune the univariate models per series")
    for model in UNIVARIATE:
        sarima = [tuned[name].get(SARIMA, {}) for name in VALUE_COLUMNS]
        for column in columns:
            result = tune_model(
                model, [column], levels, dates, origins, validation_end, scales, reference, sarima
            )
            store(model, result, [column])
    begin_phase("Tune the joint models on both series")
    logger.info(
        f"{', '.join(GPU_MODELS)} fit on the {fit_device(SEQUENCE_MODELS[GPU_MODELS[0]])}, "
        f"{workers(GPU_MODELS[0])} origins at a time"
    )
    for model in JOINT_MODELS:
        result = tune_model(
            model, columns, levels, dates, origins, validation_end, scales, reference
        )
        store(model, result, columns)
    begin_phase("Choose the champions and write the outputs")
    # The parallel fits are done: stop joblib's workers now, releasing their memory.
    get_reusable_executor().shutdown(wait=True)
    champions = choose_champions(
        {c: {m: v for m, v in scores[c].items() if m not in BENCHMARKS} for c in columns}, errors
    )
    table = metrics_rows(metrics, scores, errors, champions)
    TUNED_PATH.write_text(json.dumps(tuned, indent=2) + "\n", encoding="utf-8")
    champion_record = {
        VALUE_COLUMNS[column]: {
            family: {
                "model": record["model"],
                "settings": tuned[VALUE_COLUMNS[column]][record["model"]],
                "validation_score": scores[column][record["model"]],
                "validation_score_se": errors[column][record["model"]],
                "tied_with": record["tied_with"],
                "validation_MASE": {
                    str(h): values["MASE"]
                    for h, values in metrics[column][record["model"]].items()
                },
            }
            for family, record in by_family.items()
        }
        for column, by_family in champions.items()
    }
    CHAMPIONS_PATH.write_text(json.dumps(champion_record, indent=2) + "\n", encoding="utf-8")
    table.to_csv(VALIDATION_METRICS_PATH, index=False)
    pd.DataFrame(
        [
            {
                "model": model,
                "trial": number,
                "settings": json.dumps(trial["settings"]),
                "score": trial["score"],
                **{f"{VALUE_COLUMNS[c]}_score": s for c, s in trial["per_series"].items()},
            }
            for model, rows in trials.items()
            for number, trial in enumerate(rows)
        ]
    ).to_csv(SCORES_PATH, index=False)
    report = tuning_report(table, tuned, champions, origins, data[DATE_COLUMN], validation_end)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    figure = tuning_figure(trials, table)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"Champions written to {CHAMPIONS_PATH}")
    finish_run()


if __name__ == "__main__":
    app()
