"""
Tuning and champion selection on the validation split: the middle part of the textbook three-way
split (Hastie, Tibshirani and Friedman 2009, section 7.2), in which training data fits each
model's parameters, validation chooses its hyperparameters and each family's champion, and the
test split (forecast_evaluation) measures the champions once.

Validation is rolling-origin with an expanding window (Tashman 2000): an origin every
VALIDATION_STEP months through the validation split, at least 10 of them. At each origin a model
is fitted on every month from the start of the data up to the origin and forecasts up to 12
months ahead; only targets inside the validation split are scored, so test months are never
seen. A setting's score is its MASE (Hyndman and Koehler 2006) averaged over the forecast
horizons, so it must forecast well at every horizon.

Each model's settings are searched by Optuna: spaces of at most CANDIDATES settings in full by
its grid sampler, larger ones by its seeded TPE sampler (Bergstra et al. 2011) with CANDIDATES
trials, run one after another so the choices repeat exactly. Settings that only exist together
(SARIMA's p and q under the order budget, exponential smoothing's damping and trend) are tried
as one choice of their valid combinations. A setting without a usable forecast ranks last.
Univariate models are tuned per series; VAR, VECM and the lag-feature models forecast both
series together and are tuned once, on the two series' average score. Each origin's fit runs in
a separate process, in parallel; every fit gives the same forecast wherever it runs.

Each family's champion per series (econometric, machine learning, neural network; see
model_forecasts) is its model with the lowest validation score.

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` (train and validation months only).

Output:

- ``models/tuned_hyperparameters.json``: every model's tuned settings per series
- ``models/champions.json``: each series' champion per family, with its settings and validation
  scores, read by forecast_evaluation
- ``reports/tuning.txt``: the origins, and per series every model's validation MASE by horizon,
  its settings, and the champions
- ``reports/validation_metrics.csv``: every tuned model's and naive's MAE, MSE, RMSE, MAPE and
  MASE per series and horizon on validation
- ``reports/tuning_scores.csv``: every setting tried and its score
- ``reports/figures/tuning.png``: per series, every setting's validation score relative to naive's,
  the tuned settings and the champions marked
"""

from collections.abc import Mapping, Sequence
import itertools
import json
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator
import numpy as np
import pandas as pd
import typer

from autoregressive_experiments.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
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
    series_title,
    split_boundaries,
    split_by_time,
)
from autoregressive_experiments.metrics import HORIZONS, MAX_HORIZON, accuracy, mase_scale
from autoregressive_experiments.ml_forecasting import ML_SPACES
from autoregressive_experiments.model_forecasts import (
    COMBINATION,
    EXPONENTIAL,
    FAMILIES,
    JOINT_MODELS,
    NAIVE,
    SARIMA,
    SARIMAX,
    SMA,
    VAR_MODEL,
    VECM_MODEL,
    family_of,
    joint_paths,
    series_path,
)
from autoregressive_experiments.nn_forecasting import NN_SPACES
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

# Settings; each value's reasoning is in configs/hyperparameters.json.
SETTINGS = HYPERPARAMETERS.tuning
VALIDATION_STEP = SETTINGS.validation_step
CANDIDATES = SETTINGS.candidates
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
LINKED: dict[str, tuple[str, ...]] = {SARIMA: ("p", "q"), EXPONENTIAL: ("trend", "damped_trend")}
# The univariate models, in tuning order: SARIMA first, since SARIMAX and the average use it.
UNIVARIATE = (SARIMA, SARIMAX, COMBINATION, SMA, EXPONENTIAL, PROPHET)

TUNED_PATH = MODELS_DIR / "tuned_hyperparameters.json"
CHAMPIONS_PATH = MODELS_DIR / "champions.json"
REPORT_PATH = REPORTS_DIR / "tuning.txt"
VALIDATION_METRICS_PATH = REPORTS_DIR / "validation_metrics.csv"
SCORES_PATH = REPORTS_DIR / "tuning_scores.csv"
FIGURE_PATH = FIGURES_DIR / "tuning.png"

# Per-horizon metrics of one series: horizon -> {"n", "MAE", "MSE", "RMSE", "MAPE", "MASE"}.
Metrics = dict[int, dict[str, float]]


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
        False for SARIMA orders over the budget p + d + q <= MAX_ORDER_SUM, and for exponential
        smoothing with a damped trend but no trend; True otherwise.
    """
    if model == SARIMA:
        return bool(setting["p"] + DIFFERENCING_ORDER + setting["q"] <= MAX_ORDER_SUM)
    if model == EXPONENTIAL:
        return bool(setting["trend"] is not None or not setting["damped_trend"])
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
        NAIVE, or any model of model_forecasts.FAMILIES.
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
    series. The fits are independent, so they run in parallel processes.
    """
    steps = [min(MAX_HORIZON, last_target - origin) for origin in origins]
    if model in JOINT_MODELS:
        paths = Parallel(n_jobs=-1)(
            delayed(joint_paths)(model, settings, levels[: origin + 1], count)
            for origin, count in zip(origins, steps)
        )
        return {column: [path[:, column] for path in paths] for column in columns}
    jobs = [(column, origin, count) for column in columns for origin, count in zip(origins, steps)]
    paths = Parallel(n_jobs=-1)(
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


def mean_mase(metrics: Metrics) -> float:
    """
    MASE averaged over the horizons: the validation score.

    Parameters
    ----------
    metrics : dict of int to dict of str to float
        From horizon_metrics().

    Returns
    -------
    float
        The mean of the horizons' MASE (NaN if a forecast was not finite).
    """
    return float(np.mean([metrics[horizon]["MASE"] for horizon in HORIZONS]))


def tune_model(
    model: str,
    columns: Sequence[int],
    levels: np.ndarray,
    dates: np.ndarray,
    origins: Sequence[int],
    last_target: int,
    scales: Sequence[float],
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
    sarima : Sequence of Mapping of str to Any, default=()
        Both series' tuned SARIMA settings (for SARIMAX and the SARIMA + naive average).

    Returns
    -------
    dict of str to Any
        "settings" (the best), "score" (its validation score), "metrics" (column -> its
        per-horizon metrics), "trials" (every trial's settings, score and per-series score) and
        "size" (settings in the space; None with a continuous range).

    Notes
    -----
    Processing:
    1. Build the Optuna space (search_space()); use the grid sampler when it has at most
       CANDIDATES settings and no ranges, otherwise the TPE sampler seeded with RANDOM_SEED.
    2. Each trial forecasts from every origin and scores the mean of the horizons' MASE,
       averaged over the series tuned on; an unusable score counts as infinity.
    3. Keep the trial with the lowest score (the first one on a tie).
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
            The mean over the series of their horizon-averaged MASE; inf when unusable.
        """
        params = {
            name: trial.suggest_float(name, choices["low"], choices["high"], log=choices["log"])
            if isinstance(choices, dict)
            else trial.suggest_categorical(name, choices)
            for name, choices in space.items()
        }
        settings = decode(params)
        paths = origin_paths(model, settings, levels, dates, origins, last_target, columns, sarima)
        metrics = {
            column: horizon_metrics(paths[column], origins, levels[:, column], scales[column])
            for column in columns
        }
        per_series = {column: mean_mase(metrics[column]) for column in columns}
        score = usable(float(np.mean(list(per_series.values()))))
        results.append(
            {"settings": settings, "score": score, "per_series": per_series, "metrics": metrics}
        )
        return score

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
    return {
        "settings": best["settings"],
        "score": best["score"],
        "metrics": best["metrics"],
        "trials": [
            {key: row[key] for key in ("settings", "score", "per_series")} for row in results
        ],
        "size": size,
    }


def choose_champions(
    metrics: Mapping[int, Mapping[str, Metrics]],
) -> dict[int, dict[str, str]]:
    """
    Each series' champion per family: the model with the lowest validation score.

    Parameters
    ----------
    metrics : Mapping of int to Mapping of str to dict
        Column -> model -> its tuned settings' validation metrics.

    Returns
    -------
    dict of int to dict of str to str
        Column -> family -> champion model.
    """
    return {
        column: {
            family: min(models, key=lambda model: usable(mean_mase(by_model[model])))
            for family, models in FAMILIES.items()
        }
        for column, by_model in metrics.items()
    }


def metrics_rows(
    metrics: Mapping[int, Mapping[str, Metrics]], champions: Mapping[int, Mapping[str, str]]
) -> pd.DataFrame:
    """
    The validation metrics as a table.

    Parameters
    ----------
    metrics : Mapping of int to Mapping of str to dict
        Column -> model (naive included) -> per-horizon metrics.
    champions : Mapping of int to Mapping of str to str
        From choose_champions().

    Returns
    -------
    pd.DataFrame
        One row per series, model and horizon: "series", "family" ("benchmark" for naive),
        "model", "champion", "horizon", "n", "MAE", "MSE", "RMSE", "MAPE", "MASE", and
        "mean_MASE" (the validation score).
    """
    rows = []
    for column, by_model in metrics.items():
        winners = set(champions[column].values())
        for model, per_horizon in by_model.items():
            family = "benchmark" if model == NAIVE else family_of(model)
            for horizon, values in per_horizon.items():
                rows.append(
                    {
                        "series": VALUE_COLUMNS[column],
                        "family": family,
                        "model": model,
                        "champion": model in winners,
                        "horizon": horizon,
                        **values,
                        "mean_MASE": mean_mase(per_horizon),
                    }
                )
    return pd.DataFrame(rows)


def tuning_report(
    table: pd.DataFrame,
    tuned: Mapping[str, Mapping[str, Any]],
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
    origins : Sequence of int
        The validation origins.
    dates : pd.Series
        Month-end dates of the whole sample.
    validation_end : int
        Position of the last validation month.

    Returns
    -------
    str
        The report text.
    """
    lines = [
        "Tuning and champion selection on the validation split",
        (
            f"{len(origins)} expanding-window origins, every {VALIDATION_STEP} months from "
            f"{dates.iloc[origins[0]]:%Y-%m} to {dates.iloc[origins[-1]]:%Y-%m}: each model is "
            f"fitted on every month from {dates.iloc[0]:%Y-%m} to the origin and forecasts the "
            f"validation months up to {dates.iloc[validation_end]:%Y-%m}, at most "
            f"{MAX_HORIZON} ahead. Score: MASE averaged over horizons "
            f"{', '.join(map(str, HORIZONS))}. Settings searched by Optuna (grid when the space "
            f"has at most {CANDIDATES} settings, seeded TPE with {CANDIDATES} trials "
            "otherwise); joint models (VAR, VECM, ML, networks) tuned once on both series. "
            "Champion per family and series: the lowest score. Search spaces and their "
            "reasoning: autoregressive_experiments/configs/hyperparameters.json."
        ),
        "",
    ]
    for column in VALUE_COLUMNS:
        rows = table[table["series"] == column]
        scores = rows.drop_duplicates("model").sort_values("mean_MASE")
        lines.append(f"{column}  (validation MASE by horizon; * champion of its family)")
        header = "".join(f"{f'h={h}':>8}" for h in HORIZONS)
        lines.append(f"  {'model':<26}{'family':<18}{'mean':>8}{header}   tuned settings")
        for row in scores.itertuples():
            per_horizon = rows[rows["model"] == row.model].set_index("horizon")["MASE"]
            cells = "".join(f"{per_horizon.get(h, np.nan):8.3f}" for h in HORIZONS)
            settings = ", ".join(f"{k}={v}" for k, v in tuned[column].get(row.model, {}).items())
            mark = "*" if row.champion else " "
            lines.append(
                f" {mark}{row.model:<26}{row.family:<18}{row.mean_MASE:8.3f}{cells}   {settings}"
            )
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
        log scale around naive at 1. The caller saves and closes it.
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
        reference = rows.loc[NAIVE, "mean_MASE"]
        axis.set_facecolor(SURFACE)
        for position, model in enumerate(models):
            scores = np.array(
                [trial["per_series"].get(column, np.nan) for trial in trials[model]], dtype=float
            )
            ratios = scores[np.isfinite(scores)] / reference
            axis.scatter(
                ratios,
                np.full(len(ratios), position),
                s=20,
                facecolors="none",
                edgecolors=MUTED_INK,
                linewidths=1,
            )
            champion = bool(rows.loc[model, "champion"])
            axis.scatter(
                [rows.loc[model, "mean_MASE"] / reference],
                [position],
                s=70 if champion else 45,
                color=SERIES_PALETTE[1] if champion else SERIES_PALETTE[0],
                zorder=3,
            )
        axis.axvline(1, color=PRIMARY_INK, linewidth=1.5)
        axis.set_xscale("log")
        # Fixed ratios: the log locator leaves 1 to 2 unlabelled when the range is wide.
        low, high = axis.get_xlim()
        ratios = (0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 5, 7, 10, 20, 50)
        axis.set_xticks([r for r in ratios if low <= r <= high])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.xaxis.set_major_formatter(plain)
        axis.set_yticks(range(len(models)), [f"{m} ({families[m]})" for m in models])
        axis.set_xlabel("Validation MASE (mean over horizons) / naive's", color=SECONDARY_INK)
        axis.set_title(
            f"{series_title(name)}: every setting tried on the validation origins\n"
            "left of the line at 1: more accurate than naive",
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
        plt.Line2D([], [], color=PRIMARY_INK, label="naive"),
    ]
    figure.legend(handles=handles, loc="lower center", ncols=4, fontsize=9, frameon=False)
    axes[0][0].invert_yaxis()
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


@app.command()
def main(levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv") -> None:
    """
    Tune every model on the validation origins, choose each family's champion, and write the
    settings, champions, report, tables and figure.

    Parameters
    ----------
    levels_path : Path, default=INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv"
        Cleaned monthly levels of both series, trillions.

    Returns
    -------
    None
        Writes TUNED_PATH, CHAMPIONS_PATH, REPORT_PATH, VALIDATION_METRICS_PATH, SCORES_PATH and
        FIGURE_PATH, and prints the report.

    Notes
    -----
    Processing:
    1. Keep the train and validation months; test months are dropped before anything else.
    2. Build the validation origins and each series' MASE scale (on the training split).
    3. Tune SARIMA per series, then the other univariate models (SARIMAX and the average use the
       tuned SARIMA), then the joint models on both series.
    4. Score naive on the same origins, as the reference.
    5. Stop joblib's workers, choose the champions and write every output.
    """
    start_run_log("tune")
    begin_phase("Read the data and build the validation origins")
    data = pd.read_csv(levels_path, parse_dates=[DATE_COLUMN])
    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)
    splits = split_by_time(data, split_boundaries(data))
    train_end = len(splits["train"]) - 1
    validation_end = train_end + len(splits["validation"])
    known = data.iloc[: validation_end + 1]
    levels = known[list(VALUE_COLUMNS)].to_numpy(dtype=float)
    dates = known[DATE_COLUMN].to_numpy()
    origins = validation_origins(train_end, validation_end)
    scales = [
        mase_scale(splits["train"][column].to_numpy(dtype=float)) for column in VALUE_COLUMNS
    ]
    columns = list(range(len(VALUE_COLUMNS)))
    tuned: dict[str, dict[str, Any]] = {name: {} for name in VALUE_COLUMNS}
    metrics: dict[int, dict[str, Metrics]] = {column: {} for column in columns}
    trials: dict[str, list[dict[str, Any]]] = {}

    def store(model: str, result: Mapping[str, Any], tuned_columns: Sequence[int]) -> None:
        """
        Keep one search's winner, metrics and trials.

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
        trials.setdefault(model, []).extend(result["trials"])

    begin_phase("Tune the univariate models per series")
    for model in UNIVARIATE:
        sarima = [tuned[name].get(SARIMA, {}) for name in VALUE_COLUMNS]
        for column in columns:
            result = tune_model(
                model, [column], levels, dates, origins, validation_end, scales, sarima
            )
            store(model, result, [column])
    begin_phase("Tune the joint models on both series")
    for model in JOINT_MODELS:
        result = tune_model(model, columns, levels, dates, origins, validation_end, scales)
        store(model, result, columns)
    begin_phase("Score naive, choose the champions, write the outputs")
    naive = origin_paths(NAIVE, {}, levels, dates, origins, validation_end, columns)
    for column in columns:
        metrics[column][NAIVE] = horizon_metrics(
            naive[column], origins, levels[:, column], scales[column]
        )
    # The parallel fits are done: stop joblib's workers now, releasing their memory.
    get_reusable_executor().shutdown(wait=True)
    champions = choose_champions(
        {c: {m: v for m, v in metrics[c].items() if m != NAIVE} for c in columns}
    )
    table = metrics_rows(metrics, champions)
    TUNED_PATH.write_text(json.dumps(tuned, indent=2) + "\n", encoding="utf-8")
    champion_record = {
        VALUE_COLUMNS[column]: {
            family: {
                "model": model,
                "settings": tuned[VALUE_COLUMNS[column]][model],
                "validation_mean_MASE": mean_mase(metrics[column][model]),
                "validation_MASE": {
                    str(h): values["MASE"] for h, values in metrics[column][model].items()
                },
            }
            for family, model in by_family.items()
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
                **{f"{VALUE_COLUMNS[c]}_mean_MASE": s for c, s in trial["per_series"].items()},
            }
            for model, rows in trials.items()
            for number, trial in enumerate(rows)
        ]
    ).to_csv(SCORES_PATH, index=False)
    report = tuning_report(table, tuned, origins, data[DATE_COLUMN], validation_end)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    figure = tuning_figure(trials, table)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"Champions written to {CHAMPIONS_PATH}")
    finish_run()


if __name__ == "__main__":
    app()
