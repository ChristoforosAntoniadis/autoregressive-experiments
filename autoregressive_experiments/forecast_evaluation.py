"""
The test of the champions: the last part of the textbook three-way split (Hastie, Tibshirani and
Friedman 2009, section 7.2). Each series' champion per family (econometric, machine learning,
neural network), chosen on validation by tuning, is run on the test split exactly once, with
a benchmark as the reference; nothing is tuned or chosen here.

Rolling origin with an expanding window (Tashman 2000): at every test period a champion is fitted
on every period from the start of the data up to it, with its tuned settings, and forecasts up to
MAX_HORIZON ahead; forecasts are scored at the HORIZONS by MAE, MSE, RMSE, MAPE and MASE (Hyndman
and Koehler 2006), by MAE relative to naive's, by their bias (the mean error and the share of
forecasts above the actual: a model that keeps forecasting a trend that has ended shows up here,
not in MAE), and against naive by the Diebold-Mariano test with the Harvey-Leybourne-Newbold
correction. The one-step errors also get a Ljung-Box test: autocorrelated errors mean the model
leaves structure a better one could use. The config's benchmarks run beside the champions
(naive, drift, the seasonal naive, the moving average; see model_forecasts), and so does the
equal-weight mean of the three champions, the combination that is hard to beat out of sample
(Clemen 1989).

Uncertainty bands: every model's 2.5, 10, 50, 90 and 97.5% quantiles are its forecast plus the
same quantiles of its own past errors at that horizon, the errors of every forecast made from the
end of training whose target month is already known at the origin (an empirical prediction
interval: Williams and Goodman 1971; Lee and Scholtes 2014; split conformal prediction, Vovk et
al. 2005). They need no assumption about the model, so every model gets the same kind of band,
and they widen with the horizon by themselves. They are scored by coverage, width and pinball
loss, as in the M5 uncertainty competition.

Sensitivity, labelled as such and not the result: each model's MAE relative to naive again,
without the forecasts that start or end in a test month whose change the outliers module's
Hampel filter flags. It shows how much a few extreme months decide the comparison.

Input: the cleaned levels (features.LEVELS_PATH), ``models/<dataset>/champions.json`` and
``models/<dataset>/tuned_hyperparameters.json`` (written by ``make tune``).

Output, in ``reports/``:

- ``test_metrics.csv``: per series, model (champions, their mean and the benchmarks) and horizon:
  MAE, MAPE, MASE, MSE, RMSE, MAE relative to naive, the mean error, the share of forecasts
  above the actual, the Diebold-Mariano test against naive and the Ljung-Box p-value of the
  one-step errors
- ``test_intervals.csv``: the bands' coverage, width, pinball loss and below-quantile shares
- ``test_forecasts.csv``: every test forecast with its band
- ``champions.csv``: each champion's family, validation scores and hyperparameters
- ``test_sensitivity.csv``: per series, model and horizon, MAE and MAE relative to naive
  without the forecasts that start or end in a flagged test month
- ``forecast_evaluation.txt``: the method, the champions, the test tables and the sensitivity
  table
- ``figures/test_forecasts_<series>.png``: per series, one panel per champion (and their mean)
  and horizon: the actual levels against the forecasts, with their 80% and 95% bands and naive
"""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from joblib.externals.loky import get_reusable_executor
from loguru import logger
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
import typer

from autoregressive_experiments.config import FIGURES_DIR, REPORTS_DIR
from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.diagnostics import (
    SEASONAL_PERIOD,
    SIGNIFICANCE_LEVEL,
    VALUE_COLUMNS,
    repo_path,
)
from autoregressive_experiments.features import (
    CUMULATIVE,
    DATE_COLUMN,
    DATE_FORMAT,
    LEVELS_PATH,
    PERIOD,
    UNIT,
    UNIT_SUFFIX,
    load_levels,
    series_title,
)
from autoregressive_experiments.metrics import (
    HORIZONS,
    MAX_HORIZON,
    QUANTILES,
    accuracy,
    diebold_mariano,
    empirical_band,
    interval_scores,
    mase_scale,
)
from autoregressive_experiments.model_forecasts import (
    BENCHMARKS,
    FAMILIES,
    MASE_LAG,
    MASE_WINDOW,
    REFERENCE,
    SARIMA,
)
from autoregressive_experiments.outliers import hampel
from autoregressive_experiments.residual_checks import (
    MUTED_INK,
    PRIMARY_INK,
    SECONDARY_INK,
    SERIES_PALETTE,
    SURFACE,
    apply_chart_style,
)
from autoregressive_experiments.runlog import begin_phase, finish_run, start_run_log
from autoregressive_experiments.tuning import (
    CHAMPIONS_PATH,
    TUNED_PATH,
    VALIDATION_STEP,
    origin_paths,
)

app = typer.Typer()

# Settings; each value's reasoning is in the dataset's config file (configs/<dataset>.json).
# Periods between test origins (1: a refit at every test period).
TEST_STEP = HYPERPARAMETERS.forecast_evaluation.test_step

BENCHMARK = "benchmark"
# The equal-weight mean of the three champions' forecasts, tested as a fourth line.
COMBINED, COMBINED_FAMILY = "mean of champions", "combination"
# Quantile column names, e.g. "q2.5", in QUANTILES order.
QUANTILE_COLUMNS = [f"q{100 * level:g}" for level in QUANTILES]
DM_COLUMNS = ("DM_absolute", "DM_absolute_p", "DM_squared", "DM_squared_p")
# MAE relative to the reference benchmark's on the same forecasts, e.g. "MAE_vs_naive".
VS_REFERENCE = f"MAE_vs_{REFERENCE.replace(' ', '_')}"
BIAS_COLUMNS = (VS_REFERENCE, "ME", "share_over", "LB_p")
# Each family's colour in the figures (palette slots 1-4); the reference is drawn in grey.
FAMILY_COLORS = dict(zip((*FAMILIES, COMBINED_FAMILY), SERIES_PALETTE))
# The report's columns for the reference: "/naive" and "DM vs naive", sized to the name.
RATIO_WIDTH = max(7, len(REFERENCE) + 2)
DM_WIDTH = max(18, len(REFERENCE) + 8)

METRICS_PATH = REPORTS_DIR / "test_metrics.csv"
INTERVALS_PATH = REPORTS_DIR / "test_intervals.csv"
FORECASTS_PATH = REPORTS_DIR / "test_forecasts.csv"
CHAMPIONS_TABLE_PATH = REPORTS_DIR / "champions.csv"
REPORT_PATH = REPORTS_DIR / "forecast_evaluation.txt"
SENSITIVITY_PATH = REPORTS_DIR / "test_sensitivity.csv"


def figure_path(column: str) -> Path:
    """
    Where a series' test figure is written.

    Parameters
    ----------
    column : str
        The series, e.g. "fed_total_assets_T".

    Returns
    -------
    Path
        ``reports/<dataset>/figures/test_forecasts_<series>.png``, with the unit suffix dropped.
    """
    return FIGURES_DIR / f"test_forecasts_{column.removesuffix(UNIT_SUFFIX)}.png"


def load_json(path: Path) -> dict[str, Any]:
    """
    A JSON file written by the tuning.

    Parameters
    ----------
    path : Path
        CHAMPIONS_PATH or TUNED_PATH.

    Returns
    -------
    dict of str to Any
        Its contents.

    Raises
    ------
    FileNotFoundError
        If it doesn't exist yet; the message says to run the tuning first.
    """
    if not path.exists():
        raise FileNotFoundError(f"{repo_path(path)} is missing: run make tune first")
    content: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return content


def forecast_rows(
    paths: Sequence[np.ndarray], origins: Sequence[int], actual: np.ndarray, dates: np.ndarray
) -> pd.DataFrame:
    """
    One row per forecast: origin, horizon, target month, actual level, forecast and error.

    Parameters
    ----------
    paths : Sequence of np.ndarray
        One level path per origin, in origin order.
    origins : Sequence of int
        The origins (positions).
    actual : np.ndarray
        The series' levels over the whole sample, shape (months,), trillions.
    dates : np.ndarray
        Their month-end dates.

    Returns
    -------
    pd.DataFrame
        Columns "origin_index", "target_index", "origin", "target" (dates), "horizon",
        "origin_level" (the level at the origin, for the figure of a cumulative series),
        "actual", "forecast" and "error" (actual - forecast), trillions.
    """
    rows = [
        {
            "origin_index": origin,
            "target_index": origin + step,
            "origin": dates[origin],
            "target": dates[origin + step],
            "horizon": step,
            "origin_level": actual[origin],
            "actual": actual[origin + step],
            "forecast": path[step - 1],
        }
        for origin, path in zip(origins, paths)
        for step in range(1, len(path) + 1)
    ]
    table = pd.DataFrame(rows)
    return table.assign(error=table["actual"] - table["forecast"])


def add_bands(rows: pd.DataFrame, first_origin: int) -> pd.DataFrame:
    """
    The forecasts from `first_origin` on, each with its empirical band.

    Parameters
    ----------
    rows : pd.DataFrame
        One model's forecasts for one series from every origin since the end of training
        (forecast_rows()).
    first_origin : int
        The first origin to keep (the last validation month, so every target is a test month).

    Returns
    -------
    pd.DataFrame
        The rows with origin_index >= first_origin, plus one column per QUANTILE_COLUMNS: the
        forecast plus that quantile of the model's errors at the same horizon whose target was
        known at the origin (target_index <= origin_index).

    Notes
    -----
    Only errors already observed at the origin enter its band, so the band never uses the
    month it forecasts or any later one; it grows as the test months' own errors arrive.
    """
    kept = rows[rows["origin_index"] >= first_origin].copy()
    offsets = np.vstack(
        [
            empirical_band(
                rows.loc[
                    (rows["horizon"] == row.horizon) & (rows["target_index"] <= row.origin_index),
                    "error",
                ].to_numpy()
            )
            for row in kept.itertuples()
        ]
    )
    for name, column in zip(QUANTILE_COLUMNS, offsets.T):
        kept[name] = kept["forecast"].to_numpy() + column
    return kept


def score_model(test: pd.DataFrame, reference: pd.DataFrame, scale: float) -> list[dict[str, Any]]:
    """
    Point metrics, the Diebold-Mariano test against the reference benchmark, the Ljung-Box test
    of the one-step errors and the band scores, per horizon.

    Parameters
    ----------
    test : pd.DataFrame
        One model's test forecasts with bands (add_bands()).
    reference : pd.DataFrame
        The reference benchmark's (REFERENCE), for the same series.
    scale : float
        The series' MASE denominator.

    Returns
    -------
    list of dict of str to Any
        One record per horizon in HORIZONS: "horizon", "n", the METRICS, VS_REFERENCE (MAE /
        the reference's MAE on the same forecasts), the DM_COLUMNS (NaN for the reference
        itself), "LB_p" (the
        Ljung-Box p-value of the one-step errors at SEASONAL_PERIOD lags; NaN at the other
        horizons, whose errors overlap by construction) and interval_scores' columns.

    Notes
    -----
    The Diebold-Mariano pairs are matched on origin, so both losses in a pair are for the same
    target month; absolute loss matches MAE, squared loss MSE. One-step errors of an adequate
    model are white noise, so a small Ljung-Box p-value says the model misses structure that its
    own past errors could predict (the forecaster's usual check; Diebold 2007).
    """
    records = []
    for horizon in HORIZONS:
        rows = test[test["horizon"] == horizon].set_index("origin_index")
        against = reference[reference["horizon"] == horizon].set_index("origin_index")
        against = against.loc[rows.index]
        record: dict[str, Any] = {"horizon": horizon, "n": len(rows)}
        record |= accuracy(rows["actual"].to_numpy(), rows["forecast"].to_numpy(), scale)
        record[VS_REFERENCE] = record["MAE"] / float(np.mean(np.abs(against["error"])))
        # Consecutive one-step errors only: with spaced origins the errors are TEST_STEP periods
        # apart and the test would not be of the structure the column claims.
        record["LB_p"] = (
            float(acorr_ljungbox(rows["error"], lags=[SEASONAL_PERIOD])["lb_pvalue"].iloc[0])
            if horizon == 1 and TEST_STEP == 1
            else np.nan
        )
        for loss, function in (("absolute", np.abs), ("squared", np.square)):
            if rows["forecast"].equals(against["forecast"]):
                statistic, p_value = np.nan, np.nan
            else:
                statistic, p_value = diebold_mariano(
                    function(rows["error"].to_numpy()),
                    function(against["error"].to_numpy()),
                    overlapping_origins(horizon),
                )
            record |= {f"DM_{loss}": statistic, f"DM_{loss}_p": p_value}
        record |= interval_scores(rows["actual"].to_numpy(), rows[QUANTILE_COLUMNS].to_numpy())
        records.append(record)
    return records


def flagged_months(levels: np.ndarray) -> np.ndarray:
    """
    Which months' changes the Hampel filter flags, over a series' whole history.

    Parameters
    ----------
    levels : np.ndarray
        One series' levels, shape (months,), trillions.

    Returns
    -------
    np.ndarray
        Shape (months,), bool: True for a month whose change from the month before is flagged;
        the first month has no change and is never flagged.
    """
    return np.concatenate([[False], hampel(np.diff(levels))[2]])


def sensitivity_rows(tests: Mapping[str, pd.DataFrame], flags: np.ndarray) -> list[dict[str, Any]]:
    """
    Each model's MAE per horizon without forecasts that start or end in a flagged month.

    Parameters
    ----------
    tests : Mapping of str to pd.DataFrame
        Model -> its test forecasts (add_bands(), with origin_index, target_index, horizon and
        error), naive included.
    flags : np.ndarray
        From flagged_months(), indexed like origin_index and target_index.

    Returns
    -------
    list of dict of str to Any
        Per model and horizon: "model", "horizon", "n" (forecasts kept), "MAE" (in the series'
        unit) and VS_REFERENCE (below 1: more accurate than the reference on the same forecasts).
    """
    kept = {
        model: rows[~flags[rows["origin_index"]] & ~flags[rows["target_index"]]]
        for model, rows in tests.items()
    }
    against = kept[REFERENCE].groupby("horizon")["error"].apply(lambda e: e.abs().mean())
    records: list[dict[str, Any]] = []
    for model, rows in kept.items():
        mae = rows.groupby("horizon")["error"].apply(lambda e: e.abs().mean())
        counts = rows.groupby("horizon").size()
        records.extend(
            {
                "model": model,
                "horizon": horizon,
                "n": int(counts.get(horizon, 0)),
                "MAE": float(mae.get(horizon, np.nan)),
                VS_REFERENCE: float(mae.get(horizon, np.nan) / against.get(horizon, np.nan)),
            }
            for horizon in HORIZONS
        )
    return records


def sensitivity_text(table: pd.DataFrame, flagged: Mapping[str, Sequence[str]]) -> str:
    """
    The sensitivity section of the report.

    Parameters
    ----------
    table : pd.DataFrame
        From sensitivity_rows(), with a "series" and a "family" column added.
    flagged : Mapping of str to Sequence of str
        Series -> its flagged test periods, as dates in DATE_FORMAT.

    Returns
    -------
    str
        Per series, the flagged periods, then MAE relative to the reference per model and horizon.
    """
    lines = [
        (
            f"Sensitivity (not the result): MAE relative to {REFERENCE} without the forecasts "
            f"that start or end in a test {PERIOD} whose change the Hampel filter flags (outliers "
            f"module). Below 1: more accurate than {REFERENCE} on the same forecasts."
        ),
        "",
    ]
    for column, part in table.groupby("series", sort=False):
        periods = ", ".join(flagged[str(column)]) or "none"
        lines.append(f"{column}  (flagged test {PERIOD}s: {periods})")
        lines.append(
            f"    {'model':24s} {'family':18s}" + "".join(f"{f'h={h}':>8s}" for h in HORIZONS)
        )
        for (model, family), rows in part.groupby(["model", "family"], sort=False):
            ratios = rows.set_index("horizon")[VS_REFERENCE]
            lines.append(
                f"    {model:24s} {family:18s}"
                + "".join(f"{ratios.get(h, np.nan):8.2f}" for h in HORIZONS)
            )
        lines.append("")
    return "\n".join(lines)


def overlapping_origins(horizon: int) -> int:
    """
    How many consecutive test origins forecast a common target at a horizon, this one included.

    Parameters
    ----------
    horizon : int
        Forecast horizon, in periods.

    Returns
    -------
    int
        1 + floor((horizon - 1) / TEST_STEP): an origin's targets run to origin + horizon, and
        the origin k steps later starts forecasting at origin + k x TEST_STEP + 1, so they share
        a target while k x TEST_STEP < horizon. The Diebold-Mariano test takes this as its
        horizon argument (one autocovariance lag per overlapping neighbour); with TEST_STEP 1
        it is the horizon itself.
    """
    return (horizon - 1) // TEST_STEP + 1


def test_origins(train_end: int, validation_end: int, last: int) -> list[int]:
    """
    The forecast origins: every TEST_STEP periods, anchored so the last validation period is one.

    Parameters
    ----------
    train_end : int
        Position of the last training period.
    validation_end : int
        Position of the last validation period, the first origin whose targets are all test
        periods.
    last : int
        Position of the last period.

    Returns
    -------
    list of int
        Positions from the first origin at or after `train_end` on the grid through
        `validation_end`, up to but excluding `last`. The origins before `validation_end`
        supply the first bands' error history (add_bands); from `validation_end` on they are
        scored.

    Notes
    -----
    Anchoring on `validation_end` rather than `train_end` keeps the scored origins the same
    whatever the validation split's length: a grid from `train_end` lands on `validation_end`
    only when that length is a multiple of TEST_STEP, otherwise the first test periods are
    never forecast and the number of scored origins drifts with the split.
    """
    return list(range(train_end + (validation_end - train_end) % TEST_STEP, last, TEST_STEP))


def listed(names: Sequence[str]) -> str:
    """
    Names as English prose: "naive and drift", "naive, drift and seasonal naive".

    Parameters
    ----------
    names : Sequence of str
        At least one name.

    Returns
    -------
    str
        The names joined by commas, the last with "and".
    """
    return " and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]


def dm_text(statistic: float, p_value: float) -> str:
    """
    Short report text for one Diebold-Mariano result against the reference benchmark.

    Parameters
    ----------
    statistic : float
        DM statistic (negative: the model beats the reference).
    p_value : float
        Its two-sided p-value; NaN for the reference itself.

    Returns
    -------
    str
        "-" without a test, "p=0.xx" when not significant at SIGNIFICANCE_LEVEL, otherwise
        "better p=0.xxx" or "worse p=0.xxx".
    """
    if not np.isfinite(p_value):
        return "-"
    if p_value >= SIGNIFICANCE_LEVEL:
        return f"p={p_value:.2f}"
    return f"{'better' if statistic < 0 else 'worse'} p={p_value:.3f}"


def evaluation_report(
    champions: Mapping[str, Mapping[str, Any]],
    metrics: pd.DataFrame,
    periods: Mapping[str, str],
    scales: Mapping[str, float],
) -> str:
    """
    The text report: the method, each series' champions with their settings, and the test
    tables.

    Parameters
    ----------
    champions : Mapping of str to Mapping of str to Any
        Series -> family -> {"model", "settings", "validation_score", "validation_score_se",
        "tied_with", ...}.
    metrics : pd.DataFrame
        score_model()'s records with "series", "family" and "model".
    periods : Mapping of str to str
        "data", "validation" and "test" -> their months, as text.
    scales : Mapping of str to float
        Series -> MASE denominator, trillions.

    Returns
    -------
    str
        The report text.
    """
    tests = int(metrics[DM_COLUMNS[1]].notna().sum() + metrics[DM_COLUMNS[3]].notna().sum())
    origins_text = (
        f"at every test {PERIOD}"
        if TEST_STEP == 1
        else f"at every {TEST_STEP}th test {PERIOD} (an origin every {TEST_STEP} {PERIOD}s)"
    )
    scale_text = (
        f"change minus the mean of the {MASE_WINDOW} changes before it (the {REFERENCE}'s "
        "one-step error)"
        if MASE_WINDOW
        else f"{PERIOD}ly change"
        if MASE_LAG == 1
        else f"change over {MASE_LAG} {PERIOD}s (the {REFERENCE}'s one-step error)"
    )
    lines = [
        "Test of the champions (one per family and series, chosen on validation)",
        (
            f"Three-way split: training fits, validation ({periods['validation']}; an origin "
            f"every {VALIDATION_STEP} {PERIOD}s) tunes and chooses, test ({periods['test']}) "
            f"measures. This step runs each champion, the mean of the three champions, "
            f"{listed(BENCHMARKS)} over the test {PERIOD}s once; nothing is tuned or chosen after "
            f"it. Rolling origin, expanding window: {origins_text} each model is refitted on all "
            f"{PERIOD}s from {periods['data']} to it, with its tuned settings, and forecasts up "
            f"to {MAX_HORIZON} {PERIOD}s ahead."
        ),
        (
            f"Errors are on the levels: MAE, RMSE in {UNIT}; MSE in {UNIT} squared; MAPE "
            f"in %; MASE = MAE / the training split's mean absolute {scale_text} (scales: "
            + ", ".join(f"{column} {scale:.4f}" for column, scale in scales.items())
            + f"), so {REFERENCE}'s own MASE is above 1 when the test moves more than training "
            f"did; MAE/{REFERENCE} compares like with like. ME: mean of actual - forecast "
            "(negative: the model forecasts too high); over: share of forecasts above the actual "
            f"(0.50 when unbiased). DM: Diebold-Mariano test against {REFERENCE}, absolute-error "
            "loss, with the "
            "Harvey-Leybourne-Newbold correction; 'better' / 'worse' when significant at "
            f"{SIGNIFICANCE_LEVEL:.0%}. {tests} such tests are reported (absolute and squared "
            f"loss); about {tests * SIGNIFICANCE_LEVEL:.0f} would come out significant by chance "
            "alone, so one borderline p-value is weak evidence. LB p: Ljung-Box p-value of the "
            f"1-{PERIOD} errors at {SEASONAL_PERIOD} lags; below {SIGNIFICANCE_LEVEL:.0%}, the "
            "errors are autocorrelated and the model leaves structure unused"
            + (
                "."
                if TEST_STEP == 1
                else f" (not applicable here: the origins are {TEST_STEP} {PERIOD}s apart)."
            )
        ),
        (
            "Bands: each forecast plus the 2.5 / 10 / 50 / 90 / 97.5% quantiles of the same "
            "model's errors at that horizon from every earlier forecast since the end of "
            "training whose target was already known (empirical prediction intervals, "
            "split-conformal style); coverage targets 0.80 and 0.95."
        ),
        "",
    ]
    for column in VALUE_COLUMNS:
        lines.append(f"{column}")
        lines.append(
            f"  Champions (validation score: MAE / {REFERENCE}'s, mean over horizons; its "
            "standard error; models tied within one standard error) and their settings:"
        )
        for family, record in champions[column].items():
            settings = ", ".join(f"{k}={v}" for k, v in record["settings"].items())
            tied = ", ".join(record["tied_with"]) or "none"
            lines.append(
                f"    {family:<18}{record['model']:<26}{record['validation_score']:6.3f} "
                f"(se {record['validation_score_se']:.3f}; tied with: {tied})   {settings}"
            )
        table = metrics[metrics["series"] == column]
        for horizon in HORIZONS:
            rows = table[table["horizon"] == horizon].sort_values("MAE")
            lines.append(f"  {horizon}-{PERIOD} horizon ({int(rows['n'].iloc[0])} forecasts)")
            lines.append(
                f"    {'model':<26}{'family':<18}{'MAE':>8}{'/' + REFERENCE:>{RATIO_WIDTH}}"
                f"{'MAPE %':>8}{'MASE':>8}"
                f"{'RMSE':>8}{'ME':>8}{'over':>6}   {'DM vs ' + REFERENCE:<{DM_WIDTH}}{'LB p':>6}"
                f"{'cov 80%':>8}{'cov 95%':>8}{'pinball':>9}"
            )
            for row in rows.itertuples():
                lb = f"{row.LB_p:6.2f}" if np.isfinite(row.LB_p) else f"{'-':>6}"
                lines.append(
                    f"    {row.model:<26}{row.family:<18}{row.MAE:8.4f}"
                    f"{getattr(row, VS_REFERENCE):{RATIO_WIDTH}.2f}"
                    f"{row.MAPE:8.2f}{row.MASE:8.3f}{row.RMSE:8.4f}{row.ME:8.3f}"
                    f"{row.share_over:6.2f}   "
                    f"{dm_text(row.DM_absolute, row.DM_absolute_p):<{DM_WIDTH}}"
                    f"{lb}{row.coverage_80:8.2f}{row.coverage_95:8.2f}{row.pinball:9.4f}"
                )
        lines.append("")
    lines.append(
        f"All numbers: {repo_path(METRICS_PATH)}, {repo_path(INTERVALS_PATH)}, "
        f"{repo_path(FORECASTS_PATH)}; champions and settings: {repo_path(CHAMPIONS_TABLE_PATH)}"
    )
    lines.append(
        "Figures: " + ", ".join(repo_path(figure_path(column)) for column in VALUE_COLUMNS)
    )
    return "\n".join(lines)


def shown_values(rows: pd.DataFrame) -> pd.DataFrame:
    """
    The values the test figure plots: the levels, or the flows since the origin.

    Parameters
    ----------
    rows : pd.DataFrame
        Forecasts with "origin_level", "actual", "forecast" and the QUANTILE_COLUMNS.

    Returns
    -------
    pd.DataFrame
        The rows unchanged, or, when CUMULATIVE, with the origin level subtracted from the
        actual, the forecast and every quantile: the forecast's flow over its horizon.

    Notes
    -----
    A running total climbs by far more over the test split than any forecast misses by, so on
    the level axis the actual, the forecast and the bands coincide; the flow since the origin
    is on the scale of the errors.
    """
    if not CUMULATIVE:
        return rows
    columns = ["actual", "forecast", *QUANTILE_COLUMNS]
    return rows.assign(**{c: rows[c] - rows["origin_level"] for c in columns})


def test_figure(forecasts: pd.DataFrame, metrics: pd.DataFrame, column: str) -> Figure:
    """
    One series' test: the actual levels against each champion's forecasts, per horizon.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Every test forecast with its band ("series", "family", "model", "target", "horizon",
        "origin_level", "actual", "forecast" and the QUANTILE_COLUMNS).
    metrics : pd.DataFrame
        The test metrics, for the panel titles.
    column : str
        The series.

    Returns
    -------
    Figure
        One row per family's champion and one for their mean, one column per horizon. Each
        panel: the actual levels (ink), the model's forecasts placed on the period they forecast
        (its family's colour) inside their 80% (darker) and 95% (lighter) bands, and naive's
        forecasts (dashed grey); the title gives both MASE values. A cumulative series shows
        every value as the flow since the forecast's origin (shown_values()).
    """
    apply_chart_style()
    series = forecasts[forecasts["series"] == column]
    families = [*FAMILIES, COMBINED_FAMILY]
    figure, axes = plt.subplots(
        len(families),
        len(HORIZONS),
        figsize=(4.2 * len(HORIZONS), 3.3 * len(families)),
        squeeze=False,
        sharex=True,
        facecolor=SURFACE,
    )
    reference_mase = metrics[
        (metrics["series"] == column) & (metrics["model"] == REFERENCE)
    ].set_index("horizon")["MASE"]
    for row, family in enumerate(families):
        rows = series[series["family"] == family]
        model = rows["model"].iloc[0]
        color = FAMILY_COLORS[family]
        scores = metrics[(metrics["series"] == column) & (metrics["model"] == model)].set_index(
            "horizon"
        )["MASE"]
        for col, horizon in enumerate(HORIZONS):
            axis = axes[row][col]
            axis.set_facecolor(SURFACE)
            at = shown_values(rows[rows["horizon"] == horizon].sort_values("target"))
            reference = series[(series["model"] == REFERENCE) & (series["horizon"] == horizon)]
            reference = shown_values(reference.sort_values("target"))
            axis.fill_between(at["target"], at["q2.5"], at["q97.5"], color=color, alpha=0.12, lw=0)
            axis.fill_between(at["target"], at["q10"], at["q90"], color=color, alpha=0.28, lw=0)
            axis.plot(at["target"], at["actual"], color=PRIMARY_INK, linewidth=1.6)
            axis.plot(
                reference["target"],
                reference["forecast"],
                color=MUTED_INK,
                linewidth=1.1,
                linestyle="--",
            )
            axis.plot(at["target"], at["forecast"], color=color, linewidth=1.6)
            axis.set_title(
                f"{model}, {horizon} {PERIOD}{'s' if horizon > 1 else ''} ahead\n"
                f"MASE {scores[horizon]:.2f} ({REFERENCE} {reference_mase[horizon]:.2f})",
                loc="left",
                fontsize=9,
            )
            # Ticks chosen from the span shown, so the axis reads for any spacing of the data.
            locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            if col == 0:
                quantity = f"flow since the origin, {UNIT}" if CUMULATIVE else UNIT
                axis.set_ylabel(f"{family}\n{quantity}", color=SECONDARY_INK)
    handles = [
        plt.Line2D([], [], color=PRIMARY_INK, linewidth=1.6, label="actual"),
        *(
            plt.Line2D([], [], color=FAMILY_COLORS[f], linewidth=1.6, label=f"{f} champion")
            for f in families
        ),
        plt.Line2D([], [], color=MUTED_INK, linewidth=1.1, linestyle="--", label=REFERENCE),
        plt.Rectangle((0, 0), 1, 1, color=MUTED_INK, alpha=0.28, label="80% band"),
        plt.Rectangle((0, 0), 1, 1, color=MUTED_INK, alpha=0.12, label="95% band"),
    ]
    shown = "flow since each forecast's origin" if CUMULATIVE else "actual"
    figure.suptitle(
        f"{series_title(column)}, test: {shown} vs each family champion's forecasts, by horizon "
        f"(each forecast shown on the {PERIOD} it forecasts)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    figure.legend(
        handles=handles, loc="lower center", ncols=len(handles), fontsize=9, frameon=False
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    return figure


@app.command()
def main(levels_path: Path = LEVELS_PATH) -> None:
    """
    Run each champion and naive over the test months once, and write the metrics, bands,
    forecasts, champions table, report and figures.

    Parameters
    ----------
    levels_path : Path, default=LEVELS_PATH
        Cleaned monthly levels of both series, trillions.

    Returns
    -------
    None
        Writes METRICS_PATH, INTERVALS_PATH, FORECASTS_PATH, CHAMPIONS_TABLE_PATH, REPORT_PATH
        and one figure per series, and prints the report.

    Raises
    ------
    FileNotFoundError
        If the champions or tuned settings are missing (run make tune first).
    ValueError
        If the champions file predates the tie reporting (run make tune first).

    Notes
    -----
    Processing, per series and model (the three champions, their mean and the benchmarks):
    1. Forecast from every TEST_STEPth period from the end of training on, with the tuned
       settings; the
       validation-period forecasts only supply past errors for the bands. The mean of the
       champions is averaged from their forecasts, so it needs no fit of its own.
    2. Keep the forecasts from the last validation period on (every target a test period) and
       give each its empirical band from the errors already known at its origin.
    3. Score each horizon: the metrics, the bias, Diebold-Mariano against naive, Ljung-Box of
       the one-step errors, and the bands.
    """
    start_run_log("evaluate")
    begin_phase("Read the champions and the data")
    champions = load_json(CHAMPIONS_PATH)
    if any("tied_with" not in r for by_family in champions.values() for r in by_family.values()):
        raise ValueError(
            f"{repo_path(CHAMPIONS_PATH)} is from an older tuning: run make tune first"
        )
    tuned = load_json(TUNED_PATH)
    data, splits = load_levels(levels_path)
    train_end = len(splits["train"]) - 1
    validation_end = train_end + len(splits["validation"])
    last = len(data) - 1
    levels = data[list(VALUE_COLUMNS)].to_numpy(dtype=float)
    dates = data[DATE_COLUMN].to_numpy()
    origins = test_origins(train_end, validation_end, last)
    scales = {
        column: mase_scale(splits["train"][column].to_numpy(dtype=float), MASE_LAG, MASE_WINDOW)
        for column in VALUE_COLUMNS
    }
    sarima = [tuned[column][SARIMA] for column in VALUE_COLUMNS]
    periods = {
        "data": f"{pd.Timestamp(dates[0]):{DATE_FORMAT}}",
        "validation": f"{pd.Timestamp(dates[train_end + 1]):{DATE_FORMAT}} to "
        f"{pd.Timestamp(dates[validation_end]):{DATE_FORMAT}}",
        "test": f"{pd.Timestamp(dates[validation_end + 1]):{DATE_FORMAT}} to "
        f"{pd.Timestamp(dates[last]):{DATE_FORMAT}}",
    }

    begin_phase(f"Run each champion and benchmark over the test {PERIOD}s")
    all_forecasts, all_metrics = [], []
    all_sensitivity: list[dict[str, Any]] = []
    flagged: dict[str, list[str]] = {}
    for index, column in enumerate(VALUE_COLUMNS):
        entries = [(f, r["model"], r["settings"]) for f, r in champions[column].items()]
        entries.extend((BENCHMARK, model, {}) for model in BENCHMARKS)
        rows_by_model: dict[str, pd.DataFrame] = {}
        for family, model, settings in entries:
            logger.info(f"TEST | {series_title(column)} | {family} | {model}")
            try:
                paths = origin_paths(
                    model, settings, levels, dates, origins, last, [index], sarima
                )
            except Exception as error:  # noqa: BLE001  hours of other results must survive
                # A fit can fail at a test origin although every validation origin fitted; the
                # model then has no test forecast (NaN metrics), which the report shows.
                logger.warning(
                    f"{model} could not be fitted on test and gets no forecast: {error!r}"
                )
                paths = {index: [np.full(min(MAX_HORIZON, last - o), np.nan) for o in origins]}
            rows_by_model[model] = forecast_rows(paths[index], origins, levels[:, index], dates)
        # The champions' forecasts share origins and horizons in the same row order, so their
        # mean is a row-wise average of the forecast column.
        champion_rows = [rows_by_model[record["model"]] for record in champions[column].values()]
        combined = champion_rows[0].assign(
            forecast=np.mean([rows["forecast"].to_numpy() for rows in champion_rows], axis=0)
        )
        rows_by_model[COMBINED] = combined.assign(error=combined["actual"] - combined["forecast"])
        entries.insert(len(champions[column]), (COMBINED_FAMILY, COMBINED, {}))
        tests = {
            model: add_bands(rows_by_model[model], validation_end).assign(
                series=column, family=family, model=model
            )
            for family, model, _ in entries
        }
        for family, model, _ in entries:
            for record in score_model(tests[model], tests[REFERENCE], scales[column]):
                all_metrics.append({"series": column, "family": family, "model": model, **record})
            all_forecasts.append(tests[model])
        flags = flagged_months(levels[:, index])
        flagged[column] = [
            f"{pd.Timestamp(dates[i]):{DATE_FORMAT}}"
            for i in np.flatnonzero(flags[validation_end + 1 :]) + validation_end + 1
        ]
        families = {model: family for family, model, _ in entries}
        all_sensitivity.extend(
            {"series": column, "family": families[record["model"]], **record}
            for record in sensitivity_rows(tests, flags)
        )
    # The parallel fits are done: stop joblib's workers now, releasing their memory.
    get_reusable_executor().shutdown(wait=True)

    begin_phase("Write the tables, report and figures")
    forecasts = pd.concat(all_forecasts, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)
    keys = ["series", "family", "model", "horizon", "n"]
    metrics[[*keys, "MAE", "MAPE", "MASE", "MSE", "RMSE", *BIAS_COLUMNS, *DM_COLUMNS]].to_csv(
        METRICS_PATH, index=False
    )
    interval_columns = [c for c in metrics.columns if c.startswith(("coverage", "width", "below"))]
    metrics[[*keys, *interval_columns, "pinball"]].to_csv(INTERVALS_PATH, index=False)
    forecasts[
        [
            "series",
            "family",
            "model",
            "origin",
            "target",
            "horizon",
            "actual",
            "forecast",
            *QUANTILE_COLUMNS,
        ]
    ].to_csv(FORECASTS_PATH, index=False)
    pd.DataFrame(
        [
            {
                "series": column,
                "family": family,
                "model": record["model"],
                "validation_score": record["validation_score"],
                "validation_score_se": record["validation_score_se"],
                "tied_with": ", ".join(record["tied_with"]),
                **{f"validation_MASE_h{h}": v for h, v in record["validation_MASE"].items()},
                "settings": json.dumps(record["settings"]),
            }
            for column, by_family in champions.items()
            for family, record in by_family.items()
        ]
    ).to_csv(CHAMPIONS_TABLE_PATH, index=False)
    sensitivity = pd.DataFrame(all_sensitivity)
    sensitivity.to_csv(SENSITIVITY_PATH, index=False)
    report = (
        evaluation_report(champions, metrics, periods, scales)
        + "\n\n"
        + sensitivity_text(sensitivity, flagged)
    )
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    for column in VALUE_COLUMNS:
        figure = test_figure(forecasts, metrics, column)
        figure.savefig(figure_path(column), dpi=130, facecolor=SURFACE)
        plt.close(figure)
    logger.success(f"Test results written to {REPORT_PATH} and {FIGURES_DIR}")
    finish_run()


if __name__ == "__main__":
    app()
