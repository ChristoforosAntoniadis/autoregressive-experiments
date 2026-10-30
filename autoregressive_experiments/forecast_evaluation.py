"""
The test of the champions: the last part of the textbook three-way split (Hastie, Tibshirani and
Friedman 2009, section 7.2). Each series' champion per family (econometric, machine learning,
neural network), chosen on validation by tuning, is run on the test split exactly once, with
naive as the reference; nothing is tuned or chosen here.

Rolling origin with an expanding window (Tashman 2000): at every test month a champion is fitted
on every month from the start of the data up to it, with its tuned settings, and forecasts up to
12 months ahead; forecasts are scored at horizons 1, 3, 6, 9 and 12 months by MAE, MSE, RMSE,
MAPE and MASE (Hyndman and Koehler 2006), and against naive by the Diebold-Mariano test with the
Harvey-Leybourne-Newbold correction.

Uncertainty bands: every model's 2.5, 10, 50, 90 and 97.5% quantiles are its forecast plus the
same quantiles of its own past errors at that horizon, the errors of every forecast made from the
end of training whose target month is already known at the origin (an empirical prediction
interval: Williams and Goodman 1971; Lee and Scholtes 2014; split conformal prediction, Vovk et
al. 2005). They need no assumption about the model, so every model gets the same kind of band,
and they widen with the horizon by themselves. They are scored by coverage, width and pinball
loss, as in the M5 uncertainty competition.

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` and ``models/champions.json`` and
``models/tuned_hyperparameters.json`` (written by ``make tune``).

Output, in ``reports/``:

- ``test_metrics.csv``: per series, champion (and naive) and horizon: MAE, MAPE, MASE, MSE, RMSE
  and the Diebold-Mariano test against naive
- ``test_intervals.csv``: the bands' coverage, width, pinball loss and below-quantile shares
- ``test_forecasts.csv``: every test forecast with its band
- ``champions.csv``: each champion's family, validation scores and hyperparameters
- ``forecast_evaluation.txt``: the method, the champions and the test tables
- ``figures/test_forecasts_<series>.png``: per series, one panel per champion and horizon: the
  actual levels against the forecasts, with their 80% and 95% bands and naive
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
import typer

from autoregressive_experiments.config import FIGURES_DIR, INTERIM_DATA_DIR, REPORTS_DIR
from autoregressive_experiments.diagnostics import SIGNIFICANCE_LEVEL, VALUE_COLUMNS, repo_path
from autoregressive_experiments.features import (
    DATE_COLUMN,
    series_title,
    split_boundaries,
    split_by_time,
)
from autoregressive_experiments.metrics import (
    HORIZONS,
    QUANTILES,
    accuracy,
    diebold_mariano,
    empirical_band,
    interval_scores,
    mase_scale,
)
from autoregressive_experiments.model_forecasts import FAMILIES, NAIVE, SARIMA
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

BENCHMARK = "benchmark"
# Quantile column names, e.g. "q2.5", in QUANTILES order.
QUANTILE_COLUMNS = [f"q{100 * level:g}" for level in QUANTILES]
DM_COLUMNS = ("DM_absolute", "DM_absolute_p", "DM_squared", "DM_squared_p")
# Each family's colour in the figures (palette slots 1-3); naive is drawn in grey.
FAMILY_COLORS = dict(zip(FAMILIES, SERIES_PALETTE))

METRICS_PATH = REPORTS_DIR / "test_metrics.csv"
INTERVALS_PATH = REPORTS_DIR / "test_intervals.csv"
FORECASTS_PATH = REPORTS_DIR / "test_forecasts.csv"
CHAMPIONS_TABLE_PATH = REPORTS_DIR / "champions.csv"
REPORT_PATH = REPORTS_DIR / "forecast_evaluation.txt"


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
        ``reports/figures/test_forecasts_<series>.png``, with the "_T" suffix dropped.
    """
    return FIGURES_DIR / f"test_forecasts_{column.removesuffix('_T')}.png"


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
        "actual", "forecast" and "error" (actual - forecast), trillions.
    """
    rows = [
        {
            "origin_index": origin,
            "target_index": origin + step,
            "origin": dates[origin],
            "target": dates[origin + step],
            "horizon": step,
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


def score_model(test: pd.DataFrame, naive: pd.DataFrame, scale: float) -> list[dict[str, Any]]:
    """
    Point metrics, the Diebold-Mariano test against naive and the band scores, per horizon.

    Parameters
    ----------
    test : pd.DataFrame
        One model's test forecasts with bands (add_bands()).
    naive : pd.DataFrame
        Naive's, for the same series.
    scale : float
        The series' MASE denominator.

    Returns
    -------
    list of dict of str to Any
        One record per horizon in HORIZONS: "horizon", "n", the METRICS, the DM_COLUMNS (NaN for
        naive itself) and interval_scores' columns.

    Notes
    -----
    The Diebold-Mariano pairs are matched on origin, so both losses in a pair are for the same
    target month; absolute loss matches MAE, squared loss MSE.
    """
    records = []
    for horizon in HORIZONS:
        rows = test[test["horizon"] == horizon].set_index("origin_index")
        reference = naive[naive["horizon"] == horizon].set_index("origin_index").loc[rows.index]
        record: dict[str, Any] = {"horizon": horizon, "n": len(rows)}
        record |= accuracy(rows["actual"].to_numpy(), rows["forecast"].to_numpy(), scale)
        for loss, function in (("absolute", np.abs), ("squared", np.square)):
            if rows["forecast"].equals(reference["forecast"]):
                statistic, p_value = np.nan, np.nan
            else:
                statistic, p_value = diebold_mariano(
                    function(rows["error"].to_numpy()),
                    function(reference["error"].to_numpy()),
                    horizon,
                )
            record |= {f"DM_{loss}": statistic, f"DM_{loss}_p": p_value}
        record |= interval_scores(rows["actual"].to_numpy(), rows[QUANTILE_COLUMNS].to_numpy())
        records.append(record)
    return records


def dm_text(statistic: float, p_value: float) -> str:
    """
    Short report text for one Diebold-Mariano result against naive.

    Parameters
    ----------
    statistic : float
        DM statistic (negative: the model beats naive).
    p_value : float
        Its two-sided p-value; NaN for naive itself.

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
        Series -> family -> {"model", "settings", "validation_mean_MASE", ...}.
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
    lines = [
        "Test of the champions (one per family and series, chosen on validation)",
        (
            f"Three-way split: training fits, validation ({periods['validation']}; an origin "
            f"every {VALIDATION_STEP} months) tunes and chooses, test ({periods['test']}) "
            "measures. This step runs each champion and naive over the test months once; "
            "nothing is tuned or chosen after it. Rolling origin, expanding window: at every "
            f"test month each model is refitted on all months from {periods['data']} to it, "
            "with its tuned settings, and forecasts up to 12 months ahead."
        ),
        (
            "Errors are on the levels: MAE, RMSE in trillions; MSE in trillions squared; MAPE "
            "in %; MASE = MAE / the training split's mean absolute monthly change (scales: "
            + ", ".join(f"{column} {scale:.4f}" for column, scale in scales.items())
            + "). DM: Diebold-Mariano test against naive, absolute-error loss, with the "
            "Harvey-Leybourne-Newbold correction; 'better' / 'worse' when significant at "
            f"{SIGNIFICANCE_LEVEL:.0%}."
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
            "  Champions (validation MASE averaged over the horizons) and their settings:"
        )
        for family, record in champions[column].items():
            settings = ", ".join(f"{k}={v}" for k, v in record["settings"].items())
            lines.append(
                f"    {family:<18}{record['model']:<26}{record['validation_mean_MASE']:7.3f}   "
                f"{settings}"
            )
        table = metrics[metrics["series"] == column]
        for horizon in HORIZONS:
            rows = table[table["horizon"] == horizon].sort_values("MASE")
            lines.append(f"  {horizon}-month horizon ({int(rows['n'].iloc[0])} forecasts)")
            lines.append(
                f"    {'model':<26}{'family':<18}{'MAE':>8}{'MAPE %':>8}{'MASE':>8}{'MSE':>11}"
                f"{'RMSE':>8}   {'DM vs naive':<18}{'cov 80%':>8}{'cov 95%':>8}{'pinball':>9}"
            )
            for row in rows.itertuples():
                lines.append(
                    f"    {row.model:<26}{row.family:<18}{row.MAE:8.4f}{row.MAPE:8.2f}"
                    f"{row.MASE:8.3f}{row.MSE:11.6f}{row.RMSE:8.4f}   "
                    f"{dm_text(row.DM_absolute, row.DM_absolute_p):<18}{row.coverage_80:8.2f}"
                    f"{row.coverage_95:8.2f}{row.pinball:9.4f}"
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


def test_figure(forecasts: pd.DataFrame, metrics: pd.DataFrame, column: str) -> Figure:
    """
    One series' test: the actual levels against each champion's forecasts, per horizon.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Every test forecast with its band ("series", "family", "model", "target", "horizon",
        "actual", "forecast" and the QUANTILE_COLUMNS).
    metrics : pd.DataFrame
        The test metrics, for the panel titles.
    column : str
        The series.

    Returns
    -------
    Figure
        One row per family's champion and one column per horizon. Each panel: the actual
        levels (ink), the champion's forecasts placed on the month they forecast (its family's
        colour) inside their 80% (darker) and 95% (lighter) bands, and naive's forecasts
        (dashed grey); the title gives both MASE values.
    """
    apply_chart_style()
    series = forecasts[forecasts["series"] == column]
    families = list(FAMILIES)
    figure, axes = plt.subplots(
        len(families),
        len(HORIZONS),
        figsize=(4.2 * len(HORIZONS), 3.3 * len(families)),
        squeeze=False,
        sharex=True,
        facecolor=SURFACE,
    )
    naive_mase = metrics[(metrics["series"] == column) & (metrics["model"] == NAIVE)].set_index(
        "horizon"
    )["MASE"]
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
            at = rows[rows["horizon"] == horizon].sort_values("target")
            reference = series[(series["model"] == NAIVE) & (series["horizon"] == horizon)]
            reference = reference.sort_values("target")
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
                f"{model}, {horizon} month{'s' if horizon > 1 else ''} ahead\n"
                f"MASE {scores[horizon]:.2f} (naive {naive_mase[horizon]:.2f})",
                loc="left",
                fontsize=9,
            )
            axis.xaxis.set_major_locator(mdates.YearLocator(2))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            if col == 0:
                axis.set_ylabel(f"{family}\ntrillions", color=SECONDARY_INK)
    handles = [
        plt.Line2D([], [], color=PRIMARY_INK, linewidth=1.6, label="actual"),
        *(
            plt.Line2D([], [], color=FAMILY_COLORS[f], linewidth=1.6, label=f"{f} champion")
            for f in families
        ),
        plt.Line2D([], [], color=MUTED_INK, linewidth=1.1, linestyle="--", label="naive"),
        plt.Rectangle((0, 0), 1, 1, color=MUTED_INK, alpha=0.28, label="80% band"),
        plt.Rectangle((0, 0), 1, 1, color=MUTED_INK, alpha=0.12, label="95% band"),
    ]
    figure.suptitle(
        f"{series_title(column)}, test: actual vs each family champion's forecasts, by horizon "
        "(each forecast shown on the month it forecasts)",
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
def main(levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv") -> None:
    """
    Run each champion and naive over the test months once, and write the metrics, bands,
    forecasts, champions table, report and figures.

    Parameters
    ----------
    levels_path : Path, default=INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv"
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

    Notes
    -----
    Processing, per series and model (the three champions and naive):
    1. Forecast from every month from the end of training on, with the tuned settings; the
       validation-period forecasts only supply past errors for the bands.
    2. Keep the forecasts from the last validation month on (every target a test month) and
       give each its empirical band from the errors already known at its origin.
    3. Score each horizon: the five metrics, Diebold-Mariano against naive, and the bands.
    """
    start_run_log("evaluate")
    begin_phase("Read the champions and the data")
    champions = load_json(CHAMPIONS_PATH)
    tuned = load_json(TUNED_PATH)
    data = pd.read_csv(levels_path, parse_dates=[DATE_COLUMN])
    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)
    splits = split_by_time(data, split_boundaries(data))
    train_end = len(splits["train"]) - 1
    validation_end = train_end + len(splits["validation"])
    last = len(data) - 1
    levels = data[list(VALUE_COLUMNS)].to_numpy(dtype=float)
    dates = data[DATE_COLUMN].to_numpy()
    origins = list(range(train_end, last))
    scales = {
        column: mase_scale(splits["train"][column].to_numpy(dtype=float))
        for column in VALUE_COLUMNS
    }
    sarima = [tuned[column][SARIMA] for column in VALUE_COLUMNS]
    periods = {
        "data": f"{pd.Timestamp(dates[0]):%Y-%m}",
        "validation": f"{pd.Timestamp(dates[train_end + 1]):%Y-%m} to "
        f"{pd.Timestamp(dates[validation_end]):%Y-%m}",
        "test": f"{pd.Timestamp(dates[validation_end + 1]):%Y-%m} to "
        f"{pd.Timestamp(dates[last]):%Y-%m}",
    }

    begin_phase("Run each champion and naive over the test months")
    all_forecasts, all_metrics = [], []
    for index, column in enumerate(VALUE_COLUMNS):
        entries = [(f, r["model"], r["settings"]) for f, r in champions[column].items()]
        entries.append((BENCHMARK, NAIVE, {}))
        tests = {}
        for family, model, settings in entries:
            logger.info(f"TEST | {series_title(column)} | {family} | {model}")
            paths = origin_paths(model, settings, levels, dates, origins, last, [index], sarima)
            rows = forecast_rows(paths[index], origins, levels[:, index], dates)
            tests[model] = add_bands(rows, validation_end).assign(
                series=column, family=family, model=model
            )
        for family, model, _ in entries:
            for record in score_model(tests[model], tests[NAIVE], scales[column]):
                all_metrics.append({"series": column, "family": family, "model": model, **record})
            all_forecasts.append(tests[model])
    # The parallel fits are done: stop joblib's workers now, releasing their memory.
    get_reusable_executor().shutdown(wait=True)

    begin_phase("Write the tables, report and figures")
    forecasts = pd.concat(all_forecasts, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)
    keys = ["series", "family", "model", "horizon", "n"]
    metrics[[*keys, "MAE", "MAPE", "MASE", "MSE", "RMSE", *DM_COLUMNS]].to_csv(
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
                "validation_mean_MASE": record["validation_mean_MASE"],
                **{f"validation_MASE_h{h}": v for h, v in record["validation_MASE"].items()},
                "settings": json.dumps(record["settings"]),
            }
            for column, by_family in champions.items()
            for family, record in by_family.items()
        ]
    ).to_csv(CHAMPIONS_TABLE_PATH, index=False)
    report = evaluation_report(champions, metrics, periods, scales)
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
