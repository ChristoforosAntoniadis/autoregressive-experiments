"""
Forecast evaluation (step U6): how accurate are the SARIMA models out of sample, do they beat
naive forecasts, and are their forecast intervals honest?

Rolling origin with an expanding window, on validation and then on test: at every month from the
end of the previous split onwards, each model is refitted on all data up to that month and
forecasts up to 12 months ahead; only forecasts whose target falls inside the split are scored.
Each monthly origin is one fold (56 per split). Model orders come from the training split and the
selected model per series from validation (lowest one-month MASE); test is run once, with all
models, and nothing changes after it. Errors are measured on the levels (trillions), so MAPE stays
well defined: the levels are always positive, while the differences cross zero.

Models: the SARIMA models that pass the residual checks (fitted on the first differences, their
forecasts added back onto the last level), and three baselines: naive (no change), drift (the
average monthly change so far) and seasonal naive (each month changes as it did a year earlier).

Point accuracy per split, series, model and horizon: MAE, MSE, RMSE (trillions, trillions
squared), MAPE (%) and MASE (MAE divided by the training split's in-sample MAE of the one-month
naive forecast; below 1 beats it), plus a Diebold-Mariano test against naive (absolute and squared
error losses), so "beats naive" means more than a lucky sample.

Forecast quantiles (2.5, 10, 50, 90 and 97.5%) come from each model's own forecast distribution
on the levels: the SARIMA's error weights, summed over the horizon, with either a constant shock
variance or the forecast variance path of the GARCH variant chosen in step U5b (refitted at every
origin); normal quantiles for constant variance, unit-variance Student-t for GARCH-t. Naive gets
random-walk quantiles as the reference. Scored by 80% and 95% interval coverage and width, the
share of actuals below each quantile, and pinball (quantile) loss. GARCH only changes these
intervals, never the point forecasts. Refits that fail to converge are counted and listed.

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` and, through ``residual_checks``, the
first differences and ``reports/information_criteria_candidates.csv``.

Output, in ``reports/``:

- ``forecast_evaluation.txt``: per split, point-accuracy tables with Diebold-Mariano results,
  interval tables, and the refits that failed to converge
- ``forecast_metrics.csv`` and ``forecast_intervals.csv``: every number behind those tables
- ``figures/forecast_evaluation.png``: per series, each model's MAE relative to naive by horizon
  on validation and on test, and the actual path against the selected model's and the naive
  one-month forecasts
- ``figures/forecast_intervals.png``: per series and split, the selected model's 80% and 95%
  interval coverage by horizon, with constant variance and with GARCH, against the targets
"""

from collections.abc import Callable
from pathlib import Path

from loguru import logger
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import typer

from autoregressive_experiments.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from autoregressive_experiments.diagnostics import (
    CANDIDATES_PATH,
    SEASONAL_PERIOD,
    SIGNIFICANCE_LEVEL,
    VALUE_COLUMNS,
    fit_sarima,
    repo_path,
)
from autoregressive_experiments.features import DATE_COLUMN, split_boundaries, split_by_time
from autoregressive_experiments.residual_checks import (
    MUTED_INK,
    PRIMARY_INK,
    SECONDARY_INK,
    SURFACE,
    apply_chart_style,
    checked_models,
    model_label,
)
from autoregressive_experiments.volatility_models import garch_check, garch_variance_path

app = typer.Typer()

HORIZONS = (1, 3, 6, 12)
MAX_HORIZON = max(HORIZONS)
METRICS = ("MAE", "MSE", "RMSE", "MAPE", "MASE")
DM_COLUMNS = ("DM_absolute", "DM_absolute_p", "DM_squared", "DM_squared_p")
NAIVE = "naive"
BASELINES = (NAIVE, "drift", "seasonal naive")
SPLITS = ("validation", "test")
QUANTILES = (0.025, 0.1, 0.5, 0.9, 0.975)
VARIANCES = ("constant", "garch")

REPORT_PATH = REPORTS_DIR / "forecast_evaluation.txt"
METRICS_PATH = REPORTS_DIR / "forecast_metrics.csv"
INTERVALS_PATH = REPORTS_DIR / "forecast_intervals.csv"
FIGURE_PATH = FIGURES_DIR / "forecast_evaluation.png"
INTERVALS_FIGURE_PATH = FIGURES_DIR / "forecast_intervals.png"

# Reference palette, categorical slots 1-3 for the SARIMA models (validated; slot 3's contrast
# warning is covered by the legend and the report's tables). Baselines are ink and greys.
MODEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
BASELINE_STYLES = {
    NAIVE: (PRIMARY_INK, 2),
    "drift": (MUTED_INK, 1.2),
    "seasonal naive": ("#c3c2b7", 1.2),
}
# Slots 1 and 2 (validated together) for constant variance vs GARCH in the interval figure.
VARIANCE_COLORS = {"constant": MODEL_COLORS[0], "garch": MODEL_COLORS[1]}

# A forecaster takes the levels up to an origin and a step count, and returns "forecast" (the
# level path) plus, optionally, "sd_constant" / "sd_garch" paths, "nu_garch" and "converged".
Forecaster = Callable[[np.ndarray, int], dict]


def naive_forecast(levels: np.ndarray, steps: int) -> dict:
    """No change; random-walk spread: the monthly changes' standard deviation times sqrt(h)."""
    spread = np.std(np.diff(levels), ddof=1) * np.sqrt(np.arange(1, steps + 1))
    return {"forecast": np.full(steps, levels[-1]), "sd_constant": spread}


def drift_forecast(levels: np.ndarray, steps: int) -> dict:
    slope = (levels[-1] - levels[0]) / (len(levels) - 1)
    return {"forecast": levels[-1] + slope * np.arange(1, steps + 1)}


def seasonal_naive_forecast(levels: np.ndarray, steps: int, period: int = SEASONAL_PERIOD) -> dict:
    """Each future month changes by as much as the same month one period earlier."""
    if steps > period:
        raise ValueError(f"Seasonal naive forecasts at most {period} steps here, not {steps}.")
    last_changes = np.diff(levels)[-period:]
    return {"forecast": levels[-1] + np.cumsum(last_changes[:steps])}


def level_variances(psi: np.ndarray, shock_variances: np.ndarray) -> np.ndarray:
    """Forecast-error variance of the level at each horizon h = 1, ..., len(shock_variances).

    The level is the running total of the differences, so its error at h is
    sum over j = 1..h of Psi_(h-j) * shock_(t+j), where Psi_k = psi_0 + ... + psi_k are the
    summed ARMA error weights (psi_0 = 1), and its variance weights each shock's variance."""
    summed = np.cumsum(psi)
    return np.array(
        [
            np.sum(summed[h - 1 - np.arange(h)] ** 2 * shock_variances[:h])
            for h in range(1, len(shock_variances) + 1)
        ]
    )


def sarima_forecaster(model: dict, garch_variant: str) -> Forecaster:
    """Refit the model on the history's first differences; add its forecasts to the last level;
    give the level's spread with a constant shock variance and with the GARCH variant's path."""

    def forecast(levels: np.ndarray, steps: int) -> dict:
        fitted = fit_sarima(
            pd.Series(np.diff(levels)), model["p"], model["q"], model["P"], model["Q"]
        )
        psi = np.asarray(fitted.impulse_responses(steps=steps)).ravel()[:steps]
        residuals = fitted.resid.iloc[fitted.loglikelihood_burn :]
        garch_variances, nu = garch_variance_path(residuals, garch_variant, steps)
        constant = np.full(steps, float(fitted.params["sigma2"]))
        return {
            "forecast": levels[-1] + np.cumsum(np.asarray(fitted.forecast(steps))),
            "sd_constant": np.sqrt(level_variances(psi, constant)),
            "sd_garch": np.sqrt(level_variances(psi, garch_variances)),
            "nu_garch": nu,
            "converged": bool(fitted.mle_retvals.get("converged", False)),
        }

    return forecast


def rolling_forecasts(
    dates: pd.Series,
    levels: np.ndarray,
    first_origin: int,
    last_target: int,
    forecasters: dict[str, Forecaster],
) -> pd.DataFrame:
    """Forecast from every origin in [first_origin, last_target) using only levels up to it,
    keeping targets up to last_target (inclusive) and up to MAX_HORIZON steps ahead."""
    rows = []
    for origin in range(first_origin, last_target):
        history = levels[: origin + 1]
        steps = min(MAX_HORIZON, last_target - origin)
        for name, forecaster in forecasters.items():
            output = forecaster(history, steps)
            for horizon in range(1, steps + 1):
                row = {
                    "model": name,
                    "origin": dates.iloc[origin],
                    "horizon": horizon,
                    "target": dates.iloc[origin + horizon],
                    "actual": levels[origin + horizon],
                }
                # Paths give one value per horizon; scalars (nu, converged) apply to all.
                for key, value in output.items():
                    row[key] = value[horizon - 1] if np.ndim(value) else value
                rows.append(row)
    return pd.DataFrame(rows)


def mase_scale(train_levels: np.ndarray) -> float:
    """In-sample MAE of the one-step naive forecast: the mean absolute monthly change."""
    return float(np.mean(np.abs(np.diff(train_levels))))


def accuracy(actual: np.ndarray, forecast: np.ndarray, scale: float) -> dict[str, float]:
    errors = np.asarray(actual) - np.asarray(forecast)
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors**2))
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(100 * np.mean(np.abs(errors / np.asarray(actual)))),
        "MASE": mae / scale,
    }


def metrics_table(forecasts: pd.DataFrame, scale: float) -> pd.DataFrame:
    rows = []
    for (model, horizon), group in forecasts.groupby(["model", "horizon"], sort=False):
        if horizon in HORIZONS:
            scores = accuracy(group["actual"].to_numpy(), group["forecast"].to_numpy(), scale)
            rows.append({"model": model, "horizon": horizon, "n": len(group), **scores})
    return pd.DataFrame(rows)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> tuple[float, float]:
    """Diebold-Mariano test of equal expected loss, with the Harvey-Leybourne-Newbold small-sample
    correction. A negative statistic means model a has the lower loss. Two-sided p-value from a
    t distribution with n - 1 degrees of freedom."""
    differences = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    count = len(differences)
    deviations = differences - differences.mean()
    # h-step forecasts from consecutive origins overlap, so their loss differences are
    # autocorrelated up to lag h - 1.
    autocovariances = [
        np.sum(deviations[lag:] * deviations[: count - lag]) / count for lag in range(horizon)
    ]
    long_run_variance = autocovariances[0] + 2 * sum(autocovariances[1:])
    # shortcut: the rectangular kernel can turn negative for h > 1; fall back to the lag-0
    # variance then, as common implementations do. Upgrade to a Bartlett kernel if this recurs.
    if long_run_variance <= 0:
        long_run_variance = autocovariances[0]
    if long_run_variance == 0:
        return 0.0, 1.0
    statistic = differences.mean() / np.sqrt(long_run_variance / count)
    statistic *= np.sqrt((count + 1 - 2 * horizon + horizon * (horizon - 1) / count) / count)
    return float(statistic), float(2 * stats.t.sf(abs(statistic), count - 1))


def dm_against_naive(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Each model against naive, per horizon, with absolute and squared error losses."""
    errors = forecasts.assign(error=forecasts["actual"] - forecasts["forecast"])
    naive = errors[errors["model"] == NAIVE].set_index(["origin", "horizon"])["error"]
    rows = []
    for (model, horizon), group in errors[errors["model"] != NAIVE].groupby(
        ["model", "horizon"], sort=False
    ):
        if horizon not in HORIZONS:
            continue
        model_errors = group.set_index(["origin", "horizon"])["error"]
        naive_errors = naive.loc[model_errors.index]
        row = {"model": model, "horizon": horizon}
        for loss, function in (("absolute", np.abs), ("squared", np.square)):
            statistic, p_value = diebold_mariano(
                function(model_errors.to_numpy()), function(naive_errors.to_numpy()), horizon
            )
            row |= {f"DM_{loss}": statistic, f"DM_{loss}_p": p_value}
        rows.append(row)
    return pd.DataFrame(rows)


def dm_verdict(statistic: float, p_value: float) -> str:
    if not np.isfinite(p_value):
        return "-"
    if p_value >= SIGNIFICANCE_LEVEL:
        return f"p={p_value:.2f}"
    return f"{'better' if statistic < 0 else 'worse'} p={p_value:.3f}"


def standardized_quantiles(nu: float) -> np.ndarray:
    """QUANTILES of a zero-mean, unit-variance shock: normal, or Student-t rescaled to unit
    variance (arch's parametrization)."""
    if not np.isfinite(nu):
        return stats.norm.ppf(QUANTILES)
    return stats.t.ppf(QUANTILES, nu) * np.sqrt((nu - 2) / nu)


def quantile_forecasts(forecast: np.ndarray, spread: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """One row of QUANTILES per forecast: the point forecast plus the spread times the shock's
    standardized quantiles."""
    # shortcut: a sum of Student-t shocks isn't exactly Student-t; treating the level error as
    # one is the usual approximation. Upgrade to simulated paths if the tails miscalibrate.
    offsets = np.vstack([standardized_quantiles(value) for value in nu])
    return np.asarray(forecast)[:, None] + np.asarray(spread)[:, None] * offsets


def pinball_loss(actual: np.ndarray, quantiles: np.ndarray) -> float:
    """Mean quantile loss over all forecasts and QUANTILES: under-forecasts cost q per unit,
    over-forecasts 1 - q. Lower is better; it rewards intervals that are calibrated and narrow."""
    gaps = np.asarray(actual)[:, None] - quantiles
    levels = np.array(QUANTILES)
    return float(np.mean(np.maximum(levels * gaps, (levels - 1) * gaps)))


def interval_scores(actual: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    low_95, low_80, _, high_80, high_95 = quantiles.T
    scores = {
        "coverage_80": float(np.mean((actual >= low_80) & (actual <= high_80))),
        "coverage_95": float(np.mean((actual >= low_95) & (actual <= high_95))),
        "width_80": float(np.mean(high_80 - low_80)),
        "width_95": float(np.mean(high_95 - low_95)),
        "pinball": pinball_loss(actual, quantiles),
    }
    for level, column in zip(QUANTILES, quantiles.T):
        scores[f"below_{100 * level:g}"] = float(np.mean(actual < column))
    return scores


def interval_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variance in VARIANCES:
        spread_column = f"sd_{variance}"
        if spread_column not in forecasts:
            continue
        with_spread = forecasts.dropna(subset=[spread_column])
        for (model, horizon), group in with_spread.groupby(["model", "horizon"], sort=False):
            if horizon not in HORIZONS:
                continue
            nu = (
                group[f"nu_{variance}"]
                if f"nu_{variance}" in group
                else np.full(len(group), np.inf)
            )
            quantiles = quantile_forecasts(group["forecast"], group[spread_column], np.asarray(nu))
            rows.append(
                {
                    "model": model,
                    "variance": variance,
                    "horizon": horizon,
                    "n": len(group),
                    **interval_scores(group["actual"].to_numpy(), quantiles),
                }
            )
    return pd.DataFrame(rows)


def convergence_failures(forecasts: pd.DataFrame) -> dict[str, list[str]]:
    """Per SARIMA model, the origins (months) whose refit didn't converge."""
    if "converged" not in forecasts:
        return {}
    refits = forecasts.dropna(subset=["converged"]).drop_duplicates(["model", "origin"])
    return {
        model: [
            f"{origin:%Y-%m}" for origin in group.loc[~group["converged"].astype(bool), "origin"]
        ]
        for model, group in refits.groupby("model", sort=False)
    }


def accuracy_report(metrics: pd.DataFrame, selected: dict[str, str]) -> list[str]:
    lines = []
    for column in VALUE_COLUMNS:
        series_metrics = metrics[metrics["series"] == column]
        lines.append(f"{column}  (selected on validation: {selected[column]})")
        for horizon in HORIZONS:
            table = series_metrics[series_metrics["horizon"] == horizon].sort_values("MASE")
            naive_mase = table.loc[table["model"] == NAIVE, "MASE"].iloc[0]
            lines.append(f"  {horizon}-month horizon ({table['n'].iloc[0]} forecasts)")
            lines.append(
                f"    {'model':<30}{'MAE':>9}{'MSE':>11}{'RMSE':>9}{'MAPE %':>9}{'MASE':>8}"
                f"   {'DM vs naive (abs)':<20}{'DM vs naive (sq)':<20}"
            )
            for row in table.itertuples():
                lines.append(
                    f"    {row.model:<30}{row.MAE:9.4f}{row.MSE:11.6f}{row.RMSE:9.4f}"
                    f"{row.MAPE:9.2f}{row.MASE:8.3f}"
                    f"   {dm_verdict(row.DM_absolute, row.DM_absolute_p):<20}"
                    f"{dm_verdict(row.DM_squared, row.DM_squared_p):<20}"
                )
            best = table.iloc[0]
            verdict = (
                f"beats naive by {100 * (1 - best['MASE'] / naive_mase):.0f}% in MAE"
                if best["model"] != NAIVE
                else "nothing beats naive"
            )
            lines.append(f"    Lowest MAE: {best['model']} ({verdict})")
        lines.append("")
    return lines


def interval_report(intervals: pd.DataFrame) -> list[str]:
    lines = [
        "  Forecast intervals (targets: 80% and 95% coverage; below_q should be close to q)",
    ]
    for column in VALUE_COLUMNS:
        series_intervals = intervals[intervals["series"] == column]
        lines.append(f"  {column}")
        for horizon in HORIZONS:
            table = series_intervals[series_intervals["horizon"] == horizon].sort_values("pinball")
            lines.append(f"    {horizon}-month horizon")
            lines.append(
                f"      {'model':<30}{'variance':<10}{'cov 80%':>8}{'cov 95%':>8}"
                f"{'width 80%':>11}{'width 95%':>11}{'pinball':>9}   below 2.5 / 10 / 50 / 90 / 97.5%"
            )
            # Plain dictionaries: itertuples would rename the "below_2.5" columns.
            for row in table.to_dict("records"):
                below = " / ".join(f"{row[f'below_{100 * q:g}']:.2f}" for q in QUANTILES)
                lines.append(
                    f"      {row['model']:<30}{row['variance']:<10}{row['coverage_80']:8.2f}"
                    f"{row['coverage_95']:8.2f}{row['width_80']:11.4f}{row['width_95']:11.4f}"
                    f"{row['pinball']:9.4f}   {below}"
                )
        lines.append("")
    return lines


def evaluation_report(results: dict, periods: dict[str, str], scales: dict, selected: dict) -> str:
    lines = [
        "Out-of-sample forecast evaluation (step U6): validation, then test",
        (
            "Rolling origin, expanding window: each model is refitted at every monthly origin on "
            f"all data up to it (data start {periods['start']}); each origin is one fold. "
            "Orders come from training, the selected model per series from validation (lowest "
            "one-month MASE), fixed before test. Test is run once, with all models."
        ),
        (
            "Point errors are on the levels: MAE, RMSE in trillions; MSE in trillions squared; "
            "MAPE in %; MASE = MAE / the training split's in-sample MAE of the one-month naive "
            "forecast (scales: "
            + ", ".join(f"{column} {scale:.4f}" for column, scale in scales.items())
            + ")."
        ),
        (
            "DM vs naive: Diebold-Mariano test of equal expected loss against the naive "
            "forecast, absolute-error loss (matches MAE) and squared-error loss (matches MSE), "
            "with the Harvey-Leybourne-Newbold correction and h - 1 autocovariance lags. "
            f"'better' / 'worse' when significant at {SIGNIFICANCE_LEVEL:.0%}, otherwise the "
            "p-value."
        ),
        (
            "Intervals: quantiles of each model's forecast distribution on the levels, with a "
            "constant shock variance or the GARCH variant chosen in step U5b (refitted at every "
            "origin); normal for constant variance, unit-variance Student-t for GARCH-t; naive "
            "uses random-walk quantiles. Pinball = mean quantile loss over the 5 quantiles."
        ),
        "The COVID months (2020) are in validation and are kept, by the user's decision.",
        "",
    ]
    for split in SPLITS:
        metrics, intervals, failures = results[split]
        lines.append(f"===== {split.upper()}: targets {periods[split]} =====")
        lines.append("")
        lines.extend(accuracy_report(metrics, selected))
        lines.extend(interval_report(intervals))
        failed = {model: months for model, months in failures.items() if months}
        refits = sum(len(months) for months in failures.values())
        lines.append(
            "  Refits that failed to converge: "
            + (
                "; ".join(f"{model}: {', '.join(months)}" for model, months in failed.items())
                if failed
                else "none"
            )
        )
        lines.append(f"  ({refits} failed refits in total for this split)" if failed else "")
        lines.append("")
    lines.append(f"All numbers: {repo_path(METRICS_PATH)}, {repo_path(INTERVALS_PATH)}")
    lines.append(f"Figures: {repo_path(FIGURE_PATH)}, {repo_path(INTERVALS_FIGURE_PATH)}")
    return "\n".join(lines)


def series_title(column: str) -> str:
    return column.removesuffix("_T").replace("_", " ")


def model_styles(sarima_names: list[str]) -> dict[str, tuple[str, float]]:
    styles = {name: (color, 2) for name, color in zip(sarima_names, MODEL_COLORS)}
    return styles | BASELINE_STYLES


def evaluation_figure(
    metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    sarima_names: dict[str, list[str]],
    selected: dict[str, str],
    test_start: pd.Timestamp,
) -> plt.Figure:
    """Per series: each model's MAE relative to naive by horizon on validation and on test, and
    the actual path against the selected model's and the naive one-month forecasts."""
    apply_chart_style()
    rows = len(VALUE_COLUMNS)
    figure, axes = plt.subplots(
        rows,
        3,
        figsize=(21, 5.4 * rows),
        squeeze=False,
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": (1, 1, 1.4)},
    )
    for row, column in enumerate(VALUE_COLUMNS):
        styles = model_styles(sarima_names[column])
        for axis in axes[row]:
            axis.set_facecolor(SURFACE)
        for split, axis in zip(SPLITS, axes[row][:2]):
            split_metrics = metrics[(metrics["series"] == column) & (metrics["split"] == split)]
            # MASE grows with the horizon for every model, which hides the differences between
            # them; dividing by naive's MASE at the same horizon shows the decision directly.
            naive_mase = split_metrics[split_metrics["model"] == NAIVE].set_index("horizon")[
                "MASE"
            ]
            for name, (color, width) in styles.items():
                if name == NAIVE:
                    continue
                model_metrics = split_metrics[split_metrics["model"] == name].set_index("horizon")
                relative = model_metrics["MASE"] / naive_mase
                axis.plot(relative.index, relative, color=color, linewidth=width, label=name)
                # Filled: significantly different from naive (DM, absolute loss); hollow: not.
                significant = model_metrics["DM_absolute_p"] < SIGNIFICANCE_LEVEL
                for filled, face in ((significant, color), (~significant, SURFACE)):
                    axis.scatter(
                        relative.index[filled],
                        relative[filled],
                        s=36,
                        color=face,
                        edgecolors=color,
                        linewidths=1.5,
                        zorder=3,
                    )
            axis.axhline(1, color=PRIMARY_INK, linewidth=2, label=NAIVE)
            axis.set_xticks(HORIZONS)
            axis.set_xlabel("Horizon (months)")
            axis.set_ylabel("MAE relative to naive")
            axis.set_title(
                f"{series_title(column)}, {split}: forecast error relative to naive\n"
                "below the naive line at 1: more accurate than naive\n"
                "filled point: significantly different from naive (Diebold-Mariano, 5%)",
                loc="left",
                fontsize=10,
            )
            # Below the axes: the lines can sit anywhere between the worst and the best model.
            axis.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3, fontsize=8, frameon=False
            )

        path_axis = axes[row][2]
        one_month = forecasts[(forecasts["series"] == column) & (forecasts["horizon"] == 1)]
        actual = one_month[one_month["model"] == NAIVE]
        path_axis.plot(
            actual["target"], actual["actual"], color=PRIMARY_INK, linewidth=2, label="actual"
        )
        for name in (selected[column], NAIVE):
            model_path = one_month[one_month["model"] == name]
            path_axis.plot(
                model_path["target"],
                model_path["forecast"],
                color=styles[name][0] if name != NAIVE else SECONDARY_INK,
                linewidth=1.5,
                marker="o",
                markersize=2.5,
                label=f"{name}, 1 month ahead",
            )
        path_axis.axvline(test_start, color=SECONDARY_INK, linewidth=1)
        path_axis.annotate(
            "validation | test ",
            (test_start, 1),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            color=SECONDARY_INK,
            fontsize=9,
        )
        path_axis.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(path_axis.xaxis.get_major_locator())
        )
        path_axis.set_ylabel("Trillions")
        path_axis.set_title(
            f"{series_title(column)}: actual vs one-month forecasts\n"
            f"selected model {selected[column]}, and naive\n",
            loc="left",
            fontsize=10,
        )
        path_axis.legend(loc="upper left", fontsize=8, frameon=False)

    figure.tight_layout()
    return figure


def intervals_figure(intervals: pd.DataFrame, selected: dict[str, str]) -> plt.Figure:
    """Per series and split: the selected model's 80% and 95% coverage by horizon, constant
    variance vs GARCH, against the targets."""
    apply_chart_style()
    rows = len(VALUE_COLUMNS)
    figure, axes = plt.subplots(
        rows, len(SPLITS), figsize=(14, 4.4 * rows), squeeze=False, facecolor=SURFACE
    )
    for row, column in enumerate(VALUE_COLUMNS):
        for split, axis in zip(SPLITS, axes[row]):
            axis.set_facecolor(SURFACE)
            chosen = intervals[
                (intervals["series"] == column)
                & (intervals["split"] == split)
                & (intervals["model"] == selected[column])
            ]
            for target in (0.8, 0.95):
                axis.axhline(target, color=PRIMARY_INK, linewidth=1)
                axis.annotate(
                    f" {target:.0%} target",
                    (1, target),
                    xycoords=("axes fraction", "data"),
                    va="center",
                    color=SECONDARY_INK,
                    fontsize=9,
                )
            for variance in VARIANCES:
                lines = chosen[chosen["variance"] == variance].set_index("horizon")
                for level, marker in (("95", "o"), ("80", "s")):
                    axis.plot(
                        lines.index,
                        lines[f"coverage_{level}"],
                        color=VARIANCE_COLORS[variance],
                        linewidth=2,
                        marker=marker,
                        markersize=6,
                        label=f"{level}% interval, "
                        + ("constant variance" if variance == "constant" else "with GARCH"),
                    )
            axis.set_ylim(0, 1.05)
            axis.set_xticks(HORIZONS)
            axis.set_xlabel("Horizon (months)")
            axis.set_ylabel("Share of actuals inside the interval")
            axis.set_title(
                f"{series_title(column)}, {split}: {selected[column]}\n"
                "interval coverage; on its target line: honest; below: too narrow",
                loc="left",
                fontsize=10,
            )
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncols=4, frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0, 0.96, 1 - 0.35 / figure.get_figheight()))
    return figure


@app.command()
def main(
    levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv",
    differences_path: Path = PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv",
    candidates_path: Path = CANDIDATES_PATH,
) -> None:
    checked, _ = checked_models(levels_path, differences_path, candidates_path)
    passed = [result for result in checked if result["passed"]]
    # The GARCH variant each model carries forward, chosen on the training residuals (step U5b).
    garch_variants = {
        (result["series"], model_label(result)): garch_check(result["residuals"])["garch_name"]
        for result in passed
    }
    levels = pd.read_csv(levels_path, parse_dates=[DATE_COLUMN])
    levels = levels.sort_values(DATE_COLUMN).reset_index(drop=True)
    splits = split_by_time(levels, split_boundaries(levels))
    train_end = len(splits["train"]) - 1
    validation_end = train_end + len(splits["validation"])
    windows = {
        "validation": (train_end, validation_end),
        "test": (validation_end, len(levels) - 1),
    }
    dates = levels[DATE_COLUMN]
    periods = {"start": f"{dates.iloc[0]:%Y-%m}"} | {
        split: f"{dates.iloc[first + 1]:%Y-%m} to {dates.iloc[last]:%Y-%m}"
        for split, (first, last) in windows.items()
    }

    scales = {
        column: mase_scale(splits["train"][column].to_numpy(dtype=float))
        for column in VALUE_COLUMNS
    }
    sarima_names = {
        column: [model_label(m) for m in passed if m["series"] == column]
        for column in VALUE_COLUMNS
    }
    all_forecasts, all_metrics, all_intervals, failures = [], [], [], {}
    for split, (first_origin, last_target) in windows.items():
        failures[split] = {}
        for column in VALUE_COLUMNS:
            values = levels[column].to_numpy(dtype=float)
            forecasters: dict[str, Forecaster] = {
                model_label(m): sarima_forecaster(m, garch_variants[(column, model_label(m))])
                for m in passed
                if m["series"] == column
            }
            forecasters |= {
                NAIVE: naive_forecast,
                "drift": drift_forecast,
                "seasonal naive": seasonal_naive_forecast,
            }
            logger.info(f"{split}, {column}: rolling forecasts for {len(forecasters)} models")
            forecasts = rolling_forecasts(dates, values, first_origin, last_target, forecasters)
            labels = {"split": split, "series": column}
            all_forecasts.append(forecasts.assign(**labels))
            all_metrics.append(
                metrics_table(forecasts, scales[column])
                .merge(dm_against_naive(forecasts), on=["model", "horizon"], how="left")
                .assign(**labels)
            )
            all_intervals.append(interval_table(forecasts).assign(**labels))
            failures[split] |= convergence_failures(forecasts)

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    intervals = pd.concat(all_intervals, ignore_index=True)
    validation_one_month = metrics[
        (metrics["split"] == "validation")
        & (metrics["horizon"] == 1)
        & ~metrics["model"].isin(BASELINES)
    ]
    selected = {
        column: group.sort_values("MASE")["model"].iloc[0]
        for column, group in validation_one_month.groupby("series")
    }
    metrics[["split", "series", "model", "horizon", "n", *METRICS, *DM_COLUMNS]].to_csv(
        METRICS_PATH, index=False
    )
    intervals.to_csv(INTERVALS_PATH, index=False)

    results = {
        split: (
            metrics[metrics["split"] == split],
            intervals[intervals["split"] == split],
            failures[split],
        )
        for split in SPLITS
    }
    report = evaluation_report(results, periods, scales, selected)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    test_start = dates.iloc[validation_end + 1]
    figure = evaluation_figure(metrics, forecasts, sarima_names, selected, test_start)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    figure = intervals_figure(intervals, selected)
    figure.savefig(INTERVALS_FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"Forecast evaluation written to {REPORT_PATH} and {FIGURES_DIR}")


if __name__ == "__main__":
    app()
