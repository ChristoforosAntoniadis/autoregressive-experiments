"""
Volatility checks (step U5): is the variance of each model's errors constant, and if not, is it
one shift in variance or ongoing volatility clustering?

The two need different fixes. Clustering (large errors follow large errors throughout) calls for
GARCH on the residuals. A single variance break (a calm regime, then a volatile one) calls for
treating the regimes separately instead. So each model's residuals get Engle's ARCH-LM test
twice: as they are, and after dividing each regime by its own standard deviation, with the
break date estimated from the data.

Input: the same as ``residual_checks``: the training split and
``reports/information_criteria_candidates.csv``. Only models that pass the residual checks are
examined.

Output, in ``reports/``:

- ``volatility_diagnostics.txt``: per model, the estimated variance break, ARCH-LM p-values
  before and after removing it, and a verdict
- ``figures/volatility_diagnostics.png``: per model, the residuals with each regime's
  +/- 2 standard deviation band, and the ARCH-LM p-value by lag before and after removing
  the break
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_arch
import typer

from autoregressive_experiments.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from autoregressive_experiments.diagnostics import (
    CANDIDATES_PATH,
    SIGNIFICANCE_LEVEL,
    VALUE_COLUMNS,
    repo_path,
)
from autoregressive_experiments.features import DATE_COLUMN
from autoregressive_experiments.residual_checks import (
    CRITICAL_COLOR,
    GRIDLINE,
    SECONDARY_INK,
    SERIES_COLOR,
    SURFACE,
    apply_chart_style,
    checked_models,
    model_label,
    model_name,
)

app = typer.Typer()

ARCH_TEST_LAGS = (1, 6, 12)
PLOT_LAGS = 12
# Each regime keeps at least this share of the months, so neither variance rests on a few points.
BREAK_TRIM = 0.15

VERDICTS = {
    "NO ARCH": "No ARCH effects: a constant variance is adequate.",
    "VARIANCE BREAK": (
        "One shift in variance explains the ARCH effects: treat the two regimes separately "
        "(model the post-break period, or add a variance break) rather than fit GARCH."
    ),
    "GARCH NEEDED": (
        "ARCH effects remain after removing the variance break: volatility clusters, so fit "
        "GARCH on the residuals."
    ),
}

REPORT_PATH = REPORTS_DIR / "volatility_diagnostics.txt"
FIGURE_PATH = FIGURES_DIR / "volatility_diagnostics.png"

# Reference palette, categorical slot 2 (validated against slot 1 for colour-blind separation).
ADJUSTED_COLOR = "#eb6834"


def arch_lm_p_values(residuals: np.ndarray, lags: tuple[int, ...] | range) -> np.ndarray:
    """Engle's ARCH-LM p-value at each lag (null hypothesis: no ARCH effects)."""
    values = np.asarray(residuals, dtype=float)
    return np.array([het_arch(values, nlags=lag, result_object=True).lmpval for lag in lags])


def variance_break(residuals: np.ndarray) -> int:
    """Index where the second variance regime starts: the split that maximizes the likelihood of
    a normal model with one variance before it and another after it."""
    squares = np.asarray(residuals, dtype=float) ** 2
    count = len(squares)
    shortest = max(2, int(count * BREAK_TRIM))
    splits = range(shortest, count - shortest + 1)
    # Residuals have mean ~0, so each regime's variance is its mean square. Up to constants, the
    # negative log-likelihood of a split at k is k log(var_before) + (n - k) log(var_after).
    costs = [
        k * np.log(squares[:k].mean()) + (count - k) * np.log(squares[k:].mean()) for k in splits
    ]
    return splits[int(np.argmin(costs))]


def regime_scales(residuals: np.ndarray, break_index: int) -> np.ndarray:
    """Each month's regime root mean square: one value before the break, another after."""
    values = np.asarray(residuals, dtype=float)
    before, after = values[:break_index], values[break_index:]
    return np.concatenate(
        [
            np.full(len(before), np.sqrt(np.mean(before**2))),
            np.full(len(after), np.sqrt(np.mean(after**2))),
        ]
    )


def standardize_by_regime(residuals: np.ndarray, break_index: int) -> np.ndarray:
    """Divide each regime's residuals by that regime's root mean square."""
    return np.asarray(residuals, dtype=float) / regime_scales(residuals, break_index)


def volatility_check(residuals: np.ndarray) -> dict:
    values = np.asarray(residuals, dtype=float)
    break_index = variance_break(values)
    adjusted = standardize_by_regime(values, break_index)
    raw_p = arch_lm_p_values(values, ARCH_TEST_LAGS)
    adjusted_p = arch_lm_p_values(adjusted, ARCH_TEST_LAGS)
    if (raw_p >= SIGNIFICANCE_LEVEL).all():
        verdict = "NO ARCH"
    elif (adjusted_p >= SIGNIFICANCE_LEVEL).all():
        verdict = "VARIANCE BREAK"
    else:
        verdict = "GARCH NEEDED"
    return {
        "break_index": break_index,
        "sd_before": float(np.sqrt(np.mean(values[:break_index] ** 2))),
        "sd_after": float(np.sqrt(np.mean(values[break_index:] ** 2))),
        "adjusted": adjusted,
        "raw_p": raw_p,
        "adjusted_p": adjusted_p,
        "verdict": verdict,
    }


def volatility_report(results: list[dict], train: pd.DataFrame) -> str:
    dates = train[DATE_COLUMN]
    lines = [
        "Volatility checks of the models that passed the residual checks (step U5)",
        (
            f"Training split only: {dates.min():%Y-%m} to {dates.max():%Y-%m}, "
            f"{len(train)} first differences."
        ),
        (
            "Engle's ARCH-LM null hypothesis: no ARCH effects (constant conditional variance). "
            f"Significance level: {SIGNIFICANCE_LEVEL:.0%}; "
            f"lags {', '.join(map(str, ARCH_TEST_LAGS))}."
        ),
        (
            "Variance break: the split that best fits one variance before and another after, "
            f"with at least {BREAK_TRIM:.0%} of the months on each side. A searched break has a "
            "non-standard test distribution, so the decisive evidence is ARCH-LM after removing it."
        ),
        "Break-adjusted: each regime's residuals divided by that regime's standard deviation.",
        "",
    ]
    for column in VALUE_COLUMNS:
        for result in (r for r in results if r["series"] == column):
            residual_dates = dates.iloc[result["residuals"].index]
            break_index = result["break_index"]
            lines.append(f"{column}  {model_label(result)}")
            lines.append(
                f"  Variance break: {residual_dates.iloc[break_index]:%Y-%m} "
                f"({break_index} months before, {len(residual_dates) - break_index} after)"
            )
            lines.append(
                f"  Standard deviation: {result['sd_before']:.4f} before, "
                f"{result['sd_after']:.4f} after (x{result['sd_after'] / result['sd_before']:.1f})"
            )
            for lag, raw, adjusted in zip(ARCH_TEST_LAGS, result["raw_p"], result["adjusted_p"]):
                lines.append(
                    f"  lag {lag:>2}: ARCH-LM p={raw:.4g} as is, {adjusted:.4g} break-adjusted"
                )
            lines.append(f"  Verdict: {result['verdict']}. {VERDICTS[result['verdict']]}")
            lines.append("")
    lines.append(f"Figure: {repo_path(FIGURE_PATH)}")
    return "\n".join(lines)


def volatility_figure(results: list[dict], dates: pd.Series) -> plt.Figure:
    """One row per model: residuals with each regime's band, and ARCH-LM p-value by lag."""
    apply_chart_style()
    figure, axes = plt.subplots(
        len(results), 2, figsize=(14, 3.2 * len(results)), squeeze=False, facecolor=SURFACE
    )
    lags = np.arange(1, PLOT_LAGS + 1)
    for row, result in enumerate(results):
        residuals = result["residuals"]
        residual_dates = dates.iloc[residuals.index].reset_index(drop=True)
        break_index = result["break_index"]
        time_axis, p_axis = axes[row]
        for axis in axes[row]:
            axis.set_facecolor(SURFACE)

        regimes = [
            (residual_dates.iloc[:break_index], result["sd_before"]),
            (residual_dates.iloc[break_index:], result["sd_after"]),
        ]
        for regime_dates, sd in regimes:
            time_axis.fill_between(
                regime_dates, -2 * sd, 2 * sd, color=GRIDLINE, alpha=0.8, linewidth=0
            )
        time_axis.plot(residual_dates, residuals.to_numpy(), color=SERIES_COLOR, linewidth=1)
        break_date = residual_dates.iloc[break_index]
        time_axis.axvline(break_date, color=SECONDARY_INK, linewidth=1)
        time_axis.annotate(
            f" break {break_date:%Y-%m}",
            (break_date, 1),
            xycoords=("data", "axes fraction"),
            va="top",
            color=SECONDARY_INK,
            fontsize=9,
        )
        time_axis.set_title(
            f"{model_name(result)}: {model_label(result)}  [{result['verdict']}]\n"
            "Residuals; shaded: +/- 2 standard deviations of each regime",
            loc="left",
            fontsize=10,
        )
        time_axis.set_ylabel("Residual (trillions)")

        raw = arch_lm_p_values(residuals, lags)
        adjusted = arch_lm_p_values(result["adjusted"], lags)
        curves = [
            (raw, SERIES_COLOR, "residuals as they are"),
            (adjusted, ADJUSTED_COLOR, "after removing the variance break"),
        ]
        for p_values, color, label in curves:
            p_axis.plot(
                lags, p_values, color=color, linewidth=2, marker="o", markersize=5, label=label
            )
        p_axis.axhline(SIGNIFICANCE_LEVEL, color=CRITICAL_COLOR, linewidth=1)
        # Right of the panel, so the label never covers a point near the line.
        p_axis.annotate(
            " 5%",
            (1, SIGNIFICANCE_LEVEL),
            xycoords=("axes fraction", "data"),
            va="center",
            color=SECONDARY_INK,
            fontsize=9,
        )
        p_axis.set_yscale("log")
        p_axis.set_ylim(max(1e-300, min(raw.min(), adjusted.min()) / 3), 2)
        p_axis.set_xlim(0.5, PLOT_LAGS + 0.5)
        p_axis.set_xticks([1, 3, 6, 9, 12])
        p_axis.set_xlabel("Lag (months)")
        p_axis.set_title(
            "ARCH-LM p-value by lag (below the line: ARCH effects)", loc="left", fontsize=10
        )

    # One legend for the whole figure, above the rows, so it never covers a curve.
    handles, labels = axes[0][1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncols=2, frameon=False, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.35 / figure.get_figheight()))
    return figure


@app.command()
def main(
    levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv",
    differences_path: Path = PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv",
    candidates_path: Path = CANDIDATES_PATH,
) -> None:
    checked, train = checked_models(levels_path, differences_path, candidates_path)
    results = [
        {**result, **volatility_check(result["residuals"])}
        for result in checked
        if result["passed"]
    ]

    report = volatility_report(results, train)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = volatility_figure(results, train[DATE_COLUMN])
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"Volatility checks written to {REPORT_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    app()
