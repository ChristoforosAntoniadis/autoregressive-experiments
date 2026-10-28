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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from matplotlib.figure import Figure
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
from autoregressive_experiments.config_loader import HYPERPARAMETERS
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
    SERIES_PALETTE,
    SURFACE,
    apply_chart_style,
    checked_models,
    model_label,
    model_name,
)

app = typer.Typer()

# Settings; each value's justification is in configs/hyperparameters.json.
# Lags (months) for the reported ARCH-LM tests.
ARCH_TEST_LAGS = HYPERPARAMETERS.volatility_checks.arch_test_lags
# Lags shown in the figure's ARCH-LM panel (every lag from 1).
PLOT_LAGS = HYPERPARAMETERS.volatility_checks.plot_lags
# Each regime keeps at least this share of the months.
BREAK_TRIM = HYPERPARAMETERS.volatility_checks.break_trim

# Verdict key -> the sentence explaining it in the report. Decided in volatility_check().
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
ADJUSTED_COLOR = SERIES_PALETTE[1]


def arch_lm_p_values(
    residuals: np.ndarray | pd.Series, lags: Sequence[int] | np.ndarray
) -> np.ndarray:
    """
    Engle's ARCH-LM p-value at each lag (null hypothesis: no ARCH effects).

    Parameters
    ----------
    residuals : np.ndarray or pd.Series
        Residuals (or standardized residuals) with mean about zero, shape (n,). Their units
        don't matter: the test is scale-free.
    lags : sequence of int or np.ndarray
        The numbers of lags to test, e.g. ARCH_TEST_LAGS or np.arange(1, 13). Each lag gets
        its own test.

    Returns
    -------
    np.ndarray
        One p-value per entry of `lags`, shape (len(lags),). A small p-value means the
        squared residuals are predictable from their own past, i.e. volatility clusters.

    Notes
    -----
    ARCH-LM regresses the squared residuals on a constant and their own first `lag` lags;
    n * R^2 is chi-squared with `lag` degrees of freedom under the null.
    """
    values = np.asarray(residuals, dtype=float)
    return np.array([het_arch(values, nlags=lag, result_object=True).lmpval for lag in lags])


def variance_break(residuals: np.ndarray) -> int:
    """
    Index where the second variance regime starts.

    The split that maximizes the likelihood of a normal model with one variance before it and
    another after it.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals with mean about zero, shape (n,), in time order.

    Returns
    -------
    int
        k, the position of the first month of the second regime: the first regime is
        residuals[:k], the second residuals[k:]. Each regime holds at least
        max(2, int(n * BREAK_TRIM)) months.

    Notes
    -----
    Processing:
    1. Square the residuals.
    2. For every allowed split k, compute k log(mean square before) + (n - k)
       log(mean square after).
    3. Return the k with the smallest value (the first one on a tie).

    Why this cost: for zero-mean normal data with variance s1 before and s2 after, the
    maximized log-likelihood is -(k log s1 + (n - k) log s2 + n (1 + log 2 pi)) / 2, so
    minimizing the cost maximizes the likelihood (the cost is twice the negative
    log-likelihood, up to a constant). The mean square is the variance's maximum-likelihood
    estimate when the mean is zero, as it is for SARIMA residuals.

    Why the trim (BREAK_TRIM, 15%): without it the search would favour splits a few months
    from either end, where one variance rests on a handful of points. Because the date is
    searched for, the usual test distributions don't apply to it; volatility_check() relies
    on ARCH-LM after removing the break instead.
    """
    squares = np.asarray(residuals, dtype=float) ** 2
    count = len(squares)
    # At least 2 months per regime even for very short series, so each mean square exists.
    shortest = max(2, int(count * BREAK_TRIM))
    # The + 1 makes the last split leave exactly `shortest` months after it.
    splits = range(shortest, count - shortest + 1)
    # Residuals have mean ~0, so each regime's variance is its mean square. Up to a constant,
    # twice the negative log-likelihood of a split at k is
    # k log(var_before) + (n - k) log(var_after).
    costs = [
        k * np.log(squares[:k].mean()) + (count - k) * np.log(squares[k:].mean()) for k in splits
    ]
    return splits[int(np.argmin(costs))]


def regime_scales(residuals: np.ndarray, break_index: int) -> np.ndarray:
    """
    Each month's regime root mean square: one value before the break, another after.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals with mean about zero, shape (n,), in trillions.
    break_index : int
        First position of the second regime, as returned by variance_break().

    Returns
    -------
    np.ndarray
        Shape (n,), in trillions: the first regime's root mean square repeated for the
        first break_index months, then the second regime's for the rest.

    Notes
    -----
    The root mean square (not np.std) is used because the residuals are taken to have zero
    mean, matching the zero-mean variance models fitted on them in volatility_models.
    """
    values = np.asarray(residuals, dtype=float)
    before, after = values[:break_index], values[break_index:]
    return np.concatenate(
        [
            np.full(len(before), np.sqrt(np.mean(before**2))),
            np.full(len(after), np.sqrt(np.mean(after**2))),
        ]
    )


def standardize_by_regime(residuals: np.ndarray, break_index: int) -> np.ndarray:
    """
    Divide each regime's residuals by that regime's root mean square.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals with mean about zero, shape (n,).
    break_index : int
        First position of the second regime, as returned by variance_break().

    Returns
    -------
    np.ndarray
        Shape (n,), unitless: each regime has a mean square of exactly 1, so the variance
        break is removed and only within-regime clustering is left for ARCH-LM to find.
    """
    return np.asarray(residuals, dtype=float) / regime_scales(residuals, break_index)


def volatility_check(residuals: np.ndarray | pd.Series) -> dict[str, Any]:
    """
    Test one model's residuals for ARCH effects, before and after removing a variance break.

    Parameters
    ----------
    residuals : np.ndarray or pd.Series
        One SARIMA model's residuals, shape (n,), in trillions.

    Returns
    -------
    dict of str to Any
        - "break_index": int, first position of the second regime (variance_break())
        - "sd_before", "sd_after": float, each regime's root mean square, in trillions
        - "adjusted": np.ndarray, shape (n,), the break-adjusted (regime-standardized)
          residuals
        - "raw_p", "adjusted_p": np.ndarray, ARCH-LM p-values at ARCH_TEST_LAGS, as they are
          and break-adjusted
        - "verdict": str, a key of VERDICTS

    Notes
    -----
    Processing:
    1. Estimate the variance break and standardize each regime by its own scale.
    2. Run ARCH-LM at ARCH_TEST_LAGS on both versions.
    3. Decide the verdict, in order:
       - every raw p-value >= 5%: "NO ARCH" (a constant variance is enough)
       - otherwise, every break-adjusted p-value >= 5%: "VARIANCE BREAK" (one shift
         explains the ARCH effects)
       - otherwise: "GARCH NEEDED" (clustering survives removing the break)
    """
    values = np.asarray(residuals, dtype=float)
    break_index = variance_break(values)
    adjusted = standardize_by_regime(values, break_index)
    raw_p = arch_lm_p_values(values, ARCH_TEST_LAGS)
    adjusted_p = arch_lm_p_values(adjusted, ARCH_TEST_LAGS)
    if (raw_p >= SIGNIFICANCE_LEVEL).all():
        verdict = "NO ARCH"
    # Only reached when the raw residuals show ARCH effects at some lag.
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


def volatility_report(results: list[dict[str, Any]], train: pd.DataFrame) -> str:
    """
    Write the text report of the volatility checks.

    Parameters
    ----------
    results : list of dict of str to Any
        One dict per model that passed the residual checks: its check_model() result merged
        with its volatility_check() result.
    train : pd.DataFrame
        The training split of the first differences, with DATE_COLUMN.

    Returns
    -------
    str
        The report: a header explaining the method, then per model the break date and the
        months on each side, both regimes' standard deviations and their ratio, the ARCH-LM
        p-values at each lag as is and break-adjusted, and the verdict; then the figure's
        path.
    """
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
            "non-standard test distribution, so the decisive evidence is ARCH-LM after "
            "removing it."
        ),
        "Break-adjusted: each regime's residuals divided by that regime's standard deviation.",
        "",
    ]
    for column in VALUE_COLUMNS:
        for result in (r for r in results if r["series"] == column):
            # The split resets the index to 0..n-1, so residual index labels are positions.
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


def volatility_figure(results: list[dict[str, Any]], dates: pd.Series) -> Figure:
    """
    One row per model: residuals with each regime's band, and ARCH-LM p-value by lag.

    Parameters
    ----------
    results : list of dict of str to Any
        As in volatility_report(): check_model() results merged with volatility_check().
    dates : pd.Series
        The training split's dates, positionally aligned with the first differences.

    Returns
    -------
    Figure
        A figure with len(results) rows and 2 columns and one shared legend on top. The
        caller saves and closes it.

    Notes
    -----
    Processing, per model row:
    1. Left: the residuals over time, each regime shaded at +/- 2 of its own standard
       deviations, and a vertical line at the break date.
    2. Right: the ARCH-LM p-value at every lag 1..PLOT_LAGS, as is and break-adjusted, on
       a log scale against the 5% line.

    Then one figure legend above the rows, with room reserved for it in tight_layout.
    """
    apply_chart_style()
    figure, axes = plt.subplots(
        len(results), 2, figsize=(14, 3.2 * len(results)), squeeze=False, facecolor=SURFACE
    )
    lags = np.arange(1, PLOT_LAGS + 1)
    for row, result in enumerate(results):
        residuals = result["residuals"]
        # reset_index: positions 0..n-1 match break_index for the .iloc slices below.
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
        # x in dates, y in axes fraction (1 = top), so the label sits at the top of the line.
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
        # A third below the smallest p-value; 1e-300 floors it, since p-values can underflow
        # to 0, which a log axis can't show.
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
    # Leave 0.35 inch free at the top for the legend, whatever the figure's height.
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.35 / figure.get_figheight()))
    return figure


@app.command()
def main(
    levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv",
    differences_path: Path = PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv",
    candidates_path: Path = CANDIDATES_PATH,
) -> None:
    """
    Run the volatility checks and write the report and the figure (``make volatility``).

    Parameters
    ----------
    levels_path : Path, default=INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv"
        The cleaned levels CSV.
    differences_path : Path, default=PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv"
        The first-differences CSV.
    candidates_path : Path, default=CANDIDATES_PATH
        The information-criteria grid CSV.

    Returns
    -------
    None
        Writes REPORT_PATH and FIGURE_PATH, and prints the report. Only the models that
        passed the residual checks (U4) are examined.
    """
    checked, train = checked_models(levels_path, differences_path, candidates_path)
    # Each result keeps its U4 fields (series, orders, residuals) and gains the U5 ones.
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
    # Free the figure's memory; it is already saved.
    plt.close(figure)
    logger.success(f"Volatility checks written to {REPORT_PATH} and {FIGURE_PATH}")


# Running this file directly runs the typer CLI (main).
if __name__ == "__main__":
    app()
