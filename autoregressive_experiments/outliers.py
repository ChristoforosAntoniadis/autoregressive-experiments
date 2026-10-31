"""
Outliers: finding unusual monthly changes and taming them in a model's training history.

A monthly change is flagged by a Hampel filter (Hampel 1974; Pearson 2002) when it lies more than
THRESHOLD robust standard deviations from the median of the changes within HALF_WIDTH months on
either side; the robust standard deviation is 1.4826 x the median absolute deviation, which
estimates the standard deviation of normal data (Rousseeuw and Croux 1993) without being moved
by the outliers themselves. Windows are cut at the ends of the history they are given, so a
model's history up to a forecast origin is judged on that history alone.

Each model's tuned `outliers` setting chooses what happens to the flagged changes before it is
fitted:

- "none": nothing
- "clip": each flagged change is set to the nearest bound of its window, and the levels are
  rebuilt from the clipped changes backwards from the last actual level, so every forecast still
  starts from the real latest value
- "dummies" (SARIMA and SARIMAX): one pulse regressor per flagged change, which absorbs that
  month's jump in the level (an intervention, Box and Tiao 1975); its future values are zero

Forecasts are always scored against the actual data.

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` (train and validation months only).

Output (make outliers):

- ``reports/outliers.txt``: per series, the flagged months of the history the last validation
  origin sees, with each change and its window's bounds
- ``reports/figures/outliers.png``: per series, the monthly changes with the bounds shaded and
  the flagged months marked
"""

from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from autoregressive_experiments.config import FIGURES_DIR, INTERIM_DATA_DIR, REPORTS_DIR
from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.diagnostics import repo_path
from autoregressive_experiments.features import (
    DATE_COLUMN,
    VALUE_COLUMNS,
    series_title,
    split_boundaries,
    split_by_time,
)
from autoregressive_experiments.residual_checks import (
    CRITICAL_COLOR,
    MUTED_INK,
    PRIMARY_INK,
    SECONDARY_INK,
    SERIES_PALETTE,
    SURFACE,
    apply_chart_style,
)
from autoregressive_experiments.runlog import begin_phase, finish_run, start_run_log

app = typer.Typer()

# Settings; each value's reasoning is in configs/hyperparameters.json.
SETTINGS = HYPERPARAMETERS.outliers
HALF_WIDTH = SETTINGS.half_width
THRESHOLD = SETTINGS.threshold
HUBER_DELTA = SETTINGS.huber_delta
# The median absolute deviation times this estimates the standard deviation of normal data.
MAD_TO_SD = 1.4826
NONE, CLIP, DUMMIES = "none", "clip", "dummies"

REPORT_PATH = REPORTS_DIR / "outliers.txt"
FIGURE_PATH = FIGURES_DIR / "outliers.png"


def hampel(changes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Each change's rolling median, its window's bounds and whether it falls outside them.

    Parameters
    ----------
    changes : np.ndarray
        One series' monthly changes, shape (n,), trillions, oldest first.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        The lower bounds, the upper bounds (both shape (n,), trillions) and the flags (bool,
        shape (n,)).

    Notes
    -----
    Processing, for each change:
    1. Take the changes within HALF_WIDTH months on either side (fewer at the ends).
    2. Bounds = their median +/- THRESHOLD x MAD_TO_SD x their median absolute deviation.
    3. Flag the change if it lies outside its bounds. A window whose changes are all equal has
       no spread and flags nothing.
    """
    lower, upper = np.empty(len(changes)), np.empty(len(changes))
    for index in range(len(changes)):
        window = changes[max(0, index - HALF_WIDTH) : index + HALF_WIDTH + 1]
        median = np.median(window)
        spread = THRESHOLD * MAD_TO_SD * np.median(np.abs(window - median))
        lower[index], upper[index] = median - spread, median + spread
    return lower, upper, (changes < lower) | (changes > upper)


def clip_levels(levels: np.ndarray) -> np.ndarray:
    """
    Levels rebuilt from changes with every flagged change clipped to its window's bounds.

    Parameters
    ----------
    levels : np.ndarray
        One series' levels, shape (months,), or several series', shape (months, series),
        trillions, oldest first.

    Returns
    -------
    np.ndarray
        The same shape; the last level is unchanged, and the earlier ones differ by the clipped
        amounts that follow them.
    """
    if levels.ndim == 2:
        return np.column_stack(
            [clip_levels(levels[:, column]) for column in range(levels.shape[1])]
        )
    changes = np.diff(levels)
    lower, upper, _ = hampel(changes)
    clipped = np.clip(changes, lower, upper)
    # Anchored at the end: the forecast starts from the actual last level.
    return np.asarray(levels[-1] - np.concatenate([np.cumsum(clipped[::-1])[::-1], [0.0]]))


def pulse_dummies(changes: np.ndarray, start: int = 0) -> np.ndarray | None:
    """
    One pulse regressor per flagged change, for the rows a model is fitted on.

    Parameters
    ----------
    changes : np.ndarray
        One series' monthly changes, shape (n,), trillions, oldest first.
    start : int, default=0
        The first change the model is fitted on (SARIMAX drops its first `lags`).

    Returns
    -------
    np.ndarray or None
        Shape (n - start, flagged changes from `start` on): 1 in the flagged change's row, 0
        elsewhere; None when no change from `start` on is flagged.
    """
    _, _, flags = hampel(changes)
    rows = np.flatnonzero(flags[start:])
    if not len(rows):
        return None
    pulses = np.zeros((len(changes) - start, len(rows)))
    pulses[rows, np.arange(len(rows))] = 1.0
    return pulses


def outlier_report(data: pd.DataFrame) -> str:
    """
    The flagged months of each series, with their changes and bounds.

    Parameters
    ----------
    data : pd.DataFrame
        The train and validation months: DATE_COLUMN and every VALUE_COLUMNS level.

    Returns
    -------
    str
        The report text.
    """
    lines = [
        "Outliers: monthly changes flagged by a Hampel filter",
        (
            f"History: {data[DATE_COLUMN].iloc[0]:%Y-%m} to {data[DATE_COLUMN].iloc[-1]:%Y-%m} "
            "(train and validation; the test months are never examined here). A change is flagged "
            f"when it lies more than {THRESHOLD:g} robust standard deviations (1.4826 x the "
            "median absolute deviation) from the median of the changes within "
            f"{HALF_WIDTH} months on either side. Each model's tuned 'outliers' setting decides "
            "whether flagged changes are left as they are, clipped to the bounds, or given a "
            "pulse regressor; forecasts are always scored against the actual data."
        ),
        "",
    ]
    dates = data[DATE_COLUMN].iloc[1:]
    for column in VALUE_COLUMNS:
        changes = np.diff(data[column].to_numpy(dtype=float))
        lower, upper, flags = hampel(changes)
        lines.append(f"{column}: {int(flags.sum())} of {len(changes)} changes flagged")
        lines.append(f"  {'month':8s} {'change':>9s} {'lower':>9s} {'upper':>9s}")
        for index in np.flatnonzero(flags):
            lines.append(
                f"  {dates.iloc[index]:%Y-%m}  {changes[index]:9.4f} {lower[index]:9.4f} "
                f"{upper[index]:9.4f}"
            )
        lines.append("")
    lines.append(f"Figure: {repo_path(FIGURE_PATH)}")
    return "\n".join(lines)


def outlier_figure(data: pd.DataFrame) -> Figure:
    """
    Per series, the monthly changes, their window bounds and the flagged months.

    Parameters
    ----------
    data : pd.DataFrame
        The train and validation months: DATE_COLUMN and every VALUE_COLUMNS level.

    Returns
    -------
    Figure
        One panel per series: the changes (palette slot 1), the band between the bounds
        shaded, flagged changes marked in the critical colour with a legend label. The caller
        saves and closes it.
    """
    apply_chart_style()
    figure, axes = plt.subplots(
        len(VALUE_COLUMNS), 1, figsize=(12, 3.6 * len(VALUE_COLUMNS)), facecolor=SURFACE
    )
    dates = data[DATE_COLUMN].iloc[1:].to_numpy()
    for axis, column in zip(np.atleast_1d(axes), VALUE_COLUMNS):
        changes = np.diff(data[column].to_numpy(dtype=float))
        lower, upper, flags = hampel(changes)
        axis.set_facecolor(SURFACE)
        axis.fill_between(
            dates, lower, upper, color=MUTED_INK, alpha=0.18, linewidth=0, label="window bounds"
        )
        axis.plot(dates, changes, color=SERIES_PALETTE[0], linewidth=1.2, label="monthly change")
        axis.scatter(
            dates[flags],
            changes[flags],
            color=CRITICAL_COLOR,
            s=28,
            zorder=3,
            label=f"flagged ({int(flags.sum())})",
        )
        axis.axhline(0, color=PRIMARY_INK, linewidth=0.8)
        axis.set_ylabel("Change (trillions)", color=SECONDARY_INK)
        axis.set_title(
            f"{series_title(column)}: monthly changes outside {THRESHOLD:g} robust standard "
            f"deviations of their {2 * HALF_WIDTH + 1}-month window are flagged",
            loc="left",
            fontsize=10,
        )
        axis.legend(fontsize=8.5, frameon=False, loc="upper left")
    figure.tight_layout()
    return figure


@app.command()
def main(levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv") -> None:
    """
    Flag the outliers of the train and validation history and write the report and figure.

    Parameters
    ----------
    levels_path : Path, default=INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv"
        Cleaned monthly levels of both series, trillions.

    Returns
    -------
    None
        Writes REPORT_PATH and FIGURE_PATH and prints the report.

    Notes
    -----
    Processing:
    1. Keep the train and validation months; the test months are dropped before anything else.
    2. Flag each series' changes and write the report and figure.
    """
    start_run_log("outliers")
    begin_phase("Flag the outliers of the train and validation history")
    data = pd.read_csv(levels_path, parse_dates=[DATE_COLUMN])
    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)
    splits = split_by_time(data, split_boundaries(data))
    known = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    report = outlier_report(known)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    figure = outlier_figure(known)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"Outliers written to {REPORT_PATH} and {FIGURE_PATH}")
    finish_run()


if __name__ == "__main__":
    app()
