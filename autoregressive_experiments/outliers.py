"""
Outliers: finding unusual changes and taming them in a model's training history.

A change is flagged by a Hampel filter (Hampel 1974; Pearson 2002) when it lies more than
THRESHOLD robust standard deviations from the median of the changes within HALF_WIDTH periods on
either side; the robust standard deviation is 1.4826 x the median absolute deviation, which
estimates the standard deviation of normal data (Rousseeuw and Croux 1993) without being moved
by the outliers themselves. Windows are cut at the ends of the history they are given, so a
model's history up to a forecast origin is judged on that history alone. With SEASONAL_PROFILE
(daily data with a weekly rhythm), each change is compared with the median of its own season
position within the window (the same weekday) and the spread is measured after that profile is
removed, so a regular payday is ordinary and only a departure from the usual payday is flagged.

Each model's tuned `outliers` setting chooses what happens to the flagged changes before it is
fitted:

- "none": nothing
- "clip": each flagged change is set to the nearest bound of its window, and the levels are
  rebuilt from the clipped changes backwards from the last actual level, so every forecast still
  starts from the real latest value
- "dummies" (SARIMA and SARIMAX): one pulse regressor per flagged change, which absorbs that
  period's jump in the level (an intervention, Box and Tiao 1975); its future values are zero

Forecasts are always scored against the actual data.

Input: the cleaned levels, ``data/interim/<dataset>/levels.csv`` (train and validation periods
only).

Output (make outliers):

- ``reports/<dataset>/outliers.txt``: per series, the flagged periods of the history the last
  validation origin sees, with each change and its window's bounds
- ``reports/<dataset>/figures/outliers.png``: per series, the changes with the bounds shaded and
  the flagged periods marked
"""

from pathlib import Path
import warnings

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import typer

from autoregressive_experiments.config import FIGURES_DIR, REPORTS_DIR
from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.diagnostics import SEASONAL_PERIOD, repo_path
from autoregressive_experiments.features import (
    DATE_COLUMN,
    DATE_FORMAT,
    LEVELS_PATH,
    PERIOD,
    PERIODLY,
    UNIT,
    VALUE_COLUMNS,
    load_levels,
    series_title,
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

# Settings; each value's reasoning is in the dataset's config file (configs/<dataset>.json).
SETTINGS = HYPERPARAMETERS.outliers
HALF_WIDTH = SETTINGS.half_width
SEASONAL_PROFILE = SETTINGS.seasonal_profile
THRESHOLD = SETTINGS.threshold
HUBER_DELTA = SETTINGS.huber_delta
# The median absolute deviation times this estimates the standard deviation of normal data; the
# mean absolute deviation times sqrt(pi / 2) does the same, for windows whose MAD is zero.
MAD_TO_SD = 1.4826
MEAN_AD_TO_SD = float(np.sqrt(np.pi / 2))
NONE, CLIP, DUMMIES = "none", "clip", "dummies"

REPORT_PATH = REPORTS_DIR / "outliers.txt"
FIGURE_PATH = FIGURES_DIR / "outliers.png"


def hampel(changes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Each change's rolling median, its window's bounds and whether it falls outside them.

    Parameters
    ----------
    changes : np.ndarray
        One series' changes, shape (n,), in the series' unit, oldest first.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        The lower bounds, the upper bounds (both shape (n,), in the series' unit) and the flags
        (bool, shape (n,)).

    Notes
    -----
    Processing, for each change:
    1. Take the changes within HALF_WIDTH periods on either side (fewer at the ends).
    2. With SEASONAL_PROFILE, subtract from each change in the window the median of the
       window's changes at the same season position (index modulo SEASONAL_PERIOD; a weekday
       profile for daily data), and judge the change by the residue; the bounds get the
       change's own position's median added back.
    3. Bounds = their median +/- THRESHOLD x MAD_TO_SD x their median absolute deviation.
       When more than half the window's changes are equal (zero-flow days, for example), that
       deviation is zero and would flag every other change, so the mean absolute deviation,
       scaled to a standard deviation as well, stands in.
    4. Flag the change if it lies outside its bounds. A window whose changes are all equal has
       no spread and flags nothing.
    """
    # One row per change: its window, padded with NaN where it is cut at the ends; the nan-
    # functions then ignore the padding. Vectorized because clip_levels runs this filter at
    # every forecast origin of every trial.
    count, width = len(changes), 2 * HALF_WIDTH + 1
    padded = np.pad(changes.astype(float), HALF_WIDTH, constant_values=np.nan)
    windows = sliding_window_view(padded, width)
    profile = np.zeros(count)
    with warnings.catch_warnings():
        # A window edge can hold no member of a season position: an all-NaN slice, ignored.
        warnings.simplefilter("ignore", RuntimeWarning)
        if SEASONAL_PROFILE:
            # Each window member's season position, the window's median per position, and the
            # residues from it; the change's own position's median is its profile.
            offsets = np.arange(-HALF_WIDTH, HALF_WIDTH + 1)
            positions = (np.arange(count)[:, None] + offsets[None, :]) % SEASONAL_PERIOD
            medians = np.column_stack(
                [
                    np.nanmedian(np.where(positions == p, windows, np.nan), axis=1)
                    for p in range(SEASONAL_PERIOD)
                ]
            )
            windows = windows - medians[np.arange(count)[:, None], positions]
            profile = medians[np.arange(count), np.arange(count) % SEASONAL_PERIOD]
        median = np.nanmedian(windows, axis=1)
        deviations = np.abs(windows - median[:, None])
        mad = np.nanmedian(deviations, axis=1)
        mean_deviation = np.nanmean(deviations, axis=1)
    scale = np.where(mad > 0, MAD_TO_SD * mad, MEAN_AD_TO_SD * mean_deviation)
    spread = THRESHOLD * scale
    lower, upper = profile + median - spread, profile + median + spread
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
    profile = (
        f" The median and spread are taken after removing the window's profile by season "
        f"position (every {SEASONAL_PERIOD} {PERIOD}s), so a regular rhythm is not flagged."
        if SEASONAL_PROFILE
        else ""
    )
    lines = [
        f"Outliers: {PERIODLY} changes flagged by a Hampel filter",
        (
            f"History: {data[DATE_COLUMN].iloc[0]:{DATE_FORMAT}} to "
            f"{data[DATE_COLUMN].iloc[-1]:{DATE_FORMAT}} "
            f"(train and validation; the test {PERIOD}s are never examined here). A change is "
            f"flagged when it lies more than {THRESHOLD:g} robust standard deviations (1.4826 x "
            "the median absolute deviation) from the median of the changes within "
            f"{HALF_WIDTH} {PERIOD}s on either side.{profile} Each model's tuned 'outliers' "
            "setting decides whether flagged changes are left as they are, clipped to the "
            "bounds, or given a pulse regressor; forecasts are always scored against the actual "
            "data."
        ),
        "",
    ]
    dates = data[DATE_COLUMN].iloc[1:]
    width = max(8, len(f"{dates.iloc[0]:{DATE_FORMAT}}"))
    for column in VALUE_COLUMNS:
        changes = np.diff(data[column].to_numpy(dtype=float))
        lower, upper, flags = hampel(changes)
        lines.append(f"{column}: {int(flags.sum())} of {len(changes)} changes flagged")
        lines.append(f"  {PERIOD:{width}s} {'change':>9s} {'lower':>9s} {'upper':>9s}")
        for index in np.flatnonzero(flags):
            lines.append(
                f"  {dates.iloc[index]:{DATE_FORMAT}}  {changes[index]:9.4f} {lower[index]:9.4f} "
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
        axis.plot(
            dates, changes, color=SERIES_PALETTE[0], linewidth=1.2, label=f"{PERIODLY} change"
        )
        axis.scatter(
            dates[flags],
            changes[flags],
            color=CRITICAL_COLOR,
            s=28,
            zorder=3,
            label=f"flagged ({int(flags.sum())})",
        )
        axis.axhline(0, color=PRIMARY_INK, linewidth=0.8)
        axis.set_ylabel(f"Change ({UNIT})", color=SECONDARY_INK)
        axis.set_title(
            f"{series_title(column)}: {PERIODLY} changes outside {THRESHOLD:g} robust standard "
            f"deviations of their {2 * HALF_WIDTH + 1}-{PERIOD} window are flagged",
            loc="left",
            fontsize=10,
        )
        axis.legend(fontsize=8.5, frameon=False, loc="upper left")
    figure.tight_layout()
    return figure


@app.command()
def main(levels_path: Path = LEVELS_PATH) -> None:
    """
    Flag the outliers of the train and validation history and write the report and figure.

    Parameters
    ----------
    levels_path : Path, default=LEVELS_PATH
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
    _, splits = load_levels(levels_path)
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
