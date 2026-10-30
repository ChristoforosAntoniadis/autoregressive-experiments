"""
VAR / VECM interpretation: does one series help predict the other, and how does
a shock to one feed through to both?

Runs on the two systems carried forward by the residual checks, VAR(5) on the first differences and
VECM(5) on the levels, fitted on the training split.

- Granger causality in both directions. VAR: Wald test that all lags of the causing series
  are zero in the caused series' equation. VECM: statsmodels' F-test, which also covers the
  long-run error-correction term.
- Orthogonalized impulse responses over 24 months, as responses of the **levels** (the VAR's
  cumulative responses, the VECM's direct ones), with 95% asymptotic bands for the VAR, and the
  forecast-error variance decomposition of the levels. Orthogonalizing needs an ordering: total
  assets first (the Fed sets its balance sheet and the monetary base largely follows), and the
  reverse ordering as a check of how much the conclusions depend on it.

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` and
``data/processed/fed_balance_sheet_first_difference.csv``, training split only.

Output, in ``reports/``:

- ``var_analysis.txt``: Granger tests, impulse responses at 1, 6, 12 and 24 months (with the
  VAR's bands), and variance shares at those horizons for both orderings
- ``figures/var_analysis.png``: Granger p-values against the 5% line; the 2 x 2 impulse responses;
  the share of each series' forecast uncertainty that comes from the other series' shocks
"""

from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from statsmodels.tsa.vector_ar.vecm import VECM, VECMResults
import typer

from autoregressive_experiments.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.diagnostics import SIGNIFICANCE_LEVEL, VALUE_COLUMNS, repo_path
from autoregressive_experiments.features import DATE_COLUMN, split_boundaries, split_by_time
from autoregressive_experiments.residual_checks import (
    BASELINE,
    CRITICAL_COLOR,
    PRIMARY_INK,
    SECONDARY_INK,
    SERIES_PALETTE,
    SURFACE,
    apply_chart_style,
)
from autoregressive_experiments.runlog import begin_phase, finish_run, start_run_log
from autoregressive_experiments.var_models import VECM_RANK

app = typer.Typer()

# Settings; each value's reasoning is in configs/hyperparameters.json.
# The lag order carried forward by the residual checks.
CARRIED_LAGS = HYPERPARAMETERS.var_analysis.carried_lags
# Months the impulse responses and variance shares run after the shock, and those reported.
HORIZON = HYPERPARAMETERS.var_analysis.horizon
REPORT_HORIZONS = HYPERPARAMETERS.var_analysis.report_horizons
# Normal quantile of the 95% bands: value +- BAND_Z standard errors.
BAND_Z = HYPERPARAMETERS.var_analysis.band_z

REPORT_PATH = REPORTS_DIR / "var_analysis.txt"
FIGURE_PATH = FIGURES_DIR / "var_analysis.png"

# Reference palette slots 1-3 (validated together).
VAR_COLOR, VECM_COLOR, REVERSED_COLOR = SERIES_PALETTE[:3]
SHORT_NAMES = {"fed_total_assets_T": "total assets", "monetary_base_T": "monetary base"}


def possessive(name: str) -> str:
    """
    Make the English possessive of a name.

    Parameters
    ----------
    name : str
        A short series name, e.g. "total assets" or "monetary base".

    Returns
    -------
    str
        The name with "'" if it ends in "s" ("total assets'"), otherwise with "'s"
        ("monetary base's").
    """
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def fit_systems(
    levels: pd.DataFrame, differences: pd.DataFrame
) -> dict[str, VARResultsWrapper | VECMResults]:
    """
    Fit the two systems carried forward by the residual checks.

    VAR(5) on the differences and VECM(5) on the levels, in the frames' column order.

    Parameters
    ----------
    levels : pd.DataFrame
        Levels of both series, shape (months, 2), indexed by month-end date with a monthly
        frequency; trillions USD. The column order is the ordering used to orthogonalize
        the shocks (first column first).
    differences : pd.DataFrame
        First differences, same columns in the same order, one row fewer.

    Returns
    -------
    dict of str to VARResultsWrapper or VECMResults
        Two entries, VAR first: "VAR(5)", the VAR fitted with a constant by OLS on the
        differences, and "VECM(5)", the rank-VECM_RANK VECM with 5 lagged differences and a
        constant outside the long-run relation, fitted by maximum likelihood on the levels.

    Notes
    -----
    Same specifications as var_models' candidates, so the analysis interprets exactly the models
    the residual checks carried forward. The column order matters only for orthogonalized impulse
    responses (the Cholesky factor follows it), which is why main refits with the columns reversed.
    """
    return {
        f"VAR({CARRIED_LAGS})": VAR(differences).fit(CARRIED_LAGS, trend="c"),
        f"VECM({CARRIED_LAGS})": VECM(
            levels, k_ar_diff=CARRIED_LAGS, coint_rank=VECM_RANK, deterministic="co"
        ).fit(),
    }


def granger_tests(
    systems: dict[str, VARResultsWrapper | VECMResults], columns: list[str]
) -> pd.DataFrame:
    """
    Run Granger-causality tests in both directions for each system.

    Each system, each direction: does the causing series help predict the caused one?

    Parameters
    ----------
    systems : dict of str to VARResultsWrapper or VECMResults
        The fitted systems from fit_systems. Names starting with "VAR" are treated as VARs,
        the rest as VECMs.
    columns : list of str
        The two series names, in the systems' column order.

    Returns
    -------
    pd.DataFrame
        One row per (system, direction), 4 in all, with columns "model", "causing", "caused",
        "statistic" (the test statistic: chi-squared for the VAR, F for the VECM) and
        "p_value".

    Notes
    -----
    Processing, for each system and each direction:
    1. VAR: statsmodels' test_causality, a Wald test that every lag of the causing series is
       zero in the caused series' equation.
    2. VECM: statsmodels' test_granger_causality, an F-test on the VECM's levels VAR
       representation (Lütkepohl 2005, 7.6.3). Its restriction covers the long-run
       error-correction term as well as the lagged differences, since in a VECM the other
       series can also act through the long-run relation.
    3. Collect the statistic and p-value.

    "Granger causes" means only "helps predict": past values of one series carry
    information about the other beyond its own past. It says nothing about economic cause.
    """
    rows = []
    for name, fitted in systems.items():
        # Unpacks [a, b] then [b, a]: both directions.
        for causing, caused in (columns, columns[::-1]):
            if name.startswith("VAR"):
                result = fitted.test_causality(caused, [causing], kind="wald")
            else:
                result = fitted.test_granger_causality(caused=caused, causing=[causing])
            rows.append(
                {
                    "model": name,
                    "causing": causing,
                    "caused": caused,
                    "statistic": float(result.test_statistic),
                    "p_value": float(result.pvalue),
                }
            )
    return pd.DataFrame(rows)


def level_responses(
    name: str, fitted: VARResultsWrapper | VECMResults
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute one system's orthogonalized impulse responses of the levels.

    Orthogonalized responses of the levels, [month, response, shock], and their standard
    errors (VAR only: statsmodels' VECM errors ignore the estimated long-run relation).

    Parameters
    ----------
    name : str
        The system's name; one starting with "VAR" means a VAR on the differences, anything
        else a VECM on the levels.
    fitted : VARResultsWrapper or VECMResults
        The fitted system, from fit_systems.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray or None)
        - responses: shape (HORIZON + 1, 2, 2), indexed [month, response, shock], months 0
          (impact) to HORIZON; the response of each series' level, in trillions USD, to a
          one-standard-deviation orthogonal shock.
        - standard errors: the asymptotic standard errors of the responses, same shape, for
          the VAR; None for the VECM.

    Notes
    -----
    Orthogonalized shocks come from the Cholesky factor of the residual covariance, so the
    first column's shock moves both series on impact and the second's moves only itself on
    impact. That makes the shocks uncorrelated with unit variance, at the price of depending
    on the column order.

    The VAR models the differences, so its plain responses are responses of the monthly
    changes; the response of the level after h months is their running total over months
    0..h, which statsmodels gives as the cumulative effects. The VECM is written in levels,
    so its responses already are level responses.
    """
    analysis = fitted.irf(HORIZON)
    if name.startswith("VAR"):
        # The VAR models the differences, so the level response is the running total.
        return analysis.orth_cum_effects, analysis.cum_effect_stderr(orth=True)
    return analysis.orth_irfs, None


def variance_decomposition(responses: np.ndarray) -> np.ndarray:
    """
    Compute the forecast-error variance decomposition from orthogonalized responses.

    Share of each variable's h-step forecast-error variance that comes from each orthogonal
    shock: shares[h - 1, variable, shock] for h = 1..len(responses).

    The h-step error is the sum over s = 0..h-1 of responses[s] times that month's shocks, which
    are uncorrelated with unit variance, so each shock contributes its squared responses.

    Parameters
    ----------
    responses : np.ndarray
        Orthogonalized responses, shape (H, k, k), indexed [month, variable, shock], months 0
        to H - 1; level responses (level_responses) give the decomposition of the levels.

    Returns
    -------
    np.ndarray
        Shares, shape (H, k, k), indexed [h - 1, variable, shock] for horizons h = 1..H; each
        [h - 1, variable, :] sums to 1.

    Notes
    -----
    Formula: with Theta_s = responses[s], the h-step forecast-error variance of variable i is
    sum over s = 0..h-1 and shocks j of Theta_s[i, j]^2, and shock j's share is its part of
    that sum:

        share[h - 1, i, j] = sum_s Theta_s[i, j]^2 / sum_s sum_j' Theta_s[i, j']^2

    Processing:
    1. Cumulative sums of the squared responses over months give each shock's contribution
       at every horizon.
    2. Dividing by the sum over shocks turns contributions into shares.

    One function serves both systems (statsmodels has a fevd() for the VAR only, and the
    VAR's would decompose the differences, not the levels). Checked against statsmodels'
    VAR fevd() in the tests.
    """
    # Axis 0 is the month: cumsum gives, at index h - 1, the sum over months 0..h-1.
    contributions = np.cumsum(responses**2, axis=0)
    # Axis 2 is the shock: dividing by the total over shocks gives each shock's share.
    return contributions / contributions.sum(axis=2, keepdims=True)


def analysis_report(
    granger: pd.DataFrame,
    responses: dict[str, tuple[np.ndarray, np.ndarray | None]],
    shares: dict[tuple[str, str], np.ndarray],
    period: str,
    columns: list[str],
) -> str:
    """
    Write the interpretation text report.

    Parameters
    ----------
    granger : pd.DataFrame
        The Granger tests, from granger_tests.
    responses : dict of str to tuple of (np.ndarray, np.ndarray or None)
        Per system name, its level responses and standard errors (level_responses), with
        total assets ordered first.
    shares : dict of tuple of (str, str) to np.ndarray
        Variance shares (variance_decomposition), keyed by (system name, ordering), where the
        ordering is "total assets first" or "monetary base first". Every array is indexed
        [h - 1, variable, shock] in `columns` order, whatever the ordering.
    period : str
        Description of the training period, e.g. "2002-12 to 2016-10".
    columns : list of str
        The two series names, in the main ordering.

    Returns
    -------
    str
        The report, lines joined by newlines: the setup, each Granger test with its verdict,
        the impulse responses at REPORT_HORIZONS (starred where the VAR's 95% band excludes
        zero), and the share of each series' forecast-error variance from the other series'
        shocks for every (system, ordering).
    """
    lines = [
        "VAR / VECM interpretation: Granger causality, impulse responses and FEVD",
        (
            f"Training split only: {period}. Systems carried forward by the residual checks: "
            f"VAR({CARRIED_LAGS}) on the differences, VECM({CARRIED_LAGS}) on the levels."
        ),
        (
            "Granger: VAR, Wald chi-squared test; VECM, F-test including the error-correction "
            f"term. Null: the causing series doesn't help predict the caused one. Significance "
            f"level {SIGNIFICANCE_LEVEL:.0%}."
        ),
        (
            "Impulse responses: orthogonalized, one-standard-deviation shocks, responses of the "
            "levels in trillions (the VAR's cumulative responses). 95% asymptotic bands for the "
            "VAR only."
        ),
        (
            "Variance shares: of each series' level forecast-error variance, from its own and "
            "the other series' shocks. Ordering: total assets first (main), and reversed."
        ),
        "",
        "Granger causality",
    ]
    for row in granger.itertuples():
        verdict = "helps predict" if row.p_value < SIGNIFICANCE_LEVEL else "no significant help"
        lines.append(
            f"  {row.model:<9} {SHORT_NAMES[row.causing]:>14} -> {SHORT_NAMES[row.caused]:<14} "
            f"statistic {row.statistic:8.3f}, p={row.p_value:.4f}: {verdict}"
        )
    lines.append("")
    lines.append("Impulse responses of the levels (trillions), total assets ordered first")
    for name, (values, errors) in responses.items():
        lines.append(f"  {name}")
        for shock_index, shock in enumerate(columns):
            for response_index, response in enumerate(columns):
                cells = []
                for month in REPORT_HORIZONS:
                    # Index = months after the shock (0 is the impact month).
                    value = values[month, response_index, shock_index]
                    if errors is None:
                        cells.append(f"{month}m {value:+.4f}")
                    else:
                        error = errors[month, response_index, shock_index]
                        # The band value +- 1.96 x error excludes zero exactly when this holds.
                        excludes_zero = abs(value) > BAND_Z * error
                        cells.append(f"{month}m {value:+.4f}{'*' if excludes_zero else ' '}")
                lines.append(
                    f"    {SHORT_NAMES[shock]:>14} shock -> {SHORT_NAMES[response]:<14} "
                    + "  ".join(cells)
                )
    lines.append("  * = the 95% band excludes zero (VAR only)")
    lines.append("")
    lines.append("Share of each series' forecast-error variance from the other series' shocks")
    for (name, ordering), values in shares.items():
        cells = []
        for response_index, response in enumerate(columns):
            # With two series, the other series' index is 1 - this one's.
            other = 1 - response_index
            months = ", ".join(
                # Shares are indexed by h - 1: row 0 is the 1-month horizon.
                f"{month}m {100 * values[month - 1, response_index, other]:.0f}%"
                for month in REPORT_HORIZONS
            )
            cells.append(
                f"    {SHORT_NAMES[response]:<14} from {SHORT_NAMES[columns[other]]}: {months}"
            )
        lines.append(f"  {name}, {ordering}")
        lines.extend(cells)
    lines.append("")
    lines.append(f"Figure: {repo_path(FIGURE_PATH)}")
    return "\n".join(lines)


def analysis_figure(
    granger: pd.DataFrame,
    responses: dict[str, tuple[np.ndarray, np.ndarray | None]],
    shares: dict[tuple[str, str], np.ndarray],
    columns: list[str],
) -> Figure:
    """
    Draw the interpretation figure: Granger tests, impulse responses and variance shares.

    Parameters
    ----------
    granger : pd.DataFrame
        The Granger tests, from granger_tests.
    responses : dict of str to tuple of (np.ndarray, np.ndarray or None)
        Level responses and standard errors per system, as in analysis_report. Exactly two
        entries, the VAR first (its standard errors are used for the bands), the VECM second.
    shares : dict of tuple of (str, str) to np.ndarray
        Variance shares keyed by (system name, ordering), as in analysis_report; must hold
        both systems with "total assets first" and the VAR with "monetary base first".
    columns : list of str
        The two series names, in the main ordering.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, not yet saved; the caller saves and closes it.

    Notes
    -----
    Processing:
    1. Top row: each Granger p-value on a log scale against the 5% line; significant ones in
       the data colour, the rest grey.
    2. Middle 2 x 2: for each (shock column, response row), the VAR's level response with its
       95% band and the VECM's response without one.
    3. Bottom row: for each series, the share of its forecast-error variance from the other
       series' shocks by horizon, for the VAR and the VECM with total assets first and the VAR
       reversed, to show how much the ordering matters.
    """
    apply_chart_style()
    figure = plt.figure(figsize=(14, 15.5), facecolor=SURFACE)
    grid = figure.add_gridspec(4, 2, height_ratios=(0.8, 1, 1, 1))
    # Months 0 (impact) to HORIZON, matching the responses' first axis.
    months = np.arange(HORIZON + 1)

    granger_axis = figure.add_subplot(grid[0, :], facecolor=SURFACE)
    labels = [
        f"{row.model}: {SHORT_NAMES[row.causing]} -> {SHORT_NAMES[row.caused]}"
        for row in granger.itertuples()
    ]
    # Reversed so the first test is drawn at the top.
    positions = np.arange(len(granger))[::-1]
    significant = granger["p_value"].to_numpy() < SIGNIFICANCE_LEVEL
    granger_axis.scatter(
        granger["p_value"],
        positions,
        s=60,
        zorder=3,
        color=np.where(significant, VAR_COLOR, BASELINE),
    )
    for position, row in zip(positions, granger.itertuples()):
        verdict = "helps predict" if row.p_value < SIGNIFICANCE_LEVEL else "no significant help"
        granger_axis.annotate(
            f"  p={row.p_value:.3g}: {verdict}",
            (row.p_value, position),
            va="center",
            color=SECONDARY_INK,
            fontsize=9,
        )
    granger_axis.axvline(SIGNIFICANCE_LEVEL, color=CRITICAL_COLOR, linewidth=1)
    granger_axis.annotate(
        " 5%",
        (SIGNIFICANCE_LEVEL, 1),
        xycoords=("data", "axes fraction"),
        va="bottom",
        color=SECONDARY_INK,
        fontsize=9,
    )
    granger_axis.set_xscale("log")
    granger_axis.set_xlim(
        min(1e-6, granger["p_value"].min() / 3), 1e4
    )  # room for the labels right of the points
    granger_axis.set_yticks(positions, labels)
    granger_axis.set_xlabel(
        "Granger p-value (left of the line: one series helps predict the other)"
    )
    granger_axis.set_title("Granger causality, both directions", loc="left", fontsize=10)

    # fit_systems puts the VAR first, so the first entry is the one with standard errors.
    var_name, vecm_name = list(responses)
    var_values, var_errors = responses[var_name]
    vecm_values, _ = responses[vecm_name]
    for shock_index, shock in enumerate(columns):
        for response_index, response in enumerate(columns):
            axis = figure.add_subplot(grid[1 + response_index, shock_index], facecolor=SURFACE)
            value = var_values[:, response_index, shock_index]
            # mypy can't see that the VAR's standard errors are never None (only the VECM's).
            error = var_errors[:, response_index, shock_index]  # type: ignore[index]  # see above
            axis.fill_between(
                months,
                value - BAND_Z * error,
                value + BAND_Z * error,
                color=VAR_COLOR,
                alpha=0.15,
                linewidth=0,
                label=f"{var_name} 95% band",
            )
            axis.plot(months, value, color=VAR_COLOR, linewidth=2, label=var_name)
            axis.plot(
                months,
                vecm_values[:, response_index, shock_index],
                color=VECM_COLOR,
                linewidth=2,
                label=f"{vecm_name} (no band)",
            )
            axis.axhline(0, color=PRIMARY_INK, linewidth=1)
            axis.set_xticks([0, 6, 12, 18, 24])
            axis.set_xlabel("Months after the shock")
            axis.set_ylabel("Response of the level (trillions)")
            axis.set_title(
                f"{SHORT_NAMES[shock]} shock -> {SHORT_NAMES[response]}\n"
                "band clear of zero: a real response",
                loc="left",
                fontsize=10,
            )
            if response_index == 0 and shock_index == 0:
                axis.legend(loc="upper left", fontsize=8, frameon=False)

    styles = {
        (var_name, "total assets first"): VAR_COLOR,
        (vecm_name, "total assets first"): VECM_COLOR,
        (var_name, "monetary base first"): REVERSED_COLOR,
    }
    for response_index, response in enumerate(columns):
        axis = figure.add_subplot(grid[3, response_index], facecolor=SURFACE)
        other = 1 - response_index
        for key, color in styles.items():
            values = shares[key][:, response_index, other]
            axis.plot(
                # Share row h - 1 is horizon h; the last row (HORIZON + 1) is cut off by xlim.
                np.arange(1, HORIZON + 2),
                100 * values,
                color=color,
                linewidth=2,
                label=f"{key[0]}, {key[1]}",
            )
        axis.set_ylim(0, 100)
        axis.set_xticks([1, 6, 12, 18, 24])
        axis.set_xlim(1, HORIZON)
        axis.set_xlabel("Forecast horizon (months)")
        axis.set_ylabel("% of forecast-error variance")
        axis.set_title(
            f"Share of {possessive(SHORT_NAMES[response])} forecast uncertainty\n"
            f"that comes from {SHORT_NAMES[columns[other]]} shocks",
            loc="left",
            fontsize=10,
        )
        if response_index == 0:
            axis.legend(loc="upper left", fontsize=8, frameon=False)
    figure.tight_layout()
    return figure


@app.command()
def main(
    levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv",
    differences_path: Path = PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv",
) -> None:
    """
    Run the Granger tests, impulse responses and FEVD on the training split and write the report
    and figure.

    Parameters
    ----------
    levels_path : Path, default=INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv"
        The cleaned levels, from ``make features``.
    differences_path : Path, default=PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv"
        The first differences, from ``make features``.

    Returns
    -------
    None
        Writes REPORT_PATH and FIGURE_PATH, and prints the report.

    Raises
    ------
    FileNotFoundError
        If either input file is missing.

    Notes
    -----
    Processing:
    1. Load both files, keep only the training split, and index them by month-end date.
    2. Fit VAR(5) and VECM(5) with total assets first.
    3. Granger tests in both directions.
    4. Level impulse responses and their variance shares.
    5. Refit both systems with the columns reversed, compute their variance shares, and swap
       them back onto the main ordering's axes.
    6. Write and print the report; draw the figure.

    The reverse ordering is a robustness check: orthogonalized shocks depend on which series
    comes first, so a conclusion that holds under both orderings doesn't rest on that choice.
    """
    start_run_log("var-analysis")
    begin_phase("Granger causality, impulse responses and FEVD")
    for path in (levels_path, differences_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Generate it with make features.")
    levels = pd.read_csv(levels_path, parse_dates=[DATE_COLUMN])
    boundaries = split_boundaries(levels)
    columns = list(VALUE_COLUMNS)
    # Month-end frequency, as in var_models: a missing month would surface as NaN rows.
    train_levels = split_by_time(levels, boundaries)["train"].set_index(DATE_COLUMN)[columns]
    train_differences = split_by_time(
        pd.read_csv(differences_path, parse_dates=[DATE_COLUMN]), boundaries
    )["train"].set_index(DATE_COLUMN)[columns]
    train_levels, train_differences = train_levels.asfreq("ME"), train_differences.asfreq("ME")
    period = f"{train_levels.index.min():%Y-%m} to {train_levels.index.max():%Y-%m}"

    systems = fit_systems(train_levels, train_differences)
    granger = granger_tests(systems, columns)
    responses = {name: level_responses(name, fitted) for name, fitted in systems.items()}

    # The reverse ordering refits the same systems with the columns swapped; swapping the
    # resulting shares back puts them on the main ordering's axes.
    reversed_systems = fit_systems(train_levels[columns[::-1]], train_differences[columns[::-1]])
    shares = {
        (name, "total assets first"): variance_decomposition(values)
        for name, (values, _) in responses.items()
    }
    for name, fitted in reversed_systems.items():
        values, _ = level_responses(name, fitted)
        # Reverse the variable and shock axes: index 0 is total assets again, as in the main run.
        shares[(name, "monetary base first")] = variance_decomposition(values)[:, ::-1, ::-1]

    report = analysis_report(granger, responses, shares, period, columns)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = analysis_figure(granger, responses, shares, columns)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"VAR / VECM analysis written to {REPORT_PATH} and {FIGURE_PATH}")
    finish_run()


if __name__ == "__main__":
    app()
