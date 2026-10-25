"""
VAR / VECM interpretation, steps V7 and V8: does one series help predict the other, and how does
a shock to one feed through to both?

Runs on the two systems carried forward from step V6, VAR(5) on the first differences and VECM(5)
on the levels, fitted on the training split.

- V7: Granger causality in both directions. VAR: Wald test that all lags of the causing series
  are zero in the caused series' equation. VECM: statsmodels' F-test, which also covers the
  long-run error-correction term.
- V8: orthogonalized impulse responses over 24 months, as responses of the **levels** (the VAR's
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
import typer

from autoregressive_experiments.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from autoregressive_experiments.diagnostics import SIGNIFICANCE_LEVEL, VALUE_COLUMNS, repo_path
from autoregressive_experiments.features import DATE_COLUMN, split_boundaries, split_by_time
from autoregressive_experiments.residual_checks import (
    BASELINE,
    CRITICAL_COLOR,
    PRIMARY_INK,
    SECONDARY_INK,
    SURFACE,
    apply_chart_style,
)
from autoregressive_experiments.var_models import VECM_RANK

app = typer.Typer()

# The lag order carried forward from step V6 (the user's decision).
CARRIED_LAGS = 5
HORIZON = 24
REPORT_HORIZONS = (1, 6, 12, 24)
BAND_Z = 1.96

REPORT_PATH = REPORTS_DIR / "var_analysis.txt"
FIGURE_PATH = FIGURES_DIR / "var_analysis.png"

# Reference palette slots 1-3 (validated together).
VAR_COLOR, VECM_COLOR, REVERSED_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
SHORT_NAMES = {"fed_total_assets_T": "total assets", "monetary_base_T": "monetary base"}


def possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def fit_systems(levels: pd.DataFrame, differences: pd.DataFrame) -> dict:
    """VAR(5) on the differences and VECM(5) on the levels, in the frames' column order."""
    return {
        f"VAR({CARRIED_LAGS})": VAR(differences).fit(CARRIED_LAGS, trend="c"),
        f"VECM({CARRIED_LAGS})": VECM(
            levels, k_ar_diff=CARRIED_LAGS, coint_rank=VECM_RANK, deterministic="co"
        ).fit(),
    }


def granger_tests(systems: dict, columns: list[str]) -> pd.DataFrame:
    """Each system, each direction: does the causing series help predict the caused one?"""
    rows = []
    for name, fitted in systems.items():
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


def level_responses(name: str, fitted) -> tuple[np.ndarray, np.ndarray | None]:
    """Orthogonalized responses of the levels, [month, response, shock], and their standard
    errors (VAR only: statsmodels' VECM errors ignore the estimated long-run relation)."""
    analysis = fitted.irf(HORIZON)
    if name.startswith("VAR"):
        # The VAR models the differences, so the level response is the running total.
        return analysis.orth_cum_effects, analysis.cum_effect_stderr(orth=True)
    return analysis.orth_irfs, None


def variance_decomposition(responses: np.ndarray) -> np.ndarray:
    """Share of each variable's h-step forecast-error variance that comes from each orthogonal
    shock: shares[h - 1, variable, shock] for h = 1..len(responses).

    The h-step error is the sum over s = 0..h-1 of responses[s] times that month's shocks, which
    are uncorrelated with unit variance, so each shock contributes its squared responses."""
    contributions = np.cumsum(responses**2, axis=0)
    return contributions / contributions.sum(axis=2, keepdims=True)


def analysis_report(
    granger: pd.DataFrame, responses: dict, shares: dict, period: str, columns: list[str]
) -> str:
    lines = [
        "VAR / VECM interpretation, steps V7 (Granger causality) and V8 (impulse responses, FEVD)",
        (
            f"Training split only: {period}. Systems carried forward from V6: "
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
        "V7: Granger causality",
    ]
    for row in granger.itertuples():
        verdict = "helps predict" if row.p_value < SIGNIFICANCE_LEVEL else "no significant help"
        lines.append(
            f"  {row.model:<9} {SHORT_NAMES[row.causing]:>14} -> {SHORT_NAMES[row.caused]:<14} "
            f"statistic {row.statistic:8.3f}, p={row.p_value:.4f}: {verdict}"
        )
    lines.append("")
    lines.append("V8: impulse responses of the levels (trillions), total assets ordered first")
    for name, (values, errors) in responses.items():
        lines.append(f"  {name}")
        for shock_index, shock in enumerate(columns):
            for response_index, response in enumerate(columns):
                cells = []
                for month in REPORT_HORIZONS:
                    value = values[month, response_index, shock_index]
                    if errors is None:
                        cells.append(f"{month}m {value:+.4f}")
                    else:
                        error = errors[month, response_index, shock_index]
                        excludes_zero = abs(value) > BAND_Z * error
                        cells.append(f"{month}m {value:+.4f}{'*' if excludes_zero else ' '}")
                lines.append(
                    f"    {SHORT_NAMES[shock]:>14} shock -> {SHORT_NAMES[response]:<14} "
                    + "  ".join(cells)
                )
    lines.append("  * = the 95% band excludes zero (VAR only)")
    lines.append("")
    lines.append("V8: share of each series' forecast-error variance from the other series' shocks")
    for (name, ordering), values in shares.items():
        cells = []
        for response_index, response in enumerate(columns):
            other = 1 - response_index
            months = ", ".join(
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
    granger: pd.DataFrame, responses: dict, shares: dict, columns: list[str]
) -> plt.Figure:
    apply_chart_style()
    figure = plt.figure(figsize=(14, 15.5), facecolor=SURFACE)
    grid = figure.add_gridspec(4, 2, height_ratios=(0.8, 1, 1, 1))
    months = np.arange(HORIZON + 1)

    granger_axis = figure.add_subplot(grid[0, :], facecolor=SURFACE)
    labels = [
        f"{row.model}: {SHORT_NAMES[row.causing]} -> {SHORT_NAMES[row.caused]}"
        for row in granger.itertuples()
    ]
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
    granger_axis.set_title("V7: Granger causality, both directions", loc="left", fontsize=10)

    var_name, vecm_name = list(responses)
    var_values, var_errors = responses[var_name]
    vecm_values, _ = responses[vecm_name]
    for shock_index, shock in enumerate(columns):
        for response_index, response in enumerate(columns):
            axis = figure.add_subplot(grid[1 + response_index, shock_index], facecolor=SURFACE)
            value = var_values[:, response_index, shock_index]
            error = var_errors[:, response_index, shock_index]
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
                f"V8: {SHORT_NAMES[shock]} shock -> {SHORT_NAMES[response]}\n"
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
            f"V8: share of {possessive(SHORT_NAMES[response])} forecast uncertainty\n"
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
        shares[(name, "monetary base first")] = variance_decomposition(values)[:, ::-1, ::-1]

    report = analysis_report(granger, responses, shares, period, columns)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = analysis_figure(granger, responses, shares, columns)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"VAR / VECM analysis written to {REPORT_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    app()
