"""
VAR / VECM diagnostics, steps V2 and V3: how many lags, and do the two series share a long-run
equilibrium?

V2 fits vector autoregressions on the first differences of both series together, with a constant
(drift), for 0 to 10 lags on a common sample, and picks the lag order by AIC, BIC and HQIC. V3 runs
the Johansen cointegration test (trace and maximum eigenvalue) on the levels, with a constant and
the selected number of lagged differences. A cointegrating rank of 0 means a VAR on the
differences; a rank of 1 means a VECM, which also pulls the series back towards their long-run
relationship. BIC's lag order drives the decision (Johansen is about identification, which BIC
targets); the other criteria's orders are checked too, so a rank that depends on the lag choice is
visible.

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` and
``data/processed/fed_balance_sheet_first_difference.csv``, training split only.

Output, in ``reports/``:

- ``var_diagnostics.txt``: the information criteria by lag, the selected orders, and the Johansen
  statistics, critical values and rank for each selected order
- ``figures/var_diagnostics.png``: each criterion's distance from its best lag, the Johansen
  statistics against their 5% critical values, and the most stationary combination of the two
  series over time
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
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
    PRIMARY_INK,
    SECONDARY_INK,
    SERIES_COLOR,
    SURFACE,
    apply_chart_style,
)

app = typer.Typer()

MAX_VAR_LAGS = 10
CRITERIA = ("aic", "bic", "hqic")
PRIMARY_CRITERION = "bic"
# coint_johansen's critical-value columns are the 90%, 95% and 99% levels.
CRITICAL_COLUMN = {0.10: 0, 0.05: 1, 0.01: 2}[SIGNIFICANCE_LEVEL]
# det_order=0: a constant in the error-correction model, so the levels may drift (both trend up).
JOHANSEN_DETERMINISTIC = 0

REPORT_PATH = REPORTS_DIR / "var_diagnostics.txt"
FIGURE_PATH = FIGURES_DIR / "var_diagnostics.png"

# Reference palette, categorical slots 1-3 (validated together) for AIC, BIC and HQIC.
CRITERION_COLORS = {"aic": "#2a78d6", "bic": "#eb6834", "hqic": "#1baf7a"}


def select_var_lags(differences: np.ndarray) -> tuple[dict[str, int], pd.DataFrame]:
    """Lag order chosen by each criterion, and every criterion's value for 0..MAX_VAR_LAGS lags,
    all fitted on the same sample so the values are comparable."""
    selection = VAR(differences).select_order(maxlags=MAX_VAR_LAGS, trend="c")
    selected = {criterion: int(selection.selected_orders[criterion]) for criterion in CRITERIA}
    values = pd.DataFrame({criterion: selection.ics[criterion] for criterion in CRITERIA})
    return selected, values.rename_axis("lags")


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def johansen_rank(statistics: np.ndarray, critical_values: np.ndarray) -> int:
    """Sequential testing: the first rank r whose null "rank <= r" isn't rejected."""
    for rank, (statistic, critical) in enumerate(zip(statistics, critical_values)):
        if statistic < critical:
            return rank
    return len(statistics)


def johansen_test(levels: np.ndarray, lagged_differences: int) -> dict:
    result = coint_johansen(levels, JOHANSEN_DETERMINISTIC, lagged_differences)
    trace_critical = result.cvt[:, CRITICAL_COLUMN]
    max_eigen_critical = result.cvm[:, CRITICAL_COLUMN]
    # The first eigenvector is the most stationary combination; scale it to 1 on the first series.
    vector = np.real(result.evec[:, 0])
    return {
        "lagged_differences": lagged_differences,
        "trace": result.lr1,
        "trace_critical": trace_critical,
        "trace_rank": johansen_rank(result.lr1, trace_critical),
        "max_eigen": result.lr2,
        "max_eigen_critical": max_eigen_critical,
        "max_eigen_rank": johansen_rank(result.lr2, max_eigen_critical),
        "vector": vector / vector[0],
    }


def decision(rank: int) -> str:
    if rank == 0:
        return "no cointegration: model the first differences with a VAR"
    if rank == 1:
        return "one cointegrating relation: model the levels with a VECM (rank 1)"
    return (
        "full rank: both series look stationary in levels, which contradicts U1; check the setup"
    )


def var_report(selected: dict, criteria: pd.DataFrame, tests: dict, periods: dict) -> str:
    lines = [
        "VAR / VECM diagnostics, steps V2 (lag order) and V3 (Johansen cointegration)",
        (
            f"Training split only: levels {periods['levels']}, first differences "
            f"{periods['differences']}. Series: {', '.join(VALUE_COLUMNS)}."
        ),
        (
            f"V2: VARs on the first differences with a constant, 0 to {MAX_VAR_LAGS} lags, all on "
            "the same sample; lower is better."
        ),
        (
            f"V3: Johansen on the levels with a constant (det_order={JOHANSEN_DETERMINISTIC}) and "
            "the selected number of lagged differences (a VAR on the differences with p lags is a "
            "VECM with p lagged differences). Null of each row: rank <= r. Critical values at "
            f"{SIGNIFICANCE_LEVEL:.0%}; the rank is the first r that isn't rejected."
        ),
        (
            "Caveat: Johansen assumes errors with a constant variance. The residuals have a "
            "variance break at 2008-09 and GARCH effects (steps U5, U5b), which can distort the "
            "test's size, so read borderline results with care."
        ),
        "",
        "V2: information criteria by number of lags",
        f"  {'lags':>4}" + "".join(f"{criterion.upper():>10}" for criterion in CRITERIA),
    ]
    for lags, row in criteria.iterrows():
        marks = "".join(
            f"{row[criterion]:9.4f}{'*' if lags == selected[criterion] else ' '}"
            for criterion in CRITERIA
        )
        lines.append(f"  {lags:>4} {marks}")
    lines.append(
        "  * = selected: "
        + ", ".join(f"{criterion.upper()} {selected[criterion]}" for criterion in CRITERIA)
    )
    lines.append("")
    lines.append("V3: Johansen cointegration test on the levels")
    for lags, test in tests.items():
        chosen_by = [c.upper() for c in CRITERIA if selected[c] == lags]
        primary = " (drives the decision)" if PRIMARY_CRITERION.upper() in chosen_by else ""
        lines.append(
            f"  {plural(lags, 'lagged difference')}, chosen by {', '.join(chosen_by)}{primary}"
        )
        for name, key in (("trace", "trace"), ("max eigenvalue", "max_eigen")):
            for rank, (statistic, critical) in enumerate(zip(test[key], test[f"{key}_critical"])):
                verdict = "reject" if statistic >= critical else "don't reject"
                lines.append(
                    f"    {name:<15} rank <= {rank}: statistic {statistic:8.3f}, "
                    f"critical {critical:7.3f} -> {verdict}"
                )
        lines.append(
            f"    Rank: trace {test['trace_rank']}, max eigenvalue {test['max_eigen_rank']}"
        )
        vector = test["vector"]
        lines.append(
            f"    Most stationary combination: {VALUE_COLUMNS[0]} {vector[1]:+.4f} x "
            f"{VALUE_COLUMNS[1]}"
        )
    primary_test = tests[selected[PRIMARY_CRITERION]]
    rank = primary_test["trace_rank"]
    agree = primary_test["max_eigen_rank"] == rank
    lines.append("")
    lines.append(
        f"Decision (BIC's lag order, trace test{'' if agree else '; max eigenvalue disagrees'}): "
        f"rank {rank}, {decision(rank)}."
    )
    lines.append(f"Figure: {repo_path(FIGURE_PATH)}")
    return "\n".join(lines)


def var_figure(
    selected: dict, criteria: pd.DataFrame, test: dict, levels: pd.DataFrame
) -> plt.Figure:
    """Information criteria by lag, Johansen statistics against critical values, and the most
    stationary combination of the two series over time."""
    apply_chart_style()
    figure, (lag_axis, test_axis, combination_axis) = plt.subplots(
        1, 3, figsize=(19, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": (1, 1, 1.3)}
    )
    for axis in (lag_axis, test_axis, combination_axis):
        axis.set_facecolor(SURFACE)

    for criterion in CRITERIA:
        distance = criteria[criterion] - criteria[criterion].min()
        color = CRITERION_COLORS[criterion]
        lag_axis.plot(
            distance.index,
            distance,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{criterion.upper()} (best: {plural(selected[criterion], 'lag')})",
        )
        lag_axis.scatter(
            [selected[criterion]], [0], s=120, facecolor="none", edgecolors=color, linewidths=2
        )
    lag_axis.set_xticks(range(MAX_VAR_LAGS + 1))
    lag_axis.set_xlabel("Lags in the VAR on the differences")
    lag_axis.set_ylabel("Distance from the criterion's best value")
    lag_axis.set_title(
        "V2: lag order\neach criterion's best lag sits at 0 (circled)", loc="left", fontsize=10
    )
    lag_axis.legend(loc="upper center", fontsize=9, frameon=False)

    rows = []
    for name, key in (("trace", "trace"), ("max eigenvalue", "max_eigen")):
        for rank, (statistic, critical) in enumerate(zip(test[key], test[f"{key}_critical"])):
            rows.append((f"{name}\nrank <= {rank}", statistic, critical))
    positions = np.arange(len(rows))[::-1]
    for position, (label, statistic, critical) in zip(positions, rows):
        test_axis.barh(position, statistic, color=SERIES_COLOR, height=0.5)
        test_axis.plot(
            [critical, critical],
            [position - 0.35, position + 0.35],
            color=PRIMARY_INK,
            linewidth=2.5,
        )
        verdict = "reject" if statistic >= critical else "don't reject"
        test_axis.annotate(
            f"  {statistic:.1f} vs {critical:.1f}: {verdict}",
            (max(statistic, critical), position),
            va="center",
            color=SECONDARY_INK,
            fontsize=9,
        )
    test_axis.set_yticks(positions, [row[0] for row in rows])
    test_axis.set_xlim(0, max(max(r[1], r[2]) for r in rows) * 1.6)
    test_axis.set_xlabel("Statistic (bar) and 5% critical value (black tick)")
    test_axis.set_title(
        f"V3: Johansen, {plural(test['lagged_differences'], 'lagged difference')} (BIC)\n"
        "bar past the tick: reject that null",
        loc="left",
        fontsize=10,
    )

    vector = test["vector"]
    combination = levels[VALUE_COLUMNS[0]] + vector[1] * levels[VALUE_COLUMNS[1]]
    combination_axis.plot(levels[DATE_COLUMN], combination, color=SERIES_COLOR, linewidth=1.5)
    combination_axis.axhline(combination.mean(), color=BASELINE, linewidth=1)
    combination_axis.set_ylabel("Trillions")
    meaning = (
        "cointegrated, so this should hover around a level"
        if test["trace_rank"]
        else "not cointegrated, so this needn't settle"
    )
    combination_axis.set_title(
        f"Most stationary combination: total assets {vector[1]:+.3f} x monetary base\n"
        f"rank {test['trace_rank']}: {meaning}",
        loc="left",
        fontsize=10,
    )
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
    train_levels = split_by_time(levels, boundaries)["train"]
    train_differences = split_by_time(
        pd.read_csv(differences_path, parse_dates=[DATE_COLUMN]), boundaries
    )["train"]
    periods = {
        name: f"{frame[DATE_COLUMN].min():%Y-%m} to {frame[DATE_COLUMN].max():%Y-%m} "
        f"({len(frame)} months)"
        for name, frame in (("levels", train_levels), ("differences", train_differences))
    }

    selected, criteria = select_var_lags(
        train_differences[list(VALUE_COLUMNS)].to_numpy(dtype=float)
    )
    level_values = train_levels[list(VALUE_COLUMNS)].to_numpy(dtype=float)
    tests = {lags: johansen_test(level_values, lags) for lags in sorted(set(selected.values()))}

    report = var_report(selected, criteria, tests, periods)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = var_figure(selected, criteria, tests[selected[PRIMARY_CRITERION]], train_levels)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"VAR diagnostics written to {REPORT_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    app()
