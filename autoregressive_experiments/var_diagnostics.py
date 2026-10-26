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
from typing import Any

from loguru import logger
from matplotlib.figure import Figure
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
    SERIES_PALETTE,
    SURFACE,
    apply_chart_style,
)

app = typer.Typer()

# Lag orders tried in V2: 0..10 months. At 10 lags each equation has 20 lag coefficients plus
# the constant, estimated from 156 usable training months.
MAX_VAR_LAGS = 10
CRITERIA = ("aic", "bic", "hqic")
# BIC drives the Johansen decision: it's consistent (picks the true order as the sample grows),
# which suits a question about the model's structure; AIC targets forecast accuracy instead.
PRIMARY_CRITERION = "bic"
# coint_johansen's critical-value columns are the 90%, 95% and 99% levels.
CRITICAL_COLUMN = {0.10: 0, 0.05: 1, 0.01: 2}[SIGNIFICANCE_LEVEL]
# det_order=0: a constant in the error-correction model, so the levels may drift (both trend up).
JOHANSEN_DETERMINISTIC = 0

REPORT_PATH = REPORTS_DIR / "var_diagnostics.txt"
FIGURE_PATH = FIGURES_DIR / "var_diagnostics.png"

# Reference palette, categorical slots 1-3 (validated together) for AIC, BIC and HQIC.
CRITERION_COLORS = dict(zip(("aic", "bic", "hqic"), SERIES_PALETTE))


def select_var_lags(differences: np.ndarray) -> tuple[dict[str, int], pd.DataFrame]:
    """
    Choose the VAR lag order by AIC, BIC and HQIC (step V2).

    Lag order chosen by each criterion, and every criterion's value for 0..MAX_VAR_LAGS lags,
    all fitted on the same sample so the values are comparable.

    Parameters
    ----------
    differences : np.ndarray
        First differences of both series, shape (months, 2), in VALUE_COLUMNS order, trillions
        USD per month. Training split only.

    Returns
    -------
    tuple of (dict of str to int, pd.DataFrame)
        - selected: maps each criterion ("aic", "bic", "hqic") to the lag order that minimizes
          it.
        - values: one row per lag order 0..MAX_VAR_LAGS (index named "lags"), one column per
          criterion, holding the criterion's value (lower is better).

    Notes
    -----
    Processing:
    1. Fit VARs with a constant (drift in the levels) for 0..MAX_VAR_LAGS lags with
       statsmodels' select_order.
    2. Read each criterion's minimizing order and its value at every order.

    select_order drops the first MAX_VAR_LAGS months from every fit, not only from the long
    ones, so all orders use the same months. Otherwise a shorter model would be scored on more
    data, and the criteria (sums over the sample) wouldn't be comparable across orders.
    """
    selection = VAR(differences).select_order(maxlags=MAX_VAR_LAGS, trend="c")
    selected = {criterion: int(selection.selected_orders[criterion]) for criterion in CRITERIA}
    values = pd.DataFrame({criterion: selection.ics[criterion] for criterion in CRITERIA})
    return selected, values.rename_axis("lags")


def plural(count: int, noun: str) -> str:
    """
    Write a count with its noun, adding "s" unless the count is 1.

    Parameters
    ----------
    count : int
        How many.
    noun : str
        The singular noun, e.g. "lag" or "lagged difference".

    Returns
    -------
    str
        For example "1 lag" or "3 lags".
    """
    return f"{count} {noun}{'' if count == 1 else 's'}"


def johansen_rank(statistics: np.ndarray, critical_values: np.ndarray) -> int:
    """
    Read the cointegrating rank off a sequence of Johansen tests.

    Sequential testing: the first rank r whose null "rank <= r" isn't rejected.

    Parameters
    ----------
    statistics : np.ndarray
        Trace or maximum-eigenvalue statistics, shape (number of series,); entry r tests the
        null "rank <= r".
    critical_values : np.ndarray
        The matching critical values at SIGNIFICANCE_LEVEL, same shape.

    Returns
    -------
    int
        The cointegrating rank, from 0 (no cointegration) to the number of series (every null
        rejected: full rank, the levels themselves look stationary).

    Notes
    -----
    The nulls are nested (rank <= 0, rank <= 1, ...), so testing goes upwards and stops at the
    first null that survives: that's the smallest rank the data don't reject. Starting from
    rank 0 keeps the test from finding cointegration unless the data demand it.
    """
    for rank, (statistic, critical) in enumerate(zip(statistics, critical_values)):
        # Below the critical value: "rank <= r" isn't rejected, so r is the rank.
        if statistic < critical:
            return rank
    return len(statistics)


def johansen_test(levels: np.ndarray, lagged_differences: int) -> dict[str, Any]:
    """
    Run the Johansen cointegration test on the levels (step V3).

    Parameters
    ----------
    levels : np.ndarray
        Levels of both series, shape (months, 2), in VALUE_COLUMNS order, trillions USD.
        Training split only, undifferenced.
    lagged_differences : int
        Number of lagged differences in the test's error-correction model (k_ar_diff). A VAR
        on the differences with p lags matches a VECM with p lagged differences, so this is a
        lag order chosen in V2.

    Returns
    -------
    dict of str to Any
        - "lagged_differences": int, the value passed in.
        - "trace": np.ndarray, shape (2,), the trace statistics for rank <= 0 and rank <= 1.
        - "trace_critical": np.ndarray, shape (2,), their critical values at
          SIGNIFICANCE_LEVEL.
        - "trace_rank": int, the rank by the trace test (johansen_rank).
        - "max_eigen": np.ndarray, shape (2,), the maximum-eigenvalue statistics.
        - "max_eigen_critical": np.ndarray, shape (2,), their critical values.
        - "max_eigen_rank": int, the rank by the maximum-eigenvalue test.
        - "vector": np.ndarray, shape (2,), the most stationary combination of the two
          levels, scaled so the first series' weight is 1.

    Notes
    -----
    Processing:
    1. Run statsmodels' coint_johansen with det_order JOHANSEN_DETERMINISTIC (0: a constant,
       so the levels may drift) and the given number of lagged differences.
    2. Pick the critical values at SIGNIFICANCE_LEVEL from its 90 / 95 / 99% columns.
    3. Read both tests' ranks with johansen_rank.
    4. Take the first eigenvector and scale it to 1 on the first series.

    The test runs on the undifferenced levels because cointegration is a property of the
    levels: a combination of two I(1) series that is itself I(0). The trace test's null is
    "rank <= r" against "rank > r"; the maximum-eigenvalue test's is "rank = r" against
    "rank = r + 1". The eigenvectors come sorted by eigenvalue, largest first, so the first one
    is the combination most correlated with the changes, i.e. the most stationary one.
    """
    result = coint_johansen(levels, JOHANSEN_DETERMINISTIC, lagged_differences)
    # Column CRITICAL_COLUMN of the (series x 3) critical-value tables, for each null r.
    trace_critical = result.cvt[:, CRITICAL_COLUMN]
    max_eigen_critical = result.cvm[:, CRITICAL_COLUMN]
    # The first eigenvector is the most stationary combination; scale it to 1 on the first series.
    # np.real: the eigen solver can return complex numbers with a zero imaginary part.
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
    """
    Translate a cointegrating rank into the modelling decision it implies.

    Parameters
    ----------
    rank : int
        The Johansen rank for two series: 0, 1 or 2.

    Returns
    -------
    str
        The decision, as a phrase for the report: rank 0, a VAR on the first differences;
        rank 1, a VECM; rank 2 (full), a warning, since full rank means the levels are
        stationary, which contradicts the I(1) finding of step U1.
    """
    if rank == 0:
        return "no cointegration: model the first differences with a VAR"
    if rank == 1:
        return "one cointegrating relation: model the levels with a VECM (rank 1)"
    return (
        "full rank: both series look stationary in levels, which contradicts U1; check the setup"
    )


def var_report(
    selected: dict[str, int],
    criteria: pd.DataFrame,
    tests: dict[int, dict[str, Any]],
    periods: dict[str, str],
) -> str:
    """
    Write the V2 / V3 text report.

    Parameters
    ----------
    selected : dict of str to int
        Each criterion's chosen lag order, from select_var_lags.
    criteria : pd.DataFrame
        Each criterion's value by lag order (index "lags", 0..MAX_VAR_LAGS; one column per
        criterion), from select_var_lags.
    tests : dict of int to dict
        Johansen results (johansen_test) keyed by the number of lagged differences, one per
        distinct selected lag order, in ascending order.
    periods : dict of str to str
        Descriptions of the training periods used, under the keys "levels" and "differences",
        e.g. "2002-12 to 2016-10 (167 months)".

    Returns
    -------
    str
        The report, lines joined by newlines: the setup, the criteria table with each
        criterion's best lag starred, every Johansen test row with its verdict, the ranks and
        the most stationary combination for each lag order, and the decision.

    Notes
    -----
    Processing:
    1. Describe the setup and the caveat (Johansen assumes a constant error variance).
    2. Tabulate the criteria by lag, marking each criterion's selected order with "*".
    3. For each tested lag order, list which criteria chose it and both tests' rows.
    4. State the decision from BIC's lag order and the trace test, flagging it when the
       maximum-eigenvalue test disagrees.

    The trace test drives the decision, as is conventional (it tends to hold up better than
    the maximum-eigenvalue test when errors aren't normal); the maximum-eigenvalue rank is
    reported as a check.
    """
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
        # Each value is followed by "*" when this row is the criterion's best order, else a space
        # so the columns stay aligned.
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
        # Several criteria can pick the same order, so list every one that chose it.
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
    selected: dict[str, int], criteria: pd.DataFrame, test: dict[str, Any], levels: pd.DataFrame
) -> Figure:
    """
    Draw the V2 / V3 figure in three panels.

    Information criteria by lag, Johansen statistics against critical values, and the most
    stationary combination of the two series over time.

    Parameters
    ----------
    selected : dict of str to int
        Each criterion's chosen lag order, from select_var_lags.
    criteria : pd.DataFrame
        Each criterion's value by lag order (index "lags"; one column per criterion).
    test : dict of str to Any
        The Johansen result (johansen_test) at BIC's lag order, the one that drives the
        decision.
    levels : pd.DataFrame
        The training-split levels, with DATE_COLUMN and the VALUE_COLUMNS (trillions USD).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, not yet saved; the caller saves and closes it.

    Notes
    -----
    Processing:
    1. Left: each criterion minus its own minimum, by lag, so the three criteria share one
       scale and each best lag sits at 0 (circled).
    2. Middle: one bar per Johansen null (trace and maximum eigenvalue, rank <= 0 and
       rank <= 1) with a tick at its critical value; a bar past its tick rejects that null.
    3. Right: the most stationary combination of the levels (first series plus the scaled
       weight times the second) over time, with its mean. If the series were cointegrated it
       would hover around a level; if not, it can wander.

    The criteria are shown as distances because their raw values sit at different levels
    (each adds its own penalty per parameter), while only the position of each minimum
    matters for the decision.
    """
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
    # Reversed so the first row (trace, rank <= 0) is drawn at the top.
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
    # 1.6x the longest bar or tick leaves room for the annotations to their right.
    test_axis.set_xlim(0, max(max(r[1], r[2]) for r in rows) * 1.6)
    test_axis.set_xlabel("Statistic (bar) and 5% critical value (black tick)")
    test_axis.set_title(
        f"V3: Johansen, {plural(test['lagged_differences'], 'lagged difference')} (BIC)\n"
        "bar past the tick: reject that null",
        loc="left",
        fontsize=10,
    )

    vector = test["vector"]
    # vector[0] is 1 after scaling, so the combination is total assets + vector[1] x base.
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
    """
    Run steps V2 and V3 on the training split and write the report and figure.

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
    1. Load both files and keep only the training split. The boundaries come from the levels,
       so levels and differences are cut at the same months.
    2. V2: choose the VAR lag order on the differences by each criterion.
    3. V3: run Johansen on the levels once per distinct selected lag order.
    4. Write and print the report; draw the figure for BIC's lag order.
    """
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
    # One test per distinct order: criteria that agree share a test, and the report can show
    # whether the rank depends on the lag choice.
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
