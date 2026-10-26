"""
VAR / VECM estimation and checks, steps V4 to V6: fit the candidate systems, check they're
stable, and check their residuals carry no autocorrelation left.

Candidates (the user's decisions): VARs on the first differences, with a constant, and a rank-1
VECM on the levels with a constant, even though Johansen found no cointegration (step V3), so
forecast accuracy can decide later whether a long-run link helps. Lag orders 1 (BIC) and 3 (AIC,
HQIC) come from step V2; both leave autocorrelated residuals, so by the roadmap's rule the order
was raised to 5, the smallest that passes. Only candidates that are stable and pass the residual
checks are carried forward to the evaluation: VAR(5) and VECM(5).

- V4: estimate each candidate on the training split (OLS per equation for the VARs, Johansen's
  maximum likelihood for the VECMs).
- V5: stability. A VAR on the differences is stable when every inverse root of its characteristic
  polynomial lies inside the unit circle (moduli below 1). A rank-1 VECM on two I(1) series has,
  by construction, exactly one unit root (one shared stochastic trend); it's stable when that is
  the only root on the circle and every other one lies inside.
- V6: multivariate Portmanteau test (adjusted) on the system residuals at lags 6, 12 and 24;
  no autocorrelation left means the lag order captured the dynamics.

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` and
``data/processed/fed_balance_sheet_first_difference.csv``, training split only.

Output, in ``reports/``:

- ``var_models.txt``: each candidate's coefficients (with p-values), its root moduli, its
  Portmanteau p-values, and a verdict
- ``figures/var_models.png``: per candidate, its roots against the unit circle, and the
  Portmanteau p-value by lag against the 5% line
"""

from pathlib import Path
from typing import Any

from loguru import logger
from matplotlib.figure import Figure
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
    SECONDARY_INK,
    SERIES_COLOR,
    SURFACE,
    apply_chart_style,
)

app = typer.Typer()

# 1 (BIC) and 3 (AIC, HQIC) from step V2, and 5: the smallest order whose residuals pass V6,
# added by the roadmap's rule (autocorrelated residuals: raise the lag order).
VAR_LAGS = (1, 3, 5)
# Portmanteau lags that decide the verdict: half a year, a year, two years.
TEST_LAGS = (6, 12, 24)
# The figure shows the Portmanteau p-value at every lag the model allows, up to this one.
PLOT_LAGS = 24
# Rank 1 even though Johansen found rank 0 (step V3): the user's choice, so that forecast
# accuracy, not the test alone, decides whether a long-run link helps.
VECM_RANK = 1
# A unit root estimated from data lands within a hair of 1, never exactly on it.
UNIT_ROOT_TOLERANCE = 1e-6

REPORT_PATH = REPORTS_DIR / "var_models.txt"
FIGURE_PATH = FIGURES_DIR / "var_models.png"


def companion_roots(coefficients: np.ndarray) -> np.ndarray:
    """
    Compute a VAR's inverse characteristic roots from its companion matrix.

    Eigenvalues of the companion matrix of a VAR with coefficient matrices A_1..A_p
    (shape p x k x k), largest modulus first. These are the inverse roots of the
    characteristic polynomial: all inside the unit circle means stable.

    Parameters
    ----------
    coefficients : np.ndarray
        The lag coefficient matrices A_1..A_p stacked along the first axis, shape (p, k, k):
        coefficients[i] multiplies the series' values i + 1 months back.

    Returns
    -------
    np.ndarray
        The p * k eigenvalues (complex), sorted by modulus, largest first.

    Notes
    -----
    Processing:
    1. Build the (p * k) x (p * k) companion matrix: the top block row is [A_1 ... A_p]; the
       identity below it shifts each lag one step back.
    2. Take its eigenvalues and sort them by modulus.

    The companion form rewrites a VAR(p) as a VAR(1) on the stacked vector
    (y_t, y_t-1, ..., y_t-p+1). A VAR(1) is stable when every eigenvalue of its matrix lies
    inside the unit circle, and those eigenvalues are exactly the inverse roots of
    det(I - A_1 z - ... - A_p z^p), so this checks stability without solving the polynomial.
    """
    lags, size, _ = coefficients.shape
    companion = np.zeros((lags * size, lags * size))
    # Top block row: y_t = A_1 y_t-1 + ... + A_p y_t-p.
    companion[:size, :] = np.hstack(list(coefficients))
    # Below it, identities carry y_t-1..y_t-p+1 down one slot (empty when p = 1).
    companion[size:, :-size] = np.eye((lags - 1) * size)
    roots = np.linalg.eigvals(companion)
    return roots[np.argsort(-np.abs(roots))]


def is_stable(moduli: np.ndarray, unit_roots: int) -> bool:
    """
    Check a system's root moduli against the stability rule.

    Exactly `unit_roots` moduli at 1 (within tolerance), every other one inside the circle.

    Parameters
    ----------
    moduli : np.ndarray
        Absolute values of the inverse roots, shape (p * k,), from companion_roots.
    unit_roots : int
        How many roots should sit on the unit circle: 0 for a VAR on the differences; the
        number of series minus the cointegrating rank for a VECM (1 here).

    Returns
    -------
    bool
        True if exactly `unit_roots` moduli are within UNIT_ROOT_TOLERANCE of 1 and every
        other modulus is below 1.

    Notes
    -----
    A VECM is written in levels, and two I(1) series with one cointegrating relation share
    k - r = 1 stochastic trend, which shows up as one root of exactly 1 in the implied levels
    VAR. That root is part of the model, not instability; any further root on or outside
    the circle would be.
    """
    on_circle = np.abs(moduli - 1) < UNIT_ROOT_TOLERANCE
    # bool(): numpy's comparisons give np.bool_; this returns a plain bool, as annotated.
    return bool(on_circle.sum() == unit_roots and (moduli[~on_circle] < 1).all())


def fit_candidates(levels: pd.DataFrame, differences: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Fit every VAR and VECM candidate (step V4).

    Every candidate fitted on the given (training) data, with what V5 and V6 need.

    Parameters
    ----------
    levels : pd.DataFrame
        Levels of both series, shape (months, 2), VALUE_COLUMNS as columns, indexed by
        month-end date with a monthly frequency; trillions USD. Used by the VECMs.
    differences : pd.DataFrame
        First differences, same layout, one row fewer. Used by the VARs.

    Returns
    -------
    dict of str to dict
        Keyed by candidate name, "VAR(p)" then "VECM(p)" for each p in VAR_LAGS. Each value
        holds:

        - "kind": "VAR" or "VECM".
        - "lags": p, the VAR's lags or the VECM's lagged differences.
        - "fitted": the fitted statsmodels results (VARResultsWrapper or VECMResults).
        - "roots": np.ndarray, the inverse roots from companion_roots, largest modulus first.
        - "unit_roots": int, the number of roots expected on the unit circle (0 for a VAR,
          number of series minus VECM_RANK for a VECM).
        - "model_lags": int, the lags of the system in its VAR form (p for a VAR, p + 1 levels
          lags for a VECM), which the Portmanteau test must exceed.

    Notes
    -----
    Processing:
    1. For each p, fit a VAR(p) with a constant on the differences by OLS per equation, and
       get its roots from its coefficient matrices.
    2. For each p, fit a rank-VECM_RANK VECM with p lagged differences and a constant
       outside the long-run relation ("co") on the levels, by Johansen's maximum likelihood,
       and get its roots from its levels VAR representation (var_rep, p + 1 matrices).

    A VECM with p lagged differences is a VAR with p + 1 lags in the levels, which is why its
    roots and its Portmanteau lag limit use p + 1.
    """
    candidates = {}
    for lags in VAR_LAGS:
        fitted = VAR(differences).fit(lags, trend="c")
        candidates[f"VAR({lags})"] = {
            "kind": "VAR",
            "lags": lags,
            "fitted": fitted,
            "roots": companion_roots(fitted.coefs),
            "unit_roots": 0,
            # Portmanteau lags must exceed the lags in the model (levels lags for the VECM).
            "model_lags": lags,
        }
    for lags in VAR_LAGS:
        fitted = VECM(levels, k_ar_diff=lags, coint_rank=VECM_RANK, deterministic="co").fit()
        candidates[f"VECM({lags})"] = {
            "kind": "VECM",
            "lags": lags,
            "fitted": fitted,
            # var_rep: the VECM rewritten as a VAR in levels, shape (p + 1, k, k).
            "roots": companion_roots(np.asarray(fitted.var_rep)),
            # k series with rank r share k - r stochastic trends, each a unit root.
            "unit_roots": levels.shape[1] - VECM_RANK,
            "model_lags": lags + 1,
        }
    return candidates


def check_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """
    Run the V5 stability and V6 residual checks on one candidate and give its verdict.

    Parameters
    ----------
    candidate : dict of str to Any
        One entry of fit_candidates' result (keys "kind", "lags", "fitted", "roots",
        "unit_roots", "model_lags").

    Returns
    -------
    dict of str to Any
        The candidate's keys, plus:

        - "portmanteau": dict of int to float, the adjusted Portmanteau p-value at every lag
          from model_lags + 1 to PLOT_LAGS.
        - "stable": bool, the V5 result (is_stable).
        - "white": bool, True when every TEST_LAGS lag that can be tested has a p-value at or
          above SIGNIFICANCE_LEVEL (no autocorrelation left); False if none can be tested.
        - "verdict": "CARRY FORWARD" if stable and white, otherwise "FAIL".

    Notes
    -----
    Processing:
    1. Run statsmodels' multivariate Portmanteau test (test_whiteness, adjusted) at each lag
       the model allows.
    2. Check stability from the root moduli.
    3. Combine both into the verdict.

    The Portmanteau statistic at lag h has k^2 (h - p) degrees of freedom for a model with p
    lags in VAR form, so it only exists for h > p: lag 6 can't be tested for VECM(5), whose
    VAR form has 6 lags. The adjusted version (Hosking's small-sample correction) is used
    because the training sample is short. The test assumes a constant error variance, so the
    GARCH effects found in U5 can make it reject too often.
    """
    fitted = candidate["fitted"]
    portmanteau = {
        lag: fitted.test_whiteness(nlags=lag, adjusted=True).pvalue
        for lag in range(candidate["model_lags"] + 1, PLOT_LAGS + 1)
    }
    stable = is_stable(np.abs(candidate["roots"]), candidate["unit_roots"])
    # The test needs more lags than the model has, so short test lags may not exist (n/a).
    tested = [portmanteau[lag] for lag in TEST_LAGS if lag in portmanteau]
    # No testable lag means no evidence of white residuals, so that fails too.
    white = bool(tested) and all(p_value >= SIGNIFICANCE_LEVEL for p_value in tested)
    return {
        **candidate,
        "portmanteau": portmanteau,
        "stable": stable,
        "white": white,
        "verdict": "CARRY FORWARD" if stable and white else "FAIL",
    }


def coefficient_lines(candidate: dict[str, Any]) -> list[str]:
    """
    Describe one candidate's estimated coefficients (V4), for the report.

    Parameters
    ----------
    candidate : dict of str to Any
        A candidate from fit_candidates or check_candidate; uses "kind" and "fitted".

    Returns
    -------
    list of str
        Indented report lines. For a VAR: a table of every coefficient's estimate and p-value,
        one row per regressor (constant and each lag of each series), and for each equation an
        (equation, estimate) and (equation, p-value) column. For a VECM: the long-run relation
        (beta, scaled to 1 on the first series), the speed of adjustment towards it (alpha)
        with its p-value in each equation, and a note that the short-run terms aren't listed.

    Notes
    -----
    beta is only identified up to scale, so it's divided by its first entry to read as
    "total assets + b x monetary base". alpha then says how fast each series moves back
    when that combination is away from its mean; a significant alpha is the evidence that a
    long-run link matters.
    """
    fitted = candidate["fitted"]
    if candidate["kind"] == "VAR":
        # params and pvalues are (regressor x equation); swapping the column levels groups
        # each equation's estimate and p-value side by side.
        table = pd.concat(
            {"estimate": fitted.params, "p-value": fitted.pvalues}, axis=1
        ).swaplevel(axis=1)
        return ["    " + line for line in table.round(4).to_string().splitlines()]
    beta = fitted.beta.ravel() / fitted.beta.ravel()[0]
    lines = [
        f"    Long-run relation (beta): {VALUE_COLUMNS[0]} {beta[1]:+.4f} x {VALUE_COLUMNS[1]}",
        "    Adjustment towards it (alpha), per equation:",
    ]
    for column, alpha, p_value in zip(
        VALUE_COLUMNS, fitted.alpha.ravel(), fitted.pvalues_alpha.ravel()
    ):
        lines.append(f"      {column:<22} {alpha:+.4f} (p={p_value:.3f})")
    lines.append(
        "    Short-run terms (gamma) and constants are in the fitted model; the forecasts use them."
    )
    return lines


def models_report(results: dict[str, dict[str, Any]], period: str) -> str:
    """
    Write the V4 to V6 text report.

    Parameters
    ----------
    results : dict of str to dict
        The checked candidates (check_candidate), keyed by name, in report order.
    period : str
        Description of the training levels used, e.g. "levels 2002-12 to 2016-10
        (167 months)".

    Returns
    -------
    str
        The report, lines joined by newlines: the setup, then per candidate its verdict,
        coefficients, root moduli with the stability result and Portmanteau p-values with the
        residual result, and finally the candidates carried forward.
    """
    lines = [
        "VAR / VECM estimation and checks, steps V4 (estimate), V5 (stability), V6 (residuals)",
        f"Training split only: {period}. Series: {', '.join(VALUE_COLUMNS)}.",
        (
            "VAR(p): first differences, constant, p lags, OLS per equation. VECM(p): levels, "
            f"rank {VECM_RANK}, p lagged differences, constant, maximum likelihood."
        ),
        (
            "Stability: inverse-root moduli of the implied VAR. A VAR on the differences needs "
            f"all below 1; a rank-{VECM_RANK} VECM on 2 series needs exactly one at 1 (its shared "
            "stochastic trend) and the rest below 1."
        ),
        (
            "Residuals: adjusted multivariate Portmanteau test (null: no autocorrelation up to "
            f"the lag) at lags {', '.join(map(str, TEST_LAGS))}; passes when every p-value is at "
            f"or above {SIGNIFICANCE_LEVEL:.0%}."
        ),
        "",
    ]
    for name, result in results.items():
        lines.append(f"{name}  [{result['verdict']}]")
        lines.append("  V4 coefficients:")
        lines.extend(coefficient_lines(result))
        moduli = ", ".join(f"{value:.4f}" for value in np.abs(result["roots"]))
        lines.append(
            f"  V5 inverse-root moduli: {moduli} -> {'stable' if result['stable'] else 'NOT stable'}"
        )
        portmanteau = ", ".join(
            f"lag {lag} p={result['portmanteau'][lag]:.4f}"
            if lag in result["portmanteau"]
            else f"lag {lag} n/a (the model has {result['model_lags']} lags)"
            for lag in TEST_LAGS
        )
        lines.append(
            f"  V6 Portmanteau: {portmanteau} -> "
            f"{'no autocorrelation left' if result['white'] else 'autocorrelation remains'}"
        )
        lines.append("")
    carried = [name for name, result in results.items() if result["verdict"] == "CARRY FORWARD"]
    lines.append(f"Carry forward to the evaluation: {', '.join(carried) if carried else 'none'}")
    lines.append(f"Figure: {repo_path(FIGURE_PATH)}")
    return "\n".join(lines)


def models_figure(results: dict[str, dict[str, Any]]) -> Figure:
    """
    Draw the V5 / V6 figure, one row per candidate.

    Per candidate: its roots against the unit circle, and the Portmanteau p-value by lag.

    Parameters
    ----------
    results : dict of str to dict
        The checked candidates (check_candidate), keyed by name; one figure row each, in this
        order.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, not yet saved; the caller saves and closes it.

    Notes
    -----
    Processing, for each candidate:
    1. Left: the unit circle and the inverse roots in the complex plane; roots within
       UNIT_ROOT_TOLERANCE of the circle are drawn hollow and labelled as the VECM's shared
       trend, the rest filled (in the "critical" red if outside the circle).
    2. Right: the Portmanteau p-value at every testable lag on a log scale, against the 5%
       line; failing lags in the "critical" red.

    The log scale keeps both tiny p-values (clear failures) and ones near 1 readable on the
    same axis.
    """
    apply_chart_style()
    figure, axes = plt.subplots(
        len(results),
        2,
        figsize=(12, 3.6 * len(results)),
        squeeze=False,
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": (1, 2.2)},
    )
    # Angles for drawing the unit circle as a 200-point polygon.
    circle = np.linspace(0, 2 * np.pi, 200)
    for row, (name, result) in enumerate(results.items()):
        root_axis, p_axis = axes[row]
        for axis in axes[row]:
            axis.set_facecolor(SURFACE)
        root_axis.plot(np.cos(circle), np.sin(circle), color=BASELINE, linewidth=1)
        roots = result["roots"]
        # Same tolerance as is_stable, so the drawing agrees with the verdict.
        on_circle = np.abs(np.abs(roots) - 1) < UNIT_ROOT_TOLERANCE
        outside = ~on_circle & (np.abs(roots) > 1)
        groups = (
            (~on_circle & ~outside, {"color": SERIES_COLOR, "s": 40}, "inside: shocks die out"),
            (outside, {"color": CRITICAL_COLOR, "s": 40}, "outside: explosive, NOT stable"),
            (
                on_circle,
                {"facecolor": SURFACE, "edgecolors": SECONDARY_INK, "s": 60, "linewidths": 2},
                "unit root (the VECM's shared trend)",
            ),
        )
        for mask, style, label in groups:
            # Skip empty groups (a VAR has no unit root) so the legend doesn't list them.
            if mask.any():
                root_axis.scatter(
                    roots[mask].real, roots[mask].imag, zorder=3, label=label, **style
                )
        root_axis.set_xlim(-1.2, 1.2)
        root_axis.set_ylim(-1.2, 1.2)
        root_axis.set_aspect("equal")
        root_axis.set_xlabel("Real part")
        root_axis.set_ylabel("Imaginary part")
        root_axis.set_title(
            f"{name}  [{result['verdict']}]\nroots vs the unit circle "
            f"({'stable' if result['stable'] else 'NOT stable'})",
            loc="left",
            fontsize=10,
        )
        root_axis.legend(loc="lower left", fontsize=8, frameon=False)

        lags = np.array(list(result["portmanteau"]))
        p_values = np.array(list(result["portmanteau"].values()))
        failing = p_values < SIGNIFICANCE_LEVEL
        p_axis.scatter(lags[~failing], p_values[~failing], color=SERIES_COLOR, s=24, zorder=3)
        p_axis.scatter(lags[failing], p_values[failing], color=CRITICAL_COLOR, s=24, zorder=3)
        p_axis.axhline(SIGNIFICANCE_LEVEL, color=CRITICAL_COLOR, linewidth=1)
        p_axis.annotate(
            " 5%",
            (1, SIGNIFICANCE_LEVEL),
            xycoords=("axes fraction", "data"),
            va="center",
            color=SECONDARY_INK,
            fontsize=9,
        )
        p_axis.set_yscale("log")
        # Down to 1e-4 at least, lower when a p-value is smaller, so every point stays visible.
        p_axis.set_ylim(min(1e-4, p_values.min() / 3), 1.5)
        p_axis.set_xlim(0.5, PLOT_LAGS + 0.5)
        p_axis.set_xticks([1, 6, 12, 18, 24])
        p_axis.set_xlabel("Lag (months)")
        p_axis.set_title(
            "Portmanteau p-value by lag (below the line: autocorrelation remains; red)",
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
    Run steps V4 to V6 on the training split and write the report and figure.

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
    1. Load both files and keep only the training split (boundaries from the levels).
    2. Index both by month-end date with an explicit monthly frequency.
    3. V4: fit every candidate.
    4. V5, V6: check each one and give its verdict.
    5. Write and print the report; draw the figure.
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
    period = (
        f"levels {train_levels[DATE_COLUMN].min():%Y-%m} to {train_levels[DATE_COLUMN].max():%Y-%m}"
        f" ({len(train_levels)} months)"
    )

    # Month-end frequency on the index: statsmodels then knows the spacing, and a missing month
    # would surface as NaN rows that the fit rejects, instead of passing silently.
    candidates = fit_candidates(
        train_levels.set_index(DATE_COLUMN)[list(VALUE_COLUMNS)].asfreq("ME"),
        train_differences.set_index(DATE_COLUMN)[list(VALUE_COLUMNS)].asfreq("ME"),
    )
    results = {name: check_candidate(candidate) for name, candidate in candidates.items()}

    report = models_report(results, period)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = models_figure(results)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"VAR / VECM checks written to {REPORT_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    app()
