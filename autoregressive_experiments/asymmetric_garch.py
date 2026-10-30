"""
Asymmetric GARCH: does volatility react differently to falls than to rises, and do
EGARCH or GJR-GARCH fit the SARIMA residuals better than the GARCH model?

These are new volatility models next to the GARCH choice, which stays as it is and stays unchanged.
For each SARIMA model that passed the residual checks, EGARCH(1,1) and GJR-GARCH(1,1), both with
Student-t errors, are fitted plain and break-adjusted (each regime divided by its own standard
deviation, as in volatility_models); BIC picks one version per family. Likelihoods are in original
units, so the BIC compares directly with the chosen GARCH variant.

- GJR: variance_t = omega + (alpha + gamma * [e_(t-1) < 0]) e_(t-1)^2 + beta variance_(t-1);
  gamma > 0 means falls raise volatility more. Persistence alpha + gamma / 2 + beta.
- EGARCH: log variance_t = omega + alpha (|z| - E|z|) + gamma z_(t-1) + beta log variance_(t-1);
  gamma < 0 means falls raise volatility more. Persistence beta.

Input: the same as ``residual_checks``: the training split and
``reports/information_criteria_candidates.csv``.

Output, in ``reports/``:

- ``asymmetric_garch.txt``: per model and family, the BIC against the GARCH model, the asymmetry
  coefficient with its p-value, persistence, the standardized-residual checks, and a verdict
- ``figures/asymmetric_garch.png``: each family's BIC difference from the GARCH model, and the
  asymmetry coefficient with its 95% interval
"""

from pathlib import Path
from typing import Any

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from autoregressive_experiments.residual_checks import (
    PRIMARY_INK,
    SERIES_PALETTE,
    SURFACE,
    apply_chart_style,
    checked_models,
    model_label,
    model_name,
)
from autoregressive_experiments.runlog import begin_phase, finish_run, start_run_log
from autoregressive_experiments.volatility_checks import ARCH_TEST_LAGS
from autoregressive_experiments.volatility_models import (
    BOOTSTRAP_NOTE,
    BOOTSTRAP_REPLICATIONS,
    BREAK_SUFFIX,
    EGARCH_T,
    GJR_T,
    bootstrap_checks,
    fit_candidate,
    garch_check,
    information_criteria,
)

app = typer.Typer()

# The two asymmetric families, in the order they are reported and drawn.
FAMILIES = (EGARCH_T, GJR_T)
REPORT_PATH = REPORTS_DIR / "asymmetric_garch.txt"
FIGURE_PATH = FIGURES_DIR / "asymmetric_garch.png"
# Reference palette slots 2 and 3 (validated with slot 1, which stays GARCH's elsewhere).
FAMILY_COLORS = {EGARCH_T: SERIES_PALETTE[1], GJR_T: SERIES_PALETTE[2]}


def persistence(family: str, params: pd.Series) -> float:
    """
    How much of a volatility shock carries into the next month, by family.

    Parameters
    ----------
    family : str
        EGARCH_T or GJR_T. Anything other than EGARCH_T is treated as GJR.
    params : pd.Series
        The fitted parameters, with "alpha[1]", "gamma[1]" and "beta[1]" (arch's names).

    Returns
    -------
    float
        EGARCH: beta, the coefficient on last month's log variance. GJR: alpha + gamma / 2 +
        beta. The variance is stationary when this is below 1 (in absolute value for EGARCH).

    Notes
    -----
    Why the formulas differ: EGARCH models log variance, whose dependence on its own past is
    beta alone; alpha and gamma act on the standardized shock z, which has a fixed
    distribution, so they add a fresh impulse each month without feeding back. In GJR, the
    expected effect of last month's squared shock on this month's variance is alpha plus
    gamma times the probability of a fall, which is 1/2 for a symmetric shock distribution.
    """
    if family == EGARCH_T:
        return float(params["beta[1]"])
    # A symmetric shock distribution is negative half the time, hence gamma / 2.
    return float(params["alpha[1]"] + params["gamma[1]"] / 2 + params["beta[1]"])


def asymmetric_check(
    residuals: np.ndarray | pd.Series,
    replications: int = BOOTSTRAP_REPLICATIONS,
    label: str = "",
) -> dict[str, Any]:
    """
    Fit EGARCH and GJR-GARCH and compare them with the chosen GARCH variant.

    Per family, the better of the plain and break-adjusted version by BIC, its BIC difference
    from the chosen GARCH variant (negative: better), its asymmetry and its checks.

    Parameters
    ----------
    residuals : np.ndarray or pd.Series
        One SARIMA model's residuals, shape (n,), in trillions.
    replications : int, default=BOOTSTRAP_REPLICATIONS
        Bootstrap replications for the checks; 0 skips the checks and verdicts, for the
        forecast evaluation, which only needs each family's chosen variant.
    label : str, default=""
        Names the model in the progress lines, e.g. "monetary_base_T SARIMA(0,1,1)".

    Returns
    -------
    dict of str to Any
        - "reference": str, the chosen GARCH variant (garch_check()'s "garch_name")
        - "reference_bic": float, its BIC
        - one entry per family in FAMILIES (EGARCH_T, GJR_T), a dict with:

          - "variant": str, the chosen version's name (plain or break-adjusted)
          - "bic", "delta_bic": float, its BIC and BIC minus "reference_bic"
          - "gamma", "gamma_se", "gamma_p": float, the asymmetry coefficient, its standard
            error and its p-value against gamma = 0
          - "nu": float, the Student-t degrees of freedom
          - "persistence": float, see persistence()
          - with replications > 0: bootstrap_checks()'s keys ("arch_p_curve", "arch_p",
            "squared_ljung_box_p", "failed_fits") and
          - "verdict": str, "FALLS SHORT", "BETTER THAN GARCH" or "NOT BETTER THAN GARCH"

    Notes
    -----
    Processing:
    1. Rerun volatility_models' garch_check() to get the reference variant and its BIC.
    2. Per family, fit the plain and the break-adjusted version with fit_candidate(), whose
       likelihoods are in original units (Jacobian included), and keep the lower BIC.
    3. Check the chosen version as volatility_models does: ARCH-LM and Ljung-Box on the squared
       standardized residuals with bootstrap p-values (bootstrap_checks()), and a
       stationary variance (|persistence| < 1).
    4. Verdict: "FALLS SHORT" if a check fails; otherwise "BETTER THAN GARCH" when its BIC
       is below the reference's, else "NOT BETTER THAN GARCH".

    gamma's standard error and p-value are arch's defaults, which use the robust
    (sandwich) covariance.
    """
    count = len(residuals)
    # Only the reference's BIC choice is needed here, so its bootstrap checks are skipped.
    reference = garch_check(residuals, replications=0)
    reference_bic = reference["information_criteria"][reference["garch_name"]][1]
    # Annotated as Any: the values are strings, floats and, per family, dicts.
    results: dict[str, Any] = {
        "reference": reference["garch_name"],
        "reference_bic": reference_bic,
    }
    for family in FAMILIES:
        versions = {
            name: fit_candidate(residuals, name) for name in (family, family + BREAK_SUFFIX)
        }
        bics = {
            name: information_criteria(c["loglikelihood"], c["parameters"], count)[1]
            for name, c in versions.items()
        }
        # mypy can't match the overloads of dict.get to min's key type; at runtime it returns
        # the BIC.
        chosen = min(bics, key=bics.get)  # type: ignore[arg-type]
        fit = versions[chosen]["fit"]
        delta = bics[chosen] - reference_bic
        results[family] = {
            "variant": chosen,
            "bic": bics[chosen],
            "delta_bic": delta,
            "gamma": float(fit.params["gamma[1]"]),
            "gamma_se": float(fit.std_err["gamma[1]"]),
            "gamma_p": float(fit.pvalues["gamma[1]"]),
            "nu": float(fit.params["nu"]),
            "persistence": persistence(family, fit.params),
        }
        if not replications:
            continue
        checks = bootstrap_checks(versions[chosen], chosen, replications, label)
        clean = not (
            np.concatenate([checks["arch_p"], checks["squared_ljung_box_p"]]) < SIGNIFICANCE_LEVEL
        ).any()
        # abs(): EGARCH's beta may be negative; GJR's persistence is positive anyway.
        stationary = abs(results[family]["persistence"]) < 1
        if not (clean and stationary):
            verdict = "FALLS SHORT"
        elif delta < 0:
            verdict = "BETTER THAN GARCH"
        else:
            verdict = "NOT BETTER THAN GARCH"
        results[family] |= {**checks, "verdict": verdict}
    return results


def asymmetric_report(results: list[dict[str, Any]]) -> str:
    """
    Write the text report of the asymmetric GARCH checks.

    Parameters
    ----------
    results : list of dict of str to Any
        One dict per model that passed the residual checks: its check_model() result merged
        with its asymmetric_check() result.

    Returns
    -------
    str
        The report: a header explaining the method, then per model the reference GARCH and
        its BIC, and per family the chosen version, its BIC and delta, gamma with its
        p-value, nu, persistence, the standardized-residual checks and the verdict; then
        the figure's path.
    """
    lines = [
        "Asymmetric GARCH on the residuals of the models that passed the residual checks",
        "Training split only. New models next to the GARCH choice, which stays unchanged.",
        (
            "Per family, BIC picks plain or break-adjusted; delta BIC = family BIC - the chosen "
            "GARCH variant's BIC (negative: the asymmetric model fits better). All in "
            "original units."
        ),
        (
            "Asymmetry: GJR gamma > 0, or EGARCH gamma < 0, means falls raise volatility more "
            f"than rises; p-value against gamma = 0 at {SIGNIFICANCE_LEVEL:.0%}."
        ),
        (
            "Verdict: BETTER / NOT BETTER THAN GARCH by BIC, or FALLS SHORT when ARCH effects "
            "remain in the standardized residuals or the variance isn't stationary."
        ),
        BOOTSTRAP_NOTE,
        "",
    ]
    for column in VALUE_COLUMNS:
        for result in (r for r in results if r["series"] == column):
            lines.append(
                f"{column}  {model_label(result)}  (GARCH: {result['reference']}, "
                f"BIC {result['reference_bic']:.2f})"
            )
            for family in FAMILIES:
                f = result[family]
                lines.append(
                    f"  {f['variant']:<42} BIC {f['bic']:9.2f}  (delta {f['delta_bic']:+7.2f})  "
                    f"gamma {f['gamma']:+.3f} (p={f['gamma_p']:.3f})  nu {f['nu']:.1f}  "
                    f"persistence {f['persistence']:.3f}"
                )
                checks = ", ".join(
                    f"lag {lag} ARCH-LM p={a:.3f} / LB p={b:.3f}"
                    for lag, a, b in zip(ARCH_TEST_LAGS, f["arch_p"], f["squared_ljung_box_p"])
                )
                lines.append(
                    f"    standardized residuals: {checks}; {f['failed_fits']} of "
                    f"{BOOTSTRAP_REPLICATIONS} bootstrap re-estimations didn't converge (kept)"
                )
                lines.append(f"    Verdict: {f['verdict']}")
            lines.append("")
    lines.append(f"Figure: {repo_path(FIGURE_PATH)}")
    return "\n".join(lines)


def asymmetric_figure(results: list[dict[str, Any]]) -> Figure:
    """
    Each family's BIC difference from the GARCH model, and its asymmetry coefficient.

    Parameters
    ----------
    results : list of dict of str to Any
        As in asymmetric_report(): check_model() results merged with asymmetric_check().

    Returns
    -------
    Figure
        One row of two panels sharing the model labels on the y axis. The caller saves and
        closes it.

    Notes
    -----
    Processing:
    1. Left: per model, one horizontal bar per family with its BIC difference from the
       GARCH model; a bar left of 0 means the asymmetric model fits better.
    2. Right: per model and family, gamma with its 95% interval (+/- 1.96 standard errors);
       an interval clear of 0 means falls and rises move volatility differently.

    The two families sit side by side, offset half a bar height above and below each
    model's position.
    """
    apply_chart_style()
    figure, (bic_axis, gamma_axis) = plt.subplots(1, 2, figsize=(15, 5), facecolor=SURFACE)
    labels = [f"{model_name(r)}: {model_label(r)}" for r in results]
    # Reversed, so the first model is drawn at the top.
    positions = np.arange(len(results))[::-1]
    height = 0.36
    for offset, family in zip((height / 2, -height / 2), FAMILIES):
        color = FAMILY_COLORS[family]
        deltas = [r[family]["delta_bic"] for r in results]
        bic_axis.barh(positions + offset, deltas, height=height, color=color, label=family)
        gammas = np.array([r[family]["gamma"] for r in results])
        errors = 1.96 * np.array([r[family]["gamma_se"] for r in results])
        gamma_axis.errorbar(
            gammas,
            positions + offset,
            xerr=errors,
            fmt="o",
            color=color,
            markersize=6,
            capsize=3,
            label=family,
        )
    for axis in (bic_axis, gamma_axis):
        axis.set_facecolor(SURFACE)
        axis.axvline(0, color=PRIMARY_INK, linewidth=1.5)
        axis.set_yticks(positions, labels)
        # Grid lines behind the bars and points.
        axis.set_axisbelow(True)
    bic_axis.set_xlabel("BIC difference from the GARCH model (left of 0: fits better)")
    bic_axis.set_title(
        "Does an asymmetric GARCH fit better than the GARCH model?", loc="left", fontsize=10
    )
    bic_axis.legend(loc="upper right", fontsize=8, frameon=False)
    # Same tick positions as the left panel, without repeating the labels.
    gamma_axis.set_yticks(positions, [""] * len(positions))
    gamma_axis.set_xlabel("Asymmetry coefficient gamma, 95% interval (black line: gamma = 0)")
    gamma_axis.set_title(
        "Asymmetry: interval clear of 0 = falls and rises move volatility differently\n"
        "GJR gamma > 0 or EGARCH gamma < 0: falls raise it more\n"
        "GJR gamma < 0 or EGARCH gamma > 0: rises raise it more",
        loc="left",
        fontsize=10,
        color=PRIMARY_INK,
    )
    figure.tight_layout()
    return figure


@app.command()
def main(
    levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv",
    differences_path: Path = PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv",
    candidates_path: Path = CANDIDATES_PATH,
) -> None:
    """
    Run the asymmetric GARCH checks and write the report and the figure.

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
        passed the residual checks are examined.
    """
    start_run_log("asymmetric-garch")
    begin_phase("EGARCH / GJR fits and bootstrap checks")
    # The training frame isn't needed: this report doesn't print dates.
    checked, _ = checked_models(levels_path, differences_path, candidates_path)
    # Each result keeps its residual-check fields (series, orders, residuals) and gains the
    # asymmetric ones.
    results = [
        {
            **result,
            **asymmetric_check(
                result["residuals"], label=f"{model_name(result)} {model_label(result)}"
            ),
        }
        for result in checked
        if result["passed"]
    ]

    report = asymmetric_report(results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = asymmetric_figure(results)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    # Free the figure's memory; it is already saved.
    plt.close(figure)
    logger.success(f"Asymmetric GARCH checks written to {REPORT_PATH} and {FIGURE_PATH}")
    finish_run()


# Running this file directly runs the typer CLI (main).
if __name__ == "__main__":
    app()
