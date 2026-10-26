"""
Asymmetric GARCH (step U5c): does volatility react differently to falls than to rises, and do
EGARCH or GJR-GARCH fit the SARIMA residuals better than step U5b's GARCH?

These are new volatility models next to U5b's choice, which is final (its test run is done) and
stays unchanged. For each SARIMA model that passed the residual checks, EGARCH(1,1) and
GJR-GARCH(1,1), both with Student-t errors, are fitted plain and break-adjusted (each regime
divided by its own standard deviation, as in U5b); BIC picks one version per family. Likelihoods
are in original units, so the BIC compares directly with U5b's chosen variant.

- GJR: variance_t = omega + (alpha + gamma * [e_(t-1) < 0]) e_(t-1)^2 + beta variance_(t-1); gamma > 0
  means falls raise volatility more. Persistence alpha + gamma / 2 + beta.
- EGARCH: log variance_t = omega + alpha (|z| - E|z|) + gamma z_(t-1) + beta log variance_(t-1);
  gamma < 0 means falls raise volatility more. Persistence beta.

Input: the same as ``residual_checks``: the training split and
``reports/information_criteria_candidates.csv``.

Output, in ``reports/``:

- ``asymmetric_garch.txt``: per model and family, the BIC against U5b's GARCH, the asymmetry
  coefficient with its p-value, persistence, the standardized-residual checks, and a verdict
- ``figures/asymmetric_garch.png``: each family's BIC difference from U5b's GARCH, and the
  asymmetry coefficient with its 95% interval
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
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
    SURFACE,
    apply_chart_style,
    checked_models,
    model_label,
    model_name,
)
from autoregressive_experiments.volatility_checks import ARCH_TEST_LAGS, arch_lm_p_values
from autoregressive_experiments.volatility_models import (
    BREAK_SUFFIX,
    EGARCH_T,
    GJR_T,
    fit_candidate,
    garch_check,
    information_criteria,
)

app = typer.Typer()

FAMILIES = (EGARCH_T, GJR_T)
REPORT_PATH = REPORTS_DIR / "asymmetric_garch.txt"
FIGURE_PATH = FIGURES_DIR / "asymmetric_garch.png"
# Reference palette slots 2 and 3 (validated with slot 1, which stays GARCH's elsewhere).
FAMILY_COLORS = {EGARCH_T: "#eb6834", GJR_T: "#1baf7a"}


def persistence(family: str, params: pd.Series) -> float:
    if family == EGARCH_T:
        return float(params["beta[1]"])
    # A symmetric shock distribution is negative half the time, hence gamma / 2.
    return float(params["alpha[1]"] + params["gamma[1]"] / 2 + params["beta[1]"])


def asymmetric_check(residuals: np.ndarray) -> dict:
    """Per family, the better of the plain and break-adjusted version by BIC, its BIC difference
    from U5b's chosen GARCH variant (negative: better), its asymmetry and its checks."""
    count = len(residuals)
    reference = garch_check(residuals)
    reference_bic = reference["information_criteria"][reference["garch_name"]][1]
    results = {"reference": reference["garch_name"], "reference_bic": reference_bic}
    for family in FAMILIES:
        versions = {
            name: fit_candidate(residuals, name) for name in (family, family + BREAK_SUFFIX)
        }
        bics = {
            name: information_criteria(c["loglikelihood"], c["parameters"], count)[1]
            for name, c in versions.items()
        }
        chosen = min(bics, key=bics.get)
        fit = versions[chosen]["fit"]
        standardized = np.asarray(fit.std_resid)
        arch_p = arch_lm_p_values(standardized, ARCH_TEST_LAGS)
        squared_p = acorr_ljungbox(standardized**2, lags=list(ARCH_TEST_LAGS))["lb_pvalue"]
        clean = bool((np.concatenate([arch_p, squared_p.to_numpy()]) >= SIGNIFICANCE_LEVEL).all())
        stationary = abs(persistence(family, fit.params)) < 1
        delta = bics[chosen] - reference_bic
        if not (clean and stationary):
            verdict = "FALLS SHORT"
        elif delta < 0:
            verdict = "BETTER THAN GARCH"
        else:
            verdict = "NOT BETTER THAN GARCH"
        results[family] = {
            "variant": chosen,
            "bic": bics[chosen],
            "delta_bic": delta,
            "gamma": float(fit.params["gamma[1]"]),
            "gamma_se": float(fit.std_err["gamma[1]"]),
            "gamma_p": float(fit.pvalues["gamma[1]"]),
            "nu": float(fit.params["nu"]),
            "persistence": persistence(family, fit.params),
            "arch_p": arch_p,
            "squared_ljung_box_p": squared_p.to_numpy(),
            "verdict": verdict,
        }
    return results


def asymmetric_report(results: list[dict]) -> str:
    lines = [
        "Asymmetric GARCH on the residuals of the models that passed the residual checks (U5c)",
        "Training split only. New models next to U5b's GARCH choice, which stays unchanged.",
        (
            "Per family, BIC picks plain or break-adjusted; delta BIC = family BIC - U5b's chosen "
            "variant's BIC (negative: the asymmetric model fits better). All in original units."
        ),
        (
            "Asymmetry: GJR gamma > 0, or EGARCH gamma < 0, means falls raise volatility more "
            f"than rises; p-value against gamma = 0 at {SIGNIFICANCE_LEVEL:.0%}."
        ),
        (
            "Verdict: BETTER / NOT BETTER THAN GARCH by BIC, or FALLS SHORT when ARCH effects "
            "remain in the standardized residuals or the variance isn't stationary."
        ),
        "",
    ]
    for column in VALUE_COLUMNS:
        for result in (r for r in results if r["series"] == column):
            lines.append(
                f"{column}  {model_label(result)}  (U5b GARCH: {result['reference']}, "
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
                lines.append(f"    standardized residuals: {checks}")
                lines.append(f"    Verdict: {f['verdict']}")
            lines.append("")
    lines.append(f"Figure: {repo_path(FIGURE_PATH)}")
    return "\n".join(lines)


def asymmetric_figure(results: list[dict]) -> plt.Figure:
    """Each family's BIC difference from U5b's GARCH, and its asymmetry coefficient."""
    apply_chart_style()
    figure, (bic_axis, gamma_axis) = plt.subplots(1, 2, figsize=(15, 5), facecolor=SURFACE)
    labels = [f"{model_name(r)}: {model_label(r)}" for r in results]
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
        axis.set_axisbelow(True)
    bic_axis.set_xlabel("BIC difference from U5b's GARCH (left of 0: fits better)")
    bic_axis.set_title(
        "U5c: does an asymmetric GARCH fit better than U5b's GARCH?", loc="left", fontsize=10
    )
    bic_axis.legend(loc="upper right", fontsize=8, frameon=False)
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
    checked, _ = checked_models(levels_path, differences_path, candidates_path)
    results = [
        {**result, **asymmetric_check(result["residuals"])}
        for result in checked
        if result["passed"]
    ]

    report = asymmetric_report(results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = asymmetric_figure(results)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"Asymmetric GARCH checks written to {REPORT_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    app()
