"""
Volatility models (step U5b): does GARCH(1,1) earn its place on top of each SARIMA model?

Two-step estimation: the SARIMA mean models stay as fitted, and four variance models are fitted
to their residuals with the ``arch`` package: a constant variance, GARCH(1,1) with normal errors,
GARCH(1,1) with Student-t errors, and GARCH(1,1) Student-t on the break-adjusted residuals (each
regime divided by its own standard deviation, at the variance break that ``volatility_checks``
estimates). The last one stops the 2008 variance shift from posing as volatility persistence.
BIC picks between them. GARCH is carried forward only when BIC prefers it, its variance is
stationary (alpha + beta < 1), and the residuals divided by its monthly standard deviation no
longer show ARCH effects.

Input: the same as ``residual_checks``: the training split and
``reports/information_criteria_candidates.csv``. Only models that pass the residual checks are
examined.

Output, in ``reports/``:

- ``garch_diagnostics.txt``: per model, the AIC / BIC of the four variance models, the GARCH
  parameters, persistence and half-life, the checks on the standardized residuals, and a verdict
- ``figures/garch_diagnostics.png``: per model, the absolute residuals with GARCH's monthly
  standard deviation and the constant one, and ARCH-LM p-values before and after GARCH
"""

from pathlib import Path

from arch import arch_model
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
from autoregressive_experiments.features import DATE_COLUMN
from autoregressive_experiments.residual_checks import (
    CRITICAL_COLOR,
    SECONDARY_INK,
    SERIES_COLOR,
    SURFACE,
    apply_chart_style,
    checked_models,
    model_label,
    model_name,
)
from autoregressive_experiments.volatility_checks import (
    ADJUSTED_COLOR,
    ARCH_TEST_LAGS,
    PLOT_LAGS,
    arch_lm_p_values,
    regime_scales,
    variance_break,
)

app = typer.Typer()

CONSTANT = "constant variance"
GARCH_NORMAL = "GARCH(1,1), normal"
GARCH_T = "GARCH(1,1), Student-t"
VARIANCE_MODELS = {
    CONSTANT: {"vol": "Constant", "dist": "normal"},
    GARCH_NORMAL: {"vol": "GARCH", "p": 1, "q": 1, "dist": "normal"},
    GARCH_T: {"vol": "GARCH", "p": 1, "q": 1, "dist": "t"},
}
GARCH_T_BREAK = "GARCH(1,1), Student-t, break-adjusted"
# The regime scaling adds two parameters: the ratio of the two regimes' scales (the overall
# scale is GARCH's omega) and the break date.
BREAK_PARAMETERS = 2

VERDICTS = {
    "CARRY GARCH FORWARD": (
        "BIC prefers GARCH, its variance is stationary, and the standardized residuals show no "
        "ARCH effects left."
    ),
    "CONSTANT VARIANCE": "BIC prefers a constant variance: GARCH doesn't earn its parameters.",
    "GARCH FALLS SHORT": (
        "BIC prefers GARCH, but its variance isn't stationary or ARCH effects remain: GARCH(1,1) "
        "is not enough on its own."
    ),
}

RESIDUALS_LABEL = "residuals as they are (left: absolute value)"
GARCH_LABEL = "with GARCH (left: its monthly SD; right: residuals divided by it)"

REPORT_PATH = REPORTS_DIR / "garch_diagnostics.txt"
FIGURE_PATH = FIGURES_DIR / "garch_diagnostics.png"


def fit_variance_models(residuals: np.ndarray) -> dict:
    """The three plain variance models on zero-mean residuals, rescaled by arch for its optimizer."""
    values = np.asarray(residuals, dtype=float)
    return {
        name: arch_model(values, mean="Zero", rescale=True, **spec).fit(disp="off")
        for name, spec in VARIANCE_MODELS.items()
    }


def original_units_loglikelihood(fit) -> float:
    """Log-likelihood of the data arch was given: arch fits c * y, and log f(y) = log f(c y) + log c."""
    return float(fit.loglikelihood + fit.nobs * np.log(fit.scale))


def information_criteria(loglikelihood: float, parameters: int, count: int) -> tuple[float, float]:
    """(AIC, BIC)."""
    return -2 * loglikelihood + 2 * parameters, -2 * loglikelihood + parameters * np.log(count)


def variance_candidates(residuals: np.ndarray) -> dict[str, dict]:
    """All four variance models, with log-likelihoods of the same residuals in trillions, their
    parameter counts, and each model's monthly standard deviation in trillions."""
    values = np.asarray(residuals, dtype=float)
    candidates = {
        name: {
            "fit": fit,
            "loglikelihood": original_units_loglikelihood(fit),
            "parameters": fit.num_params,
            "sd": np.asarray(fit.conditional_volatility) / fit.scale,
        }
        for name, fit in fit_variance_models(values).items()
    }
    scales = regime_scales(values, variance_break(values))
    break_fit = arch_model(
        values / scales, mean="Zero", rescale=True, **VARIANCE_MODELS[GARCH_T]
    ).fit(disp="off")
    candidates[GARCH_T_BREAK] = {
        "fit": break_fit,
        # Change of variables u = e / s: log f(e) = log f(u) - log s, summed over the months.
        "loglikelihood": original_units_loglikelihood(break_fit) - float(np.log(scales).sum()),
        "parameters": break_fit.num_params + BREAK_PARAMETERS,
        "sd": scales * np.asarray(break_fit.conditional_volatility) / break_fit.scale,
    }
    return candidates


def garch_variance_path(
    residuals: np.ndarray, variant: str, steps: int
) -> tuple[np.ndarray, float]:
    """Forecast shock variances for the next steps months (trillions squared) from the chosen
    GARCH variant, and its Student-t degrees of freedom (inf for normal errors)."""
    values = np.asarray(residuals, dtype=float)
    regime_scale = 1.0
    if variant == GARCH_T_BREAK:
        scales = regime_scales(values, variance_break(values))
        values, regime_scale = values / scales, scales[-1]
    spec = VARIANCE_MODELS[GARCH_T if variant == GARCH_T_BREAK else variant]
    fit = arch_model(values, mean="Zero", rescale=True, **spec).fit(disp="off")
    variances = fit.forecast(horizon=steps, reindex=False).variance.to_numpy()[-1]
    # Back from arch's rescaled units, and onto the latest (post-break) regime's scale.
    return variances / fit.scale**2 * regime_scale**2, float(fit.params.get("nu", np.inf))


def half_life(persistence: float) -> float:
    """Months for a volatility shock to halve: persistence ** h = 0.5."""
    return float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else np.inf


def describe_half_life(months: float) -> str:
    return (
        f"half-life {months:.1f} months" if np.isfinite(months) else "no decay (alpha + beta = 1)"
    )


def garch_check(residuals: np.ndarray) -> dict:
    candidates = variance_candidates(residuals)
    count = len(residuals)
    criteria = {
        name: information_criteria(c["loglikelihood"], c["parameters"], count)
        for name, c in candidates.items()
    }
    best = min(criteria, key=lambda name: criteria[name][1])
    garch_names = [name for name in criteria if name != CONSTANT]
    garch_name = best if best != CONSTANT else min(garch_names, key=lambda n: criteria[n][1])
    garch = candidates[garch_name]["fit"]
    persistence = float(garch.params["alpha[1]"] + garch.params["beta[1]"])
    standardized = np.asarray(garch.std_resid)
    arch_p = arch_lm_p_values(standardized, ARCH_TEST_LAGS)
    squared_ljung_box_p = acorr_ljungbox(standardized**2, lags=list(ARCH_TEST_LAGS))[
        "lb_pvalue"
    ].to_numpy()
    clustering_left = (np.concatenate([arch_p, squared_ljung_box_p]) < SIGNIFICANCE_LEVEL).any()

    if best == CONSTANT:
        verdict = "CONSTANT VARIANCE"
    elif persistence >= 1 or clustering_left:
        verdict = "GARCH FALLS SHORT"
    else:
        verdict = "CARRY GARCH FORWARD"
    return {
        "information_criteria": criteria,
        "best": best,
        "garch_name": garch_name,
        "garch_params": garch.params,
        "persistence": persistence,
        "half_life": half_life(persistence),
        "garch_sd": candidates[garch_name]["sd"],
        "constant_sd": float(candidates[CONSTANT]["sd"][0]),
        "standardized": standardized,
        "arch_p": arch_p,
        "squared_ljung_box_p": squared_ljung_box_p,
        "verdict": verdict,
    }


def garch_report(results: list[dict], train: pd.DataFrame) -> str:
    dates = train[DATE_COLUMN]
    lines = [
        "GARCH(1,1) on the residuals of the models that passed the residual checks (step U5b)",
        (
            f"Training split only: {dates.min():%Y-%m} to {dates.max():%Y-%m}, "
            f"{len(train)} first differences."
        ),
        (
            "Two-step estimation: the SARIMA mean model as fitted, then each variance model on "
            "its residuals (zero mean). BIC picks the variance model; AIC is shown alongside."
        ),
        (
            "Break-adjusted: GARCH-t on the residuals divided by each regime's standard "
            "deviation, at the variance break from the volatility checks. Its likelihood is "
            "converted back to the residuals, and it counts 2 extra parameters (the regime scale "
            "ratio and the break date), so all four AIC / BIC values are comparable."
        ),
        (
            "Standardized residuals = residuals / GARCH's monthly standard deviation. If GARCH "
            "captured the clustering, ARCH-LM and Ljung-Box on their squares pass at "
            f"{SIGNIFICANCE_LEVEL:.0%} (lags {', '.join(map(str, ARCH_TEST_LAGS))})."
        ),
        "Half-life: months for a volatility shock to halve, log(0.5) / log(alpha + beta).",
        "",
    ]
    for column in VALUE_COLUMNS:
        for result in (r for r in results if r["series"] == column):
            params = result["garch_params"]
            lines.append(f"{column}  {model_label(result)}")
            for name, (aic, bic) in result["information_criteria"].items():
                marker = "  <- best by BIC" if name == result["best"] else ""
                lines.append(f"  {name:<38} AIC={aic:9.2f}  BIC={bic:9.2f}{marker}")
            shape = f", nu={params['nu']:.1f}" if "nu" in params else ""
            lines.append(
                f"  {result['garch_name']}: alpha={params['alpha[1]']:.3f}, "
                f"beta={params['beta[1]']:.3f}{shape}; persistence alpha + beta="
                f"{result['persistence']:.3f}, {describe_half_life(result['half_life'])}"
            )
            for lag, arch_p, squared_p in zip(
                ARCH_TEST_LAGS, result["arch_p"], result["squared_ljung_box_p"]
            ):
                lines.append(
                    f"  lag {lag:>2}: standardized residuals ARCH-LM p={arch_p:.4f}, "
                    f"Ljung-Box on squares p={squared_p:.4f}"
                )
            lines.append(f"  Verdict: {result['verdict']}. {VERDICTS[result['verdict']]}")
            lines.append("")
    lines.append(f"Figure: {repo_path(FIGURE_PATH)}")
    return "\n".join(lines)


def garch_figure(results: list[dict], dates: pd.Series) -> plt.Figure:
    """One row per model: absolute residuals against GARCH's and the constant standard
    deviation, and ARCH-LM p-value by lag before and after GARCH."""
    apply_chart_style()
    figure, axes = plt.subplots(
        len(results), 2, figsize=(14, 3.3 * len(results)), squeeze=False, facecolor=SURFACE
    )
    lags = np.arange(1, PLOT_LAGS + 1)
    for row, result in enumerate(results):
        residuals = result["residuals"]
        residual_dates = dates.iloc[residuals.index]
        time_axis, p_axis = axes[row]
        for axis in axes[row]:
            axis.set_facecolor(SURFACE)

        curves = [
            (np.abs(residuals.to_numpy()), SERIES_COLOR, 1, RESIDUALS_LABEL),
            (result["garch_sd"], ADJUSTED_COLOR, 2, GARCH_LABEL),
        ]
        for values, color, width, label in curves:
            time_axis.plot(residual_dates, values, color=color, linewidth=width, label=label)
        time_axis.axhline(result["constant_sd"], color=SECONDARY_INK, linewidth=1)
        time_axis.annotate(
            " constant SD",
            (residual_dates.iloc[0], result["constant_sd"]),
            va="bottom",
            color=SECONDARY_INK,
            fontsize=9,
        )
        bics = result["information_criteria"]
        time_axis.set_title(
            f"{model_name(result)}: {model_label(result)}  [{result['verdict']}]\n"
            f"Best by BIC: {result['best']}\n"
            f"BIC {bics[result['best']][1]:.1f} vs constant {bics[CONSTANT][1]:.1f}; "
            f"{describe_half_life(result['half_life'])}",
            loc="left",
            fontsize=10,
        )
        time_axis.set_ylabel("Trillions")

        before = arch_lm_p_values(residuals, lags)
        after = arch_lm_p_values(result["standardized"], lags)
        for p_values, color in ((before, SERIES_COLOR), (after, ADJUSTED_COLOR)):
            p_axis.plot(lags, p_values, color=color, linewidth=2, marker="o", markersize=5)
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
        p_axis.set_ylim(max(1e-300, min(before.min(), after.min()) / 3), 2)
        p_axis.set_xlim(0.5, PLOT_LAGS + 0.5)
        p_axis.set_xticks([1, 3, 6, 9, 12])
        p_axis.set_xlabel("Lag (months)")
        p_axis.set_title(
            "ARCH-LM p-value by lag (above the line: no ARCH effects left)",
            loc="left",
            fontsize=10,
        )

    # One legend for the whole figure, above the rows, so it never covers a curve.
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncols=2, frameon=False, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.35 / figure.get_figheight()))
    return figure


@app.command()
def main(
    levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv",
    differences_path: Path = PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv",
    candidates_path: Path = CANDIDATES_PATH,
) -> None:
    checked, train = checked_models(levels_path, differences_path, candidates_path)
    results = [
        {**result, **garch_check(result["residuals"])} for result in checked if result["passed"]
    ]

    report = garch_report(results, train)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = garch_figure(results, train[DATE_COLUMN])
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"GARCH checks written to {REPORT_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    app()
