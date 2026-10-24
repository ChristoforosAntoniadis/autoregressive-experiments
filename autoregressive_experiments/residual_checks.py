"""
Residual checks (step U4): do the recommended SARIMA models leave autocorrelation behind?

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` and
``data/processed/fed_balance_sheet_first_difference.csv`` (training split only), and
``reports/information_criteria_candidates.csv`` from ``make diagnostics``.

Output, in ``reports/``:

- ``residual_diagnostics.txt``: Ljung-Box and Breusch-Godfrey p-values per model and lag,
  a pass / fail verdict, and the models to carry forward
- ``figures/residual_diagnostics.png``: per model, the residuals over time, their ACF, and
  the Ljung-Box p-value at every lag against the 5% line
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox
from statsmodels.tsa.stattools import acf
import typer

from autoregressive_experiments.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from autoregressive_experiments.diagnostics import (
    CANDIDATES_PATH,
    SEASONAL_PERIOD,
    SIGNIFICANCE_LEVEL,
    VALUE_COLUMNS,
    fit_sarima,
    repo_path,
    series,
)
from autoregressive_experiments.features import DATE_COLUMN, split_boundaries, split_by_time

app = typer.Typer()

TEST_LAGS = (6, 12, 24)
# Burnham & Anderson: a model within 2 AIC of the best still has substantial support.
AIC_TIE_MARGIN = 2.0
PLOT_LAGS = 24

REPORT_PATH = REPORTS_DIR / "residual_diagnostics.txt"
FIGURE_PATH = FIGURES_DIR / "residual_diagnostics.png"

# Reference palette: slot 1 for the data, the reserved "critical" status color for failures.
SERIES_COLOR = "#2a78d6"
CRITICAL_COLOR = "#d03b3b"
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def model_label(model: dict) -> str:
    p, d, q, seasonal_p, seasonal_q = (model[k] for k in ("p", "d", "q", "P", "Q"))
    seasonal = (
        f"x({seasonal_p},0,{seasonal_q},{SEASONAL_PERIOD})" if seasonal_p or seasonal_q else ""
    )
    return f"SARIMA({p},{d},{q}){seasonal}"


def recommended_orders(candidates: pd.DataFrame) -> list[dict]:
    """Per series: the best converged model by AIC, AICc and BIC, plus the AIC runner-up if tied."""
    models: list[dict] = []
    for column, group in candidates[candidates["converged"]].groupby("series", sort=False):
        picks: dict[tuple, list[str]] = {}
        for criterion in ("AIC", "AICc", "BIC"):
            best = group.loc[group[criterion].idxmin()]
            picks.setdefault(order_key(best), []).append(criterion)
        ranked = group.sort_values("AIC")
        gap = ranked["AIC"].iloc[1] - ranked["AIC"].iloc[0]
        if gap < AIC_TIE_MARGIN:
            # Three decimals, so a gap of 1.997 doesn't print as a margin-breaking "+2.00".
            picks.setdefault(order_key(ranked.iloc[1]), []).append(f"AIC tie (+{gap:.3f})")
        for (p, d, q, seasonal_p, seasonal_q), reasons in picks.items():
            models.append(
                {
                    "series": column,
                    "p": p,
                    "d": d,
                    "q": q,
                    "P": seasonal_p,
                    "Q": seasonal_q,
                    "reason": ", ".join(reasons),
                }
            )
    return models


def order_key(row: pd.Series) -> tuple[int, int, int, int, int]:
    return tuple(int(row[k]) for k in ("p", "d", "q", "P", "Q"))


def residual_tests(residuals: pd.Series, arma_terms: int) -> pd.DataFrame:
    """Ljung-Box and Breusch-Godfrey p-values at TEST_LAGS; a NaN means too few degrees of freedom."""
    # Ljung-Box's chi-squared loses one degree of freedom per fitted ARMA coefficient.
    ljung_box = acorr_ljungbox(residuals, lags=list(TEST_LAGS), model_df=arma_terms)
    # statsmodels only offers Breusch-Godfrey on regression results, so regress the residuals on
    # a constant: its auxiliary regression is then residuals on their own lags (n * R^2 ~ chi2).
    constant_fit = OLS(np.asarray(residuals), np.ones(len(residuals))).fit()
    return pd.DataFrame(
        {
            "lag": TEST_LAGS,
            "ljung_box_p": ljung_box["lb_pvalue"].to_numpy(),
            "breusch_godfrey_p": [
                acorr_breusch_godfrey(constant_fit, nlags=lag, result_object=True).lmpval
                for lag in TEST_LAGS
            ],
        }
    )


def passes(tests: pd.DataFrame) -> bool:
    p_values = tests[["ljung_box_p", "breusch_godfrey_p"]].to_numpy(dtype=float)
    return bool((p_values[np.isfinite(p_values)] >= SIGNIFICANCE_LEVEL).all())


def check_model(values: pd.Series, model: dict) -> dict:
    fitted = fit_sarima(values, model["p"], model["q"], model["P"], model["Q"])
    residuals = fitted.resid.iloc[fitted.loglikelihood_burn :]
    arma_terms = model["p"] + model["q"] + model["P"] + model["Q"]
    tests = residual_tests(residuals, arma_terms)
    return {
        **model,
        "residuals": residuals,
        "arma_terms": arma_terms,
        "tests": tests,
        "passed": passes(tests),
    }


def check_series(
    values: pd.Series, column: str, candidates: pd.DataFrame, models: list[dict]
) -> list[dict]:
    """Check the recommended models; if none passes, walk down the AIC ranking to the first that does."""
    checked = [check_model(values, model) for model in models]
    if any(result["passed"] for result in checked):
        return checked

    tried = {order_key(pd.Series(model)) for model in models}
    ranked = candidates[(candidates["series"] == column) & candidates["converged"]]
    for rank, (_, row) in enumerate(ranked.sort_values("AIC").iterrows(), start=1):
        if order_key(row) in tried:
            continue
        p, d, q, seasonal_p, seasonal_q = order_key(row)
        reason = f"best passing alternative (AIC rank {rank})"
        model = {"series": column, "p": p, "d": d, "q": q, "P": seasonal_p, "Q": seasonal_q}
        result = check_model(values, {**model, "reason": reason})
        if result["passed"]:
            return [*checked, result]
    return checked


def residual_report(results: list[dict], train: pd.DataFrame) -> str:
    dates = train[DATE_COLUMN]
    lines = [
        "Residual checks of the recommended SARIMA models (step U4)",
        (
            f"Training split only: {dates.min():%Y-%m} to {dates.max():%Y-%m}, "
            f"{len(train)} first differences. Models are fitted exactly as in the information-"
            "criteria grid."
        ),
        (
            "Recommended = best by AIC, AICc and BIC, plus the AIC runner-up when it is within "
            f"{AIC_TIE_MARGIN:g} AIC of the best."
        ),
        (
            "Both tests' null hypothesis: no autocorrelation up to the lag. "
            f"Significance level: {SIGNIFICANCE_LEVEL:.0%}."
        ),
        "Ljung-Box degrees of freedom = lag - (p + q + P + Q); n/a when that is not positive.",
        (
            "Breusch-Godfrey is the auxiliary regression of the residuals on their own lags; for "
            "models with MA terms this is an approximation of the exact LM test."
        ),
        "A model passes when every available p-value is at or above the significance level.",
        "",
    ]
    for column in VALUE_COLUMNS:
        series_results = [result for result in results if result["series"] == column]
        for result in series_results:
            verdict = "PASS" if result["passed"] else "FAIL: residual autocorrelation remains"
            lines.append(f"{column}  {model_label(result)}  ({result['reason']})")
            lines.append(
                f"  Residuals: {len(result['residuals'])}, ARMA terms: {result['arma_terms']}"
            )
            for row in result["tests"].itertuples():
                lines.append(
                    f"  lag {row.lag:>2}: Ljung-Box p={format_p(row.ljung_box_p)}, "
                    f"Breusch-Godfrey p={format_p(row.breusch_godfrey_p)}"
                )
            lines.append(f"  Verdict: {verdict}")
            lines.append("")
        carried = [model_label(result) for result in series_results if result["passed"]]
        lines.append(
            f"{column}: carry forward {', '.join(carried)}"
            if carried
            else f"{column}: no candidate within p + d + q <= 5 passes; revisit the grid."
        )
        lines.append("")
    lines.append(f"Figure: {repo_path(FIGURE_PATH)}")
    return "\n".join(lines)


def format_p(p_value: float) -> str:
    return "n/a" if not np.isfinite(p_value) else f"{p_value:.4f}"


def apply_chart_style() -> None:
    """Recessive hairline grid and axes in the reference palette's chrome colors."""
    plt.rcParams.update(
        {
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": SECONDARY_INK,
            "axes.titlecolor": PRIMARY_INK,
            "xtick.color": MUTED_INK,
            "ytick.color": MUTED_INK,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def model_name(result: dict) -> str:
    return result["series"].removesuffix("_T").replace("_", " ")


def residual_figure(results: list[dict], dates: pd.Series) -> plt.Figure:
    """One row per model: residuals over time, residual ACF, and Ljung-Box p-value by lag."""
    apply_chart_style()
    figure, axes = plt.subplots(
        len(results), 3, figsize=(15, 3.1 * len(results)), squeeze=False, facecolor=SURFACE
    )
    for row, result in enumerate(results):
        residuals = result["residuals"]
        residual_dates = dates.iloc[residuals.index]
        verdict = "PASS" if result["passed"] else "FAIL"
        name = model_name(result)
        time_axis, acf_axis, p_axis = axes[row]
        for axis in axes[row]:
            axis.set_facecolor(SURFACE)

        time_axis.plot(residual_dates, residuals, color=SERIES_COLOR, linewidth=1)
        time_axis.axhline(0, color=BASELINE, linewidth=0.8)
        time_axis.set_title(
            f"{name}: {model_label(result)}  [{verdict}]\n{result['reason']}",
            loc="left",
            fontsize=10,
        )
        time_axis.set_ylabel("Residual (trillions)")

        autocorrelations = acf(residuals, nlags=PLOT_LAGS, fft=False)[1:]
        band = 1.96 / np.sqrt(len(residuals))
        lags = np.arange(1, PLOT_LAGS + 1)
        outside = np.abs(autocorrelations) > band
        acf_axis.axhspan(-band, band, color=GRIDLINE, alpha=0.7, linewidth=0)
        acf_axis.vlines(lags, 0, autocorrelations, color=SERIES_COLOR, linewidth=2)
        acf_axis.scatter(
            lags[outside], autocorrelations[outside], color=CRITICAL_COLOR, s=30, zorder=3
        )
        acf_axis.axhline(0, color=BASELINE, linewidth=0.8)
        acf_axis.set_title(
            "Residual ACF (shaded: 95% band; red: outside)", loc="left", fontsize=10
        )
        acf_axis.set_xlabel("Lag (months)")

        p_values = acorr_ljungbox(residuals, lags=PLOT_LAGS, model_df=result["arma_terms"])[
            "lb_pvalue"
        ].to_numpy()
        failing = p_values < SIGNIFICANCE_LEVEL
        p_axis.scatter(lags[~failing], p_values[~failing], color=SERIES_COLOR, s=24, zorder=3)
        p_axis.scatter(lags[failing], p_values[failing], color=CRITICAL_COLOR, s=24, zorder=3)
        p_axis.axhline(SIGNIFICANCE_LEVEL, color=CRITICAL_COLOR, linewidth=1)
        p_axis.text(1, SIGNIFICANCE_LEVEL * 1.3, "5% level", color=SECONDARY_INK, fontsize=9)
        p_axis.set_yscale("log")
        p_axis.set_ylim(1e-4, 1.5)
        p_axis.set_title(
            "Ljung-Box p-value by lag (below the line: fails)", loc="left", fontsize=10
        )
        p_axis.set_xlabel("Lag (months)")
        # Same lag axis everywhere; lags up to the ARMA term count have no Ljung-Box p-value.
        for axis in (acf_axis, p_axis):
            axis.set_xlim(0.5, PLOT_LAGS + 0.5)
            axis.set_xticks([1, 6, 12, 18, 24])

    figure.tight_layout()
    return figure


def checked_models(
    levels_path: Path, differences_path: Path, candidates_path: Path
) -> tuple[list[dict], pd.DataFrame]:
    """Fit and check the recommended models on the training split; also returns that split."""
    required = {
        levels_path: "features",
        differences_path: "features",
        candidates_path: "diagnostics",
    }
    for path, command in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Generate it with make {command}.")
    levels = pd.read_csv(levels_path, parse_dates=[DATE_COLUMN])
    train = split_by_time(
        pd.read_csv(differences_path, parse_dates=[DATE_COLUMN]), split_boundaries(levels)
    )["train"]
    candidates = pd.read_csv(candidates_path)
    models = recommended_orders(candidates)

    results = []
    for column in VALUE_COLUMNS:
        series_models = [model for model in models if model["series"] == column]
        results.extend(check_series(series(train, column), column, candidates, series_models))
    return results, train


@app.command()
def main(
    levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv",
    differences_path: Path = PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv",
    candidates_path: Path = CANDIDATES_PATH,
) -> None:
    results, train = checked_models(levels_path, differences_path, candidates_path)

    report = residual_report(results, train)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = residual_figure(results, train[DATE_COLUMN])
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"Residual checks written to {REPORT_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    app()
