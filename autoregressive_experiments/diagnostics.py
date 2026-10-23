"""
Pre-estimation diagnostics for SARIMA: stationarity, model identification and order selection.

Input: ``data/processed/fed_balance_sheet_first_difference.csv``.

Output, in ``reports/``:

- ``stationarity_tests.txt``: ADF and KPSS on each differenced series
- ``acf_pacf_analysis.txt`` and ``figures/fed_balance_sheet_acf_pacf.png``:
  sample ACF / PACF and their significant lags
- ``information_criteria_candidates.csv`` and ``information_criteria_summary.txt``:
  AIC / AICc / BIC of a SARIMA candidate grid, and the best model per criterion
"""

from itertools import product
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
import typer

from autoregressive_experiments.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
)

app = typer.Typer()

VALUE_COLUMNS = ("fed_total_assets_T", "monetary_base_T")
SIGNIFICANCE_LEVEL = 0.05
MAX_LAGS = 36
MAX_AR_ORDER = 2
MAX_MA_ORDER = 2
SEASONAL_AR_ORDERS = (0, 1)
SEASONAL_MA_ORDERS = (0, 1)
SEASONAL_PERIOD = 12
MAX_ITERATIONS = 200

ACF_PACF_FIGURE_PATH = FIGURES_DIR / "fed_balance_sheet_acf_pacf.png"
CANDIDATES_PATH = REPORTS_DIR / "information_criteria_candidates.csv"


def repo_path(path: Path) -> str:
    """Path relative to the project root, so reports read the same on every computer."""
    return path.relative_to(PROJ_ROOT).as_posix()


def series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        raise ValueError(f"Dataset is missing required column: {column}")
    return pd.to_numeric(dataframe[column], errors="raise").dropna()


def assess_stationarity(adf_p_value: float, kpss_p_value: float) -> str:
    # ADF's null is a unit root, KPSS's null is stationarity, so they agree when exactly one rejects.
    adf_stationary = adf_p_value < SIGNIFICANCE_LEVEL
    kpss_stationary = kpss_p_value >= SIGNIFICANCE_LEVEL
    if adf_stationary and kpss_stationary:
        return "Evidence supports stationarity (tests agree)."
    if not adf_stationary and not kpss_stationary:
        return "Evidence supports non-stationarity (tests agree)."
    return "Tests conflict; treat the stationarity result as inconclusive."


def significant_lags(intervals: np.ndarray) -> list[int]:
    """Lags (from 1) whose confidence interval excludes zero; row 0 is lag 0."""
    return [
        lag for lag in range(1, len(intervals)) if intervals[lag, 0] > 0 or intervals[lag, 1] < 0
    ]


def aicc(aic: float, parameter_count: int, observation_count: int) -> float:
    denominator = observation_count - parameter_count - 1
    if denominator <= 0:
        return np.nan
    return aic + (2 * parameter_count * (parameter_count + 1)) / denominator


def stationarity_report(dataframe: pd.DataFrame) -> str:
    report_lines = [
        "ADF and KPSS stationarity checks on first-order differences",
        f"Significance level: {SIGNIFICANCE_LEVEL:.0%}",
        "ADF null hypothesis: the series has a unit root (non-stationary).",
        "KPSS null hypothesis: the series is stationary.",
        "Both tests use a constant term; ADF lag selection uses AIC.",
        "",
    ]

    for column in VALUE_COLUMNS:
        values = series(dataframe, column)
        adf_result = adfuller(values, regression="c", autolag="AIC")
        kpss_result = kpss(values, regression="c", nlags="auto")
        report_lines.extend(
            [
                f"{column} (observations: {len(values)})",
                (
                    f"  ADF: statistic={adf_result[0]:.6f}, "
                    f"p-value={adf_result[1]:.6g}, selected lags={adf_result[2]}"
                ),
                f"  ADF critical values: {adf_result[4]}",
                (
                    f"  KPSS: statistic={kpss_result[0]:.6f}, "
                    f"p-value={kpss_result[1]:.6g}, lags={kpss_result[2]}"
                ),
                f"  KPSS critical values: {kpss_result[3]}",
                f"  Assessment: {assess_stationarity(adf_result[1], kpss_result[1])}",
                "",
            ]
        )

    return "\n".join(report_lines)


def acf_pacf_report(dataframe: pd.DataFrame) -> tuple[str, Figure]:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    report_lines = [
        "Sample ACF and PACF analysis for first-order differences",
        f"Significance level: {SIGNIFICANCE_LEVEL:.0%}",
        f"Maximum displayed lag: {MAX_LAGS}",
        "Bars outside the 95% confidence interval indicate significant autocorrelation.",
        (
            "Use significant ACF/PACF lag patterns as model-identification guidance, "
            "not an automatic order selection."
        ),
        "",
    ]

    for row, column in enumerate(VALUE_COLUMNS):
        values = series(dataframe, column)
        if len(values) < 4:
            raise ValueError(f"At least four usable values are required for {column}.")

        lag_count = min(MAX_LAGS, len(values) // 2 - 1)
        acf_values, acf_intervals = acf(
            values, nlags=lag_count, alpha=SIGNIFICANCE_LEVEL, fft=False
        )
        pacf_values, pacf_intervals = pacf(
            values, nlags=lag_count, alpha=SIGNIFICANCE_LEVEL, method="ywm"
        )

        plot_acf(
            values,
            lags=lag_count,
            alpha=SIGNIFICANCE_LEVEL,
            ax=axes[row, 0],
            title=f"{column} - ACF",
        )
        plot_pacf(
            values,
            lags=lag_count,
            alpha=SIGNIFICANCE_LEVEL,
            method="ywm",
            ax=axes[row, 1],
            title=f"{column} - PACF",
        )
        axes[row, 0].set_xlabel("Lag")
        axes[row, 1].set_xlabel("Lag")

        report_lines.extend(
            [
                f"{column} (observations: {len(values)})",
                (
                    "  Significant ACF lags: "
                    f"{significant_lags(acf_intervals) or 'none at the selected level'}"
                ),
                (
                    "  Significant PACF lags: "
                    f"{significant_lags(pacf_intervals) or 'none at the selected level'}"
                ),
                f"  ACF values, lags 1-{lag_count}: {np.round(acf_values[1:], 4).tolist()}",
                f"  PACF values, lags 1-{lag_count}: {np.round(pacf_values[1:], 4).tolist()}",
                "",
            ]
        )

    figure.tight_layout()
    report_lines.extend(
        [
            f"ACF/PACF figure saved to: {repo_path(ACF_PACF_FIGURE_PATH)}",
            "Interpret the ACF and PACF shapes together when considering candidate ARMA orders.",
        ]
    )
    return "\n".join(report_lines), figure


def information_criteria(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    candidate_orders = list(
        product(
            range(MAX_AR_ORDER + 1),
            range(MAX_MA_ORDER + 1),
            SEASONAL_AR_ORDERS,
            SEASONAL_MA_ORDERS,
        )
    )
    results: list[dict[str, object]] = []

    for column in VALUE_COLUMNS:
        values = series(dataframe, column)
        if len(values) <= 2 * SEASONAL_PERIOD:
            raise ValueError(
                f"At least {2 * SEASONAL_PERIOD + 1} observations are required for "
                f"seasonal model selection; {column} has {len(values)}."
            )

        for p, q, seasonal_p, seasonal_q in candidate_orders:
            fitted_model = SARIMAX(
                values,
                order=(p, 0, q),
                seasonal_order=(seasonal_p, 0, seasonal_q, SEASONAL_PERIOD),
                trend="c",
                enforce_stationarity=True,
                enforce_invertibility=True,
            ).fit(disp=False, maxiter=MAX_ITERATIONS)

            parameter_count = len(fitted_model.params)
            observation_count = int(fitted_model.nobs)
            aic = float(fitted_model.aic)
            results.append(
                {
                    "series": column,
                    "p": p,
                    "d": 0,
                    "q": q,
                    "P": seasonal_p,
                    "D": 0,
                    "Q": seasonal_q,
                    "s": SEASONAL_PERIOD,
                    "nobs": observation_count,
                    "parameter_count": parameter_count,
                    "converged": bool(fitted_model.mle_retvals.get("converged", False)),
                    "AIC": aic,
                    "AICc": aicc(aic, parameter_count, observation_count),
                    "BIC": float(fitted_model.bic),
                }
            )

    results_frame = pd.DataFrame(results)
    report_lines = [
        "SARIMA information criteria model selection",
        "Input: first-order-differenced balance-sheet data.",
        "Therefore all candidates use d=0 (equivalent to d=1 from original levels).",
        f"Monthly seasonal period: {SEASONAL_PERIOD}; all candidates use D=0.",
        (
            f"Candidate orders: p=0..{MAX_AR_ORDER}, q=0..{MAX_MA_ORDER}, "
            f"P in {SEASONAL_AR_ORDERS}, Q in {SEASONAL_MA_ORDERS}."
        ),
        "Lower AIC, AICc, and BIC indicate a preferred fit/complexity tradeoff.",
        "AICc applies the finite-sample correction to AIC.",
        "",
    ]

    for column in VALUE_COLUMNS:
        series_results = results_frame[results_frame["series"] == column]
        converged_results = series_results[series_results["converged"]]
        if converged_results.empty:
            raise ValueError(f"No candidate SARIMA models converged for {column}.")

        report_lines.append(
            f"{column}: {len(converged_results)} of {len(series_results)} candidates converged."
        )
        for criterion in ("AIC", "AICc", "BIC"):
            eligible = converged_results[np.isfinite(converged_results[criterion])]
            if eligible.empty:
                raise ValueError(f"No finite {criterion} values are available for {column}.")
            best = eligible.loc[eligible[criterion].idxmin()]
            report_lines.append(
                f"  Best by {criterion}: "
                f"SARIMA({int(best['p'])},0,{int(best['q'])})"
                f"x({int(best['P'])},0,{int(best['Q'])},{int(best['s'])}), "
                f"{criterion}={best[criterion]:.4f}"
            )
        report_lines.append("")

    report_lines.append(f"Full candidate table: {repo_path(CANDIDATES_PATH)}")
    return results_frame, "\n".join(report_lines)


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv",
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"First-difference dataset not found at {input_path}. Generate it with make features."
        )
    dataframe = pd.read_csv(input_path)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    report = stationarity_report(dataframe)
    (REPORTS_DIR / "stationarity_tests.txt").write_text(report, encoding="utf-8")
    print(report)

    report, figure = acf_pacf_report(dataframe)
    figure.savefig(ACF_PACF_FIGURE_PATH, dpi=150)
    plt.close(figure)
    (REPORTS_DIR / "acf_pacf_analysis.txt").write_text(report, encoding="utf-8")
    print(report)

    candidates, report = information_criteria(dataframe)
    candidates.to_csv(CANDIDATES_PATH, index=False)
    (REPORTS_DIR / "information_criteria_summary.txt").write_text(report, encoding="utf-8")
    print(report)

    logger.success(f"Diagnostics written to {REPORTS_DIR}")


if __name__ == "__main__":
    app()
