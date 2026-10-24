"""
Pre-estimation diagnostics for SARIMA: stationarity, model identification and order selection.

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` (levels) and
``data/processed/fed_balance_sheet_first_difference.csv``. Only the training split
(the first 60% of months) is used, so validation and test stay unseen until evaluation.

Output, in ``reports/``:

- ``stationarity_tests.txt``: ADF and KPSS on each series, in levels and in first differences
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
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
)
from autoregressive_experiments.features import DATE_COLUMN, split_boundaries, split_by_time

app = typer.Typer()

VALUE_COLUMNS = ("fed_total_assets_T", "monetary_base_T")
SIGNIFICANCE_LEVEL = 0.05
MAX_LAGS = 36
# Every non-seasonal order with p + d + q <= MAX_ORDER_SUM. d = 1 comes from the stationarity
# tests (both series are I(1)) and is applied by fitting on the first differences.
MAX_ORDER_SUM = 5
DIFFERENCING_ORDER = 1
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


def split_header(train: pd.DataFrame, what: str) -> str:
    dates = train[DATE_COLUMN]
    return (
        f"Training split only (first 60% of months), {what}: "
        f"{dates.min():%Y-%m} to {dates.max():%Y-%m}, {len(train)} observations.\n"
        "Validation and test months are held out for model evaluation.\n\n"
    )


def stationarity_report(dataframe: pd.DataFrame, description: str, regression: str) -> str:
    """ADF and KPSS on each series; regression is "c" (constant) or "ct" (constant and trend)."""
    deterministic_terms = {"c": "a constant", "ct": "a constant and a linear trend"}[regression]
    report_lines = [
        f"ADF and KPSS stationarity checks on {description}",
        f"Significance level: {SIGNIFICANCE_LEVEL:.0%}",
        "ADF null hypothesis: the series has a unit root (non-stationary).",
        "KPSS null hypothesis: the series is stationary.",
        f"Both tests use {deterministic_terms}; ADF lag selection uses AIC.",
        "",
    ]

    for column in VALUE_COLUMNS:
        values = series(dataframe, column)
        adf_result = adfuller(values, regression=regression, autolag="AIC")
        kpss_result = kpss(values, regression=regression, nlags="auto")
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


def nonseasonal_orders() -> list[tuple[int, int]]:
    """Every (p, q) with p + d + q <= MAX_ORDER_SUM, for the fixed d = DIFFERENCING_ORDER."""
    budget = MAX_ORDER_SUM - DIFFERENCING_ORDER
    return [(p, q) for p in range(budget + 1) for q in range(budget + 1 - p)]


def fit_sarima(values: pd.Series, p: int, q: int, seasonal_p: int, seasonal_q: int):
    """ARMA(p, q)x(P, 0, Q, 12) with a constant, fitted on the first differences (so d = 1)."""
    return SARIMAX(
        values,
        order=(p, 0, q),
        seasonal_order=(seasonal_p, 0, seasonal_q, SEASONAL_PERIOD),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(disp=False, maxiter=MAX_ITERATIONS)


def information_criteria(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    candidate_orders = [
        (p, q, seasonal_p, seasonal_q)
        for (p, q), seasonal_p, seasonal_q in product(
            nonseasonal_orders(), SEASONAL_AR_ORDERS, SEASONAL_MA_ORDERS
        )
    ]
    results: list[dict[str, object]] = []

    for column in VALUE_COLUMNS:
        values = series(dataframe, column)
        if len(values) <= 2 * SEASONAL_PERIOD:
            raise ValueError(
                f"At least {2 * SEASONAL_PERIOD + 1} observations are required for "
                f"seasonal model selection; {column} has {len(values)}."
            )

        for p, q, seasonal_p, seasonal_q in candidate_orders:
            fitted_model = fit_sarima(values, p, q, seasonal_p, seasonal_q)

            parameter_count = len(fitted_model.params)
            observation_count = int(fitted_model.nobs)
            aic = float(fitted_model.aic)
            results.append(
                {
                    "series": column,
                    "p": p,
                    "d": DIFFERENCING_ORDER,
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
        (
            f"d={DIFFERENCING_ORDER} is applied by the differencing: each candidate is an "
            f"ARMA(p, q) with a constant on the differences, i.e. ARIMA(p, {DIFFERENCING_ORDER}, q) "
            "with drift on the levels."
        ),
        f"Monthly seasonal period: {SEASONAL_PERIOD}; all candidates use D=0.",
        (
            f"Candidate orders: every p, q with p + d + q <= {MAX_ORDER_SUM} "
            f"({len(nonseasonal_orders())} pairs), P in {SEASONAL_AR_ORDERS}, "
            f"Q in {SEASONAL_MA_ORDERS}: {len(candidate_orders)} candidates per series."
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
                f"SARIMA({int(best['p'])},{int(best['d'])},{int(best['q'])})"
                f"x({int(best['P'])},0,{int(best['Q'])},{int(best['s'])}), "
                f"{criterion}={best[criterion]:.4f}"
            )
        report_lines.append("")

    report_lines.append(f"Full candidate table: {repo_path(CANDIDATES_PATH)}")
    return results_frame, "\n".join(report_lines)


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
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Trending levels are tested against trend-stationarity; their differences against a constant mean.
    report = (
        split_header(train_levels, "levels")
        + stationarity_report(train_levels, "levels", regression="ct")
        + "\n"
        + split_header(train_differences, "first differences")
        + stationarity_report(train_differences, "first-order differences", regression="c")
    )
    (REPORTS_DIR / "stationarity_tests.txt").write_text(report, encoding="utf-8")
    print(report)

    report, figure = acf_pacf_report(train_differences)
    report = split_header(train_differences, "first differences") + report
    figure.savefig(ACF_PACF_FIGURE_PATH, dpi=150)
    plt.close(figure)
    (REPORTS_DIR / "acf_pacf_analysis.txt").write_text(report, encoding="utf-8")
    print(report)

    candidates, report = information_criteria(train_differences)
    report = split_header(train_differences, "first differences") + report
    candidates.to_csv(CANDIDATES_PATH, index=False)
    (REPORTS_DIR / "information_criteria_summary.txt").write_text(report, encoding="utf-8")
    print(report)

    logger.success(f"Diagnostics written to {REPORTS_DIR}")


if __name__ == "__main__":
    app()
