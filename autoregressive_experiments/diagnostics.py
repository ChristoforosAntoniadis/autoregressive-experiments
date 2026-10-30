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
from typing import Literal

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
import typer

from autoregressive_experiments.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
)
from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS

# VALUE_COLUMNS (the two series) is defined once, in features; the other analysis modules
# import it from here.
from autoregressive_experiments.features import (
    DATE_COLUMN,
    VALUE_COLUMNS,
    split_boundaries,
    split_by_time,
)
from autoregressive_experiments.runlog import (
    Progress,
    begin_phase,
    finish_run,
    start_run_log,
)

app = typer.Typer()

# Settings; each value's reasoning is in configs/hyperparameters.json.
# The level for every test and confidence band in the project.
SIGNIFICANCE_LEVEL = HYPERPARAMETERS.tests.significance_level
# Lags shown in the ACF / PACF analysis.
MAX_LAGS = HYPERPARAMETERS.diagnostics.max_acf_lags
# Every non-seasonal order with p + d + q <= MAX_ORDER_SUM; d is applied by fitting on the
# first differences.
MAX_ORDER_SUM = HYPERPARAMETERS.diagnostics.max_order_sum
DIFFERENCING_ORDER = HYPERPARAMETERS.diagnostics.differencing_order
# Seasonal AR (P) and MA (Q) orders tried; no seasonal differencing (D = 0).
SEASONAL_AR_ORDERS = HYPERPARAMETERS.diagnostics.seasonal_ar_orders
SEASONAL_MA_ORDERS = HYPERPARAMETERS.diagnostics.seasonal_ma_orders
SEASONAL_PERIOD = HYPERPARAMETERS.diagnostics.seasonal_period
# The optimizer's iteration cap per fit; a fit that hits it is recorded as not converged.
MAX_ITERATIONS = HYPERPARAMETERS.diagnostics.max_iterations
# How the stationarity tests pick their lag length / bandwidth, and the PACF estimator.
ADF_AUTOLAG = HYPERPARAMETERS.diagnostics.adf_autolag
KPSS_NLAGS = HYPERPARAMETERS.diagnostics.kpss_nlags
PACF_METHOD = HYPERPARAMETERS.diagnostics.pacf_method

ACF_PACF_FIGURE_PATH = FIGURES_DIR / "fed_balance_sheet_acf_pacf.png"
CANDIDATES_PATH = REPORTS_DIR / "information_criteria_candidates.csv"


def repo_path(path: Path) -> str:
    """
    Path relative to the project root, so reports read the same on every computer.

    Parameters
    ----------
    path : Path
        An absolute path inside the project, e.g. REPORTS_DIR / "stationarity_tests.txt".

    Returns
    -------
    str
        The path from the project root with forward slashes, e.g.
        "reports/stationarity_tests.txt".

    Raises
    ------
    ValueError
        If the path isn't inside PROJ_ROOT (raised by Path.relative_to).

    Notes
    -----
    Forward slashes (as_posix) keep the committed reports identical whether they were written
    on Windows or elsewhere, so re-running doesn't show up as a change in git.
    """
    return path.relative_to(PROJ_ROOT).as_posix()


def series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """
    One column as numbers, without missing values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A table holding the column, e.g. the training levels or differences.
    column : str
        The column to take, one of VALUE_COLUMNS.

    Returns
    -------
    pd.Series
        The column's values as numbers (trillions USD), with missing values dropped and the
        original row index kept.

    Raises
    ------
    ValueError
        If the column is missing, or (from pd.to_numeric) if a value isn't a number.
    """
    if column not in dataframe.columns:
        raise ValueError(f"Dataset is missing required column: {column}")
    return pd.to_numeric(dataframe[column], errors="raise").dropna()


def assess_stationarity(adf_p_value: float, kpss_p_value: float) -> str:
    """
    Combine the ADF and KPSS p-values into one verdict.

    Parameters
    ----------
    adf_p_value : float
        ADF p-value; small means "reject the unit root", i.e. evidence of stationarity.
    kpss_p_value : float
        KPSS p-value; small means "reject stationarity".

    Returns
    -------
    str
        "Evidence supports stationarity (tests agree).", "Evidence supports non-stationarity
        (tests agree)." or "Tests conflict; ...".

    Notes
    -----
    The two tests have opposite null hypotheses, so using both guards against each one's
    weakness: ADF has little power to reject a unit root in short samples, and KPSS tends to
    reject stationarity in persistent but stationary series. Only a pair that points the same
    way counts as a verdict.
    """
    # ADF's null is a unit root, KPSS's null is stationarity, so they agree when exactly one
    # rejects.
    adf_stationary = adf_p_value < SIGNIFICANCE_LEVEL
    kpss_stationary = kpss_p_value >= SIGNIFICANCE_LEVEL
    if adf_stationary and kpss_stationary:
        return "Evidence supports stationarity (tests agree)."
    if not adf_stationary and not kpss_stationary:
        return "Evidence supports non-stationarity (tests agree)."
    return "Tests conflict; treat the stationarity result as inconclusive."


def significant_lags(intervals: np.ndarray) -> list[int]:
    """
    Lags (from 1) whose confidence interval excludes zero; row 0 is lag 0.

    Parameters
    ----------
    intervals : np.ndarray
        Shape (lags + 1, 2): the lower and upper confidence bound of the ACF or PACF at each
        lag, as returned by statsmodels' acf / pacf with ``alpha``. Row 0 is lag 0.

    Returns
    -------
    list of int
        The lags, in increasing order, whose interval lies entirely above or below zero.

    Notes
    -----
    Lag 0 is skipped because its autocorrelation is 1 by definition. statsmodels centres
    these intervals on the estimate, so "the interval excludes zero" is the same test as
    "the bar lies outside the shaded band" in plot_acf / plot_pacf.
    """
    return [
        lag for lag in range(1, len(intervals)) if intervals[lag, 0] > 0 or intervals[lag, 1] < 0
    ]


def aicc(aic: float, parameter_count: int, observation_count: int) -> float:
    """
    AIC with the small-sample correction (AICc).

    Parameters
    ----------
    aic : float
        The model's AIC.
    parameter_count : int
        k, the number of estimated parameters, including the constant and the shock
        variance.
    observation_count : int
        n, the number of observations the model was fitted on.

    Returns
    -------
    float
        AIC + 2k(k + 1) / (n - k - 1), or NaN when n - k - 1 <= 0 (the correction is
        undefined there).

    Notes
    -----
    AIC's penalty is derived for large samples and lets models with many parameters through
    too easily when n isn't much larger than k. The extra term grows quickly as k approaches
    n, and vanishes as n grows, so AICc tends to AIC for long series. With n = 166 training
    months and k up to 8 here, it is a small but real correction.
    """
    denominator = observation_count - parameter_count - 1
    if denominator <= 0:
        return np.nan
    return aic + (2 * parameter_count * (parameter_count + 1)) / denominator


def split_header(train: pd.DataFrame, what: str) -> str:
    """
    The header that opens each report: which split, which months, how many observations.

    Parameters
    ----------
    train : pd.DataFrame
        The training split, with ``date`` parsed to datetime.
    what : str
        What the rows are, e.g. "levels" or "first differences".

    Returns
    -------
    str
        Two lines and a blank line, naming the first and last month (YYYY-MM) and the
        number of rows, and saying validation and test are held out.
    """
    dates = train[DATE_COLUMN]
    return (
        f"Training split only (first 60% of months), {what}: "
        f"{dates.min():%Y-%m} to {dates.max():%Y-%m}, {len(train)} observations.\n"
        "Validation and test months are held out for model evaluation.\n\n"
    )


def stationarity_report(
    dataframe: pd.DataFrame, description: str, regression: Literal["c", "ct"]
) -> str:
    """
    ADF and KPSS on each series; regression is "c" (constant) or "ct" (constant and trend).

    Parameters
    ----------
    dataframe : pd.DataFrame
        A table holding both VALUE_COLUMNS (levels or first differences, trillions USD).
    description : str
        What the data is, for the report's first line, e.g. "levels".
    regression : {"c", "ct"}
        The deterministic terms both tests include: "c" for a constant, "ct" for a constant
        and a linear trend.

    Returns
    -------
    str
        The report: the hypotheses, then per series the statistic, p-value, lags and
        critical values of each test, and the combined verdict.

    Raises
    ------
    KeyError
        If ``regression`` is neither "c" nor "ct".
    ValueError
        If a value column is missing (raised by series()).

    Notes
    -----
    Processing:
    1. Write the header: significance level, both null hypotheses, the deterministic terms.
    2. For each series: run ADF (lag count chosen by AIC) and KPSS (bandwidth chosen
       automatically), and add their results and assess_stationarity()'s verdict.

    The deterministic terms must match the alternative being tested: trending levels are
    tested against trend-stationarity ("ct"), otherwise a deterministic trend would be
    mistaken for a unit root; differences have no trend, so a constant ("c") is enough.
    KPSS p-values come from a lookup table and are capped at 0.01 and 0.1, which is why
    statsmodels warns (InterpolationWarning) at the extremes.
    """
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
        # result_object=True: named fields (statistic, pvalue, lags, critical_values) instead of
        # plain tuples, which statsmodels 0.16 drops.
        adf_result = adfuller(
            values, regression=regression, autolag=ADF_AUTOLAG, result_object=True
        )
        kpss_result = kpss(values, regression=regression, nlags=KPSS_NLAGS, result_object=True)
        report_lines.extend(
            [
                f"{column} (observations: {len(values)})",
                (
                    f"  ADF: statistic={adf_result.statistic:.6f}, "
                    f"p-value={adf_result.pvalue:.6g}, selected lags={adf_result.lags}"
                ),
                f"  ADF critical values: {adf_result.critical_values}",
                (
                    f"  KPSS: statistic={kpss_result.statistic:.6f}, "
                    f"p-value={kpss_result.pvalue:.6g}, lags={kpss_result.lags}"
                ),
                f"  KPSS critical values: {kpss_result.critical_values}",
                f"  Assessment: {assess_stationarity(adf_result.pvalue, kpss_result.pvalue)}",
                "",
            ]
        )

    return "\n".join(report_lines)


def acf_pacf_report(dataframe: pd.DataFrame) -> tuple[str, Figure]:
    """
    Sample ACF and PACF of both series: a text report and a 2 x 2 figure.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The training first differences, holding both VALUE_COLUMNS (trillions USD).

    Returns
    -------
    tuple of (str, Figure)
        The report (significant lags and the values at lags 1..L, per series) and the
        figure: one row per series, ACF on the left, PACF on the right. The caller saves
        and closes the figure.

    Raises
    ------
    ValueError
        If a series has fewer than four values, or a value column is missing.

    Notes
    -----
    Processing:
    1. For each series, choose L = min(MAX_LAGS, n // 2 - 1) lags.
    2. Compute the ACF and PACF with their 95% intervals, and draw both into the figure.
    3. Report the lags whose interval excludes zero (significant_lags), and the values.
    4. Lay out the figure and add the figure path to the report.

    The cap n // 2 - 1 is because statsmodels' PACF only allows fewer than n / 2 lags.
    The PACF uses method "ywm" (Yule-Walker with the biased, divide-by-n autocovariances),
    which always yields a valid PACF within [-1, 1], in both the numbers and the plot.
    fft=False computes the ACF directly; on about 166 points speed doesn't matter.
    """
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
        # With alpha, both give the values and their intervals, shape (L + 1, 2); acf's result
        # object replaces the plain tuple that statsmodels 0.16 drops.
        acf_result = acf(
            values, nlags=lag_count, alpha=SIGNIFICANCE_LEVEL, fft=False, result_object=True
        )
        acf_values, acf_intervals = acf_result.acf, acf_result.confint
        pacf_values, pacf_intervals = pacf(
            values, nlags=lag_count, alpha=SIGNIFICANCE_LEVEL, method=PACF_METHOD
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
                # [1:] skips lag 0, which is always 1.
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
    """
    Every (p, q) with p + d + q <= MAX_ORDER_SUM, for the fixed d = DIFFERENCING_ORDER.

    Returns
    -------
    list of tuple of (int, int)
        The (p, q) pairs, ordered by p, then q. With MAX_ORDER_SUM 5 and d = 1, p + q <= 4:
        15 pairs, from (0, 0) to (4, 0).

    Notes
    -----
    A total-order budget, rather than separate caps on p and q, covers pure AR, pure MA and
    mixed models of the same total complexity.
    """
    budget = MAX_ORDER_SUM - DIFFERENCING_ORDER
    return [(p, q) for p in range(budget + 1) for q in range(budget + 1 - p)]


def fit_sarima(
    values: pd.Series,
    p: int,
    q: int,
    seasonal_p: int,
    seasonal_q: int,
    exog: np.ndarray | None = None,
    trend: str = "c",
) -> SARIMAXResultsWrapper:
    """
    ARMA(p, q)x(P, 0, Q, 12), with a constant by default, fitted on the first differences (so
    d = 1).

    Parameters
    ----------
    values : pd.Series
        The first differences of one series (trillions USD per month), in time order.
    p : int
        Non-seasonal AR order.
    q : int
        Non-seasonal MA order.
    seasonal_p : int
        Seasonal AR order P, at multiples of SEASONAL_PERIOD (12 months).
    seasonal_q : int
        Seasonal MA order Q, at multiples of SEASONAL_PERIOD.
    exog : np.ndarray or None, default=None
        Extra regressors, shape (len(values), k), one row per month of `values` (the SARIMAX
        family of the forecast evaluation); None fits the plain SARIMA.
    trend : str, default="c"
        "c" for a constant (a drift in the levels), "n" for none; the tuning searches both.

    Returns
    -------
    SARIMAXResultsWrapper
        The fitted model by maximum likelihood. Its params are the constant ("intercept"),
        the regression coefficients when `exog` is given, the AR / MA coefficients and the
        shock variance ("sigma2"). Convergence is in
        ``mle_retvals["converged"]``.

    Notes
    -----
    Fitting an ARMA with a constant on the differences is ARIMA(p, 1, q) with drift on the
    levels: the constant sets the average monthly change (the drift).
    enforce_stationarity / enforce_invertibility keep the AR part stationary and the MA part
    invertible, so every fit is a well-defined model with a unique representation;
    disp=False silences the optimizer's output.
    """
    return SARIMAX(
        values,
        exog=exog,
        order=(p, 0, q),
        seasonal_order=(seasonal_p, 0, seasonal_q, SEASONAL_PERIOD),
        trend=trend,
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(disp=False, maxiter=MAX_ITERATIONS)


def information_criteria(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Fit the SARIMA candidate grid on both series and rank it by AIC, AICc and BIC.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The training first differences, holding both VALUE_COLUMNS (trillions USD).

    Returns
    -------
    tuple of (pd.DataFrame, str)
        The candidate table, one row per (series, model): the orders p, d, q, P, D, Q, s,
        nobs, parameter_count, converged, AIC, AICc and BIC (120 rows: 60 per series); and
        the summary report naming the best converged model per series and criterion.

    Raises
    ------
    ValueError
        If a series has 24 or fewer observations (too few for seasonal lags at 12 and 24),
        if no candidate converged for a series, or if a criterion has no finite value.

    Notes
    -----
    Processing:
    1. Build the grid: every (p, q) from nonseasonal_orders() x P in {0, 1} x Q in {0, 1}.
    2. For each series and candidate, fit with fit_sarima() and record the orders, the
       parameter count, convergence, AIC, AICc and BIC.
    3. For each series and criterion, keep the converged candidates with a finite value and
       report the one with the lowest value.

    The three criteria differ in how hard they penalize parameters: AIC aims at the best
    forecasts and tends to pick larger models, BIC (penalty log n per parameter) aims at
    the true order and picks smaller ones, AICc sits in between for short samples. Reporting
    all three shows whether the choice is robust. Non-converged fits stay in the table but
    are never chosen, since their criteria aren't at a likelihood maximum.
    """
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

        progress = Progress(f"SARIMA GRID | {column}", len(candidate_orders))
        for done, (p, q, seasonal_p, seasonal_q) in enumerate(candidate_orders, start=1):
            fitted_model = fit_sarima(values, p, q, seasonal_p, seasonal_q)
            progress.step(done, f"({p},{DIFFERENCING_ORDER},{q})x({seasonal_p},0,{seasonal_q})")

            # Includes the constant and sigma2, which AICc's k must count.
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
            "ARMA(p, q) with a constant on the differences, i.e. "
            f"ARIMA(p, {DIFFERENCING_ORDER}, q) "
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
            # AICc can be NaN (aicc() returns NaN when n - k - 1 <= 0), which idxmin can't rank.
            eligible = converged_results[np.isfinite(converged_results[criterion])]
            if eligible.empty:
                raise ValueError(f"No finite {criterion} values are available for {column}.")
            # idxmin returns the first row on a tie, i.e. the earlier candidate in grid order.
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
    """
    Run the stationarity tests, ACF / PACF and SARIMA grid on the training split and write their
    reports (make diagnostics).

    Parameters
    ----------
    levels_path : Path, default=INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv"
        The cleaned levels, written by ``make features``.
    differences_path : Path, default=PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv"
        The first differences, written by ``make features``.

    Returns
    -------
    None
        Writes the reports and the figure listed in the module docstring, and prints each
        report.

    Raises
    ------
    FileNotFoundError
        If either input file is missing (run ``make features`` first).

    Notes
    -----
    Processing:
    1. Read the levels, take the split boundaries from them, and keep the training part of
       both the levels and the differences.
    2. ADF / KPSS on the levels (constant and trend) and on the differences (constant).
    3. ACF / PACF of the differences; save the figure and the report.
    4. Fit the SARIMA grid on the differences; save the candidate table and the summary.

    Only train is used, so nothing in the order selection has seen a validation or test
    month. The boundaries come from the levels for both inputs, so the differences are cut
    at the same months (train has 167 levels and 166 differences).
    """
    start_run_log("diagnostics")
    begin_phase("Stationarity, ACF / PACF and the SARIMA grid")
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

    # Trending levels are tested against trend-stationarity; their differences against a
    # constant mean.
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

    finish_run()


if __name__ == "__main__":
    app()
