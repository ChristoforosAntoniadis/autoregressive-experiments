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
from typing import Any

from loguru import logger
from matplotlib.figure import Figure
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
from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
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
from autoregressive_experiments.runlog import begin_phase, finish_run, start_run_log

app = typer.Typer()

# Settings; each value's justification is in configs/hyperparameters.json.
# Lags (months) at which both tests are reported.
TEST_LAGS = HYPERPARAMETERS.residual_checks.test_lags
# A runner-up within this many AIC points of the best is checked too.
AIC_TIE_MARGIN = HYPERPARAMETERS.residual_checks.aic_tie_margin
# Lags shown in the figure's ACF and Ljung-Box panels.
PLOT_LAGS = HYPERPARAMETERS.residual_checks.plot_lags

REPORT_PATH = REPORTS_DIR / "residual_diagnostics.txt"
FIGURE_PATH = FIGURES_DIR / "residual_diagnostics.png"

# Reference palette: categorical slots 1-5 for the data (validated together; 3-5 need a legend),
# the reserved "critical" status color for failures, and the chrome colors. Every figure in the
# project takes its colors from here.
SERIES_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SERIES_COLOR = SERIES_PALETTE[0]
CRITICAL_COLOR = "#d03b3b"
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def model_label(model: dict[str, Any]) -> str:
    """
    Format a model's orders as a SARIMA label, e.g. "SARIMA(0,1,1)x(1,0,1,12)".

    Parameters
    ----------
    model : dict of str to Any
        Must hold the integer orders under "p", "d", "q", "P" and "Q". Other keys (the
        series name, the reason it was picked, fit results...) are ignored.

    Returns
    -------
    str
        "SARIMA(p,d,q)" followed by "x(P,0,Q,12)" when the model has a seasonal term. The
        seasonal differencing order is always 0 in this project, and the period is
        SEASONAL_PERIOD (12 months).
    """
    p, d, q, seasonal_p, seasonal_q = (model[k] for k in ("p", "d", "q", "P", "Q"))
    # A non-seasonal model gets no "x(0,0,0,12)" suffix, so labels stay short.
    seasonal = (
        f"x({seasonal_p},0,{seasonal_q},{SEASONAL_PERIOD})" if seasonal_p or seasonal_q else ""
    )
    return f"SARIMA({p},{d},{q}){seasonal}"


def recommended_orders(candidates: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Per series: the best converged model by AIC, AICc and BIC, plus the AIC runner-up if tied.

    Parameters
    ----------
    candidates : pd.DataFrame
        The information-criteria grid from ``make diagnostics``, one row per fitted
        candidate, with columns "series", "p", "d", "q", "P", "Q", "AIC", "AICc", "BIC"
        and "converged" (bool).

    Returns
    -------
    list of dict of str to Any
        One dict per distinct recommended order, in the order the series appear in
        `candidates`. Keys: "series" (column name), "p", "d", "q", "P", "Q" (int orders)
        and "reason" (str: the criteria that picked it, e.g. "AIC, AICc", or
        "AIC tie (+0.330)").

    Notes
    -----
    Processing:
    1. Keep only the converged candidates and group them by series.
    2. For each of AIC, AICc and BIC, find the minimizing row and record the criterion
       under that row's orders; when several criteria agree, the model appears once with
       all of them listed.
    3. Rank by AIC; if the runner-up is less than AIC_TIE_MARGIN (2) above the best, add it
       too, since by Burnham & Anderson it has substantial support.
    4. Turn each recorded order into one dict.

    A series with a single converged candidate would make ``iloc[1]`` raise IndexError;
    the 60-candidate grid makes that unlikely.
    """
    models: list[dict[str, Any]] = []
    # sort=False keeps the series in their order in the CSV (VALUE_COLUMNS order).
    for column, group in candidates[candidates["converged"]].groupby("series", sort=False):
        # Keyed by the order tuple, so criteria that agree are merged into one entry.
        picks: dict[tuple[int, ...], list[str]] = {}
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


def order_key(row: pd.Series) -> tuple[int, ...]:
    """
    The (p, d, q, P, Q) orders of a candidate, as a hashable tuple of ints.

    Parameters
    ----------
    row : pd.Series
        A candidate row (or a model dict wrapped in a Series) with "p", "d", "q", "P"
        and "Q" entries.

    Returns
    -------
    tuple of int
        Always five values, (p, d, q, P, Q). int() turns the CSV's numpy integers (or
        an object Series' values) into plain ints, so keys from different sources compare
        equal.
    """
    return tuple(int(row[k]) for k in ("p", "d", "q", "P", "Q"))


def residual_tests(residuals: pd.Series, arma_terms: int) -> pd.DataFrame:
    """
    Ljung-Box and Breusch-Godfrey p-values at TEST_LAGS; a NaN means too few degrees of freedom.

    Parameters
    ----------
    residuals : pd.Series
        One-step residuals of a fitted SARIMA model, in trillions, burn-in removed.
    arma_terms : int
        Number of fitted ARMA coefficients, p + q + P + Q (the constant not counted).

    Returns
    -------
    pd.DataFrame
        One row per lag in TEST_LAGS, with columns "lag" (int), "ljung_box_p" and
        "breusch_godfrey_p" (float). Both tests' null hypothesis: no autocorrelation up to
        that lag. "ljung_box_p" is NaN when lag - arma_terms is not positive.

    Notes
    -----
    Processing:
    1. Ljung-Box at every lag in TEST_LAGS, with ``model_df=arma_terms``.
    2. Fit an OLS regression of the residuals on a constant only.
    3. Breusch-Godfrey on that regression at every lag in TEST_LAGS.

    Why ``model_df``: residuals of an estimated ARMA model are fitted to look
    uncorrelated, so the Ljung-Box statistic is compared with a chi-squared of
    lag - (p + q + P + Q) degrees of freedom, not lag. Without the correction the test
    would pass too easily.

    Why the constant-only regression: statsmodels' Breusch-Godfrey only accepts
    regression results. With a constant as the only regressor, its auxiliary regression
    is the residuals on a constant and their own lags, and n * R^2 is the LM statistic.
    For models with MA terms this approximates the exact LM test, as the report says.
    """
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
    """
    Whether every available p-value is at or above SIGNIFICANCE_LEVEL (5%).

    Parameters
    ----------
    tests : pd.DataFrame
        The output of residual_tests(): columns "ljung_box_p" and "breusch_godfrey_p".

    Returns
    -------
    bool
        True when no test rejects "no autocorrelation". NaN p-values (Ljung-Box lags with
        no degrees of freedom left) are skipped rather than counted as failures.
    """
    p_values = tests[["ljung_box_p", "breusch_godfrey_p"]].to_numpy(dtype=float)
    # bool(): .all() returns numpy.bool_, and callers store the verdict in plain dicts.
    return bool((p_values[np.isfinite(p_values)] >= SIGNIFICANCE_LEVEL).all())


def check_model(values: pd.Series, model: dict[str, Any]) -> dict[str, Any]:
    """
    Fit one SARIMA model and run the residual tests on it.

    Parameters
    ----------
    values : pd.Series
        One series' training first differences, in trillions per month.
    model : dict of str to Any
        The model to check: "series", "p", "d", "q", "P", "Q" and "reason", as made by
        recommended_orders().

    Returns
    -------
    dict of str to Any
        Every key of `model`, plus:

        - "residuals": pd.Series, the one-step residuals after the burn-in, in trillions
        - "arma_terms": int, p + q + P + Q
        - "tests": pd.DataFrame, the output of residual_tests()
        - "passed": bool, the output of passes()

    Notes
    -----
    Processing:
    1. Fit the model with diagnostics.fit_sarima(), exactly as in the information-criteria
       grid, so the checked model is the one that was ranked.
    2. Drop the first ``loglikelihood_burn`` residuals: statsmodels excludes them from the
       likelihood because they come from the filter's initialization, not the model.
    3. Run residual_tests() and passes() on what is left.
    """
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
    values: pd.Series, column: str, candidates: pd.DataFrame, models: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Check the recommended models; if none passes, walk down the AIC ranking to the first that does.

    Parameters
    ----------
    values : pd.Series
        The series' training first differences, in trillions per month.
    column : str
        The series name, e.g. "fed_total_assets_T".
    candidates : pd.DataFrame
        The full information-criteria grid (all series), as in recommended_orders().
    models : list of dict of str to Any
        This series' recommended models, from recommended_orders().

    Returns
    -------
    list of dict of str to Any
        One check_model() result per recommended model, in the same order. If none of them
        passed, the first passing alternative from the AIC ranking is appended (its
        "reason" says "best passing alternative (AIC rank N)"). If no candidate passes at
        all, only the failing recommended models are returned.

    Notes
    -----
    Processing:
    1. Check every recommended model; stop here if at least one passes.
    2. Otherwise rank this series' converged candidates by AIC and skip the orders already
       tried.
    3. Check each remaining candidate in rank order and return as soon as one passes.

    The fallback keeps the selection rule "the best by information criteria among the
    models whose residuals are clean", rather than inventing a new order.
    """
    checked = [check_model(values, model) for model in models]
    if any(result["passed"] for result in checked):
        return checked

    # Wrapped in a Series so order_key() reads the dicts the same way as the CSV rows.
    tried = {order_key(pd.Series(model)) for model in models}
    ranked = candidates[(candidates["series"] == column) & candidates["converged"]]
    # The rank counts every converged candidate, including the ones already tried.
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


def residual_report(results: list[dict[str, Any]], train: pd.DataFrame) -> str:
    """
    Write the text report of the residual checks.

    Parameters
    ----------
    results : list of dict of str to Any
        check_model() results for every series, as returned by checked_models().
    train : pd.DataFrame
        The training split of the first differences, with DATE_COLUMN; used for the date
        range and the observation count in the header.

    Returns
    -------
    str
        The report: a header explaining the method, then per series and model the
        p-values at each lag in TEST_LAGS and a PASS / FAIL verdict, then per series the
        models to carry forward (or a note that none passed), then the figure's path.
    """
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
    """
    Format a p-value with four decimals, or "n/a" when it is NaN or infinite.

    Parameters
    ----------
    p_value : float
        A test p-value; NaN where Ljung-Box has no degrees of freedom left.

    Returns
    -------
    str
        E.g. "0.0612", or "n/a".
    """
    return "n/a" if not np.isfinite(p_value) else f"{p_value:.4f}"


def apply_chart_style() -> None:
    """
    Recessive hairline grid and axes in the reference palette's chrome colors.

    Returns
    -------
    None
        Updates matplotlib's global rcParams, so every figure created afterwards in this
        process uses the style. The other analysis modules call it before drawing, so all
        the project's figures look the same.
    """
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
            # No box around the plot: only the left and bottom axes are drawn.
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def model_name(result: dict[str, Any]) -> str:
    """
    A readable series name for chart titles, e.g. "fed total assets".

    Parameters
    ----------
    result : dict of str to Any
        Any model dict with a "series" key holding the column name.

    Returns
    -------
    str
        The column name without its "_T" (trillions) suffix, underscores turned into
        spaces.
    """
    return result["series"].removesuffix("_T").replace("_", " ")


def residual_figure(results: list[dict[str, Any]], dates: pd.Series) -> Figure:
    """
    One row per model: residuals over time, residual ACF, and Ljung-Box p-value by lag.

    Parameters
    ----------
    results : list of dict of str to Any
        check_model() results, one figure row each.
    dates : pd.Series
        The training split's dates, positionally aligned with the first differences (the
        split resets the index to 0..n-1, so the residuals' index positions pick them).

    Returns
    -------
    Figure
        A figure with len(results) rows and 3 columns, in the reference palette. The
        caller saves and closes it.

    Notes
    -----
    Processing, per model row:
    1. Left: the residuals over time (trillions) with a zero line; the title holds the
       model, its PASS / FAIL verdict and the reason it was checked.
    2. Middle: the residual ACF at lags 1..PLOT_LAGS with the approximate 95% band
       +/- 1.96 / sqrt(n); spikes outside the band are marked in the critical color.
    3. Right: the Ljung-Box p-value at every lag 1..PLOT_LAGS (degrees of freedom
       corrected as in residual_tests()), on a log scale against the 5% line; failing
       lags in the critical color.
    """
    apply_chart_style()
    # squeeze=False keeps axes 2-D even when there is only one model.
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

        # [1:] drops lag 0, whose autocorrelation is always 1.
        autocorrelations = acf(residuals, nlags=PLOT_LAGS, fft=False)[1:]
        # Approximate 95% band for white noise: +/- 1.96 / sqrt(n).
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

        # lags=PLOT_LAGS (an int) gives a p-value at every lag 1..PLOT_LAGS.
        p_values = acorr_ljungbox(residuals, lags=PLOT_LAGS, model_df=result["arma_terms"])[
            "lb_pvalue"
        ].to_numpy()
        # NaN p-values compare False, so they land in the passing (blue) group, but a log axis
        # can't draw them, so they simply don't appear.
        failing = p_values < SIGNIFICANCE_LEVEL
        p_axis.scatter(lags[~failing], p_values[~failing], color=SERIES_COLOR, s=24, zorder=3)
        p_axis.scatter(lags[failing], p_values[failing], color=CRITICAL_COLOR, s=24, zorder=3)
        p_axis.axhline(SIGNIFICANCE_LEVEL, color=CRITICAL_COLOR, linewidth=1)
        # Slightly above the line (x1.3 on the log axis), so the label doesn't sit on it.
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
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Fit and check the recommended models on the training split; also returns that split.

    Shared entry point of the U4, U5, U5b and U5c modules, so they all examine the same
    models fitted on the same data.

    Parameters
    ----------
    levels_path : Path
        The cleaned levels CSV (``make features``); only used to compute the split dates.
    differences_path : Path
        The first-differences CSV (``make features``).
    candidates_path : Path
        The information-criteria grid CSV (``make diagnostics``).

    Returns
    -------
    tuple of (list of dict of str to Any, pd.DataFrame)
        The check_model() results of every series, in VALUE_COLUMNS order (see
        check_series() for which models are included), and the training split of the
        first differences (with DATE_COLUMN and both series, index 0..n-1).

    Raises
    ------
    FileNotFoundError
        If any of the three input files is missing; the message names the make target
        that produces it.

    Notes
    -----
    Processing:
    1. Check that the inputs exist.
    2. Compute the 60 / 20 / 20 split boundaries from the levels and cut the differences
       at the same dates, keeping train only.
    3. Pick the recommended orders from the grid with recommended_orders().
    4. Check each series' models with check_series().
    """
    # Each input mapped to the make target that generates it, for the error message.
    required = {
        levels_path: "features",
        differences_path: "features",
        candidates_path: "diagnostics",
    }
    for path, command in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Generate it with make {command}.")
    levels = pd.read_csv(levels_path, parse_dates=[DATE_COLUMN])
    # The boundaries come from the levels, so levels and differences are cut at the same months.
    train = split_by_time(
        pd.read_csv(differences_path, parse_dates=[DATE_COLUMN]), split_boundaries(levels)
    )["train"]
    candidates = pd.read_csv(candidates_path)
    models = recommended_orders(candidates)

    results: list[dict[str, Any]] = []
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
    """
    Run the residual checks and write the report and the figure (``make residuals``).

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
        Writes REPORT_PATH and FIGURE_PATH, and prints the report.
    """
    start_run_log("residuals")
    begin_phase("Residual checks of the recommended SARIMAs (U4)")
    results, train = checked_models(levels_path, differences_path, candidates_path)

    report = residual_report(results, train)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = residual_figure(results, train[DATE_COLUMN])
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    # Free the figure's memory; it is already saved.
    plt.close(figure)
    logger.success(f"Residual checks written to {REPORT_PATH} and {FIGURE_PATH}")
    finish_run()


# Running this file directly runs the typer CLI (main).
if __name__ == "__main__":
    app()
