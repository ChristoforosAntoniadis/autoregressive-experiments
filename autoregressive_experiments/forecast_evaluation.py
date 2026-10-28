"""
Forecast evaluation (step U6, then the VAR / VECM systems): how accurate are the models out of
sample, do they beat naive forecasts, does modelling the two series jointly beat modelling them
alone, and are the forecast intervals honest?

The systems, VAR(5) on the differences and VECM(5) on the levels (carried forward from step
V6), are refitted on both series up to each origin and forecast both; their intervals are normal,
from the system's forecast-error covariance (GARCH is univariate here). Each system is also
tested against the series' selected SARIMA with Diebold-Mariano.

Rolling origin with an expanding window, on validation and then on test: at every month from the
end of the previous split onwards, each model is refitted on all data up to that month and
forecasts up to 12 months ahead; only forecasts whose target falls inside the split are scored.
Each monthly origin is one fold (56 per split). Model orders come from the training split and the
selected model per series and family (SARIMA, system, smoothing, ML, SARIMAX) from validation
(lowest one-month MASE); test is run once, with all models, and nothing changes after it. Errors
are measured on the levels (trillions), so MAPE stays well defined: the levels are always
positive, while the differences cross zero.

Models: the SARIMA models that pass the residual checks (fitted on the first differences, their
forecasts added back onto the last level), and three baselines: naive (no change), drift (the
average monthly change so far) and seasonal naive (each month changes as it did a year earlier).
Smoothing baselines: simple moving averages of the last 3, 6 and 12 levels, and single, double
(Holt, plain and damped) and triple (Holt-Winters) exponential smoothing on the levels. A
combination: the equal-weight average of the selected SARIMA and naive. SARIMAX: the selected
SARIMA plus the other series' last 1, 3 or 5 monthly changes as regressors; for forecast months
whose regressors aren't known yet, the other series' own selected-SARIMA forecasts stand in.
Lag-feature ML (ml_forecasting): Ridge, Lasso, random forest and gradient boosting on the
last 1, 3 or 5 monthly changes of both series, forecasting recursively. Smoothing, the
combination, SARIMAX and ML give point forecasts only.

Point accuracy per split, series, model and horizon: MAE, MSE, RMSE (trillions, trillions
squared), MAPE (%) and MASE (MAE divided by the training split's in-sample MAE of the one-month
naive forecast; below 1 beats it), plus a Diebold-Mariano test against naive (absolute and squared
error losses), so "beats naive" means more than a lucky sample. Every non-SARIMA model is also
tested against the series' selected SARIMA the same way.

Forecast quantiles (2.5, 10, 50, 90 and 97.5%) come from each model's own forecast distribution
on the levels: the SARIMA's error weights, summed over the horizon, with either a constant shock
variance or the forecast variance path of a volatility model, refitted at every origin: the
GARCH variant chosen in step U5b, or the EGARCH and GJR-GARCH variants of step U5c; normal
quantiles for constant variance, unit-variance Student-t for the volatility models. Naive gets
random-walk quantiles as the reference. Scored by 80% and 95% interval coverage and width, the
share of actuals below each quantile, and pinball (quantile) loss. The volatility models only
change these intervals, never the point forecasts. Refits that fail to converge are counted and
listed.

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` and, through ``residual_checks``, the
first differences and ``reports/information_criteria_candidates.csv``.

Output, in ``reports/``:

- ``forecast_evaluation.txt``: per split, point-accuracy tables with Diebold-Mariano results,
  interval tables, and the refits that failed to converge
- ``forecast_metrics.csv`` and ``forecast_intervals.csv``: every number behind those tables
- ``figures/forecast_evaluation.png``: per series, each family's selected model's MAE relative to
  naive by horizon on validation and on test, and the actual path against the selected SARIMA's,
  the best other family's and the naive one-month forecasts
- ``figures/ml_evaluation.png``: per series and split, each ML model's best lag count (chosen
  on validation) against naive, the selected SARIMA and the selected smoothing model
- ``figures/forecast_intervals.png``: per series and split, the selected SARIMA's 80% and 95%
  interval coverage by horizon, with constant variance, GARCH, EGARCH and GJR-GARCH, against the
  targets
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.vecm import VECM
import typer

from autoregressive_experiments.asymmetric_garch import EGARCH_T, GJR_T, asymmetric_check
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
)
from autoregressive_experiments.features import (
    DATE_COLUMN,
    lagged_exog,
    split_boundaries,
    split_by_time,
)
from autoregressive_experiments.ml_forecasting import ML_MODELS, ML_NAMES, ml_forecaster
from autoregressive_experiments.residual_checks import (
    BASELINE,
    MUTED_INK,
    PRIMARY_INK,
    SECONDARY_INK,
    SERIES_PALETTE,
    SURFACE,
    apply_chart_style,
    checked_models,
    model_label,
)
from autoregressive_experiments.runlog import (
    Progress,
    begin_phase,
    finish_run,
    start_run_log,
)
from autoregressive_experiments.var_analysis import CARRIED_LAGS
from autoregressive_experiments.var_models import VAR_LAGS, VECM_RANK
from autoregressive_experiments.volatility_models import garch_variance_path

app = typer.Typer()

# Settings; each value's justification is in configs/hyperparameters.json.
# Forecast horizons that are scored, in months ahead; every model forecasts up to the longest.
HORIZONS = HYPERPARAMETERS.forecast_evaluation.horizons
MAX_HORIZON = max(HORIZONS)
METRICS = ("MAE", "MSE", "RMSE", "MAPE", "MASE")
# Diebold-Mariano statistic and p-value per loss, against naive; the DM_SARIMA_* twins hold the
# same test against the series' selected SARIMA.
DM_COLUMNS = ("DM_absolute", "DM_absolute_p", "DM_squared", "DM_squared_p")
DM_SARIMA_COLUMNS = tuple(column.replace("DM", "DM_SARIMA") for column in DM_COLUMNS)
NAIVE = "naive"
BASELINES = (NAIVE, "drift", "seasonal naive")
# The systems carried forward from step V6; each forecasts both series at once.
SYSTEMS = (f"VAR({CARRIED_LAGS})", f"VECM({CARRIED_LAGS})")
# The standard normal's 97.5% quantile (about 1.96): a normal 95% interval is mean +- this x SD.
NORMAL_95 = stats.norm.ppf(0.975)
# Smoothing baselines: simple moving averages of the last k levels, and exponential smoothing
# via statsmodels' holtwinters.ExponentialSmoothing.
MOVING_AVERAGE_WINDOWS = HYPERPARAMETERS.forecast_evaluation.moving_average_windows
# Model name -> keyword arguments for ExponentialSmoothing; no trend or season means single. A
# seasonal model gets the project's season length, so it isn't configured twice.
EXPONENTIAL_SMOOTHING: dict[str, dict[str, Any]] = {
    name: dict(options) | ({"seasonal_periods": SEASONAL_PERIOD} if "seasonal" in options else {})
    for name, options in HYPERPARAMETERS.forecast_evaluation.exponential_smoothing.items()
}
SMOOTHING = (*(f"SMA({window})" for window in MOVING_AVERAGE_WINDOWS), *EXPONENTIAL_SMOOTHING)
# A weighted average of the selected SARIMA and naive forecasts; COMBINATION_WEIGHT is the
# SARIMA's share.
COMBINATION = "SARIMA + naive average"
COMBINATION_WEIGHT = HYPERPARAMETERS.forecast_evaluation.combination_weight
# SARIMAX: the series' selected SARIMA plus the other series' lagged monthly changes as
# regressors. The candidate lag counts are the VAR candidates' orders, so no new number is
# introduced; validation picks one per series.
SARIMAX_LAGS = VAR_LAGS
SARIMAX_NAMES = {
    lags: f"SARIMAX ({lags} exog lag{'' if lags == 1 else 's'})" for lags in SARIMAX_LAGS
}
SPLITS = ("validation", "test")
# Ordered low to high: interval_scores unpacks them as 95% low, 80% low, median, 80% high,
# 95% high.
QUANTILES = HYPERPARAMETERS.forecast_evaluation.quantiles
# Interval variants: constant variance, U5b's GARCH, and U5c's EGARCH and GJR-GARCH.
VARIANCES = ("constant", "garch", "egarch", "gjr")
VARIANCE_NAMES = {
    "constant": "constant variance",
    "garch": "with GARCH",
    "egarch": "with EGARCH",
    "gjr": "with GJR-GARCH",
}

REPORT_PATH = REPORTS_DIR / "forecast_evaluation.txt"
METRICS_PATH = REPORTS_DIR / "forecast_metrics.csv"
INTERVALS_PATH = REPORTS_DIR / "forecast_intervals.csv"
FIGURE_PATH = FIGURES_DIR / "forecast_evaluation.png"
INTERVALS_FIGURE_PATH = FIGURES_DIR / "forecast_intervals.png"
ML_FIGURE_PATH = FIGURES_DIR / "ml_evaluation.png"

# Reference palette: baselines are ink and greys; slots 3-4's below-3:1 contrast is backed by
# the legends and the report's tables.
# Baseline name -> (line color, line width).
BASELINE_STYLES: dict[str, tuple[str, float]] = {
    NAIVE: (PRIMARY_INK, 2),
    "drift": (MUTED_INK, 1.2),
    "seasonal naive": (BASELINE, 1.2),
}
# Slots 1-4 (validated together) for the interval variants in the interval figure; zip stops
# at the 4 variants.
VARIANCE_COLORS = dict(zip(VARIANCES, SERIES_PALETTE))
# The accuracy figure shows each family's selected model in slots 1-5 (validated together).
FAMILY_COLORS = dict(
    zip(("SARIMA", "system", "smoothing", "combination", "SARIMAX"), SERIES_PALETTE)
)

# A forecaster takes the levels up to an origin and a step count, and returns "forecast" (the
# level path) plus, optionally, "sd_constant" / "sd_garch" paths, "nu_garch" and "converged".
# The levels are a 1-D array in trillions, oldest first, ending at the origin month; every path
# in the returned dict has one value per step (horizons 1..steps). The bare dict keeps the
# alias as it was; each forecaster states its own, narrower return type.
Forecaster = Callable[[np.ndarray, int], dict]


def naive_forecast(levels: np.ndarray, steps: int) -> dict[str, np.ndarray]:
    """
    No change; random-walk spread: the monthly changes' standard deviation times sqrt(h).

    Parameters
    ----------
    levels : np.ndarray
        Levels up to and including the origin, shape (months,), trillions, oldest first.
    steps : int
        Number of months to forecast.

    Returns
    -------
    dict of str to np.ndarray
        "forecast": the last level repeated `steps` times, shape (steps,), trillions.
        "sd_constant": the forecast-error standard deviation at h = 1..steps, shape (steps,),
        trillions.

    Notes
    -----
    Processing:
    1. Take the sample standard deviation (ddof=1) of the monthly changes so far.
    2. Multiply it by sqrt(h) for each horizon h.
    3. Repeat the last level as the point forecast.

    Why sqrt(h): under a random walk the level error at h is the sum of h independent monthly
    changes, so its variance is h times the one-month variance. This makes naive's intervals
    the reference that every model's intervals are compared with.
    """
    spread = np.std(np.diff(levels), ddof=1) * np.sqrt(np.arange(1, steps + 1))
    return {"forecast": np.full(steps, levels[-1]), "sd_constant": spread}


def drift_forecast(levels: np.ndarray, steps: int) -> dict[str, np.ndarray]:
    """
    Extend the last level by the average monthly change over the whole history.

    Parameters
    ----------
    levels : np.ndarray
        Levels up to and including the origin, shape (months,), trillions, oldest first.
    steps : int
        Number of months to forecast.

    Returns
    -------
    dict of str to np.ndarray
        "forecast": the level path at h = 1..steps, shape (steps,), trillions. Point forecast
        only, so this baseline gets no intervals.

    Notes
    -----
    The average monthly change is (last - first) / (months - 1), the mean of the first
    differences, so it uses the whole expanding window, not a recent slope.
    """
    slope = (levels[-1] - levels[0]) / (len(levels) - 1)
    return {"forecast": levels[-1] + slope * np.arange(1, steps + 1)}


def seasonal_naive_forecast(
    levels: np.ndarray, steps: int, period: int = SEASONAL_PERIOD
) -> dict[str, np.ndarray]:
    """
    Each future month changes by as much as the same month one period earlier.

    Parameters
    ----------
    levels : np.ndarray
        Levels up to and including the origin, shape (months,), trillions, oldest first. Needs
        at least `period` + 1 months.
    steps : int
        Number of months to forecast; at most `period`.
    period : int, default=SEASONAL_PERIOD
        Season length in months (12).

    Returns
    -------
    dict of str to np.ndarray
        "forecast": the level path at h = 1..steps, shape (steps,), trillions. Point forecast
        only.

    Raises
    ------
    ValueError
        If `steps` is larger than `period`.

    Notes
    -----
    Processing:
    1. Take the last `period` monthly changes; the first of them is the change into the month
       one period before the first forecast month.
    2. Add their running total to the last level.

    Applied to the changes rather than the levels, because the levels trend: repeating last
    year's level would forecast a year of reversal. Beyond one period the changes would have to
    be reused, which this simple version doesn't do, hence the error.
    """
    if steps > period:
        raise ValueError(f"Seasonal naive forecasts at most {period} steps here, not {steps}.")
    last_changes = np.diff(levels)[-period:]
    return {"forecast": levels[-1] + np.cumsum(last_changes[:steps])}


def moving_average_forecaster(window: int) -> Forecaster:
    """
    Every horizon gets the average of the last `window` levels.

    Parameters
    ----------
    window : int
        Number of most recent levels to average (the k of SMA(k)), months.

    Returns
    -------
    Forecaster
        A forecaster for the harness: given the levels up to an origin and a step count, it
        returns {"forecast": the flat average path, shape (steps,), trillions}. Point forecast
        only.
    """

    def forecast(levels: np.ndarray, steps: int) -> dict[str, np.ndarray]:
        """
        Forecast the mean of the last `window` levels at every horizon.

        Parameters
        ----------
        levels : np.ndarray
            Levels up to and including the origin, shape (months,), trillions, oldest first.
        steps : int
            Number of months to forecast.

        Returns
        -------
        dict of str to np.ndarray
            "forecast": the average repeated `steps` times, shape (steps,), trillions.

        Notes
        -----
        On a trending series the average of past levels lags behind the last level, so this
        baseline is expected to do worse than naive; it is here because the user asked for the
        smoothing family.
        """
        return {"forecast": np.full(steps, np.mean(levels[-window:]))}

    return forecast


def exponential_smoothing_forecaster(settings: dict[str, Any]) -> Forecaster:
    """
    statsmodels' ExponentialSmoothing on the levels, refitted (parameters re-estimated) at
    every origin: single (no settings), Holt (additive trend, optionally damped), or
    Holt-Winters (additive trend and season).

    Parameters
    ----------
    settings : dict of str to Any
        Keyword arguments for ExponentialSmoothing, one value of EXPONENTIAL_SMOOTHING, such as
        {"trend": "add", "damped_trend": True}. Empty for single exponential smoothing.

    Returns
    -------
    Forecaster
        A forecaster for the harness: given the levels up to an origin and a step count, it
        returns {"forecast": the level path, shape (steps,), trillions}. Point forecast only.
    """

    def forecast(levels: np.ndarray, steps: int) -> dict[str, np.ndarray]:
        """
        Fit ExponentialSmoothing on the levels so far and forecast `steps` months.

        Parameters
        ----------
        levels : np.ndarray
            Levels up to and including the origin, shape (months,), trillions, oldest first.
        steps : int
            Number of months to forecast.

        Returns
        -------
        dict of str to np.ndarray
            "forecast": the level path at h = 1..steps, shape (steps,), trillions.

        Notes
        -----
        Fitted on the levels, not the differences: the trend component plays the role that
        differencing plays for SARIMA. initialization_method="estimated" estimates the initial
        level (and trend, season) together with the smoothing parameters, so each refit uses
        only the data up to the origin.
        """
        fitted = ExponentialSmoothing(levels, initialization_method="estimated", **settings).fit()
        return {"forecast": np.asarray(fitted.forecast(steps))}

    return forecast


def combine_forecasts(forecasts: pd.DataFrame, first: str, second: str, name: str) -> pd.DataFrame:
    """
    Rows for `name`: the weighted average of two models' forecasts (COMBINATION_WEIGHT on
    the first), same origin and horizon. Point forecasts only; no intervals.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Rolling forecasts of one series and split, one row per model, origin and horizon, as
        made by rolling_forecasts; needs the columns "model", "origin", "horizon", "target",
        "actual" and "forecast" (trillions).
    first : str
        Name of the first model to average (the selected SARIMA).
    second : str
        Name of the second model to average (naive).
    name : str
        Model name for the new rows (COMBINATION).

    Returns
    -------
    pd.DataFrame
        One row per (origin, horizon) that both models forecast, with the columns "origin",
        "horizon", "target", "actual", "model" (= `name`) and "forecast" (the average,
        trillions).

    Notes
    -----
    Processing:
    1. Keep each model's rows and the key columns plus "forecast".
    2. Inner-join them on origin, horizon, target and actual, so the two forecasts in a row
       are for the same month made at the same origin.
    3. Average the two forecasts, COMBINATION_WEIGHT on the first and the rest on the
       second.

    Built from forecasts already made, so the combination adds no refits and can't see more
    data than its two parts did.
    """
    keys = ["origin", "horizon", "target", "actual"]
    pair = forecasts[forecasts["model"] == first][[*keys, "forecast"]].merge(
        forecasts[forecasts["model"] == second][[*keys, "forecast"]],
        on=keys,
        suffixes=("_first", "_second"),
    )
    average = (
        COMBINATION_WEIGHT * pair["forecast_first"]
        + (1 - COMBINATION_WEIGHT) * pair["forecast_second"]
    )
    return pair[keys].assign(model=name, forecast=average)


def level_variances(psi: np.ndarray, shock_variances: np.ndarray) -> np.ndarray:
    """
    Forecast-error variance of the level at each horizon h = 1, ..., len(shock_variances).

    The level is the running total of the differences, so its error at h is
    sum over j = 1..h of Psi_(h-j) * shock_(t+j), where Psi_k = psi_0 + ... + psi_k are the
    summed ARMA error weights (psi_0 = 1), and its variance weights each shock's variance.

    Parameters
    ----------
    psi : np.ndarray
        ARMA error weights (impulse responses) of the differences, psi_0 = 1 first, shape
        (steps,) with at least len(shock_variances) values.
    shock_variances : np.ndarray
        Variance of the shock in each future month t+1..t+steps, shape (steps,), trillions
        squared: constant for a constant-variance model, a GARCH forecast path otherwise.

    Returns
    -------
    np.ndarray
        Level forecast-error variance at h = 1..steps, shape (steps,), trillions squared.

    Notes
    -----
    Processing:
    1. Sum the error weights cumulatively: Psi_k is how much a shock moves the level k months
       later, since the level adds up every difference it touched.
    2. For each h, weight shock j (j = 1..h) by Psi_(h-j) squared and its own variance, and
       sum; the shocks are uncorrelated, so no cross terms.

    Why levels: the errors are scored on the levels, and a shock to the differences keeps
    moving the level, so the level variance grows faster than the differences' variance.
    Indexing: in the code j - 1 runs over np.arange(h), so the weight index is h - 1 - (j - 1)
    = h - j and shock_variances[j - 1] is the shock at t+j.
    """
    summed = np.cumsum(psi)
    return np.array(
        [
            np.sum(summed[h - 1 - np.arange(h)] ** 2 * shock_variances[:h])
            for h in range(1, len(shock_variances) + 1)
        ]
    )


def sarima_forecaster(model: dict[str, Any], variants: dict[str, str]) -> Forecaster:
    """
    Refit the model on the history's first differences; add its forecasts to the last level;
    give the level's spread with a constant shock variance and with each volatility model's
    variance path (variants maps "garch" / "egarch" / "gjr" to the chosen variant names).

    Parameters
    ----------
    model : dict of str to Any
        One checked SARIMA model from residual_checks.checked_models; uses its orders "p", "q",
        "P" and "Q" (d = 1 is fixed by fitting on the first differences, D = 0).
    variants : dict of str to str
        Volatility key ("garch", "egarch", "gjr") -> the variance-model name to refit at every
        origin, as chosen on the training residuals in steps U5b and U5c.

    Returns
    -------
    Forecaster
        A forecaster for the harness; see the nested forecast() for what it returns.
    """

    def forecast(levels: np.ndarray, steps: int) -> dict[str, np.ndarray | float | bool]:
        """
        Fit the SARIMA on the differences so far and forecast the levels with their spreads.

        Parameters
        ----------
        levels : np.ndarray
            Levels up to and including the origin, shape (months,), trillions, oldest first.
        steps : int
            Number of months to forecast.

        Returns
        -------
        dict of str to np.ndarray, float or bool
            "forecast": level path at h = 1..steps, shape (steps,), trillions.
            "sd_constant": level forecast-error SD with a constant shock variance, shape
            (steps,), trillions.
            "converged": whether the SARIMA's likelihood optimizer reported convergence.
            "sd_<key>" and "nu_<key>" for each key of `variants`: the level SD path with that
            volatility model's variance forecasts (shape (steps,), trillions), and its
            Student-t degrees of freedom (inf for normal shocks).

        Notes
        -----
        Processing:
        1. Fit the SARIMA on the first differences of the levels so far (d = 1).
        2. Get its error weights psi_0..psi_(steps-1) and the residuals after the burn-in.
        3. Point forecast: the last level plus the running total of the forecast differences.
        4. Constant spread: level_variances with the estimated sigma2 at every horizon.
        5. For each volatility model: refit it on the residuals, forecast the shock variances,
           and turn them into a level spread the same way.

        The volatility models change only the spread, never the point forecast (two-step
        approach, as in U5b).
        """
        fitted = fit_sarima(
            pd.Series(np.diff(levels)), model["p"], model["q"], model["P"], model["Q"]
        )
        # impulse_responses returns psi_0..psi_steps (steps + 1 values, possibly 2-D); keep the
        # first `steps`, the weights needed for h = 1..steps.
        psi = np.asarray(fitted.impulse_responses(steps=steps)).ravel()[:steps]
        # The first residuals come from the diffuse start-up of the state-space filter and
        # aren't real one-step errors; the likelihood skips them, and so does the GARCH refit.
        residuals = fitted.resid.iloc[fitted.loglikelihood_burn :]
        constant = np.full(steps, float(fitted.params["sigma2"]))
        output: dict[str, np.ndarray | float | bool] = {
            "forecast": levels[-1] + np.cumsum(np.asarray(fitted.forecast(steps))),
            "sd_constant": np.sqrt(level_variances(psi, constant)),
            "converged": bool(fitted.mle_retvals.get("converged", False)),
        }
        # One spread path per volatility model (U5b's GARCH, U5c's EGARCH and GJR).
        for key, variant in variants.items():
            variances, nu = garch_variance_path(residuals, variant, steps)
            output[f"sd_{key}"] = np.sqrt(level_variances(psi, variances))
            output[f"nu_{key}"] = nu
        return output

    return forecast


def system_level_variances(psi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Level forecast-error variance of each series at h = 1..len(psi), for a VAR on the
    differences with error-weight matrices psi (psi_0 = I) and shock covariance sigma.

    The matrix version of level_variances: the level error at h is the sum over s = 0..h-1 of
    C_s u_(t+h-s), with C_s = psi_0 + ... + psi_s, so its covariance sums C_s sigma C_s'.
    Returns the diagonals, shape (h, series).

    Parameters
    ----------
    psi : np.ndarray
        The VAR's moving-average (error-weight) matrices psi_0..psi_(steps-1) of the
        differences, shape (steps, series, series).
    sigma : np.ndarray
        Covariance matrix of the VAR's one-step shocks, shape (series, series), trillions
        squared.

    Returns
    -------
    np.ndarray
        Level forecast-error variance of each series at h = 1..steps, shape (steps, series),
        trillions squared.

    Notes
    -----
    Processing:
    1. Sum the error-weight matrices cumulatively to get C_s.
    2. Compute C_s sigma C_s' for every s at once with einsum ("sij,jk,slk->sil": for each s,
       row i of C_s times sigma times column l of C_s').
    3. Sum those terms cumulatively over s, so entry h - 1 is the covariance at horizon h.
    4. Keep each covariance matrix's diagonal: the per-series variances.

    The cumulative sum in step 3 works because the shock at t+h-s gets weight C_s for every h,
    unlike the univariate version's explicit loop; the off-diagonal terms (the series' error
    covariance) are dropped, since each series is scored on its own.
    """
    summed = np.cumsum(psi, axis=0)
    covariances = np.cumsum(np.einsum("sij,jk,slk->sil", summed, sigma, summed), axis=0)
    return np.diagonal(covariances, axis1=1, axis2=2)


def system_forecast(kind: str, history: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit VAR(5) on the differences or VECM(5) on the levels of both series; return the level
    forecasts and their standard deviations, each of shape (steps, series).

    Parameters
    ----------
    kind : str
        SYSTEMS[0] for the VAR; anything else (SYSTEMS[1]) fits the VECM.
    history : np.ndarray
        Levels of both series up to and including the origin, shape (months, series), trillions,
        oldest first, columns in VALUE_COLUMNS order.
    steps : int
        Number of months to forecast.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        The level forecasts and their normal standard deviations, each shape (steps, series),
        trillions.

    Notes
    -----
    Processing, VAR:
    1. Difference the levels, fit a VAR with CARRIED_LAGS lags and a constant by OLS.
    2. Forecast the differences from the last CARRIED_LAGS of them and add their running total
       to the last levels.
    3. Spread: system_level_variances from the error-weight matrices psi_0..psi_(steps-1)
       (ma_rep returns steps + 1 of them) and the shock covariance.

    Processing, VECM:
    1. Fit a VECM on the levels with CARRIED_LAGS lagged differences, cointegration rank
       VECM_RANK and a constant outside the long-run relation ("co").
    2. Take statsmodels' level forecasts and 95% intervals and convert the interval's upper
       half-width back into a standard deviation.
    """
    if kind == SYSTEMS[0]:
        differences = np.diff(history, axis=0)
        fitted = VAR(differences).fit(CARRIED_LAGS, trend="c")
        path = fitted.forecast(differences[-CARRIED_LAGS:], steps)
        spread = np.sqrt(system_level_variances(fitted.ma_rep(steps)[:steps], fitted.sigma_u))
        return history[-1] + np.cumsum(path, axis=0), spread
    fitted = VECM(history, k_ar_diff=CARRIED_LAGS, coint_rank=VECM_RANK, deterministic="co").fit()
    mean, _, upper = fitted.predict(steps, alpha=0.05)
    # statsmodels gives normal 95% level intervals; recover the standard deviation from them.
    return mean, (upper - mean) / NORMAL_95


def system_forecaster(
    kind: str,
    all_levels: np.ndarray,
    target: int,
    cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]],
) -> Forecaster:
    """
    A forecaster for one series of a system. The harness passes that series' levels up to
    the origin; both series are cut at the same month. Each origin's fit is shared between the
    two series through the cache.

    Parameters
    ----------
    kind : str
        Which system, one of SYSTEMS.
    all_levels : np.ndarray
        Levels of both series over the whole sample, shape (months, series), trillions, columns
        in VALUE_COLUMNS order. Only the months the harness hands over are ever used.
    target : int
        Column of `all_levels` this forecaster scores (the series' position in VALUE_COLUMNS).
    cache : dict of tuple of (str, int, int) to tuple of (np.ndarray, np.ndarray)
        Shared, mutated store of system_forecast results, keyed by (kind, months of history,
        steps); the same dict is passed to both series' forecasters.

    Returns
    -------
    Forecaster
        A forecaster for the harness; see the nested forecast() for what it returns.
    """

    def forecast(levels: np.ndarray, steps: int) -> dict[str, np.ndarray]:
        """
        Forecast this forecaster's series from the system fitted on both series so far.

        Parameters
        ----------
        levels : np.ndarray
            This series' levels up to and including the origin, shape (months,), trillions,
            oldest first; only its length is used to cut both series, after the leakage check.
        steps : int
            Number of months to forecast.

        Returns
        -------
        dict of str to np.ndarray
            "forecast": this series' level path, shape (steps,), trillions.
            "sd_constant": its normal level forecast-error SD, shape (steps,), trillions.

        Raises
        ------
        AssertionError
            If the history cut from `all_levels` doesn't match the levels the harness passed,
            which would mean the system could see other months than the univariate models.

        Notes
        -----
        Processing:
        1. Cut both series to the length the harness handed over.
        2. Check that this series' column equals the handed-over levels (leakage guard).
        3. Fit and forecast the system once per (kind, origin, steps); the second series reuses
           the cached result.
        4. Return this series' column.
        """
        history = all_levels[: len(levels)]
        # Leakage guard: the system sees exactly the months the harness handed over.
        assert np.array_equal(history[:, target], levels)
        # len(levels) identifies the origin: both series' histories end at the same month.
        key = (kind, len(levels), steps)
        if key not in cache:
            cache[key] = system_forecast(kind, history, steps)
        mean, spread = cache[key]
        return {"forecast": mean[:, target], "sd_constant": spread[:, target]}

    return forecast


def sarimax_forecaster(
    model: dict[str, Any],
    other_model: dict[str, Any],
    all_levels: np.ndarray,
    target: int,
    lags: int,
    plans: dict[tuple[int, int, int], np.ndarray],
) -> Forecaster:
    """
    A forecaster for the SARIMAX family: one series' selected SARIMA plus the other series'
    last `lags` monthly changes as regressors.

    Parameters
    ----------
    model : dict of str to Any
        This series' selected SARIMA (orders "p", "q", "P", "Q"), as in sarima_forecaster.
    other_model : dict of str to Any
        The other series' selected SARIMA, which forecasts that series' future changes.
    all_levels : np.ndarray
        Levels of both series over the whole sample, shape (months, 2), trillions, columns in
        VALUE_COLUMNS order. Only the months the harness hands over are ever used.
    target : int
        Column of `all_levels` this forecaster scores; the other column is 1 - target.
    lags : int
        How many of the other series' past changes enter as regressors (one of SARIMAX_LAGS).
    plans : dict of tuple of (int, int, int) to np.ndarray
        Shared, mutated store of the other series' forecast changes, keyed by (its column,
        months of history, steps), so the three lag candidates fit it once per origin.

    Returns
    -------
    Forecaster
        A forecaster for the harness; see the nested forecast() for what it returns.
    """
    other = 1 - target

    def forecast(levels: np.ndarray, steps: int) -> dict[str, np.ndarray | bool]:
        """
        Fit the SARIMAX on the changes so far and forecast this series' levels.

        Parameters
        ----------
        levels : np.ndarray
            This series' levels up to and including the origin, shape (months,), trillions,
            oldest first; only its length is used to cut both series, after the leakage check.
        steps : int
            Number of months to forecast.

        Returns
        -------
        dict of str to np.ndarray or bool
            "forecast": the level path at h = 1..steps, shape (steps,), trillions.
            "converged": whether the SARIMAX's likelihood optimizer reported convergence.

        Raises
        ------
        AssertionError
            If the history cut from `all_levels` doesn't match the levels the harness passed.

        Notes
        -----
        Processing:
        1. Cut both series to the handed-over length and check this series against it
           (leakage guard, as in system_forecaster).
        2. Forecast the other series' changes with its selected SARIMA, once per origin.
        3. Fit the SARIMAX on this series' changes from month `lags` on, with the lag matrix of
           the other series' changes as regressors.
        4. Regressors for the forecast months: the lag matrix of the other series' actual
           changes followed by its forecast ones; the last `steps` rows belong to the forecast
           months. From h = 2 on they include forecast changes, never actual future ones.
        5. Point forecast: the last level plus the running total of the forecast changes.

        No intervals: the SARIMAX's forecast variance treats the planned regressors as known,
        so its intervals would be too narrow by construction.
        """
        history = all_levels[: len(levels)]
        # Leakage guard: both series end at the month the harness handed over.
        assert np.array_equal(history[:, target], levels)
        own_changes = np.diff(history[:, target])
        other_changes = np.diff(history[:, other])
        key = (other, len(levels), steps)
        if key not in plans:
            other_fit = fit_sarima(
                pd.Series(other_changes),
                other_model["p"],
                other_model["q"],
                other_model["P"],
                other_model["Q"],
            )
            plans[key] = np.asarray(other_fit.forecast(steps))
        fitted = fit_sarima(
            pd.Series(own_changes[lags:]),
            model["p"],
            model["q"],
            model["P"],
            model["Q"],
            exog=lagged_exog(other_changes, lags),
        )
        # The forecast months' regressors; the extended series' last forecast change is never
        # used, since a row only holds changes before its own month.
        future = lagged_exog(np.concatenate([other_changes, plans[key]]), lags)[-steps:]
        path = np.asarray(fitted.forecast(steps, exog=future))
        return {
            "forecast": levels[-1] + np.cumsum(path),
            "converged": bool(fitted.mle_retvals.get("converged", False)),
        }

    return forecast


def rolling_forecasts(
    dates: pd.Series,
    levels: np.ndarray,
    first_origin: int,
    last_target: int,
    forecasters: dict[str, Forecaster],
    label: str = "",
) -> pd.DataFrame:
    """
    Forecast from every origin in [first_origin, last_target) using only levels up to it,
    keeping targets up to last_target (inclusive) and up to MAX_HORIZON steps ahead.

    Parameters
    ----------
    dates : pd.Series
        Month of each level, same length as `levels`.
    levels : np.ndarray
        One series' levels over the whole sample, shape (months,), trillions.
    first_origin : int
        Position of the first origin: the last month of the previous split, so the first
        target is the split's first month.
    last_target : int
        Position of the split's last month; no forecast targets a month after it.
    forecasters : dict of str to Forecaster
        Model name -> forecaster.
    label : str, default=""
        Names the run in the progress lines, e.g. "validation | total assets".

    Returns
    -------
    pd.DataFrame
        One row per model, origin and horizon, with the columns "model", "origin" and "target"
        (dates), "horizon" (months), "actual" (trillions) and every key the forecaster
        returned: path values at that horizon ("forecast", "sd_*") and scalars ("nu_*",
        "converged") repeated on every horizon. Keys a model doesn't return are NaN.

    Notes
    -----
    Processing:
    1. For each origin, cut the history at it (expanding window: everything from the start).
    2. Forecast min(MAX_HORIZON, months left in the split) steps, so no target leaves the split.
    3. Write one row per horizon, with the actual level of the target month.

    Leakage: a forecaster only ever receives levels[: origin + 1], so every refit uses data up
    to the origin only. Near the split's end fewer horizons are scored, which is why the
    longer horizons have fewer forecasts.
    """
    rows = []
    progress = Progress(
        " | ".join(filter(None, [label, f"{len(forecasters)} models", "origin"])),
        last_target - first_origin,
    )
    for done, origin in enumerate(range(first_origin, last_target), start=1):
        history = levels[: origin + 1]
        steps = min(MAX_HORIZON, last_target - origin)
        for name, forecaster in forecasters.items():
            output = forecaster(history, steps)
            for horizon in range(1, steps + 1):
                row = {
                    "model": name,
                    "origin": dates.iloc[origin],
                    "horizon": horizon,
                    "target": dates.iloc[origin + horizon],
                    "actual": levels[origin + horizon],
                }
                # Paths give one value per horizon; scalars (nu, converged) apply to all.
                for key, value in output.items():
                    row[key] = value[horizon - 1] if np.ndim(value) else value
                rows.append(row)
        progress.step(done, f"{dates.iloc[origin]:%Y-%m}")
    return pd.DataFrame(rows)


def mase_scale(train_levels: np.ndarray) -> float:
    """
    In-sample MAE of the one-step naive forecast: the mean absolute monthly change.

    Parameters
    ----------
    train_levels : np.ndarray
        The training split's levels of one series, shape (months,), trillions.

    Returns
    -------
    float
        The MASE denominator, trillions.

    Notes
    -----
    Taken from the training split only and kept fixed for validation and test, so MASE is
    comparable across splits and the scaling never uses out-of-sample data.
    """
    return float(np.mean(np.abs(np.diff(train_levels))))


def accuracy(actual: np.ndarray, forecast: np.ndarray, scale: float) -> dict[str, float]:
    """
    Point-accuracy metrics of a set of forecasts.

    Parameters
    ----------
    actual : np.ndarray
        Actual levels, shape (forecasts,), trillions; must be non-zero (for MAPE).
    forecast : np.ndarray
        Forecast levels, same shape, trillions.
    scale : float
        MASE denominator from mase_scale, trillions.

    Returns
    -------
    dict of str to float
        "MAE" and "RMSE" (trillions), "MSE" (trillions squared), "MAPE" (%), "MASE" (unitless;
        below 1 is better than the training split's one-month naive).
    """
    errors = np.asarray(actual) - np.asarray(forecast)
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors**2))
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(100 * np.mean(np.abs(errors / np.asarray(actual)))),
        "MASE": mae / scale,
    }


def metrics_table(forecasts: pd.DataFrame, scale: float) -> pd.DataFrame:
    """
    Point-accuracy metrics per model and scored horizon.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Rolling forecasts of one series and split (rolling_forecasts' columns "model",
        "horizon", "actual" and "forecast").
    scale : float
        The series' MASE denominator, trillions.

    Returns
    -------
    pd.DataFrame
        One row per model and horizon in HORIZONS, with "model", "horizon", "n" (number of
        forecasts) and the METRICS columns, models in their first-appearance order.
    """
    rows = []
    for (model, horizon), group in forecasts.groupby(["model", "horizon"], sort=False):
        # Every horizon 1..12 is forecast (the paths need them), but only HORIZONS are scored.
        if horizon in HORIZONS:
            scores = accuracy(group["actual"].to_numpy(), group["forecast"].to_numpy(), scale)
            rows.append({"model": model, "horizon": horizon, "n": len(group), **scores})
    return pd.DataFrame(rows)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> tuple[float, float]:
    """
    Diebold-Mariano test of equal expected loss, with the Harvey-Leybourne-Newbold small-sample
    correction. A negative statistic means model a has the lower loss. Two-sided p-value from a
    t distribution with n - 1 degrees of freedom.

    Parameters
    ----------
    loss_a : np.ndarray
        Loss of model a per forecast (e.g. absolute or squared error), shape (forecasts,),
        in forecast-origin order.
    loss_b : np.ndarray
        Loss of model b for the same targets, same shape and order.
    horizon : int
        Forecast horizon h in months; sets the number of autocovariance lags (h - 1).

    Returns
    -------
    tuple of (float, float)
        The corrected DM statistic and its two-sided p-value. (0.0, 1.0) if the loss
        differences have no variance (e.g. identical forecasts).

    Notes
    -----
    Processing:
    1. Loss differences d = loss_a - loss_b and their deviations from the mean.
    2. Autocovariances of d at lags 0..h-1 (divided by n, not n - lag).
    3. Long-run variance: lag 0 plus twice the others (rectangular kernel); fall back to lag 0
       if that isn't positive, and return "no difference" if even that is zero.
    4. DM = mean(d) / sqrt(long-run variance / n).
    5. HLN correction: multiply by sqrt((n + 1 - 2h + h(h - 1) / n) / n).
    6. Two-sided p-value from Student-t with n - 1 degrees of freedom.

    Why HLN and a t distribution: with 56 or fewer forecasts per horizon, the plain DM test
    over-rejects; the correction and the t reference make it closer to its nominal size.
    """
    differences = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    count = len(differences)
    deviations = differences - differences.mean()
    # h-step forecasts from consecutive origins overlap, so their loss differences are
    # autocorrelated up to lag h - 1.
    autocovariances = [
        np.sum(deviations[lag:] * deviations[: count - lag]) / count for lag in range(horizon)
    ]
    long_run_variance = autocovariances[0] + 2 * sum(autocovariances[1:])
    # shortcut: the rectangular kernel can turn negative for h > 1; fall back to the lag-0
    # variance then, as common implementations do. Upgrade to a Bartlett kernel if this recurs.
    if long_run_variance <= 0:
        long_run_variance = autocovariances[0]
    if long_run_variance == 0:
        return 0.0, 1.0
    statistic = differences.mean() / np.sqrt(long_run_variance / count)
    statistic *= np.sqrt((count + 1 - 2 * horizon + horizon * (horizon - 1) / count) / count)
    return float(statistic), float(2 * stats.t.sf(abs(statistic), count - 1))


def dm_against(forecasts: pd.DataFrame, reference: str, prefix: str = "DM") -> pd.DataFrame:
    """
    Each other model against the reference model, per horizon, with absolute and squared
    error losses; columns are named prefix_absolute, prefix_absolute_p, and so on.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Rolling forecasts of one series and split (rolling_forecasts' columns "model",
        "origin", "horizon", "actual" and "forecast"); must include the reference model at
        every (origin, horizon) the other models forecast.
    reference : str
        Model every other model is tested against (naive, or the selected SARIMA).
    prefix : str, default="DM"
        Prefix of the result columns ("DM" or "DM_SARIMA").

    Returns
    -------
    pd.DataFrame
        One row per non-reference model and horizon in HORIZONS: "model", "horizon",
        "<prefix>_absolute", "<prefix>_absolute_p", "<prefix>_squared", "<prefix>_squared_p".
        A negative statistic means the model's loss is lower than the reference's.

    Notes
    -----
    Processing:
    1. Compute every forecast's error (actual - forecast).
    2. Index the reference's errors by (origin, horizon).
    3. For each other model and scored horizon, pair its errors with the reference's at the
       same origins, and run diebold_mariano on their absolute and on their squared errors.

    Absolute loss matches MAE and squared loss matches MSE, so the test answers the same
    question as each metric.
    """
    errors = forecasts.assign(error=forecasts["actual"] - forecasts["forecast"])
    reference_errors = errors[errors["model"] == reference].set_index(["origin", "horizon"])
    rows = []
    for (model, horizon), group in errors[errors["model"] != reference].groupby(
        ["model", "horizon"], sort=False
    ):
        if horizon not in HORIZONS:
            continue
        model_errors = group.set_index(["origin", "horizon"])["error"]
        # Aligned by (origin, horizon), so both losses in a pair are for the same target month.
        matching = reference_errors.loc[model_errors.index, "error"]
        row = {"model": model, "horizon": horizon}
        for loss, function in (("absolute", np.abs), ("squared", np.square)):
            statistic, p_value = diebold_mariano(
                function(model_errors.to_numpy()), function(matching.to_numpy()), horizon
            )
            row |= {f"{prefix}_{loss}": statistic, f"{prefix}_{loss}_p": p_value}
        rows.append(row)
    return pd.DataFrame(rows)


def dm_verdict(statistic: float, p_value: float) -> str:
    """
    Short report text for one Diebold-Mariano result.

    Parameters
    ----------
    statistic : float
        DM statistic (negative: the model beats the reference).
    p_value : float
        Its two-sided p-value; NaN when the test wasn't run (e.g. for the reference itself).

    Returns
    -------
    str
        "-" when there is no test, "p=0.xx" when not significant at SIGNIFICANCE_LEVEL,
        otherwise "better p=0.xxx" or "worse p=0.xxx".
    """
    if not np.isfinite(p_value):
        return "-"
    if p_value >= SIGNIFICANCE_LEVEL:
        return f"p={p_value:.2f}"
    return f"{'better' if statistic < 0 else 'worse'} p={p_value:.3f}"


def standardized_quantiles(nu: float) -> np.ndarray:
    """
    QUANTILES of a zero-mean, unit-variance shock: normal, or Student-t rescaled to unit
    variance (arch's parametrization).

    Parameters
    ----------
    nu : float
        Student-t degrees of freedom (> 2); inf (or NaN) for normal shocks.

    Returns
    -------
    np.ndarray
        One quantile per QUANTILES level, shape (5,), unitless.

    Notes
    -----
    A Student-t with nu degrees of freedom has variance nu / (nu - 2); multiplying by
    sqrt((nu - 2) / nu) gives it unit variance, so the forecast SD keeps its meaning and only
    the tails change. Any non-finite nu, including the NaN of a model without that volatility
    column, falls back to normal.
    """
    if not np.isfinite(nu):
        return stats.norm.ppf(QUANTILES)
    return stats.t.ppf(QUANTILES, nu) * np.sqrt((nu - 2) / nu)


def quantile_forecasts(
    forecast: np.ndarray | pd.Series, spread: np.ndarray | pd.Series, nu: np.ndarray
) -> np.ndarray:
    """
    One row of QUANTILES per forecast: the point forecast plus the spread times the shock's
    standardized quantiles.

    Parameters
    ----------
    forecast : np.ndarray or pd.Series
        Point forecasts, shape (forecasts,), trillions.
    spread : np.ndarray or pd.Series
        Their level forecast-error standard deviations, same shape, trillions.
    nu : np.ndarray
        Student-t degrees of freedom per forecast (inf for normal), same shape.

    Returns
    -------
    np.ndarray
        Quantile forecasts, shape (forecasts, 5), trillions, columns in QUANTILES order.
    """
    # shortcut: a sum of Student-t shocks isn't exactly Student-t; treating the level error as
    # one is the usual approximation. Upgrade to simulated paths if the tails miscalibrate.
    offsets = np.vstack([standardized_quantiles(value) for value in nu])
    return np.asarray(forecast)[:, None] + np.asarray(spread)[:, None] * offsets


def pinball_loss(actual: np.ndarray, quantiles: np.ndarray) -> float:
    """
    Mean quantile loss over all forecasts and QUANTILES: under-forecasts cost q per unit,
    over-forecasts 1 - q. Lower is better; it rewards intervals that are calibrated and narrow.

    Parameters
    ----------
    actual : np.ndarray
        Actual levels, shape (forecasts,), trillions.
    quantiles : np.ndarray
        Quantile forecasts, shape (forecasts, 5), trillions, columns in QUANTILES order.

    Returns
    -------
    float
        Mean pinball loss, trillions.

    Notes
    -----
    With gap = actual - quantile, the loss is q * gap when the actual is above the quantile
    (gap > 0) and (q - 1) * gap when below; max() of the two picks the right branch, since the
    other one is negative.
    """
    gaps = np.asarray(actual)[:, None] - quantiles
    levels = np.array(QUANTILES)
    return float(np.mean(np.maximum(levels * gaps, (levels - 1) * gaps)))


def interval_scores(actual: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    """
    Calibration and sharpness scores of a set of quantile forecasts.

    Parameters
    ----------
    actual : np.ndarray
        Actual levels, shape (forecasts,), trillions.
    quantiles : np.ndarray
        Quantile forecasts, shape (forecasts, 5), trillions, columns in QUANTILES order.

    Returns
    -------
    dict of str to float
        "coverage_80" / "coverage_95": share of actuals inside the 10-90% / 2.5-97.5% interval
        (targets 0.80 / 0.95); "width_80" / "width_95": mean interval width, trillions;
        "pinball": mean pinball loss, trillions; "below_<q>" for each quantile (e.g.
        "below_2.5"): share of actuals below it (target q / 100).
    """
    # Relies on QUANTILES' order: 2.5, 10, 50, 90, 97.5%.
    low_95, low_80, _, high_80, high_95 = quantiles.T
    scores = {
        "coverage_80": float(np.mean((actual >= low_80) & (actual <= high_80))),
        "coverage_95": float(np.mean((actual >= low_95) & (actual <= high_95))),
        "width_80": float(np.mean(high_80 - low_80)),
        "width_95": float(np.mean(high_95 - low_95)),
        "pinball": pinball_loss(actual, quantiles),
    }
    for level, column in zip(QUANTILES, quantiles.T):
        # :g drops trailing zeros, so the names are below_2.5, below_10, ..., below_97.5.
        scores[f"below_{100 * level:g}"] = float(np.mean(actual < column))
    return scores


def interval_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Interval scores per variance variant, model and scored horizon.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Rolling forecasts of one series and split (rolling_forecasts' columns), with an
        "sd_<variance>" column for every variant some model produced, and "nu_<variance>" for
        the volatility models.

    Returns
    -------
    pd.DataFrame
        One row per variance in VARIANCES, model with that spread, and horizon in HORIZONS:
        "model", "variance", "horizon", "n" and interval_scores' columns.

    Notes
    -----
    Processing:
    1. For each variance variant whose "sd_<variance>" column exists, keep the rows that have
       it (models without intervals, such as smoothing, are NaN there and drop out).
    2. Per model and scored horizon, build the quantiles: Student-t with the rows' nu where a
       "nu_<variance>" column exists (rows without it give NaN, i.e. normal), normal otherwise.
    3. Score them with interval_scores.
    """
    rows = []
    for variance in VARIANCES:
        spread_column = f"sd_{variance}"
        if spread_column not in forecasts:
            continue
        with_spread = forecasts.dropna(subset=[spread_column])
        for (model, horizon), group in with_spread.groupby(["model", "horizon"], sort=False):
            if horizon not in HORIZONS:
                continue
            # Constant variance has no nu column: all-inf means normal quantiles throughout.
            nu = (
                group[f"nu_{variance}"]
                if f"nu_{variance}" in group
                else np.full(len(group), np.inf)
            )
            quantiles = quantile_forecasts(group["forecast"], group[spread_column], np.asarray(nu))
            rows.append(
                {
                    "model": model,
                    "variance": variance,
                    "horizon": horizon,
                    "n": len(group),
                    **interval_scores(group["actual"].to_numpy(), quantiles),
                }
            )
    return pd.DataFrame(rows)


def convergence_failures(forecasts: pd.DataFrame) -> dict[str, list[str]]:
    """
    Per SARIMA model, the origins (months) whose refit didn't converge.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Rolling forecasts of one series and split; only models that report "converged" (the
        SARIMAs) have a value in that column.

    Returns
    -------
    dict of str to list of str
        Model name -> origins whose refit failed, as "YYYY-MM"; an empty list for a model whose
        refits all converged. Empty if no model reports convergence.

    Notes
    -----
    "converged" is repeated on every horizon of a refit, so rows are de-duplicated per model
    and origin first: each refit is counted once.
    """
    if "converged" not in forecasts:
        return {}
    refits = forecasts.dropna(subset=["converged"]).drop_duplicates(["model", "origin"])
    return {
        model: [
            f"{origin:%Y-%m}" for origin in group.loc[~group["converged"].astype(bool), "origin"]
        ]
        for model, group in refits.groupby("model", sort=False)
    }


def accuracy_report(
    metrics: pd.DataFrame, selected_by_family: dict[str, dict[str, str]]
) -> list[str]:
    """
    Report lines for one split's point accuracy: per series and horizon, a table of every model.

    Parameters
    ----------
    metrics : pd.DataFrame
        One split's metrics: "series", "model", "horizon", "n", the METRICS columns, and the
        DM_COLUMNS and DM_SARIMA_COLUMNS (NaN where the test wasn't run).
    selected_by_family : dict of str to dict of str to str
        Family ("SARIMA", "system", "smoothing", "combination", "SARIMAX") -> series -> selected
        model.

    Returns
    -------
    list of str
        Report lines: per series, a header naming the selected models, then per horizon a
        table sorted by MASE (best first) and a line naming the lowest-MAE model and its gain
        over naive.

    Notes
    -----
    Sorting by MASE is the same as sorting by MAE within a series, since MASE is MAE divided
    by one constant per series; that is also why the gain over naive can be read from the MASE
    ratio and reported "in MAE".
    """
    lines = []
    for column in VALUE_COLUMNS:
        series_metrics = metrics[metrics["series"] == column]
        chosen = ", ".join(
            f"{family} {picks[column]}" for family, picks in selected_by_family.items()
        )
        lines.append(f"{column}  (selected on validation: {chosen})")
        for horizon in HORIZONS:
            table = series_metrics[series_metrics["horizon"] == horizon].sort_values("MASE")
            naive_mase = table.loc[table["model"] == NAIVE, "MASE"].iloc[0]
            lines.append(f"  {horizon}-month horizon ({table['n'].iloc[0]} forecasts)")
            lines.append(
                f"    {'model':<30}{'MAE':>9}{'MSE':>11}{'RMSE':>9}{'MAPE %':>9}{'MASE':>8}"
                f"   {'DM vs naive (abs)':<20}{'DM vs naive (sq)':<20}"
                f"{'DM vs SARIMA (abs)':<20}{'DM vs SARIMA (sq)':<20}"
            )
            for row in table.itertuples():
                lines.append(
                    f"    {row.model:<30}{row.MAE:9.4f}{row.MSE:11.6f}{row.RMSE:9.4f}"
                    f"{row.MAPE:9.2f}{row.MASE:8.3f}"
                    f"   {dm_verdict(row.DM_absolute, row.DM_absolute_p):<20}"
                    f"{dm_verdict(row.DM_squared, row.DM_squared_p):<20}"
                    f"{dm_verdict(row.DM_SARIMA_absolute, row.DM_SARIMA_absolute_p):<20}"
                    f"{dm_verdict(row.DM_SARIMA_squared, row.DM_SARIMA_squared_p):<20}"
                )
            best = table.iloc[0]
            verdict = (
                f"beats naive by {100 * (1 - best['MASE'] / naive_mase):.0f}% in MAE"
                if best["model"] != NAIVE
                else "nothing beats naive"
            )
            lines.append(f"    Lowest MAE: {best['model']} ({verdict})")
        lines.append("")
    return lines


def interval_report(intervals: pd.DataFrame) -> list[str]:
    """
    Report lines for one split's forecast intervals: per series and horizon, every model and
    variance variant.

    Parameters
    ----------
    intervals : pd.DataFrame
        One split's interval scores (interval_table's columns plus "series").

    Returns
    -------
    list of str
        Report lines: a header with the targets, then per series and horizon a table sorted by
        pinball loss (best first) with coverage, width, pinball and the below-quantile shares.
    """
    lines = [
        "  Forecast intervals (targets: 80% and 95% coverage; below_q should be close to q)",
    ]
    for column in VALUE_COLUMNS:
        series_intervals = intervals[intervals["series"] == column]
        lines.append(f"  {column}")
        for horizon in HORIZONS:
            table = series_intervals[series_intervals["horizon"] == horizon].sort_values("pinball")
            lines.append(f"    {horizon}-month horizon")
            lines.append(
                f"      {'model':<30}{'variance':<10}{'cov 80%':>8}{'cov 95%':>8}"
                f"{'width 80%':>11}{'width 95%':>11}{'pinball':>9}"
                "   below 2.5 / 10 / 50 / 90 / 97.5%"
            )
            # Plain dictionaries: itertuples would rename the "below_2.5" columns.
            for row in table.to_dict("records"):
                below = " / ".join(f"{row[f'below_{100 * q:g}']:.2f}" for q in QUANTILES)
                lines.append(
                    f"      {row['model']:<30}{row['variance']:<10}{row['coverage_80']:8.2f}"
                    f"{row['coverage_95']:8.2f}{row['width_80']:11.4f}{row['width_95']:11.4f}"
                    f"{row['pinball']:9.4f}   {below}"
                )
        lines.append("")
    return lines


def evaluation_report(
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]],
    periods: dict[str, str],
    scales: dict[str, float],
    selected_by_family: dict[str, dict[str, str]],
) -> str:
    """
    The full text report: the method, then each split's accuracy, intervals and failed refits.

    Parameters
    ----------
    results : dict of str to tuple of (pd.DataFrame, pd.DataFrame, dict of str to list of str)
        Split name -> (that split's metrics, its interval scores, its convergence failures as
        model -> failed origins "YYYY-MM").
    periods : dict of str to str
        "start" -> the first month of the data ("YYYY-MM"), and each split -> its target months
        ("YYYY-MM to YYYY-MM").
    scales : dict of str to float
        Series -> MASE denominator, trillions.
    selected_by_family : dict of str to dict of str to str
        Family -> series -> model selected on validation.

    Returns
    -------
    str
        The report text, lines joined with newlines.

    Notes
    -----
    Processing:
    1. Write the method paragraphs (setup, systems, smoothing, combination, metrics, DM tests,
       intervals, the COVID decision).
    2. Per split: the accuracy tables, the interval tables, and the refits that failed to
       converge, with their total.
    3. Point to the CSVs and figures, as project-relative paths.
    """
    lines = [
        (
            "Out-of-sample forecast evaluation: SARIMA, VAR / VECM, smoothing baselines, and the "
            "SARIMA + naive average"
        ),
        (
            "Rolling origin, expanding window: each model is refitted at every monthly origin on "
            f"all data up to it (data start {periods['start']}); each origin is one fold. "
            "Orders come from training; the selected model per series and family (SARIMA, "
            "system, smoothing, ML, SARIMAX) from validation (lowest one-month MASE), fixed "
            "before test. "
            "Each family's test is run once, with all its models."
        ),
        (
            f"Systems: {SYSTEMS[0]} on the differences and {SYSTEMS[1]} (rank {VECM_RANK}) on "
            "the levels, fitted on both series up to each origin and forecasting both; normal "
            "intervals from the system's forecast-error covariance (no GARCH)."
        ),
        (
            "Smoothing: SMA(k) forecasts the average of the last k levels; single, double "
            "(Holt, plain and damped) and triple (Holt-Winters, 12-month season) exponential "
            "smoothing, statsmodels' ExponentialSmoothing on the levels, re-estimated at every "
            "origin. Point forecasts only."
        ),
        (
            f"{COMBINATION}: the equal-weight average of the series' selected SARIMA and the "
            "naive (no-change) forecast, from the same origins. Point forecasts only."
        ),
        (
            "SARIMAX: the series' selected SARIMA (same orders) plus the other series' last "
            f"{', '.join(map(str, SARIMAX_LAGS))} monthly changes as regressors (the VAR orders "
            "of steps V2 and V6), so DM vs SARIMA measures what the other series adds. Forecast "
            "months whose regressors aren't known yet use the other series' own selected-SARIMA "
            "forecasts from the same origin. Point forecasts only: the intervals would treat "
            "those planned regressors as known."
        ),
        (
            "Lag-feature ML: Ridge and Lasso (standardized features, penalty by time-ordered "
            "cross-validation in each window), random forest and histogram gradient boosting "
            "(trees chosen on the window's last quarter, in time order), each predicting next "
            "month's change of one series from the last "
            f"{', '.join(map(str, SARIMAX_LAGS))} changes of both series, refitted at every "
            "origin and forecasting recursively. Point forecasts only. Settings and their "
            "justifications: autoregressive_experiments/configs/hyperparameters.json."
        ),
        (
            "Point errors are on the levels: MAE, RMSE in trillions; MSE in trillions squared; "
            "MAPE in %; MASE = MAE / the training split's in-sample MAE of the one-month naive "
            "forecast (scales: "
            + ", ".join(f"{column} {scale:.4f}" for column, scale in scales.items())
            + ")."
        ),
        (
            "DM vs naive: Diebold-Mariano test of equal expected loss against the naive "
            "forecast, absolute-error loss (matches MAE) and squared-error loss (matches MSE), "
            "with the Harvey-Leybourne-Newbold correction and h - 1 autocovariance lags. "
            f"'better' / 'worse' when significant at {SIGNIFICANCE_LEVEL:.0%}, otherwise the "
            "p-value. DM vs SARIMA: the same test, each non-SARIMA model against the series' "
            "selected SARIMA: does any other approach beat it?"
        ),
        (
            "Intervals: quantiles of each model's forecast distribution on the levels, with a "
            "constant shock variance, U5b's GARCH, or U5c's EGARCH / GJR-GARCH (each refitted "
            "at every origin; EGARCH's multi-step variances simulated with a fixed seed); normal "
            "for constant variance, unit-variance Student-t otherwise; naive uses random-walk "
            "quantiles. Pinball = mean quantile loss over the 5 quantiles."
        ),
        "The COVID months (2020) are in validation and are kept, by the user's decision.",
        "",
    ]
    for split in SPLITS:
        metrics, intervals, failures = results[split]
        lines.append(f"===== {split.upper()}: targets {periods[split]} =====")
        lines.append("")
        lines.extend(accuracy_report(metrics, selected_by_family))
        lines.extend(interval_report(intervals))
        # Models with at least one failed refit; the others map to empty lists.
        failed = {model: months for model, months in failures.items() if months}
        refits = sum(len(months) for months in failures.values())
        lines.append(
            "  Refits that failed to converge: "
            + (
                "; ".join(f"{model}: {', '.join(months)}" for model, months in failed.items())
                if failed
                else "none"
            )
        )
        lines.append(f"  ({refits} failed refits in total for this split)" if failed else "")
        lines.append("")
    lines.append(f"All numbers: {repo_path(METRICS_PATH)}, {repo_path(INTERVALS_PATH)}")
    lines.append(f"Figures: {repo_path(FIGURE_PATH)}, {repo_path(INTERVALS_FIGURE_PATH)}")
    return "\n".join(lines)


def series_title(column: str) -> str:
    """
    Readable series name for chart titles.

    Parameters
    ----------
    column : str
        Column name, e.g. "fed_total_assets_T".

    Returns
    -------
    str
        The name without the "_T" (trillions) suffix and with spaces, e.g. "fed total assets".
    """
    return column.removesuffix("_T").replace("_", " ")


def family_styles(
    selected_by_family: dict[str, dict[str, str]], column: str
) -> dict[str, tuple[str, float, str]]:
    """
    Each family's selected model for the series, then drift and seasonal naive: name ->
    (color, line width, legend label).

    Parameters
    ----------
    selected_by_family : dict of str to dict of str to str
        Family -> series -> model selected on validation.
    column : str
        The series.

    Returns
    -------
    dict of str to tuple of (str, float, str)
        Model name -> (line color, line width, legend label), in drawing order. Naive isn't
        included: the accuracy panels draw it as the reference line at 1.
    """
    styles: dict[str, tuple[str, float, str]] = {
        picks[column]: (FAMILY_COLORS[family], 2, f"{family}: {picks[column]}")
        for family, picks in selected_by_family.items()
    }
    for name in ("drift", "seasonal naive"):
        color, width = BASELINE_STYLES[name]
        styles[name] = (color, width, name)
    return styles


def evaluation_figure(
    metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    selected_by_family: dict[str, dict[str, str]],
    test_start: pd.Timestamp,
) -> Figure:
    """
    Per series: each family's selected model's MAE relative to naive by horizon, on
    validation and on test (every model is in the report's tables), and the actual path against
    the selected SARIMA's, the best other family's and the naive one-month forecasts.

    Parameters
    ----------
    metrics : pd.DataFrame
        All metrics: "split", "series", "model", "horizon", "MASE" and "DM_absolute_p".
    forecasts : pd.DataFrame
        All rolling forecasts, with "split" and "series" columns added.
    selected_by_family : dict of str to dict of str to str
        Family -> series -> model selected on validation.
    test_start : pd.Timestamp
        First test month, where the validation | test divider is drawn.

    Returns
    -------
    Figure
        One row per series, three panels: validation and test accuracy relative to naive, and
        the one-month forecast paths.

    Notes
    -----
    Processing, per series:
    1. Accuracy panels (validation, test): each style's MASE divided by naive's at the same
       horizon, filled markers where the DM test (absolute loss) against naive is significant.
    2. Path panel: the actual levels and the one-month forecasts of the selected SARIMA, of the
       other family's pick with the lowest one-month validation MASE, and of naive.

    The rival is chosen on validation only, so the figure's choice is fixed before test like
    every other selection.
    """
    apply_chart_style()
    rows = len(VALUE_COLUMNS)
    figure, axes = plt.subplots(
        rows,
        3,
        figsize=(21, 5.4 * rows),
        squeeze=False,
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": (1, 1, 1.4)},
    )
    for row, column in enumerate(VALUE_COLUMNS):
        styles = family_styles(selected_by_family, column)
        for axis in axes[row]:
            axis.set_facecolor(SURFACE)
        for split, axis in zip(SPLITS, axes[row][:2]):
            split_metrics = metrics[(metrics["series"] == column) & (metrics["split"] == split)]
            # MASE grows with the horizon for every model, which hides the differences between
            # them; dividing by naive's MASE at the same horizon shows the decision directly.
            naive_mase = split_metrics[split_metrics["model"] == NAIVE].set_index("horizon")[
                "MASE"
            ]
            for name, (color, width, label) in styles.items():
                model_metrics = split_metrics[split_metrics["model"] == name].set_index("horizon")
                relative = model_metrics["MASE"] / naive_mase
                axis.plot(relative.index, relative, color=color, linewidth=width, label=label)
                # Filled: significantly different from naive (DM, absolute loss); hollow: not.
                significant = model_metrics["DM_absolute_p"] < SIGNIFICANCE_LEVEL
                for filled, face in ((significant, color), (~significant, SURFACE)):
                    axis.scatter(
                        relative.index[filled],
                        relative[filled],
                        s=36,
                        color=face,
                        edgecolors=color,
                        linewidths=1.5,
                        zorder=3,
                    )
            axis.axhline(1, color=PRIMARY_INK, linewidth=2, label=NAIVE)
            axis.set_xticks(HORIZONS)
            axis.set_xlabel("Horizon (months)")
            axis.set_ylabel("MAE relative to naive")
            axis.set_title(
                f"{series_title(column)}, {split}: each family's selected model vs naive\n"
                "below the naive line at 1: more accurate than naive\n"
                "filled point: significantly different from naive (Diebold-Mariano, 5%)",
                loc="left",
                fontsize=10,
            )
            # Below the axes: the lines can sit anywhere between the worst and the best model.
            axis.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2, fontsize=8, frameon=False
            )

        # The SARIMA pick against whichever other family's pick did best on validation.
        validation = metrics[
            (metrics["series"] == column)
            & (metrics["split"] == "validation")
            & (metrics["horizon"] == 1)
        ].set_index("model")["MASE"]
        sarima = selected_by_family["SARIMA"][column]
        rival = min(
            (picks[column] for family, picks in selected_by_family.items() if family != "SARIMA"),
            key=lambda name: validation[name],
        )
        path_axis = axes[row][2]
        one_month = forecasts[(forecasts["series"] == column) & (forecasts["horizon"] == 1)]
        # Sorted by date: the average's rows are appended split by split, not in time order.
        one_month = one_month.sort_values("target")
        # Naive's rows carry the actual level of every target month once, so they give the path.
        actual = one_month[one_month["model"] == NAIVE]
        path_axis.plot(
            actual["target"], actual["actual"], color=PRIMARY_INK, linewidth=2, label="actual"
        )
        for name in (sarima, rival, NAIVE):
            model_path = one_month[one_month["model"] == name]
            path_axis.plot(
                model_path["target"],
                model_path["forecast"],
                color=styles[name][0] if name != NAIVE else SECONDARY_INK,
                linewidth=1.5,
                marker="o",
                markersize=2.5,
                label=f"{name}, 1 month ahead",
            )
        path_axis.axvline(test_start, color=SECONDARY_INK, linewidth=1)
        # x in data units (the date), y in axes units (1 = the top edge).
        path_axis.annotate(
            "validation | test ",
            (test_start, 1),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            color=SECONDARY_INK,
            fontsize=9,
        )
        path_axis.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(path_axis.xaxis.get_major_locator())
        )
        path_axis.set_ylabel("Trillions")
        path_axis.set_title(
            f"{series_title(column)}: actual vs one-month forecasts\n"
            f"selected {sarima}, the best other family's {rival}, and naive\n",
            loc="left",
            fontsize=10,
        )
        path_axis.legend(loc="upper left", fontsize=8, frameon=False)

    figure.tight_layout()
    return figure


def intervals_figure(intervals: pd.DataFrame, selected: dict[str, str]) -> Figure:
    """
    Per series, interval level and split: the selected SARIMA's coverage by horizon with
    constant variance, GARCH, EGARCH and GJR-GARCH, against the target.

    Parameters
    ----------
    intervals : pd.DataFrame
        All interval scores: "split", "series", "model", "variance", "horizon",
        "coverage_80" and "coverage_95".
    selected : dict of str to str
        Series -> the SARIMA selected on validation.

    Returns
    -------
    Figure
        One row per (series, interval level) pair and one column per split, with one shared
        legend at the top right.
    """
    apply_chart_style()
    panels = [(column, level) for column in VALUE_COLUMNS for level in ("80", "95")]
    figure, axes = plt.subplots(
        len(panels), len(SPLITS), figsize=(14, 3.6 * len(panels)), squeeze=False, facecolor=SURFACE
    )
    for row, (column, level) in enumerate(panels):
        target = int(level) / 100
        for split, axis in zip(SPLITS, axes[row]):
            axis.set_facecolor(SURFACE)
            chosen = intervals[
                (intervals["series"] == column)
                & (intervals["split"] == split)
                & (intervals["model"] == selected[column])
            ]
            axis.axhline(target, color=PRIMARY_INK, linewidth=1)
            # x in axes units (1 = the right edge), y in data units (the target coverage).
            axis.annotate(
                f" {level}% target",
                (1, target),
                xycoords=("axes fraction", "data"),
                va="center",
                color=SECONDARY_INK,
                fontsize=9,
            )
            for variance in VARIANCES:
                lines = chosen[chosen["variance"] == variance].set_index("horizon")
                axis.plot(
                    lines.index,
                    lines[f"coverage_{level}"],
                    color=VARIANCE_COLORS[variance],
                    linewidth=2,
                    marker="o",
                    markersize=6,
                    label=VARIANCE_NAMES[variance],
                )
            axis.set_ylim(0, 1.05)
            axis.set_xticks(HORIZONS)
            axis.set_xlabel("Horizon (months)")
            axis.set_ylabel("Share of actuals inside")
            axis.set_title(
                f"{series_title(column)}, {split}: {selected[column]}, {level}% interval\n"
                "on the target line: honest; below: too narrow; above: too wide",
                loc="left",
                fontsize=10,
            )
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncols=4, frameon=False, fontsize=9)
    # Leave room on the right for the target labels and about 0.35 inch at the top for the
    # shared legend.
    figure.tight_layout(rect=(0, 0, 0.95, 1 - 0.35 / figure.get_figheight()))
    return figure


def ml_figure(metrics: pd.DataFrame, selected_by_family: dict[str, dict[str, str]]) -> Figure:
    """
    One row per series, one column per split: each ML model's MAE relative to naive by horizon,
    against the selected SARIMA and the selected smoothing model.

    Parameters
    ----------
    metrics : pd.DataFrame
        Every model's metrics, both splits, with "MASE" and "DM_absolute_p".
    selected_by_family : dict of str to dict of str to str
        Family -> series -> model selected on validation; uses "SARIMA" and "smoothing".

    Returns
    -------
    Figure
        A figure with one row per series and one column per split. The caller saves and
        closes it.

    Notes
    -----
    Processing, per series:
    1. For each ML model type, pick its lag count on validation (lowest one-month MASE, the
       harness's rule), so the lines show what each model type can do without looking at test.
    2. Per split, draw each pick's MASE divided by naive's at the same horizon, with the
       selected SARIMA and the selected smoothing model for comparison.
    3. Filled points are significantly different from naive (DM, absolute loss, 5%).

    Palette slots 1-4 go to the four ML model types and slot 5 to SARIMA; the smoothing model
    is drawn in the baseline grey, as a reference.
    """
    apply_chart_style()
    figure, axes = plt.subplots(
        len(VALUE_COLUMNS),
        2,
        figsize=(15, 5.4 * len(VALUE_COLUMNS)),
        squeeze=False,
        facecolor=SURFACE,
    )
    for row, column in enumerate(VALUE_COLUMNS):
        validation = metrics[
            (metrics["series"] == column)
            & (metrics["split"] == "validation")
            & (metrics["horizon"] == 1)
        ].set_index("model")["MASE"]
        picks = {
            model: min(
                (name for (kind, _), name in ML_NAMES.items() if kind == model),
                key=lambda name: validation[name],
            )
            for model in ML_MODELS
        }
        styles = {name: (color, 2.0) for name, color in zip(picks.values(), SERIES_PALETTE)}
        styles[selected_by_family["SARIMA"][column]] = (SERIES_PALETTE[4], 2.0)
        styles[selected_by_family["smoothing"][column]] = (MUTED_INK, 1.5)
        for split, axis in zip(SPLITS, axes[row]):
            axis.set_facecolor(SURFACE)
            split_metrics = metrics[(metrics["series"] == column) & (metrics["split"] == split)]
            naive_mase = split_metrics[split_metrics["model"] == NAIVE].set_index("horizon")[
                "MASE"
            ]
            for name, (color, width) in styles.items():
                model_metrics = split_metrics[split_metrics["model"] == name].set_index("horizon")
                relative = model_metrics["MASE"] / naive_mase
                axis.plot(relative.index, relative, color=color, linewidth=width, label=name)
                # Filled: significantly different from naive (DM, absolute loss); hollow: not.
                significant = model_metrics["DM_absolute_p"] < SIGNIFICANCE_LEVEL
                for filled, face in ((significant, color), (~significant, SURFACE)):
                    axis.scatter(
                        relative.index[filled],
                        relative[filled],
                        s=36,
                        color=face,
                        edgecolors=color,
                        linewidths=1.5,
                        zorder=3,
                    )
            axis.axhline(1, color=PRIMARY_INK, linewidth=2, label=NAIVE)
            axis.set_xticks(HORIZONS)
            axis.set_xlabel("Horizon (months)")
            axis.set_ylabel("MAE relative to naive")
            axis.set_title(
                f"{series_title(column)}, {split}: each ML model (lags chosen on validation)\n"
                "vs the selected SARIMA and smoothing model; below 1: more accurate than naive\n"
                "filled point: significantly different from naive (Diebold-Mariano, 5%)",
                loc="left",
                fontsize=10,
            )
            axis.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2, fontsize=8, frameon=False
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
    Run the rolling-origin evaluation on validation and test and write the report, CSVs and
    figures.

    Parameters
    ----------
    levels_path : Path, default=INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv"
        Cleaned monthly levels of both series, trillions.
    differences_path : Path, default=PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv"
        Their first differences, used to fit and check the SARIMA candidates on train.
    candidates_path : Path, default=CANDIDATES_PATH
        The SARIMA information-criteria table from step U3.

    Returns
    -------
    None
        Writes REPORT_PATH, METRICS_PATH, INTERVALS_PATH, FIGURE_PATH and
        INTERVALS_FIGURE_PATH, and prints the report.

    Raises
    ------
    FileNotFoundError
        If an input file is missing (raised by residual_checks.checked_models).

    Notes
    -----
    Processing:
    1. Fit and check the recommended SARIMA models on train; keep those that pass.
    2. For each passing model, choose its GARCH, EGARCH and GJR-GARCH variants on its training
       residuals.
    3. Split the levels 60 / 20 / 20 in time order; the validation origins start at train's
       last month, the test origins at validation's last month.
    4. Compute each series' MASE scale on train.
    5. For each split and series, run every forecaster (SARIMAs, systems, baselines,
       smoothing, lag-feature ML) from every origin, and score the point forecasts, DM
       against naive, the intervals and the failed refits.
    6. Select each family's model per series on validation (lowest one-month MASE).
    7. Add the SARIMA + naive average from the forecasts already made, and score it.
    8. Run the SARIMAX candidates (built on the selected SARIMAs) from the same origins, score
       them with DM against naive, and select one per series on validation.
    9. Test every non-SARIMA model against the series' selected SARIMA with DM.
    10. Write the CSVs, the report and the three figures.

    Selection uses validation only, and the test rows are scored by the same code, so the
    test numbers can't influence any choice.
    """
    start_run_log("evaluate")
    begin_phase("Fit and check the SARIMA models; choose their volatility variants")
    checked, _ = checked_models(levels_path, differences_path, candidates_path)
    passed = [result for result in checked if result["passed"]]
    # Each model's volatility variants, chosen on the training residuals: U5b's GARCH (final,
    # unchanged) and U5c's EGARCH and GJR-GARCH.
    volatility_variants: dict[tuple[str, str], dict[str, str]] = {}
    for result in passed:
        # Only the BIC-chosen variants are needed here, so the bootstrap checks are skipped.
        checks = asymmetric_check(result["residuals"], replications=0)
        volatility_variants[(result["series"], model_label(result))] = {
            "garch": checks["reference"],
            "egarch": checks[EGARCH_T]["variant"],
            "gjr": checks[GJR_T]["variant"],
        }
    levels = pd.read_csv(levels_path, parse_dates=[DATE_COLUMN])
    levels = levels.sort_values(DATE_COLUMN).reset_index(drop=True)
    splits = split_by_time(levels, split_boundaries(levels))
    # Positions in the full sample: the last month of train and of validation.
    train_end = len(splits["train"]) - 1
    validation_end = train_end + len(splits["validation"])
    # Split -> (first origin, last target): the origins start at the previous split's last
    # month, so the first target is the split's first month.
    windows = {
        "validation": (train_end, validation_end),
        "test": (validation_end, len(levels) - 1),
    }
    dates = levels[DATE_COLUMN]
    periods = {"start": f"{dates.iloc[0]:%Y-%m}"} | {
        split: f"{dates.iloc[first + 1]:%Y-%m} to {dates.iloc[last]:%Y-%m}"
        for split, (first, last) in windows.items()
    }

    scales = {
        column: mase_scale(splits["train"][column].to_numpy(dtype=float))
        for column in VALUE_COLUMNS
    }
    all_levels = levels[list(VALUE_COLUMNS)].to_numpy(dtype=float)
    # One fit per system and origin, shared by both series' forecasters.
    system_fits: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    # One fit per ML model, lag count and origin, shared the same way.
    ml_fits: dict[tuple[str, int, int, int], dict[str, np.ndarray]] = {}
    # Split -> model -> origins whose refit failed.
    failures: dict[str, dict[str, list[str]]]
    all_forecasts, all_metrics, all_intervals, failures = [], [], [], {}
    for split, (first_origin, last_target) in windows.items():
        begin_phase(f"{split.capitalize()}: SARIMA, systems, baselines, smoothing and ML")
        failures[split] = {}
        for target, column in enumerate(VALUE_COLUMNS):
            values = levels[column].to_numpy(dtype=float)
            forecasters: dict[str, Forecaster] = {
                model_label(m): sarima_forecaster(m, volatility_variants[(column, model_label(m))])
                for m in passed
                if m["series"] == column
            }
            forecasters |= {
                kind: system_forecaster(kind, all_levels, target, system_fits) for kind in SYSTEMS
            }
            forecasters |= {
                NAIVE: naive_forecast,
                "drift": drift_forecast,
                "seasonal naive": seasonal_naive_forecast,
            }
            forecasters |= {
                f"SMA({window})": moving_average_forecaster(window)
                for window in MOVING_AVERAGE_WINDOWS
            }
            forecasters |= {
                name: exponential_smoothing_forecaster(settings)
                for name, settings in EXPONENTIAL_SMOOTHING.items()
            }
            forecasters |= {
                name: ml_forecaster(model, lags, all_levels, target, ml_fits)
                for (model, lags), name in ML_NAMES.items()
            }
            forecasts = rolling_forecasts(
                dates,
                values,
                first_origin,
                last_target,
                forecasters,
                f"{split} | {series_title(column)}",
            )
            labels = {"split": split, "series": column}
            all_forecasts.append(forecasts.assign(**labels))
            all_metrics.append(
                metrics_table(forecasts, scales[column])
                .merge(dm_against(forecasts, NAIVE), on=["model", "horizon"], how="left")
                .assign(**labels)
            )
            all_intervals.append(interval_table(forecasts).assign(**labels))
            # Extend rather than replace: the series could share a SARIMA label.
            for model, months in convergence_failures(forecasts).items():
                failures[split].setdefault(model, []).extend(months)

    forecasts = pd.concat(all_forecasts, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    intervals = pd.concat(all_intervals, ignore_index=True)
    # Selection per family, on validation only: earlier families' choices are unchanged by new
    # families joining the harness.
    validation_one_month = metrics[(metrics["split"] == "validation") & (metrics["horizon"] == 1)]
    model = validation_one_month["model"]
    # SARIMA is whatever isn't a baseline, system or smoothing model: the labels vary by order.
    families = {
        "SARIMA": ~model.isin([*BASELINES, *SYSTEMS, *SMOOTHING, *ML_NAMES.values()]),
        "system": model.isin(SYSTEMS),
        "smoothing": model.isin(SMOOTHING),
        "ML": model.isin(ML_NAMES.values()),
    }
    selected_by_family: dict[str, dict[str, str]] = {
        family: {
            column: group.sort_values("MASE")["model"].iloc[0]
            for column, group in validation_one_month[mask].groupby("series")
        }
        for family, mask in families.items()
    }
    selected = selected_by_family["SARIMA"]
    # The average of the validation-selected SARIMA and naive, from the forecasts already made.
    combined = []
    for (split, column), group in forecasts.groupby(["split", "series"]):
        labels = {"split": split, "series": column}
        rows = combine_forecasts(group, selected[column], NAIVE, COMBINATION)
        combined.append(rows.assign(**labels))
        # dm_against needs the reference (naive) next to the new rows.
        pool = pd.concat([group[group["model"] == NAIVE], rows])
        metrics = pd.concat(
            [
                metrics,
                metrics_table(rows, scales[column])
                .merge(dm_against(pool, NAIVE), on=["model", "horizon"], how="left")
                .assign(**labels),
            ],
            ignore_index=True,
        )
    forecasts = pd.concat([forecasts, *combined], ignore_index=True)
    selected_by_family["combination"] = {column: COMBINATION for column in VALUE_COLUMNS}

    # SARIMAX builds on the selected SARIMA, so it runs after the selection above, from the same
    # origins; the other series' future changes come from its own selected SARIMA.
    by_label = {(m["series"], model_label(m)): m for m in passed}
    plans: dict[tuple[int, int, int], np.ndarray] = {}
    sarimax_rows = []
    for split, (first_origin, last_target) in windows.items():
        begin_phase(f"{split.capitalize()}: SARIMAX")
        for target, column in enumerate(VALUE_COLUMNS):
            other_column = VALUE_COLUMNS[1 - target]
            sarimax_forecasters = {
                name: sarimax_forecaster(
                    by_label[(column, selected[column])],
                    by_label[(other_column, selected[other_column])],
                    all_levels,
                    target,
                    lags,
                    plans,
                )
                for lags, name in SARIMAX_NAMES.items()
            }
            values = levels[column].to_numpy(dtype=float)
            rows = rolling_forecasts(
                dates,
                values,
                first_origin,
                last_target,
                sarimax_forecasters,
                f"SARIMAX | {split} | {series_title(column)}",
            )
            labels = {"split": split, "series": column}
            sarimax_rows.append(rows.assign(**labels))
            # dm_against needs the reference (naive) next to the new rows.
            naive_rows = forecasts[
                (forecasts["split"] == split)
                & (forecasts["series"] == column)
                & (forecasts["model"] == NAIVE)
            ]
            metrics = pd.concat(
                [
                    metrics,
                    metrics_table(rows, scales[column])
                    .merge(
                        dm_against(pd.concat([naive_rows, rows]), NAIVE),
                        on=["model", "horizon"],
                        how="left",
                    )
                    .assign(**labels),
                ],
                ignore_index=True,
            )
            for model_name, months in convergence_failures(rows).items():
                failures[split].setdefault(model_name, []).extend(months)
    forecasts = pd.concat([forecasts, *sarimax_rows], ignore_index=True)
    sarimax_one_month = metrics[
        (metrics["split"] == "validation")
        & (metrics["horizon"] == 1)
        & metrics["model"].isin(SARIMAX_NAMES.values())
    ]
    selected_by_family["SARIMAX"] = {
        column: group.sort_values("MASE")["model"].iloc[0]
        for column, group in sarimax_one_month.groupby("series")
    }
    begin_phase("DM tests against SARIMA, report and figures")
    # Does any other approach beat univariate SARIMA? Each one against the selected SARIMA.
    others = [*SYSTEMS, *SMOOTHING, COMBINATION, *SARIMAX_NAMES.values(), *ML_NAMES.values()]
    against_sarima = [
        dm_against(
            group[group["model"].isin([*others, selected[column]])],
            selected[column],
            prefix="DM_SARIMA",
        ).assign(split=split, series=column)
        for (split, column), group in forecasts.groupby(["split", "series"])
    ]
    # Left join: the baselines and the SARIMAs keep NaN in the DM_SARIMA columns.
    metrics = metrics.merge(
        pd.concat(against_sarima), on=["split", "series", "model", "horizon"], how="left"
    )
    metrics[
        ["split", "series", "model", "horizon", "n", *METRICS, *DM_COLUMNS, *DM_SARIMA_COLUMNS]
    ].to_csv(METRICS_PATH, index=False)
    intervals.to_csv(INTERVALS_PATH, index=False)

    results = {
        split: (
            metrics[metrics["split"] == split],
            intervals[intervals["split"] == split],
            failures[split],
        )
        for split in SPLITS
    }
    report = evaluation_report(results, periods, scales, selected_by_family)
    # The boosting cap must never bind: early stopping, not the cap, should end training.
    trees = forecasts["trees"].max()
    cap = HYPERPARAMETERS.ml.boosting_max_iterations
    report += (
        f"\nGradient boosting: at most {trees:.0f} trees chosen at any origin; the cap of {cap} "
        + (
            "was never reached.\n"
            if trees < cap
            else "WAS REACHED: the cap, not early stopping, ended training.\n"
        )
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    test_start = dates.iloc[validation_end + 1]
    # The main figure keeps the families it has colours for; ML gets its own figure.
    main_families = {f: picks for f, picks in selected_by_family.items() if f in FAMILY_COLORS}
    figure = evaluation_figure(metrics, forecasts, main_families, test_start)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    figure = intervals_figure(intervals, selected)
    figure.savefig(INTERVALS_FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    figure = ml_figure(metrics, selected_by_family)
    figure.savefig(ML_FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"Forecast evaluation written to {REPORT_PATH} and {FIGURES_DIR}")
    finish_run()


if __name__ == "__main__":
    app()
