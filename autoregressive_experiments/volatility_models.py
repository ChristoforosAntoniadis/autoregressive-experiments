"""
Volatility models (step U5b): does GARCH(1,1) earn its place on top of each SARIMA model?

Two-step estimation: the SARIMA mean models stay as fitted, and four variance models are fitted
to their residuals with the ``arch`` package: a constant variance, GARCH(1,1) with normal errors,
GARCH(1,1) with Student-t errors, and GARCH(1,1) Student-t on the break-adjusted residuals (each
regime divided by its own standard deviation, at the variance break that ``volatility_checks``
estimates). The last one stops the 2008 variance shift from posing as volatility persistence.
BIC picks between them. GARCH is carried forward only when BIC prefers it, its variance is
stationary (alpha + beta < 1), and the residuals divided by its monthly standard deviation no
longer show ARCH effects, judged by parametric-bootstrap p-values (the chi-squared tables don't
hold for residuals standardized by a fitted model).

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
from typing import Any

from arch import arch_model
from arch.univariate.base import ARCHModelResult
from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
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
from autoregressive_experiments.runlog import Progress, begin_phase, finish_run, start_run_log
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
# Report name -> keyword arguments for arch_model(). p = 1 ARCH term, q = 1 GARCH term.
VARIANCE_MODELS: dict[str, dict[str, Any]] = {
    CONSTANT: {"vol": "Constant", "dist": "normal"},
    GARCH_NORMAL: {"vol": "GARCH", "p": 1, "q": 1, "dist": "normal"},
    GARCH_T: {"vol": "GARCH", "p": 1, "q": 1, "dist": "t"},
}
# Appended to a model name to mean "fitted on the break-adjusted residuals" (see fit_candidate).
BREAK_SUFFIX = ", break-adjusted"
GARCH_T_BREAK = GARCH_T + BREAK_SUFFIX
# The regime scaling adds two parameters: the ratio of the two regimes' scales (the overall
# scale is GARCH's omega) and the break date.
BREAK_PARAMETERS = 2
# Asymmetric GARCH (step U5c): volatility may react more to falls than to rises. Separate new
# models; U5b's choice above is final and doesn't consider them.
EGARCH_T = "EGARCH(1,1), Student-t"
GJR_T = "GJR-GARCH(1,1), Student-t"
# o = 1 adds one asymmetric (leverage) term; arch's GJR is its GARCH volatility with o > 0.
ASYMMETRIC_MODELS: dict[str, dict[str, Any]] = {
    EGARCH_T: {"vol": "EGARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    GJR_T: {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
}
# Settings; each value's reasoning is in configs/hyperparameters.json.
# EGARCH has no closed-form multi-step variance forecast, so it is simulated, with a fixed
# seed so every rerun gives identical numbers.
SIMULATIONS = HYPERPARAMETERS.volatility_models.simulations
SIMULATION_SEED = HYPERPARAMETERS.volatility_models.simulation_seed
# Parametric bootstrap of the standardized-residual checks (Davison & Hinkley 1997, ch. 4): the
# chi-squared tables don't hold for residuals standardized by a fitted variance model (Li & Mak
# 1994), so each model's own simulations give the null distribution.
BOOTSTRAP_REPLICATIONS = HYPERPARAMETERS.volatility_models.bootstrap_replications
BOOTSTRAP_SEED = HYPERPARAMETERS.volatility_models.bootstrap_seed
# Months simulated and dropped before each bootstrap series, so it starts from the model's
# stationary distribution, not arch's starting value.
BURN_IN = HYPERPARAMETERS.volatility_models.burn_in
BOOTSTRAP_NOTE = (
    f"p-values by parametric bootstrap: {BOOTSTRAP_REPLICATIONS} series simulated from the "
    "fitted variance model and re-estimated the same way (break search included); "
    "p = (1 + simulated statistics >= observed) / (replications + 1). The chi-squared tables "
    "don't hold for residuals standardized by a fitted model (Li & Mak 1994)."
)

# Verdict key -> the sentence explaining it in the report. Decided in garch_check().
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

# Legend labels shared by both panels of a figure row (same colour in both).
RESIDUALS_LABEL = "residuals as they are (left: absolute value)"
GARCH_LABEL = "with GARCH (left: its monthly SD; right: residuals divided by it)"

REPORT_PATH = REPORTS_DIR / "garch_diagnostics.txt"
FIGURE_PATH = FIGURES_DIR / "garch_diagnostics.png"


def fit_variance_models(residuals: np.ndarray) -> dict[str, ARCHModelResult]:
    """
    The three plain variance models on zero-mean residuals, rescaled by arch for its optimizer.

    Parameters
    ----------
    residuals : np.ndarray
        SARIMA residuals, shape (n,), in trillions.

    Returns
    -------
    dict of str to ARCHModelResult
        One fit per VARIANCE_MODELS name (constant, GARCH normal, GARCH Student-t). The
        fits are in arch's rescaled units: see original_units_loglikelihood().

    Notes
    -----
    ``mean="Zero"``: the SARIMA mean model already removed the mean (two-step estimation),
    so only the variance is modelled. ``rescale=True``: the residuals are small (hundredths
    of a trillion), which makes arch's optimizer struggle, so arch multiplies them by a
    power of 10 (``fit.scale``) before fitting.
    """
    values = np.asarray(residuals, dtype=float)
    return {
        name: arch_model(values, mean="Zero", rescale=True, **spec).fit(disp="off")
        for name, spec in VARIANCE_MODELS.items()
    }


def original_units_loglikelihood(fit: ARCHModelResult) -> float:
    """
    Log-likelihood of the data arch was given: arch fits c * y, and log f(y) = log f(c y) + log c.

    Parameters
    ----------
    fit : ARCHModelResult
        A fit made with ``rescale=True``; ``fit.scale`` is the factor c arch applied (1 when
        it didn't rescale).

    Returns
    -------
    float
        The log-likelihood of the unscaled data, summed over the fit's nobs observations.

    Notes
    -----
    Why: arch reports the likelihood of c * y. The density of y is c times the density of
    c * y (change of variables), so each observation adds log c. Without this correction,
    models fitted with different c (or on different data, such as the break-adjusted
    residuals) would have AIC / BIC values that can't be compared.
    """
    return float(fit.loglikelihood + fit.nobs * np.log(fit.scale))


def information_criteria(loglikelihood: float, parameters: int, count: int) -> tuple[float, float]:
    """
    (AIC, BIC).

    Parameters
    ----------
    loglikelihood : float
        The model's maximized log-likelihood, in original units.
    parameters : int
        The number of estimated parameters.
    count : int
        The number of observations the likelihood is summed over.

    Returns
    -------
    tuple of (float, float)
        AIC = -2 log L + 2 k, and BIC = -2 log L + k log n. Lower is better.
    """
    return -2 * loglikelihood + 2 * parameters, -2 * loglikelihood + parameters * np.log(count)


def variance_spec(name: str) -> dict[str, Any]:
    """
    arch settings for a variance model, with or without the break-adjustment suffix.

    Parameters
    ----------
    name : str
        A key of VARIANCE_MODELS or ASYMMETRIC_MODELS, optionally followed by BREAK_SUFFIX.

    Returns
    -------
    dict of str to Any
        Keyword arguments for arch_model() ("vol", "dist" and the orders "p", "o", "q").

    Raises
    ------
    KeyError
        If the name, without the suffix, is not a known model.
    """
    return (VARIANCE_MODELS | ASYMMETRIC_MODELS)[name.removesuffix(BREAK_SUFFIX)]


def fit_candidate(residuals: np.ndarray | pd.Series, name: str) -> dict[str, Any]:
    """
    Fit one variance model on zero-mean residuals.

    Its fit, the log-likelihood of the residuals in trillions, its parameter count, and its
    monthly standard deviation in trillions. A break-adjusted model is fitted on each regime
    divided by its own standard deviation.

    Parameters
    ----------
    residuals : np.ndarray or pd.Series
        SARIMA residuals, shape (n,), in trillions.
    name : str
        A model name accepted by variance_spec(); a BREAK_SUFFIX ending selects the
        break-adjusted version.

    Returns
    -------
    dict of str to Any
        - "fit": ARCHModelResult, in arch's rescaled units (and, when break-adjusted, on the
          regime-standardized residuals)
        - "loglikelihood": float, log-likelihood of the original residuals in trillions
        - "parameters": int, the parameter count for AIC / BIC
        - "sd": np.ndarray, shape (n,), the model's monthly standard deviation in trillions
        - "scales": np.ndarray, shape (n,), each month's regime scale s (all 1 for a plain
          model); simulate_residuals() needs it to rebuild residuals in trillions

    Notes
    -----
    Processing, plain model:
    1. Fit arch with a zero mean and rescaling.
    2. Convert the log-likelihood and the conditional volatility back from arch's scale c.

    Processing, break-adjusted model:
    1. Find U5's variance break and each month's regime scale s (volatility_checks).
    2. Fit arch on u = e / s.
    3. Log-likelihood of e: by the change of variables u = e / s, log f(e) = log f(u) -
       log s, so subtract the sum of log s over the months (the Jacobian term). This puts
       the break-adjusted model's likelihood on the same data as the plain ones.
    4. Count BREAK_PARAMETERS (2) extra parameters: the regimes' scale ratio and the break
       date, both estimated from the data.
    5. Monthly SD of e = s times the SD of u.
    """
    values = np.asarray(residuals, dtype=float)
    spec = variance_spec(name)
    if not name.endswith(BREAK_SUFFIX):
        fit = arch_model(values, mean="Zero", rescale=True, **spec).fit(disp="off")
        return {
            "fit": fit,
            "loglikelihood": original_units_loglikelihood(fit),
            "parameters": fit.num_params,
            # arch's volatility is of c * e, so divide by c for trillions.
            "sd": np.asarray(fit.conditional_volatility) / fit.scale,
            "scales": np.ones_like(values),
        }
    # Only break-adjusted names reach this point.
    scales = regime_scales(values, variance_break(values))
    fit = arch_model(values / scales, mean="Zero", rescale=True, **spec).fit(disp="off")
    return {
        "fit": fit,
        # Change of variables u = e / s: log f(e) = log f(u) - log s, summed over the months.
        "loglikelihood": original_units_loglikelihood(fit) - float(np.log(scales).sum()),
        "parameters": fit.num_params + BREAK_PARAMETERS,
        "sd": scales * np.asarray(fit.conditional_volatility) / fit.scale,
        "scales": scales,
    }


def variance_candidates(residuals: np.ndarray | pd.Series) -> dict[str, dict[str, Any]]:
    """
    All four U5b variance models, fitted on the same residuals.

    With log-likelihoods of the same residuals in trillions, their parameter counts, and
    each model's monthly standard deviation in trillions.

    Parameters
    ----------
    residuals : np.ndarray or pd.Series
        SARIMA residuals, shape (n,), in trillions.

    Returns
    -------
    dict of str to dict of str to Any
        Maps CONSTANT, GARCH_NORMAL, GARCH_T and GARCH_T_BREAK to their fit_candidate()
        results. The asymmetric models (U5c) are not included.
    """
    return {name: fit_candidate(residuals, name) for name in [*VARIANCE_MODELS, GARCH_T_BREAK]}


def garch_variance_path(
    residuals: np.ndarray | pd.Series, variant: str, steps: int
) -> tuple[np.ndarray, float]:
    """
    Forecast shock variances for the next months from a named GARCH variant.

    Parameters
    ----------
    residuals : np.ndarray or pd.Series
        The SARIMA residuals up to the forecast origin, shape (n,), in trillions.
    variant : str
        A model name accepted by variance_spec(), e.g. GARCH_T, GARCH_T_BREAK or EGARCH_T.
    steps : int
        The forecast horizon in months.

    Returns
    -------
    tuple of (np.ndarray, float)
        The forecast shock variances for months 1..steps after the origin, shape (steps,),
        in trillions squared; and the Student-t degrees of freedom nu (inf for normal
        errors).

    Notes
    -----
    A named front end to variance_forecast(), for the variants of steps U5b and U5c.
    """
    variances, shape = variance_forecast(
        residuals, variance_spec(variant), variant.endswith(BREAK_SUFFIX), steps
    )
    return variances, float(shape[0]) if len(shape) else np.inf


def variance_forecast(
    residuals: np.ndarray | pd.Series, spec: dict[str, Any], break_adjusted: bool, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forecast shock variances for the next months from any arch variance model.

    Used by the forecast evaluation at every rolling origin, so the variance model is
    refitted on the residuals available at that origin only.

    Parameters
    ----------
    residuals : np.ndarray or pd.Series
        The SARIMA residuals up to the forecast origin, shape (n,), in trillions.
    spec : dict of str to Any
        Keyword arguments for arch_model(): "vol", "p", "o", "q" and "dist".
    break_adjusted : bool
        Whether to fit on the residuals with step U5's variance break removed.
    steps : int
        The forecast horizon in months.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        The forecast shock variances for months 1..steps after the origin, shape (steps,),
        in trillions squared; and the error distribution's shape parameters as arch orders
        them (none for normal, [nu] for Student-t, [eta, lambda] for skewed Student-t).

    Notes
    -----
    Processing:
    1. For a break-adjusted model, divide the residuals by their regime scales and keep
       the latest (post-break) regime's scale.
    2. Fit the variance model with arch (zero mean, rescaled).
    3. Forecast the variance path: analytic for GARCH and GJR; for EGARCH beyond one step,
       by simulation (EGARCH works on log variance, so its multi-step forecast has no closed
       form). The simulation uses SIMULATIONS paths whose standardized shocks are drawn by
       inverting the fitted error distribution at uniforms from a generator seeded with
       SIMULATION_SEED, created afresh at each call, so the same inputs always give the
       same path.
    4. Undo arch's rescaling (divide by c^2) and multiply by the latest regime's scale
       squared, so the forecasts are in trillions squared and assume the post-break regime
       continues.
    """
    values = np.asarray(residuals, dtype=float)
    regime_scale = 1.0
    if break_adjusted:
        scales = regime_scales(values, variance_break(values))
        # scales[-1]: the forecasts are for months after the sample, in its last regime.
        values, regime_scale = values / scales, scales[-1]
    fit = arch_model(values, mean="Zero", rescale=True, **spec).fit(disp="off")
    distribution = fit.model.distribution
    # The distribution's parameters come last in arch's parameter vector.
    shape = fit.params.to_numpy()[len(fit.params) - distribution.num_params :]
    # Extra keyword arguments for fit.forecast(); empty means arch's analytic forecast.
    options: dict[str, Any] = {}
    # One step ahead, EGARCH's variance is known exactly from the fit, so analytic is fine.
    if spec["vol"] == "EGARCH" and steps > 1:
        generator = np.random.default_rng(SIMULATION_SEED)
        options = {
            "method": "simulation",
            "simulations": SIMULATIONS,
            # Seeded shocks from the fitted, unit-variance error distribution (inverse CDF of
            # uniforms); arch ignores random_state here.
            "rng": lambda size: distribution.ppf(generator.uniform(size=size), shape),
        }
    # reindex=False returns only the last origin's row; [-1] takes it as a (steps,) array.
    variances = fit.forecast(horizon=steps, reindex=False, **options).variance.to_numpy()[-1]
    # Back from arch's rescaled units, and onto the latest (post-break) regime's scale.
    return variances / fit.scale**2 * regime_scale**2, shape


def half_life(persistence: float) -> float:
    """
    Months for a volatility shock to halve: persistence ** h = 0.5.

    Parameters
    ----------
    persistence : float
        alpha + beta of a GARCH(1,1): the share of a variance shock still there a month
        later.

    Returns
    -------
    float
        h = log(0.5) / log(persistence) in months; inf when persistence is outside (0, 1),
        where a shock never decays (or the formula is undefined).
    """
    return float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else np.inf


def describe_half_life(months: float) -> str:
    """
    Format a half-life for the report and the figure titles.

    Parameters
    ----------
    months : float
        The output of half_life().

    Returns
    -------
    str
        E.g. "half-life 8.3 months", or "no decay (alpha + beta = 1)" when it is infinite.
    """
    return (
        f"half-life {months:.1f} months" if np.isfinite(months) else "no decay (alpha + beta = 1)"
    )


def bootstrap_p_values(observed: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    """
    Parametric-bootstrap p-values: how often the model's own simulations look at least as
    extreme as the data.

    Parameters
    ----------
    observed : np.ndarray
        The statistics on the data, shape (k,).
    simulated : np.ndarray
        The same statistics on each simulated series, shape (replications, k).

    Returns
    -------
    np.ndarray
        Shape (k,): (1 + number of simulated statistics >= observed) / (replications + 1).

    Notes
    -----
    The + 1 in both places counts the observed series as one draw from the null (Davison &
    Hinkley 1997, eq. 4.11), so a p-value is never 0 and the test keeps its level exactly.
    """
    return (1 + (simulated >= observed).sum(axis=0)) / (len(simulated) + 1)


def check_statistics(standardized: np.ndarray) -> np.ndarray:
    """
    The statistics of the standardized-residual checks, larger meaning more clustering left.

    Parameters
    ----------
    standardized : np.ndarray
        A fitted variance model's standardized residuals, shape (n,).

    Returns
    -------
    np.ndarray
        Shape (PLOT_LAGS + len(ARCH_TEST_LAGS),): the ARCH-LM statistic n * R^2 at lags
        1..PLOT_LAGS (the figure's curve), then the Ljung-Box Q of the squares at
        ARCH_TEST_LAGS.
    """
    arch_lm = [
        het_arch(standardized, nlags=lag, result_object=True).lm for lag in range(1, PLOT_LAGS + 1)
    ]
    ljung_box = acorr_ljungbox(standardized**2, lags=list(ARCH_TEST_LAGS))["lb_stat"]
    return np.concatenate([arch_lm, ljung_box.to_numpy()])


def simulate_residuals(
    candidate: dict[str, Any], name: str, generator: np.random.Generator
) -> np.ndarray:
    """
    One series of residuals simulated from a fitted variance model, in trillions.

    Parameters
    ----------
    candidate : dict of str to Any
        The model's fit_candidate() result.
    name : str
        The same name that fitted it (variance_spec() decides the family and distribution).
    generator : np.random.Generator
        Draws the shocks; the caller seeds it so the bootstrap is reproducible.

    Returns
    -------
    np.ndarray
        Shape (n,), as long as the data the model was fitted on.

    Notes
    -----
    Processing:
    1. Build an arch model of the same family and error distribution, with the shocks drawn
       from `generator` (arch's own random state is global otherwise).
    2. Simulate BURN_IN + n months at the fitted parameters and keep the last n.
    3. Undo arch's rescaling (divide by c) and multiply by the regime scales, so a
       break-adjusted model's series has the same variance break as the data.
    """
    fit = candidate["fit"]
    count = len(candidate["sd"])
    # The zeros are never fitted: simulate() only needs the model's structure.
    model = arch_model(np.zeros(count), mean="Zero", rescale=False, **variance_spec(name))
    model.distribution = type(model.distribution)(seed=generator)
    simulated = model.simulate(fit.params, count, burn=BURN_IN)["data"].to_numpy()
    return candidate["scales"] * simulated / fit.scale


def bootstrap_checks(
    candidate: dict[str, Any],
    name: str,
    replications: int = BOOTSTRAP_REPLICATIONS,
    label: str = "",
) -> dict[str, Any]:
    """
    Bootstrap p-values of the ARCH-LM and Ljung-Box-on-squares checks for one fitted model.

    Parameters
    ----------
    candidate : dict of str to Any
        The model's fit_candidate() result on the data.
    name : str
        The model's name (fit_candidate() and variance_spec() accept it).
    replications : int, default=BOOTSTRAP_REPLICATIONS
        Simulated series; tests use fewer to stay fast.
    label : str, default=""
        Names the model in the progress lines, e.g. "monetary_base_T SARIMA(0,1,1)".

    Returns
    -------
    dict of str to Any
        - "arch_p_curve": np.ndarray, shape (PLOT_LAGS,), ARCH-LM p-values at lags
          1..PLOT_LAGS (the figure)
        - "arch_p", "squared_ljung_box_p": np.ndarray, shape (len(ARCH_TEST_LAGS),), the
          ARCH-LM and Ljung-Box-on-squares p-values at ARCH_TEST_LAGS (the verdict)
        - "failed_fits": int, simulated series whose re-estimation didn't converge (kept:
          dropping them would favour well-behaved samples)

    Notes
    -----
    Processing:
    1. Compute check_statistics() on the model's standardized residuals.
    2. Repeat `replications` times: simulate a series from the fitted model, re-estimate it
       with fit_candidate() (the break search included, for a break-adjusted model), and
       compute the same statistics on its standardized residuals.
    3. Turn the two into p-values with bootstrap_p_values().

    Re-estimating each series reproduces exactly how fitting the model shrinks the
    autocorrelation left in its own standardized residuals, which the chi-squared tables
    ignore. The SARIMA mean model stays fixed: estimating an ARMA mean doesn't change the
    large-sample distribution of squared-residual autocorrelations (McLeod & Li 1983). A new
    generator seeded with BOOTSTRAP_SEED per call makes every model's p-values reproducible,
    whatever order the models are checked in.
    """
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    observed = check_statistics(np.asarray(candidate["fit"].std_resid))
    simulated = np.empty((replications, observed.size))
    failed = 0
    progress = Progress(" | ".join(filter(None, ["BOOTSTRAP", label, name])), replications)
    for row in range(replications):
        refit = fit_candidate(simulate_residuals(candidate, name, generator), name)
        failed += int(refit["fit"].convergence_flag != 0)
        simulated[row] = check_statistics(np.asarray(refit["fit"].std_resid))
        progress.step(row + 1, f"{failed} re-estimations not converged")
    p_values = bootstrap_p_values(observed, simulated)
    return {
        "arch_p_curve": p_values[:PLOT_LAGS],
        # Lag k of the curve sits at position k - 1.
        "arch_p": p_values[[lag - 1 for lag in ARCH_TEST_LAGS]],
        "squared_ljung_box_p": p_values[PLOT_LAGS:],
        "failed_fits": failed,
    }


def garch_check(
    residuals: np.ndarray | pd.Series,
    replications: int = BOOTSTRAP_REPLICATIONS,
    label: str = "",
) -> dict[str, Any]:
    """
    Choose a variance model by BIC and check whether GARCH captured the clustering.

    Parameters
    ----------
    residuals : np.ndarray or pd.Series
        One SARIMA model's residuals, shape (n,), in trillions.
    replications : int, default=BOOTSTRAP_REPLICATIONS
        Bootstrap replications for the checks; 0 skips the checks and the verdict, for
        callers that only need the BIC choice (the forecast evaluation, U5c's reference).
    label : str, default=""
        Names the model in the progress lines, e.g. "monetary_base_T SARIMA(0,1,1)".

    Returns
    -------
    dict of str to Any
        - "information_criteria": dict of model name to (AIC, BIC), all four U5b models
        - "best": str, the model with the lowest BIC
        - "garch_name": str, the GARCH variant examined: "best" itself, or the best GARCH
          by BIC when the constant variance wins
        - "garch_params": pd.Series, that variant's parameters (omega, alpha[1], beta[1],
          and nu for Student-t), in arch's units
        - "persistence": float, alpha + beta
        - "half_life": float, months (half_life())
        - "garch_sd": np.ndarray, shape (n,), that variant's monthly SD in trillions
        - "constant_sd": float, the constant-variance model's SD in trillions
        - "standardized": np.ndarray, shape (n,), arch's standardized residuals of the
          variant (for a break-adjusted one, also divided by the regime scales)

        With replications > 0, also bootstrap_checks()'s keys ("arch_p_curve", "arch_p",
        "squared_ljung_box_p", "failed_fits") and "verdict": str, a key of VERDICTS.

    Notes
    -----
    Processing:
    1. Fit the four variance models and compute AIC / BIC, all in original units.
    2. Pick the lowest BIC; if it is the constant variance, still examine the best GARCH, so
       the report can show what GARCH would have estimated.
    3. Persistence alpha + beta: how much of a variance shock carries into the next month.
       At 1 or above, the variance is not stationary (IGARCH), which often signals an
       unmodelled variance break rather than real persistence.
    4. If GARCH captured the clustering, its standardized residuals have no ARCH effects
       left: test them with ARCH-LM and with Ljung-Box on their squares, with p-values by
       parametric bootstrap (bootstrap_checks()).
    5. Verdict: "CONSTANT VARIANCE" if BIC prefers it; else "GARCH FALLS SHORT" if the
       variance isn't stationary or any check rejects; else "CARRY GARCH FORWARD".
    """
    candidates = variance_candidates(residuals)
    count = len(residuals)
    criteria = {
        name: information_criteria(c["loglikelihood"], c["parameters"], count)
        for name, c in candidates.items()
    }
    # [1] is BIC: BIC chooses, AIC is only reported.
    best = min(criteria, key=lambda name: criteria[name][1])
    garch_names = [name for name in criteria if name != CONSTANT]
    garch_name = best if best != CONSTANT else min(garch_names, key=lambda n: criteria[n][1])
    garch = candidates[garch_name]["fit"]
    persistence = float(garch.params["alpha[1]"] + garch.params["beta[1]"])
    result = {
        "information_criteria": criteria,
        "best": best,
        "garch_name": garch_name,
        "garch_params": garch.params,
        "persistence": persistence,
        "half_life": half_life(persistence),
        "garch_sd": candidates[garch_name]["sd"],
        # A constant variance gives the same SD every month, so the first value is all of it.
        "constant_sd": float(candidates[CONSTANT]["sd"][0]),
        "standardized": np.asarray(garch.std_resid),
    }
    if not replications:
        return result
    checks = bootstrap_checks(candidates[garch_name], garch_name, replications, label)
    clustering_left = (
        np.concatenate([checks["arch_p"], checks["squared_ljung_box_p"]]) < SIGNIFICANCE_LEVEL
    ).any()
    if best == CONSTANT:
        verdict = "CONSTANT VARIANCE"
    elif persistence >= 1 or clustering_left:
        verdict = "GARCH FALLS SHORT"
    else:
        verdict = "CARRY GARCH FORWARD"
    return {**result, **checks, "verdict": verdict}


def garch_report(results: list[dict[str, Any]], train: pd.DataFrame) -> str:
    """
    Write the text report of the GARCH checks.

    Parameters
    ----------
    results : list of dict of str to Any
        One dict per model that passed the residual checks: its check_model() result merged
        with its garch_check() result.
    train : pd.DataFrame
        The training split of the first differences, with DATE_COLUMN.

    Returns
    -------
    str
        The report: a header explaining the method, then per model the AIC / BIC of the
        four variance models (the BIC winner marked), the examined GARCH's alpha, beta, nu,
        persistence and half-life, the standardized-residual checks at each lag, and the
        verdict; then the figure's path.
    """
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
        BOOTSTRAP_NOTE,
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
            # Only Student-t variants have a shape parameter nu.
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
            lines.append(
                f"  Bootstrap: {result['failed_fits']} of {BOOTSTRAP_REPLICATIONS} re-estimations "
                "didn't converge (kept)."
            )
            lines.append(f"  Verdict: {result['verdict']}. {VERDICTS[result['verdict']]}")
            lines.append("")
    lines.append(f"Figure: {repo_path(FIGURE_PATH)}")
    return "\n".join(lines)


def garch_figure(results: list[dict[str, Any]], dates: pd.Series) -> Figure:
    """
    One row per model: residuals against GARCH's SD, and ARCH-LM p-values before and after.

    The left panel shows the absolute residuals against GARCH's and the constant standard
    deviation; the right one the ARCH-LM p-value by lag before and after GARCH.

    Parameters
    ----------
    results : list of dict of str to Any
        As in garch_report(): check_model() results merged with garch_check().
    dates : pd.Series
        The training split's dates, positionally aligned with the first differences.

    Returns
    -------
    Figure
        A figure with len(results) rows and 2 columns and one shared legend on top. The
        caller saves and closes it.

    Notes
    -----
    Processing, per model row:
    1. Left: |residual| and GARCH's monthly SD over time (trillions), with the constant SD
       as a horizontal line; |residual| is the natural one-month estimate of the SD, so the
       GARCH line should track its swings.
    2. Right: ARCH-LM p-values at lags 1..PLOT_LAGS for the residuals (chi-squared tables,
       valid for SARIMA residuals by McLeod & Li 1983) and for the GARCH standardized
       residuals (bootstrap, as in the verdict), on a log scale against the 5% line.

    Then one figure legend above the rows, with room reserved for it in tight_layout.
    """
    apply_chart_style()
    figure, axes = plt.subplots(
        len(results), 2, figsize=(14, 3.3 * len(results)), squeeze=False, facecolor=SURFACE
    )
    lags = np.arange(1, PLOT_LAGS + 1)
    for row, result in enumerate(results):
        residuals = result["residuals"]
        # The split resets the index to 0..n-1, so residual index labels are positions.
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
        # After GARCH: the bootstrap p-values, the ones the verdict uses.
        after = result["arch_p_curve"]
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
        # A third below the smallest p-value; 1e-300 floors it, since p-values can underflow
        # to 0, which a log axis can't show.
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
    # Leave 0.35 inch free at the top for the legend, whatever the figure's height.
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.35 / figure.get_figheight()))
    return figure


@app.command()
def main(
    levels_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv",
    differences_path: Path = PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv",
    candidates_path: Path = CANDIDATES_PATH,
) -> None:
    """
    Run the GARCH checks and write the report and the figure (``make garch``).

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
        passed the residual checks (U4) are examined.
    """
    start_run_log("garch")
    begin_phase("GARCH fits and bootstrap checks (U5b)")
    checked, train = checked_models(levels_path, differences_path, candidates_path)
    # Each result keeps its U4 fields (series, orders, residuals) and gains the U5b ones.
    results = [
        {
            **result,
            **garch_check(
                result["residuals"], label=f"{model_name(result)} {model_label(result)}"
            ),
        }
        for result in checked
        if result["passed"]
    ]

    report = garch_report(results, train)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = garch_figure(results, train[DATE_COLUMN])
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    # Free the figure's memory; it is already saved.
    plt.close(figure)
    logger.success(f"GARCH checks written to {REPORT_PATH} and {FIGURE_PATH}")
    finish_run()


# Running this file directly runs the typer CLI (main).
if __name__ == "__main__":
    app()
