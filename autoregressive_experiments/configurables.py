"""
Definitions of every configurable setting: one frozen dataclass per analysis step, listing the
settings it accepts, their types and the rule each value must satisfy.

The values themselves live in ``configs/hyperparameters.json``, each with its justification;
``config_loader`` reads that file, checks it against these definitions and builds the objects
the modules import. Names, paths, colours, labels, model identities (such as GARCH(1,1) or a VAR
on the differences, which the code's formulas assume) and the evaluation's fixed rules (such as
selecting on one-month MASE) are code, not settings, and stay in their modules.

Input: none.

Output: the dataclasses; nothing is read or written.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

# Keyword arguments ExponentialSmoothing may receive from the config; seasonal_periods is
# added by the code from the seasonal period, so it isn't configured twice.
SMOOTHING_OPTIONS = frozenset({"trend", "damped_trend", "seasonal"})
SMOOTHING_RULE = f"at least one model; options only from {sorted(SMOOTHING_OPTIONS)}"


def setting(check: Callable[[Any], bool], rule: str) -> Any:
    """
    A required dataclass field with a validity check, read by the config loader.

    Parameters
    ----------
    check : callable
        Takes the converted value and returns True when it is acceptable.
    rule : str
        The rule in words, quoted in the loader's error message when `check` fails.

    Returns
    -------
    Any
        A dataclasses.field with no default (every value must come from the config file) and
        the check and rule in its metadata.
    """
    return field(metadata={"check": check, "rule": rule})


def positive(values: tuple[int, ...]) -> bool:
    """
    Whether a tuple of integers is non-empty, every value is at least 1, and none repeats.

    Parameters
    ----------
    values : tuple of int
        Lags, horizons or window lengths, in months.

    Returns
    -------
    bool
        True when the tuple is a non-empty set of distinct positive integers.
    """
    return len(values) > 0 and min(values) >= 1 and len(set(values)) == len(values)


def increasing(values: tuple[int, ...]) -> bool:
    """
    Whether a tuple of integers is non-empty, every value is at least 1, and it strictly
    increases.

    Parameters
    ----------
    values : tuple of int
        Lags or horizons, in months.

    Returns
    -------
    bool
        True when the values are positive and in strictly increasing order.
    """
    return positive(values) and all(a < b for a, b in pairwise(values))


def orders(values: tuple[int, ...]) -> bool:
    """
    Whether a tuple of model orders is non-empty, non-negative, distinct and increasing.

    Parameters
    ----------
    values : tuple of int
        Seasonal AR or MA orders to try.

    Returns
    -------
    bool
        True when the orders are distinct, at least 0 and in increasing order.
    """
    return len(values) > 0 and min(values) >= 0 and all(a < b for a, b in pairwise(values))


def smoothing_models(models: Mapping[str, Mapping[str, Any]]) -> bool:
    """
    Whether every exponential-smoothing model uses only the options the code supports.

    Parameters
    ----------
    models : Mapping of str to Mapping of str to Any
        Model name -> ExponentialSmoothing keyword arguments.

    Returns
    -------
    bool
        True when there is at least one model and each uses only SMOOTHING_OPTIONS keys.
    """
    return len(models) > 0 and all(
        set(options) <= SMOOTHING_OPTIONS for options in models.values()
    )


@dataclass(frozen=True)
class SplitSettings:
    """
    The chronological train / validation / test split (features.split_boundaries).

    Attributes
    ----------
    train_percent : int
        Share of the months in train, in whole percent.
    validation_percent : int
        Share in validation; test takes the rest.
    """

    train_percent: int = setting(lambda v: 0 < v < 100, "between 1 and 99")
    validation_percent: int = setting(lambda v: 0 < v < 100, "between 1 and 99")


@dataclass(frozen=True)
class StatisticalTestSettings:
    """
    Settings shared by every statistical test in the project.

    Attributes
    ----------
    significance_level : float
        The level for every test and confidence band.
    """

    significance_level: float = setting(
        lambda v: v in (0.01, 0.05, 0.10), "0.01, 0.05 or 0.10 (Johansen's critical-value tables)"
    )


@dataclass(frozen=True)
class DiagnosticsSettings:
    """
    Steps U1-U3 (diagnostics.py): stationarity tests, ACF / PACF and the SARIMA grid.

    Attributes
    ----------
    max_acf_lags : int
        Lags shown in the ACF / PACF analysis.
    max_order_sum : int
        Budget p + d + q for the non-seasonal orders.
    differencing_order : int
        d, applied by fitting on the first differences.
    seasonal_ar_orders, seasonal_ma_orders : tuple of int
        Seasonal P and Q values tried.
    seasonal_period : int
        Season length in months.
    max_iterations : int
        The likelihood optimizer's iteration cap per fit.
    adf_autolag : str
        How adfuller picks its lag length.
    kpss_nlags : str
        How kpss picks its bandwidth.
    pacf_method : str
        The PACF estimator.
    """

    max_acf_lags: int = setting(lambda v: v >= 1, "at least 1")
    max_order_sum: int = setting(lambda v: v >= 1, "at least 1")
    differencing_order: int = setting(lambda v: v == 1, "1 (the code fits on first differences)")
    seasonal_ar_orders: tuple[int, ...] = setting(orders, "distinct orders >= 0, increasing")
    seasonal_ma_orders: tuple[int, ...] = setting(orders, "distinct orders >= 0, increasing")
    seasonal_period: int = setting(lambda v: v >= 2, "at least 2")
    max_iterations: int = setting(lambda v: v >= 1, "at least 1")
    adf_autolag: str = setting(lambda v: v in ("AIC", "BIC", "t-stat"), "AIC, BIC or t-stat")
    kpss_nlags: str = setting(lambda v: v in ("auto", "legacy"), "auto or legacy")
    pacf_method: str = setting(
        lambda v: v in ("ywm", "yw", "ols", "burg", "ldb"), "a statsmodels pacf method"
    )


@dataclass(frozen=True)
class ResidualCheckSettings:
    """
    Step U4 (residual_checks.py): Ljung-Box and Breusch-Godfrey on the SARIMA residuals.

    Attributes
    ----------
    test_lags : tuple of int
        Lags at which both tests decide.
    aic_tie_margin : float
        A runner-up within this many AIC points of the best is checked too.
    plot_lags : int
        Lags shown in the figure.
    """

    test_lags: tuple[int, ...] = setting(increasing, "distinct lags >= 1, increasing")
    aic_tie_margin: float = setting(lambda v: v >= 0, "at least 0")
    plot_lags: int = setting(lambda v: v >= 1, "at least 1")


@dataclass(frozen=True)
class VolatilityCheckSettings:
    """
    Step U5 (volatility_checks.py): ARCH-LM and the variance-break search.

    Attributes
    ----------
    arch_test_lags : tuple of int
        Lags at which ARCH-LM decides.
    plot_lags : int
        Lags shown in the figure.
    break_trim : float
        Least share of the months each variance regime keeps.
    """

    arch_test_lags: tuple[int, ...] = setting(increasing, "distinct lags >= 1, increasing")
    plot_lags: int = setting(lambda v: v >= 1, "at least 1")
    break_trim: float = setting(lambda v: 0 < v < 0.5, "between 0 and 0.5")


@dataclass(frozen=True)
class VolatilityModelSettings:
    """
    Steps U5b / U5c (volatility_models.py, asymmetric_garch.py): simulation and bootstrap.

    Attributes
    ----------
    simulations : int
        Paths simulated for EGARCH's multi-step variance forecasts.
    simulation_seed : int
        Seed of those simulations.
    bootstrap_replications : int
        Series simulated per model for the bootstrap p-values.
    bootstrap_seed : int
        Seed of the bootstrap.
    burn_in : int
        Months simulated and dropped before each bootstrap series.
    """

    simulations: int = setting(lambda v: v >= 1, "at least 1")
    simulation_seed: int = setting(lambda v: v >= 0, "at least 0")
    bootstrap_replications: int = setting(lambda v: v >= 1, "at least 1")
    bootstrap_seed: int = setting(lambda v: v >= 0, "at least 0")
    burn_in: int = setting(lambda v: v >= 0, "at least 0")


@dataclass(frozen=True)
class VarDiagnosticsSettings:
    """
    Steps V2-V3 (var_diagnostics.py): VAR lag order and the Johansen test.

    Attributes
    ----------
    max_lags : int
        Highest VAR lag order tried.
    criteria : tuple of str
        Information criteria reported.
    primary_criterion : str
        The criterion whose lag choice drives the Johansen decision.
    johansen_deterministic : int
        coint_johansen's det_order.
    """

    max_lags: int = setting(lambda v: v >= 1, "at least 1")
    criteria: tuple[str, ...] = setting(
        lambda v: len(v) > 0 and set(v) <= {"aic", "bic", "hqic", "fpe"},
        "a non-empty set of aic, bic, hqic, fpe",
    )
    primary_criterion: str = setting(
        lambda v: v in ("aic", "bic", "hqic", "fpe"), "aic, bic, hqic or fpe"
    )
    johansen_deterministic: int = setting(lambda v: v in (-1, 0, 1), "-1, 0 or 1")


@dataclass(frozen=True)
class VarModelSettings:
    """
    Steps V4-V6 (var_models.py): the VAR / VECM candidates and their checks.

    Attributes
    ----------
    var_lags : tuple of int
        Lag orders of the candidates (also the SARIMAX lag candidates).
    test_lags : tuple of int
        Portmanteau lags that decide the verdict.
    plot_lags : int
        Highest lag shown in the figure.
    vecm_rank : int
        Cointegration rank of the VECM candidates.
    unit_root_tolerance : float
        How close to 1 a root's modulus must be to count as a unit root.
    """

    var_lags: tuple[int, ...] = setting(increasing, "distinct lags >= 1, increasing")
    test_lags: tuple[int, ...] = setting(increasing, "distinct lags >= 1, increasing")
    plot_lags: int = setting(lambda v: v >= 1, "at least 1")
    vecm_rank: int = setting(lambda v: v == 1, "1 (two series: rank 1 is the only VECM)")
    unit_root_tolerance: float = setting(lambda v: 0 < v < 0.01, "between 0 and 0.01")


@dataclass(frozen=True)
class VarAnalysisSettings:
    """
    Steps V7-V8 (var_analysis.py): Granger tests, impulse responses and FEVD.

    Attributes
    ----------
    carried_lags : int
        Lag order of the systems carried forward (and evaluated out of sample).
    horizon : int
        Months the impulse responses and variance shares run after the shock.
    report_horizons : tuple of int
        Months reported in the text.
    band_z : float
        Normal quantile for the 95% impulse-response bands.
    """

    carried_lags: int = setting(lambda v: v >= 1, "at least 1")
    horizon: int = setting(lambda v: v >= 1, "at least 1")
    report_horizons: tuple[int, ...] = setting(increasing, "distinct months >= 1, increasing")
    band_z: float = setting(lambda v: v > 0, "positive")


@dataclass(frozen=True)
class ForecastEvaluationSettings:
    """
    The out-of-sample evaluation (forecast_evaluation.py).

    Attributes
    ----------
    horizons : tuple of int
        Months ahead that are scored.
    quantiles : tuple of float
        Forecast quantile levels: the 95% and 80% interval ends and the median.
    moving_average_windows : tuple of int
        Window lengths of the SMA baselines.
    exponential_smoothing : Mapping of str to Mapping of str to Any
        Model name -> ExponentialSmoothing options (seasonal_periods is added by the code);
        read-only, like every setting.
    combination_weight : float
        Weight of the selected SARIMA in the SARIMA + naive combination.
    """

    horizons: tuple[int, ...] = setting(increasing, "distinct months >= 1, increasing")
    quantiles: tuple[float, ...] = setting(
        lambda v: (
            len(v) == 5
            and v[2] == 0.5
            and all(0 < a < b < 1 for a, b in pairwise(v))
            and all(abs(v[i] + v[4 - i] - 1) < 1e-12 for i in range(2))
        ),
        "five increasing levels in (0, 1), symmetric around a 0.5 median",
    )
    moving_average_windows: tuple[int, ...] = setting(
        increasing, "distinct months >= 1, increasing"
    )
    exponential_smoothing: Mapping[str, Mapping[str, Any]] = setting(
        smoothing_models, SMOOTHING_RULE
    )
    combination_weight: float = setting(lambda v: 0 < v < 1, "between 0 and 1")


@dataclass(frozen=True)
class Hyperparameters:
    """
    Every configurable setting of the project, one section per analysis step.

    Attributes
    ----------
    split, tests, diagnostics, residual_checks, volatility_checks, volatility_models,
    var_diagnostics, var_models, var_analysis, forecast_evaluation
        The sections, in pipeline order; see each class.
    """

    split: SplitSettings
    tests: StatisticalTestSettings
    diagnostics: DiagnosticsSettings
    residual_checks: ResidualCheckSettings
    volatility_checks: VolatilityCheckSettings
    volatility_models: VolatilityModelSettings
    var_diagnostics: VarDiagnosticsSettings
    var_models: VarModelSettings
    var_analysis: VarAnalysisSettings
    forecast_evaluation: ForecastEvaluationSettings
