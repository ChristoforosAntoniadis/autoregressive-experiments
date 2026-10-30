"""
Definitions of every configurable setting: one frozen dataclass per analysis step, listing the
settings it accepts, their types and the rule each value must satisfy.

The values themselves live in ``hyperparameters.json`` beside this file, each with its
reasoning; ``config_loader`` reads that file, checks it against these definitions and
builds the objects the modules import. Names, paths, colours, labels, model identities (such as
GARCH(1,1) or a VAR on the differences, which the code's formulas assume) and the evaluation's
fixed rules (such as selecting on validation MASE averaged over the horizons) are code, not
settings, and stay in their modules.

Input: none.

Output: the dataclasses and the config file's path; nothing is read or written.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

# This module's folder; hyperparameters.json sits beside it.
CONFIG_DIR = Path(__file__).resolve().parent
HYPERPARAMETERS_PATH = CONFIG_DIR / "hyperparameters.json"
# The one seed of every random draw in the project: EGARCH's simulated variances, the
# bootstrap, the tuning's draws, the random forest and XGBoost. Fixed so every rerun gives
# identical numbers; the value itself carries no meaning.
RANDOM_SEED = 1337
# The settings each model's search space must name: exactly these keys, each with a list of
# values to try (a single value fixes the setting).
SPACES: dict[str, frozenset[str]] = {
    "sarima": frozenset({"p", "q", "P", "Q", "trend"}),
    "var": frozenset({"lags", "trend"}),
    "vecm": frozenset({"lags", "deterministic"}),
    "sma": frozenset({"window"}),
    "exponential_smoothing": frozenset(
        {"trend", "damped_trend", "seasonal", "use_boxcox", "initialization_method"}
    ),
    "sarimax": frozenset({"lags"}),
    "combination": frozenset({"weight"}),
    "ridge": frozenset({"lags", "alpha"}),
    "lasso": frozenset({"lags", "alpha", "max_iter"}),
    "random_forest": frozenset(
        {
            "lags",
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "bootstrap",
            "max_samples",
            "ccp_alpha",
        }
    ),
    "xgboost": frozenset(
        {
            "lags",
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        }
    ),
    "knn": frozenset({"lags", "n_neighbors", "weights", "p"}),
    "prophet": frozenset(
        {
            "growth",
            "changepoint_prior_scale",
            "changepoint_range",
            "seasonality_mode",
            "seasonality_prior_scale",
            "yearly_seasonality",
        }
    ),
    "lstm": frozenset(
        {
            "lags",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "use_bias",
            "epochs",
            "patience",
            "min_delta",
            "min_epochs",
            "validation_share",
            "hidden_units",
            "layers",
            "dropout",
        }
    ),
    "gru": frozenset(
        {
            "lags",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "use_bias",
            "epochs",
            "patience",
            "min_delta",
            "min_epochs",
            "validation_share",
            "hidden_units",
            "layers",
            "dropout",
        }
    ),
    "tcn": frozenset(
        {
            "lags",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "use_bias",
            "epochs",
            "patience",
            "min_delta",
            "min_epochs",
            "validation_share",
            "hidden_units",
            "layers",
            "kernel_size",
            "dropout",
        }
    ),
    "nbeats": frozenset(
        {
            "lags",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "use_bias",
            "epochs",
            "patience",
            "min_delta",
            "min_epochs",
            "validation_share",
            "hidden_units",
            "layers",
            "blocks",
            "dropout",
        }
    ),
    "dlinear": frozenset(
        {
            "lags",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "use_bias",
            "epochs",
            "patience",
            "min_delta",
            "min_epochs",
            "validation_share",
            "kernel_size",
        }
    ),
    "tensorflow_mlp": frozenset(
        {
            "lags",
            "use_bias",
            "architecture",
            "hidden_units",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "dropout",
            "batch_size",
            "epochs",
            "patience",
            "min_delta",
            "min_epochs",
            "validation_share",
        }
    ),
    "pytorch_mlp": frozenset(
        {
            "lags",
            "use_bias",
            "architecture",
            "hidden_units",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "dropout",
            "batch_size",
            "epochs",
            "patience",
            "min_delta",
            "min_epochs",
            "validation_share",
        }
    ),
    "numpy_a2": frozenset(
        {
            "lags",
            "use_bias",
            "dropout",
            "hidden_units",
            "optimizer",
            "learning_rate",
            "batch_size",
            "epochs",
            "patience",
            "min_delta",
            "min_epochs",
            "validation_share",
        }
    ),
    "numpy_a1": frozenset(
        {
            "lags",
            "use_bias",
            "dropout",
            "hidden_units",
            "optimizer",
            "learning_rate",
            "batch_size",
            "epochs",
            "patience",
            "min_delta",
            "min_epochs",
            "validation_share",
        }
    ),
    "mlp": frozenset(
        {
            "lags",
            "hidden_layer_sizes",
            "activation",
            "solver",
            "alpha",
            "learning_rate_init",
            "max_iter",
            "early_stopping",
        }
    ),
}


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


def search_range(spec: Any) -> bool:
    """
    Whether a setting's search range is usable: {"low": a, "high": b, "log": flag}.

    Parameters
    ----------
    spec : Any
        The setting's entry in a search space, as read from the JSON.

    Returns
    -------
    bool
        True when it has exactly the keys low, high and log, low and high are numbers with
        low < high, log is true or false, and a log scale starts above 0.
    """
    if not isinstance(spec, dict) or set(spec) != {"low", "high", "log"}:
        return False
    low, high, log = spec["low"], spec["high"], spec["log"]
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (low, high))
    return numbers and isinstance(log, bool) and low < high and (not log or low > 0)


def space(name: str) -> Any:
    """
    A required search-space field: an object naming exactly the model's settings, each with a
    non-empty list of distinct values to try, or a numeric range (search_range) for the models
    tuned by Optuna.

    Parameters
    ----------
    name : str
        The model's key in SPACES, e.g. "random_forest".

    Returns
    -------
    Any
        A dataclasses.field (see setting()) whose check accepts a mapping with exactly the
        keys SPACES[name], each mapped to a non-empty list without repeats.
    """
    keys = SPACES[name]

    def check(values: Mapping[str, Any]) -> bool:
        """
        Whether the space names exactly the expected settings with usable value lists.

        Parameters
        ----------
        values : Mapping of str to Any
            Setting name -> the list of values to try, as read from the JSON.

        Returns
        -------
        bool
            True when the keys are exactly `keys` and every value is a non-empty list of
            distinct values or a usable numeric range.
        """
        return set(values) == keys and all(
            search_range(options)
            or (
                isinstance(options, list)
                and len(options) > 0
                and len(set(map(repr, options))) == len(options)
            )
            for options in values.values()
        )

    return setting(
        check,
        f"exactly the settings {sorted(keys)}, each a non-empty list or a "
        '{"low", "high", "log"} range',
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
    bootstrap_replications : int
        Series simulated per model for the bootstrap p-values.
    burn_in : int
        Months simulated and dropped before each bootstrap series.
    """

    simulations: int = setting(lambda v: v >= 1, "at least 1")
    bootstrap_replications: int = setting(lambda v: v >= 1, "at least 1")
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
        Lag orders of the candidates checked in steps V4-V6.
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


@dataclass(frozen=True)
class TuningSettings:
    """
    The tuning on the validation split (tuning.py) and the econometric models' search spaces.

    Attributes
    ----------
    validation_step : int
        Months between consecutive validation origins.
    candidates : int
        Optuna trials per model; spaces of at most this many settings are tried in full.
    sarima, var, vecm, sma, exponential_smoothing, sarimax, combination
        Mapping of str to list: each model's settings and the values tried.
    """

    validation_step: int = setting(lambda v: v >= 1, "at least 1")
    candidates: int = setting(lambda v: v >= 1, "at least 1")
    sarima: Mapping[str, Any] = space("sarima")
    var: Mapping[str, Any] = space("var")
    vecm: Mapping[str, Any] = space("vecm")
    sma: Mapping[str, Any] = space("sma")
    exponential_smoothing: Mapping[str, Any] = space("exponential_smoothing")
    sarimax: Mapping[str, Any] = space("sarimax")
    combination: Mapping[str, Any] = space("combination")


@dataclass(frozen=True)
class MlSettings:
    """
    The lag-feature ML family (ml_forecasting.py): each model's search space, tuned by
    tuning.py.

    Attributes
    ----------
    ridge, lasso, random_forest, xgboost, knn
        Mapping of str to list: the model's settings (scikit-learn / XGBoost keyword names,
        plus "lags") and the values tried.
    """

    ridge: Mapping[str, Any] = space("ridge")
    lasso: Mapping[str, Any] = space("lasso")
    random_forest: Mapping[str, Any] = space("random_forest")
    xgboost: Mapping[str, Any] = space("xgboost")
    knn: Mapping[str, Any] = space("knn")


@dataclass(frozen=True)
class NnSettings:
    """
    The neural-network family (nn_forecasting.py): each network's search space, tuned by
    tuning.py.

    Attributes
    ----------
    mlp
        Mapping of str to list: scikit-learn MLPRegressor's settings (keyword names, plus
        "lags") and the values tried.
    numpy_a1, numpy_a2
        Mapping of str to list: the hand-written NumPy networks' settings (NumpyNetwork's
        keyword names, plus "lags") and the values tried.
    tensorflow_mlp, pytorch_mlp
        Mapping of str to list or range: the Keras and PyTorch networks' settings (their
        keyword names, plus "lags"), searched by Optuna.
    lstm, gru, tcn, nbeats, dlinear
        Mapping of str to list or range: the PyTorch sequence networks' settings (TorchNetwork's
        keyword names, plus "lags", the window length), searched by Optuna.
    """

    mlp: Mapping[str, Any] = space("mlp")
    numpy_a1: Mapping[str, Any] = space("numpy_a1")
    numpy_a2: Mapping[str, Any] = space("numpy_a2")
    tensorflow_mlp: Mapping[str, Any] = space("tensorflow_mlp")
    pytorch_mlp: Mapping[str, Any] = space("pytorch_mlp")
    lstm: Mapping[str, Any] = space("lstm")
    gru: Mapping[str, Any] = space("gru")
    tcn: Mapping[str, Any] = space("tcn")
    nbeats: Mapping[str, Any] = space("nbeats")
    dlinear: Mapping[str, Any] = space("dlinear")


@dataclass(frozen=True)
class ReadyMadeSettings:
    """
    The ready-made model, Prophet (prophet_forecasting.py), part of the econometric family.

    Attributes
    ----------
    prophet
        Mapping of str to list or range: Prophet's settings (its keyword names) and the values
        searched by Optuna.
    """

    prophet: Mapping[str, Any] = space("prophet")


@dataclass(frozen=True)
class Hyperparameters:
    """
    Every configurable setting of the project, one section per analysis step.

    Attributes
    ----------
    split, tests, diagnostics, residual_checks, volatility_checks, volatility_models,
    var_diagnostics, var_models, var_analysis, forecast_evaluation, tuning, ml, nn,
    ready_made
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
    tuning: TuningSettings
    ml: MlSettings
    nn: NnSettings
    ready_made: ReadyMadeSettings
