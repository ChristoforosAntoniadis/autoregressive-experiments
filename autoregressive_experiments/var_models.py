"""
VAR / VECM estimation and checks, steps V4 to V6: fit the candidate systems, check they're
stable, and check their residuals carry no autocorrelation left.

Candidates (the user's decisions): VARs on the first differences, with a constant, and a rank-1
VECM on the levels with a constant, even though Johansen found no cointegration (step V3), so
forecast accuracy can decide later whether a long-run link helps. Lag orders 1 (BIC) and 3 (AIC,
HQIC) come from step V2; both leave autocorrelated residuals, so by the roadmap's rule the order
was raised to 5, the smallest that passes. Only candidates that are stable and pass the residual
checks are carried forward to the evaluation: VAR(5) and VECM(5).

- V4: estimate each candidate on the training split (OLS per equation for the VARs, Johansen's
  maximum likelihood for the VECMs).
- V5: stability. A VAR on the differences is stable when every inverse root of its characteristic
  polynomial lies inside the unit circle (moduli below 1). A rank-1 VECM on two I(1) series has,
  by construction, exactly one unit root (one shared stochastic trend); it's stable when that is
  the only root on the circle and every other one lies inside.
- V6: multivariate Portmanteau test (adjusted) on the system residuals at lags 6, 12 and 24;
  no autocorrelation left means the lag order captured the dynamics.

Input: ``data/interim/fed_balance_sheet_cleaned.csv`` and
``data/processed/fed_balance_sheet_first_difference.csv``, training split only.

Output, in ``reports/``:

- ``var_models.txt``: each candidate's coefficients (with p-values), its root moduli, its
  Portmanteau p-values, and a verdict
- ``figures/var_models.png``: per candidate, its roots against the unit circle, and the
  Portmanteau p-value by lag against the 5% line
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
import typer

from autoregressive_experiments.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from autoregressive_experiments.diagnostics import SIGNIFICANCE_LEVEL, VALUE_COLUMNS, repo_path
from autoregressive_experiments.features import DATE_COLUMN, split_boundaries, split_by_time
from autoregressive_experiments.residual_checks import (
    BASELINE,
    CRITICAL_COLOR,
    SECONDARY_INK,
    SERIES_COLOR,
    SURFACE,
    apply_chart_style,
)

app = typer.Typer()

# 1 (BIC) and 3 (AIC, HQIC) from step V2, and 5: the smallest order whose residuals pass V6,
# added by the roadmap's rule (autocorrelated residuals: raise the lag order).
VAR_LAGS = (1, 3, 5)
TEST_LAGS = (6, 12, 24)
PLOT_LAGS = 24
VECM_RANK = 1
# A unit root estimated from data lands within a hair of 1, never exactly on it.
UNIT_ROOT_TOLERANCE = 1e-6

REPORT_PATH = REPORTS_DIR / "var_models.txt"
FIGURE_PATH = FIGURES_DIR / "var_models.png"


def companion_roots(coefficients: np.ndarray) -> np.ndarray:
    """Eigenvalues of the companion matrix of a VAR with coefficient matrices A_1..A_p
    (shape p x k x k), largest modulus first. These are the inverse roots of the
    characteristic polynomial: all inside the unit circle means stable."""
    lags, size, _ = coefficients.shape
    companion = np.zeros((lags * size, lags * size))
    companion[:size, :] = np.hstack(list(coefficients))
    companion[size:, :-size] = np.eye((lags - 1) * size)
    roots = np.linalg.eigvals(companion)
    return roots[np.argsort(-np.abs(roots))]


def is_stable(moduli: np.ndarray, unit_roots: int) -> bool:
    """Exactly `unit_roots` moduli at 1 (within tolerance), every other one inside the circle."""
    on_circle = np.abs(moduli - 1) < UNIT_ROOT_TOLERANCE
    return bool(on_circle.sum() == unit_roots and (moduli[~on_circle] < 1).all())


def fit_candidates(levels: pd.DataFrame, differences: pd.DataFrame) -> dict[str, dict]:
    """Every candidate fitted on the given (training) data, with what V5 and V6 need."""
    candidates = {}
    for lags in VAR_LAGS:
        fitted = VAR(differences).fit(lags, trend="c")
        candidates[f"VAR({lags})"] = {
            "kind": "VAR",
            "lags": lags,
            "fitted": fitted,
            "roots": companion_roots(fitted.coefs),
            "unit_roots": 0,
            # Portmanteau lags must exceed the lags in the model (levels lags for the VECM).
            "model_lags": lags,
        }
    for lags in VAR_LAGS:
        fitted = VECM(levels, k_ar_diff=lags, coint_rank=VECM_RANK, deterministic="co").fit()
        candidates[f"VECM({lags})"] = {
            "kind": "VECM",
            "lags": lags,
            "fitted": fitted,
            "roots": companion_roots(np.asarray(fitted.var_rep)),
            "unit_roots": levels.shape[1] - VECM_RANK,
            "model_lags": lags + 1,
        }
    return candidates


def check_candidate(candidate: dict) -> dict:
    fitted = candidate["fitted"]
    portmanteau = {
        lag: fitted.test_whiteness(nlags=lag, adjusted=True).pvalue
        for lag in range(candidate["model_lags"] + 1, PLOT_LAGS + 1)
    }
    stable = is_stable(np.abs(candidate["roots"]), candidate["unit_roots"])
    # The test needs more lags than the model has, so short test lags may not exist (n/a).
    white = all(portmanteau[lag] >= SIGNIFICANCE_LEVEL for lag in TEST_LAGS if lag in portmanteau)
    return {
        **candidate,
        "portmanteau": portmanteau,
        "stable": stable,
        "white": white,
        "verdict": "CARRY FORWARD" if stable and white else "FAIL",
    }


def coefficient_lines(candidate: dict) -> list[str]:
    fitted = candidate["fitted"]
    if candidate["kind"] == "VAR":
        table = pd.concat(
            {"estimate": fitted.params, "p-value": fitted.pvalues}, axis=1
        ).swaplevel(axis=1)
        return ["    " + line for line in table.round(4).to_string().splitlines()]
    beta = fitted.beta.ravel() / fitted.beta.ravel()[0]
    lines = [
        f"    Long-run relation (beta): {VALUE_COLUMNS[0]} {beta[1]:+.4f} x {VALUE_COLUMNS[1]}",
        "    Adjustment towards it (alpha), per equation:",
    ]
    for column, alpha, p_value in zip(
        VALUE_COLUMNS, fitted.alpha.ravel(), fitted.pvalues_alpha.ravel()
    ):
        lines.append(f"      {column:<22} {alpha:+.4f} (p={p_value:.3f})")
    lines.append(
        "    Short-run terms (gamma) and constants are in the fitted model; the forecasts use them."
    )
    return lines


def models_report(results: dict[str, dict], period: str) -> str:
    lines = [
        "VAR / VECM estimation and checks, steps V4 (estimate), V5 (stability), V6 (residuals)",
        f"Training split only: {period}. Series: {', '.join(VALUE_COLUMNS)}.",
        (
            "VAR(p): first differences, constant, p lags, OLS per equation. VECM(p): levels, "
            f"rank {VECM_RANK}, p lagged differences, constant, maximum likelihood."
        ),
        (
            "Stability: inverse-root moduli of the implied VAR. A VAR on the differences needs "
            f"all below 1; a rank-{VECM_RANK} VECM on 2 series needs exactly one at 1 (its shared "
            "stochastic trend) and the rest below 1."
        ),
        (
            "Residuals: adjusted multivariate Portmanteau test (null: no autocorrelation up to "
            f"the lag) at lags {', '.join(map(str, TEST_LAGS))}; passes when every p-value is at "
            f"or above {SIGNIFICANCE_LEVEL:.0%}."
        ),
        "",
    ]
    for name, result in results.items():
        lines.append(f"{name}  [{result['verdict']}]")
        lines.append("  V4 coefficients:")
        lines.extend(coefficient_lines(result))
        moduli = ", ".join(f"{value:.4f}" for value in np.abs(result["roots"]))
        lines.append(
            f"  V5 inverse-root moduli: {moduli} -> {'stable' if result['stable'] else 'NOT stable'}"
        )
        portmanteau = ", ".join(
            f"lag {lag} p={result['portmanteau'][lag]:.4f}"
            if lag in result["portmanteau"]
            else f"lag {lag} n/a (the model has {result['model_lags']} lags)"
            for lag in TEST_LAGS
        )
        lines.append(
            f"  V6 Portmanteau: {portmanteau} -> "
            f"{'no autocorrelation left' if result['white'] else 'autocorrelation remains'}"
        )
        lines.append("")
    carried = [name for name, result in results.items() if result["verdict"] == "CARRY FORWARD"]
    lines.append(f"Carry forward to the evaluation: {', '.join(carried) if carried else 'none'}")
    lines.append(f"Figure: {repo_path(FIGURE_PATH)}")
    return "\n".join(lines)


def models_figure(results: dict[str, dict]) -> plt.Figure:
    """Per candidate: its roots against the unit circle, and the Portmanteau p-value by lag."""
    apply_chart_style()
    figure, axes = plt.subplots(
        len(results),
        2,
        figsize=(12, 3.6 * len(results)),
        squeeze=False,
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": (1, 2.2)},
    )
    circle = np.linspace(0, 2 * np.pi, 200)
    for row, (name, result) in enumerate(results.items()):
        root_axis, p_axis = axes[row]
        for axis in axes[row]:
            axis.set_facecolor(SURFACE)
        root_axis.plot(np.cos(circle), np.sin(circle), color=BASELINE, linewidth=1)
        roots = result["roots"]
        on_circle = np.abs(np.abs(roots) - 1) < UNIT_ROOT_TOLERANCE
        groups = (
            (~on_circle, {"color": SERIES_COLOR, "s": 40}, "inside: shocks die out"),
            (
                on_circle,
                {"facecolor": SURFACE, "edgecolors": SECONDARY_INK, "s": 60, "linewidths": 2},
                "unit root (the VECM's shared trend)",
            ),
        )
        for mask, style, label in groups:
            if mask.any():
                root_axis.scatter(
                    roots[mask].real, roots[mask].imag, zorder=3, label=label, **style
                )
        root_axis.set_xlim(-1.2, 1.2)
        root_axis.set_ylim(-1.2, 1.2)
        root_axis.set_aspect("equal")
        root_axis.set_xlabel("Real part")
        root_axis.set_ylabel("Imaginary part")
        root_axis.set_title(
            f"{name}  [{result['verdict']}]\nroots vs the unit circle "
            f"({'stable' if result['stable'] else 'NOT stable'})",
            loc="left",
            fontsize=10,
        )
        root_axis.legend(loc="lower left", fontsize=8, frameon=False)

        lags = np.array(list(result["portmanteau"]))
        p_values = np.array(list(result["portmanteau"].values()))
        failing = p_values < SIGNIFICANCE_LEVEL
        p_axis.scatter(lags[~failing], p_values[~failing], color=SERIES_COLOR, s=24, zorder=3)
        p_axis.scatter(lags[failing], p_values[failing], color=CRITICAL_COLOR, s=24, zorder=3)
        p_axis.axhline(SIGNIFICANCE_LEVEL, color=CRITICAL_COLOR, linewidth=1)
        p_axis.annotate(
            " 5%",
            (1, SIGNIFICANCE_LEVEL),
            xycoords=("axes fraction", "data"),
            va="center",
            color=SECONDARY_INK,
            fontsize=9,
        )
        p_axis.set_yscale("log")
        p_axis.set_ylim(min(1e-4, p_values.min() / 3), 1.5)
        p_axis.set_xlim(0.5, PLOT_LAGS + 0.5)
        p_axis.set_xticks([1, 6, 12, 18, 24])
        p_axis.set_xlabel("Lag (months)")
        p_axis.set_title(
            "Portmanteau p-value by lag (below the line: autocorrelation remains; red)",
            loc="left",
            fontsize=10,
        )
    figure.tight_layout()
    return figure


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
    period = (
        f"levels {train_levels[DATE_COLUMN].min():%Y-%m} to {train_levels[DATE_COLUMN].max():%Y-%m}"
        f" ({len(train_levels)} months)"
    )

    # Month-end frequency on the index: statsmodels then knows the spacing, and a missing month
    # would surface as NaN rows that the fit rejects, instead of passing silently.
    candidates = fit_candidates(
        train_levels.set_index(DATE_COLUMN)[list(VALUE_COLUMNS)].asfreq("ME"),
        train_differences.set_index(DATE_COLUMN)[list(VALUE_COLUMNS)].asfreq("ME"),
    )
    results = {name: check_candidate(candidate) for name, candidate in candidates.items()}

    report = models_report(results, period)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)

    figure = models_figure(results)
    figure.savefig(FIGURE_PATH, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    logger.success(f"VAR / VECM checks written to {REPORT_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    app()
