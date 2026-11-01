"""
A smoke test of the forecast pipeline: the outlier report, the tuning and the test of the
champions, run end to end on a small synthetic file with one trial per model, so a model path
that breaks (a fit that cannot converge, a forecast that overflows, a report column that is
missing) fails here in a minute instead of hours into a full run.
"""

from collections.abc import Iterator
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from autoregressive_experiments import forecast_evaluation, outliers, tuning
from autoregressive_experiments.config import LOGS_DIR
from autoregressive_experiments.features import DATE_COLUMN, VALUE_COLUMNS
from autoregressive_experiments.model_forecasts import BENCHMARKS, FAMILIES


@pytest.fixture
def work_dir() -> Iterator[Path]:
    """
    A scratch folder inside the project, removed afterwards.

    Yields
    ------
    Path
        ``logs/pytest_smoke/``: the reports print paths relative to the project root, so the
        outputs must live inside it; ``logs/`` is not committed. The run logs themselves go to
        ``logs/`` as in any run (the log file stays open, so it could not be removed here).
    """
    folder = LOGS_DIR / "pytest_smoke"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


def synthetic_levels(months: int, path: Path) -> Path:
    """
    Write a small cleaned-levels file: two positive random walks with a drift and one jump.

    Parameters
    ----------
    months : int
        Number of month-end rows.
    path : Path
        Where to write the CSV.

    Returns
    -------
    Path
        `path`.
    """
    rng = np.random.default_rng(0)
    steps = rng.normal(0.02, 0.05, size=(months, 2))
    steps[months // 3] += 1.0
    levels = 2.0 + np.cumsum(steps, axis=0)
    table = pd.DataFrame(levels, columns=VALUE_COLUMNS)
    table.insert(0, DATE_COLUMN, pd.date_range("2000-01-31", periods=months, freq="ME"))
    table.to_csv(path, index=False)
    return path


@pytest.mark.slow
def test_tuning_and_evaluation_run_end_to_end_on_synthetic_data(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With one Optuna trial per model, every model is fitted and scored on validation, one
    champion per family is chosen, and the champions, their mean, naive and drift are scored on
    test with bands and the sensitivity table; every output file exists and the test metrics
    hold every model at every horizon."""
    tmp_path = work_dir
    levels = synthetic_levels(72, tmp_path / "levels.csv")
    monkeypatch.setattr(tuning, "CANDIDATES", 1)
    monkeypatch.setattr(tuning, "BOOTSTRAP_SAMPLES", 20)
    for module, names in (
        (outliers, ("REPORT_PATH", "FIGURE_PATH")),
        (
            tuning,
            (
                "TUNED_PATH",
                "CHAMPIONS_PATH",
                "REPORT_PATH",
                "VALIDATION_METRICS_PATH",
                "SCORES_PATH",
                "FIGURE_PATH",
            ),
        ),
        (
            forecast_evaluation,
            (
                "METRICS_PATH",
                "INTERVALS_PATH",
                "FORECASTS_PATH",
                "CHAMPIONS_TABLE_PATH",
                "REPORT_PATH",
                "SENSITIVITY_PATH",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, tmp_path / getattr(module, name).name)
    # forecast_evaluation reads the tuning's files under its own names, and builds the figure
    # paths from the figures folder.
    monkeypatch.setattr(forecast_evaluation, "CHAMPIONS_PATH", tuning.CHAMPIONS_PATH)
    monkeypatch.setattr(forecast_evaluation, "TUNED_PATH", tuning.TUNED_PATH)
    monkeypatch.setattr(forecast_evaluation, "FIGURES_DIR", tmp_path)

    outliers.main(levels)
    tuning.main(levels)
    forecast_evaluation.main(levels)

    champions = pd.read_csv(forecast_evaluation.CHAMPIONS_TABLE_PATH)
    metrics = pd.read_csv(forecast_evaluation.METRICS_PATH)
    assert sorted(champions["family"].unique()) == sorted(FAMILIES)
    assert len(champions) == len(FAMILIES) * len(VALUE_COLUMNS)
    models_per_series = len(FAMILIES) + 1 + len(BENCHMARKS)
    assert len(metrics) == models_per_series * len(VALUE_COLUMNS) * len(tuning.HORIZONS)
    assert metrics["MAE"].notna().all() and metrics["MAE_vs_naive"].notna().all()
    assert forecast_evaluation.SENSITIVITY_PATH.exists()
    assert (tmp_path / "tuning.png").exists() and (tmp_path / "outliers.png").exists()
    assert forecast_evaluation.figure_path(VALUE_COLUMNS[0]).exists()
