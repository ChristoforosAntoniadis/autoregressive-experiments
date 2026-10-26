"""
Training entry point from the Cookiecutter Data Science template (a placeholder).

This is the template's untouched stub: it reads nothing, trains nothing and writes nothing. The
project's pipeline doesn't use it; every model is fitted and evaluated by the analysis modules
run by the make targets (``forecast_evaluation`` above all).

Input: none in practice (the default paths below are the template's examples).
Output: log messages only.
"""

from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from autoregressive_experiments.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
) -> None:
    """
    Pretend to train a model: the Cookiecutter template's placeholder.

    Parameters
    ----------
    features_path : Path, default=PROCESSED_DATA_DIR / "features.csv"
        Where a real implementation would read the feature matrix. Unused; the file doesn't
        exist in this project.
    labels_path : Path, default=PROCESSED_DATA_DIR / "labels.csv"
        Where a real implementation would read the targets. Unused; the file doesn't exist.
    model_path : Path, default=MODELS_DIR / "model.pkl"
        Where a real implementation would save the fitted model. Unused.

    Returns
    -------
    None
        Only logs a start message, a message at iteration 5 and a success message.

    Notes
    -----
    Kept as the template left it, so the CCDS layout stays complete. It isn't part of the
    project's pipeline and no make target runs it.
    """
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    # A dummy loop that only shows the tqdm progress bar working with the logger.
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
