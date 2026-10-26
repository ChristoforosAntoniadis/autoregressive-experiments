"""
Shared project paths, following the Cookiecutter Data Science layout.

Every other module imports its folder locations (raw, interim and processed
data, models, reports and figures) from here.

Input: none; the paths are computed from where this file lives.
Output: the path constants below, and the loguru logger routed through tqdm.
"""

from pathlib import Path

from loguru import logger

# Paths
# The package folder (autoregressive_experiments/) sits directly inside the project root, so
# parents[1] of this file is the root on every computer, wherever the project is cloned.
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# data/: raw (as downloaded, committed), interim (cleaned levels), processed (first
# differences) and external (the CCDS slot for third-party data; unused here).
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# models/: the CCDS slot for saved models; only the template stub in modeling/train.py uses it.
MODELS_DIR = PROJ_ROOT / "models"

# reports/: the committed text reports and CSVs; reports/figures/: every generated figure.
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Route log messages through tqdm.write, so they print above an active progress bar instead of
# breaking it. Handler 0 is loguru's default stderr handler, replaced here. Without tqdm
# installed, loguru's default handler is kept.
try:
    from tqdm import tqdm

    logger.remove(0)
    # end="": loguru's message already ends in a newline.
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
