"""
Shared project paths, following the Cookiecutter Data Science layout.

Every other module imports its folder locations (raw, interim and processed
data, models, reports and figures) from here.

Input: the ``DATASET`` environment variable (the Makefile sets it; default
"fed_balance_sheet"), naming the dataset profile: its config file is
``autoregressive_experiments/configs/<DATASET>.json``, and its data, models, reports and logs
live in a ``<DATASET>/`` folder of each of those directories, so several datasets' results
coexist.
Output: the path constants below, and the loguru logger routed through tqdm.
"""

import os
from pathlib import Path

from loguru import logger

DATASET = os.environ.get("DATASET", "fed_balance_sheet")

# Paths
# The package folder (autoregressive_experiments/) sits directly inside the project root, so
# parents[1] of this file is the root on every computer, wherever the project is cloned.
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# data/: raw (as downloaded, committed), interim (cleaned levels), processed (first
# differences) and external (the CCDS slot for third-party data; unused here).
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / DATASET
INTERIM_DATA_DIR = DATA_DIR / "interim" / DATASET
PROCESSED_DATA_DIR = DATA_DIR / "processed" / DATASET
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# models/: every model's tuned settings and each family's champion (tuning).
MODELS_DIR = PROJ_ROOT / "models" / DATASET

# reports/: the committed text reports and CSVs; reports/<dataset>/figures/: every figure.
REPORTS_DIR = PROJ_ROOT / "reports" / DATASET
FIGURES_DIR = REPORTS_DIR / "figures"

# logs/: one timestamped log per run (runlog); not committed, since logs differ per computer.
LOGS_DIR = PROJ_ROOT / "logs" / DATASET
# A new profile's folders do not exist yet; created here so no step fails when it writes.
for folder in (INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

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
