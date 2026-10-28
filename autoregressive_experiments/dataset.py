"""
Download the Kaggle dataset used by the project.

Input: the Kaggle dataset "Fed Balance Sheet & M2 Money Supply 1950 - Present"
(CC0, aggregated from FRED), reached through ``kagglehub``.

Output: ``fed_balance_sheet.csv`` and ``m2_inflation.csv`` in ``data/raw/``.
Everything already in ``data/raw/`` except ``.gitkeep`` is replaced.
"""

from pathlib import Path
import shutil
import tempfile

import kagglehub

from autoregressive_experiments.config import RAW_DATA_DIR
from autoregressive_experiments.runlog import begin_phase, finish_run, start_run_log

# The dataset's "owner/slug" identifier on Kaggle.
DATASET_HANDLE = "belbino/fed-balance-sheet-and-m2-money-supply-1950-present"


def download_dataset() -> Path:
    """
    Download the Kaggle dataset and replace the contents of ``data/raw/`` with it.

    Returns
    -------
    Path
        RAW_DATA_DIR, the folder now holding the downloaded files.

    Raises
    ------
    RuntimeError
        If the download finishes but leaves no files. ``data/raw/`` is left untouched then.

    Notes
    -----
    Processing:
    1. Create ``data/raw/`` if it doesn't exist.
    2. Download a fresh copy (``force_download``, ignoring kagglehub's cache) into a temporary
       folder next to ``data/raw/``.
    3. Stop with RuntimeError if nothing was downloaded.
    4. Delete everything in ``data/raw/`` except ``.gitkeep``.
    5. Move the downloaded files into ``data/raw/``; the temporary folder is then removed.

    Why a temporary folder: the old files are deleted only after the new ones have arrived,
    so a failed download never leaves ``data/raw/`` empty. It is created inside ``data/``
    rather than the system temp folder so the final move stays on the same drive.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="kaggle-download-", dir=RAW_DATA_DIR.parent
    ) as temporary_directory:
        # kagglehub returns the folder it wrote into as a str.
        downloaded_path = Path(
            kagglehub.dataset_download(
                DATASET_HANDLE,
                force_download=True,
                output_dir=temporary_directory,
            )
        )
        downloaded_items = list(downloaded_path.iterdir())
        if not downloaded_items:
            raise RuntimeError(f"No dataset files were downloaded to {downloaded_path}")

        for existing_path in RAW_DATA_DIR.iterdir():
            # .gitkeep keeps the empty folder in git (CCDS convention), so it survives.
            if existing_path.name == ".gitkeep":
                continue
            if existing_path.is_dir():
                shutil.rmtree(existing_path)
            else:
                existing_path.unlink()

        for downloaded_item in downloaded_items:
            shutil.move(
                str(downloaded_item),
                str(RAW_DATA_DIR / downloaded_item.name),
            )

    return RAW_DATA_DIR


# Running this file (make data) downloads the dataset and prints where it went.
if __name__ == "__main__":
    start_run_log("data")
    begin_phase("Download the dataset")
    print(f"Dataset files downloaded to: {download_dataset()}")
    finish_run()
