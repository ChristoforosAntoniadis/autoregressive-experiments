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

DATASET_HANDLE = "belbino/fed-balance-sheet-and-m2-money-supply-1950-present"


def download_dataset() -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="kaggle-download-", dir=RAW_DATA_DIR.parent
    ) as temporary_directory:
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


if __name__ == "__main__":
    print(f"Dataset files downloaded to: {download_dataset()}")
