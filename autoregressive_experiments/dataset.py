"""
Download the Kaggle dataset of the dataset profile.

Input: the Kaggle dataset the profile's config names (dataset.kaggle_handle), reached through
``kagglehub``: "Fed Balance Sheet & M2 Money Supply 1950 - Present" (CC0, aggregated from
FRED) or "Daily cash and debt operations of U.S. Treasury" (the Daily Treasury Statement,
2005-10 to 2023-04, US government data).

Output: the downloaded files in ``data/raw/<dataset>/``; everything already there except
``.gitkeep`` is replaced. The Treasury download (100 MB of statement tables) is reduced to the
one table the project uses, the daily deposits and withdrawals of the main account summed over
their categories (REDUCERS), so the committed raw file stays small.
"""

from collections.abc import Callable
from pathlib import Path
import shutil
import tempfile

import kagglehub
import pandas as pd

from autoregressive_experiments.config import DATASET, RAW_DATA_DIR
from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.features import DATE_COLUMN, FLOW_COLUMNS, RAW_FILE
from autoregressive_experiments.runlog import begin_phase, finish_run, start_run_log

# The dataset's "owner/slug" identifier on Kaggle.
DATASET_HANDLE = HYPERPARAMETERS.dataset.kaggle_handle
# The Daily Treasury Statement's deposits and withdrawals table and the names the statement
# gives the Treasury's main account (renamed on 2021-10-01), whose rows are kept; its
# sub-total rows are dropped, so the categories sum to the published daily totals.
TREASURY_TABLE = "DTS_OpCashDpstWdrl"
TREASURY_ACCOUNTS = ("Federal Reserve Account", "Treasury General Account (TGA)")
TREASURY_COLUMNS = {
    "Record Date": DATE_COLUMN,
    "Type of Account": FLOW_COLUMNS[0],
    "Transaction Type": FLOW_COLUMNS[1],
    "Transactions Today": FLOW_COLUMNS[2],
}


def reduce_treasury(folder: Path) -> None:
    """
    Replace the downloaded Daily Treasury Statement tables by the one reduced file the project
    reads.

    Parameters
    ----------
    folder : Path
        The download folder; its ``DTS_OpCashDpstWdrl_*.csv`` is read and every file in it is
        replaced by RAW_FILE.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the deposits and withdrawals table is not among the downloaded files.

    Notes
    -----
    Keeps the main account's rows (both names it had), drops the sub-total rows, and sums the
    amounts per day, account and transaction type. The result has about 9,000 rows instead of
    326,000.
    """
    tables = sorted(folder.glob(f"{TREASURY_TABLE}_*.csv"))
    if not tables:
        raise FileNotFoundError(f"no {TREASURY_TABLE} table among the files in {folder}")
    table = pd.read_csv(tables[0], na_values=["null"], low_memory=False)
    table = table[table["Type of Account"].isin(TREASURY_ACCOUNTS)]
    table = table[~table["Transaction Category"].fillna("").str.contains("Total", case=False)]
    table = table.rename(columns=TREASURY_COLUMNS)[list(TREASURY_COLUMNS.values())]
    reduced = table.groupby(list(TREASURY_COLUMNS.values())[:3], as_index=False).sum()
    # Files only: kagglehub's .complete marker folder stays and is git-ignored.
    for path in folder.iterdir():
        if path.is_file():
            path.unlink()
    reduced.to_csv(folder / RAW_FILE, index=False)


# Dataset profile -> how its download is reduced before it is kept; profiles without an entry
# keep every downloaded file.
REDUCERS: dict[str, Callable[[Path], None]] = {"daily_treasury": reduce_treasury}


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
    1. Create the raw folder if it doesn't exist.
    2. Download a fresh copy (``force_download``, ignoring kagglehub's cache) into a temporary
       folder next to it, and reduce it when the profile has a REDUCERS entry.
    3. Stop with RuntimeError if nothing was downloaded.
    4. Delete everything in the raw folder except ``.gitkeep``.
    5. Move the downloaded files into it; the temporary folder is then removed.

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
        if not any(downloaded_path.iterdir()):
            raise RuntimeError(f"No dataset files were downloaded to {downloaded_path}")
        if DATASET in REDUCERS:
            REDUCERS[DATASET](downloaded_path)
        downloaded_items = list(downloaded_path.iterdir())

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
