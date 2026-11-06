"""
Build the two level series of the dataset profile from its raw file and first-difference them
for modeling.

Input: the profile's raw CSV, ``data/raw/<dataset>/<raw_file>`` (the dataset section of the
config names it).

Output: ``data/interim/<dataset>/levels.csv`` (the cleaned levels: one row per period, ``date``
plus the two value columns; duplicates are reported, not removed) and
``data/processed/<dataset>/first_differences.csv`` (period-over-period changes of both series).

Each profile has a builder (BUILDERS) that turns its raw table into the levels: the Fed balance
sheet is already two level series, so its builder keeps the complete rows; the Daily Treasury
Statement's flows are cumulated into two level series on business days (treasury_levels).

Also provides the chronological 60 / 20 / 20 train / validation / test split that every
analysis and model uses. Its boundaries are dates taken from the cleaned levels, so the
levels and the first differences are always cut at the same periods. And lagged_exog, the lag
matrix of a series' changes that SARIMAX and the lag-feature ML models use.
"""

from collections.abc import Callable
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from autoregressive_experiments.config import (
    DATASET,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from autoregressive_experiments.configs.config_loader import HYPERPARAMETERS
from autoregressive_experiments.runlog import begin_phase, finish_run, start_run_log

app = typer.Typer()

# The date column of the cleaned levels, the same for every dataset profile.
DATE_COLUMN = "date"
# Settings; each value's reasoning is in the dataset's config file (configs/<dataset>.json).
# The two series the project models, their unit and the dataset's name.
VALUE_COLUMNS = list(HYPERPARAMETERS.dataset.value_columns)
UNIT = HYPERPARAMETERS.dataset.unit
UNIT_SUFFIX = HYPERPARAMETERS.dataset.unit_suffix
# Running totals: the test figure shows each forecast as the flow since its origin.
CUMULATIVE = HYPERPARAMETERS.dataset.cumulative
TITLE = HYPERPARAMETERS.dataset.title
RAW_FILE = HYPERPARAMETERS.dataset.raw_file
# Whole percentages rather than fractions, so split_boundaries can use exact integer
# arithmetic. Test takes the rest.
TRAIN_PERCENT = HYPERPARAMETERS.split.train_percent
VALIDATION_PERCENT = HYPERPARAMETERS.split.validation_percent
# The series' spacing: pandas' offset alias, the word for one step in reports, its adjective
# ("monthly change") and how reports print a date.
FREQUENCY = HYPERPARAMETERS.frequency.alias
PERIOD = HYPERPARAMETERS.frequency.period
PERIODLY = {"day": "daily", "week": "weekly", "month": "monthly", "year": "yearly"}.get(
    PERIOD, f"per-{PERIOD}"
)
DATE_FORMAT = HYPERPARAMETERS.frequency.date_format
# The cleaned levels and their first differences, written by make features and read by every
# later step.
LEVELS_PATH = INTERIM_DATA_DIR / "levels.csv"
DIFFERENCES_PATH = PROCESSED_DATA_DIR / "first_differences.csv"
# The Daily Treasury Statement's reduced raw file (dataset.py): its columns, the two transaction
# types as the statement names them (in VALUE_COLUMNS order) and its unit.
FLOW_COLUMNS = ("account", "transaction_type", "amount_millions")
FLOW_TYPES = ("Deposits", "Withdrawals")
MILLIONS_PER_BILLION = 1000.0


def read_dataset(input_path: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Read the profile's raw CSV.

    Parameters
    ----------
    input_path : Path, default=RAW_DATA_DIR
        The folder holding RAW_FILE (a folder, not the file itself).

    Returns
    -------
    pd.DataFrame
        The raw table as read. The Fed balance sheet: one row per month, ``date`` (as text) and
        the two series in trillions USD; 808 rows, 1959-01 to 2026-04, with empty cells where a
        series isn't published (total assets before 2002-12, the monetary base for the last two
        months). The Daily Treasury Statement: one row per business day, account and
        transaction type with the day's amount in millions USD (see treasury_levels).
    """
    logger.info(f"Reading dataset from {input_path}")
    return pd.read_csv(input_path / RAW_FILE)


def treasury_levels(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative deposits and withdrawals of the Treasury's operating cash, in billions USD, one
    row per business day.

    Parameters
    ----------
    raw : pd.DataFrame
        The reduced Daily Treasury Statement: ``date`` and FLOW_COLUMNS, one row per business
        day, account and transaction type (Deposits or Withdrawals), amounts in millions USD.

    Returns
    -------
    pd.DataFrame
        ``date`` (every weekday from the first to the last date) and VALUE_COLUMNS: the running
        totals of deposits and withdrawals since the first day, billions USD.

    Notes
    -----
    Processing:
    1. Sum each day's amounts over the accounts, per transaction type.
    2. Put the flows on every weekday (pandas' "B" days): a weekday without a statement is a
       holiday, when nothing flows, so its flow is zero. Every fifth row is then the same
       weekday, which the seasonal period of 5 and the seasonal naive rely on; skipping holidays
       would shift the weekday rhythm by one position at each.
    3. Cumulate each flow and convert to billions.

    The levels are the running totals rather than the flows themselves, so the pipeline's first
    differences are the daily flows and the error of a level forecast h days ahead is the error
    of the h-day total, the quantity a cash manager plans with. The cumulated series are I(1) by
    construction and positive from the first day, so every model, benchmark and transform of the
    project applies unchanged.
    """
    _account, kind, amount = FLOW_COLUMNS
    # fill_value: a day with deposits but no withdrawal row (or the reverse) had no such flow.
    flows = raw.pivot_table(
        index=DATE_COLUMN, columns=kind, values=amount, aggfunc="sum", fill_value=0.0
    )
    flows.index = pd.to_datetime(flows.index)
    flows = flows.reindex(pd.bdate_range(flows.index.min(), flows.index.max()), fill_value=0.0)
    levels = pd.DataFrame({DATE_COLUMN: flows.index})
    for column, flow_type in zip(VALUE_COLUMNS, FLOW_TYPES):
        levels[column] = flows[flow_type].to_numpy().cumsum() / MILLIONS_PER_BILLION
    return levels


# Dataset profile -> how its raw table becomes the two level series; a profile without an entry
# is two level series already (as the Fed balance sheet), and diagnose_dataset keeps its
# complete rows.
BUILDERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "daily_treasury": treasury_levels,
    # A locally supplied file in the same layout (config.RAW_DATA_DIR).
    "cash_flow": treasury_levels,
}


def diagnose_dataset(name: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Print data-quality checks and keep only the complete rows.

    Parameters
    ----------
    name : str
        A title printed above the checks.
    dataframe : pd.DataFrame
        The raw table, as returned by read_dataset.

    Returns
    -------
    pd.DataFrame
        The rows with no missing cell, renumbered from 0. For the committed data: 279 months,
        2002-12 to 2026-02, with no gaps in between.

    Notes
    -----
    Processing:
    1. Print the column types.
    2. Turn empty and whitespace-only cells into missing values, and print the missing
       counts per column.
    3. Drop every row with any missing cell, renumber the rows, and print the counts again
       (all zero now), the frame's info and its shape.
    4. Print how many rows are duplicates and list them, or say there are none.

    Duplicates are only reported, not removed; the raw data has none, so the cleaned file is
    de-duplicated in practice.
    """
    print(f"\n{name}")
    # Check data types
    print("Dtypes:")
    print(dataframe.dtypes)
    # Treat empty and whitespace-only cells as missing before keeping complete rows.
    df_complete = dataframe.replace(r"^\s*$", pd.NA, regex=True)
    print("\nMissing values per column before dropping incomplete rows:")
    print(df_complete.isna().sum())
    print(f"Total missing values before dropping rows: {df_complete.isna().sum().sum()}")
    df_clean = df_complete.dropna(how="any").reset_index(drop=True)
    print("\nMissing values per column after dropping incomplete rows:")
    print(df_clean.isna().sum())
    print(f"Total missing values after dropping rows: {df_clean.isna().sum().sum()}")
    # Print info
    print("\nDataFrame info:")
    df_clean.info()
    print(f"\nShape after dropping rows: {df_clean.shape}")
    # Check for duplicates
    # keep=False marks every copy, so the listing shows each duplicated row with its twins;
    # the count uses the default keep="first", so it counts only the extra copies.
    duplicate_rows = df_clean[df_clean.duplicated(keep=False)]
    duplicate_count = df_clean.duplicated().sum()
    print(f"\nDuplicate rows (excluding first occurrences): {duplicate_count}")
    if duplicate_rows.empty:
        print("No duplicate rows.")
    else:
        print(duplicate_rows)
    return df_clean


def first_difference(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Month-over-month changes of both series.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The cleaned levels: ``date`` plus VALUE_COLUMNS, in trillions USD, one row per month.
        Other columns are ignored.

    Returns
    -------
    pd.DataFrame
        Columns ``date`` and VALUE_COLUMNS, sorted by date, renumbered from 0. Each value is
        this month's level minus last month's, in trillions USD. The first month has no
        predecessor and is dropped, so there is one row fewer than the input (278 here).

    Raises
    ------
    ValueError
        If ``date`` or a value column is missing, or (from pd.to_numeric) if a value isn't a
        number.

    Notes
    -----
    Processing:
    1. Check the required columns are present.
    2. Sort by date and renumber the rows.
    3. Convert the value columns to numbers and subtract each row's predecessor.
    4. Drop the first row, whose difference is undefined.

    Differencing once is what makes both series stationary: the ADF / KPSS tests show they
    are I(1). The difference is taken between consecutive rows, so it is a one-month change
    only because the cleaned months have no gaps.
    """
    required_columns = {DATE_COLUMN, *VALUE_COLUMNS}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    dataframe = dataframe.sort_values(DATE_COLUMN).reset_index(drop=True)
    differences = dataframe.loc[:, [DATE_COLUMN, *VALUE_COLUMNS]].copy()
    # errors="raise": a stray non-numeric cell should stop the pipeline, not become NaN.
    differences[VALUE_COLUMNS] = (
        dataframe[VALUE_COLUMNS].apply(pd.to_numeric, errors="raise").diff()
    )
    return differences.dropna(subset=VALUE_COLUMNS).reset_index(drop=True)


def split_boundaries(levels: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Last training date and last validation date: the first 60% of months, then the next 20%.

    Parameters
    ----------
    levels : pd.DataFrame
        The cleaned levels, with a ``date`` column (text or datetime), one row per month.

    Returns
    -------
    tuple of (pd.Timestamp, pd.Timestamp)
        (train_end, validation_end). Train is every month up to and including train_end,
        validation the months after it up to and including validation_end, and test the rest.
        For the 279 cleaned months: 2016-10-31 and 2021-06-30, giving 167 / 56 / 56 months.

    Raises
    ------
    ValueError
        If there are too few months for every part to get at least one month.

    Notes
    -----
    Processing:
    1. Parse and sort the dates.
    2. train_count = floor(n * 60 / 100); validation_end = floor(n * 80 / 100), both as
       counts of months from the start.
    3. Check train has at least one month, and validation and test are non-empty.
    4. Return the dates at those positions (count - 1, because positions start at 0).

    The boundaries are dates, not row numbers, so split_by_time can cut the levels and the
    first differences (one row shorter) at the same months.
    """
    dates = pd.to_datetime(levels[DATE_COLUMN]).sort_values().reset_index(drop=True)
    # Integer arithmetic, so a float product such as 5 * 0.6 can't truncate a month away.
    train_count = len(dates) * TRAIN_PERCENT // 100
    validation_end = len(dates) * (TRAIN_PERCENT + VALIDATION_PERCENT) // 100
    # validation_end >= len(dates) would leave test empty.
    if train_count < 1 or validation_end <= train_count or validation_end >= len(dates):
        raise ValueError(f"{len(dates)} {PERIOD}s are too few for a 60 / 20 / 20 split.")
    return dates[train_count - 1], dates[validation_end - 1]


def split_by_time(
    dataframe: pd.DataFrame, boundaries: tuple[pd.Timestamp, pd.Timestamp]
) -> dict[str, pd.DataFrame]:
    """
    Cut a dated frame into train / validation / test at the given boundaries, in date order.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Any frame with a ``date`` column (text or datetime): the levels or the differences.
    boundaries : tuple of (pd.Timestamp, pd.Timestamp)
        (train_end, validation_end), from split_boundaries.

    Returns
    -------
    dict of str to pd.DataFrame
        Keys "train", "validation" and "test", each sorted by date and renumbered from 0,
        with ``date`` parsed to datetime. Together they hold every row exactly once.

    Notes
    -----
    Processing:
    1. Parse ``date`` to datetime (on a copy) and sort by it.
    2. train: dates <= train_end; validation: train_end < date <= validation_end;
       test: dates > validation_end.
    3. Return the three parts, each renumbered.

    Never shuffled: every part lies strictly after the one before it, so a model chosen on
    validation and scored on test has never seen a later month.
    """
    train_end, validation_end = boundaries
    # assign returns a copy, so the caller's frame keeps its original date column.
    dataframe = dataframe.assign(**{DATE_COLUMN: pd.to_datetime(dataframe[DATE_COLUMN])})
    dataframe = dataframe.sort_values(DATE_COLUMN).reset_index(drop=True)
    dates = dataframe[DATE_COLUMN]
    masks = {
        "train": dates <= train_end,
        "validation": (dates > train_end) & (dates <= validation_end),
        "test": dates > validation_end,
    }
    return {name: dataframe[mask].reset_index(drop=True) for name, mask in masks.items()}


def load_levels(path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    The cleaned levels in date order, and their train / validation / test split.

    Parameters
    ----------
    path : Path
        The cleaned levels CSV (LEVELS_PATH).

    Returns
    -------
    tuple of (pd.DataFrame, dict of str to pd.DataFrame)
        The whole table, sorted by date and renumbered from 0, and split_by_time's parts.
    """
    data = pd.read_csv(path, parse_dates=[DATE_COLUMN]).sort_values(DATE_COLUMN)
    data = data.reset_index(drop=True)
    return data, split_by_time(data, split_boundaries(data))


def load_training_split(
    levels_path: Path, differences_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The training split of the cleaned levels and of their first differences.

    Parameters
    ----------
    levels_path : Path
        The cleaned levels CSV (``data/interim/``).
    differences_path : Path
        The first differences CSV (``data/processed/``).

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        The training rows of each, with ``date`` parsed; cut at the same months, since the
        boundaries come from the levels.

    Raises
    ------
    FileNotFoundError
        If either file is missing (run make features).
    """
    for path in (levels_path, differences_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Generate it with make features.")
    levels = pd.read_csv(levels_path, parse_dates=[DATE_COLUMN])
    differences = pd.read_csv(differences_path, parse_dates=[DATE_COLUMN])
    boundaries = split_boundaries(levels)
    return split_by_time(levels, boundaries)["train"], split_by_time(differences, boundaries)[
        "train"
    ]


def series_title(column: str) -> str:
    """
    Readable series name for chart titles.

    Parameters
    ----------
    column : str
        Column name, e.g. "fed_total_assets_T".

    Returns
    -------
    str
        The name without the unit suffix (UNIT_SUFFIX, e.g. "_T") and with spaces, e.g. "fed
        total assets".
    """
    return column.removesuffix(UNIT_SUFFIX).replace("_", " ")


def lagged_exog(changes: np.ndarray, lags: int) -> np.ndarray:
    """
    The lag matrix of a series of monthly changes: row i holds the `lags` changes before month
    i + lags, most recent first.

    Parameters
    ----------
    changes : np.ndarray
        Monthly changes, shape (n,), trillions, oldest first.
    lags : int
        How many past changes each row holds.

    Returns
    -------
    np.ndarray
        Shape (n - lags, lags): row i is changes[i + lags - 1], changes[i + lags - 2], ...,
        changes[i], the regressors for month i + lags.

    Notes
    -----
    Months without a full set of earlier changes (the first `lags`) get no row, rather than
    zeros standing in for unknown values; the SARIMAX fit drops the same first months of its
    own series. A row never holds its own month's change, so a forecast never needs the other
    series' value for the month being forecast (that would be leakage).
    """
    return np.column_stack(
        [changes[lags - lag : len(changes) - lag] for lag in range(1, lags + 1)]
    )


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    cleaned_path: Path = LEVELS_PATH,
    output_path: Path = DIFFERENCES_PATH,
) -> None:
    """
    Build the levels from the raw data and write them and their first differences (make
    features).

    Parameters
    ----------
    input_path : Path, default=RAW_DATA_DIR
        Folder holding the raw RAW_FILE.
    cleaned_path : Path, default=LEVELS_PATH
        Where to write the complete rows (the levels).
    output_path : Path, default=DIFFERENCES_PATH
        Where to write the period-over-period differences.

    Returns
    -------
    None
        Writes the two CSVs and prints the data-quality checks plus the first and last rows.

    Notes
    -----
    Processing:
    1. Read the raw CSV, build the levels with the profile's BUILDERS entry and keep the
       complete rows (diagnose_dataset), printing the checks.
    2. Print the first and last five rows, and write the cleaned levels.
    3. Read the cleaned file back with parsed dates, first-difference it and write the result
       with ISO dates (YYYY-MM-DD).

    Reading the cleaned file back, rather than reusing the frame in memory, gives the dates a
    real datetime type (the raw ones are text), and the differences are computed from exactly
    what the later steps read from disk.
    """
    start_run_log("features")
    begin_phase("Clean the data and take first differences")
    build = BUILDERS.get(DATASET, lambda raw: raw)
    df_clean = diagnose_dataset(TITLE, build(read_dataset(input_path)))
    print("\nFirst few rows of the dataset:")
    print(df_clean.head())
    print("\nLast few rows of the dataset:")
    print(df_clean.tail())
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(cleaned_path, index=False)
    logger.success(f"Cleaned dataset saved to {cleaned_path}")

    differences = first_difference(pd.read_csv(cleaned_path, parse_dates=[DATE_COLUMN]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    differences.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    logger.success(f"First-difference dataset saved to {output_path}")

    finish_run()


if __name__ == "__main__":
    app()
