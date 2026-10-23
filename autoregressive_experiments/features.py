"""
Clean the raw Fed balance-sheet data and first-difference it for modeling.

Input: ``data/raw/fed_balance_sheet.csv``.

Output: ``data/interim/fed_balance_sheet_cleaned.csv`` (complete, de-duplicated
rows) and ``data/processed/fed_balance_sheet_first_difference.csv`` (month-over-month
changes of both series).
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from autoregressive_experiments.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()

DATE_COLUMN = "date"
VALUE_COLUMNS = ["fed_total_assets_T", "monetary_base_T"]


def read_dataset(input_path: Path = RAW_DATA_DIR) -> pd.DataFrame:
    logger.info(f"Reading dataset from {input_path}")
    return pd.read_csv(input_path / "fed_balance_sheet.csv")


def diagnose_dataset(name: str, dataframe: pd.DataFrame) -> pd.DataFrame:
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
    duplicate_rows = df_clean[df_clean.duplicated(keep=False)]
    duplicate_count = df_clean.duplicated().sum()
    print(f"\nDuplicate rows (excluding first occurrences): {duplicate_count}")
    if duplicate_rows.empty:
        print("No duplicate rows.")
    else:
        print(duplicate_rows)
    return df_clean


def first_difference(dataframe: pd.DataFrame) -> pd.DataFrame:
    required_columns = {DATE_COLUMN, *VALUE_COLUMNS}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    dataframe = dataframe.sort_values(DATE_COLUMN).reset_index(drop=True)
    differences = dataframe.loc[:, [DATE_COLUMN, *VALUE_COLUMNS]].copy()
    differences[VALUE_COLUMNS] = (
        dataframe[VALUE_COLUMNS].apply(pd.to_numeric, errors="raise").diff()
    )
    return differences.dropna(subset=VALUE_COLUMNS).reset_index(drop=True)


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    cleaned_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv",
    output_path: Path = PROCESSED_DATA_DIR / "fed_balance_sheet_first_difference.csv",
) -> None:
    df_clean = diagnose_dataset("Fed balance sheet dataset", read_dataset(input_path))
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


if __name__ == "__main__":
    app()
