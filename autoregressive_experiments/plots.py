"""
Plot the cleaned Fed balance-sheet series.

Input: ``data/interim/fed_balance_sheet_cleaned.csv``.

Output: ``reports/figures/fed_balance_sheet.png``, both series over time with
their Pearson correlation in the legend.
"""

from pathlib import Path

from loguru import logger
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import typer

from autoregressive_experiments.config import FIGURES_DIR, INTERIM_DATA_DIR

app = typer.Typer()

DATE_COLUMN = "date"
VALUE_COLUMNS = ["fed_total_assets_T", "monetary_base_T"]
LINE_COLORS = ("tab:orange", "tab:blue")


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv",
    output_path: Path = FIGURES_DIR / "fed_balance_sheet.png",
) -> None:
    logger.info(f"Reading cleaned dataset from {input_path}")
    dataframe = pd.read_csv(input_path, parse_dates=[DATE_COLUMN])
    required_columns = {DATE_COLUMN, *VALUE_COLUMNS}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    dataframe = dataframe.sort_values(DATE_COLUMN)
    dataframe.loc[:, VALUE_COLUMNS] = dataframe.loc[:, VALUE_COLUMNS].apply(
        pd.to_numeric, errors="raise"
    )
    correlation = dataframe[VALUE_COLUMNS[0]].corr(dataframe[VALUE_COLUMNS[1]])
    figure, axis = plt.subplots(figsize=(12, 6))

    for column, color in zip(VALUE_COLUMNS, LINE_COLORS):
        values = dataframe[column]
        axis.plot(dataframe[DATE_COLUMN], values, label=column, color=color)
        axis.fill_between(dataframe[DATE_COLUMN], values, color=color, alpha=0.18)

    axis.set(title="Federal Reserve Balance Sheet", xlabel="Time", ylabel="Value (trillions)")
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    axis.grid(True, alpha=0.3)
    axis.legend(title=f"Pearson correlation: r = {correlation:.3f}")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    logger.success(f"Plot saved to {output_path}")


if __name__ == "__main__":
    app()
