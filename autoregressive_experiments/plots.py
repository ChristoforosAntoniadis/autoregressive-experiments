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
from autoregressive_experiments.features import DATE_COLUMN, VALUE_COLUMNS
from autoregressive_experiments.residual_checks import SERIES_PALETTE, apply_chart_style
from autoregressive_experiments.runlog import begin_phase, finish_run, start_run_log

app = typer.Typer()

# Palette slots 1 and 2, paired with VALUE_COLUMNS in order (total assets, monetary base).
LINE_COLORS = SERIES_PALETTE[:2]


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv",
    output_path: Path = FIGURES_DIR / "fed_balance_sheet.png",
) -> None:
    """
    Draw both cleaned series over time and save the figure (make plots).

    Parameters
    ----------
    input_path : Path, default=INTERIM_DATA_DIR / "fed_balance_sheet_cleaned.csv"
        The cleaned levels written by ``make features``.
    output_path : Path, default=FIGURES_DIR / "fed_balance_sheet.png"
        Where to save the PNG (150 dpi); its folder is created if needed.

    Returns
    -------
    None
        Writes the figure and logs where it went.

    Raises
    ------
    ValueError
        If ``date`` or a value column is missing, or (from pd.to_numeric) if a value isn't a
        number.

    Notes
    -----
    Processing:
    1. Read the cleaned levels with parsed dates and check the required columns.
    2. Sort by date and make sure both series are numeric.
    3. Compute the Pearson correlation of the two levels.
    4. Draw each series as a line with a light fill down to zero, on a 12 x 6 inch figure.
    5. Add the title, axis labels, compact date ticks and the legend (titled with the
       correlation; the grid comes from apply_chart_style), then save and close the figure.

    The correlation is between the levels, which both trend upwards, so a value near 1
    mostly reflects the shared trend; it isn't evidence of a causal or cointegrating link
    (the Johansen test, step V3, finds none on train).
    """
    start_run_log("plots")
    begin_phase("Plot the cleaned series")
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
    # The project's shared chart style (hairline grid, chrome colors), as in every other figure.
    apply_chart_style()
    figure, axis = plt.subplots(figsize=(12, 6))

    for column, color in zip(VALUE_COLUMNS, LINE_COLORS):
        values = dataframe[column]
        axis.plot(dataframe[DATE_COLUMN], values, label=column, color=color)
        # Low alpha, so the two filled areas stay readable where they overlap.
        axis.fill_between(dataframe[DATE_COLUMN], values, color=color, alpha=0.18)

    axis.set(title="Federal Reserve Balance Sheet", xlabel="Time", ylabel="Value (trillions)")
    # ConciseDateFormatter keeps tick labels short: only the date parts that change are shown.
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    axis.legend(title=f"Pearson correlation: r = {correlation:.3f}")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    # Free the figure's memory; pyplot keeps every open figure alive until closed.
    plt.close(figure)
    logger.success(f"Plot saved to {output_path}")

    finish_run()


if __name__ == "__main__":
    app()
