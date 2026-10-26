"""
Tests for first-differencing and the chronological 60 / 20 / 20 split in
autoregressive_experiments.features.
"""

import pandas as pd
import pytest

from autoregressive_experiments.features import (
    first_difference,
    split_boundaries,
    split_by_time,
)


def monthly_levels(month_count: int) -> pd.DataFrame:
    """Month-end levels 1, 2, 3, ... from January 2020, given in reverse date order.

    Parameters
    ----------
    month_count : int
        Number of months to generate.

    Returns
    -------
    pandas.DataFrame
        Columns date, fed_total_assets_T and monetary_base_T (both series equal 1, 2, 3, ...),
        newest month first, so the tests also check that the functions sort by date.
    """
    dates = pd.date_range("2020-01-31", periods=month_count, freq="ME")
    values = [float(i + 1) for i in range(month_count)]
    frame = pd.DataFrame({"date": dates, "fed_total_assets_T": values, "monetary_base_T": values})
    return frame.iloc[::-1].reset_index(drop=True)


def months(frame: pd.DataFrame) -> list[str]:
    """The frame's dates as "YYYY-MM" strings, in row order.

    Parameters
    ----------
    frame : pandas.DataFrame
        A frame with a datetime "date" column.

    Returns
    -------
    list of str
        One "YYYY-MM" string per row.
    """
    return frame["date"].dt.strftime("%Y-%m").tolist()


def test_split_boundaries_take_the_first_60_and_next_20_percent_of_months() -> None:
    """10 months: train ends at month 6 (June), validation at month 8 (August)."""
    # 10 * 60 // 100 = 6 training months; 10 * 80 // 100 = 8 months up to validation's end.
    train_end, validation_end = split_boundaries(monthly_levels(10))

    assert (train_end.strftime("%Y-%m"), validation_end.strftime("%Y-%m")) == (
        "2020-06",
        "2020-08",
    )


def test_split_boundaries_round_down_for_this_dataset_size() -> None:
    """279 months (the real dataset): 279 * 60 // 100 = 167 and 279 * 80 // 100 = 223."""
    dates = pd.date_range("2020-01-31", periods=279, freq="ME")
    train_end, validation_end = split_boundaries(monthly_levels(279))

    # 0-based: the 167th month is dates[166] and the 223rd is dates[222].
    assert (train_end, validation_end) == (dates[166], dates[222])


def test_split_by_time_cuts_levels_in_date_order_without_overlap() -> None:
    """10 levels given in reverse order become 6 / 2 / 2 months, each block sorted."""
    levels = monthly_levels(10)

    splits = split_by_time(levels, split_boundaries(levels))

    assert months(splits["train"]) == [f"2020-0{m}" for m in range(1, 7)]
    assert months(splits["validation"]) == ["2020-07", "2020-08"]
    assert months(splits["test"]) == ["2020-09", "2020-10"]
    assert splits["train"]["fed_total_assets_T"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_split_by_time_cuts_differences_at_the_same_months_as_levels() -> None:
    """The 9 differences (February to October) split 5 / 2 / 2, at the level boundaries."""
    levels = monthly_levels(10)
    differences = first_difference(levels)

    # The boundaries come from the levels (June, August), not from the 9 differences.
    splits = split_by_time(differences, split_boundaries(levels))

    assert months(splits["train"]) == [f"2020-0{m}" for m in range(2, 7)]
    assert months(splits["validation"]) == ["2020-07", "2020-08"]
    assert months(splits["test"]) == ["2020-09", "2020-10"]
    # Each level rises by 1 a month, so every difference is 1.
    assert splits["train"]["fed_total_assets_T"].tolist() == [1.0] * 5


def test_split_boundaries_reject_too_few_months() -> None:
    """2 months give 1 training month and no validation month, which can't be split."""
    # 2 * 60 // 100 = 1 and 2 * 80 // 100 = 1: validation would end where train ends.
    with pytest.raises(ValueError, match="too few"):
        split_boundaries(monthly_levels(2))


def test_first_difference_sorts_by_date_and_drops_the_first_row() -> None:
    """Rows given out of order are sorted first; each value becomes the change from the month before."""
    # Sorted by date, total assets is 1, 4, 9 (changes 3, 5) and the base 2.0, 2.5, 2.0
    # (changes 0.5, -0.5).
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-29", "2020-01-31", "2020-03-31"]),
            "fed_total_assets_T": [4.0, 1.0, 9.0],
            "monetary_base_T": [2.5, 2.0, 2.0],
        }
    )

    differences = first_difference(dataframe)

    assert differences["date"].dt.strftime("%Y-%m-%d").tolist() == ["2020-02-29", "2020-03-31"]
    assert differences["fed_total_assets_T"].tolist() == [3.0, 5.0]
    assert differences["monetary_base_T"].tolist() == [0.5, -0.5]


def test_first_difference_rejects_missing_columns() -> None:
    """A dataset without one of the value columns is refused with a clear error."""
    dataframe = pd.DataFrame({"date": ["2020-01-31"], "fed_total_assets_T": [1.0]})

    with pytest.raises(ValueError, match="monetary_base_T"):
        first_difference(dataframe)
