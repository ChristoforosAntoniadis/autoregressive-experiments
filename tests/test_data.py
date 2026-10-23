"""
Tests for first-differencing in autoregressive_experiments.features.
"""

import pandas as pd
import pytest

from autoregressive_experiments.features import first_difference


def test_first_difference_sorts_by_date_and_drops_the_first_row():
    """Rows given out of order are sorted first; each value becomes the change from the month before."""
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


def test_first_difference_rejects_missing_columns():
    """A dataset without one of the value columns is refused with a clear error."""
    dataframe = pd.DataFrame({"date": ["2020-01-31"], "fed_total_assets_T": [1.0]})

    with pytest.raises(ValueError, match="monetary_base_T"):
        first_difference(dataframe)
