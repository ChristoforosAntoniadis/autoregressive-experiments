"""
Tests for the dataset profiles: the Daily Treasury Statement's reduction (dataset) and its
cumulative-flow builder (features), the seasonal naive benchmark and the seasonal MASE scale
(model_forecasts, metrics), the seasonal Hampel profile (outliers) and the daily config file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoregressive_experiments import dataset, forecast_evaluation, outliers
from autoregressive_experiments.configs.config_loader import ConfigLoader
from autoregressive_experiments.configs.configurables import CONFIG_DIR
from autoregressive_experiments.features import (
    DATE_COLUMN,
    FLOW_COLUMNS,
    RAW_FILE,
    VALUE_COLUMNS,
    treasury_levels,
)
from autoregressive_experiments.metrics import mase_scale
from autoregressive_experiments.model_forecasts import SEASONAL_NAIVE, series_path

ACCOUNT, KIND, AMOUNT = FLOW_COLUMNS


def test_the_daily_treasury_config_loads_and_links_its_settings() -> None:
    """The second profile passes every check of the loader: business days, a five-day season,
    the seasonal naive among the benchmarks and as the score's reference."""
    settings = ConfigLoader(CONFIG_DIR / "daily_treasury.json").load()

    assert settings.frequency.alias == "B" and settings.diagnostics.seasonal_period == 5
    assert settings.tuning.score_reference in settings.forecast_evaluation.benchmarks
    assert settings.outliers.seasonal_profile and settings.forecast_evaluation.test_step == 20


def test_treasury_levels_cumulate_the_flows_on_business_days_by_hand() -> None:
    """Flows on Thu 2024-01-04 (deposits 1000 + 500 over two accounts, withdrawals 200), none on
    Fri the 5th (a holiday in this toy), Mon the 8th deposits 100: three business days; the
    holiday gets a zero flow, so the cumulative deposits are 1.5, 1.5, 1.6 billion and the
    withdrawals 0.2, 0.2, 0.2."""
    raw = pd.DataFrame(
        {
            DATE_COLUMN: ["2024-01-04", "2024-01-04", "2024-01-04", "2024-01-08"],
            ACCOUNT: ["Federal Reserve Account", "TGA", "TGA", "TGA"],
            KIND: ["Deposits", "Deposits", "Withdrawals", "Deposits"],
            AMOUNT: [1000.0, 500.0, 200.0, 100.0],
        }
    )

    levels = treasury_levels(raw)

    assert levels[DATE_COLUMN].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
    ]
    assert levels[VALUE_COLUMNS[0]].tolist() == pytest.approx([1.5, 1.5, 1.6])
    assert levels[VALUE_COLUMNS[1]].tolist() == pytest.approx([0.2, 0.2, 0.2])


def test_reduce_treasury_keeps_the_main_account_without_its_total_rows(tmp_path: Path) -> None:
    """Of five statement rows, the Tax and Loan account's and the Sub-Total row are dropped; the
    two Federal Reserve Account deposit categories are summed to 30, the withdrawal kept at 7;
    every other downloaded file is removed."""
    table = pd.DataFrame(
        {
            "Record Date": ["2010-03-01"] * 5,
            "Type of Account": [
                "Federal Reserve Account",
                "Federal Reserve Account",
                "Federal Reserve Account",
                "Tax and Loan Note Accounts",
                "Federal Reserve Account",
            ],
            "Transaction Type": ["Deposits", "Deposits", "Withdrawals", "Deposits", "Deposits"],
            "Transaction Category": ["Taxes", "Fees", "Medicare", "Taxes", "Sub-Total Deposits"],
            "Transactions Today": [10, 20, 7, 99, 30],
        }
    )
    table.to_csv(tmp_path / "DTS_OpCashDpstWdrl_20051003_20230403.csv", index=False)
    (tmp_path / "DTS_OpCashBal_20051003_20230403.csv").write_text("x", encoding="utf-8")

    dataset.reduce_treasury(tmp_path)

    assert [p.name for p in tmp_path.iterdir()] == [RAW_FILE]
    reduced = pd.read_csv(tmp_path / RAW_FILE)
    assert reduced.columns.tolist() == [DATE_COLUMN, ACCOUNT, KIND, AMOUNT]
    assert reduced.set_index(KIND)[AMOUNT].to_dict() == {"Deposits": 30, "Withdrawals": 7}


def test_the_seasonal_naive_repeats_the_last_seasons_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a season of 3 and levels 0, 1, 3, 6, 10 (changes 1, 2, 3, 4), the last season's
    changes are 2, 3, 4, so the forecasts are 12, 15, 19, 21, 24."""
    import autoregressive_experiments.model_forecasts as forecasts

    monkeypatch.setattr(forecasts, "SEASONAL_PERIOD", 3)
    history = np.column_stack([[0.0, 1.0, 3.0, 6.0, 10.0]] * 2)

    path = series_path(SEASONAL_NAIVE, {}, history, np.array([]), 0, 5)

    assert path.tolist() == pytest.approx([12.0, 15.0, 19.0, 21.0, 24.0])


def test_the_mase_scale_with_a_seasonal_lag_by_hand() -> None:
    """Levels 1, 2, 4, 7, 11, changes 1, 2, 3, 4: naive's one-step errors are the changes, mean
    2.5; a seasonal naive with season 2 repeats the change two steps back, errors 3 - 1 and
    4 - 2, mean 2."""
    levels = np.array([1.0, 2.0, 4.0, 7.0, 11.0])

    assert mase_scale(levels, 2) == pytest.approx(2.0)
    assert mase_scale(levels) == pytest.approx(2.5)


def test_the_seasonal_profile_spares_a_regular_rhythm_and_flags_a_departure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changes alternating 0 and 10 with a season of 2 (a 'payday' every second period) plus one
    payday of 30: with the profile each change is compared with its own position's median, so
    the regular paydays sit inside their bounds and only the 30 is flagged. Its bounds: position
    median 10, the window's residues are all 0 except 20, so the median absolute deviation is 0
    and the mean absolute deviation stands in: 20 / 13 x sqrt(pi / 2) x 3 above and below 10."""
    monkeypatch.setattr(outliers, "SEASONAL_PERIOD", 2)
    monkeypatch.setattr(outliers, "HALF_WIDTH", 6)
    monkeypatch.setattr(outliers, "SEASONAL_PROFILE", True)
    changes = np.array([0.0, 10.0] * 3 + [0.0, 30.0] + [0.0, 10.0] * 3)

    lower, upper, flags = outliers.hampel(changes)

    assert flags.tolist() == [index == 7 for index in range(14)]
    spread = 3 * outliers.MEAN_AD_TO_SD * 20 / 13
    assert (lower[7], upper[7]) == pytest.approx((10 - spread, 10 + spread))


def test_without_the_profile_a_weekly_payday_is_flagged_every_week(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four days of 1 and a payday of 10, three weeks, season 5. Plain filter, the first payday:
    its window (11 changes) holds two 10s and nine 1s, median 1, deviations nine 0s and two 9s,
    so the mean absolute deviation 18 / 11 stands in and the bounds are 1 +/- 3 x 1.2533 x
    1.636 = +/- 6.15: the payday's deviation of 9 is flagged, and so is the second payday. The
    last payday's window is cut at the end (7 changes, two of them 10s): mean deviation 18 / 7,
    bounds +/- 9.67, so it escapes. With the profile every residue is 0 and nothing is flagged."""
    monkeypatch.setattr(outliers, "SEASONAL_PERIOD", 5)
    monkeypatch.setattr(outliers, "HALF_WIDTH", 6)
    changes = np.array([1.0, 1.0, 1.0, 1.0, 10.0] * 3)

    monkeypatch.setattr(outliers, "SEASONAL_PROFILE", False)
    plain_flags = outliers.hampel(changes)[2]
    monkeypatch.setattr(outliers, "SEASONAL_PROFILE", True)
    profile_flags = outliers.hampel(changes)[2]

    assert plain_flags.tolist() == [index in (4, 9) for index in range(15)]
    assert not profile_flags.any()


def test_overlapping_origins_by_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Origins every 20 periods: a 21-period forecast shares its last target with the next
    origin (2 origins, one lag); a 5-period forecast shares none (1 origin, no lag). Origins
    every period: the horizon itself, as before."""
    monkeypatch.setattr(forecast_evaluation, "TEST_STEP", 20)
    assert forecast_evaluation.overlapping_origins(21) == 2
    assert forecast_evaluation.overlapping_origins(5) == 1
    monkeypatch.setattr(forecast_evaluation, "TEST_STEP", 1)
    assert forecast_evaluation.overlapping_origins(12) == 12


def test_test_origins_include_the_last_validation_period(monkeypatch: pytest.MonkeyPatch) -> None:
    """Origins every 20 periods with a 33-period validation split (train ends at 100, validation
    at 133, data at 160): 33 mod 20 = 13, so the grid starts at 113 and passes 133; from
    train_end it would be 100, 120, 140 and never 133. Every period: 100 to 159, as before."""
    monkeypatch.setattr(forecast_evaluation, "TEST_STEP", 20)
    assert forecast_evaluation.test_origins(100, 133, 160) == [113, 133, 153]
    monkeypatch.setattr(forecast_evaluation, "TEST_STEP", 1)
    assert forecast_evaluation.test_origins(100, 133, 160) == list(range(100, 160))
