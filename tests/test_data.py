import pandas as pd
import pytest

from ml.config import COMMERCIAL, PRIVATE
from ml.data.hourly import build_hourly_panel
from ml.data.loaders import SESSION_COLUMNS, is_valid_session, load_sample_sessions


@pytest.fixture(scope="module")
def sample_sessions() -> pd.DataFrame:
    return load_sample_sessions()


def test_sample_has_normalized_schema(sample_sessions: pd.DataFrame) -> None:
    assert set(SESSION_COLUMNS) <= set(sample_sessions.columns)
    assert set(sample_sessions["charge_point_type"]) == {PRIVATE, COMMERCIAL}
    assert sample_sessions["session_id"].is_unique


def test_derived_columns_are_consistent(sample_sessions: pd.DataFrame) -> None:
    row = sample_sessions.iloc[0]
    expected_power = row["energy_kwh"] / (row["duration_minutes"] / 60)

    assert row["average_power_kw"] == pytest.approx(expected_power)
    assert row["start_hour"] == row["plugin_time"].hour
    assert row["day_of_week"] == row["plugin_time"].dayofweek


def test_invalid_sessions_are_flagged() -> None:
    sessions = pd.DataFrame(
        {
            "charge_point_type": [PRIVATE, PRIVATE, COMMERCIAL, PRIVATE],
            "energy_kwh": [10.0, 0.1, 30.0, 5.0],
            "duration_minutes": [120.0, 120.0, 20.0, 60.0],
            "average_power_kw": [5.0, 0.05, 90.0, 5.0],
            "idle_minutes": [10.0, 10.0, 0.0, None],
        }
    )

    assert is_valid_session(sessions).tolist() == [True, False, False, False]


def test_hourly_panel_counts_concurrent_sessions() -> None:
    sessions = pd.DataFrame(
        {
            "site_id": ["A", "A"],
            "charge_point_type": [PRIVATE, PRIVATE],
            "plugin_time": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:30"]),
            "plugout_time": pd.to_datetime(["2024-01-01 12:00", "2024-01-01 11:00"]),
            "duration_minutes": [120.0, 30.0],
        }
    )

    panel = build_hourly_panel(sessions).set_index("timestamp")

    assert panel.loc["2024-01-01 10:00", "connected"] == pytest.approx(1.5)
    assert panel.loc["2024-01-01 10:00", "arrivals"] == 2
    assert panel.loc["2024-01-01 11:00", "connected"] == pytest.approx(1.0)
    assert panel["occupancy_ratio"].between(0, 1).all()
    assert (panel["queue_length"] >= 0).all()


def test_hourly_panel_on_sample(sample_sessions: pd.DataFrame) -> None:
    panel = build_hourly_panel(sample_sessions)

    assert set(panel["site_id"]) == set(sample_sessions["site_id"])
    assert panel["hour"].between(0, 23).all()
    assert panel["day_of_week"].between(0, 6).all()
    assert (panel["capacity"] >= 1).all()
