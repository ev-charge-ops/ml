import numpy as np
import pandas as pd
import pytest

from ml.config import PRIVATE
from ml.data.hourly import build_hourly_panel
from ml.data.loaders import load_sample_sessions
from ml.training.datasets import HORIZON_HOURS, build_demand_dataset
from ml.training.evaluate import ANOMALY_KINDS, inject_anomalies


def test_demand_target_is_mean_of_next_hours() -> None:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h")
    panel = pd.DataFrame(
        {
            "site_id": "A",
            "timestamp": timestamps,
            "demand_index": np.linspace(0.0, 0.9, 10),
        }
    )

    dataset = build_demand_dataset(panel)

    assert len(dataset) == len(panel) - (HORIZON_HOURS - 1)
    assert dataset["next_demand"].iloc[0] == pytest.approx(0.1)
    assert dataset["is_test"].iloc[-1]
    assert not dataset["is_test"].iloc[0]


def test_demand_dataset_skips_gaps() -> None:
    timestamps = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00", "2024-01-08 00:00"]
    )
    panel = pd.DataFrame(
        {"site_id": "A", "timestamp": timestamps, "demand_index": [0.1, 0.2, 0.3, 0.4]}
    )

    dataset = build_demand_dataset(panel)

    assert dataset["timestamp"].tolist() == [timestamps[0]]


def test_demand_dataset_from_sample() -> None:
    dataset = build_demand_dataset(build_hourly_panel(load_sample_sessions()))

    assert dataset["next_demand"].between(0, 1.5).all()
    assert 0.1 < dataset["is_test"].mean() < 0.3


def test_injected_anomalies_cover_every_kind() -> None:
    sessions = load_sample_sessions()
    sessions = sessions[
        sessions["is_valid"] & (sessions["charge_point_type"] == PRIVATE)
    ]

    injected = inject_anomalies(sessions, 5, np.random.default_rng(0))

    assert injected["anomaly_kind"].value_counts().to_dict() == dict.fromkeys(
        ANOMALY_KINDS, 5
    )
    long = injected[injected["anomaly_kind"] == "extended_occupation"]
    assert (long["duration_minutes"] >= 72 * 60).all()
    assert (long["idle_minutes"] <= long["duration_minutes"]).all()
