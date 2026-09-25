import numpy as np
import pytest

from ml.config import COMMERCIAL, PRIVATE
from ml.features import (
    ANOMALY_FEATURES,
    DEMAND_FEATURES,
    build_anomaly_features,
    build_demand_features,
)
from ml.features.demand import MAX_QUEUE_LENGTH


def demand_input(**overrides):
    data = {
        "hour": [22, 9],
        "day_of_week": [5, 1],
        "occupancy_ratio": [0.8, 0.2],
        "queue_length": [2, 0],
        "charge_point_type": [PRIVATE, COMMERCIAL],
    }
    return data | overrides


def anomaly_input(**overrides):
    data = {
        "energy_kwh": [12.0, 30.0],
        "duration_minutes": [600.0, 0.0],
        "idle_minutes": [420.0, 10.0],
        "average_power_kw": [1.2, 0.0],
        "start_hour": [18, 6],
        "day_of_week": [2, 6],
        "charge_point_type": [PRIVATE, COMMERCIAL],
    }
    return data | overrides


def test_demand_features_shape_and_encoding() -> None:
    features = build_demand_features(demand_input())

    assert features.shape == (2, len(DEMAND_FEATURES))
    columns = dict(zip(DEMAND_FEATURES, features.T))
    assert columns["is_weekend"].tolist() == [1.0, 0.0]
    assert columns["is_commercial"].tolist() == [0.0, 1.0]


def test_demand_features_clip_out_of_range_values() -> None:
    features = build_demand_features(
        demand_input(occupancy_ratio=[1.7, -0.2], queue_length=[50, 0])
    )
    columns = dict(zip(DEMAND_FEATURES, features.T))

    assert columns["occupancy_ratio"].tolist() == [1.0, 0.0]
    assert columns["queue_length"].tolist() == [MAX_QUEUE_LENGTH, 0]


def test_unknown_charge_point_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown charge point types"):
        build_demand_features(demand_input(charge_point_type=["PRIVATE", "PUBLIC"]))


def test_anomaly_features_are_finite_for_edge_cases() -> None:
    features = build_anomaly_features(anomaly_input())

    assert features.shape == (2, len(ANOMALY_FEATURES))
    assert np.isfinite(features).all()


def test_anomaly_features_derive_charging_power() -> None:
    features = build_anomaly_features(anomaly_input())
    columns = dict(zip(ANOMALY_FEATURES, features.T))

    assert columns["duration_hours"][0] == pytest.approx(10.0)
    assert columns["idle_hours"][0] == pytest.approx(7.0)
    assert columns["charging_power_kw"][0] == pytest.approx(12.0 / 3.0)
    assert columns["idle_hours"][1] == 0.0
