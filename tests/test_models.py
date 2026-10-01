import json

import numpy as np
import pytest

from ml.config import (
    ANOMALY_MODEL,
    ANOMALY_VERSION,
    COMMERCIAL,
    DEMAND_FACTOR_MODEL,
    DEMAND_FACTOR_VERSION,
    PRIVATE,
)
from ml.models.anomaly import NORMALIZED_THRESHOLD, AnomalyModel
from ml.models.artifacts import METRICS_FILE, artifact_dir
from ml.models.demand_factor import (
    MAX_FACTOR,
    MIN_FACTOR,
    DemandFactorModel,
    demand_to_factor,
)


@pytest.fixture(scope="module")
def demand_model() -> DemandFactorModel:
    return DemandFactorModel.load(DEMAND_FACTOR_VERSION)


@pytest.fixture(scope="module")
def anomaly_model() -> AnomalyModel:
    return AnomalyModel.load(ANOMALY_VERSION)


def test_demand_to_factor_mapping() -> None:
    factors = demand_to_factor(np.array([0.0, 0.25, 0.5, 0.9, 1.5]))

    assert factors.tolist() == pytest.approx([0.8, 0.8, 1.0, 1.5, 1.5])


def test_demand_model_loads_and_predicts_within_bounds(
    demand_model: DemandFactorModel,
) -> None:
    hours = np.arange(24)
    data = {
        "hour": np.tile(hours, 2),
        "day_of_week": np.zeros(48, dtype=int),
        "occupancy_ratio": np.linspace(0, 1, 48),
        "queue_length": np.zeros(48, dtype=int),
        "charge_point_type": [PRIVATE] * 24 + [COMMERCIAL] * 24,
    }

    factors = demand_model.predict_factor(data)

    assert demand_model.version == DEMAND_FACTOR_VERSION
    assert ((factors >= MIN_FACTOR) & (factors <= MAX_FACTOR)).all()


def test_demand_factor_grows_with_occupancy(demand_model: DemandFactorModel) -> None:
    data = {
        "hour": [2, 2],
        "day_of_week": [2, 2],
        "occupancy_ratio": [0.05, 1.0],
        "queue_length": [0, 3],
        "charge_point_type": [PRIVATE, PRIVATE],
    }

    low, high = demand_model.predict_factor(data)

    assert low < high


def test_anomaly_model_separates_normal_and_impossible_sessions(
    anomaly_model: AnomalyModel,
) -> None:
    data = {
        "energy_kwh": [10.0, 400.0],
        "duration_minutes": [660.0, 30.0],
        "idle_minutes": [480.0, 0.0],
        "average_power_kw": [0.9, 800.0],
        "start_hour": [17, 3],
        "day_of_week": [1, 1],
        "charge_point_type": [PRIVATE, PRIVATE],
    }

    scores = anomaly_model.score(data)

    assert anomaly_model.version == ANOMALY_VERSION
    assert ((scores >= 0) & (scores <= 1)).all()
    assert scores[0] < NORMALIZED_THRESHOLD <= scores[1]
    assert anomaly_model.is_anomaly(scores).tolist() == [False, True]


def test_anomaly_model_is_deterministic(anomaly_model: AnomalyModel) -> None:
    data = {
        "energy_kwh": [5.0],
        "duration_minutes": [120.0],
        "idle_minutes": [30.0],
        "average_power_kw": [2.5],
        "start_hour": [9],
        "day_of_week": [3],
        "charge_point_type": [COMMERCIAL],
    }

    assert anomaly_model.score(data)[0] == anomaly_model.score(data)[0]


@pytest.mark.parametrize(
    ("model_name", "version"),
    [(DEMAND_FACTOR_MODEL, DEMAND_FACTOR_VERSION), (ANOMALY_MODEL, ANOMALY_VERSION)],
)
def test_artifacts_ship_metrics_and_model_card(model_name: str, version: str) -> None:
    directory = artifact_dir(model_name, version)
    metrics = json.loads((directory / METRICS_FILE).read_text(encoding="utf-8"))

    assert metrics["version"] == version
    assert (directory / "model-card.md").exists()
