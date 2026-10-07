from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ml.app import app
from ml.config import ANOMALY_VERSION, DEMAND_FACTOR_VERSION

DEMAND_REQUEST = {
    "hour": 19,
    "dayOfWeek": 2,
    "occupancyRatio": 0.6,
    "queueLength": 0,
    "chargePointType": "PRIVATE",
}
ANOMALY_REQUEST = {
    "energyKwh": 9.3,
    "durationMinutes": 660,
    "idleMinutes": 480,
    "averagePowerKw": 0.85,
    "startHour": 17,
    "dayOfWeek": 1,
    "chargePointType": "PRIVATE",
}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def demand(client: TestClient, **overrides: Any) -> dict[str, Any]:
    response = client.post("/demand-factor", json=DEMAND_REQUEST | overrides)
    assert response.status_code == 200, response.text
    return response.json()


def anomaly(client: TestClient, **overrides: Any) -> dict[str, Any]:
    response = client.post("/anomaly-score", json=ANOMALY_REQUEST | overrides)
    assert response.status_code == 200, response.text
    return response.json()


def test_health_reports_model_versions(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "models": {"demandFactor": DEMAND_FACTOR_VERSION, "anomaly": ANOMALY_VERSION},
    }


@pytest.mark.parametrize("charge_point_type", ["PRIVATE", "COMMERCIAL"])
def test_demand_factor_returns_factor_and_version(
    client: TestClient, charge_point_type: str
) -> None:
    body = demand(client, chargePointType=charge_point_type)

    assert set(body) == {"factor", "modelVersion"}
    assert 0.8 <= body["factor"] <= 1.5
    assert body["modelVersion"] == DEMAND_FACTOR_VERSION


def test_demand_factor_is_clipped_to_bounds(client: TestClient) -> None:
    empty = demand(client, hour=3, occupancyRatio=0.0, queueLength=0)
    crowded = demand(client, hour=22, occupancyRatio=1.0, queueLength=50)

    assert empty["factor"] == 0.8
    assert crowded["factor"] == 1.5


def test_demand_factor_grows_with_occupancy(client: TestClient) -> None:
    factors = [
        demand(client, occupancyRatio=ratio)["factor"] for ratio in (0.0, 0.3, 0.6, 1.0)
    ]

    assert factors == sorted(factors)
    assert factors[0] < factors[-1]


def test_demand_factor_is_deterministic(client: TestClient) -> None:
    assert demand(client) == demand(client)


def test_demand_factor_accepts_integer_ratio(client: TestClient) -> None:
    assert demand(client, occupancyRatio=1)["factor"] <= 1.5


@pytest.mark.parametrize(
    "overrides",
    [
        {"hour": 24},
        {"hour": -1},
        {"hour": 3.5},
        {"hour": "3"},
        {"dayOfWeek": 7},
        {"occupancyRatio": 1.01},
        {"occupancyRatio": -0.1},
        {"queueLength": -1},
        {"chargePointType": "PUBLIC"},
        {"chargePointType": "private"},
    ],
)
def test_demand_factor_rejects_invalid_input(
    client: TestClient, overrides: dict[str, Any]
) -> None:
    response = client.post("/demand-factor", json=DEMAND_REQUEST | overrides)

    assert response.status_code == 422


def test_demand_factor_rejects_missing_field(client: TestClient) -> None:
    payload = {key: value for key, value in DEMAND_REQUEST.items() if key != "hour"}

    assert client.post("/demand-factor", json=payload).status_code == 422


def test_anomaly_score_for_typical_session(client: TestClient) -> None:
    body = anomaly(client)

    assert set(body) == {"score", "isAnomaly", "modelVersion"}
    assert 0 <= body["score"] < 0.5
    assert body["isAnomaly"] is False
    assert body["modelVersion"] == ANOMALY_VERSION


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "energyKwh": 45,
            "durationMinutes": 10,
            "idleMinutes": 0,
            "averagePowerKw": 270,
        },
        {"durationMinutes": 6000, "idleMinutes": 5820, "averagePowerKw": 0.09},
        {
            "energyKwh": 200,
            "durationMinutes": 120,
            "idleMinutes": 0,
            "averagePowerKw": 100,
            "chargePointType": "COMMERCIAL",
        },
    ],
)
def test_anomaly_score_flags_impossible_sessions(
    client: TestClient, overrides: dict[str, Any]
) -> None:
    body = anomaly(client, **overrides)

    assert 0.5 <= body["score"] <= 1
    assert body["isAnomaly"] is True


def test_anomaly_score_is_deterministic(client: TestClient) -> None:
    assert anomaly(client) == anomaly(client)


def test_anomaly_score_tolerates_idle_longer_than_duration(client: TestClient) -> None:
    body = anomaly(client, idleMinutes=700)

    assert 0 <= body["score"] <= 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"energyKwh": -1},
        {"durationMinutes": -5},
        {"idleMinutes": -1},
        {"averagePowerKw": -0.1},
        {"startHour": 24},
        {"dayOfWeek": -1},
        {"chargePointType": None},
        {"energyKwh": "10"},
    ],
)
def test_anomaly_score_rejects_invalid_input(
    client: TestClient, overrides: dict[str, Any]
) -> None:
    response = client.post("/anomaly-score", json=ANOMALY_REQUEST | overrides)

    assert response.status_code == 422
