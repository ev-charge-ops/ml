from typing import Annotated

from fastapi import APIRouter, Depends

from ..models.anomaly import NORMALIZED_THRESHOLD
from .registry import ModelRegistry, load_registry
from .schemas import (
    AnomalyScoreRequest,
    AnomalyScoreResponse,
    DemandFactorRequest,
    DemandFactorResponse,
    HealthResponse,
    ModelVersions,
)

DECIMALS = 4

router = APIRouter()
Registry = Annotated[ModelRegistry, Depends(load_registry)]


@router.get("/health")
def health(registry: Registry) -> HealthResponse:
    return HealthResponse(
        status="ok",
        models=ModelVersions(
            demandFactor=registry.demand_factor.version,
            anomaly=registry.anomaly.version,
        ),
    )


@router.post("/demand-factor")
def demand_factor(
    request: DemandFactorRequest, registry: Registry
) -> DemandFactorResponse:
    model = registry.demand_factor
    factor = model.predict_factor(
        {
            "hour": [request.hour],
            "day_of_week": [request.dayOfWeek],
            "occupancy_ratio": [request.occupancyRatio],
            "queue_length": [request.queueLength],
            "charge_point_type": [request.chargePointType],
        }
    )[0]
    return DemandFactorResponse(
        factor=round(float(factor), DECIMALS), modelVersion=model.version
    )


@router.post("/anomaly-score")
def anomaly_score(
    request: AnomalyScoreRequest, registry: Registry
) -> AnomalyScoreResponse:
    model = registry.anomaly
    raw_score = model.score(
        {
            "energy_kwh": [request.energyKwh],
            "duration_minutes": [request.durationMinutes],
            "idle_minutes": [request.idleMinutes],
            "average_power_kw": [request.averagePowerKw],
            "start_hour": [request.startHour],
            "day_of_week": [request.dayOfWeek],
            "charge_point_type": [request.chargePointType],
        }
    )[0]
    score = round(float(raw_score), DECIMALS)
    return AnomalyScoreResponse(
        score=score,
        isAnomaly=score >= NORMALIZED_THRESHOLD,
        modelVersion=model.version,
    )
