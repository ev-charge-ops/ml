from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChargePointType = Literal["PRIVATE", "COMMERCIAL"]


class ApiModel(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)


class DemandFactorRequest(ApiModel):
    hour: int = Field(ge=0, le=23)
    dayOfWeek: int = Field(ge=0, le=6, description="0 = Monday")
    occupancyRatio: float = Field(ge=0, le=1)
    queueLength: int = Field(ge=0)
    chargePointType: ChargePointType


class DemandFactorResponse(ApiModel):
    factor: float
    modelVersion: str


class AnomalyScoreRequest(ApiModel):
    energyKwh: float = Field(ge=0)
    durationMinutes: float = Field(ge=0)
    idleMinutes: float = Field(ge=0)
    averagePowerKw: float = Field(ge=0)
    startHour: int = Field(ge=0, le=23)
    dayOfWeek: int = Field(ge=0, le=6, description="0 = Monday")
    chargePointType: ChargePointType


class AnomalyScoreResponse(ApiModel):
    score: float
    isAnomaly: bool
    modelVersion: str


class ModelVersions(ApiModel):
    demandFactor: str
    anomaly: str


class HealthResponse(ApiModel):
    status: Literal["ok"]
    models: ModelVersions
