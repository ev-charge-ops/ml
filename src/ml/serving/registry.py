from dataclasses import dataclass
from functools import lru_cache

from ..config import ANOMALY_VERSION, DEMAND_FACTOR_VERSION
from ..models.anomaly import AnomalyModel
from ..models.demand_factor import DemandFactorModel


@dataclass(frozen=True)
class ModelRegistry:
    demand_factor: DemandFactorModel
    anomaly: AnomalyModel


@lru_cache(maxsize=1)
def load_registry() -> ModelRegistry:
    return ModelRegistry(
        demand_factor=DemandFactorModel.load(DEMAND_FACTOR_VERSION),
        anomaly=AnomalyModel.load(ANOMALY_VERSION),
    )
