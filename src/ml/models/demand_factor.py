from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from ..config import ARTIFACTS_DIR, DEMAND_FACTOR_MODEL, RANDOM_SEED
from ..features.common import Columns
from ..features.demand import DEMAND_FEATURES, build_demand_features
from .artifacts import artifact_dir, load_payload

MIN_FACTOR = 0.8
MAX_FACTOR = 1.5
DEMAND_KNOTS = (0.25, 0.5, 0.9)
FACTOR_KNOTS = (MIN_FACTOR, 1.0, MAX_FACTOR)

ESTIMATOR_PARAMS = {
    "loss": "huber",
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": RANDOM_SEED,
}


def demand_to_factor(demand: np.ndarray) -> np.ndarray:
    factor = np.interp(np.asarray(demand, dtype=float), DEMAND_KNOTS, FACTOR_KNOTS)
    return np.clip(factor, MIN_FACTOR, MAX_FACTOR)


def train_demand_estimator(
    features: np.ndarray, target: np.ndarray
) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(**ESTIMATOR_PARAMS).fit(features, target)


@dataclass(frozen=True)
class DemandFactorModel:
    estimator: GradientBoostingRegressor
    version: str

    def predict_demand(self, data: Columns) -> np.ndarray:
        return np.clip(self.estimator.predict(build_demand_features(data)), 0.0, None)

    def predict_factor(self, data: Columns) -> np.ndarray:
        return demand_to_factor(self.predict_demand(data))

    def to_payload(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "version": self.version,
            "features": list(DEMAND_FEATURES),
            "demand_knots": list(DEMAND_KNOTS),
            "factor_knots": list(FACTOR_KNOTS),
        }

    @classmethod
    def load(cls, version: str, root: Path = ARTIFACTS_DIR) -> "DemandFactorModel":
        payload = load_payload(artifact_dir(DEMAND_FACTOR_MODEL, version, root))
        if payload["features"] != list(DEMAND_FEATURES):
            raise ValueError("demand factor artifact was built with other features")
        return cls(estimator=payload["estimator"], version=payload["version"])
