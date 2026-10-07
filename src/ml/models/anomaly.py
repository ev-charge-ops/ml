from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from ..config import ANOMALY_MODEL, ARTIFACTS_DIR, CHARGE_POINT_TYPES, RANDOM_SEED
from ..features.anomaly import ANOMALY_FEATURES, build_anomaly_features
from ..features.common import Columns
from .artifacts import artifact_dir, load_payload

THRESHOLD_QUANTILE = 0.98
FLOOR_QUANTILE = 0.01
NORMALIZED_THRESHOLD = 0.5

ESTIMATOR_PARAMS = {
    "n_estimators": 300,
    "max_samples": 1024,
    "random_state": RANDOM_SEED,
}


@dataclass(frozen=True)
class ScoreScale:
    floor: float
    threshold: float

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        anchors = (self.floor, self.threshold, 1.0)
        return np.interp(raw, anchors, (0.0, NORMALIZED_THRESHOLD, 1.0))

    @classmethod
    def fit(cls, raw: np.ndarray) -> "ScoreScale":
        return cls(
            floor=float(np.quantile(raw, FLOOR_QUANTILE)),
            threshold=float(np.quantile(raw, THRESHOLD_QUANTILE)),
        )


def raw_scores(estimator: IsolationForest, features: np.ndarray) -> np.ndarray:
    return -estimator.score_samples(features)


@dataclass(frozen=True)
class AnomalyModel:
    estimators: dict[str, IsolationForest]
    scales: dict[str, ScoreScale]
    version: str

    def score(self, data: Columns) -> np.ndarray:
        features = build_anomaly_features(data)
        types = np.asarray(data["charge_point_type"], dtype=object)
        unknown = set(types.tolist()) - set(self.estimators)
        if unknown:
            raise ValueError(f"unknown charge point types: {sorted(unknown)}")
        scores = np.zeros(len(types))
        for name, estimator in self.estimators.items():
            mask = types == name
            if mask.any():
                raw = raw_scores(estimator, features[mask])
                scores[mask] = self.scales[name].normalize(raw)
        return scores

    @staticmethod
    def is_anomaly(scores: np.ndarray) -> np.ndarray:
        return scores >= NORMALIZED_THRESHOLD

    def to_payload(self) -> dict[str, Any]:
        return {
            "estimators": self.estimators,
            "scales": {
                name: {"floor": scale.floor, "threshold": scale.threshold}
                for name, scale in self.scales.items()
            },
            "version": self.version,
            "features": list(ANOMALY_FEATURES),
        }

    @classmethod
    def train(
        cls, features: np.ndarray, types: np.ndarray, version: str
    ) -> "AnomalyModel":
        estimators, scales = {}, {}
        for name in CHARGE_POINT_TYPES:
            subset = features[types == name]
            estimator = IsolationForest(**ESTIMATOR_PARAMS).fit(subset)
            estimators[name] = estimator
            scales[name] = ScoreScale.fit(raw_scores(estimator, subset))
        return cls(estimators=estimators, scales=scales, version=version)

    @classmethod
    def load(cls, version: str, root: Path = ARTIFACTS_DIR) -> "AnomalyModel":
        payload = load_payload(artifact_dir(ANOMALY_MODEL, version, root))
        if payload["features"] != list(ANOMALY_FEATURES):
            raise ValueError("anomaly artifact was built with other features")
        return cls(
            estimators=payload["estimators"],
            scales={
                name: ScoreScale(**scale) for name, scale in payload["scales"].items()
            },
            version=payload["version"],
        )
