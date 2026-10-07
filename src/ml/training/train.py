import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from ..config import (
    ANOMALY_MODEL,
    ANOMALY_VERSION,
    ARTIFACTS_DIR,
    DEMAND_FACTOR_MODEL,
    DEMAND_FACTOR_VERSION,
    RANDOM_SEED,
    RAW_DATA_DIR,
)
from ..data.hourly import build_hourly_panel
from ..data.loaders import MAX_DURATION_MINUTES, load_sessions
from ..features.anomaly import ANOMALY_FEATURES, build_anomaly_features
from ..features.demand import DEMAND_FEATURES, build_demand_features
from ..models import anomaly, demand_factor
from ..models.anomaly import AnomalyModel
from ..models.artifacts import artifact_dir, save_artifact
from ..models.demand_factor import DemandFactorModel
from .datasets import build_anomaly_dataset, build_demand_dataset
from .evaluate import (
    anomaly_report,
    demand_report,
    hourly_mean_baseline,
    inject_anomalies,
)

INJECTED_PER_KIND = 100
FACTOR_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
DATA_SOURCES = [
    "Sorensen (2024), residential EV charging in Norway, doi:10.5281/zenodo.13896176",
    "Andrenacci, Bosch, Kulla (2021), Turku public charging, doi:10.5281/zenodo.5721233",
]


def split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return dataset[~dataset["is_test"]], dataset[dataset["is_test"]]


def train_demand_factor(
    sessions: pd.DataFrame, version: str
) -> tuple[DemandFactorModel, dict[str, Any]]:
    train, test = split(build_demand_dataset(build_hourly_panel(sessions)))
    estimator = demand_factor.train_demand_estimator(
        build_demand_features(train), train["next_demand"].to_numpy()
    )
    model = DemandFactorModel(estimator=estimator, version=version)
    prediction = model.predict_demand(test)
    factors = demand_factor.demand_to_factor(prediction)
    importances = estimator.feature_importances_.round(4).tolist()
    metrics = {
        "model": DEMAND_FACTOR_MODEL,
        "version": version,
        "algorithm": "GradientBoostingRegressor",
        "sklearn_version": sklearn.__version__,
        "data_sources": DATA_SOURCES,
        "target": "demand index: mean connected vehicles over the next 3 hours / capacity",
        "split": "temporal: last 20% of the hours of each site are the test set",
        "rows": {"train": len(train), "test": len(test)},
        "params": demand_factor.ESTIMATOR_PARAMS,
        "features": list(DEMAND_FEATURES),
        "feature_importances": dict(zip(DEMAND_FEATURES, importances)),
        "test_metrics": demand_report(
            test, prediction, hourly_mean_baseline(train, test)
        ),
        "factor_mapping": {
            "demand_knots": list(demand_factor.DEMAND_KNOTS),
            "factor_knots": list(demand_factor.FACTOR_KNOTS),
            "clip": [demand_factor.MIN_FACTOR, demand_factor.MAX_FACTOR],
        },
        "test_factor_quantiles": {
            str(q): float(np.quantile(factors, q)) for q in FACTOR_QUANTILES
        },
    }
    return model, metrics


def long_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    usable = sessions.dropna(subset=["idle_minutes", "average_power_kw"])
    return usable[usable["duration_minutes"] > MAX_DURATION_MINUTES]


def train_anomaly(
    sessions: pd.DataFrame, version: str
) -> tuple[AnomalyModel, dict[str, Any]]:
    train, test = split(build_anomaly_dataset(sessions))
    model = AnomalyModel.train(
        build_anomaly_features(train),
        train["charge_point_type"].to_numpy(),
        version,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    injected = inject_anomalies(test, INJECTED_PER_KIND, rng)
    evaluation = pd.concat(
        [test.assign(anomaly_kind="normal"), injected], ignore_index=True
    )
    scores = model.score(evaluation)
    flagged = model.is_anomaly(scores)
    kinds = evaluation["anomaly_kind"].to_numpy()
    types = evaluation["charge_point_type"].to_numpy()
    is_injected = kinds != "normal"
    real_long = long_sessions(sessions)
    metrics = {
        "model": ANOMALY_MODEL,
        "version": version,
        "algorithm": "IsolationForest (one per charge point type)",
        "sklearn_version": sklearn.__version__,
        "data_sources": DATA_SOURCES,
        "split": "random 80/20 over valid sessions",
        "rows": {
            "train": len(train),
            "test_normal": len(test),
            "test_injected": len(injected),
        },
        "params": anomaly.ESTIMATOR_PARAMS,
        "features": list(ANOMALY_FEATURES),
        "score_normalization": {
            "rule": "raw isolation score mapped piecewise-linearly per charge point type",
            "anchors": {
                "0.0": f"quantile {anomaly.FLOOR_QUANTILE} of training raw scores",
                "0.5": f"quantile {anomaly.THRESHOLD_QUANTILE} of training raw scores",
                "1.0": "raw score 1.0",
            },
            "is_anomaly": f"score >= {anomaly.NORMALIZED_THRESHOLD}",
            "raw_scales": model.to_payload()["scales"],
        },
        "test_metrics": anomaly_report(is_injected, scores, model.is_anomaly),
        "test_metrics_by_type": {
            name: anomaly_report(
                is_injected[types == name], scores[types == name], model.is_anomaly
            )
            for name in sorted(set(types))
        },
        "recall_by_kind": {
            kind: float(flagged[kinds == kind].mean())
            for kind in sorted(set(kinds) - {"normal"})
        },
        "real_sessions_over_72h_flagged": {
            "sessions": len(real_long),
            "flagged_share": float(model.is_anomaly(model.score(real_long)).mean()),
        },
    }
    return model, metrics


TRAINERS = {
    DEMAND_FACTOR_MODEL: (train_demand_factor, DEMAND_FACTOR_VERSION),
    ANOMALY_MODEL: (train_anomaly, ANOMALY_VERSION),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist the ML models")
    parser.add_argument("--model", choices=[*TRAINERS, "all"], default="all")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()
    sessions = load_sessions(args.raw_dir)
    names = list(TRAINERS) if args.model == "all" else [args.model]
    for name in names:
        trainer, version = TRAINERS[name]
        model, metrics = trainer(sessions, version)
        save_artifact(
            model.to_payload(), metrics, artifact_dir(name, version, args.artifacts_dir)
        )
        print(f"{name} {version}: {metrics['test_metrics']}")


if __name__ == "__main__":
    main()
