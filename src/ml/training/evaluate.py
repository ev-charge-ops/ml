from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from ..data.loaders import MAX_AVERAGE_POWER_KW

ANOMALY_KINDS = (
    "meter_spike",
    "short_burst",
    "extended_occupation",
    "phantom_occupation",
)
SEED_SPACE = 2**31


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(root_mean_squared_error(target, prediction)),
        "r2": float(r2_score(target, prediction)),
    }


def hourly_mean_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    keys = ["charge_point_type", "hour", "day_of_week"]
    means = train.groupby(keys)["next_demand"].mean().rename("baseline")
    fallback = train["next_demand"].mean()
    return test.join(means, on=keys)["baseline"].fillna(fallback).to_numpy()


def demand_report(
    test: pd.DataFrame, prediction: np.ndarray, hourly_baseline: np.ndarray
) -> dict[str, dict[str, dict[str, float]]]:
    frame = test.assign(
        model=prediction,
        persistence=test["occupancy_ratio"],
        hourly_mean=hourly_baseline,
    )
    groups = {"ALL": frame} | dict(tuple(frame.groupby("charge_point_type")))
    return {
        name: {
            method: regression_metrics(group["next_demand"], group[method])
            for method in ("model", "persistence", "hourly_mean")
        }
        for name, group in groups.items()
    }


def recompute_power(sessions: pd.DataFrame) -> pd.DataFrame:
    return sessions.assign(
        average_power_kw=sessions["energy_kwh"] / (sessions["duration_minutes"] / 60)
    )


def inject_anomalies(
    sessions: pd.DataFrame, per_kind: int, rng: np.random.Generator
) -> pd.DataFrame:
    frames = []
    for kind in ANOMALY_KINDS:
        seed = int(rng.integers(SEED_SPACE))
        base = sessions.sample(n=per_kind, random_state=seed).copy()
        if kind == "meter_spike":
            max_power = base["charge_point_type"].map(MAX_AVERAGE_POWER_KW)
            hours = base["duration_minutes"] / 60
            base["energy_kwh"] = max_power * rng.uniform(1.5, 3, per_kind) * hours
            base["idle_minutes"] = 0.0
        elif kind == "short_burst":
            base["duration_minutes"] = rng.uniform(5, 15, per_kind)
            base["energy_kwh"] = rng.uniform(20, 60, per_kind)
            base["idle_minutes"] = 0.0
        elif kind == "extended_occupation":
            extra = rng.uniform(72, 120, per_kind) * 60 - base["duration_minutes"]
            base["duration_minutes"] += extra
            base["idle_minutes"] += extra
        elif kind == "phantom_occupation":
            base["duration_minutes"] = rng.uniform(12, 48, per_kind) * 60
            base["energy_kwh"] = rng.uniform(0.5, 1.0, per_kind)
            base["idle_minutes"] = base["duration_minutes"] - rng.uniform(
                5, 15, per_kind
            )
        frames.append(recompute_power(base).assign(anomaly_kind=kind))
    return pd.concat(frames, ignore_index=True)


def anomaly_report(
    is_anomaly: np.ndarray,
    scores: np.ndarray,
    classify: Callable[[np.ndarray], np.ndarray],
) -> dict[str, float]:
    flagged = classify(scores)
    return {
        "precision": float(precision_score(is_anomaly, flagged)),
        "recall": float(recall_score(is_anomaly, flagged)),
        "f1": float(f1_score(is_anomaly, flagged)),
        "false_positive_rate": float(flagged[~is_anomaly].mean()),
        "roc_auc": float(roc_auc_score(is_anomaly, scores)),
        "average_precision": float(average_precision_score(is_anomaly, scores)),
    }
