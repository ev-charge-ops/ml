import numpy as np

from .common import Columns, column, cyclic_hour

ANOMALY_INPUTS = (
    "energy_kwh",
    "duration_minutes",
    "idle_minutes",
    "average_power_kw",
    "start_hour",
    "day_of_week",
    "charge_point_type",
)
ANOMALY_FEATURES = (
    "energy_kwh",
    "duration_hours",
    "idle_hours",
    "average_power_kw",
    "charging_power_kw",
    "start_hour_sin",
    "start_hour_cos",
)
MIN_CHARGING_HOURS = 1 / 60


def build_anomaly_features(data: Columns) -> np.ndarray:
    energy = np.clip(column(data, "energy_kwh"), 0.0, None)
    duration = np.clip(column(data, "duration_minutes"), 0.0, None)
    idle = np.clip(column(data, "idle_minutes"), 0.0, None)
    idle = np.minimum(idle, duration)
    average_power = np.clip(column(data, "average_power_kw"), 0.0, None)
    charging_hours = np.maximum((duration - idle) / 60, MIN_CHARGING_HOURS)
    hour_sin, hour_cos = cyclic_hour(column(data, "start_hour"))
    return np.column_stack(
        [
            energy,
            duration / 60,
            idle / 60,
            average_power,
            energy / charging_hours,
            hour_sin,
            hour_cos,
        ]
    )
