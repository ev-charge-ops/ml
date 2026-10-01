import numpy as np

from .common import Columns, column, is_commercial, is_weekend

DEMAND_INPUTS = (
    "hour",
    "day_of_week",
    "occupancy_ratio",
    "queue_length",
    "charge_point_type",
)
DEMAND_FEATURES = (
    "hour",
    "day_of_week",
    "is_weekend",
    "occupancy_ratio",
    "queue_length",
    "is_commercial",
)
MAX_QUEUE_LENGTH = 10


def build_demand_features(data: Columns) -> np.ndarray:
    hour = column(data, "hour")
    day_of_week = column(data, "day_of_week")
    return np.column_stack(
        [
            hour,
            day_of_week,
            is_weekend(day_of_week),
            np.clip(column(data, "occupancy_ratio"), 0.0, 1.0),
            np.clip(column(data, "queue_length"), 0, MAX_QUEUE_LENGTH),
            is_commercial(data),
        ]
    )
