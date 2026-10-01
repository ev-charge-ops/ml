from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..config import CHARGE_POINT_TYPES, COMMERCIAL

Columns = Mapping[str, Sequence[Any] | np.ndarray]

WEEKEND_START = 5


def column(data: Columns, name: str) -> np.ndarray:
    return np.asarray(data[name], dtype=float)


def is_commercial(data: Columns) -> np.ndarray:
    types = np.asarray(data["charge_point_type"], dtype=object)
    unknown = set(types.tolist()) - set(CHARGE_POINT_TYPES)
    if unknown:
        raise ValueError(f"unknown charge point types: {sorted(unknown)}")
    return (types == COMMERCIAL).astype(float)


def is_weekend(day_of_week: np.ndarray) -> np.ndarray:
    return (day_of_week >= WEEKEND_START).astype(float)


def cyclic_hour(hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * hour / 24
    return np.sin(angle), np.cos(angle)
