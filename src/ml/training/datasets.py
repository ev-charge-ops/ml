import numpy as np
import pandas as pd

from ..config import RANDOM_SEED

TEST_SHARE = 0.2
MAX_DEMAND_INDEX = 1.5
HORIZON_HOURS = 3


def build_demand_dataset(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    by_site = panel.groupby("site_id")
    upcoming = [by_site["demand_index"].shift(-step) for step in range(HORIZON_HOURS)]
    horizon_end = by_site["timestamp"].shift(-(HORIZON_HOURS - 1))
    is_contiguous = horizon_end == panel["timestamp"] + pd.Timedelta(
        hours=HORIZON_HOURS - 1
    )
    target = pd.concat(upcoming, axis=1).mean(axis=1, skipna=False)
    dataset = panel.assign(next_demand=target.clip(upper=MAX_DEMAND_INDEX))
    dataset = dataset[is_contiguous].reset_index(drop=True)
    position = dataset.groupby("site_id").cumcount()
    size = dataset.groupby("site_id")["timestamp"].transform("size")
    return dataset.assign(is_test=position >= size * (1 - TEST_SHARE))


def build_anomaly_dataset(sessions: pd.DataFrame) -> pd.DataFrame:
    valid = sessions[sessions["is_valid"]].reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_SEED)
    return valid.assign(is_test=rng.random(len(valid)) < TEST_SHARE)
