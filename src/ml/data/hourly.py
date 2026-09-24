import numpy as np
import pandas as pd

from .loaders import MAX_DURATION_MINUTES

CAPACITY_QUANTILE = 0.95
MINUTES_PER_HOUR = 60


def site_concurrency(site_sessions: pd.DataFrame) -> pd.DataFrame:
    start = site_sessions["plugin_time"].min().floor("h")
    end = site_sessions["plugout_time"].max().ceil("h") + pd.Timedelta(hours=1)
    total_minutes = int((end - start).total_seconds() // 60)
    plugin_offsets = (site_sessions["plugin_time"] - start).dt.total_seconds() // 60
    plugout_offsets = (site_sessions["plugout_time"] - start).dt.total_seconds() // 60
    deltas = np.zeros(total_minutes + 1, dtype=np.int64)
    np.add.at(deltas, plugin_offsets.to_numpy(dtype=np.int64), 1)
    np.add.at(deltas, plugout_offsets.to_numpy(dtype=np.int64), -1)
    per_minute = np.cumsum(deltas[:total_minutes])
    per_hour = per_minute.reshape(-1, MINUTES_PER_HOUR)
    timestamps = pd.date_range(start, periods=per_hour.shape[0], freq="h")
    arrivals = (
        site_sessions["plugin_time"]
        .dt.floor("h")
        .value_counts()
        .reindex(timestamps, fill_value=0)
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "connected_at_start": per_hour[:, 0],
            "connected": per_hour.mean(axis=1),
            "peak_connected": per_hour.max(axis=1),
            "arrivals": arrivals.to_numpy(),
        }
    )


def drop_inactive_weeks(panel: pd.DataFrame) -> pd.DataFrame:
    week = panel["timestamp"].dt.to_period("W")
    weekly_arrivals = panel.groupby(week)["arrivals"].transform("sum")
    return panel[weekly_arrivals > 0]


def build_hourly_panel(sessions: pd.DataFrame) -> pd.DataFrame:
    usable = sessions.dropna(subset=["plugin_time", "plugout_time"])
    usable = usable[
        (usable["plugout_time"] > usable["plugin_time"])
        & (usable["duration_minutes"] <= MAX_DURATION_MINUTES)
    ]
    frames = []
    for (site_id, charge_point_type), site_sessions in usable.groupby(
        ["site_id", "charge_point_type"]
    ):
        panel = drop_inactive_weeks(site_concurrency(site_sessions))
        capacity = max(1, int(np.ceil(panel["connected"].quantile(CAPACITY_QUANTILE))))
        frames.append(
            panel.assign(
                site_id=site_id,
                charge_point_type=charge_point_type,
                capacity=capacity,
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    snapshot = panel["connected_at_start"]
    return panel.assign(
        hour=panel["timestamp"].dt.hour,
        day_of_week=panel["timestamp"].dt.dayofweek,
        occupancy_ratio=(snapshot / panel["capacity"]).clip(upper=1.0),
        queue_length=(snapshot - panel["capacity"]).clip(lower=0).astype(int),
        demand_index=panel["connected"] / panel["capacity"],
    )
