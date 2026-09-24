from pathlib import Path

import numpy as np
import pandas as pd

from ..config import COMMERCIAL, PRIVATE, RAW_DATA_DIR, SAMPLE_DATA_DIR
from .sources import NORWAY_SESSION_PREDICTIONS, NORWAY_SESSIONS, TURKU_SESSIONS

SESSION_COLUMNS = [
    "session_id",
    "site_id",
    "charge_point_type",
    "connector",
    "plugin_time",
    "plugout_time",
    "energy_kwh",
    "duration_minutes",
    "idle_minutes",
    "source",
]

EFFECTIVE_POWER_KW = {"AC": 3.7, "DC": 37.0}
MAX_AVERAGE_POWER_KW = {PRIVATE: 22.0, COMMERCIAL: 50.0}
MIN_ENERGY_KWH = 0.5
MIN_DURATION_MINUTES = 5.0
MAX_DURATION_MINUTES = 72 * 60.0

SAMPLE_FILE = "sessions.csv"


def load_norway_sessions(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    reports = pd.read_csv(raw_dir / NORWAY_SESSIONS.name, sep=";", decimal=",")
    predictions = pd.read_csv(
        raw_dir / NORWAY_SESSION_PREDICTIONS.name,
        sep=";",
        decimal=",",
        usecols=["session_id", "idle_time"],
    )
    frame = reports.merge(predictions, on="session_id", how="left")
    plugin = pd.to_datetime(frame["plugin_time"])
    plugout = pd.to_datetime(frame["plugout_time"])
    return pd.DataFrame(
        {
            "session_id": frame["session_id"],
            "site_id": "NO_" + frame["location"],
            "charge_point_type": PRIVATE,
            "connector": "AC",
            "plugin_time": plugin,
            "plugout_time": plugout,
            "energy_kwh": frame["energy_session"].astype(float),
            "duration_minutes": (plugout - plugin).dt.total_seconds() / 60,
            "idle_minutes": frame["idle_time"].astype(float) * 60,
            "source": "norway_residential",
        }
    )


def load_turku_sessions(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    frame = pd.read_csv(raw_dir / TURKU_SESSIONS.name, sep=";", encoding="latin1")
    time_format = "%d/%m/%Y %H.%M"
    plugin = pd.to_datetime(frame["Start time"], format=time_format)
    plugout = pd.to_datetime(frame["Stop time"], format=time_format)
    energy_kwh = frame["Energy (Wh)"].astype(float) / 1000
    duration_minutes = (plugout - plugin).dt.total_seconds() / 60
    effective_power = frame["Plug type"].map(EFFECTIVE_POWER_KW)
    charging_minutes = energy_kwh.clip(lower=0) / effective_power * 60
    return pd.DataFrame(
        {
            "session_id": "TKU_" + frame.index.astype(str),
            "site_id": "TKU_" + frame["Station ID"].astype(str),
            "charge_point_type": COMMERCIAL,
            "connector": frame["Plug type"],
            "plugin_time": plugin,
            "plugout_time": plugout,
            "energy_kwh": energy_kwh,
            "duration_minutes": duration_minutes,
            "idle_minutes": (duration_minutes - charging_minutes).clip(lower=0),
            "source": "turku_public",
        }
    )


def add_derived_columns(sessions: pd.DataFrame) -> pd.DataFrame:
    hours = sessions["duration_minutes"] / 60
    return sessions.assign(
        average_power_kw=np.where(hours > 0, sessions["energy_kwh"] / hours, np.nan),
        start_hour=sessions["plugin_time"].dt.hour,
        day_of_week=sessions["plugin_time"].dt.dayofweek,
    )


def is_valid_session(sessions: pd.DataFrame) -> pd.Series:
    max_power = sessions["charge_point_type"].map(MAX_AVERAGE_POWER_KW)
    return (
        sessions["energy_kwh"].ge(MIN_ENERGY_KWH)
        & sessions["duration_minutes"].between(
            MIN_DURATION_MINUTES, MAX_DURATION_MINUTES
        )
        & sessions["average_power_kw"].le(max_power)
        & sessions["idle_minutes"].notna()
    )


def load_sessions(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    sessions = pd.concat(
        [load_norway_sessions(raw_dir), load_turku_sessions(raw_dir)], ignore_index=True
    )
    sessions = add_derived_columns(sessions)
    return sessions.assign(is_valid=is_valid_session(sessions)).sort_values(
        ["site_id", "plugin_time"], ignore_index=True
    )


def load_sample_sessions(sample_dir: Path = SAMPLE_DATA_DIR) -> pd.DataFrame:
    sessions = pd.read_csv(
        sample_dir / SAMPLE_FILE, parse_dates=["plugin_time", "plugout_time"]
    )
    sessions = add_derived_columns(sessions)
    return sessions.assign(is_valid=is_valid_session(sessions))
