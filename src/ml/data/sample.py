import argparse
from pathlib import Path

import pandas as pd

from ..config import RAW_DATA_DIR, SAMPLE_DATA_DIR
from .loaders import SAMPLE_FILE, SESSION_COLUMNS, load_sessions

SAMPLE_SITES = ("NO_ASK", "NO_BAR", "TKU_1100", "TKU_1099", "TKU_1138")
SAMPLE_START = "2019-09-01"
SAMPLE_END = "2019-11-01"


def build_sample(sessions: pd.DataFrame) -> pd.DataFrame:
    in_window = sessions["plugin_time"].between(
        SAMPLE_START, SAMPLE_END, inclusive="left"
    )
    selected = sessions[in_window & sessions["site_id"].isin(SAMPLE_SITES)]
    rounded = selected.round(
        {"energy_kwh": 3, "duration_minutes": 3, "idle_minutes": 3}
    )
    return rounded[SESSION_COLUMNS].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the committed sample of sessions"
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--sample-dir", type=Path, default=SAMPLE_DATA_DIR)
    args = parser.parse_args()
    sample = build_sample(load_sessions(args.raw_dir))
    args.sample_dir.mkdir(parents=True, exist_ok=True)
    target = args.sample_dir / SAMPLE_FILE
    sample.to_csv(target, index=False)
    print(f"{len(sample)} sessions written to {target}")


if __name__ == "__main__":
    main()
