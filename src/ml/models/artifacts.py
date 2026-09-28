import json
from pathlib import Path
from typing import Any

import joblib

from ..config import ARTIFACTS_DIR

MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"


def artifact_dir(model_name: str, version: str, root: Path = ARTIFACTS_DIR) -> Path:
    return root / model_name / version


def save_artifact(
    payload: dict[str, Any], metrics: dict[str, Any], target_dir: Path
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, target_dir / MODEL_FILE, compress=3)
    (target_dir / METRICS_FILE).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_payload(source_dir: Path) -> dict[str, Any]:
    return joblib.load(source_dir / MODEL_FILE)
