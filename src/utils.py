import os
from typing import Dict


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def get_default_paths() -> Dict[str, str]:
    """Return default paths used in the project."""
    return {
        "raw_csv": "data/raw/AirQualityUCI.csv",
        "results_dir": "results",
        "eda_summary": "results/eda_summary.json",
        "model_metrics": "results/model_metrics.json",
    }