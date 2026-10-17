import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # resolve output dirs relative to the project root and make sure they exist
    for key, rel in cfg["paths"].items():
        p = PROJECT_ROOT / rel
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = p
    cfg["data"]["csv_path"] = PROJECT_ROOT / cfg["data"]["csv_path"]
    return cfg


def feature_columns(cfg, include_call):
    """Raw input columns a model variant is allowed to see."""
    cols = list(cfg["features"]["categorical"]) + list(cfg["features"]["numeric"])
    if include_call:
        cols.append(cfg["leakage_review"]["feature"])
    return cols
