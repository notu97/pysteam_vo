"""Config loading. All thresholds / noise values come from pysteam_vo/configs/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config (defaults to pysteam_vo/configs/default.yaml)."""
    with open(path or DEFAULT_CONFIG, "r") as f:
        return yaml.safe_load(f)
