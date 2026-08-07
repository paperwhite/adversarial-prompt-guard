"""Project settings loaded from the JSON config artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/guard_config.json")


@dataclass(frozen=True)
class GuardSettings:
    database_path: Path
    embedding_model: str
    embedding_dimension: int
    model_cache_path: Path
    similarity_threshold: float
    regex_threshold: float
    default_top_k: int
    batch_size: int


def load_settings(project_root: Path) -> GuardSettings:
    data = _read_json(project_root / CONFIG_PATH)
    return GuardSettings(
        database_path=project_root / data["database_path"],
        embedding_model=data["embedding_model"],
        embedding_dimension=int(data["embedding_dimension"]),
        model_cache_path=project_root / data["model_cache_path"],
        similarity_threshold=float(data["similarity_threshold"]),
        regex_threshold=float(data["regex_threshold"]),
        default_top_k=int(data["default_top_k"]),
        batch_size=int(data["batch_size"]),
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
