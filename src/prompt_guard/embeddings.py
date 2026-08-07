"""Sentence embedding model wrapper."""

from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

import numpy as np


class SentenceEmbeddingModel:
    """Lazy wrapper around SentenceTransformers with normalized output."""

    def __init__(self, model_name: str, cache_folder: Path | None = None) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder

    @cached_property
    def model(self):
        if self.cache_folder is not None:
            self.cache_folder.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self.cache_folder))
            os.environ.setdefault("HF_HUB_CACHE", str(self.cache_folder / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(self.cache_folder / "transformers"))
            os.environ.setdefault("HF_XET_CACHE", str(self.cache_folder / "xet"))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.model_name,
            device="cpu",
            cache_folder=str(self.cache_folder) if self.cache_folder is not None else None,
        )

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
