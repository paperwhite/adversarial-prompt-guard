"""Populate the SQLite guard database from normalized JSONL records."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from prompt_guard.db import connect, db_stats, init_db, insert_prompt_batch, reset_prompt_tables
from prompt_guard.embeddings import SentenceEmbeddingModel
from prompt_guard.settings import load_settings

DEFAULT_JSONL = Path("data/interim/adversarial_prompts_step1.jsonl")


def build_database(
    project_root: Path,
    jsonl_path: Path | None = None,
    reset: bool = True,
) -> dict[str, Any]:
    settings = load_settings(project_root)
    source_path = jsonl_path or project_root / DEFAULT_JSONL
    model = SentenceEmbeddingModel(settings.embedding_model, cache_folder=settings.model_cache_path)

    conn = connect(settings.database_path)
    try:
        init_db(conn)
        if reset:
            reset_prompt_tables(conn)
        total = 0
        for batch in _iter_jsonl_batches(source_path, settings.batch_size):
            texts = [record["prompt_text"] for record in batch]
            embeddings = model.encode(texts, batch_size=settings.batch_size)
            total += insert_prompt_batch(conn, batch, embeddings, settings.embedding_model)
            conn.commit()
        stats = db_stats(conn)
        stats["inserted_records"] = total
        stats["database_path"] = str(settings.database_path)
        stats["source_path"] = str(source_path)
        return stats
    finally:
        conn.close()


def _iter_jsonl_batches(path: Path, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
