"""SQLite storage for prompt records, FTS, and local embeddings."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prompt_records (
            id TEXT PRIMARY KEY,
            attack_type TEXT NOT NULL,
            category TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            severity_score REAL NOT NULL,
            variants_json TEXT NOT NULL DEFAULT '[]',
            source_id TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_url TEXT,
            source_license TEXT,
            source_split TEXT,
            source_config TEXT,
            source_row INTEGER,
            raw_fields_json TEXT NOT NULL DEFAULT '{}',
            collected_at TEXT,
            inserted_at TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS prompt_records_fts USING fts5(
            id UNINDEXED,
            attack_type,
            category,
            prompt_text,
            source_name,
            tokenize = 'porter unicode61'
        );

        CREATE TABLE IF NOT EXISTS prompt_embeddings (
            record_id TEXT PRIMARY KEY REFERENCES prompt_records(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            embedded_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_prompt_records_attack_type
            ON prompt_records(attack_type);
        CREATE INDEX IF NOT EXISTS idx_prompt_records_category
            ON prompt_records(category);
        CREATE INDEX IF NOT EXISTS idx_prompt_embeddings_model
            ON prompt_embeddings(model_name);
        """
    )
    set_metadata(conn, "schema_version", str(SCHEMA_VERSION))


def reset_prompt_tables(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM prompt_embeddings")
    conn.execute("DELETE FROM prompt_records_fts")
    conn.execute("DELETE FROM prompt_records")


def set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def insert_prompt_batch(
    conn: sqlite3.Connection,
    records: Iterable[dict[str, Any]],
    embeddings: np.ndarray,
    model_name: str,
) -> int:
    now = datetime.now(UTC).isoformat()
    count = 0
    for record, embedding in zip(records, embeddings, strict=True):
        variants_json = json.dumps(record.get("variants", []), ensure_ascii=False, sort_keys=True)
        raw_fields_json = json.dumps(
            record.get("raw_fields", {}),
            ensure_ascii=False,
            sort_keys=True,
        )
        conn.execute(
            """
            INSERT INTO prompt_records(
                id, attack_type, category, prompt_text, severity_score, variants_json,
                source_id, source_name, source_url, source_license, source_split,
                source_config, source_row, raw_fields_json, collected_at, inserted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                attack_type = excluded.attack_type,
                category = excluded.category,
                prompt_text = excluded.prompt_text,
                severity_score = excluded.severity_score,
                variants_json = excluded.variants_json,
                source_id = excluded.source_id,
                source_name = excluded.source_name,
                source_url = excluded.source_url,
                source_license = excluded.source_license,
                source_split = excluded.source_split,
                source_config = excluded.source_config,
                source_row = excluded.source_row,
                raw_fields_json = excluded.raw_fields_json,
                collected_at = excluded.collected_at,
                inserted_at = excluded.inserted_at
            """,
            (
                record["id"],
                record["attack_type"],
                record["category"],
                record["prompt_text"],
                float(record["severity_score"]),
                variants_json,
                record["source_id"],
                record["source_name"],
                record.get("source_url"),
                record.get("license"),
                record.get("split"),
                record.get("config"),
                record.get("source_row"),
                raw_fields_json,
                record.get("collected_at"),
                now,
            ),
        )
        conn.execute("DELETE FROM prompt_records_fts WHERE id = ?", (record["id"],))
        conn.execute(
            """
            INSERT INTO prompt_records_fts(id, attack_type, category, prompt_text, source_name)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["attack_type"],
                record["category"],
                record["prompt_text"],
                record["source_name"],
            ),
        )
        embedding_array = np.asarray(embedding, dtype=np.float32)
        conn.execute(
            """
            INSERT INTO prompt_embeddings(record_id, model_name, dimension, embedding, embedded_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(record_id) DO UPDATE SET
                model_name = excluded.model_name,
                dimension = excluded.dimension,
                embedding = excluded.embedding,
                embedded_at = excluded.embedded_at
            """,
            (
                record["id"],
                model_name,
                int(embedding_array.shape[0]),
                embedding_array.tobytes(),
                now,
            ),
        )
        count += 1
    return count


def load_embedded_records(conn: sqlite3.Connection, model_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            r.id,
            r.attack_type,
            r.category,
            r.prompt_text,
            r.severity_score,
            r.source_id,
            r.source_name,
            r.source_url,
            r.source_license,
            e.dimension,
            e.embedding
        FROM prompt_records r
        JOIN prompt_embeddings e ON e.record_id = r.id
        WHERE e.model_name = ?
        """,
        (model_name,),
    ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        embedding_blob = record.pop("embedding")
        record["embedding"] = np.frombuffer(embedding_blob, dtype=np.float32)
        records.append(record)
    return records


def search_fts(conn: sqlite3.Connection, query: str, limit: int) -> list[dict[str, Any]]:
    escaped = _fts_query(query)
    if not escaped:
        return []
    rows = conn.execute(
        """
        SELECT
            r.id,
            r.attack_type,
            r.category,
            r.prompt_text,
            r.severity_score,
            r.source_name,
            bm25(prompt_records_fts) AS rank
        FROM prompt_records_fts
        JOIN prompt_records r ON r.id = prompt_records_fts.id
        WHERE prompt_records_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (escaped, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def db_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    total = conn.execute("SELECT COUNT(*) AS count FROM prompt_records").fetchone()["count"]
    embedded = conn.execute("SELECT COUNT(*) AS count FROM prompt_embeddings").fetchone()["count"]
    attack_rows = conn.execute(
        """
        SELECT attack_type, COUNT(*) AS count
        FROM prompt_records
        GROUP BY attack_type
        ORDER BY count DESC, attack_type
        """
    ).fetchall()
    model_rows = conn.execute(
        """
        SELECT model_name, dimension, COUNT(*) AS count
        FROM prompt_embeddings
        GROUP BY model_name, dimension
        ORDER BY count DESC
        """
    ).fetchall()
    return {
        "total_records": total,
        "embedded_records": embedded,
        "attack_type_counts": {row["attack_type"]: row["count"] for row in attack_rows},
        "embedding_models": [dict(row) for row in model_rows],
    }


def _fts_query(text: str) -> str:
    terms = [term for term in text.replace('"', " ").split() if len(term) >= 3]
    return " OR ".join(f'"{term}"' for term in terms[:12])
