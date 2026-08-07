"""Hybrid adversarial prompt detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prompt_guard.db import connect, load_embedded_records, search_fts
from prompt_guard.embeddings import SentenceEmbeddingModel
from prompt_guard.settings import GuardSettings, load_settings

REGEX_PATTERNS: tuple[tuple[str, str, float], ...] = (
    (
        "instruction_override",
        r"\b(ignore|disregard|forget|override)\b.{0,90}\b(previous|prior|above|system|developer|instruction|rules?)\b",
        0.86,
    ),
    (
        "system_prompt_extraction",
        r"\b(reveal|print|show|repeat|dump|exfiltrate)\b.{0,90}\b(system prompt|developer message|hidden instructions|initial prompt)\b",
        0.9,
    ),
    (
        "role_hijack",
        r"\b(you are now|act as|pretend to be|simulate)\b.{0,80}\b(dan|jailbreak|unfiltered|no restrictions|developer mode)\b",
        0.84,
    ),
    (
        "tool_or_data_exfiltration",
        r"\b(send|post|upload|forward|email|copy)\b.{0,100}\b(secret|token|api key|password|credential|private data)\b",
        0.86,
    ),
    (
        "indirect_prompt_injection",
        r"\b(when.*assistant|when.*ai|for any ai|for the llm)\b.{0,120}\b(ignore|override|reveal|exfiltrate)\b",
        0.8,
    ),
)


@dataclass(frozen=True)
class Match:
    record_id: str
    score: float
    attack_type: str
    category: str
    prompt_text: str
    source_name: str
    severity_score: float


class GuardDetector:
    """Loads the embedded corpus once and evaluates prompts locally."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.settings: GuardSettings = load_settings(self.project_root)
        self.embedder = SentenceEmbeddingModel(
            self.settings.embedding_model,
            cache_folder=self.settings.model_cache_path,
        )
        self._records: list[dict[str, Any]] | None = None
        self._matrix: np.ndarray | None = None

    def detect(self, text: str, top_k: int | None = None) -> dict[str, Any]:
        top_k = top_k or self.settings.default_top_k
        regex_hits = _regex_hits(text)
        semantic_matches = self._semantic_matches(text, top_k=top_k)
        fts_matches = self._fts_matches(text, top_k=top_k)
        heuristic = _heuristic_fallback_score(text)

        best_semantic = semantic_matches[0].score if semantic_matches else 0.0
        best_regex = max((hit["score"] for hit in regex_hits), default=0.0)
        best_score = max(best_semantic, best_regex, heuristic["score"])
        reasons: list[str] = []
        if best_semantic >= self.settings.similarity_threshold:
            reasons.append("semantic_similarity")
        if best_regex >= self.settings.regex_threshold:
            reasons.append("regex_pattern")
        if heuristic["score"] >= self.settings.regex_threshold:
            reasons.append("heuristic_fallback")

        return {
            "flagged": bool(reasons),
            "score": round(float(best_score), 4),
            "threshold": self.settings.similarity_threshold,
            "reasons": reasons,
            "model": self.settings.embedding_model,
            "regex_hits": regex_hits,
            "heuristic_fallback": heuristic,
            "semantic_matches": [match.__dict__ for match in semantic_matches],
            "fts_matches": fts_matches,
        }

    def stats(self) -> dict[str, Any]:
        records, matrix = self._load_records()
        return {
            "database_path": str(self.settings.database_path),
            "embedding_model": self.settings.embedding_model,
            "embedding_dimension": self.settings.embedding_dimension,
            "model_cache_path": str(self.settings.model_cache_path),
            "loaded_records": len(records),
            "matrix_shape": list(matrix.shape),
        }

    def _semantic_matches(self, text: str, top_k: int) -> list[Match]:
        records, matrix = self._load_records()
        if not records:
            return []
        query = self.embedder.encode([text], batch_size=1)[0]
        scores = matrix @ query
        candidate_count = min(top_k, scores.shape[0])
        if candidate_count == 0:
            return []
        top_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
        sorted_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        matches: list[Match] = []
        for idx in sorted_indices:
            record = records[int(idx)]
            matches.append(
                Match(
                    record_id=record["id"],
                    score=round(float(scores[int(idx)]), 4),
                    attack_type=record["attack_type"],
                    category=record["category"],
                    prompt_text=record["prompt_text"],
                    source_name=record["source_name"],
                    severity_score=float(record["severity_score"]),
                )
            )
        return matches

    def _fts_matches(self, text: str, top_k: int) -> list[dict[str, Any]]:
        conn = connect(self.settings.database_path)
        try:
            matches = search_fts(conn, text, top_k)
        finally:
            conn.close()
        for match in matches:
            match["rank"] = round(float(match["rank"]), 4)
        return matches

    def _load_records(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        if self._records is not None and self._matrix is not None:
            return self._records, self._matrix

        conn = connect(self.settings.database_path)
        try:
            records = load_embedded_records(conn, self.settings.embedding_model)
        finally:
            conn.close()

        if records:
            matrix = np.vstack([record["embedding"] for record in records]).astype(np.float32)
        else:
            matrix = np.empty((0, self.settings.embedding_dimension), dtype=np.float32)
        self._records = records
        self._matrix = matrix
        return records, matrix


def _regex_hits(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for name, pattern, score in REGEX_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            hits.append(
                {
                    "name": name,
                    "score": score,
                    "snippet": re.sub(r"\s+", " ", match.group(0)).strip()[:220],
                }
            )
    return hits


def _heuristic_fallback_score(text: str) -> dict[str, Any]:
    lowered = text.lower()
    signals = {
        "override_language": any(
            phrase in lowered
            for phrase in ("ignore previous", "disregard previous", "developer mode", "no restrictions")
        ),
        "secret_extraction": any(
            phrase in lowered
            for phrase in ("system prompt", "hidden instruction", "api key", "private credential")
        ),
        "role_reassignment": any(
            phrase in lowered for phrase in ("you are now", "act as dan", "pretend to be")
        ),
        "exfiltration_verb": any(
            phrase in lowered for phrase in ("send to", "post to", "upload to", "copy the secret")
        ),
    }
    score = min(0.95, 0.28 * sum(1 for hit in signals.values() if hit))
    return {
        "score": round(score, 4),
        "signals": [name for name, hit in signals.items() if hit],
    }
