"""FastAPI detection service."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from prompt_guard.db import connect, db_stats
from prompt_guard.detector import GuardDetector
from prompt_guard.settings import load_settings


class DetectRequest(BaseModel):
    text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)


class DetectResponse(BaseModel):
    flagged: bool
    score: float
    threshold: float
    reasons: list[str]
    model: str
    regex_hits: list[dict]
    heuristic_fallback: dict
    semantic_matches: list[dict]
    fts_matches: list[dict]


def create_app(project_root: Path | None = None) -> FastAPI:
    root = (project_root or Path.cwd()).resolve()
    app = FastAPI(title="Adversarial Prompt Guard", version="0.1.0")

    @lru_cache(maxsize=1)
    def detector() -> GuardDetector:
        return GuardDetector(root)

    @app.get("/health")
    def health() -> dict:
        settings = load_settings(root)
        conn = connect(settings.database_path)
        try:
            stats = db_stats(conn)
        finally:
            conn.close()
        return {"ok": True, **stats}

    @app.post("/detect", response_model=DetectResponse)
    def detect(request: DetectRequest) -> dict:
        return detector().detect(request.text, top_k=request.top_k)

    return app


app = create_app()
