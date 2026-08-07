# Steps 2-4: DB, Ingestion, and Detection Engine

## Current Status

- Step 1 data is sufficient to begin implementation: `5,805` normalized records.
- The SQLite guard DB has been populated with `5,805` records and `5,805` embeddings.
- The detection engine combines normalized-vector cosine similarity, SQLite FTS5 text search, regex patterns, and a local heuristic fallback.
- The FastAPI service exposes `GET /health` and `POST /detect`.

## Current Roadblocks From Step 1

- Hugging Face rate limits can interrupt full row collection for the NeurAlchemy prompt-injection dataset without an authenticated token.
- HackAPrompt is tracked but not fully ingested because the public row API returned authorization errors in this environment.
- X collection is intentionally not automated yet. It needs account/session access, ToS/legal review, and a curated intake process.

## Database Choice

SQLite is the project artifact for the first working guard. It is simple to inspect, version around, ship locally, and use in a Mac-hosted middleware prototype. The schema stores source provenance, attack grouping, prompt text, FTS rows, and model-specific embedding blobs.

Postgres is still a good later move when the project needs concurrent writes, hosted deployment, log ingestion, queue workers, or a vector extension such as pgvector.

## Embedding Model

The first model is `sentence-transformers/all-MiniLM-L6-v2`.

Reasons:

- It produces 384-dimensional embeddings, keeping the SQLite artifact compact.
- It is small and fast enough for CPU inference on a Mac.
- It has good general-purpose semantic similarity quality.
- It keeps local setup much lighter than larger 768-dimensional models such as `all-mpnet-base-v2`.

The schema stores `model_name` and `dimension`, so a later rebuild can compare MiniLM against a larger model without redesigning the database.

## Detection Behavior

Flagging currently happens when any of these are true:

- Top semantic match is at or above the configured similarity threshold, currently `0.8`.
- Regex pattern score is at or above the configured regex threshold, currently `0.72`.
- Local heuristic fallback reaches the regex threshold.

The local heuristic is not a replacement for a real LLM scorer. It is a deterministic placeholder for Step 4 so the API has a fallback path without requiring API keys or sending prompts to an external service.
