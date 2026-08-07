"""Command-line entry points for Adversarial Prompt Guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompt_guard.db import connect, db_stats
from prompt_guard.detector import GuardDetector
from prompt_guard.ingest import build_database
from prompt_guard.research.collect_step1 import collect_step1
from prompt_guard.settings import load_settings


def app() -> None:
    parser = argparse.ArgumentParser(
        prog="prompt-guard-data",
        description="Collect, embed, store, and run adversarial prompt detection.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing configs/ and data/ directories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect step-one research data.")
    collect_parser.add_argument(
        "--limit-hf-rows",
        type=int,
        default=None,
        help="Optional per-split row limit for quick smoke tests.",
    )

    build_parser = subparsers.add_parser("build-db", help="Populate the SQLite guard DB.")
    build_parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional normalized JSONL path. Defaults to data/interim/adversarial_prompts_step1.jsonl.",
    )
    build_parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Append/update records instead of clearing existing prompt tables first.",
    )

    detect_parser = subparsers.add_parser("detect", help="Run local detection for one prompt.")
    detect_parser.add_argument("text", help="Prompt text to evaluate.")
    detect_parser.add_argument("--top-k", type=int, default=None, help="Number of matches to return.")

    subparsers.add_parser("stats", help="Show database stats.")

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI service.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    project_root = args.project_root.resolve()
    if args.command == "collect":
        manifest = collect_step1(project_root, limit_hf_rows=args.limit_hf_rows)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "build-db":
        stats = build_database(project_root, jsonl_path=args.jsonl, reset=not args.no_reset)
        print(json.dumps(stats, indent=2, sort_keys=True))
    elif args.command == "detect":
        result = GuardDetector(project_root).detect(args.text, top_k=args.top_k)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "stats":
        settings = load_settings(project_root)
        conn = connect(settings.database_path)
        try:
            stats = db_stats(conn)
        finally:
            conn.close()
        print(json.dumps(stats, indent=2, sort_keys=True))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run(
            "prompt_guard.api:app",
            host=args.host,
            port=args.port,
            reload=False,
            app_dir=str(project_root / "src"),
        )


if __name__ == "__main__":
    app()
