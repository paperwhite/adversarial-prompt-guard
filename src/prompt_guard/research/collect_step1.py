"""Step-one dataset collection for the Adversarial Prompt Guard DB.

The collector intentionally uses only the Python standard library so the first
research bundle can be reproduced before the heavier ML stack is installed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

USER_AGENT = "adversarial-prompt-guard/0.1 defensive-research"

SOURCE_REGISTRY = Path("configs/research_sources.json")
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
NORMALIZED_JSONL = INTERIM_DIR / "adversarial_prompts_step1.jsonl"
MANIFEST_JSON = INTERIM_DIR / "manifest_step1.json"


@dataclass(frozen=True)
class SourceStat:
    source_id: str
    rows_seen: int = 0
    records_written: int = 0
    raw_files: tuple[str, ...] = ()
    notes: str = ""


def collect_step1(project_root: Path, limit_hf_rows: int | None = None) -> dict[str, Any]:
    project_root = project_root.resolve()
    registry_path = project_root / SOURCE_REGISTRY
    with registry_path.open("r", encoding="utf-8") as fh:
        registry = json.load(fh)

    raw_root = project_root / RAW_DIR
    interim_root = project_root / INTERIM_DIR
    raw_root.mkdir(parents=True, exist_ok=True)
    interim_root.mkdir(parents=True, exist_ok=True)

    collected_at = datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []
    stats: list[SourceStat] = []

    for source in registry["sources"]:
        kind = source["kind"]
        if kind == "csv_url":
            new_records, stat = _collect_csv_url(source, raw_root, collected_at)
        elif kind == "markdown_url":
            new_records, stat = _collect_markdown_url(source, raw_root, collected_at)
        elif kind == "hf_rows":
            new_records, stat = _collect_hf_rows(
                source,
                raw_root,
                collected_at,
                limit_hf_rows=limit_hf_rows,
            )
        elif kind == "reference_only":
            new_records, stat = [], SourceStat(
                source_id=source["id"],
                notes=source.get("notes", "Reference tracked for follow-up collection."),
            )
        else:
            raise ValueError(f"Unsupported source kind: {kind}")
        records.extend(new_records)
        stats.append(stat)

    deduped = _dedupe_records(records)
    jsonl_path = project_root / NORMALIZED_JSONL
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in deduped:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    grouped = _group_counts(deduped, "attack_type")
    categories = _group_counts(deduped, "category")
    manifest = {
        "collected_at": collected_at,
        "total_records": len(deduped),
        "pre_dedupe_records": len(records),
        "attack_type_counts": grouped,
        "category_counts": categories,
        "sources": [stat.__dict__ for stat in stats],
        "jsonl_path": str(jsonl_path),
        "manifest_path": str(project_root / MANIFEST_JSON),
        "registry_path": str(registry_path),
    }

    manifest_path = project_root / MANIFEST_JSON
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    _write_step1_note(project_root, manifest, registry)
    return manifest


def _collect_csv_url(
    source: dict[str, Any],
    raw_root: Path,
    collected_at: str,
) -> tuple[list[dict[str, Any]], SourceStat]:
    raw_path = _download_source_file(source, raw_root)
    text_field = source["text_field"]
    records: list[dict[str, Any]] = []
    rows_seen = 0
    with raw_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_idx, row in enumerate(reader):
            rows_seen += 1
            prompt_text = _normalize_prompt(row.get(text_field, ""))
            if not prompt_text:
                continue
            records.append(_record_from_prompt(source, prompt_text, collected_at, row_idx, row))
    return records, SourceStat(
        source_id=source["id"],
        rows_seen=rows_seen,
        records_written=len(records),
        raw_files=(str(raw_path),),
    )


def _collect_markdown_url(
    source: dict[str, Any],
    raw_root: Path,
    collected_at: str,
) -> tuple[list[dict[str, Any]], SourceStat]:
    raw_path = _download_source_file(source, raw_root)
    text = raw_path.read_text(encoding="utf-8")
    examples = _extract_markdown_payloads(text)
    records = [
        _record_from_prompt(source, prompt, collected_at, row_idx, {"extracted_from": "markdown"})
        for row_idx, prompt in enumerate(examples)
    ]
    return records, SourceStat(
        source_id=source["id"],
        rows_seen=len(examples),
        records_written=len(records),
        raw_files=(str(raw_path),),
        notes="Markdown source saved raw; extracted fenced/list payload-like examples when present.",
    )


def _collect_hf_rows(
    source: dict[str, Any],
    raw_root: Path,
    collected_at: str,
    limit_hf_rows: int | None,
) -> tuple[list[dict[str, Any]], SourceStat]:
    source_dir = raw_root / source["id"]
    source_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    raw_files: list[str] = []
    rows_seen = 0
    all_notes: list[str] = []

    for split in source["splits"]:
        offset = 0
        split_records: list[dict[str, Any]] = []
        max_rows = limit_hf_rows if limit_hf_rows is not None else split.get("limit")
        split_notes: list[str] = []
        while True:
            try:
                page = _fetch_hf_rows(
                    dataset=source["dataset"],
                    config=split["config"],
                    split=split["split"],
                    offset=offset,
                    length=100,
                )
            except (HTTPError, URLError, TimeoutError) as exc:
                split_notes.append(f"{split['config']}/{split['split']} offset {offset}: {exc}")
                break
            rows = page.get("rows", [])
            if not rows:
                break
            for wrapped in rows:
                rows_seen += 1
                row = wrapped["row"]
                prompt_text = _normalize_prompt(row.get(source["text_field"], ""))
                if not prompt_text:
                    continue
                if source.get("malicious_label_field") is not None:
                    label_field = source["malicious_label_field"]
                    if row.get(label_field) != source.get(
                        "malicious_label_value", 1
                    ) and not source.get("include_benign", False):
                        continue
                split_records.append(
                    _record_from_prompt(
                        source,
                        prompt_text,
                        collected_at,
                        wrapped["row_idx"],
                        row,
                        split_name=split["split"],
                        config_name=split["config"],
                    )
                )
                if max_rows is not None and len(split_records) >= max_rows:
                    break
            if max_rows is not None and len(split_records) >= max_rows:
                break
            offset += len(rows)
            if offset >= page.get("num_rows_total", offset):
                break
            time.sleep(0.25)

        raw_path = source_dir / f"{split['config']}__{split['split']}.json"
        with raw_path.open("w", encoding="utf-8") as fh:
            json.dump(split_records, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        raw_files.append(str(raw_path))
        all_records.extend(split_records)
        all_notes.extend(split_notes)

    return all_records, SourceStat(
        source_id=source["id"],
        rows_seen=rows_seen,
        records_written=len(all_records),
        raw_files=tuple(raw_files),
        notes="; ".join(all_notes),
    )


def _download_source_file(source: dict[str, Any], raw_root: Path) -> Path:
    source_dir = raw_root / source["id"]
    source_dir.mkdir(parents=True, exist_ok=True)
    filename = source.get("filename") or Path(urllib.parse.urlparse(source["url"]).path).name
    raw_path = source_dir / filename
    request = urllib.request.Request(source["url"], headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw_path.write_bytes(response.read())
    return raw_path


def _fetch_hf_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    last_error: HTTPError | URLError | TimeoutError | None = None
    for attempt in range(4):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code in {400, 401, 403, 404}:
                raise
            retry_after = 0.0
            if isinstance(exc, HTTPError):
                try:
                    retry_after = float(exc.headers.get("Retry-After", "0"))
                except ValueError:
                    retry_after = 0.0
            time.sleep(max(retry_after, 2.0 * (attempt + 1)))
    assert last_error is not None
    raise last_error


def _record_from_prompt(
    source: dict[str, Any],
    prompt_text: str,
    collected_at: str,
    source_row: int,
    raw_fields: dict[str, Any],
    split_name: str | None = None,
    config_name: str | None = None,
) -> dict[str, Any]:
    category = _normalize_category(raw_fields.get(source.get("category_field", ""), ""))
    if not category:
        category = source.get("default_category", "uncategorized")
    attack_type = _map_attack_type(category, source.get("default_attack_type", "injection"))
    severity = _map_severity(raw_fields.get(source.get("severity_field", ""), ""))
    record_id = _stable_id(source["id"], source_row, prompt_text)
    return {
        "id": record_id,
        "attack_type": attack_type,
        "category": category,
        "prompt_text": prompt_text,
        "severity_score": severity,
        "source_id": source["id"],
        "source_name": source["name"],
        "source_url": source.get("url") or source.get("dataset"),
        "license": source.get("license", "unknown"),
        "split": split_name,
        "config": config_name,
        "source_row": source_row,
        "variants": [],
        "collected_at": collected_at,
        "raw_fields": raw_fields,
    }


def _normalize_prompt(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u0000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_category(value: Any) -> str:
    text = _normalize_prompt(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _map_attack_type(category: str, default: str) -> str:
    benign = {
        "benign",
        "standard",
        "edge_case",
    }
    poisoning = {
        "indirect_injection",
        "embedding_manipulation",
        "data_poisoning",
        "rag_poisoning",
        "tool_output_poisoning",
    }
    evasion = {
        "jailbreak",
        "human_jailbreak",
        "human_jailbreaks",
        "harmful_behavior",
        "harmful_string",
        "encoding_obfuscation",
        "token_smuggling",
        "encoding",
        "persona_replacement",
        "many_shot",
        "crescendo",
        "context_overflow",
        "adversarial",
        "copyright",
        "training_extraction",
        "model_fingerprinting",
    }
    injection = {
        "direct_injection",
        "instruction_override",
        "prompt_leaking",
        "prompt_leak",
        "prompt_injection",
        "prompt_disclosure",
        "prompt_extraction",
        "system_extraction",
        "system_manipulation",
        "role_hijack",
        "data_exfiltration",
        "agent_manipulation",
        "response_manipulation",
        "output_manipulation",
        "payload_injection",
        "token_injection",
        "context_confusion",
        "contextual",
        "control",
        "multi_turn",
        "chain_of_thought",
        "code_execution",
    }
    if category in benign:
        return "benign"
    if category in poisoning:
        return "poisoning_prompt"
    if category in evasion:
        return "evasion"
    if category in injection:
        return "injection"
    return default


def _map_severity(value: Any) -> float:
    text = _normalize_category(value)
    if text == "critical":
        return 1.0
    if text == "high":
        return 0.8
    if text == "medium":
        return 0.55
    if text == "low":
        return 0.3
    return 0.5


def _extract_markdown_payloads(text: str) -> list[str]:
    examples: list[str] = []
    for match in re.finditer(r"```(?:[a-zA-Z0-9_-]+)?\n(.*?)```", text, flags=re.DOTALL):
        candidate = _normalize_prompt(match.group(1))
        if _looks_like_prompt_payload(candidate):
            examples.append(candidate)

    for line in text.splitlines():
        stripped = line.strip(" -*`\t")
        if _looks_like_prompt_payload(stripped):
            examples.append(_normalize_prompt(stripped))
    return list(dict.fromkeys(examples))


def _looks_like_prompt_payload(text: str) -> bool:
    if len(text) < 24 or len(text) > 2500:
        return False
    lowered = text.lower()
    signals = (
        "ignore",
        "instruction",
        "system prompt",
        "developer message",
        "reveal",
        "override",
        "do not follow",
        "forget",
        "prompt injection",
    )
    return any(signal in lowered for signal in signals)


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        key = hashlib.sha256(record["prompt_text"].casefold().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _stable_id(source_id: str, source_row: int, prompt_text: str) -> str:
    digest = hashlib.sha256(f"{source_id}:{source_row}:{prompt_text}".encode()).hexdigest()
    return f"apg_{digest[:16]}"


def _group_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get(field) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _write_step1_note(
    project_root: Path,
    manifest: dict[str, Any],
    registry: dict[str, Any],
) -> None:
    note_path = project_root / "docs" / "research_data_step1.md"
    source_lines = []
    for source in registry["sources"]:
        source_lines.append(
            f"- `{source['id']}`: {source['name']} ({source['kind']}, license: "
            f"{source.get('license', 'unknown')})"
        )

    attack_counts = "\n".join(
        f"- `{attack_type}`: {count}" for attack_type, count in manifest["attack_type_counts"].items()
    )
    note = f"""# Step 1: Research & Data

Collected at: `{manifest['collected_at']}`

## Current Bundle

- Normalized records: `{manifest['total_records']}`
- Pre-dedupe records: `{manifest['pre_dedupe_records']}`
- Normalized JSONL: `{manifest['jsonl_path']}`
- Manifest: `{manifest['manifest_path']}`

## Attack Type Counts

{attack_counts}

## Sources Tracked

{chr(10).join(source_lines)}

## Notes

- AdvBench and HarmBench are collected as direct harmful/evasion prompts with source provenance.
- OWASP AI Testing Guide pages are saved as raw Markdown references; prompt-like examples are extracted when present.
- Neuralchemy Prompt Injection Dataset contributes labeled injection, jailbreak, extraction, smuggling, and benign hard-negative examples.
- HackAPrompt is tracked as a high-value paper/dataset source, but the public row API currently requires authorization for direct row extraction in this environment. Use the registry entry for follow-up authenticated collection or full parquet/arrow processing.
- X scraping is not automated in this first pass because it requires account/session access and a separate legal/ToS review. Treat it as a curated intake task, not a blind scraper.

## Next Step

Use this JSONL as the seed input for the Week 2 schema. The record fields already mirror the planned DB columns: `id`, `attack_type`, `prompt_text`, `severity_score`, and `variants`.
"""
    note_path.write_text(note, encoding="utf-8")


if __name__ == "__main__":
    collect_step1(Path.cwd())
