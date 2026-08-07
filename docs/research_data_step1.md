# Step 1: Research & Data

Collected at: `2026-08-04T17:44:07.493825+00:00`

## Current Bundle

- Normalized records: `5805`
- Pre-dedupe records: `5832`
- Normalized JSONL: `/Users/paperwhite/Documents/Shiba-2026/adversarial-prompt-guard/adversarial-prompt-guard/data/interim/adversarial_prompts_step1.jsonl`
- Manifest: `/Users/paperwhite/Documents/Shiba-2026/adversarial-prompt-guard/adversarial-prompt-guard/data/interim/manifest_step1.json`

## Attack Type Counts

- `evasion`: 3449
- `injection`: 1603
- `benign`: 695
- `poisoning_prompt`: 58

## Sources Tracked

- `advbench_harmful_behaviors`: AdvBench harmful behaviors (csv_url, license: research)
- `advbench_harmful_strings`: AdvBench harmful strings (csv_url, license: research)
- `harmbench_text_all`: HarmBench text behavior set (csv_url, license: MIT)
- `harmbench_human_jailbreaks`: HarmBench HumanJailbreaks (hf_rows, license: MIT)
- `neuralchemy_prompt_injection_core`: Prompt Injection & Jailbreak Detection Dataset (hf_rows, license: Apache-2.0)
- `owasp_aitg_prompt_injection`: OWASP AI Testing Guide: Prompt Injection (markdown_url, license: CC-BY-SA-4.0)
- `owasp_aitg_indirect_prompt_injection`: OWASP AI Testing Guide: Indirect Prompt Injection (markdown_url, license: CC-BY-SA-4.0)
- `owasp_aitg_prompt_disclosure`: OWASP AI Testing Guide: Prompt Disclosure (markdown_url, license: CC-BY-SA-4.0)
- `owasp_aitg_embedding_manipulation`: OWASP AI Testing Guide: Embedding Manipulation (markdown_url, license: CC-BY-SA-4.0)
- `hackaprompt_dataset`: HackAPrompt public competition dataset (reference_only, license: CC-BY-4.0)

## Notes

- AdvBench and HarmBench are collected as direct harmful/evasion prompts with source provenance.
- OWASP AI Testing Guide pages are saved as raw Markdown references; prompt-like examples are extracted when present.
- Neuralchemy Prompt Injection Dataset contributes labeled injection, jailbreak, extraction, smuggling, and benign hard-negative examples.
- HackAPrompt is tracked as a high-value paper/dataset source, but the public row API currently requires authorization for direct row extraction in this environment. Use the registry entry for follow-up authenticated collection or full parquet/arrow processing.
- X scraping is not automated in this first pass because it requires account/session access and a separate legal/ToS review. Treat it as a curated intake task, not a blind scraper.

## Next Step

Use this JSONL as the seed input for the Week 2 schema. The record fields already mirror the planned DB columns: `id`, `attack_type`, `prompt_text`, `severity_score`, and `variants`.
