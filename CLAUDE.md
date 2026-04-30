# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Digest is a self-hosted, LLM-powered AI news aggregator. It fetches 25+ RSS feeds, scores articles for relevance using a two-stage LLM pipeline, and delivers curated HTML email digests.

## Commands

```bash
# Install dependencies
pip install -r src/requirements.txt

# Full run (fetch, score, email)
python src/ai_digest.py -c src/config.local.yaml

# Preview without sending email (outputs digest_preview.html)
python src/ai_digest.py -c src/config.local.yaml --dry-run

# Test SMTP configuration
python src/ai_digest.py -c src/config.local.yaml --test-email

# View statistics and history
python src/ai_digest.py -c src/config.local.yaml --stats
python src/ai_digest.py -c src/config.local.yaml --history
python src/ai_digest.py -c src/config.local.yaml --show-digest <ID>

# Reset database (re-fetches and rescores everything)
python src/ai_digest.py -c src/config.local.yaml --reset-db
```

Configuration is always passed via `-c`. The template is `src/config.yaml`; local overrides go in `src/config.local.yaml` (gitignored).

```bash
# Web UI (config editor) – run alongside the cron job
python src/web_ui.py -c src/config.local.yaml
python src/web_ui.py -c src/config.local.yaml --host 0.0.0.0 --port 5000  # LAN access
```

## Architecture

The entire application is a single file: `src/ai_digest.py` (~1,270 lines). There are no modules or packages.

### Two-Stage LLM Pipeline

```
RSS Feeds → Fetch & Dedup (SQLite) → Stage 1: Fast Filter (cheap model, all articles)
→ Stage 2: Summarize (smart model, top N only) → SQLite → HTML Email via SMTP
```

- **Stage 1** (`stage1_filter()`): Batches 10 articles at a time, scores 0–10, applies topic weights, falls back to keyword matching if LLM unavailable.
- **Stage 2** (`stage2_summarize()`): Generates headline + summary for the top N articles only.

### LLM Backends

- `OllamaClient`: Local Ollama at `http://localhost:11434`
- `OpenRouterClient`: Cloud API; API key read from environment variable `OPENROUTER_API_KEY` or `src/openrouter.txt` (gitignored)
- `create_llm_client()`: Factory function that selects backend based on config `llm.backend` field (`"ollama"` or `"openrouter"`)

Two separate clients can be configured: one for filtering (`llm.filter_model`) and one for summarization (`llm.summarize_model`).

### Database (`ArticleDB`)

SQLite at `~/.ai-digest/articles.db` (auto-created). Two tables:

- `articles`: URL hash PK, content, LLM scores/summaries, metadata. Retention: 365 days (scored), 90 days (unscored).
- `digests`: Full HTML archives + metadata. Retention: 90 days.

Schema migrations run automatically on startup.

### Key Config Fields

| Field | Purpose |
|---|---|
| `llm.backend` | `"ollama"` or `"openrouter"` |
| `llm.top_n` | Articles sent to Stage 2 (default: 15) |
| `llm.min_relevance` | Minimum score to include (default: 4) |
| `feeds[].category` | Used for color-coding in HTML output |
| `topics[].weight` | Multiplier applied to Stage 1 scores |
