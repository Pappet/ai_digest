# Project Overview

## What it is?

AI Digest is an automated, self-hosted system that curates and summarizes RSS feeds using Large Language Models (LLMs) to create highly relevant, personalized news digests delivered straight to the user's email.

## Project Stats

* **Language:** Python 3
* **Application Type:** CLI Application & Background Cron Job (with Flask Web UI)
* **Lines of Code:** ~1,270 (Main Application `ai_digest.py`)
* **Storage:** SQLite (`~/.ai-digest/articles.db`)
* **Data Retention:** 365 days (scored articles), 90 days (unscored and digest archives)

## Architectural Decisions

* **Monolithic Script:** The core application logic resides entirely in a single file (`ai_digest.py`) without nested modules/packages. This ensures simple deployment and minimal boilerplate.
* **Two-Stage Pipeline:** To balance cost, speed, and quality, a two-stage LLM evaluation strategy was adopted. A cheap model scores all articles first, allowing the more expensive/smarter model to only run on the top-N articles.
* **Abstract LLM Clients:** Support for both local inference (Ollama) and cloud APIs (OpenRouter) is abstracted behind standard client interfaces, making the platform backend-agnostic.
* **Local Database:** SQLite is used for persistent caching, deduplication, and retention management without requiring a heavy, external database setup.

## Detailed Architecture

### Pipeline Flow

```mermaid
flowchart LR
    A[RSS Feeds] --> B[Fetch & Dedup (SQLite)]
    B --> C[Stage 1: Fast Filter (cheap model)]
    C --> D[Stage 2: Summarize (smart model, top N)]
    D --> E[Store in SQLite]
    E --> F[HTML Email via SMTP]
```

* **Stage 1 (`stage1_filter()`):** Batches 10 articles at a time, scores them from 0–10, applies pre-defined topic weights from the configuration, and falls back to simple keyword matching if the LLM backend is unavailable.
* **Stage 2 (`stage2_summarize()`):** Takes only the top N articles from Stage 1 and generates a concise headline and summary for each.

## Source Files Description

* **`src/ai_digest.py`**: The core application logic containing RSS parsing, LLM pipeline execution, SQLite interaction, and email delivery.
* **`src/web_ui.py`**: A Flask-based web application to edit configurations and topics locally.
* **`src/config.yaml`**: The baseline configuration template.
* **`src/run_ai_digest.sh`**: A shell script helper to run the main application within its virtual environment and with necessary environment variables.
* **`src/requirements.txt`**: The exact Python dependencies required for the project.

## Dependencies and their purpose

All dependencies are locked to exact versions to guarantee reproducible builds.

| Dependency | Purpose |
| :--- | :--- |
| `feedparser` | Parsing and standardizing the different RSS feeds. |
| `requests` | Making HTTP requests to the LLM APIs and RSS feeds. |
| `PyYAML` | Reading and parsing the configuration files. |
| `Flask` | Serving the local web UI config editor. |
| `python-dotenv` | Loading environment variables from a `.env` file (e.g., API keys). |

## Additional References

* [Ollama Local LLM](https://ollama.com/)
* [OpenRouter API](https://openrouter.ai/)
