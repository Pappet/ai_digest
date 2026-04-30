# AI Digest

## Project Overview

AI Digest is a self-hosted, LLM-powered AI news aggregator. It is designed to automatically fetch 25+ RSS feeds, score the articles for relevance using a specialized two-stage LLM pipeline, and finally deliver the top curated articles directly as an HTML email digest.

## Features

* **RSS Aggregation:** Fetches and deduplicates content from multiple RSS feeds.
* **Two-Stage LLM Pipeline:**
  * **Stage 1 (Fast Filter):** Uses a fast/cheap LLM model to score all articles (0-10) and filter out noise. Applies topic weighting and falls back to keyword matching if the LLM is unavailable.
  * **Stage 2 (Summarize):** Uses a smarter LLM model to generate concise headlines and summaries for the top 'N' selected articles.
* **Flexible LLM Backends:** Supports running locally via Ollama or through the cloud via OpenRouter API.
* **HTML Email Delivery:** Automatically sends the curated digest to your inbox via SMTP.
* **Web UI Config Editor:** Includes a web interface to easily edit your configuration and topics.

## Quickstart Guide

### Prerequisites

* Python 3.10+
* (Optional) Ollama running locally or an OpenRouter API Key.

### Installation

1. Clone the repository and navigate to the project directory:

    ```bash
    git clone <repository_url>
    cd ai_digest
    ```

2. Create a virtual environment and install dependencies:

    ```bash
    python -m venv src/venv
    source src/venv/bin/activate
    pip install -r src/requirements.txt
    ```

### Configuration

1. Copy the template configuration:

    ```bash
    cp src/config.yaml src/config.local.yaml
    ```

2. Edit `src/config.local.yaml` with your preferred feeds, topics, and SMTP settings.
3. Set your OpenRouter API key in an `.env` file or `openrouter.txt` if using the OpenRouter backend.

### Running AI Digest

* **Full run (fetch, score, email):**

    ```bash
    python src/ai_digest.py -c src/config.local.yaml
    ```

* **Preview run (creates `digest_preview.html` without sending email):**

    ```bash
    python src/ai_digest.py -c src/config.local.yaml --dry-run
    ```

* **Start the Web UI Config Editor:**

    ```bash
    python src/web_ui.py -c src/config.local.yaml
    ```

## Contribution

Contributions are welcome! Please ensure that any feature additions are accompanied by updates to this `README.md` and the `PROJECT_OVERVIEW.md`.
All changes must go through a Pull Request to the `main` branch. Direct commits to `main` are not allowed.
Ensure dependencies in `src/requirements.txt` are locked to exact versions to maintain reproducible builds.

## License

This project is licensed under the MIT License. Please refer to the `LICENSE` file for more details.
