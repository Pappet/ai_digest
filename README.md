# 🤖 AI Digest

A self-hosted, LLM-powered AI news aggregator. Fetches 25+ RSS feeds, scores articles for relevance using a two-stage LLM pipeline (Ollama or OpenRouter), and delivers a curated HTML email digest. All LLM outputs are persisted in SQLite for history, trend analysis, and digest re-generation.

## How It Works

```
RSS Feeds (25+ sources)
        │
        ▼
   Fetch & Dedup (SQLite)
        │
        ▼
   Stage 1: Fast Filter          ← cheap model (Gemini Flash / qwen3:8b)
   Score 0-10 per article           processes ALL articles
        │
        ▼
   Stage 2: Summarize             ← smart model (Claude Sonnet / qwen3:30b)
   Top N articles only              writes headline + summary
        │
        ▼
   Persist to SQLite              → scores, summaries, topics, models used
        │
        ▼
   HTML Email Digest              → SMTP to your inbox (+ archived in DB)
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- **Either** [Ollama](https://ollama.ai) running locally **or** an [OpenRouter](https://openrouter.ai) API key
- SMTP access (Gmail app password, Postfix, etc.)

### 2. Install

```bash
git clone <your-repo> && cd ai-digest
pip install -r requirements.txt
```

### 3a. Setup – OpenRouter (cloud, no GPU needed)

```bash
# Get an API key at https://openrouter.ai/keys
export OPENROUTER_API_KEY="sk-or-v1-..."
```

In `config.yaml`:
```yaml
llm:
  backend: "openrouter"
  filter_model: "google/gemini-flash-1.5"       # ~$0.00001/article
  summary_model: "anthropic/claude-sonnet-4"     # ~$0.003/article
```

Typical daily cost: **< $0.10** for 15 summarized articles.

### 3b. Setup – Ollama (local, private)

```bash
ollama pull qwen3:8b       # Fast filter model
ollama pull qwen3:30b      # Summary model (or whatever fits your GPU)
```

In `config.yaml`:
```yaml
llm:
  backend: "ollama"
  base_url: "http://localhost:11434"
  filter_model: "qwen3:8b"
  summary_model: "qwen3:30b"
```

### 4. Configure

```bash
cp config.yaml config.local.yaml
# Edit config.local.yaml:
#   - Set your SMTP credentials
#   - Set recipient email
#   - Adjust Ollama URL if not localhost
#   - Tweak topics, weights, feeds
```

### 5. Test Email & First Run

```bash
# Verify SMTP settings first
python ai_digest.py -c config.local.yaml --test-email

# Preview without sending email
python ai_digest.py -c config.local.yaml --dry-run

# Check the generated preview
open digest_preview.html
```

### 6. Schedule (Cron)

```bash
crontab -e

# Run daily at 7:00 AM
0 7 * * * cd /path/to/ai-digest && /usr/bin/python3 ai_digest.py -c config.local.yaml >> /var/log/ai-digest.log 2>&1
```

## CLI Reference

```
python ai_digest.py [OPTIONS]
```

### Core Commands

| Flag | Description |
|------|-------------|
| `-c, --config PATH` | Config file path (default: `config.yaml`) |
| `--dry-run` | Generate digest, save as `digest_preview.html`, don't send email |
| `--stdout` | Print HTML digest to stdout (pipe-friendly) |
| `--reset-db` | Clear the article database before running (reprocess all articles) |

### Testing & Diagnostics

| Flag | Description |
|------|-------------|
| `--test-email` | Send a test email to verify SMTP settings, then exit |
| `--stats` | Show article scores, topic distribution, and digest history overview |

### Digest Archive

| Flag | Description |
|------|-------------|
| `--history` | List all archived digests with date, article count, and status |
| `--show-digest ID` | Export an archived digest as HTML file (`digest_<ID>.html`) |

### Usage Examples

```bash
# Daily run
python ai_digest.py -c config.local.yaml

# First run or after config changes – reprocess all articles
python ai_digest.py -c config.local.yaml --dry-run --reset-db

# Check what's been captured over time
python ai_digest.py -c config.local.yaml --stats

# Browse past digests
python ai_digest.py -c config.local.yaml --history
python ai_digest.py -c config.local.yaml --show-digest 7
open digest_7.html

# Test email before going to cron
python ai_digest.py -c config.local.yaml --test-email
```

## Feed Sources (25 feeds)

The default config includes feeds across six categories:

| Category | Sources |
|----------|---------|
| **Industry** | OpenAI Blog, Anthropic (News/Engineering/Research via community RSS), Google AI, Microsoft AI, Meta (Engineering/Research), NVIDIA, Apple ML, Hugging Face |
| **News** | TechCrunch AI, The Verge AI, Ars Technica, MIT Technology Review, VentureBeat AI |
| **Research** | ArXiv cs.AI, ArXiv cs.LG |
| **Community** | r/LocalLLaMA, r/MachineLearning, Hacker News (AI) |
| **Asia / China** | Hacker News (China AI), r/LocalLLaMA (China), South China Morning Post Tech, TechNode |

> **Note:** Anthropic and Meta AI don't offer official RSS feeds. The config uses community-maintained feeds from [Olshansk/rss-feeds](https://github.com/Olshansk/rss-feeds) (hourly updates) for Anthropic, and Meta's Engineering/Research blogs which do have feeds.

## Topic Profiles

Articles are scored against configurable topic profiles. Each topic has a weight (0.0–1.0) that boosts matching articles in the final ranking:

| Topic | Weight | What it catches |
|-------|--------|----------------|
| **Local & Open-Source LLMs** | 1.0 | Ollama, llama.cpp, GGUF, quantization, open weights, vLLM |
| **AI Research & Science** | 1.0 | Papers, architectures, scaling laws, reasoning, RLHF, interpretability |
| **AI aus Asien & China** | 0.9 | Qwen, DeepSeek, MiniMax, Baichuan, Yi, Alibaba, ByteDance, SenseTime |
| **AI Dev Tools & Coding Agents** | 0.9 | Claude Code, Cursor, MCP, tool use, agentic coding, code generation |
| **Major Breakthroughs & Milestones** | 0.8 | New model releases, benchmarks, frontier models, AGI |
| **Big Tech AI News** | 0.7 | Google, Microsoft, Apple, Meta, Amazon, NVIDIA, OpenAI, Anthropic |

Add your own topics in `config.yaml`:

```yaml
topics:
  - name: "My Custom Topic"
    weight: 0.9
    keywords:
      - "keyword one"
      - "keyword two"
```

## Model Selection

The two-stage approach lets you balance speed/cost vs. quality:

**OpenRouter (cloud):**

| Stage | Purpose | Recommendation | Cost per article |
|-------|---------|---------------|-----------------|
| Filter | Score all articles | `google/gemini-2.5-flash-lite`, `meta-llama/llama-3.1-8b-instruct` | ~$0.00001 |
| Summary | Summarize top picks | `anthropic/claude-sonnet-4`, `minimax/minimax-m2.5` | ~$0.002 |

Browse all models: [openrouter.ai/models](https://openrouter.ai/models)

> **Tip:** Avoid free-tier models for the filter stage – they often return `null` content on structured JSON prompts. Cheap paid models like Gemini Flash are far more reliable and cost almost nothing.

**Ollama (local):**

| Stage | Purpose | Recommendation |
|-------|---------|---------------|
| Filter | Score all articles | `qwen3:8b`, `llama3.2:3b`, `phi3:3.8b` |
| Summary | Summarize top picks | `qwen3:30b`, `llama3.3:70b`, `qwen3-coder:30b` |

Adjust `top_n` in config to control how many articles reach stage 2.

### Fallback Mode

If the LLM backend (Ollama or OpenRouter) is unreachable, the script falls back to keyword-based scoring automatically. You still get a digest, just without AI-generated summaries.

## Database & Persistence

All data is stored in a single SQLite file (default: `~/.ai-digest/articles.db`). The schema auto-migrates when upgrading from older versions.

### What's stored

| Table | Contents | Retention |
|-------|----------|-----------|
| `articles` | URL, title, source, category, published date, content snippet, relevance score, matched topics (JSON), AI headline, AI summary, which models were used | Scored articles: 365 days. Unscored: 90 days. |
| `digests` | Complete HTML, article count, status, list of article URLs (JSON) | HTML: 90 days. Metadata: indefinitely. |

### Why persist LLM outputs?

- **Re-generate digests** from stored data without re-running the LLM
- **Track how topics trend** over time (which topics score highest, which sources deliver)
- **Debug scoring** – see exactly what the LLM scored and summarized
- **Cost tracking** – know which model processed which article

## File Structure

```
ai-digest/
├── ai_digest.py        # Main script (all-in-one, ~1300 lines)
├── config.yaml         # Default config (template)
├── requirements.txt    # Python deps (feedparser, requests, pyyaml)
└── README.md           # This file

~/.ai-digest/
└── articles.db         # SQLite database (auto-created, auto-migrated)
```

## Tips

- **OpenRouter API key**: Set it as env var `OPENROUTER_API_KEY` for security instead of putting it in the config file. Track costs at [openrouter.ai/activity](https://openrouter.ai/activity).
- **Gmail SMTP**: Use an [App Password](https://support.google.com/accounts/answer/185833), not your real password. Host: `smtp.gmail.com`, Port: `587`.
- **Pi deployment**: The script itself is lightweight. Run fetching + email on a Pi, point `base_url` to your GPU machine for LLM processing.
- **ArXiv overload**: ArXiv feeds can have 100+ entries/day. Lower `max_per_feed` or raise `min_relevance` to compensate.
- **Debugging**: Set `log_level: DEBUG` in config for verbose output.
- **Adding feeds**: Just add an entry to the `feeds` list in config. If a blog doesn't have an RSS feed, check [Olshansk/rss-feeds](https://github.com/Olshansk/rss-feeds) or use [RSSHub](https://docs.rsshub.app/) as a proxy.
- **Free-tier models**: If you use OpenRouter free models and get empty results, switch to a cheap paid model. The `null` content bug is common with free-tier endpoints.

## License

MIT – Do whatever you want with it.