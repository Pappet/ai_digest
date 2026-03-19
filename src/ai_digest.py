#!/usr/bin/env python3
"""
AI Digest – Daily AI news aggregator with LLM-powered filtering and summarization.

Fetches RSS feeds, scores articles for relevance using an LLM (Ollama or OpenRouter),
summarizes the top picks, and sends a formatted HTML email digest.
"""

import argparse
import hashlib
import html
import json
import logging
import os
import re
import sqlite3
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import feedparser
import requests
import yaml

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Article:
    title: str
    url: str
    source: str
    category: str
    published: datetime
    summary: str = ""
    content_snippet: str = ""
    relevance_score: float = 0.0
    matched_topics: list = field(default_factory=list)
    ai_summary: str = ""
    ai_headline: str = ""


@dataclass
class TopicProfile:
    name: str
    weight: float
    keywords: list


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load and validate configuration from YAML file."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Copy config.yaml.example to config.yaml and edit it.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ---------------------------------------------------------------------------
# Database (deduplication)
# ---------------------------------------------------------------------------

class ArticleDB:
    """SQLite-based article store with LLM output persistence and digest archive."""

    SCHEMA_VERSION = 2

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
        self._migrate()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS articles (
                url_hash        TEXT PRIMARY KEY,
                url             TEXT NOT NULL,
                title           TEXT,
                source          TEXT,
                category        TEXT,
                published       TEXT,
                first_seen      TEXT DEFAULT (datetime('now')),
                sent            INTEGER DEFAULT 0,
                -- LLM Stage 1 outputs
                relevance_score REAL DEFAULT 0,
                matched_topics  TEXT DEFAULT '[]',
                -- LLM Stage 2 outputs
                ai_headline     TEXT,
                ai_summary      TEXT,
                -- Metadata
                content_snippet TEXT,
                filter_model    TEXT,
                summary_model   TEXT
            );

            CREATE TABLE IF NOT EXISTS digests (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT DEFAULT (datetime('now')),
                article_count   INTEGER,
                status          TEXT,
                html            TEXT,
                article_urls    TEXT DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_articles_score
                ON articles(relevance_score DESC);
            CREATE INDEX IF NOT EXISTS idx_articles_first_seen
                ON articles(first_seen);
            CREATE INDEX IF NOT EXISTS idx_digests_created
                ON digests(created_at);
        """)
        self.conn.commit()

    def _migrate(self):
        """Run schema migrations if needed."""
        row = self.conn.execute(
            "SELECT MAX(version) as v FROM schema_version"
        ).fetchone()
        current = row["v"] if row and row["v"] else 0

        if current < 2:
            # Migration: add columns if upgrading from v1
            existing = {
                col[1] for col in
                self.conn.execute("PRAGMA table_info(articles)").fetchall()
            }
            new_columns = {
                "category": "TEXT",
                "published": "TEXT",
                "relevance_score": "REAL DEFAULT 0",
                "matched_topics": "TEXT DEFAULT '[]'",
                "ai_headline": "TEXT",
                "ai_summary": "TEXT",
                "content_snippet": "TEXT",
                "filter_model": "TEXT",
                "summary_model": "TEXT",
            }
            for col, typedef in new_columns.items():
                if col not in existing:
                    self.conn.execute(
                        f"ALTER TABLE articles ADD COLUMN {col} {typedef}"
                    )

            self.conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,),
            )
            self.conn.commit()

    # -- Deduplication --

    def is_known(self, url: str) -> bool:
        """Check if we've already seen this article."""
        h = hashlib.sha256(url.encode()).hexdigest()
        row = self.conn.execute(
            "SELECT 1 FROM articles WHERE url_hash = ?", (h,)
        ).fetchone()
        return row is not None

    def mark_seen(self, article: Article, sent: bool = False):
        """Record an article as seen (basic info only)."""
        h = hashlib.sha256(article.url.encode()).hexdigest()
        self.conn.execute(
            """INSERT OR IGNORE INTO articles
               (url_hash, url, title, source, category, published, content_snippet, sent)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                h, article.url, article.title, article.source,
                article.category,
                article.published.isoformat() if article.published else None,
                article.content_snippet[:1000],
                int(sent),
            ),
        )
        self.conn.commit()

    # -- LLM Output Storage --

    def save_llm_results(self, article: Article, filter_model: str = "", summary_model: str = ""):
        """Persist LLM scoring and summary outputs for an article."""
        h = hashlib.sha256(article.url.encode()).hexdigest()
        self.conn.execute(
            """UPDATE articles SET
                relevance_score = ?,
                matched_topics  = ?,
                ai_headline     = ?,
                ai_summary      = ?,
                filter_model    = ?,
                summary_model   = ?,
                sent            = 1
            WHERE url_hash = ?""",
            (
                article.relevance_score,
                json.dumps(article.matched_topics, ensure_ascii=False),
                article.ai_headline,
                article.ai_summary,
                filter_model,
                summary_model,
                h,
            ),
        )
        self.conn.commit()

    # -- Digest Archive --

    def save_digest(self, html: str, articles: list[Article], status: str) -> int:
        """Archive a complete digest with HTML and article references."""
        article_urls = json.dumps([a.url for a in articles], ensure_ascii=False)
        cursor = self.conn.execute(
            """INSERT INTO digests (article_count, status, html, article_urls)
               VALUES (?, ?, ?, ?)""",
            (len(articles), status, html, article_urls),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_digest(self, digest_id: int) -> Optional[dict]:
        """Retrieve a stored digest by ID."""
        row = self.conn.execute(
            "SELECT * FROM digests WHERE id = ?", (digest_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_digests(self, limit: int = 20) -> list[dict]:
        """List recent digests."""
        rows = self.conn.execute(
            "SELECT id, created_at, article_count, status FROM digests "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Analytics / History --

    def get_article_history(self, limit: int = 100) -> list[dict]:
        """Get scored articles ordered by score, for trend analysis."""
        rows = self.conn.execute(
            """SELECT url, title, source, category, first_seen,
                      relevance_score, matched_topics, ai_headline, ai_summary
               FROM articles
               WHERE relevance_score > 0
               ORDER BY first_seen DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_topic_stats(self) -> list[dict]:
        """Aggregate score statistics per topic over time."""
        rows = self.conn.execute(
            """SELECT matched_topics, COUNT(*) as count,
                      ROUND(AVG(relevance_score), 1) as avg_score,
                      ROUND(MAX(relevance_score), 1) as max_score,
                      MIN(first_seen) as earliest,
                      MAX(first_seen) as latest
               FROM articles
               WHERE relevance_score > 0
               GROUP BY matched_topics
               ORDER BY count DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Maintenance --

    def log_digest(self, count: int, status: str):
        """Legacy compat – use save_digest for new code."""
        self.conn.execute(
            "INSERT INTO digests (article_count, status) VALUES (?, ?)",
            (count, status),
        )
        self.conn.commit()

    def cleanup(self, keep_days: int = 90):
        """Remove old articles. Keeps 90 days by default for trend analysis."""
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()
        self.conn.execute(
            "DELETE FROM articles WHERE first_seen < ? AND relevance_score = 0",
            (cutoff,),
        )
        # Keep scored articles longer (365 days) for analytics
        cutoff_scored = (datetime.now() - timedelta(days=365)).isoformat()
        self.conn.execute(
            "DELETE FROM articles WHERE first_seen < ?",
            (cutoff_scored,),
        )
        # Keep digest HTML for 90 days, then strip it but keep metadata
        self.conn.execute(
            "UPDATE digests SET html = NULL WHERE created_at < ?",
            (cutoff,),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# RSS Fetching
# ---------------------------------------------------------------------------

def fetch_feeds(cfg: dict) -> list[Article]:
    """Fetch all configured RSS feeds and return a list of Articles."""
    log = logging.getLogger("ai-digest.feeds")
    articles = []
    lookback = datetime.now(timezone.utc) - timedelta(
        days=cfg["general"]["lookback_days"]
    )
    headers = {"User-Agent": cfg["general"]["user_agent"]}
    max_per_feed = cfg["general"].get("max_per_feed", 30)

    for feed_cfg in cfg["feeds"]:
        name = feed_cfg["name"]
        url = feed_cfg["url"]
        category = feed_cfg.get("category", "general")

        log.info(f"Fetching: {name}")
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
        except Exception as e:
            log.warning(f"  Failed to fetch {name}: {e}")
            continue

        count = 0
        for entry in parsed.entries[:max_per_feed]:
            # Parse publication date
            pub_date = _parse_date(entry)
            if pub_date and pub_date < lookback:
                continue

            # Extract content snippet
            snippet = ""
            if hasattr(entry, "summary"):
                snippet = _strip_html(entry.summary)[:500]
            elif hasattr(entry, "content"):
                snippet = _strip_html(entry.content[0].get("value", ""))[:500]

            article = Article(
                title=entry.get("title", "Untitled").strip(),
                url=entry.get("link", ""),
                source=name,
                category=category,
                published=pub_date or datetime.now(timezone.utc),
                content_snippet=snippet,
            )
            articles.append(article)
            count += 1

        log.info(f"  → {count} articles from {name}")

    log.info(f"Total fetched: {len(articles)} articles")
    return articles


def _parse_date(entry) -> Optional[datetime]:
    """Try to extract a timezone-aware datetime from a feed entry."""
    for attr in ("published_parsed", "updated_parsed"):
        tp = getattr(entry, attr, None)
        if tp:
            try:
                return datetime(*tp[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# LLM Integration (Backend-agnostic)
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base for LLM backends."""

    @abstractmethod
    def generate(self, model: str, prompt: str, system: str = "") -> str:
        """Send a prompt and return the response text."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is reachable."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...


class OllamaClient(LLMClient):
    """Local Ollama API client."""

    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.log = logging.getLogger("ai-digest.llm.ollama")

    @property
    def name(self) -> str:
        return "Ollama"

    def generate(self, model: str, prompt: str, system: str = "") -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1024,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            self.log.warning(f"LLM timeout for model {model}")
            return ""
        except Exception as e:
            self.log.error(f"LLM error: {e}")
            return ""

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


class OpenRouterClient(LLMClient):
    """OpenRouter API client (OpenAI-compatible)."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, timeout: int = 120, site_name: str = "AI Digest"):
        self.api_key = api_key
        self.timeout = timeout
        self.site_name = site_name
        self.log = logging.getLogger("ai-digest.llm.openrouter")

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set llm.api_key in config or OPENROUTER_API_KEY env var."
            )

    @property
    def name(self) -> str:
        return "OpenRouter"

    def generate(self, model: str, prompt: str, system: str = "") -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": f"https://github.com/ai-digest",
            "X-Title": self.site_name,
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        }

        try:
            resp = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            # Handle OpenRouter error responses
            if "error" in data:
                self.log.error(f"OpenRouter error: {data['error']}")
                return ""

            content = data["choices"][0]["message"].get("content")
            if content is None:
                # Some models (especially free-tier) return null content
                # Check for refusal or empty response
                refusal = data["choices"][0]["message"].get("refusal")
                if refusal:
                    self.log.warning(f"Model refused: {refusal}")
                else:
                    self.log.warning(
                        f"Model {model} returned null content. "
                        "Try a different model or check OpenRouter status."
                    )
                return ""
            return content.strip()

        except requests.exceptions.Timeout:
            self.log.warning(f"OpenRouter timeout for model {model}")
            return ""
        except (KeyError, IndexError, AttributeError) as e:
            self.log.error(f"Unexpected OpenRouter response format: {e}")
            return ""
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After", "5")
                self.log.warning(f"Rate limited. Retry after {retry_after}s")
                time.sleep(float(retry_after))
                return self.generate(model, prompt, system)  # One retry
            self.log.error(f"OpenRouter HTTP error: {e}")
            return ""
        except Exception as e:
            self.log.error(f"OpenRouter error: {e}")
            return ""

    def is_available(self) -> bool:
        """Check OpenRouter reachability and API key validity."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            resp = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                self.log.info(
                    f"OpenRouter connected – credits: ${data.get('limit', '?')}, "
                    f"used: ${data.get('usage', '?')}"
                )
                return True
            elif resp.status_code == 401:
                self.log.error("OpenRouter API key is invalid.")
                return False
            return False
        except Exception as e:
            self.log.error(f"Cannot reach OpenRouter: {e}")
            return False


def create_llm_client(llm_cfg: dict) -> LLMClient:
    """Factory: create the right LLM client based on config."""
    backend = llm_cfg.get("backend", "ollama").lower()

    if backend == "openrouter":
        api_key = llm_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        return OpenRouterClient(
            api_key=api_key,
            timeout=llm_cfg.get("timeout", 120),
            site_name=llm_cfg.get("site_name", "AI Digest"),
        )
    elif backend == "ollama":
        return OllamaClient(
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            timeout=llm_cfg.get("timeout", 120),
        )
    else:
        raise ValueError(
            f"Unknown LLM backend: '{backend}'. Use 'ollama' or 'openrouter'."
        )


def stage1_filter(
    articles: list[Article],
    topics: list[TopicProfile],
    client: LLMClient,
    model: str,
    min_relevance: float,
) -> list[Article]:
    """
    Stage 1: Fast relevance scoring.
    Sends batches of article titles to a small model for quick filtering.
    """
    log = logging.getLogger("ai-digest.stage1")
    log.info(f"Stage 1: Scoring {len(articles)} articles with {model}...")

    topic_desc = "\n".join(
        f"- {t.name} (weight: {t.weight}): {', '.join(t.keywords[:5])}"
        for t in topics
    )

    system_prompt = textwrap.dedent(f"""\
        You are an AI news relevance scorer. You evaluate article titles and snippets
        for relevance to specific topics. Be selective – only high-quality, genuinely
        relevant articles should score above 5.

        Topics of interest (with weights):
        {topic_desc}

        IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation.""")

    # Process in batches of 10 for efficiency
    batch_size = 10
    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]

        items = []
        for idx, a in enumerate(batch):
            items.append({
                "idx": idx,
                "title": a.title[:150],
                "source": a.source,
                "snippet": a.content_snippet[:200],
            })

        prompt = textwrap.dedent(f"""\
            Score these articles for relevance (0-10) to the topics listed above.
            Return a JSON array with objects: {{"idx": <int>, "score": <float>, "topics": [<matched topic names>]}}

            Articles:
            {json.dumps(items, ensure_ascii=False, indent=2)}""")

        response = client.generate(model, prompt, system_prompt)

        # Parse JSON response
        try:
            # Try to extract JSON from response (handle markdown fences)
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                log.warning(f"  Batch {i//batch_size}: No JSON found in response")
                continue

            for score_item in scores:
                idx = score_item.get("idx", -1)
                if 0 <= idx < len(batch):
                    batch[idx].relevance_score = float(score_item.get("score", 0))
                    batch[idx].matched_topics = score_item.get("topics", [])

        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"  Batch {i//batch_size}: Failed to parse scores: {e}")
            continue

        # Be nice to the local GPU
        time.sleep(0.5)

    # Apply topic weights to final score
    for article in articles:
        if article.matched_topics:
            weight_boost = max(
                (t.weight for t in topics if t.name in article.matched_topics),
                default=1.0,
            )
            article.relevance_score *= weight_boost

    # Filter and sort
    filtered = [a for a in articles if a.relevance_score >= min_relevance]
    filtered.sort(key=lambda a: a.relevance_score, reverse=True)

    log.info(
        f"Stage 1 complete: {len(filtered)}/{len(articles)} articles passed "
        f"(threshold: {min_relevance})"
    )
    return filtered


def stage2_summarize(
    articles: list[Article],
    client: LLMClient,
    model: str,
) -> list[Article]:
    """
    Stage 2: Detailed summarization of top articles using a larger model.
    """
    log = logging.getLogger("ai-digest.stage2")
    log.info(f"Stage 2: Summarizing {len(articles)} articles with {model}...")

    system_prompt = textwrap.dedent("""\
        You are a concise tech news summarizer writing for a senior software engineer
        who follows AI/ML closely and runs local LLMs. Write in a direct, technical
        style. No fluff. Do NOT try to connect every article to a specific domain –
        just summarize what actually happened.

        For each article, provide:
        1. A punchy one-line headline (may differ from original title)
        2. A 2-3 sentence summary covering: what happened, why it matters

        IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation.""")

    for i, article in enumerate(articles):
        prompt = textwrap.dedent(f"""\
            Summarize this article:

            Title: {article.title}
            Source: {article.source}
            Content: {article.content_snippet}

            Return JSON: {{"headline": "...", "summary": "..."}}""")

        response = client.generate(model, prompt, system_prompt)

        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                article.ai_headline = data.get("headline", article.title)
                article.ai_summary = data.get("summary", article.content_snippet[:200])
            else:
                article.ai_headline = article.title
                article.ai_summary = article.content_snippet[:200]
        except (json.JSONDecodeError, ValueError):
            article.ai_headline = article.title
            article.ai_summary = article.content_snippet[:200]

        log.info(f"  [{i+1}/{len(articles)}] {article.ai_headline[:60]}...")
        time.sleep(0.5)

    return articles


# ---------------------------------------------------------------------------
# Email Generation
# ---------------------------------------------------------------------------

def generate_html(articles: list[Article], topics: list[TopicProfile]) -> str:
    """Generate a clean, readable HTML email digest."""
    today = datetime.now().strftime("%A, %d. %B %Y")

    # Build a set of canonical topic names for normalization
    canonical_names = {t.name for t in topics}

    def _normalize_topic(raw: str) -> str:
        """Map LLM-returned topic names back to canonical names.
        Handles cases like 'Local & Open-Source LLMs (weight: 1.0)'."""
        raw_clean = re.sub(r"\s*\(.*?\)\s*$", "", raw).strip()
        if raw_clean in canonical_names:
            return raw_clean
        # Fuzzy: check if any canonical name is contained in the raw string
        for canon in canonical_names:
            if canon.lower() in raw.lower() or raw.lower() in canon.lower():
                return canon
        return raw_clean

    # Normalize topic names on all articles
    for a in articles:
        a.matched_topics = [_normalize_topic(t) for t in a.matched_topics]

    # Group by primary topic, preserving config order
    topic_articles: dict[str, list[Article]] = {}
    for a in articles:
        primary_topic = a.matched_topics[0] if a.matched_topics else "Other"
        topic_articles.setdefault(primary_topic, []).append(a)

    # Sort sections: configured topics first (in config order), then "Other"
    ordered_topics = [t.name for t in topics if t.name in topic_articles]
    for key in topic_articles:
        if key not in ordered_topics:
            ordered_topics.append(key)

    # Build sections
    sections_html = ""
    for topic_name in ordered_topics:
        topic_arts = topic_articles[topic_name]
        cards = ""
        for a in topic_arts:
            score_color = (
                "#22c55e" if a.relevance_score >= 8
                else "#f59e0b" if a.relevance_score >= 6
                else "#94a3b8"
            )
            cards += f"""\
            <div style="margin-bottom:16px; padding:16px; background:#f8fafc;
                        border-left:4px solid {score_color}; border-radius:4px;">
                <div style="font-size:11px; color:#64748b; margin-bottom:4px;">
                    {html.escape(a.source)} · Score: {a.relevance_score:.1f}
                </div>
                <a href="{html.escape(a.url)}"
                   style="font-size:16px; font-weight:600; color:#0f172a;
                          text-decoration:none; line-height:1.3;">
                    {html.escape(a.ai_headline or a.title)}
                </a>
                <p style="margin:8px 0 0; font-size:14px; color:#334155; line-height:1.5;">
                    {html.escape(a.ai_summary or a.content_snippet[:200])}
                </p>
            </div>"""

        sections_html += f"""\
        <div style="margin-bottom:28px;">
            <h2 style="font-size:18px; color:#1e293b; border-bottom:2px solid #e2e8f0;
                       padding-bottom:6px; margin-bottom:12px;">
                {html.escape(topic_name)}
            </h2>
            {cards}
        </div>"""

    return f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
             max-width:680px; margin:0 auto; padding:20px; background:#ffffff;">

    <div style="text-align:center; padding:24px 0; border-bottom:3px solid #1e293b;">
        <h1 style="margin:0; font-size:28px; color:#1e293b;">🤖 AI Digest</h1>
        <p style="margin:6px 0 0; font-size:14px; color:#64748b;">{today}</p>
        <p style="margin:4px 0 0; font-size:13px; color:#94a3b8;">
            {len(articles)} articles curated from {len(set(a.source for a in articles))} sources
        </p>
    </div>

    <div style="padding:24px 0;">
        {sections_html}
    </div>

    <div style="text-align:center; padding:16px 0; border-top:1px solid #e2e8f0;
                font-size:12px; color:#94a3b8;">
        Generated by AI Digest ·
        <a href="#" style="color:#64748b;">Unsubscribe</a>
    </div>
</body>
</html>"""


def send_email(cfg: dict, html_content: str, article_count: int):
    """Send the digest email via SMTP."""
    log = logging.getLogger("ai-digest.email")
    email_cfg = cfg["email"]

    msg = MIMEMultipart("alternative")
    today = datetime.now().strftime("%Y-%m-%d")
    msg["Subject"] = email_cfg["subject"].format(date=today)
    msg["From"] = email_cfg["from_addr"]
    msg["To"] = ", ".join(email_cfg["to_addrs"])

    # Plain text fallback
    plain = f"AI Digest – {today}\n\n{article_count} articles curated. View in HTML-capable client."
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    import smtplib

    try:
        if email_cfg.get("use_tls", True):
            server = smtplib.SMTP(email_cfg["smtp_host"], email_cfg["smtp_port"])
            server.starttls()
        else:
            server = smtplib.SMTP(email_cfg["smtp_host"], email_cfg["smtp_port"])

        if email_cfg.get("smtp_user"):
            server.login(email_cfg["smtp_user"], email_cfg["smtp_password"])

        server.sendmail(
            email_cfg["from_addr"],
            email_cfg["to_addrs"],
            msg.as_string(),
        )
        server.quit()
        log.info("Email sent successfully!")
    except Exception as e:
        log.error(f"Failed to send email: {e}")
        raise


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run(config_path: str, dry_run: bool = False, stdout: bool = False, reset_db: bool = False):
    """Execute the full digest pipeline."""
    cfg = load_config(config_path)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg["general"].get("log_level", "INFO")),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("ai-digest")
    log.info("=" * 60)
    log.info("AI Digest Pipeline Starting")
    log.info("=" * 60)

    # Parse topic profiles
    topics = [
        TopicProfile(
            name=t["name"],
            weight=t.get("weight", 1.0),
            keywords=t.get("keywords", []),
        )
        for t in cfg.get("topics", [])
    ]

    # Init database
    db = ArticleDB(cfg["general"]["db_path"])

    if reset_db:
        log.info("Resetting article database...")
        db.conn.execute("DELETE FROM articles")
        db.conn.commit()
        log.info("Database cleared.")

    db.cleanup()

    # Step 1: Fetch feeds
    log.info("Step 1/4: Fetching RSS feeds...")
    all_articles = fetch_feeds(cfg)

    # Step 2: Deduplicate
    log.info("Step 2/4: Deduplicating...")
    new_articles = [a for a in all_articles if not db.is_known(a.url)]
    log.info(f"  {len(new_articles)} new articles (filtered {len(all_articles) - len(new_articles)} duplicates)")

    if not new_articles:
        log.info("No new articles found. Nothing to do.")
        db.log_digest(0, "empty")
        db.close()
        return

    # Step 3: LLM Processing
    llm_cfg = cfg["llm"]
    try:
        client = create_llm_client(llm_cfg)
    except ValueError as e:
        log.error(str(e))
        log.info("Falling back to keyword-based scoring...")
        new_articles = _keyword_fallback(new_articles, topics)
        top_articles = sorted(new_articles, key=lambda a: a.relevance_score, reverse=True)[
            : llm_cfg.get("top_n", 15)
        ]
        # skip to step 4
        client = None

    if client and not client.is_available():
        log.error(
            f"{client.name} not reachable. Check your config."
        )
        log.info("Falling back to keyword-based scoring...")
        new_articles = _keyword_fallback(new_articles, topics)
        top_articles = sorted(new_articles, key=lambda a: a.relevance_score, reverse=True)[
            : llm_cfg.get("top_n", 15)
        ]
    elif client:
        log.info(f"Step 3/4: LLM Processing via {client.name}...")

        # Stage 1: Filter
        filtered = stage1_filter(
            new_articles,
            topics,
            client,
            llm_cfg["filter_model"],
            llm_cfg.get("min_relevance", 4),
        )

        # Stage 2: Summarize top N
        top_n = llm_cfg.get("top_n", 15)
        top_articles = stage2_summarize(filtered[:top_n], client, llm_cfg["summary_model"])

    # Step 4: Generate and deliver
    log.info("Step 4/4: Generating digest...")

    if not top_articles:
        log.info("No articles scored above threshold. Nothing to send.")
        db.log_digest(0, "empty_after_filter")
        db.close()
        return

    # Persist all articles (basic info for dedup)
    for a in new_articles:
        db.mark_seen(a)

    # Persist LLM outputs for scored articles
    filter_model = llm_cfg.get("filter_model", "")
    summary_model = llm_cfg.get("summary_model", "")
    for a in top_articles:
        db.save_llm_results(a, filter_model=filter_model, summary_model=summary_model)
    log.info(f"  Saved LLM results for {len(top_articles)} articles to database.")

    html_content = generate_html(top_articles, topics)

    # Archive the digest
    digest_status = "dry_run" if dry_run else ("stdout" if stdout else "sent")
    digest_id = db.save_digest(html_content, top_articles, digest_status)
    log.info(f"  Digest #{digest_id} archived in database.")

    if stdout:
        print(html_content)
    elif dry_run:
        out_path = Path("digest_preview.html")
        out_path.write_text(html_content, encoding="utf-8")
        log.info(f"Dry run: Preview saved to {out_path}")
    else:
        send_email(cfg, html_content, len(top_articles))

    log.info(f"Done! {len(top_articles)} articles in today's digest.")
    db.close()


def _keyword_fallback(
    articles: list[Article], topics: list[TopicProfile]
) -> list[Article]:
    """Simple keyword-based scoring when LLM is not available."""
    for article in articles:
        text = f"{article.title} {article.content_snippet}".lower()
        max_score = 0
        matched = []
        for topic in topics:
            hits = sum(1 for kw in topic.keywords if kw.lower() in text)
            if hits > 0:
                score = min(hits * 2.5, 10) * topic.weight
                if score > max_score:
                    max_score = score
                matched.append(topic.name)
        article.relevance_score = max_score
        article.matched_topics = matched
    return articles


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_email_config(config_path: str):
    """Send a short test email to verify SMTP settings."""
    cfg = load_config(config_path)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("ai-digest.test-email")
    email_cfg = cfg["email"]

    log.info("=" * 50)
    log.info("Testing email configuration")
    log.info("=" * 50)
    log.info(f"  SMTP Host:  {email_cfg['smtp_host']}:{email_cfg['smtp_port']}")
    log.info(f"  SMTP User:  {email_cfg.get('smtp_user', '(none)')}")
    log.info(f"  TLS:        {email_cfg.get('use_tls', True)}")
    log.info(f"  From:       {email_cfg['from_addr']}")
    log.info(f"  To:         {', '.join(email_cfg['to_addrs'])}")
    log.info("-" * 50)

    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    test_html = f"""\
<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family:sans-serif; max-width:500px; margin:0 auto; padding:20px;">
    <div style="text-align:center; padding:20px; border:2px solid #22c55e; border-radius:8px;">
        <h1 style="color:#22c55e; margin:0;">✅ AI Digest</h1>
        <p style="color:#64748b; margin:8px 0 0;">Email test successful!</p>
        <p style="color:#94a3b8; font-size:13px; margin:4px 0 0;">Sent at {today}</p>
    </div>
    <p style="color:#334155; font-size:14px; margin-top:16px; text-align:center;">
        If you see this, your SMTP settings are correct.<br>
        Daily digests will arrive at this address.
    </p>
</body></html>"""

    try:
        send_email(cfg, test_html, 0)
        log.info("=" * 50)
        log.info("SUCCESS – Check your inbox!")
        log.info("=" * 50)
    except Exception as e:
        log.error("=" * 50)
        log.error(f"FAILED: {e}")
        log.error("=" * 50)
        log.error("Common fixes:")
        log.error("  - Gmail: Use an App Password, not your real password")
        log.error("  - Port 587 = STARTTLS, Port 465 = SSL/TLS")
        log.error("  - Check firewall / network allows outbound SMTP")
        sys.exit(1)


def show_digest_history(config_path: str):
    """List all archived digests."""
    cfg = load_config(config_path)
    db = ArticleDB(cfg["general"]["db_path"])
    digests = db.list_digests(limit=30)
    db.close()

    if not digests:
        print("No digests found in database.")
        return

    print(f"\n{'ID':>4}  {'Date':20s}  {'Articles':>8}  {'Status':12s}  {'HTML':>5}")
    print("-" * 60)
    for d in digests:
        has_html = "✓" if d.get("html") else "–"  # html not in list query
        print(
            f"{d['id']:>4}  {d['created_at']:20s}  {d['article_count']:>8}  "
            f"{d['status']:12s}"
        )
    print(f"\nUse --show-digest <ID> to export a digest as HTML.")


def export_digest(config_path: str, digest_id: int):
    """Export an archived digest as HTML file."""
    cfg = load_config(config_path)
    db = ArticleDB(cfg["general"]["db_path"])
    digest = db.get_digest(digest_id)
    db.close()

    if not digest:
        print(f"Error: Digest #{digest_id} not found.")
        sys.exit(1)

    if not digest.get("html"):
        print(f"Error: Digest #{digest_id} exists but HTML was cleaned up (older than 90 days).")
        sys.exit(1)

    out_path = Path(f"digest_{digest_id}.html")
    out_path.write_text(digest["html"], encoding="utf-8")
    print(f"Exported digest #{digest_id} ({digest['created_at']}, "
          f"{digest['article_count']} articles) → {out_path}")


def show_stats(config_path: str):
    """Show article and topic statistics from the database."""
    cfg = load_config(config_path)
    db = ArticleDB(cfg["general"]["db_path"])

    # Digest summary
    digests = db.list_digests(limit=100)
    total_digests = len(digests)
    total_articles_sent = sum(d["article_count"] for d in digests)

    print("\n" + "=" * 60)
    print("  AI Digest – Statistics")
    print("=" * 60)
    print(f"\n  Total digests:      {total_digests}")
    print(f"  Total articles sent: {total_articles_sent}")
    if digests:
        print(f"  First digest:       {digests[-1]['created_at']}")
        print(f"  Last digest:        {digests[0]['created_at']}")

    # Top articles
    articles = db.get_article_history(limit=20)
    if articles:
        print(f"\n{'─' * 60}")
        print("  Top 20 Articles by Score")
        print(f"{'─' * 60}")
        sorted_arts = sorted(articles, key=lambda a: a["relevance_score"], reverse=True)
        for i, a in enumerate(sorted_arts[:20], 1):
            score = a["relevance_score"]
            title = (a["ai_headline"] or a["title"] or "Untitled")[:55]
            source = (a["source"] or "?")[:15]
            print(f"  {i:>2}. [{score:>4.1f}] {title:55s} ({source})")

    # Topic distribution
    topic_stats = db.get_topic_stats()
    if topic_stats:
        print(f"\n{'─' * 60}")
        print("  Topic Distribution")
        print(f"{'─' * 60}")
        for ts in topic_stats:
            topics_raw = ts["matched_topics"]
            try:
                topics_list = json.loads(topics_raw)
                topic_name = ", ".join(topics_list) if topics_list else "Unscored"
            except (json.JSONDecodeError, TypeError):
                topic_name = str(topics_raw)[:40]
            print(
                f"  {topic_name:40s}  "
                f"n={ts['count']:>3}  avg={ts['avg_score']:>4.1f}  max={ts['max_score']:>4.1f}"
            )

    print()
    db.close()


def main():
    parser = argparse.ArgumentParser(
        description="AI Digest – Daily AI news aggregator with LLM-powered filtering"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate digest but don't send email (saves HTML preview)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print HTML digest to stdout",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Clear the article database before running (reprocess all articles)",
    )
    parser.add_argument(
        "--test-email",
        action="store_true",
        help="Send a test email to verify SMTP settings, then exit",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="List all archived digests",
    )
    parser.add_argument(
        "--show-digest",
        type=int,
        metavar="ID",
        help="Export an archived digest as HTML file",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show article and topic statistics",
    )
    args = parser.parse_args()

    if args.test_email:
        test_email_config(args.config)
    elif args.history:
        show_digest_history(args.config)
    elif args.show_digest is not None:
        export_digest(args.config, args.show_digest)
    elif args.stats:
        show_stats(args.config)
    else:
        run(args.config, dry_run=args.dry_run, stdout=args.stdout, reset_db=args.reset_db)


if __name__ == "__main__":
    main()