#!/usr/bin/env python3
"""
AI Digest Web UI – Browser-based configuration editor.

Run alongside your existing cron job:
    python src/web_ui.py -c src/config.local.yaml
    python src/web_ui.py -c src/config.local.yaml --host 0.0.0.0 --port 5000

Writes are atomic (temp file + rename) so the cron job is never disrupted.
Note: YAML comments are stripped on first save via the web UI.
"""

import argparse
import html as htmllib
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from flask import (
    Flask,
    flash,
    make_response,
    redirect,
    render_template_string,
    request,
    url_for,
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

CONFIG_PATH: Path = None
SCRIPT_PATH: Path = None


# ─── Config helpers ───────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    """Atomic overwrite. YAML comments are not preserved after the first save."""
    tmp = CONFIG_PATH.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False,
                  default_flow_style=False, width=120)
    tmp.replace(CONFIG_PATH)


def e(value) -> str:
    """HTML-escape a value for safe embedding in attributes/text."""
    return htmllib.escape(str(value) if value is not None else "")


# ─── Base layout ─────────────────────────────────────────────────────────────

BASE_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Digest</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
  <style>
    /* ── Design tokens ─────────────────────────────── */
    :root {
      --surface-lowest:   #0e0e0e;
      --surface:          #131313;
      --surface-low:      #191919;
      --surface-mid:      #1f1f1f;
      --surface-high:     #2a2a2a;
      --surface-highest:  #3a3a3a;
      --primary:          #A6FF00;
      --primary-dark:     #467000;
      --primary-dim:      #8ad800;
      --secondary:        #00E0FF;
      --caution:          #FF8A00;
      --on-surface:       #e2e2e2;
      --on-surface-muted: #8a8a8a;
      --on-surface-faint: #555555;
      --on-primary:       #0a1000;
      --ghost:            rgba(65,74,52,.15);
      --ghost-strong:     rgba(65,74,52,.35);
      --glow-primary:     0 0 12px rgba(166,255,0,.30);
      --glow-caution:     0 0 12px rgba(255,138,0,.30);
      --font-display:     'Space Grotesk', sans-serif;
      --font-body:        'Manrope', sans-serif;
      --font-mono:        'JetBrains Mono', monospace;
      --radius-xs:        0.125rem;
      --radius-sm:        0.25rem;
      --radius-md:        0.375rem;
      --transition:       160ms cubic-bezier(0.2,0,0,1);
    }

    /* ── Base ──────────────────────────────────────── */
    body { background: var(--surface); color: var(--on-surface); font-family: var(--font-body); font-size: 15px; min-height: 100vh; }
    a { color: var(--secondary); }
    a:hover { color: var(--primary); }
    h4, h5, h6 { font-family: var(--font-display); letter-spacing: 1px; color: var(--on-surface); }
    code {
      font-family: var(--font-mono); font-size: .8rem;
      color: var(--secondary); background: var(--surface-lowest);
      padding: .1em .3em; border-radius: var(--radius-xs);
    }

    /* ── Sidebar ───────────────────────────────────── */
    .sidebar {
      width: 210px; min-width: 210px; background: var(--surface-lowest);
      min-height: 100vh; padding: 1.25rem .75rem;
      position: sticky; top: 0; height: 100vh; overflow-y: auto;
      border-right: 1px solid var(--ghost);
    }
    .sidebar-brand {
      color: var(--on-surface); font-family: var(--font-display);
      font-weight: 700; font-size: 13px; letter-spacing: 1.5px;
      text-transform: uppercase; text-decoration: none;
      display: flex; align-items: center; gap: .5rem;
    }
    .sidebar-brand:hover { color: var(--on-surface); text-decoration: none; }
    .brand-logo {
      width: 26px; height: 26px; flex-shrink: 0;
      background: linear-gradient(135deg, var(--primary), var(--primary-dark));
      border-radius: var(--radius-xs);
      display: flex; align-items: center; justify-content: center;
      font-size: 11px; font-weight: 700; color: var(--on-primary);
    }
    .brand-sub {
      font-family: var(--font-mono); font-size: 9px; font-weight: 400;
      color: var(--on-surface-faint); text-transform: none; letter-spacing: 0;
      display: block; margin-top: -2px;
    }
    .sidebar .nav-link {
      color: var(--on-surface-muted); font-family: var(--font-display);
      font-size: 11px; font-weight: 600; letter-spacing: 1.2px;
      text-transform: uppercase; padding: .4rem .75rem;
      border-radius: var(--radius-xs); transition: var(--transition);
    }
    .sidebar .nav-link:hover  { color: var(--on-surface); background: var(--surface-high); }
    .sidebar .nav-link.active { color: var(--on-primary); background: linear-gradient(135deg, var(--primary), var(--primary-dark)); }
    .sidebar .nav-link i { margin-right: .4rem; }

    /* ── Main ──────────────────────────────────────── */
    .main { padding: 1.75rem 2rem; flex: 1; min-width: 0; }

    /* ── Cards ─────────────────────────────────────── */
    .card { background: var(--surface-low) !important; border: 1px solid var(--ghost) !important; border-radius: var(--radius-md) !important; box-shadow: none !important; }
    .card-header {
      background: var(--surface-lowest) !important; border-bottom: 1px solid var(--ghost) !important;
      font-family: var(--font-display); font-size: 10px; font-weight: 600;
      letter-spacing: 2.5px; text-transform: uppercase; color: var(--on-surface-muted);
    }
    .card-body  { background: transparent; }
    .card-footer { background: var(--surface-low) !important; border-top: 1px solid var(--ghost) !important; }

    /* ── Tables ────────────────────────────────────── */
    .table { --bs-table-bg: transparent; --bs-table-color: var(--on-surface); --bs-table-border-color: var(--ghost); --bs-table-hover-bg: var(--surface-mid); }
    .table th { font-family: var(--font-display); font-size: .7rem; text-transform: uppercase; letter-spacing: .15em; color: var(--on-surface-faint); font-weight: 600; border-bottom: 1px solid var(--ghost-strong) !important; }
    .table td  { border-bottom: 1px solid var(--ghost) !important; }

    /* ── Buttons ───────────────────────────────────── */
    .btn { font-family: var(--font-display); font-size: 11.5px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; border-radius: var(--radius-xs); transition: var(--transition); }
    .btn:active { transform: scale(0.97); }
    .btn-primary, .btn-success { background: linear-gradient(135deg, var(--primary), var(--primary-dark)); color: var(--on-primary); border: none; }
    .btn-primary:hover, .btn-success:hover { background: linear-gradient(135deg, var(--primary), var(--primary-dark)); color: var(--on-primary); box-shadow: var(--glow-primary); }
    .btn-outline-secondary { background: var(--surface-high); color: var(--on-surface-muted); border: 1px solid var(--ghost-strong); }
    .btn-outline-secondary:hover { background: var(--surface-highest); color: var(--on-surface); border-color: var(--ghost-strong); }
    .btn-outline-primary { background: var(--surface-high); color: var(--primary-dim); border: 1px solid var(--ghost-strong); }
    .btn-outline-primary:hover { background: var(--surface-highest); color: var(--primary); border-color: var(--primary); box-shadow: var(--glow-primary); }
    .btn-outline-danger { background: var(--surface-high); color: var(--on-surface-muted); border: 1px solid var(--ghost-strong); }
    .btn-outline-danger:hover { background: var(--surface-highest); color: var(--caution); border-color: var(--caution); box-shadow: var(--glow-caution); }

    /* ── Forms ─────────────────────────────────────── */
    .form-control, .form-select { background-color: var(--surface-lowest); color: var(--on-surface); border: 1px solid var(--ghost-strong); border-radius: var(--radius-xs); transition: var(--transition); }
    .form-control:focus, .form-select:focus { background-color: var(--surface-lowest); color: var(--on-surface); border-color: var(--primary); box-shadow: 0 0 0 2px rgba(166,255,0,.15); outline: none; }
    .form-control::placeholder { color: var(--on-surface-faint); }
    .form-label { font-family: var(--font-display); font-size: 10px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: var(--on-surface-muted); }
    .form-select { background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill='none' stroke='%238a8a8a' stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='m2 5 6 6 6-6'/%3e%3c/svg%3e"); }
    .form-check-input { background-color: var(--surface-high); border-color: var(--ghost-strong); }
    .form-check-input:checked { background-color: var(--primary); border-color: var(--primary); }
    .form-check-label { color: var(--on-surface); }
    textarea.kw { font-family: var(--font-mono); font-size: .8rem; }

    /* ── Alerts ────────────────────────────────────── */
    .alert-success { background: rgba(166,255,0,.07); border: 1px solid rgba(166,255,0,.25); color: var(--primary); border-radius: var(--radius-sm); }
    .alert-danger  { background: rgba(255,138,0,.07); border: 1px solid rgba(255,138,0,.25); color: var(--caution); border-radius: var(--radius-sm); }
    .btn-close { filter: invert(1); opacity: .4; }
    .btn-close:hover { opacity: 1; }

    /* ── Badges ────────────────────────────────────── */
    .badge { font-family: var(--font-mono); font-size: 10px; font-weight: 500; letter-spacing: .5px; border-radius: var(--radius-xs); }
    .text-bg-primary   { background: var(--surface-high) !important; color: var(--secondary) !important; }
    .text-bg-info      { background: var(--surface-high) !important; color: var(--secondary) !important; }
    .text-bg-success   { background: rgba(166,255,0,.10) !important; color: var(--primary) !important; }
    .text-bg-warning   { background: rgba(255,138,0,.10) !important; color: var(--caution) !important; }
    .text-bg-danger    { background: rgba(255,60,60,.10) !important; color: #ff7070 !important; }
    .text-bg-secondary { background: var(--surface-high) !important; color: var(--on-surface-muted) !important; }

    /* ── Color utilities ───────────────────────────── */
    .text-primary { color: var(--primary) !important; }
    .text-success { color: var(--primary-dim) !important; }
    .text-warning { color: var(--caution) !important; }
    .text-info    { color: var(--secondary) !important; }
    .text-muted   { color: var(--on-surface-muted) !important; }
    .text-danger  { color: #ff7070 !important; }

    /* ── Scrollbars ────────────────────────────────── */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--surface-high); border-radius: 0; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary-dark); }
  </style>
</head>
<body>
<div class="d-flex">
  <aside class="sidebar d-flex flex-column">
    <a class="sidebar-brand mb-4" href="{{ url_for('dashboard') }}">
      <span class="brand-logo">AI</span>
      <span>
        Digest
        <span class="brand-sub">// news aggregator</span>
      </span>
    </a>
    <nav class="nav flex-column gap-1 mb-auto">
      {%- for name, icon, label in [
          ('dashboard', 'bi-speedometer2',  'Dashboard'),
          ('feeds',     'bi-rss',           'Feeds'),
          ('topics',    'bi-tags',          'Topics'),
          ('settings',  'bi-sliders',       'Settings'),
          ('history',   'bi-clock-history', 'History'),
          ('logs',      'bi-terminal',      'Logs'),
        ] %}
      <a class="nav-link{{ ' active' if page == name }}" href="{{ url_for(name) }}">
        <i class="bi {{ icon }}"></i>{{ label }}
      </a>
      {%- endfor %}
    </nav>
    <form method="post" action="{{ url_for('run_digest') }}" class="mt-3">
      <button class="btn btn-success w-100 btn-sm">
        <i class="bi bi-play-fill"></i> Run now
      </button>
    </form>
  </aside>

  <div class="main">
    {%- for cat, msg in get_flashed_messages(with_categories=true) %}
    <div class="alert alert-{{ 'danger' if cat == 'error' else 'success' }} alert-dismissible fade show py-2 mb-3">
      {{ msg }}<button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    </div>
    {%- endfor %}
    {{ content | safe }}
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""


CAT_COLORS = {
    "industry": "primary", "news": "info", "research": "warning",
    "community": "success", "asia": "danger", "open-source": "secondary",
}
ALL_CATEGORIES = ["industry", "news", "research", "community", "asia", "open-source"]


def render_page(page: str, content: str) -> str:
    return render_template_string(BASE_HTML, page=page, content=content)


# ─── Dashboard ───────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    cfg = load_config()
    n_feeds = len(cfg.get("feeds", []))
    n_topics = len(cfg.get("topics", []))
    llm = cfg.get("llm", {})
    backend = llm.get("backend", "?")
    filter_model = llm.get("filter_model", "?")
    summary_model = llm.get("summary_model", "?")
    top_n = llm.get("top_n", "?")

    last_run = "No digests yet."
    n_digests = 0
    try:
        db = Path(cfg["general"]["db_path"]).expanduser()
        if db.exists():
            con = sqlite3.connect(str(db))
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT created_at, article_count, status FROM digests ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                last_run = f"{row['created_at']}  ·  {row['article_count']} articles  ·  {row['status']}"
            n_digests = con.execute("SELECT COUNT(*) FROM digests").fetchone()[0]
            con.close()
    except Exception:
        pass

    content = f"""
<h4 class="mb-4">Dashboard</h4>
<div class="row g-3 mb-4">
  <div class="col-sm-6 col-lg-3">
    <div class="card p-3 text-center">
      <div class="fs-1 fw-bold text-primary">{n_feeds}</div>
      <div class="text-muted small">RSS Feeds</div>
    </div>
  </div>
  <div class="col-sm-6 col-lg-3">
    <div class="card p-3 text-center">
      <div class="fs-1 fw-bold text-success">{n_topics}</div>
      <div class="text-muted small">Topics</div>
    </div>
  </div>
  <div class="col-sm-6 col-lg-3">
    <div class="card p-3 text-center">
      <div class="fs-1 fw-bold text-warning">{top_n}</div>
      <div class="text-muted small">Top-N articles</div>
    </div>
  </div>
  <div class="col-sm-6 col-lg-3">
    <div class="card p-3 text-center">
      <div class="fs-1 fw-bold text-info">{n_digests}</div>
      <div class="text-muted small">Digests archived</div>
    </div>
  </div>
</div>
<div class="row g-3">
  <div class="col-md-6">
    <div class="card">
      <div class="card-header">LLM Configuration</div>
      <div class="card-body">
        <table class="table table-sm mb-0">
          <tr><td class="text-muted w-50">Backend</td><td><code>{e(backend)}</code></td></tr>
          <tr><td class="text-muted">Filter model</td><td><code>{e(filter_model)}</code></td></tr>
          <tr><td class="text-muted">Summary model</td><td><code>{e(summary_model)}</code></td></tr>
        </table>
        <a href="{url_for('settings')}" class="btn btn-outline-secondary btn-sm mt-3">Edit settings</a>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card">
      <div class="card-header">Last Run</div>
      <div class="card-body">
        <p class="mb-3">{e(last_run)}</p>
        <a href="{url_for('history')}" class="btn btn-outline-secondary btn-sm">View history</a>
      </div>
    </div>
  </div>
</div>"""
    return render_page("dashboard", content)


# ─── Feeds ───────────────────────────────────────────────────────────────────

@app.route("/feeds")
def feeds():
    cfg = load_config()
    feed_list = cfg.get("feeds", [])

    rows = ""
    for i, f in enumerate(feed_list):
        color = CAT_COLORS.get(f.get("category", ""), "secondary")
        rows += f"""
<tr>
  <td class="text-muted">{i + 1}</td>
  <td>{e(f.get('name', ''))}</td>
  <td><code class="small">{e(f.get('url', ''))}</code></td>
  <td><span class="badge text-bg-{color}">{e(f.get('category', ''))}</span></td>
  <td>
    <a href="{url_for('feed_edit', idx=i)}" class="btn btn-outline-secondary btn-sm">Edit</a>
    <form method="post" action="{url_for('feed_delete', idx=i)}" class="d-inline"
          onsubmit="return confirm('Delete this feed?')">
      <button class="btn btn-outline-danger btn-sm">Delete</button>
    </form>
  </td>
</tr>"""

    cat_opts = "".join(f'<option value="{c}">{c}</option>' for c in ALL_CATEGORIES)

    content = f"""
<div class="d-flex justify-content-between align-items-center mb-3">
  <h4 class="mb-0">RSS Feeds <span class="badge text-bg-secondary">{len(feed_list)}</span></h4>
</div>
<div class="card mb-4">
  <div class="card-header">Add Feed</div>
  <div class="card-body">
    <form method="post" action="{url_for('feed_add')}">
      <div class="row g-2">
        <div class="col-md-3">
          <input class="form-control form-control-sm" name="name" placeholder="Name" required>
        </div>
        <div class="col-md-6">
          <input class="form-control form-control-sm" name="url" type="url" placeholder="https://..." required>
        </div>
        <div class="col-md-2">
          <select class="form-select form-select-sm" name="category">{cat_opts}</select>
        </div>
        <div class="col-md-1">
          <button class="btn btn-primary btn-sm w-100">Add</button>
        </div>
      </div>
    </form>
  </div>
</div>
<div class="card">
  <div class="table-responsive">
    <table class="table table-hover align-middle mb-0">
      <thead><tr><th>#</th><th>Name</th><th>URL</th><th>Category</th><th>Actions</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""
    return render_page("feeds", content)


@app.route("/feeds/add", methods=["POST"])
def feed_add():
    cfg = load_config()
    cfg.setdefault("feeds", []).append({
        "name": request.form["name"].strip(),
        "url": request.form["url"].strip(),
        "category": request.form.get("category", "news"),
    })
    save_config(cfg)
    flash(f"Feed '{request.form['name']}' added.")
    return redirect(url_for("feeds"))


@app.route("/feeds/<int:idx>/edit", methods=["GET", "POST"])
def feed_edit(idx: int):
    cfg = load_config()
    feed_list = cfg.get("feeds", [])
    if idx >= len(feed_list):
        flash("Feed not found.", "error")
        return redirect(url_for("feeds"))

    if request.method == "POST":
        feed_list[idx] = {
            "name": request.form["name"].strip(),
            "url": request.form["url"].strip(),
            "category": request.form.get("category", "news"),
        }
        save_config(cfg)
        flash("Feed updated.")
        return redirect(url_for("feeds"))

    f = feed_list[idx]
    cat_opts = "".join(
        f'<option value="{c}"{"selected" if c == f.get("category") else ""}>{c}</option>'
        for c in ALL_CATEGORIES
    )
    content = f"""
<h4 class="mb-3">Edit Feed</h4>
<div class="card" style="max-width:600px">
  <div class="card-body">
    <form method="post">
      <div class="mb-3">
        <label class="form-label">Name</label>
        <input class="form-control" name="name" value="{e(f.get('name', ''))}" required>
      </div>
      <div class="mb-3">
        <label class="form-label">URL</label>
        <input class="form-control" name="url" type="url" value="{e(f.get('url', ''))}" required>
      </div>
      <div class="mb-3">
        <label class="form-label">Category</label>
        <select class="form-select" name="category">{cat_opts}</select>
      </div>
      <div class="d-flex gap-2">
        <button class="btn btn-primary">Save</button>
        <a href="{url_for('feeds')}" class="btn btn-outline-secondary">Cancel</a>
      </div>
    </form>
  </div>
</div>"""
    return render_page("feeds", content)


@app.route("/feeds/<int:idx>/delete", methods=["POST"])
def feed_delete(idx: int):
    cfg = load_config()
    feed_list = cfg.get("feeds", [])
    if 0 <= idx < len(feed_list):
        removed = feed_list.pop(idx)
        save_config(cfg)
        flash(f"Feed '{removed.get('name', '')}' deleted.")
    return redirect(url_for("feeds"))


# ─── Topics ──────────────────────────────────────────────────────────────────

@app.route("/topics")
def topics():
    cfg = load_config()
    topic_list = cfg.get("topics", [])

    cards = ""
    for i, t in enumerate(topic_list):
        kws = t.get("keywords", [])
        kw_preview = e(", ".join(kws[:6]))
        if len(kws) > 6:
            kw_preview += f' <span class="text-muted">+{len(kws) - 6} more</span>'
        weight = t.get("weight", 1.0)
        wcolor = "success" if weight >= 0.9 else ("warning" if weight >= 0.7 else "secondary")
        cards += f"""
<div class="col-md-6">
  <div class="card h-100">
    <div class="card-header d-flex justify-content-between align-items-center">
      <span>{e(t.get('name', ''))}</span>
      <span class="badge text-bg-{wcolor}">weight: {weight}</span>
    </div>
    <div class="card-body">
      <p class="small text-muted mb-0">{len(kws)} keywords: {kw_preview}</p>
    </div>
    <div class="card-footer bg-transparent d-flex gap-2">
      <a href="{url_for('topic_edit', idx=i)}" class="btn btn-outline-secondary btn-sm">Edit</a>
      <form method="post" action="{url_for('topic_delete', idx=i)}"
            onsubmit="return confirm('Delete this topic?')">
        <button class="btn btn-outline-danger btn-sm">Delete</button>
      </form>
    </div>
  </div>
</div>"""

    content = f"""
<div class="d-flex justify-content-between align-items-center mb-3">
  <h4 class="mb-0">Topics <span class="badge text-bg-secondary">{len(topic_list)}</span></h4>
  <a href="{url_for('topic_new')}" class="btn btn-primary btn-sm">+ Add Topic</a>
</div>
<div class="row g-3">{cards}</div>"""
    return render_page("topics", content)


def _topic_form(action: str, topic: dict = None) -> str:
    t = topic or {}
    keywords_text = "\n".join(t.get("keywords", []))
    return f"""
<div class="card" style="max-width:600px">
  <div class="card-body">
    <form method="post" action="{action}">
      <div class="mb-3">
        <label class="form-label">Topic name</label>
        <input class="form-control" name="name" value="{e(t.get('name', ''))}" required>
      </div>
      <div class="mb-3">
        <label class="form-label">Weight <span class="text-muted">(0.0 – 1.0)</span></label>
        <input class="form-control" name="weight" type="number" step="0.1" min="0" max="1"
               value="{e(t.get('weight', 1.0))}" required>
      </div>
      <div class="mb-3">
        <label class="form-label">Keywords <span class="text-muted">(one per line)</span></label>
        <textarea class="form-control kw" name="keywords" rows="12">{e(keywords_text)}</textarea>
      </div>
      <div class="d-flex gap-2">
        <button class="btn btn-primary">Save</button>
        <a href="{url_for('topics')}" class="btn btn-outline-secondary">Cancel</a>
      </div>
    </form>
  </div>
</div>"""


@app.route("/topics/new", methods=["GET"])
def topic_new():
    content = f"<h4 class='mb-3'>Add Topic</h4>{_topic_form(url_for('topic_create'))}"
    return render_page("topics", content)


@app.route("/topics/new", methods=["POST"])
def topic_create():
    cfg = load_config()
    name = request.form.get("name", "").strip()
    weight = float(request.form.get("weight", 1.0))
    keywords = [k.strip() for k in request.form.get("keywords", "").splitlines() if k.strip()]
    cfg.setdefault("topics", []).append({"name": name, "weight": weight, "keywords": keywords})
    save_config(cfg)
    flash(f"Topic '{name}' added.")
    return redirect(url_for("topics"))


@app.route("/topics/<int:idx>/edit", methods=["GET", "POST"])
def topic_edit(idx: int):
    cfg = load_config()
    topic_list = cfg.get("topics", [])
    if idx >= len(topic_list):
        flash("Topic not found.", "error")
        return redirect(url_for("topics"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        weight = float(request.form.get("weight", 1.0))
        keywords = [k.strip() for k in request.form.get("keywords", "").splitlines() if k.strip()]
        topic_list[idx] = {"name": name, "weight": weight, "keywords": keywords}
        save_config(cfg)
        flash(f"Topic '{name}' updated.")
        return redirect(url_for("topics"))

    content = f"<h4 class='mb-3'>Edit Topic</h4>{_topic_form(url_for('topic_edit', idx=idx), topic_list[idx])}"
    return render_page("topics", content)


@app.route("/topics/<int:idx>/delete", methods=["POST"])
def topic_delete(idx: int):
    cfg = load_config()
    topic_list = cfg.get("topics", [])
    if 0 <= idx < len(topic_list):
        removed = topic_list.pop(idx)
        save_config(cfg)
        flash(f"Topic '{removed.get('name', '')}' deleted.")
    return redirect(url_for("topics"))


# ─── Settings ────────────────────────────────────────────────────────────────

@app.route("/settings", methods=["GET", "POST"])
def settings():
    cfg = load_config()

    if request.method == "POST":
        f = request.form
        llm = cfg.setdefault("llm", {})
        llm["backend"] = f.get("backend", "openrouter")
        llm["filter_model"] = f.get("filter_model", "").strip()
        llm["summary_model"] = f.get("summary_model", "").strip()
        if f.get("api_key"):
            llm["api_key"] = f.get("api_key", "").strip()
        if f.get("base_url"):
            llm["base_url"] = f.get("base_url", "").strip()
        llm["top_n"] = int(f.get("top_n", 15))
        llm["min_relevance"] = float(f.get("min_relevance", 4))
        llm["timeout"] = int(f.get("timeout", 120))

        email = cfg.setdefault("email", {})
        email["smtp_host"] = f.get("smtp_host", "").strip()
        email["smtp_port"] = int(f.get("smtp_port", 587))
        email["smtp_user"] = f.get("smtp_user", "").strip()
        if f.get("smtp_password"):
            email["smtp_password"] = f.get("smtp_password", "").strip()
        email["use_tls"] = f.get("use_tls") == "true"
        email["from_addr"] = f.get("from_addr", "").strip()
        email["to_addrs"] = [a.strip() for a in f.get("to_addrs", "").splitlines() if a.strip()]
        email["subject"] = f.get("subject", "").strip()

        general = cfg.setdefault("general", {})
        general["lookback_days"] = int(f.get("lookback_days", 1))
        general["max_per_feed"] = int(f.get("max_per_feed", 30))
        general["log_level"] = f.get("log_level", "INFO")

        save_config(cfg)
        flash("Settings saved.")
        return redirect(url_for("settings"))

    llm = cfg.get("llm", {})
    email = cfg.get("email", {})
    general = cfg.get("general", {})
    backend = llm.get("backend", "openrouter")
    to_addrs = "\n".join(email.get("to_addrs", []))
    log_opts = "".join(
        f'<option {"selected" if general.get("log_level", "INFO") == l else ""}>{l}</option>'
        for l in ["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    def radio(val):
        return "checked" if backend == val else ""

    def tls_sel(val):
        return "selected" if str(email.get("use_tls", True)).lower() == val else ""

    content = f"""
<h4 class="mb-3">Settings</h4>
<form method="post">

  <div class="card mb-3">
    <div class="card-header">LLM</div>
    <div class="card-body row g-3">
      <div class="col-12">
        <label class="form-label">Backend</label><br>
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="radio" name="backend" value="openrouter" {radio('openrouter')} id="or">
          <label class="form-check-label" for="or">OpenRouter (cloud)</label>
        </div>
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="radio" name="backend" value="ollama" {radio('ollama')} id="ol">
          <label class="form-check-label" for="ol">Ollama (local)</label>
        </div>
      </div>
      <div class="col-md-6">
        <label class="form-label">OpenRouter API key</label>
        <input class="form-control form-control-sm" name="api_key" type="password"
               placeholder="leave blank to keep current">
      </div>
      <div class="col-md-6">
        <label class="form-label">Ollama base URL</label>
        <input class="form-control form-control-sm" name="base_url"
               placeholder="http://localhost:11434" value="{e(llm.get('base_url', ''))}">
      </div>
      <div class="col-md-6">
        <label class="form-label">Filter model (Stage 1)</label>
        <input class="form-control form-control-sm" name="filter_model"
               value="{e(llm.get('filter_model', ''))}">
      </div>
      <div class="col-md-6">
        <label class="form-label">Summary model (Stage 2)</label>
        <input class="form-control form-control-sm" name="summary_model"
               value="{e(llm.get('summary_model', ''))}">
      </div>
      <div class="col-md-4">
        <label class="form-label">Top-N articles</label>
        <input class="form-control form-control-sm" name="top_n" type="number"
               min="1" max="100" value="{e(llm.get('top_n', 15))}">
      </div>
      <div class="col-md-4">
        <label class="form-label">Min relevance (0–10)</label>
        <input class="form-control form-control-sm" name="min_relevance" type="number"
               step="0.5" min="0" max="10" value="{e(llm.get('min_relevance', 4))}">
      </div>
      <div class="col-md-4">
        <label class="form-label">Timeout (s)</label>
        <input class="form-control form-control-sm" name="timeout" type="number"
               min="10" max="600" value="{e(llm.get('timeout', 120))}">
      </div>
    </div>
  </div>

  <div class="card mb-3">
    <div class="card-header">Email / SMTP</div>
    <div class="card-body row g-3">
      <div class="col-md-6">
        <label class="form-label">SMTP host</label>
        <input class="form-control form-control-sm" name="smtp_host" value="{e(email.get('smtp_host', ''))}">
      </div>
      <div class="col-md-3">
        <label class="form-label">Port</label>
        <input class="form-control form-control-sm" name="smtp_port" type="number"
               value="{e(email.get('smtp_port', 587))}">
      </div>
      <div class="col-md-3">
        <label class="form-label">TLS</label>
        <select class="form-select form-select-sm" name="use_tls">
          <option value="true" {tls_sel('true')}>Yes</option>
          <option value="false" {tls_sel('false')}>No</option>
        </select>
      </div>
      <div class="col-md-6">
        <label class="form-label">SMTP user</label>
        <input class="form-control form-control-sm" name="smtp_user" value="{e(email.get('smtp_user', ''))}">
      </div>
      <div class="col-md-6">
        <label class="form-label">SMTP password <span class="text-muted">(leave blank to keep)</span></label>
        <input class="form-control form-control-sm" name="smtp_password" type="password" placeholder="unchanged">
      </div>
      <div class="col-md-6">
        <label class="form-label">From address</label>
        <input class="form-control form-control-sm" name="from_addr" value="{e(email.get('from_addr', ''))}">
      </div>
      <div class="col-md-6">
        <label class="form-label">To addresses <span class="text-muted">(one per line)</span></label>
        <textarea class="form-control form-control-sm" name="to_addrs" rows="2">{e(to_addrs)}</textarea>
      </div>
      <div class="col-12">
        <label class="form-label">Subject</label>
        <input class="form-control form-control-sm" name="subject" value="{e(email.get('subject', ''))}">
      </div>
    </div>
  </div>

  <div class="card mb-3">
    <div class="card-header">General</div>
    <div class="card-body row g-3">
      <div class="col-md-4">
        <label class="form-label">Lookback days</label>
        <input class="form-control form-control-sm" name="lookback_days" type="number"
               min="1" max="30" value="{e(general.get('lookback_days', 1))}">
      </div>
      <div class="col-md-4">
        <label class="form-label">Max articles per feed</label>
        <input class="form-control form-control-sm" name="max_per_feed" type="number"
               min="1" max="200" value="{e(general.get('max_per_feed', 30))}">
      </div>
      <div class="col-md-4">
        <label class="form-label">Log level</label>
        <select class="form-select form-select-sm" name="log_level">{log_opts}</select>
      </div>
    </div>
  </div>

  <button class="btn btn-primary">Save all settings</button>
</form>"""
    return render_page("settings", content)


# ─── History ─────────────────────────────────────────────────────────────────

@app.route("/history")
def history():
    cfg = load_config()
    db = Path(cfg["general"]["db_path"]).expanduser()
    rows_html = ""

    if db.exists():
        try:
            con = sqlite3.connect(str(db))
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT id, created_at, article_count, status FROM digests ORDER BY id DESC LIMIT 50"
            ).fetchall()
            con.close()
            for r in rows:
                rows_html += f"""
<tr>
  <td>{r['id']}</td>
  <td>{e(r['created_at'])}</td>
  <td>{r['article_count'] or '?'}</td>
  <td><span class="badge text-bg-{'success' if r['status']=='sent' else 'secondary'}">{e(r['status'])}</span></td>
  <td>
    <a href="{url_for('view_digest', digest_id=r['id'])}" class="btn btn-outline-primary btn-sm" target="_blank">
      View
    </a>
  </td>
</tr>"""
        except Exception as ex:
            rows_html = f'<tr><td colspan="5" class="text-danger">DB error: {e(ex)}</td></tr>'
    else:
        rows_html = '<tr><td colspan="5" class="text-muted">No database yet – run the digest first.</td></tr>'

    content = f"""
<h4 class="mb-3">Digest History</h4>
<div class="card">
  <div class="table-responsive">
    <table class="table table-hover align-middle mb-0">
      <thead><tr><th>#</th><th>Date</th><th>Articles</th><th>Status</th><th>View</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>"""
    return render_page("history", content)


@app.route("/history/<int:digest_id>")
def view_digest(digest_id: int):
    cfg = load_config()
    db = Path(cfg["general"]["db_path"]).expanduser()
    if not db.exists():
        return "Database not found.", 404
    try:
        con = sqlite3.connect(str(db))
        row = con.execute("SELECT html FROM digests WHERE id = ?", (digest_id,)).fetchone()
        con.close()
        if not row:
            return "Digest not found.", 404
        return make_response(row[0] or "", 200, {"Content-Type": "text/html; charset=utf-8"})
    except Exception as ex:
        return f"Error: {ex}", 500


def _log_path() -> Path:
    cfg = load_config()
    return Path(cfg["general"]["db_path"]).expanduser().parent / "last_run.log"


# ─── Logs ─────────────────────────────────────────────────────────────────────

@app.route("/logs")
def logs():
    content = """
<div class="d-flex justify-content-between align-items-center mb-3">
  <h4 class="mb-0">Last Run Log</h4>
  <button onclick="clearLog()" class="btn btn-outline-danger btn-sm">Clear</button>
</div>
<div class="card">
  <div class="card-header d-flex justify-content-between align-items-center">
    <span>OUTPUT</span>
    <span id="ts" style="font-family:var(--font-mono);font-size:10px;color:var(--on-surface-faint)">—</span>
  </div>
  <div class="card-body p-0">
    <pre id="log-out" style="background:var(--surface-lowest);color:var(--primary);font-family:var(--font-mono);
         font-size:12px;line-height:1.6;padding:1rem;margin:0;min-height:400px;max-height:72vh;
         overflow-y:auto;border-radius:0 0 var(--radius-md) var(--radius-md);
         white-space:pre-wrap;word-break:break-all;"></pre>
  </div>
</div>
<script>
  let lastLen = 0;
  const pre = document.getElementById('log-out');
  const ts  = document.getElementById('ts');
  function fetchLog() {
    fetch('/logs/content').then(r => r.text()).then(text => {
      if (text.length !== lastLen) {
        const atBottom = pre.scrollHeight - pre.scrollTop <= pre.clientHeight + 60;
        pre.textContent = text || '(no log yet — run the digest first)';
        lastLen = text.length;
        if (atBottom) pre.scrollTop = pre.scrollHeight;
      }
      ts.textContent = new Date().toLocaleTimeString();
    }).catch(() => { ts.textContent = 'error'; });
  }
  fetchLog();
  setInterval(fetchLog, 2000);
  function clearLog() {
    fetch('/logs/clear', {method:'POST'}).then(fetchLog);
  }
</script>"""
    return render_page("logs", content)


@app.route("/logs/content")
def logs_content():
    p = _log_path()
    if not p.exists():
        return "", 200, {"Content-Type": "text/plain; charset=utf-8"}
    try:
        return p.read_text(encoding="utf-8", errors="replace"), 200, {"Content-Type": "text/plain; charset=utf-8"}
    except Exception as ex:
        return f"Error reading log: {ex}", 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/logs/clear", methods=["POST"])
def logs_clear():
    try:
        _log_path().write_text("", encoding="utf-8")
    except Exception:
        pass
    return "", 204


# ─── Run digest ──────────────────────────────────────────────────────────────

@app.route("/run", methods=["POST"])
def run_digest():
    if SCRIPT_PATH and SCRIPT_PATH.exists():
        try:
            log_file = open(_log_path(), "w", encoding="utf-8")
            subprocess.Popen(
                [sys.executable, str(SCRIPT_PATH), "-c", str(CONFIG_PATH)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log_file.close()
            flash("Digest started — watch progress in Logs.")
        except Exception as ex:
            flash(f"Failed to start digest: {ex}", "error")
    else:
        flash("Could not find ai_digest.py. Trigger it manually instead.", "error")
    return redirect(url_for("dashboard"))


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main():
    global CONFIG_PATH, SCRIPT_PATH

    parser = argparse.ArgumentParser(description="AI Digest Web UI")
    parser.add_argument("-c", "--config", default="src/config.local.yaml",
                        help="Path to config YAML (default: src/config.local.yaml)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host (use 0.0.0.0 for LAN access, default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    args = parser.parse_args()

    CONFIG_PATH = Path(args.config).expanduser().resolve()
    if not CONFIG_PATH.exists():
        print(f"Error: config file not found: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    SCRIPT_PATH = Path(__file__).parent / "ai_digest.py"

    print(f"AI Digest Web UI → http://{args.host}:{args.port}")
    print(f"Config: {CONFIG_PATH}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
