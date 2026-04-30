#!/bin/bash
# Datei: /home/zeroclaw/ai_digest/run_ai_digest.sh

# 1. In das Arbeitsverzeichnis wechseln
# Falls das Verzeichnis nicht existiert, bricht das Skript direkt ab (exit 1)
cd /home/zeroclaw/ai_digest/src || exit 1

# 2. Das Python-Skript ausführen
/home/zeroclaw/ai_digest/src/venv/bin/python ai_digest.py
