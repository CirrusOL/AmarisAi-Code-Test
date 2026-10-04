#!/usr/bin/env bash
cd "$(dirname "$0")"

# Create a virtual environment on first run (avoids "externally-managed-environment" pip errors)
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Open the browser after a short delay, so the server has time to start
(
  sleep 2
  if command -v open >/dev/null 2>&1; then
    open "http://127.0.0.1:5000/"        # macOS
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://127.0.0.1:5000/"    # Linux
  fi
) &

python app.py