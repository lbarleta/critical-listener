#!/usr/bin/env bash
# Start the API from the repo root with PYTHONPATH + .env loaded.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PYTHONPATH="${PYTHONPATH:-.:lastfm-recommender}"

UVICORN="${ROOT}/app/.venv/bin/uvicorn"
if [[ ! -x "$UVICORN" ]]; then
  UVICORN=uvicorn
fi

exec "$UVICORN" app.main:app --reload --host 127.0.0.1 --port 8000
