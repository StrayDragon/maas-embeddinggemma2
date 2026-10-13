#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# machine-specific config lives in .env (see .env.example); real env vars win
set -a; [ -f .env ] && . ./.env; set +a
exec .venv/bin/python -m uvicorn server:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  ${RELOAD:+--reload}
