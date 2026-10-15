#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# machine config lives in .env: PROFILE picks the profile, overrides use its
# env-prefix (CPU_* / STRIX_HALO_*); real env vars still win (pydantic precedence)
set -a; [ -f .env ] && . ./.env; set +a
[ -x .venv/bin/python ] || { echo "no .venv -- run: just setup cpu|strix-halo"; exit 1; }
# host/port resolved profile-aware by config.py; bare HOST/PORT env (one-off
# override, not in .env anymore) still wins if exported
read -r C_HOST C_PORT < <(.venv/bin/python -c 'from config import settings; print(settings.host, settings.port)')
HOST="${HOST:-$C_HOST}"
PORT="${PORT:-$C_PORT}"
exec .venv/bin/python -m uvicorn server:app \
  --host "$HOST" \
  --port "$PORT" \
  ${RELOAD:+--reload}
