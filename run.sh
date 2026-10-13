#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn server:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}"
