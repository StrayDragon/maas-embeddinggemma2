#!/usr/bin/env bash
# Export EMBG2_BASE_URL/EMBG2_MODEL(+auth via base.ts env fallback) for the
# sdk-test suites, from the same sources the server uses (endpoint.py/config).
# Usage: scripts/with_target_env.sh <command...>
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
py="$repo_root/.venv/bin/python"
[ -x "$py" ] || { echo "no .venv -- just setup <profile>"; exit 1; }
url=$(PYTHONPATH="$repo_root" "$py" -c "from endpoint import resolve_base_url; print(resolve_base_url())")
export EMBG2_BASE_URL="$url"
if PYTHONPATH="$repo_root" "$py" -c "import sys; from config import ACTIVE_PROFILE; sys.exit(0 if ACTIVE_PROFILE=='ommlx' else 1)" 2>/dev/null; then
  # shellcheck disable=SC2046
  export $(PYTHONPATH="$repo_root" "$py" -c 'from config import settings; print("EMBG2_MODEL=%s" % settings.model_path)')
fi
if [ "$#" -gt 0 ]; then exec "$@"; fi
