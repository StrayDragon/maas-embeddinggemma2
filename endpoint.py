"""Resolve the running embedding service's base URL for clients / tests.

Precedence:
  1. ``EMBG2_BASE_URL`` env          -- explicit override (remote host, custom port)
  2. ``config.py`` settings          -- profile-aware `.env` (0.0.0.0 dialled as 127.0.0.1)
  3. well-known default ``http://127.0.0.1:50233``

Every candidate is verified against ``GET /health`` (auto-discovery); the first
that answers wins. On total failure: friendly hint on stderr, exit 1.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request


def _candidates() -> list[str]:
    out: list[str] = []
    env = os.getenv("EMBG2_BASE_URL", "").strip()
    if env:
        out.append(env.rstrip("/"))
    try:
        from config import settings

        # omlx profile: the "service" IS the remote oMLX server
        if hasattr(settings, "resolved_base_url"):
            out.append(settings.resolved_base_url())
        host = settings.host
        if host in ("", "0.0.0.0", "::"):
            host = "127.0.0.1"
        out.append(f"http://{host}:{settings.port}")
    except Exception:  # noqa: BLE001 -- no .env/.venv yet; still try the default
        pass
    out.append("http://127.0.0.1:50233")
    seen: set[str] = set()
    return [u for u in out if u not in seen and not seen.add(u)]


def _probe(base: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as r:
            json.load(r)  # any JSON body counts as our service
        return True
    except Exception:  # noqa: BLE001
        return False


def resolve_base_url(*, timeout: float = 1.5, exit_on_fail: bool = True) -> str | None:
    """Return a reachable base URL; on failure print a friendly hint (exit 1 if exit_on_fail)."""
    tried: list[str] = []
    for base in _candidates():
        tried.append(base)
        if _probe(base, timeout):
            return base
    if not exit_on_fail:
        return None
    print(f"✗ 未发现 embedding 服务(探测过: {', '.join(tried)})", file=sys.stderr)
    print("  本机启动:   just serve            # 仓库根目录", file=sys.stderr)
    print("  远程/自定义: EMBG2_BASE_URL=http://host:port 后重试", file=sys.stderr)
    sys.exit(1)
