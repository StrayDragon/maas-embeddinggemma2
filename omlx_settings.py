"""oMLX server settings (profile ``omlx``).

Talks to a remote/local oMLX instance that serves EmbeddingGemma 2
(model id ``embeddinggemma-2-bf16``). The key lives in ``.env`` (git-ignored);
``.env.example`` carries the shape.

Env (prefix OMLX_*):
  OMLX_BASE_URL   server root, e.g. http://coral:11534
  OMLX_API_KEY    bearer token for /v1/*
  OMLX_MODEL      model id served by oMLX
  OMLX_TIMEOUT_S  per-request timeout
"""

from __future__ import annotations

import os

DEFAULT_BASE_URL = "http://127.0.0.1:11534"
DEFAULT_MODEL = "embeddinggemma-2-bf16"
DEFAULT_TIMEOUT = 120.0


def base_url() -> str:
    return os.getenv("OMLX_BASE_URL", DEFAULT_BASE_URL).rstrip("/") or DEFAULT_BASE_URL


def api_key() -> str:
    return os.getenv("OMLX_API_KEY", "")


def model() -> str:
    return os.getenv("OMLX_MODEL", DEFAULT_MODEL)


def timeout() -> float:
    try:
        return float(os.getenv("OMLX_TIMEOUT_S", str(DEFAULT_TIMEOUT)))
    except ValueError:
        return DEFAULT_TIMEOUT


def headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    key = api_key()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h
