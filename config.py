"""Central configuration (pydantic-settings).

Env precedence: real environment > .env file > field defaults.
Regenerate `.env.example` after changing this file:

    just env-example   # or: .venv/bin/python scripts/gen_env_example.py
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model_path: str = "models/embeddinggemma-2"
    """Local path (or HF repo id) of the EmbeddingGemma 2 weights."""

    host: str = "0.0.0.0"
    """Bind address for the HTTP server (used by run.sh / just serve)."""

    port: int = 8000
    """TCP port for the HTTP server (used by run.sh / just serve)."""

    device: str = "auto"
    """torch device: auto | cuda | rocm | cpu."""

    dtype: str = "auto"
    """Model dtype: auto | bfloat16 | float32 | float16 (float32 is the safe default)."""

    log_level: str = "INFO"
    """Logging verbosity for the application."""


settings = Settings()
