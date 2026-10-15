"""Central configuration (pydantic-settings), profile-aware.

A "profile" bundles the per-machine defaults. The active profile is chosen with
PROFILE (real env > .env), and every other setting is read through that
profile's env-prefix -- so a single .env can carry override blocks for several
machines without interfering:

    PROFILE=strix-halo          # auto | cpu | strix-halo
    STRIX_HALO_DEVICE=auto      # read only while PROFILE=strix-halo
    CPU_DEVICE=cpu              # read only while PROFILE=cpu

PROFILE=auto imports torch once and picks: GPU present -> strix-halo, else cpu.
Pin PROFILE in .env to skip that probe at startup.

Single source of truth: regenerate .env.example (contains EVERY profile) via

    just env-example
"""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

PROFILES: tuple[str, ...] = ("cpu", "strix-halo")


def _cfg(prefix: str = "") -> SettingsConfigDict:
    return SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_prefix=prefix,
    )


class _ProfileSelector(BaseSettings):
    """Bootstrap: resolves which profile's prefix the main settings use."""

    model_config = _cfg()

    profile: str = "auto"
    """auto | cpu | strix-halo."""


class _Common(BaseSettings):
    """Shared fields; per-profile subclasses below only pin defaults/prefix."""

    model_config = _cfg()

    model_path: str = "models/embeddinggemma-2"
    """Local path (or HF repo id) of the EmbeddingGemma 2 weights."""

    host: str = "0.0.0.0"
    """Bind address for the HTTP server."""

    port: int = 50233
    """TCP port for the HTTP server."""

    device: str = "auto"
    """torch device: auto | cuda | cpu."""

    dtype: str = "auto"
    """Model dtype: auto | bfloat16 | float32 | float16."""

    log_level: str = "INFO"
    """Logging verbosity for the application."""


class CpuSettings(_Common):
    """Pure-CPU machines."""

    model_config = _cfg("CPU_")

    device: str = "cpu"
    dtype: str = "float32"


class StrixHaloSettings(_Common):
    """AMD Radeon 8060S (gfx1151, ROCm); fp32 default lives in server._resolve_dtype."""

    model_config = _cfg("STRIX_HALO_")


_PROFILE_CLS: dict[str, type[BaseSettings]] = {
    "cpu": CpuSettings,
    "strix-halo": StrixHaloSettings,
}


def _auto_profile() -> str:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return "cpu"
    return "strix-halo" if torch.cuda.is_available() else "cpu"


_selected = _ProfileSelector().profile.strip().lower()
if _selected == "auto":
    _selected = _auto_profile()
if _selected not in _PROFILE_CLS:
    raise SystemExit(f"PROFILE={_selected!r} invalid; choose one of {list(PROFILES)} or 'auto'")

ACTIVE_PROFILE: str = _selected
settings = _PROFILE_CLS[_selected]()
