"""Backend registry: one module per serving recipe, chosen by ``PROFILE``.

A *recipe* is a way of getting EmbeddingGemma 2 embeddings:

- ``local``  -- this repo's FastAPI server (``server.py``); full multimodal
               surface (``/embed``, task presets, video, DashScope shape)
- ``ommlx``  -- a remote/local `oMLX <https://github.com/jundot/omlx>`_
               instance serving the model; OpenAI-shape ``/v1/embeddings``
               with the subset documented in ``docs/OMMLX.md``

The active recipe follows ``config.ACTIVE_PROFILE`` (``cpu``/``strix-halo``
run the local server process; ``ommlx`` is a pure client of the remote API).
Clients, tests and helper scripts ask the registry instead of hand-rolling
per-backend branches:

    from recipes import get_backend

    backend = get_backend()
    backend.embed_text("a cat", task="search", dim=256)
    backend.embed_media(image="photo.jpg", text="caption")   # fusion input
    backend.capabilities   # degrade cleanly instead of try/except

Adding a recipe: create ``recipes/<name>.py`` exposing ``Backend``
(class or module-level functions), register it in ``_REGISTRY`` and, if it
needs config, in ``config.PROFILES`` / ``scripts/setup_env.sh``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

from config import ACTIVE_PROFILE


class Backend(Protocol):
    """The contract every recipe fulfils. All methods return lists of
    L2-normalised floats (or raise a recipe-specific error type that also
    subclasses :class:`BackendError`)."""

    name: str
    capabilities: dict[str, bool]

    def embed_text(
        self, text: str, *, task: str | None = None, dim: int | None = None
    ) -> list[float]: ...

    def embed_batch(
        self, texts: list[str], *, task: str | None = None, dim: int | None = None
    ) -> list[list[float]]: ...

    def embed_item(self, item: dict, *, dim: int | None = None) -> list[float]:
        """item: {"text"?: str|list[str], "image"?: path|PIL|bytes|dict,
        "audio"?: path|bytes|{array,sampling_rate}, "video"?: path}"""

    def embed_many(self, items: list, *, dim: int | None = None) -> list[list[float]]: ...

    def embed_file(self, path: str, *, kind: str | None = None, dim: int | None = None) -> list[float]: ...


class BackendError(RuntimeError):
    """Base for backend failures; backends subclass to carry status/detail."""

    status: int = 0


def _profile_backend_name() -> str:
    return "ommlx" if ACTIVE_PROFILE == "ommlx" else "local"


_REGISTRY: dict[str, str] = {
    "local": "recipes.local_backend",
    "ommlx": "recipes.ommlx_backend",
}


@lru_cache(maxsize=None)
def _load(name: str) -> Backend:
    import importlib
    import types

    module = importlib.import_module(_REGISTRY[name])
    backend = types.SimpleNamespace(
        name=getattr(module, "name", name),
        capabilities=dict(getattr(module, "capabilities", {})),
        embed_text=module.embed_text,
        embed_batch=module.embed_batch,
        embed_item=module.embed_item,
        embed_many=module.embed_many,
        embed_file=module.embed_file,
        error=getattr(module, f"{'Ommlx' if name == 'ommlx' else 'Local'}BackendError"),
    )
    return backend


def get_backend(name: str | None = None) -> Backend:
    """Return the backend for ``name`` (default: the active profile's).

    Unknown names raise with the valid choices; instantiation is cached so
    repeated calls share one lazy module import.
    """
    key = (name or _profile_backend_name()).lower()
    if key == "auto":
        key = _profile_backend_name()
    if key not in _REGISTRY:
        raise KeyError(f"unknown backend {key!r}; choose one of {sorted(_REGISTRY)}")
    return _load(key)


def active_backend_name() -> str:
    return _profile_backend_name()
