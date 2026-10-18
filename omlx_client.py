"""Back-compat facade over :mod:`recipes.omlx_backend` (the implementation).

Historical import surface kept so existing scripts/tests keep working; new
code should use the registry::

    from recipes import get_backend
    backend = get_backend()          # follows the active profile

Everything here delegates; the single source of truth for request shaping,
media conversion and the capability notes lives in
``recipes/omlx_backend.py`` (see its docstring for the 0.7.1rc1 audio/video
caveats).
"""

from __future__ import annotations

import urllib.error
from pathlib import Path
from typing import Any

from recipes.omlx_backend import OmlxBackendError
from recipes.omlx_backend import (
    embed_item as _embed_item_core,
    embed_batch as _embed_batch_core,
    embed_text as _embed_text_core,
    _embed_request,
)

OmlxError = OmlxBackendError
OmlxError = OmlxBackendError  # legacy alias


from config import settings as settings  # re-exported: tests patch oc.settings

def _request(path: str, payload: dict | None = None, method: str = "POST", *, timeout_s: float | None = None):
    """Legacy raw request (tests use it); routes through recipes.omlx_backend._post-alike."""
    import urllib.request

    base = settings.resolved_base_url()
    req = urllib.request.Request(f"{base}{path}", method=method)
    if settings.api_key:
        req.add_header("Authorization", f"Bearer {settings.api_key}")
    data = None
    if payload is not None:
        import json as _json

        data = _json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    timeout = timeout_s if timeout_s is not None else min(settings.timeout_s, 60)
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
            return json_load(resp.read())
    except urllib.error.HTTPError as e:
        raise OmlxBackendError(e.code, e.read().decode(errors="replace")[:400], path) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise OmlxBackendError(0, f"unreachable ({e})", path) from e


def embed(inputs: Any, **kw) -> list[list[float]]:
    """Legacy multi-shape entry: str | [str] | item-dict | [item-dict] | parts."""
    if isinstance(inputs, str):
        return _embed_batch_core([inputs], task=kw.get("task"), dim=kw.get("dimensions"))
    if isinstance(inputs, dict) and "type" in inputs:  # single content part
        return [_embed_item_core(_parts_to_item([inputs]), dim=kw.get("dimensions") or kw.get("dim"))]
    if (
        isinstance(inputs, list)
        and inputs
        and all(isinstance(x, dict) and "type" in x for x in inputs)
    ):  # content parts describing ONE input
        return [_embed_item_core(_parts_to_item(inputs), dim=kw.get("dimensions") or kw.get("dim"))]
    if isinstance(inputs, list) and inputs and all(isinstance(x, str) for x in inputs):
        return _embed_batch_core(inputs, task=kw.get("task"), dim=kw.get("dimensions"))
    if isinstance(inputs, dict):
        return [_embed_item_core(inputs, dim=kw.get("dimensions") or kw.get("dim"))]
    if isinstance(inputs, list):
        return _embed_batch_items([_coerce(x) for x in inputs], dim=kw.get("dimensions"))
    raise ValueError(f"unsupported input shape: {type(inputs).__name__}")


def _coerce(x: Any) -> Any:
    return x  # items pass through; recipes.omlx_backend._to_item normalises


def _parts_to_item(parts: list[dict]) -> dict:
    """OpenAI content parts -> one item dict (server folds them itself when
    given via ``input``; for the items path we merge here)."""
    texts: list[str] = []
    image = audio = None
    for p in parts:
        t = p.get("type")
        if t == "text":
            texts.append(p.get("text", ""))
        elif t in ("image_url", "image", "input_image"):
            payload = p.get(t) if t != "image" else p.get("image")
            image = payload.get("url") if isinstance(payload, dict) else payload
        elif t == "input_audio":
            payload = p.get("input_audio")
            if isinstance(payload, dict):
                data = payload.get("data", "")
                fmt = payload.get("format", "wav")
                audio = data if data.startswith("data:") else f"data:audio/{fmt};base64,{data}"
            else:
                audio = payload
    item: dict[str, Any] = {}
    if any(texts):
        item["text"] = "\n".join(t for t in texts if t)
    if image:
        item["image"] = image
    if audio:
        item["audio"] = audio
    if not item:
        raise ValueError("content parts contain no text/image/audio")
    return item


def _embed_batch_items(items: list[dict], *, dim: int | None = None) -> list[list[float]]:
    from recipes.omlx_backend import _to_item

    return _embed_request(
        {"model": __import__("config").settings.model_path, "items": [_to_item(i) for i in items]},
        dim,
    )


def embed_batch(texts: list[str], **kw) -> list[list[float]]:
    return _embed_batch_core(texts, task=kw.get("task"), dim=kw.get("dimensions"))


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are unit-normalised


def models() -> list[dict]:
    import urllib.request

    from config import settings

    req = urllib.request.Request(f"{settings.resolved_base_url()}/v1/models", method="GET")
    key = settings.api_key
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=min(settings.timeout_s, 30)) as resp:
        return json_load(resp.read())["data"]


def json_load(raw: bytes) -> Any:
    import json

    return json.loads(raw)


def image_data_uri(img: Any) -> str:
    from recipes.omlx_backend import _image_value

    return _image_value(img)


def embed_text_task(text: str, task: str, dim: int | None = None) -> list[float]:
    return _embed_text_core(text, task=task, dim=dim)


__all__ = [
    "embed", "embed_batch", "embed_text_task", "cosine", "models",
    "image_data_uri", "OmlxError", "OmlxError", "OmlxBackendError",
]
