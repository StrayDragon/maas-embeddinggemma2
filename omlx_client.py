"""Thin client for the oMLX embedding API (profile ``ommlx``).

One dependency (stdlib ``urllib`` + ``config``); no torch. The oMLX server
owns model loading and dtype; this module only shapes requests.

Endpoint coverage measured against oMLX 0.7.1rc1 (see docs/OMMLX.md):

- ``input``: str | list[str] | OpenAI content parts
    (``{"type":"text"}``, ``{"type":"image_url","image_url":{"url":...}}``,
    ``{"type":"input_audio",...}``) -- parts fold into ONE multimodal input.
- ``items``: list of ``{"text"?, "image"?, "audio"?}`` -- one dict per input;
    image must be a ``data:image/...;base64,...`` URI (URLs/paths rejected);
    audio needs ``embedding_audio_enabled`` in the model's server-side settings.
- ``dimensions``: MRL truncation, server re-normalizes (verified norm=1.0).
- No video support server-side (``items.video`` -> 422); encode frames to
    images client-side instead.

Usage:
    from omlx_client import embed, cosine
    vecs = embed(["hello", {"text": "cap", "image": data_uri}])
"""

from __future__ import annotations

import json
import os
import base64
import io
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Union

from config import settings

EmbedInput = Union[str, dict, list]


class OmmlxError(RuntimeError):
    """oMLX returned an error response (status + body kept on the exception)."""

    def __init__(self, status: int, body: Any, label: str = ""):
        self.status = status
        self.body = body
        super().__init__(f"oMLX {label or 'request'} failed: HTTP {status}: {body}")


def _load_dotenv() -> None:
    """Minimal .env loader (run.sh does `set -a` for the server; scripts may not)."""
    if os.getenv("OMLX_API_KEY"):
        return
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _request(path: str, payload: dict | None = None, method: str = "POST", *, timeout_s: float | None = None) -> Any:
    _load_dotenv()
    base_url = (
        settings.resolved_base_url()
        if hasattr(settings, "resolved_base_url")
        else settings.base_url
    )
    url = f"{base_url}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    key = settings.api_key or os.getenv("OMLX_API_KEY", "")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    timeout = timeout_s if timeout_s is not None else settings.timeout_s
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:  # pragma: no cover - network paths
        raise OmmlxError(e.code, e.read().decode(errors="replace")[:500], path) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise OmmlxError(0, f"unreachable ({e}); check OMLX_BASE_URL={base_url!r}", path) from e
    return json.loads(raw)


def _media_data_uri(source: str | Path, kind: str = "image") -> str:
    """file path / bytes / already-data-URI -> data URI (oMLX rejects URLs/paths)."""
    s = str(source)
    if s.startswith("data:"):
        return s
    raw = Path(s).read_bytes() if Path(s).exists() else s.encode()
    mime = mimetypes.guess_type(s)[0] or {"image": "image/png", "audio": "audio/wav"}[kind]
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _normalize(item: str | dict) -> dict:
    """Accept server.py-style rich inputs, fold to the oMLX item shape.

    Supported keys after aliasing (mirrors server._KEY_ALIASES):
    text, image, audio -- where image/audio may be paths (made into data URIs)
    or ``{path|b64|url}`` dicts as the local server accepts.
    """
    if isinstance(item, str):
        return {"text": item}
    aliases = {"images": "image", "texts": "text", "audios": "audio", "videos": "video"}
    out: dict[str, Any] = {}
    for raw_k, v in item.items():
        k = aliases.get(str(raw_k).lower(), str(raw_k).lower())
        if k == "text":
            out["text"] = v if isinstance(v, str) else "\n".join(v)
        elif k in ("image", "audio"):
            if isinstance(v, list):
                # multiple images: oMLX takes one per item; join as separate
                # media via the first, keep text -- same as local multi-image
                # collapse. Callers needing per-image vectors should batch.
                out[k] = _media_data_uri(v[0], k)
            elif isinstance(v, dict):
                ref = v.get("path") or v.get("b64") or v.get("base64") or v.get("data") or v.get("url")
                out[k] = _media_data_uri(ref, k) if not str(ref).startswith("data:") else ref
            else:
                out[k] = _media_data_uri(v, k)
        elif k == "video":
            continue  # no server-side video; caller pre-extracts frames
        else:
            raise ValueError(f"unknown input key {raw_k!r}")
    if not out:
        raise ValueError("empty input item")
    return out


def embed(
    inputs: EmbedInput,
    *,
    dimensions: int | None = None,
    model: str | None = None,
    encoding_format: str = "float",
    timeout: float | None = None,
) -> list[list[float]]:
    """Embed any supported input shape; returns one vector per input item.

    inputs: str | list[str] | content-part list | list of item dicts.
    A list of dicts = one embedding per dict (oMLX ``items`` batch semantics).
    """
    body: dict[str, Any] = {"model": model or settings.model_path}
    if isinstance(inputs, str):
        body["input"] = inputs
    elif isinstance(inputs, dict):
        body["items"] = [_normalize(inputs)]
    elif isinstance(inputs, list) and inputs and all(isinstance(x, str) for x in inputs):
        body["input"] = inputs
    elif (
        isinstance(inputs, list)
        and inputs
        and all(isinstance(x, dict) and "type" in x for x in inputs)
    ):
        body["input"] = inputs  # content parts describing ONE input
    elif isinstance(inputs, list):
        body["items"] = [_normalize(x) for x in inputs]
    else:
        raise ValueError(f"unsupported input shape: {type(inputs).__name__}")
    if dimensions is not None:
        body["dimensions"] = int(dimensions)
    if encoding_format != "float":
        body["encoding_format"] = encoding_format
    if timeout is not None:
        pass  # per-call timeout overrides handled by caller via env; keep simple
    out = _request("/v1/embeddings", body)
    return [d["embedding"] for d in out["data"]]


def embed_batch(texts: Iterable[str], **kw) -> list[list[float]]:
    """Plain text batch (the common RAG case)."""
    return embed(list(texts), **kw)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # server L2-normalizes already


def models() -> list[dict]:
    return _request("/v1/models", method="GET")["data"]


def image_data_uri(img: "ImageLike") -> str:  # noqa: F821
    """PIL Image / path / bytes -> data URI."""
    try:
        from PIL import Image  # type: ignore

        if isinstance(img, Image.Image):
            buf = io.BytesIO()
            fmt = "PNG" if img.mode in ("RGBA", "P", "LA") else "JPEG"
            if fmt == "JPEG" and img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buf, fmt)
            return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"
    except ImportError:
        pass
    return _media_data_uri(img, "image")
