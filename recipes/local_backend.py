"""``local`` recipe: this repo's own FastAPI server (server.py).

Rich surface (see docs/API.md): ``POST /embed`` accepts text/image/audio/
video/interleaved items, path-or-URL-or-b64 media, task-name presets, raw
audio PCM arrays and per-item batch. Errors surface as ``LocalBackendError``.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from endpoint import resolve_base_url

from . import BackendError


class LocalBackendError(BackendError):
    def __init__(self, status: int, body: Any, path: str = ""):
        self.status = status
        self.body = body
        super().__init__(f"local server {path or 'request'} failed: HTTP {status}: {body}")


name = "local"

capabilities = {
    "text": True,
    "image": True,
    "audio": True,          # incl. raw {array, sampling_rate} PCM
    "video": True,          # server-side decode (ST/transformers path)
    "interleave": True,     # <{|audio|}> placeholders in one input
    "prompt_presets": True, # prompt_name / prompt per request
    "dimensions": True,     # MRL truncation, server renormalises
}

# model-card task prefixes; the server also exposes these as prompt_name
TASK_PREFIX = {
    "search": "task: search result | query: ",
    "question": "task: question answering | query: ",
    "fact": "task: fact checking | query: ",
    "code": "task: code retrieval | query: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
    "similarity": "task: sentence similarity | query: ",
    # document-side (no query marker)
    "document": "title: none | text: ",
}

# prompt_name aliases accepted by POST /embed
PROMPT_NAME = {
    "search": "SearchQuery", "question": "QuestionAnswering", "fact": "FactChecking",
    "code": "CodeRetrieval", "classification": "Classification",
    "clustering": "Clustering", "similarity": "STS", "document": "Document",
}

_BASE: str | None = None


def _base() -> str:
    global _BASE
    if _BASE is None:
        _BASE = resolve_base_url()
    return _BASE


def _post(path: str, body: dict, timeout: float = 300) -> Any:
    req = urllib.request.Request(
        _base() + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise LocalBackendError(e.code, e.read().decode(errors="replace")[:400], path) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise LocalBackendError(0, f"unreachable ({e}); start with `just serve`", path) from e


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _image_ref(v: Any) -> Any:
    """path/PIL/bytes/{b64|path|url} -> server shape ({b64}|{path}|{url}|data-uri str)."""
    if isinstance(v, (str, Path)):
        p = Path(v)
        if p.exists():
            return {"b64": _b64(p)} if p.suffix.lower() != ".svg" else str(p)
        return str(v)  # URL or bare b64: server resolves
    if isinstance(v, dict):  # passthrough {path|url|b64}
        return v
    if isinstance(v, bytes):
        return {"b64": base64.b64encode(v).decode()}
    try:  # PIL Image
        buf = io.BytesIO()
        fmt = "PNG" if v.mode in ("RGBA", "P", "LA") else "JPEG"
        (v if fmt == "PNG" else v.convert("RGB")).save(buf, fmt)
        return {"b64": base64.b64encode(buf.getvalue()).decode()}
    except AttributeError as e:  # not PIL-shaped
        raise ValueError(f"unsupported image value {type(v).__name__}") from e


def _audio_ref(v: Any) -> Any:
    if isinstance(v, dict) and "array" in v:  # raw PCM -> pass through
        return v
    if isinstance(v, (str, Path)):
        p = Path(v)
        if p.exists():
            return {"b64": _b64(p)}
        return str(v)
    if isinstance(v, bytes):
        return {"b64": base64.b64encode(v).decode()}
    if isinstance(v, dict):
        return v
    raise ValueError(f"unsupported audio value {type(v).__name__}")


def _media(kind: str, v: Any) -> Any:
    return {"image": _image_ref, "audio": _audio_ref}.get(kind, lambda x: x)(v)


def _to_server_item(item: str | dict) -> Any:
    if isinstance(item, str):
        return item
    out: dict[str, Any] = {}
    aliases = {"images": "image", "texts": "text", "audios": "audio", "videos": "video"}
    for k, v in item.items():
        k = aliases.get(str(k).lower(), str(k).lower())
        if k == "text":
            out["text"] = v if isinstance(v, str) else "\n".join(v)
        elif k in ("image", "audio", "video"):
            out[k] = [_media(k, x) for x in v] if isinstance(v, list) else _media(k, v)
        else:
            raise ValueError(f"unknown key {k!r}")
    if not out:
        raise ValueError("empty item")
    return out


def _prefix(task: str | None, text: str) -> str:
    if not task:
        return text
    try:
        return TASK_PREFIX[task] + text
    except KeyError as e:
        raise ValueError(f"unknown task {task!r}; choose one of {sorted(TASK_PREFIX)}") from e


def embed_text(text: str, *, task: str | None = None, dim: int | None = None) -> list[float]:
    return embed_batch([text], task=task, dim=dim)[0]


def embed_batch(
    texts: list[str], *, task: str | None = None, dim: int | None = None
) -> list[list[float]]:
    body: dict[str, Any] = {"input": [_prefix(task, t) for t in texts]}
    if task and task in PROMPT_NAME and not task.startswith(("document",)):
        body["prompt_name"] = PROMPT_NAME[task]
    if dim:
        body["truncate_dim"] = dim
    return _post("/embed", body)["embeddings"]


def embed_item(item: str | dict, *, dim: int | None = None) -> list[float]:
    return embed_many([item], dim=dim)[0]


def embed_many(items: list, *, dim: int | None = None) -> list[list[float]]:
    body: dict[str, Any] = {"input": [_to_server_item(i) for i in items]}
    if dim:
        body["truncate_dim"] = dim
    return _post("/embed", body)["embeddings"]


def embed_file(
    path: str | Path, *, kind: str | None = None, dim: int | None = None
) -> list[float]:
    p = Path(path)
    kind = kind or _KIND_BY_EXT.get(p.suffix.lower().lstrip("."))
    if kind is None:
        raise ValueError(f"cannot infer media kind for {p.name}; pass kind=")
    return embed_item({kind: str(p)}, dim=dim)


_KIND_BY_EXT = {
    "png": "image", "jpg": "image", "jpeg": "image", "webp": "image", "gif": "image",
    "bmp": "image", "tif": "image", "tiff": "image", "heic": "image", "avif": "image",
    "wav": "audio", "mp3": "audio", "m4a": "audio", "aac": "audio", "flac": "audio",
    "ogg": "audio", "oga": "audio", "opus": "audio", "weba": "audio",
    "mp4": "video", "mov": "video", "mkv": "video", "webm": "video", "avi": "video",
    "m4v": "video", "ts": "video",
}
