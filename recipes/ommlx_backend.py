"""``ommlx`` recipe: client of an oMLX server serving EmbeddingGemma 2.

Server contract measured on oMLX 0.7.1rc1 + ``embeddinggemma-2-bf16``
(details and the full capability matrix: docs/OMMLX.md):

- ``POST /v1/embeddings`` only; ``items[]`` of ``{text?, image?, audio?}``
  or OpenAI content parts; images must be ``data:...;base64`` URIs.
- MRL ``dimensions`` truncation is renormalised server-side.
- Task steering has no server support: prefixes travel inside the text.

Known limitations of the current server build (recipe-level workarounds
live here so callers stay clean):

- **video**: no ``video`` item field (422). ``embed_video`` extracts frames
  with ffmpeg and mean-pools the per-frame vectors -- same vision tower, and
  measured retrieval-equivalent (cos(q, video-mean) tracks the best single
  frame within ~0.01).
- **audio**: the ``embedding_audio_enabled`` server flag exposes the input,
  but audio inference currently returns HTTP 500 on every encoding /
  duration / shape tried (wav 8-48k 0.25-10 s incl. upstream's own test
  fixture, mp3/ogg/opus, items and content-part forms, across engine
  reloads). Server logs are needed to root-cause; until then ``audio`` is
  reported as a capability that may fail, and ``test_ommlx`` skips it.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from config import settings

from . import BackendError


class OmmlxBackendError(BackendError):
    def __init__(self, status: int, body: Any, path: str = ""):
        self.status = status  # 0 == unreachable
        self.body = body
        super().__init__(f"oMLX {path or 'request'} failed: HTTP {status}: {body}")


name = "ommlx"

capabilities = {
    "text": True,
    "image": True,
    "audio": True,           # server flag on; inference 500s on 0.7.1rc1 -- may raise
    "video": True,           # via client-side frame extraction (embed_video)
    "interleave": False,     # one image + one text per item; no <|audio|> markers
    "prompt_presets": False, # prefixes travel in the text (task= below)
    "dimensions": True,
}

TASK_PREFIX = {
    "search": "task: search result | query: ",
    "question": "task: question answering | query: ",
    "fact": "task: fact checking | query: ",
    "code": "task: code retrieval | query: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
    "similarity": "task: sentence similarity | query: ",
    "document": "title: none | text: ",
}

_KIND_BY_EXT = {
    "png": "image", "jpg": "image", "jpeg": "image", "webp": "image", "gif": "image",
    "bmp": "image", "tif": "image", "tiff": "image", "heic": "image", "avif": "image",
    "wav": "audio", "mp3": "audio", "m4a": "audio", "aac": "audio", "flac": "audio",
    "ogg": "audio", "oga": "audio", "opus": "audio", "weba": "audio",
}

_VIDEO_EXT = {"mp4", "mov", "mkv", "webm", "avi", "m4v", "ts"}

_AUDIO_MIME = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4",
               "aac": "audio/mp4", "flac": "audio/flac", "ogg": "audio/ogg",
               "oga": "audio/ogg", "opus": "audio/opus", "weba": "audio/webm"}


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _data_uri(path: Path, kind: str) -> str:
    mime = (
        _AUDIO_MIME.get(path.suffix.lower().lstrip("."))
        if kind == "audio"
        else None
    ) or mimetypes.guess_type(str(path))[0]
    mime = mime or ("image/png" if kind == "image" else "application/octet-stream")
    return f"data:{mime};base64,{_b64(path.read_bytes())}"


def _post(path: str, body: dict, timeout: float | None = None) -> Any:
    url = f"{settings.resolved_base_url()}{path}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    key = settings.api_key
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout or settings.timeout_s) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise OmmlxBackendError(e.code, e.read().decode(errors="replace")[:400], path) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise OmmlxBackendError(
            0, f"unreachable ({e}); check OMLX_BASE_URL={settings.resolved_base_url()!r}", path
        ) from e


def _prefix(task: str | None, text: str) -> str:
    if not task:
        return text
    try:
        return TASK_PREFIX[task] + text
    except KeyError as e:
        raise ValueError(f"unknown task {task!r}; choose one of {sorted(TASK_PREFIX)}") from e


def _image_value(v: Any) -> str:
    """path/PIL/bytes/str(data-uri|b64)/{path|b64|url-dict} -> data URI."""
    if isinstance(v, bytes):
        return "data:image/png;base64," + _b64(v)
    if isinstance(v, (str, Path)):
        s = str(v)
        if s.startswith("data:") or s.startswith("http"):
            return s
        p = Path(s)
        if p.exists():
            return _data_uri(p, "image")
        return "data:image/png;base64," + s
    if isinstance(v, dict):
        ref = v.get("path") or v.get("b64") or v.get("base64") or v.get("data")
        return _image_value(ref)
    # best effort: PIL-like
    try:
        buf = __import__("io").BytesIO()
        fmt = "PNG" if v.mode in ("RGBA", "P", "LA") else "JPEG"
        (v if fmt == "PNG" else v.convert("RGB")).save(buf, fmt)
        return f"data:image/{fmt.lower()};base64," + _b64(buf.getvalue())
    except AttributeError as e:
        raise ValueError(f"unsupported image value {type(v).__name__}") from e


def _audio_value(v: Any) -> str:
    if isinstance(v, dict) and "array" in v:  # raw PCM -> wrap a wav in-memory
        return _pcm_to_wav_uri(v["array"], int(v.get("sampling_rate", 16000)))
    if isinstance(v, bytes):
        return "data:audio/wav;base64," + _b64(v)
    if isinstance(v, (str, Path)):
        s = str(v)
        if s.startswith("data:"):
            return s
        p = Path(s)
        if p.exists():
            return _data_uri(p, "audio")
        return "data:audio/wav;base64," + s
    if isinstance(v, dict):
        return _audio_value(v.get("path") or v.get("b64") or v.get("data"))
    raise ValueError(f"unsupported audio value {type(v).__name__}")


def _pcm_to_wav_uri(samples: list[float] | "array", sr: int) -> str:  # type: ignore[name-defined]
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        ints = bytearray()
        for x in samples:
            ints += int(max(-32768, min(32767, round(x * 32767)))).to_bytes(2, "little", signed=True)
        w.writeframes(bytes(ints))
    return "data:audio/wav;base64," + _b64(buf.getvalue())


def _to_item(item: str | dict) -> dict:
    """Rich input shape (shared with the local recipe) -> oMLX item.

    Note: oMLX takes ONE image and ONE audio per item and no video; multi
    image lists collapse to the first (batch explicitly for per-image
    vectors), video is handled by :func:`embed_video`.
    """
    if isinstance(item, str):
        return {"text": item}
    aliases = {"images": "image", "texts": "text", "audios": "audio", "videos": "video"}
    out: dict[str, Any] = {}
    for raw_k, v in item.items():
        k = aliases.get(str(raw_k).lower(), str(raw_k).lower())
        if k == "text":
            out["text"] = v if isinstance(v, str) else "\n".join(v)
        elif k == "image":
            out["image"] = _image_value(v[0] if isinstance(v, list) else v)
        elif k == "audio":
            out["audio"] = _audio_value(v[0] if isinstance(v, list) else v)
        elif k == "video":
            continue  # caller should use embed_video; tolerate silently
        else:
            raise ValueError(f"unknown key {raw_k!r}")
    if not out:
        raise ValueError("empty item")
    return out


def _embed_request(body: dict, dim: int | None) -> list[list[float]]:
    if dim:
        body["dimensions"] = int(dim)
    out = _post("/v1/embeddings", body)
    return [d["embedding"] for d in out["data"]]


def embed_text(text: str, *, task: str | None = None, dim: int | None = None) -> list[float]:
    return _embed_request({"model": settings.model_path, "input": _prefix(task, text)}, dim)[0]


def embed_batch(
    texts: list[str], *, task: str | None = None, dim: int | None = None
) -> list[list[float]]:
    return _embed_request(
        {"model": settings.model_path, "input": [_prefix(task, t) for t in texts]}, dim
    )


def embed_item(item: str | dict, *, dim: int | None = None) -> list[float]:
    return embed_many([item], dim=dim)[0]


def embed_many(items: list, *, dim: int | None = None) -> list[list[float]]:
    return _embed_request(
        {"model": settings.model_path, "items": [_to_item(i) for i in items]}, dim
    )


def embed_file(
    path: str | Path, *, kind: str | None = None, dim: int | None = None
) -> list[float]:
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    if kind is None and ext in _VIDEO_EXT:
        return embed_video(p, dim=dim)
    kind = kind or _KIND_BY_EXT.get(ext)
    if kind is None:
        raise ValueError(f"cannot infer media kind for {p.name}; pass kind=")
    key = "image" if kind == "image" else "audio"
    return embed_item({key: p}, dim=dim)


def embed_video(
    path: str | Path, *, fps: float = 1.0, max_frames: int = 32, dim: int | None = None,
    long_side: int = 768,
) -> list[float]:
    """Video embedding via client-side frame extraction (no server video API).

    ffmpeg samples ``fps`` frames (capped at ``max_frames``, downscaled to
    ``long_side``), embeds each as an image and returns the L2-renormalised
    mean -- the standard proxy for shot-level retrieval on eg2.
    """
    if not shutil.which("ffmpeg"):
        raise OmmlxBackendError(0, "ffmpeg not found; required for omlx video support")
    p = Path(path)
    with tempfile.TemporaryDirectory(prefix="eg2_vid_") as td:
        out = Path(td) / "f%03d.jpg"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(p),
            "-vf", f"fps={fps},scale='if(gt(iw,ih),min({long_side},iw),-2)':'if(gt(iw,ih),-2,min({long_side},ih))'",
            "-frames:v", str(max_frames), str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        frames = sorted(Path(td).glob("f*.jpg"))
        if not frames:
            raise OmmlxBackendError(0, f"ffmpeg produced no frames for {p.name}")
        vecs = embed_many([{"image": f} for f in frames], dim=dim)
    mean = [sum(col) / len(vecs) for col in zip(*vecs)]
    norm = sum(x * x for x in mean) ** 0.5 or 1.0
    return [x / norm for x in mean]
