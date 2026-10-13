"""EmbeddingGemma 2 embedding server — text / image / audio / video / interleaved multimodal.

Run:  ./run.sh          (env: HOST PORT DEVICE DTYPE MODEL_PATH)
"""
from __future__ import annotations

import base64
import binascii
import io
import logging
import uuid

from fastapi.responses import JSONResponse
import os
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from math import gcd
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, Field

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("embeddinggemma2")

MODEL_PATH: str = settings.model_path
DEVICE_ENV = settings.device
DTYPE_ENV = settings.dtype

MRL_DIMS = [768, 512, 256, 128]

# ---------------------------------------------------------------- device / dtype


def _resolve_device() -> str:
    if DEVICE_ENV != "auto":
        return DEVICE_ENV
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    names = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
    }
    if DTYPE_ENV != "auto":
        if DTYPE_ENV not in names:
            raise ValueError(f"DTYPE must be one of {sorted(names)} or 'auto'")
        return names[DTYPE_ENV]
    # model card prefers bfloat16, but on this ROCM/gfx1151 stack bf16 produces
    # NaNs in batched attention (flaky, worst with sdpa). float32 is rock-solid
    # and still fast (~25ms/batch on the 8060S). set DTYPE=bfloat16 to use bf16.
    if device != "cpu" and DTYPE_ENV == "auto":
        return torch.float32
    if device != "cpu" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


# ---------------------------------------------------------------- media resolution

_DATAURI_RE = re.compile(r"^data:[^;,]+;base64,(.*)$", re.S)
_URL_RE = re.compile(r"^https?://", re.I)
_DEFAULT_SUFFIX = {"image": ".png", "audio": ".wav", "video": ".mp4"}
_KEY_ALIASES = {"images": "image", "texts": "text", "audios": "audio", "videos": "video"}


class BadInput(ValueError):
    pass


def _write_tmp(dir_: Path, name: str, data: bytes) -> str:
    p = dir_ / name
    p.write_bytes(data)
    return str(p)


def _b64_to_file(spec: str, kind: str, dir_: Path, n: int) -> str:
    m = _DATAURI_RE.match(spec.strip())
    raw = m.group(1) if m else spec
    try:
        data = base64.b64decode(re.sub(r"\s", "", raw), validate=False)
    except (binascii.Error, ValueError) as e:
        raise BadInput(f"invalid base64 for {kind} #{n}: {e}") from e
    if not data:
        raise BadInput(f"empty base64 payload for {kind} #{n}")
    return _write_tmp(dir_, f"{kind}_{n}{_DEFAULT_SUFFIX[kind]}", data)


def _url_to_file(url: str, kind: str, dir_: Path, n: int) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix
    suffix = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,5}", suffix or "") else _DEFAULT_SUFFIX[kind]
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except Exception as e:  # noqa: BLE001
        raise BadInput(f"failed to fetch {kind} url #{n} ({url}): {e}") from e
    return _write_tmp(dir_, f"{kind}_res{n}{suffix}", data)


def _audio_from_array(d: dict, n: int) -> dict:
    """{array: [...], sampling_rate: int} -> mono 16 kHz float32 (resampled if needed)."""
    try:
        arr = np.asarray(d["array"], dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        raise BadInput(f"audio #{n}: bad array: {e}") from e
    sr = int(d.get("sampling_rate", 16000))
    if arr.ndim == 2:  # (samples, channels) -> mono
        arr = arr.mean(axis=1)
    if arr.ndim != 1:
        raise BadInput(f"audio #{n}: array must be 1-D mono or (samples, channels)")
    if sr <= 0:
        raise BadInput(f"audio #{n}: bad sampling_rate {sr}")
    if sr != 16000:
        g = gcd(sr, 16000)
        up, down = 16000 // g, sr // g
        try:
            from scipy.signal import resample_poly

            arr = resample_poly(arr, up, down).astype(np.float32)
        except ImportError:
            idx = np.linspace(0, len(arr) - 1, max(1, int(len(arr) * 16000 / sr)))
            arr = np.interp(idx, np.arange(len(arr)), arr).astype(np.float32)
    return {"array": arr, "sampling_rate": 16000}


def _resolve_media(x: Any, kind: str, dir_: Path, n: int) -> Any:
    """path | url | b64/dataURI str | {path|url|b64} | {array,sampling_rate}(audio) -> loader-ready value"""
    if isinstance(x, bool) or isinstance(x, (int, float)):
        raise BadInput(f"{kind} #{n}: scalar is not a valid media reference")
    if isinstance(x, str):
        s = x.strip()
        if _URL_RE.match(s):
            return _url_to_file(s, kind, dir_, n)
        if Path(s).exists():
            return s
        if len(s) >= 24:  # assume bare base64
            return _b64_to_file(s, kind, dir_, n)
        raise BadInput(f"{kind} #{n}: not a path/url/base64: {s[:80]!r}")
    if isinstance(x, dict):
        if "path" in x:
            p = str(x["path"])
            if not Path(p).exists():
                raise BadInput(f"{kind} #{n}: path not found: {p}")
            return p
        if "url" in x:
            return _url_to_file(str(x["url"]), kind, dir_, n)
        for k in ("b64", "base64", "data"):
            if k in x:
                return _b64_to_file(str(x[k]), kind, dir_, n)
        if kind == "audio" and "array" in x:
            return _audio_from_array(x, n)
        raise BadInput(
            f"{kind} #{n}: dict needs 'path' | 'url' | 'b64'" + (" | 'array'+sampling_rate" if kind == "audio" else "")
        )
    raise BadInput(f"{kind} #{n}: unsupported type {type(x).__name__}")


def _prepare_item(item: Any, dir_: Path, ctr: dict) -> Any:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise BadInput(f"input items must be strings or objects, got {type(item).__name__}")
    out: dict[str, Any] = {}
    for raw_k, v in item.items():
        k = _KEY_ALIASES.get(str(raw_k).lower(), str(raw_k).lower())
        if k == "text":
            ok = isinstance(v, str) or (isinstance(v, list) and all(isinstance(t, str) for t in v))
            if not ok:
                raise BadInput("'text' must be a string or list of strings")
            out["text"] = v
        elif k in ("image", "audio", "video"):
            if isinstance(v, list):
                seq = []
                for x in v:
                    ctr[k] = ctr.get(k, 0) + 1
                    seq.append(_resolve_media(x, k, dir_, ctr[k]))
                out[k] = seq
            else:
                ctr[k] = ctr.get(k, 0) + 1
                out[k] = _resolve_media(v, k, dir_, ctr[k])
        else:
            raise BadInput(f"unknown input key {raw_k!r} (allowed: text/image/audio/video)")
    if not out:
        raise BadInput("empty input object")
    return out


def _prepare_inputs(user_input: Any, run_dir: Path) -> list[Any]:
    if isinstance(user_input, (str, dict)):
        items = [user_input]
    elif isinstance(user_input, list):
        if not user_input:
            raise BadInput("input list is empty")
        items = user_input
    else:
        raise BadInput("'input' must be a string, an object, or a list")
    ctr: dict = {}
    return [_prepare_item(it, run_dir, ctr) for it in items]


# ---------------------------------------------------------------- app / lifecycle

S: dict[str, Any] = {}


def _renormalize(emb: "np.ndarray") -> "np.ndarray":
    """L2-normalize rows; all-zero rows stay zero (matches ST eps-normalize)."""
    norms = np.linalg.norm(emb, axis=-1, keepdims=True)
    return np.divide(emb, norms, out=np.zeros_like(emb), where=norms > 0)


def _encode_finite(model, inputs: list, kwargs: dict) -> "np.ndarray":
    """encode + finite-value guard.

    This box's ROCm stack has a history of attention numerics flaking
    (see README: bf16 batched-attention NaNs); fp32+sdpa can still emit
    NaN/Inf for specific token sequences in long-lived processes, which
    later blows up JSON serialization (starlette dumps with allow_nan=False).
    Non-finite output is re-encoded on CPU (authoritative numerics); if that
    still yields non-finite values, sanitize and re-normalize as last resort.
    Clients always receive finite, JSON-safe embeddings.
    """
    emb = np.asarray(model.encode(inputs, **kwargs))
    if np.isfinite(emb).all():
        return emb
    n_bad = int((~np.isfinite(emb)).sum())
    log.warning("non-finite embedding on %s (%d/%d values); re-encoding on cpu: inputs=%r",
                S.get("device"), n_bad, emb.size, [str(x)[:60] for x in inputs])
    cpu_kwargs = {**kwargs, "device": "cpu"}
    emb = np.asarray(model.encode(inputs, **cpu_kwargs))
    if not np.isfinite(emb).all():
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        if kwargs.get("normalize_embeddings", True):
            emb = _renormalize(emb)
    return emb


def _load_model() -> None:
    from sentence_transformers import SentenceTransformer

    device = _resolve_device()
    dtype = _resolve_dtype(device)
    t0 = time.perf_counter()
    log.info("loading %s on %s (%s) ...", MODEL_PATH, device, dtype)
    model = SentenceTransformer(
        MODEL_PATH,
        device=device,
        model_kwargs={"dtype": dtype, "attn_implementation": "sdpa"},
    )
    S.update(
        model=model,
        device=device,
        dtype=str(dtype).removeprefix("torch."),
        dim=model.get_embedding_dimension(),
        prompts=dict(model.prompts or {}),
        lock=threading.Lock(),
    )
    log.info("ready in %.1fs (dim=%d, %d prompt presets)", time.perf_counter() - t0, S["dim"], len(S["prompts"]))


def _warmup(model) -> None:
    # first inference per modality drifts (lazy init), and the drift RE-ARMS
    # when modalities interleave, so warm up twice with crossed orders
    from PIL import Image
    import numpy as np

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (0, 0, 0)).save(buf, "PNG")
    img = Image.open(buf)

    p = Path("/tmp/embg2_warmup.mp4")
    import av
    with av.open(str(p), "w") as c:
        st = c.add_stream("libx264", rate=8)
        st.width, st.height, st.pix_fmt = 64, 64, "yuv420p"
        for _i in range(8):
            for pkt in st.encode(av.VideoFrame.from_image(Image.new("RGB", (64, 64), (0, 0, 0)))):
                c.mux(pkt)
        for pkt in st.encode(None):
            c.mux(pkt)
    t = np.zeros(16000, dtype=np.float32)
    rounds = [
        ["warmup text", {"image": img}, {"audio": {"array": t, "sampling_rate": 16000}}, {"video": str(p)}],
        # crossed second pass: re-arms lazy state touched by modality mixing
        [{"image": img}, {"video": str(p)}, "warmup text 2", {"audio": {"array": t, "sampling_rate": 16000}}],
    ]
    for round_ in rounds:
        for item in round_:
            model.encode([item] if isinstance(item, str) else item)
    p.unlink(missing_ok=True)
    log.info("warmup done (x2, cross-ordered)")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    try:
        _warmup(S["model"])
    except Exception:  # noqa: BLE001
        log.exception("warmup failed (continuing)")
    yield


class EmbedRequest(BaseModel):
    input: Union[str, dict, list] = Field(validation_alias=AliasChoices("input", "inputs"))
    prompt_name: str | None = None
    prompt: str | None = None
    truncate_dim: int | None = None
    normalize_embeddings: bool | None = None
    batch_size: int | None = None


app = FastAPI(title="embeddinggemma2-serve", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"]
)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """Catch-all: readable JSON 500 (with trace id) instead of a bare text 500.

    Full traceback lands in the server log under the same trace id.
    """
    trace_id = uuid.uuid4().hex[:12]
    log.exception("unhandled error [%s] on %s %s", trace_id, request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": f"{type(exc).__name__}: {exc}",
            "path": request.url.path,
            "method": request.method,
            "trace_id": trace_id,
        },
    )


@app.get("/health")
def health():
    out: dict[str, Any] = {
        "status": "ok" if S.get("model") is not None else "loading",
        "model": MODEL_PATH,
        "device": S.get("device"),
        "dtype": S.get("dtype"),
        "dimension": S.get("dim"),
    }
    if S.get("device") == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
        out["mem_allocated_mib"] = round(torch.cuda.memory_allocated() / 2**20)
        out["mem_reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20)
    return out


@app.get("/metadata")
def metadata():
    if S.get("model") is None:
        raise HTTPException(503, "model still loading")
    return {
        "model": MODEL_PATH,
        "device": S["device"],
        "dtype": S["dtype"],
        "dimension": S["dim"],
        "mrl_dimensions": MRL_DIMS,
        "modalities": ["text", "image", "audio", "video", "interleaved text+image+video+audio"],
        "prompt_names": sorted(S["prompts"]),
        "prompts": S["prompts"],
        "similarity": "cosine (embeddings unit-normalized)",
    }


@app.post("/embed/st")
@app.post("/embed")
def embed(req: EmbedRequest):
    if S.get("model") is None:
        raise HTTPException(503, "model still loading")
    model = S["model"]

    if req.prompt is not None and req.prompt_name is not None:
        raise HTTPException(400, "provide either 'prompt' or 'prompt_name', not both")
    if req.prompt_name is not None and req.prompt_name not in S["prompts"]:
        raise HTTPException(400, f"unknown prompt_name {req.prompt_name!r}; valid: {sorted(S['prompts'])}")
    if req.truncate_dim is not None and not 1 <= req.truncate_dim <= S["dim"]:
        raise HTTPException(400, f"truncate_dim must be in 1..{S['dim']}")
    if req.batch_size is not None and req.batch_size < 1:
        raise HTTPException(400, "batch_size must be >= 1")

    with tempfile.TemporaryDirectory(prefix="embg2_") as td:
        try:
            prepared = _prepare_inputs(req.input, Path(td))
        except BadInput as e:
            raise HTTPException(400, str(e)) from e

        kwargs: dict[str, Any] = {
            "show_progress_bar": False,
            "normalize_embeddings": True if req.normalize_embeddings is None else req.normalize_embeddings,
        }
        if req.prompt_name is not None:
            kwargs["prompt_name"] = req.prompt_name
        if req.prompt is not None:
            kwargs["prompt"] = req.prompt
        if req.truncate_dim is not None:
            kwargs["truncate_dim"] = req.truncate_dim
        if req.batch_size is not None:
            kwargs["batch_size"] = req.batch_size

        t0 = time.perf_counter()
        try:
            with S["lock"]:
                emb = _encode_finite(model, prepared, kwargs)
        except BadInput as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # noqa: BLE001
            log.exception("encode failed")
            raise HTTPException(500, f"inference failed: {type(e).__name__}: {e}") from e

    emb = np.asarray(emb)
    return {
        "embeddings": emb.tolist(),
        "dimension": int(emb.shape[-1]),
        "count": int(emb.shape[0]),
        "normalized": bool(kwargs["normalize_embeddings"]),
        "model": Path(MODEL_PATH).name,
        "took_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ================================================================ OpenAI-compatible API
# POST /v1/embeddings  (and /embeddings) — works with `ai`/`@ai-sdk/openai`, `openai`,
# langchain, anything speaking the OpenAI embeddings protocol.
# Model id: "embeddinggemma-2" or "embeddinggemma-2:<PromptName>" to steer the
# task prefix (e.g. :Document :SearchQuery :STS :Classification ...). GET /v1/models.


class OpenAIEmbedRequest(BaseModel):
    model: str = "embeddinggemma-2"
    input: Union[str, list]
    dimensions: int | None = None
    encoding_format: str = "float"  # float | base64
    user: str | None = None  # ignored


def _openai_model_spec(model_id: str) -> tuple[str, str | None]:
    base, sep, suffix = model_id.partition(":")
    if sep and suffix not in S["prompts"]:
        raise HTTPException(400, f"unknown task suffix :{suffix!r}; valid: {sorted(S['prompts'])}")
    return (base or "embeddinggemma-2"), (suffix or None)


def _openai_inputs(raw) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and raw and all(isinstance(x, str) for x in raw):
        return raw
    raise HTTPException(400, "'input' must be a string or an array of strings")


def _count_tokens(texts: list[str]) -> int:
    tok = S["model"].tokenizer
    enc = tok(list(texts), add_special_tokens=True)["input_ids"]
    return sum(len(x) for x in enc)


@app.get("/v1/models")
def openai_models():
    data = [
        {
            "id": "embeddinggemma-2",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "embeddinggemma2-serve",
            "max_input_tokens": 8192,
            "task_suffixes": sorted(S["prompts"]),
        }
    ]
    data += [{"id": f"embeddinggemma-2:{n}", "object": "model", "created": int(time.time()), "owned_by": "embeddinggemma2-serve"} for n in sorted(S["prompts"])]
    return {"object": "list", "data": data}


@app.post("/v1/embeddings")
@app.post("/embeddings")
def openai_embeddings(req: OpenAIEmbedRequest):
    if S.get("model") is None:
        raise HTTPException(503, "model still loading")
    if req.encoding_format not in ("float", "base64"):
        raise HTTPException(400, "encoding_format must be 'float' or 'base64'")
    if req.dimensions is not None and not 1 <= req.dimensions <= S["dim"]:
        raise HTTPException(400, f"dimensions must be in 1..{S['dim']}")

    model_id, prompt_name = _openai_model_spec(req.model)
    texts = _openai_inputs(req.input)
    if not texts:
        raise HTTPException(400, "'input' is empty")

    kwargs: dict[str, Any] = {"show_progress_bar": False, "normalize_embeddings": True}
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    if req.dimensions:
        kwargs["truncate_dim"] = req.dimensions

    try:
        with S["lock"]:
            emb = _encode_finite(S["model"], texts, kwargs)
        n_tokens = _count_tokens(texts)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("v1 embeddings failed")
        # readable 500: never a bare 'Internal Server Error' — clients get the
        # exception, the task steering and an input digest for reproduction
        raise HTTPException(500, {
            "error": f"{type(e).__name__}: {e}",
            "task": prompt_name or "(none)",
            "inputs": [t[:120] for t in texts],
            "hint": "full traceback in server log",
        }) from e

    if req.encoding_format == "base64":
        out = [base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode() for v in emb]
    else:
        out = emb.tolist()

    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(out)],
        "model": req.model,
        "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
    }


# ================================================================ DashScope-compatible multimodal embedding
# POST /api/v1/services/embeddings/multimodal-embedding/multimodal-embedding (alias /embed/dashscope)
# 阿里云百炼多模态向量 API 形状, 便于 DashScope 用户零改造切换。
# 独立向量: 每 content 一个; 单 content 多模态 => type=fused; enable_fusion=true => 全部融合, type=fusion。
# 支持字段: text/image/video/multi_images, parameters.dimension/.instruct/.enable_fusion。
# 偏差: 本服务单模型同空间; usage 视觉 token 为估计值(280/img, 1120/video)。

class DashscopeRequest(BaseModel):
    model: str = "embeddinggemma-2"
    input: dict = {}
    parameters: dict = {}


def _ds_error(status: int, code: str, msg: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"code": code, "message": msg, "request_id": str(uuid.uuid4())},
    )


def _ds_text_prep(text: str, instruct, steer: bool) -> str:
    if instruct and steer:
        return f"task: {instruct} | query: {text}"  # EmbeddingGemma instruction shape
    return text


def _ds_parse(el, i: int):
    if isinstance(el, str):
        return {"text": el}
    return el


def _ds_collect_media(el: dict, kind: str, dir_: Path, ctr: dict) -> list:
    """image + multi_images -> list of engine-ready values; video -> single-item list"""
    out: list = []
    if kind == "image":
        if "image" in el:
            ctr["image"] = ctr.get("image", 0) + 1
            out.append(_resolve_media(el["image"], "image", dir_, ctr["image"]))
        for x in el.get("multi_images") or []:
            ctr["image"] = ctr.get("image", 0) + 1
            out.append(_resolve_media(x, "image", dir_, ctr["image"]))
    elif "video" in el:
        ctr["video"] = ctr.get("video", 0) + 1
        out.append(_resolve_media(el["video"], "video", dir_, ctr["video"]))
    return out


@app.post("/embed/dashscope")
@app.post("/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding")
def dashscope_multimodal_embedding(req: DashscopeRequest):
    if S.get("model") is None:
        return _ds_error(503, "InternalError", "model still loading")
    contents = (req.input or {}).get("contents")
    if not isinstance(contents, list) or not contents:
        return _ds_error(400, "InvalidParameter", "input can not be empty: input.contents")

    params = req.parameters or {}
    dimension = params.get("dimension")
    if dimension is not None and (not isinstance(dimension, int) or not 1 <= dimension <= S["dim"]):
        return _ds_error(400, "InvalidParameter", f"dimension must be integer 1..{S['dim']}")
    enable_fusion = bool(params.get("enable_fusion", False))
    instruct = params.get("instruct")

    known = {"text", "image", "video", "multi_images"}
    inputs: list = []
    types: list[str] = []
    text_pieces: list[str] = []
    n_images = 0
    n_videos = 0
    fused_pool = {"text": [], "image": [], "video": []}

    with tempfile.TemporaryDirectory(prefix="ds_") as td:
        dir_ = Path(td)
        ctr: dict = {}

        def add(inp: dict, typ: str) -> None:
            inputs.append(inp)
            types.append(typ)

        for i, raw_el in enumerate(contents):
            el = _ds_parse(raw_el, i)
            if not isinstance(el, dict) or not el:
                return _ds_error(400, "InvalidParameter", f"contents[{i}] must be a non-empty object or string")
            bad = set(el) - known
            if bad:
                return _ds_error(400, "InvalidParameter", f"contents[{i}] unsupported keys {sorted(bad)}; allowed {sorted(known)}")
            if not any(k in el for k in known):
                return _ds_error(400, "InvalidParameter", f"contents[{i}] has no usable modality")

            texts = []
            if "text" in el:
                t = el["text"]
                if not isinstance(t, str) or not t:
                    return _ds_error(400, "InvalidParameter", f"contents[{i}].text must be a non-empty string")
                texts.append(t)

            imgs: list = []
            if "image" in el or "multi_images" in el:
                if "image" in el:
                    ctr["image"] = ctr.get("image", 0) + 1
                    imgs.append(_resolve_media(el["image"], "image", dir_, ctr["image"]))
                for x in el.get("multi_images") or []:
                    if not isinstance(x, (str, dict)):
                        return _ds_error(400, "InvalidParameter", f"contents[{i}].multi_images[{len(imgs)}] invalid")
                    ctr["image"] = ctr.get("image", 0) + 1
                    imgs.append(_resolve_media(x, "image", dir_, ctr["image"]))
            vids: list = []
            if "video" in el:
                if not isinstance(el["video"], str):
                    return _ds_error(400, "InvalidParameter", f"contents[{i}].video must be a string (path/url/b64)")
                ctr["video"] = ctr.get("video", 0) + 1
                vids.append(_resolve_media(el["video"], "video", dir_, ctr["video"]))

            n_images += len(imgs)
            n_videos += len(vids)
            has_media = bool(imgs or vids)

            if enable_fusion:  # accumulate everything, single vector at the end
                for t in texts:
                    fused_pool["text"].append(_ds_text_prep(t, instruct, True))
                fused_pool["image"].extend(imgs)
                fused_pool["video"].extend(vids)
                continue

            multimodal_element = bool(texts and has_media)
            if texts:
                t = _ds_text_prep(texts[0], instruct, multimodal_element)
                inp: dict = {"text": t}
                text_pieces.append(t)
            else:
                inp = {}

            if not has_media:
                add(inp, "text")
            elif not imgs and vids:  # video only
                if len(vids) > 1:
                    for v in vids:
                        add({"video": v}, "video")
                else:
                    inp["video"] = vids[0]
                    add(inp, "video" if not texts else "fused")
            elif imgs and not vids:  # image(s)
                if len(imgs) == 1:
                    inp["image"] = imgs[0]
                    add(inp, "fused" if multimodal_element else "image")
                else:  # multi-image element: independent per image (+text fused w/ first? no: text applies to element -> fuse text w/ each? keep simple: text fused with first, rest plain)
                    if texts and len(imgs) > 1 and "image" not in inp:
                        pass
                    for k, one in enumerate(imgs):
                        if k == 0 and texts:
                            add({"text": inp["text"], "image": one}, "fused")
                        else:
                            add({"image": one}, "image")
            else:  # images + videos mixed in one element: fuse text+first of each
                if texts:
                    inp["image"] = imgs[0]
                    inp["video"] = vids[0]
                    add(inp, "fused")
                    for one in imgs[1:]:
                        add({"image": one}, "image")
                else:
                    add({"image": imgs[0], "video": vids[0]}, "fused")
                    for one in imgs[1:]:
                        add({"image": one}, "image")

        if enable_fusion:
            merged: dict = {}
            if fused_pool["text"]:
                merged["text"] = fused_pool["text"] if len(fused_pool["text"]) > 1 else fused_pool["text"][0]
            if fused_pool["image"]:
                merged["image"] = fused_pool["image"]
            if fused_pool["video"]:
                merged["video"] = fused_pool["video"][0] if len(fused_pool["video"]) == 1 else fused_pool["video"]
            if not merged:
                return _ds_error(400, "InvalidParameter", "no usable modality for fusion")
            add(merged, "fusion")

        if not inputs:
            return _ds_error(400, "InvalidParameter", "no usable contents after parsing")
        if any(t == "fused" for t in types):
            pass

        kwargs: dict = {"show_progress_bar": False, "normalize_embeddings": True}
        if dimension:
            kwargs["truncate_dim"] = dimension
        try:
            with S["lock"]:
                emb = _encode_finite(S["model"], inputs, kwargs)
        except Exception as e:  # noqa: BLE001
            log.exception("dashscope encode failed")
            return _ds_error(500, "InternalError.Algo", f"embedding failed: {type(e).__name__}: {e}")

    n = emb.shape[0]
    out = [{"index": i, "embedding": emb[i].tolist(), "type": types[i]} for i in range(n)]
    tok = S["model"].tokenizer
    text_tokens = sum(len(x) for x in tok(text_pieces, add_special_tokens=True)["input_ids"]) if text_pieces else 0
    image_tokens = 280 * n_images + 1120 * n_videos  # estimate
    return {
        "output": {"embeddings": out},
        "usage": {
            "input_tokens": text_tokens,
            "input_tokens_details": {"image_tokens": image_tokens, "text_tokens": text_tokens},
            "output_tokens": 1,
            "total_tokens": text_tokens + image_tokens,
        },
        "request_id": str(uuid.uuid4()),
    }
