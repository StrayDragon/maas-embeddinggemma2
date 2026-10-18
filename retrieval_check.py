"""Cross-modal retrieval sanity on real assets: zh queries -> image ranking.

Works against both backends chosen by profile:
- local server: rich /embed ({image:{path}}, prompt_name)
- oMLX remote:  /v1/embeddings with base64 data URIs, task prefix in-text
"""
import base64
import json
import mimetypes
import os
import urllib.request
from pathlib import Path

import numpy as np

from endpoint import resolve_base_url

BASE = resolve_base_url()


def _auth() -> dict:
    key = os.getenv("OMLX_API_KEY", "")
    try:
        from config import settings

        key = key or getattr(settings, "api_key", "")
    except Exception:  # noqa: BLE001
        pass
    return {"Authorization": f"Bearer {key}"} if key else {}


def _remote() -> bool:
    try:
        from config import ACTIVE_PROFILE

        return ACTIVE_PROFILE == "ommlx"
    except Exception:  # noqa: BLE001
        return False


def _data_uri(p: Path) -> str:
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **_auth()},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def emb_text(text: str) -> np.ndarray:
    if _remote():
        out = _post("/v1/embeddings", {
            "model": __import__("config").settings.model_path,
            "input": f"task: search result | query: {text}",
        })
        return np.asarray(out["data"][0]["embedding"])
    return np.asarray(_post("/embed", {"input": {"text": text}, "prompt_name": "SearchQuery"})["embeddings"][0])


def emb_image(p: Path) -> np.ndarray:
    if _remote():
        out = _post("/v1/embeddings", {
            "model": __import__("config").settings.model_path,
            "items": [{"image": _data_uri(p)}],
        })
        return np.asarray(out["data"][0]["embedding"])
    return np.asarray(_post("/embed", {"input": {"image": {"path": str(p)}}})["embeddings"][0])


imgs = {p.stem: emb_image(p) for p in sorted(Path("assets/images").glob("*.png"))}
queries = {
    "猫在窗台上": "猫在窗台上",
    "狗和小孩在公园玩球": "狗和小孩在公园玩球",
    "窗户前一杯热咖啡": "窗户前一杯热咖啡",
}
top1_ok = 0
for label, q in queries.items():
    qv = emb_text(q)
    rank = sorted(((float(qv @ v), k) for k, v in imgs.items()), reverse=True)
    hit = "✓" if any(label[:2] in k for _, k in rank[:2]) else "?"
    top1_ok += hit == "✓"
    print(f"{q}: " + " > ".join(f"{k}({s:.3f})" for s, k in rank) + f"  top1{hit}")
print(f"\ntop1 hits: {top1_ok}/{len(queries)}")
