"""Quick semantic sanity against the resolved backend (local /embed or oMLX /v1).

Used by `just try "<text>"`. Asserts dim / unit-norm / finite values.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endpoint import resolve_base_url  # noqa: E402

OMLX_PREFIX = {
    "SearchQuery": "task: search result | query: ",
    "Document": "title: none | text: ",
    "STS": "task: sentence similarity | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--url", default=None, help="default: auto-discover via endpoint.py")
    p.add_argument("--prompt-name", default="SearchQuery")
    p.add_argument("--dim", type=int, default=256)
    a = p.parse_args()
    base = a.url or resolve_base_url()

    def _post(path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        try:
            from config import settings  # noqa

            if getattr(settings, "api_key", ""):
                headers["Authorization"] = f"Bearer {settings.api_key}"
        except Exception:  # noqa: BLE001
            pass
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    # ommlx remote: /embed does not exist -> /v1/embeddings, prefix in-text
    try:
        out = _post("/embed", {"input": a.text, "prompt_name": a.prompt_name, "truncate_dim": a.dim})
        vec = out["embeddings"][0]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        out = _post(
            "/v1/embeddings",
            {
                "model": __import__("config").settings.model_path,
                "input": OMLX_PREFIX.get(a.prompt_name, "") + a.text,
                "dimensions": a.dim,
            },
        )
        vec = out["data"][0]["embedding"]

    e = np.asarray(vec)
    norm = float(np.linalg.norm(e))
    ok = len(e) == a.dim and abs(norm - 1) < 1e-2 and bool(np.isfinite(e).all())
    print(f"text={a.text!r} dim={len(e)} norm={norm:.4f} head={np.round(e[:3], 3)} -> {'OK' if ok else 'BAD'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
