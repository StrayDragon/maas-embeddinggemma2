"""Quick semantic sanity: POST /embed and assert dim/unit-norm. Used by `just test-embed`."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endpoint import resolve_base_url  # noqa: E402


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--url", default=None, help="default: auto-discover via endpoint.py (EMBG2_BASE_URL / .env / probe)")
    p.add_argument("--prompt-name", default="SearchQuery")
    p.add_argument("--dim", type=int, default=256)
    a = p.parse_args()
    url = a.url or resolve_base_url() + "/embed"

    body = {"input": a.text, "prompt_name": a.prompt_name, "truncate_dim": a.dim}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        e = np.asarray(json.load(r)["embeddings"][0])

    norm = float(np.linalg.norm(e))
    ok = len(e) == a.dim and abs(norm - 1) < 1e-3 and bool(np.isfinite(e).all())
    print(f"text={a.text!r} dim={len(e)} norm={norm:.4f} head={np.round(e[:3], 3)} -> {'OK' if ok else 'BAD'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
