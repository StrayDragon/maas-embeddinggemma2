"""Cross-modal retrieval sanity on real assets: zh queries -> image ranking."""
import json
import urllib.request
from pathlib import Path

import numpy as np

from endpoint import resolve_base_url

BASE = resolve_base_url()


def emb(payload: dict, **extra) -> np.ndarray:
    req = urllib.request.Request(
        f"{BASE}/embed", data=json.dumps({"input": payload, **extra}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return np.asarray(json.load(r)["embeddings"])


imgs = {p.stem: emb({"image": {"path": str(p)}})[0] for p in sorted(Path("assets/images").glob("*.png"))}
queries = {
    "猫在窗台上": "猫在窗台上",
    "狗和小孩在公园玩球": "狗和小孩在公园玩球",
    "窗户前一杯热咖啡": "窗户前一杯热咖啡",
}
top1_ok = 0
for label, q in queries.items():
    qv = emb({"text": q}, prompt_name="SearchQuery")[0]
    rank = sorted(((float(qv @ v), k) for k, v in imgs.items()), reverse=True)
    hit = "✓" if any(label[:2] in k for _, k in rank[:2]) else "?"
    top1_ok += hit == "✓"
    print(f"{q}: " + " > ".join(f"{k}({s:.3f})" for s, k in rank) + f"  top1{hit}")
print(f"\ntop1 hits: {top1_ok}/{len(queries)}")
