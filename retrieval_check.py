"""Cross-modal retrieval sanity on real assets: zh queries -> image ranking.

Runs against whichever backend the active profile selects (local server or
oMLX) via the recipe registry -- no per-backend branching here.
"""

from pathlib import Path

import numpy as np

from recipes import get_backend

backend = get_backend()
imgs = {p.stem: np.asarray(backend.embed_file(p)) for p in sorted(Path("assets/images").glob("*.png"))}
queries = {
    "猫在窗台上": "猫在窗台上",
    "狗和小孩在公园玩球": "狗和小孩在公园玩球",
    "窗户前一杯热咖啡": "窗户前一杯热咖啡",
}
top1_ok = 0
for label, q in queries.items():
    qv = np.asarray(backend.embed_text(q, task="search"))
    rank = sorted(((float(qv @ v), k) for k, v in imgs.items()), reverse=True)
    hit = "✓" if any(label[:2] in k for _, k in rank[:2]) else "?"
    top1_ok += hit == "✓"
    print(f"{q}: " + " > ".join(f"{k}({s:.3f})" for s, k in rank) + f"  top1{hit}")
print(f"\nbackend={backend.name}  top1 hits: {top1_ok}/{len(queries)}")
