"""Quick semantic sanity via the active backend recipe.

Used by `just try "<text>"`. Asserts dim / unit-norm / finite values.
The backend (local server vs oMLX) and task-prefix handling are the
registry's concern -- this file stays shape-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recipes import get_backend  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("--task", default="search", help="search|document|similarity|... (recipe map)")
    p.add_argument("--dim", type=int, default=256)
    a = p.parse_args()

    backend = get_backend()
    vec = backend.embed_text(a.text, task=a.task, dim=a.dim)

    e = np.asarray(vec)
    norm = float(np.linalg.norm(e))
    ok = len(e) == a.dim and abs(norm - 1) < 1e-2 and bool(np.isfinite(e).all())
    print(
        f"[{backend.name}] text={a.text!r} dim={len(e)} norm={norm:.4f} "
        f"head={np.round(e[:3], 3)} -> {'OK' if ok else 'BAD'}"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
