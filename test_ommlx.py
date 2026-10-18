"""Live tests against the oMLX server (profile ommlx).

Run:  just test-ommlx        (or: .venv/bin/python test_ommlx.py [-k name])
Reads OMLX_* from .env / environment; skips cleanly when the server is down.

Covers the capability matrix measured on oMLX 0.7.1rc1 + embeddinggemma-2-bf16
(see docs/OMMLX.md): text / batch / content parts / items / image data-URI /
text+image fusion / MRL truncation+renorm / auth errors / semantic sanity.
Audio is covered only when the server has embedding_audio_enabled on.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import urllib.error

from config import settings

PASS = "✓"
FAIL = "✗"
SKIP = "·"

_results: list[tuple[str, str, str]] = []


def _record(name: str, ok: bool, note: str = "") -> bool:
    _results.append((name, "ok" if ok else "FAIL", note))
    print(f"  {PASS if ok else FAIL} {name}" + (f"  ({note})" if note else ""))
    return ok


def _skip(name: str, why: str) -> None:
    _results.append((name, "skip", why))
    print(f"  {SKIP} {name}  ({why})")


# ---------------------------------------------------------------- helpers

def _png_data_uri(rgb: tuple[int, int, int] = (200, 30, 30), size: int = 8) -> str:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        raise SystemExit("Pillow required: uv pip install --python .venv/bin/python pillow")
    buf = io.BytesIO()
    Image.new("RGB", (size, size), rgb).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _wav_data_uri(seconds: float = 0.5, freq: int = 440) -> str:
    import struct
    import wave

    buf = io.BytesIO()
    n = int(16000 * seconds)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / 16000)))
                for i in range(n)
            )
        )
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v) -> float:
    return math.sqrt(sum(x * x for x in v))


# ---------------------------------------------------------------- tests

def test_models_visible() -> None:
    from omlx_client import models

    ids = [m["id"] for m in models()]
    _record("models endpoint lists eg2", settings.model_path in ids, ",".join(ids[:3]))


def test_text_single() -> None:
    from omlx_client import embed

    v = embed("hello ommlx")
    _record("single text 768d", len(v) == 1 and len(v[0]) == 768)


def test_text_batch() -> None:
    from omlx_client import embed

    vs = embed(["alpha", "beta", "gamma"])
    _record("text batch x3", len(vs) == 3 and all(len(v) == 768 for v in vs))


def test_content_parts() -> None:
    from omlx_client import embed

    v = embed([{"type": "text", "text": "via content parts"}])
    _record("OpenAI content parts (text)", len(v[0]) == 768)


def test_items_text() -> None:
    from omlx_client import embed

    vs = embed([{"text": "items a"}, {"text": "items b"}])
    _record("items batch", len(vs) == 2)


def test_image_data_uri() -> None:
    from omlx_client import embed

    v = embed({"image": _png_data_uri()})
    ok = len(v[0]) == 768 and all(math.isfinite(x) for x in v[0])
    _record("image via items (data URI)", ok)


def test_image_parts() -> None:
    from omlx_client import embed

    v = embed([{"type": "image_url", "image_url": {"url": _png_data_uri((30, 30, 200))}}])
    _record("image via content parts image_url", len(v[0]) == 768)


def test_text_image_fusion() -> None:
    """text+image in ONE input must fuse into one vector (multimodal story)."""
    from omlx_client import embed

    v = embed([{"type": "text", "text": "a red block"},
               {"type": "image_url", "image_url": {"url": _png_data_uri()}}])
    _record("text+image fused single vector", len(v) == 1 and len(v[0]) == 768)


def test_mixed_items_batch() -> None:
    from omlx_client import embed

    uri = _png_data_uri()
    vs = embed([{"text": "t"}, {"image": uri}, {"text": "t2", "image": uri}])
    _record("mixed items batch (text|image|both)", len(vs) == 3)


def test_mrl_truncate_renorm() -> None:
    from omlx_client import embed

    for d in (768, 512, 256, 128):
        v = embed("mrl check", dimensions=d)[0]
        if not _record(f"MRL dim={d}", len(v) == d):
            return
    v = embed("renorm check", dimensions=256)[0]
    n = _norm(v)
    _record("server re-normalizes truncated vec", abs(n - 1.0) < 1e-2, f"norm={n:.4f}")


def test_usage_counts() -> None:
    """prompt_tokens reflect multimodal expansion (image=280 tokens)."""
    from omlx_client import _request

    out = _request("/v1/embeddings", {"model": settings.model_path,
                                      "items": [{"image": _png_data_uri()}]})
    tok = out["usage"]["prompt_tokens"]
    _record("usage counts image tokens", 260 <= tok <= 300, f"tokens={tok}")


def test_semantic_text() -> None:
    from omlx_client import embed

    a = embed("a cat sits on the mat")[0]
    b = embed("the feline rests on the rug")[0]
    c = embed("quarterly revenue forecast")[0]
    ab, ac = _cos(a, b), _cos(a, c)
    _record("text semantics paraphrase > unrelated", ab > ac, f"{ab:.3f} vs {ac:.3f}")


def test_semantic_cross_modal() -> None:
    from omlx_client import embed

    q = embed("task: search result | query: red color")[0]
    red = embed({"image": _png_data_uri((200, 30, 30))})[0]
    blue = embed({"image": _png_data_uri((30, 30, 200))})[0]
    _record("cross-modal red query ranks red img first",
            _cos(q, red) > _cos(q, blue),
            f"{_cos(q, red):.3f} vs {_cos(q, blue):.3f}")


def test_task_prefix_steering() -> None:
    """eg2 is trained with instruction prefixes; check both directions stay sane
    and report the actual deltas (server applies no prefix itself, so the
    steering is purely in the text we send -- document the numbers)."""
    from omlx_client import embed

    doc = ("title: none | text: Auroras appear when solar wind particles "
           "hit the atmosphere near the poles.")
    plain = embed("Auroras appear when solar wind hits the atmosphere")[0]
    steered = embed("task: search result | query: aurora borealis cause")[0]
    d = embed(doc)[0]
    ps, pd = _cos(plain, d), _cos(steered, d)
    # both routes must retrieve well above chance; exact ordering is
    # checkpoint-dependent (0.7.1rc1 bf16 measured plain 0.83 / steered 0.78)
    _record("prefix-steered and plain both retrieve >0.6", min(ps, pd) > 0.6,
            f"plain={ps:.3f} steered={pd:.3f}")


def test_long_input_truncation() -> None:
    from omlx_client import embed

    v = embed("word " * 8000)  # ~10k tokens > 8192 ctx; server truncates
    _record("over-long input truncated not rejected", len(v[0]) == 768)


def test_auth_error_surface() -> None:
    """Wrong key must map to a readable error, not a crash."""
    import omlx_client

    saved = omlx_client.settings

    class _S:
        model_path = saved.model_path
        api_key = "nope-wrong-key"
        timeout_s = 10.0

        def resolved_base_url(self):
            return saved.resolved_base_url()

    try:
        omlx_client.settings = _S()
        omlx_client._request("/v1/models", method="GET")
        ok, note = False, "expected 401"
    except omlx_client.OmmlxError as e:
        ok, note = e.status == 401, f"status={e.status}"
    finally:
        omlx_client.settings = saved
    _record("401 surfaces as OmmlxError", ok, note)


def test_unreachable_surface() -> None:
    import omlx_client

    saved = omlx_client.settings
    try:
        class _S:  # minimal stand-in
            model_path = saved.model_path
            api_key = "x"
            timeout_s = 2.0

            def resolved_base_url(self):
                return "http://127.0.0.1:9"

        omlx_client.settings = _S()
        try:
            omlx_client.embed("x")
            ok, note = False, "no error"
        except omlx_client.OmmlxError as e:
            ok, note = e.status == 0, str(e)[:60]
    finally:
        omlx_client.settings = saved
    _record("unreachable server -> OmmlxError(0)", ok, note)


def test_audio_if_enabled() -> None:
    from omlx_client import embed

    try:
        v = embed({"audio": _wav_data_uri()})
    except Exception as e:  # noqa: BLE001
        _skip("audio embedding", f"server audio disabled? ({str(e)[:80]})")
        return
    _record("audio embedding", len(v[0]) == 768)


def test_second_model_isolation() -> None:
    """bge-m3 shares the server but not the space -- document, don't mix."""
    try:
        from omlx_client import _request
    except ImportError:
        _skip("second model isolation", "import failed")
        return
    ids = [m["id"] for m in _request("/v1/models", method="GET")["data"]]
    other = [i for i in ids if i != "embeddinggemma-2-bf16"]
    if not other:
        _skip("second model isolation", "only eg2 loaded")
        return
    a = _request("/v1/embeddings", {"model": other[0], "input": "space check"})
    dim_other = len(a["data"][0]["embedding"])
    _record("other model answers with own dim", dim_other in (768, 1024), f"{other[0]} dim={dim_other}")


TESTS = [
    test_models_visible,
    test_text_single,
    test_text_batch,
    test_content_parts,
    test_items_text,
    test_image_data_uri,
    test_image_parts,
    test_text_image_fusion,
    test_mixed_items_batch,
    test_mrl_truncate_renorm,
    test_usage_counts,
    test_semantic_text,
    test_semantic_cross_modal,
    test_task_prefix_steering,
    test_long_input_truncation,
    test_auth_error_surface,
    test_unreachable_surface,
    test_audio_if_enabled,
    test_second_model_isolation,
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", dest="only", default="", help="substring filter on test name")
    args = ap.parse_args()

    # reachability gate with a friendly message
    try:
        from omlx_client import models

        models()
    except Exception as e:  # noqa: BLE001
        print(f"oMLX unreachable at {settings.resolved_base_url() if hasattr(settings, 'resolved_base_url') else '?'}: {e}")
        print("  check .env OMLX_BASE_URL / OMLX_API_KEY, or start the server")
        return 2

    print(f"oMLX live tests -> {getattr(settings, 'base_url', '') or 'default'} "
          f"model={settings.model_path}")
    n_fail = 0
    for t in TESTS:
        if args.only and args.only not in t.__name__:
            continue
        try:
            t()
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            _record(t.__name__, False, f"{type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
        n_fail += sum(1 for name, st, _ in _results if name == t.__name__ and st == "FAIL")

    fails = [r for r in _results if r[1] == "FAIL"]
    skipped = [r for r in _results if r[1] == "skip"]
    print(f"\n== {len(_results) - len(fails) - len(skipped)} passed, "
          f"{len(fails)} failed, {len(skipped)} skipped ==")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
