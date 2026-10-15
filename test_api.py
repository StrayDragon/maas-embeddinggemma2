"""End-to-end API test: text / image / audio / video / multimodal / batch / truncation / errors."""
import base64
import io
import json
import math
import os
import tempfile
import urllib.error
import urllib.request

import numpy as np
import av
import soundfile as sf
from PIL import Image, ImageDraw

from endpoint import resolve_base_url

BASE = resolve_base_url()
IMG_TOKEN = "<|image|>"
VID_TOKEN = "<|video|>"


def _post(path: str, payload: dict, timeout: int = 300):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


def b64png() -> str:
    img = Image.new("RGB", (320, 240), (30, 144, 255))
    ImageDraw.Draw(img).rectangle((50, 50, 200, 180), fill=(255, 215, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64wav() -> tuple[str, np.ndarray]:
    t = np.linspace(0, 2, 32000, endpoint=False)
    wav = (0.5 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, 16000, format="WAV")
    return base64.b64encode(buf.getvalue()).decode(), wav


def b64mp4(frames: int = 12) -> str:
    fd, p = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with av.open(p, "w") as c:
        st = c.add_stream("libx264", rate=2)
        st.width, st.height, st.pix_fmt = 224, 160, "yuv420p"
        for i in range(frames):
            img = Image.new("RGB", (224, 160), (i * 20 % 255, 100, 200))
            ImageDraw.Draw(img).ellipse((i * 8 % 80, 30, i * 8 % 80 + 40, 110), fill=(255, 180, 0))
            for pkt in st.encode(av.VideoFrame.from_image(img)):
                c.mux(pkt)
        for pkt in st.encode(None):
            c.mux(pkt)
    with open(p, "rb") as f:
        data = f.read()
    os.unlink(p)
    return base64.b64encode(data).decode()


def main() -> None:
    print("health   :", _get("/health"))
    md = _get("/metadata")
    print("metadata : dims", md["mrl_dimensions"], "| device", md["device"], "| dtype", md["dtype"])

    _, e = _post("/embed", {"input": "The northern lights are caused by solar particles."})
    print(f"text          : dim={e['dimension']} count={e['count']} {e['took_ms']}ms")

    _, e = _post("/embed", {"input": ["a", "b", "c"], "prompt_name": "STS", "truncate_dim": 256})
    print(f"batch@256 STS : {len(e['embeddings'])}x{e['dimension']}")

    _, e = _post("/embed", {"input": {"text": "waterproof trail running shoe", "image": {"b64": b64png()}}})
    print(f"text+image    : dim={e['dimension']}")

    wa, wav = b64wav()
    _, e1 = _post("/embed", {"input": {"audio": {"b64": wa}}})
    _, e2 = _post("/embed", {"input": {"audio": {"array": wav.tolist(), "sampling_rate": 16000}}})
    sim = float(np.asarray(e1["embeddings"][0]) @ np.asarray(e2["embeddings"][0]))
    print(f"audio b64/raw : dim={e1['dimension']} cos(b64,raw)={sim:.4f}")

    _, e = _post("/embed", {"input": {"video": {"b64": b64mp4()}}})
    print(f"video         : dim={e['dimension']}")

    _, e = _post(
        "/embed",
        {
            "input": [
                "plain text item",
                {
                    "text": f"Waterproof shoe review. {IMG_TOKEN} grip test {VID_TOKEN}",
                    "image": [{"b64": b64png()}],
                    "video": {"b64": b64mp4()},
                },
            ],
            "prompt_name": "Document",
            "truncate_dim": 512,
        },
    )
    print(f"mixed@512     : {len(e['embeddings'])}x{e['dimension']}")

    for payload, note in [
        ({"input": "x", "prompt_name": "Nope"}, "bad prompt_name  "),
        ({"input": {"foo": 1}}, "unknown key      "),
        ({"input": 3.14}, "scalar input     "),
        ({"prompt": "a", "prompt_name": "STS", "input": "x"}, "prompt+name      "),
        ({"input": {"audio": {"b64": "!!notb64!!"} | {} }}, "bad audio b64    "),
    ]:
        code, resp = _post("/embed", payload)
        print(f"{note}: {code} {str(resp.get('detail', ''))[:70]}")

    # --- regression: CJK + task prefix on /v1/embeddings (former NaN->500) ---
    trig = "对数组按分数排序并取前 K 个的函数"
    trig2 = "按分数从高到低取前若干个结果"
    for note, payload in [
        ("v1 client-prefix     ", {"model": "embeddinggemma-2", "input": f"task: search result | query: {trig}"}),
        ("v1 suffix :SearchQuery", {"model": "embeddinggemma-2:SearchQuery", "input": trig2}),
        ("v1 bare              ", {"model": "embeddinggemma-2", "input": trig}),
    ]:
        code, resp = _post("/v1/embeddings", payload)
        ok = code == 200 and np.isfinite(np.asarray(resp["data"][0]["embedding"])).all()
        print(f"{note}: {code} finite={ok}")
        assert ok, resp
    code, resp = _post("/v1/embeddings", {
        "model": "embeddinggemma-2", "encoding_format": "base64",
        "input": f"task: search result | query: {trig}",
    })
    vals = np.frombuffer(base64.b64decode(resp["data"][0]["embedding"]), dtype="<f4")
    print(f"v1 base64             : {code} finite={bool(np.isfinite(vals).all())}")
    assert code == 200 and np.isfinite(vals).all()

    print("ALL OK")


if __name__ == "__main__":
    main()
