#!/usr/bin/env bash
# A/B benchmark: local strix-halo server (tufa:50233) vs oMLX (coral:11534).
# Same requests, x3 runs each scenario, median. Writes JSON to stdout.
set -uo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python - <<'PY'
import base64, json, os, statistics, time, urllib.error, urllib.request

RUNS = 3

def post(url, body, timeout=600, native=True):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers=_headers(native))
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return (time.perf_counter() - t0) * 1000, d

def med(fn):
    xs = []
    for _ in range(RUNS):
        xs.append(fn())
    return {"p50": round(statistics.median(xs), 1),
            "min": round(min(xs), 1),
            "max": round(max(xs), 1)}

_KEY = ""
for _line in open(".env"):
    if _line.startswith("OMLX_API_KEY="):
        _KEY = _line.split("=", 1)[1].strip()

def _headers(native):
    h = {"Content-Type": "application/json"}
    if not native and _KEY:
        h["Authorization"] = "Bearer " + _KEY
    return h

BACKENDS = {
    "strix-halo(tufa)": {"base": "http://tufa:50233", "model": "embeddinggemma-2", "native": True},
    "ommlx(coral)": {"base": "http://coral:11534/v1", "model": "embeddinggemma-2-bf16", "native": False},
}

def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()

PNG = b64("sdk-test/assets/blue_bg.png")
WAV = b64("sdk-test/assets/tone440.wav")
MP4 = b64("sdk-test/assets/blue_clip.mp4")

def bench(name, cfg):
    base, model, native = cfg["base"], cfg["model"], cfg["native"]

    def post(url, body, timeout=600):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
            headers=_headers(native))
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            json.load(r)
        return (time.perf_counter() - t0) * 1000

    out = {"backend": name}

    # text single / batch via the right routes
    def t1():
        if native:
            ms = post(base + "/embed", {"input": "single text probe"})
        else:
            ms = post(base + "/embeddings", {"model": model, "input": "single text probe"})
        return ms
    def tN(n):
        texts = [f"doc {i} sample text" for i in range(n)]
        if native:
            ms = post(base + "/embed", {"input": texts})
        else:
            ms = post(base + "/embeddings", {"model": model, "input": texts})
        return ms
    out["text_x1"] = med(t1)
    out["text_x32"] = med(lambda: tN(32))
    out["text_x64"] = med(lambda: tN(64))

    def img1():
        if native:
            ms = post(base + "/embed", {"input": {"image": {"b64": PNG}}})
        else:
            ms = post(base + "/embeddings", {"model": model,
                        "items": [{"image": "data:image/png;base64," + PNG}]})
        return ms
    def imgN(n):
        if native:
            ms = post(base + "/embed", {"input": [{"image": {"b64": PNG}} for _ in range(n)]})
        else:
            ms = post(base + "/embeddings", {"model": model,
                        "items": [{"image": "data:image/png;base64," + PNG} for _ in range(n)]})
        return ms
    out["image_x1"] = med(img1)
    out["image_x8"] = med(lambda: imgN(8))
    out["image_x16_ms_per"] = round(med(lambda: imgN(16))["p50"] / 16, 1)

    def aud1():
        try:
            if native:
                return post(base + "/embed", {"input": {"audio": {"b64": WAV}}})
            return post(base + "/embeddings", {"model": model,
                        "items": [{"audio": "data:audio/wav;base64," + WAV}]})
        except urllib.error.HTTPError as e:
            e.read()
            return None
    vals = [aud1() for _ in range(RUNS)]
    ok = [v for v in vals if v is not None]
    out["audio_x1"] = (
        {"p50": round(statistics.median(ok), 1), "min": round(min(ok), 1), "max": round(max(ok), 1)}
        if ok else {"p50": "ERR(500)", "min": "-", "max": f"{len(ok)}/{RUNS} ok"}
    )

    def vid1():
        if native:
            ms = post(base + "/embed", {"input": {"video": {"b64": MP4}}})
        else:  # ommlx: frame-extraction proxy happens client-side; measure server part only
            import subprocess, tempfile, glob
            with tempfile.TemporaryDirectory() as td:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i",
                                "sdk-test/assets/blue_clip.mp4", "-vf", "fps=2,scale=512:-2",
                                f"{td}/f%03d.jpg"], check=True, capture_output=True)
                frames = sorted(glob.glob(td + "/f*.jpg"))
                items = [{"image": "data:image/jpeg;base64," + b64(f)} for f in frames]
                ms = post(base + "/embeddings", {"model": model, "items": items})
        return ms
    out["video_2s"] = med(vid1)
    return out

results = {name: bench(name, cfg) for name, cfg in BACKENDS.items()}
print(json.dumps(results, indent=1, ensure_ascii=False))
PY
