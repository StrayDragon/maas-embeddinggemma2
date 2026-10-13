"""Smoke test: all modalities of EmbeddingGemma 2 on ROCm."""
import io
import os
import time
import numpy as np
import torch
import av
import soundfile as sf
from PIL import Image, ImageDraw

MODEL = os.environ.get("MODEL_PATH", "models/embeddinggemma-2")

def make_video(path, seconds=6, fps=2, size=(224, 160)):
    with av.open(path, "w") as c:
        stream = c.add_stream("libx264", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for t in range(seconds * fps):
            img = Image.new("RGB", size, (t * 20 % 255, 100, 200))
            d = ImageDraw.Draw(img); d.ellipse((t * 8 % 100, 30, 80 + t * 8 % 100, 110), fill=(255, 180, 0))
            frame = av.VideoFrame.from_image(img)
            for pkt in stream.encode(frame): c.mux(pkt)
        for pkt in stream.encode(None): c.mux(pkt)

def main():
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(MODEL, device="cuda", model_kwargs={"dtype": torch.bfloat16})
    print(f"loaded in {time.time()-t0:.1f}s | dim={model.get_sentence_embedding_dimension()}")

    # 1) text
    e = model.encode(["hello world", "goodbye"], prompt_name="Document")
    print("text:", np.asarray(e).shape, float(np.asarray(e)[0] @ np.asarray(e)[1]))

    # 2) image
    img = Image.new("RGB", (320, 240), (30, 144, 255))
    ImageDraw.Draw(img).rectangle((50, 50, 200, 180), fill=(255, 215, 0))
    e = model.encode({"image": [img]})
    print("image:", np.asarray(e).shape)

    # 3) audio (AudioDict, 16k mono)
    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    wav = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    e = model.encode({"audio": {"array": wav, "sampling_rate": sr}})
    print("audio(dict):", np.asarray(e).shape)
    buf = io.BytesIO(); sf.write(buf, wav, sr, format="WAV"); buf.seek(0)
    with open("/tmp/test.wav", "wb") as f: f.write(buf.getvalue())
    try:
        e = model.encode({"audio": "/tmp/test.wav"})
        print("audio(path):", np.asarray(e).shape)
    except Exception as ex:
        print("audio(path) FAILED:", type(ex).__name__, str(ex)[:200])

    # 4) video (path, pyav backend)
    make_video("/tmp/test.mp4")
    try:
        e = model.encode({"video": "/tmp/test.mp4"})
        print("video(path):", np.asarray(e).shape)
    except Exception as ex:
        print("video(path) FAILED:", type(ex).__name__, str(ex)[:300])

    # 5) interleaved multimodal
    try:
        e = model.encode({
            "text": "A test clip.  <",  # placeholder replaced below if needed
            "image": [img],
            "video": "/tmp/test.mp4",
        })
        print("multimodal:", np.asarray(e).shape)
    except Exception as ex:
        print("multimodal FAILED:", type(ex).__name__, str(ex)[:300])
        # proper interleaved variant with explicit tokens
        try:
            e = model.encode({
                "text": "A test clip.  <", "image": [img]
            })
            print("text+image:", np.asarray(e).shape)
        except Exception as ex2:
            print("text+image FAILED:", type(ex2).__name__, str(ex2)[:300])

    # 6) batch of mixed inputs + truncation
    try:
        e = model.encode([
            "plain string input",
            {"text": "doc number two"},
            {"image": [img]},
        ], prompt_name="Document", truncate_dim=256)
        print("mixed batch@256d:", np.asarray(e).shape)
    except Exception as ex:
        print("mixed batch FAILED:", type(ex).__name__, str(ex)[:300])

    print("device mem:", torch.cuda.mem_get_info()[1] / 2**20, "MiB total")

if __name__ == "__main__":
    main()
