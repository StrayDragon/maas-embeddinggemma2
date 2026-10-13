"""One-off: generate deterministic test assets for multimodal.ts (run via project venv)."""
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

A = Path(__file__).parent / "assets"
A.mkdir(exist_ok=True)


def solid(bg, circle, name):
    im = Image.new("RGB", (224, 224), bg)
    ImageDraw.Draw(im).ellipse((52, 52, 172, 172), fill=circle)
    im.save(A / name)


# distinct background colors, same shape -> color is the discriminative feature
solid((200, 60, 60), (250, 210, 60), "red_bg.png")
solid((60, 110, 200), (250, 210, 60), "blue_bg.png")

t = np.linspace(0, 1, 16000, endpoint=False)
sf.write(A / "tone440.wav", 0.5 * np.sin(2 * np.pi * 440 * t), 16000)
sf.write(A / "tone880.wav", 0.5 * np.sin(2 * np.pi * 880 * t), 16000)

with av.open(str(A / "blue_clip.mp4"), "w") as c:  # blue-themed clip
    st = c.add_stream("libx264", rate=8)
    st.width, st.height, st.pix_fmt = 160, 120, "yuv420p"
    for i in range(16):
        im = Image.new("RGB", (160, 120), (40, 80, 160))
        ImageDraw.Draw(im).ellipse((20 + (i * 6) % 60, 40, 60 + (i * 6) % 60, 90), fill=(250, 210, 60))
        for pkt in st.encode(av.VideoFrame.from_image(im)):
            c.mux(pkt)
    for pkt in st.encode(None):
        c.mux(pkt)

for f in sorted(A.iterdir()):
    print(f, f.stat().st_size, "bytes")
