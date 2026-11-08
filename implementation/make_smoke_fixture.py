"""Tiny SYNTHETIC fixture to test the code path end to end. Not cassava data; results from it are meaningless."""
import os, sys
import numpy as np
from PIL import Image, ImageDraw

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/smoke_data"
os.makedirs(os.path.join(out, "train_images"), exist_ok=True)
rng = np.random.default_rng(0)
rows = []
for k in range(5):
    for j in range(45):
        im = Image.new("RGB", (160, 120), (90 + rng.integers(0, 30), 120 + rng.integers(0, 40), 60))
        d = ImageDraw.Draw(im)
        d.ellipse((30, 20, 130, 100), fill=(30, 110 + 15 * k, 40))
        for _ in range(k * 3):
            x, y = rng.integers(40, 120), rng.integers(30, 90)
            d.ellipse((x, y, x + 8, y + 8), fill=(200, 190, 60) if k % 2 else (120, 70, 30))
        px = np.asarray(im).astype(np.int16) + rng.integers(-12, 12, (120, 160, 3))
        name = f"{k}_{j:03d}.jpg"
        Image.fromarray(np.clip(px, 0, 255).astype(np.uint8)).save(os.path.join(out, "train_images", name), quality=85)
        rows.append((name, k))
import pandas as pd
pd.DataFrame(rows, columns=["image_id", "label"]).to_csv(os.path.join(out, "train.csv"), index=False)
print("fixture written to", out, len(rows), "images")
