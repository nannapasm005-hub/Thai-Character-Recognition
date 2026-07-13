"""สร้างรูปประกอบ README จาก dataset จริง -> assets/
  1) dataset_samples.png  : ตาราง 72 คลาส ตัวอย่าง 1 รูป/คลาส + label ตัวอักษรไทย
  2) augmentation_preview.png : original + gentle/mild/strong
รูปพวกนี้เล็กและอยู่ใน assets/ (ไม่ถูก gitignore) จึงขึ้น GitHub ได้
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams
from PIL import Image

from src.dataset import build_samples, load_class_mapping, _TRAIN_AUGMENTS

for f in ["Thonburi", "Sukhumvit Set", "Ayuthaya", "Sarabun", "Silom"]:
    if f in {x.name for x in font_manager.fontManager.ttflist}:
        rcParams["font.family"] = f
        break
rcParams["axes.unicode_minus"] = False

os.makedirs("assets", exist_ok=True)
samples, class_names, class_to_idx, idx_to_char = build_samples(".", "class_mapping.json")

first_of = {}
for i, (path, lbl) in enumerate(samples):
    first_of.setdefault(lbl, path)

# ---------- 1) dataset sample grid (72 คลาส, 9x8) ----------
ncol, nrow = 9, 8
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.5, nrow * 1.95))
axes = axes.flatten()
for ax in axes:
    ax.axis("off")
for k, name in enumerate(class_names):
    ax = axes[k]
    img = Image.open(first_of[class_to_idx[name]]).convert("L")
    ax.imshow(img, cmap="gray")
    ax.set_title(f"{idx_to_char[class_to_idx[name]]}\n{name}", fontsize=7, pad=2)
fig.suptitle(f"Thai Character Dataset — {len(class_names)} classes (1 sample each)",
             fontsize=15, weight="bold", y=1.002)
plt.tight_layout(h_pad=2.2)
plt.savefig("assets/dataset_samples.png", dpi=110, bbox_inches="tight", facecolor="white")
plt.close()
print("✅ assets/dataset_samples.png")

# ---------- 2) augmentation preview ----------
def denorm(t):
    return np.clip(t.squeeze(0).numpy() * 0.5 + 0.5, 0, 1)

demo_path = first_of[class_to_idx["kor_kai"]]
orig = Image.open(demo_path).convert("L")
titles = ["Original", "Gentle", "Mild", "Strong"]
imgs = [np.array(orig) / 255.0] + [denorm(t(orig)) for t in _TRAIN_AUGMENTS]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, im, ti in zip(axes, imgs, titles):
    ax.imshow(im, cmap="gray")
    ax.set_title(ti, fontsize=12)
    ax.axis("off")
fig.suptitle("Augmentation levels (grayscale 1-channel) — ก / kor_kai",
             fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig("assets/augmentation_preview.png", dpi=110, bbox_inches="tight", facecolor="white")
plt.close()
print("✅ assets/augmentation_preview.png")

for p in ["assets/dataset_samples.png", "assets/augmentation_preview.png"]:
    print(f"   {p}: {os.path.getsize(p)//1024} KB")
