"""Dataset utilities for Thai Character Recognition.

โหลดภาพจากโฟลเดอร์คลาส (ชื่อโรมัน เช่น kor_kai) โดยอ้างอิงรายชื่อคลาสจาก
class_mapping.json เท่านั้น จึงไม่หลงไปสแกนโฟลเดอร์ output/src/__pycache__
ที่อยู่ข้าง ๆ notebook

โครงสร้างที่คาดหวัง:
    <DATA_ROOT>/<class_name>/<image>.jpg|png|jpeg
ตัวอย่าง: round2/kor_kai/0001.jpg
"""
import json
import os
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# ---------------------------------------------------------------------------
# Transforms — training/inference ต้อง normalize เหมือนกัน:
#   Grayscale 1 channel + Normalize((0.5,), (0.5,))
# ---------------------------------------------------------------------------
gentle_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Pad(10, fill=255),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(5, fill=255),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3)),
    transforms.Normalize((0.5,), (0.5,)),
])

mild_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Pad(10, fill=255),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10, fill=255),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.7)),
    transforms.Normalize((0.5,), (0.5,)),
])

strong_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Pad(12, fill=255),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20, fill=255),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=5, fill=255),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.7, 1.0)),
    transforms.Normalize((0.5,), (0.5,)),
])

# Validation/Test/Inference — ไม่ augment (ใช้ตัวนี้กับ test_single_image ด้วย)
val_test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Pad(10, fill=255),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

_TRAIN_AUGMENTS = [gentle_transforms, mild_transforms, strong_transforms]


def load_class_mapping(mapping_path):
    """อ่าน class_mapping.json -> dict {folder_name: ตัวอักษรไทย}"""
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_samples(data_root, mapping_path):
    """สแกนภาพจาก 72 โฟลเดอร์ที่ระบุใน class_mapping.json เท่านั้น

    Returns
    -------
    samples : list[(path, label_idx)]
    class_names : list[str]              # เรียงตามตัวอักษร (เหมือน ImageFolder)
    class_to_idx : dict[str, int]
    idx_to_char : dict[int, str]         # label_idx -> ตัวอักษรไทยจริง
    """
    mapping = load_class_mapping(mapping_path)
    class_names = sorted(mapping.keys())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_char = {i: mapping[name] for name, i in class_to_idx.items()}

    samples = []
    missing = []
    for name in class_names:
        cls_dir = os.path.join(data_root, name)
        if not os.path.isdir(cls_dir):
            missing.append(name)
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if fn.lower().endswith(IMG_EXTS) and not fn.startswith("."):
                samples.append((os.path.join(cls_dir, fn), class_to_idx[name]))

    if missing:
        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์คลาสเหล่านี้ใน {data_root}: {missing}"
        )
    if not samples:
        raise ValueError(f"ไม่พบรูปภาพใน {data_root}")
    return samples, class_names, class_to_idx, idx_to_char


def stratified_split(samples, ratios=(0.8, 0.1, 0.1), seed=42):
    """แบ่ง train/val/test แบบ stratified (per-class) เก็บครบทุกคลาส

    คลาสที่มีภาพน้อย: รับประกันมีอย่างน้อย 1 ภาพใน train; ถ้ามี >=2 จะมี val/test
    เท่าที่จะแบ่งได้ (คลาส 1 ภาพ -> train อย่างเดียว)
    """
    rng = torch.Generator().manual_seed(seed)
    by_class = defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        by_class[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, idxs in by_class.items():
        perm = torch.randperm(len(idxs), generator=rng).tolist()
        idxs = [idxs[p] for p in perm]
        n = len(idxs)
        if n == 1:
            train_idx += idxs
            continue
        n_train = max(1, int(round(ratios[0] * n)))
        n_val = int(round(ratios[1] * n))
        # กันไม่ให้ train กิน test หมด เหลืออย่างน้อย 1 ให้ test เมื่อ n>=3
        if n - n_train - n_val < 1 and n >= 3:
            n_train = n - n_val - 1
        n_train = max(1, min(n_train, n - 1))
        n_val = max(0, min(n_val, n - n_train))
        train_idx += idxs[:n_train]
        val_idx += idxs[n_train:n_train + n_val]
        test_idx += idxs[n_train + n_val:]
    return train_idx, val_idx, test_idx


class ThaiCharDataset(Dataset):
    """อ่านภาพจาก samples list

    mode='train':
        augment='triple' -> แต่ละภาพกลายเป็น 3 ภาพ (gentle/mild/strong) ดาต้า ×3
        augment='single' -> แต่ละภาพ 1 ภาพ สุ่มระดับ augment ทุกครั้งที่ดึง (เร็วกว่า ~3 เท่า)
    mode='val'/'test' -> val_test_transforms (ไม่ augment)
    """

    def __init__(self, samples, indices, mode="train", augment="single"):
        self.samples = samples
        self.indices = list(indices)
        self.mode = mode
        self.augment = augment
        self.is_train = (mode == "train")
        self.triple = self.is_train and augment == "triple"

    def __len__(self):
        return len(self.indices) * 3 if self.triple else len(self.indices)

    def __getitem__(self, idx):
        if self.triple:
            base_i = self.indices[idx // 3]
            tfm = _TRAIN_AUGMENTS[idx % 3]
        elif self.is_train:
            base_i = self.indices[idx]
            tfm = _TRAIN_AUGMENTS[torch.randint(len(_TRAIN_AUGMENTS), (1,)).item()]
        else:
            base_i = self.indices[idx]
            tfm = val_test_transforms
        path, label = self.samples[base_i]
        with Image.open(path) as im:
            image = im.convert("L")  # โหลดเป็น grayscale ตั้งแต่แรก
        return tfm(image), label
