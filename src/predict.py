"""Single-image inference for Thai Character Recognition.

แก้บั๊กสำคัญจากเวอร์ชันเดิม:
  1. ใช้ val_test_transforms (Grayscale 1ch + Normalize(0.5,0.5)) ให้ตรงกับตอนเทรน
     ไม่ใช่ RGB + ImageNet normalize
  2. โหลดโมเดลถูกวิธี: create_model() แล้ว load_state_dict() (ไฟล์เซฟเป็น state_dict)
  3. ไม่มี tkinter — รับ image_path เป็น parameter
  4. แปลผล index -> ตัวอักษรไทยจริง ผ่าน class_mapping.json

ใช้งาน:
    from src.predict import test_single_image
    test_single_image("kor_kai/0001.jpg", model_name="efficientnet_b3")

หรือ command line:
    python -m src.predict path/to/image.jpg --model efficientnet_b3
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import load_class_mapping, val_test_transforms
from .model import MODEL_LABELS, create_model, get_device

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPPING_PATH = os.path.join(HERE, "class_mapping.json")
OUTPUT_DIRS = {
    "resnet50": "outputs_resnet50",
    "efficientnet_b3": "outputs_efficientnet_b3",
    "mobilenet_v3": "outputs_mobilenet_v3",
}


def load_trained_model(model_name, device=None):
    """สร้าง architecture แล้วโหลด state_dict ของ best model"""
    device = device or get_device()
    out_dir = os.path.join(HERE, OUTPUT_DIRS[model_name])
    model_path = os.path.join(out_dir, f"{model_name}_best.pt")
    idx_path = os.path.join(out_dir, "class_to_idx.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {model_path} (เทรนก่อนนะครับ)")

    with open(idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    model = create_model(model_name, num_classes, pretrained=False, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, class_to_idx, device


def test_single_image(image_path, model_name="efficientnet_b3", topk=3, show=False):
    """ทำนายภาพเดียว -> dict {predicted_folder, predicted_char, confidence, topk}

    show=True จะ plot ภาพ + ผลทำนาย (ต้องมี matplotlib + GUI/inline)
    """
    model, class_to_idx, device = load_trained_model(model_name)
    idx_to_folder = {v: k for k, v in class_to_idx.items()}
    mapping = load_class_mapping(MAPPING_PATH)  # folder -> ตัวอักษรไทย

    with Image.open(image_path) as im:
        pil = im.convert("L")
    tensor = val_test_transforms(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    conf, idx = torch.max(probs, dim=0)
    pred_folder = idx_to_folder[idx.item()]
    pred_char = mapping.get(pred_folder, "?")

    k = min(topk, probs.numel())
    top_conf, top_idx = torch.topk(probs, k)
    topk_list = [
        {
            "folder": idx_to_folder[i.item()],
            "char": mapping.get(idx_to_folder[i.item()], "?"),
            "confidence": float(c),
        }
        for c, i in zip(top_conf, top_idx)
    ]

    result = {
        "image": image_path,
        "model": model_name,
        "predicted_folder": pred_folder,
        "predicted_char": pred_char,
        "confidence": float(conf),
        "topk": topk_list,
    }

    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(pil, cmap="gray")
        ax[0].axis("off")
        ax[0].set_title("Input")
        lines = [f"Predicted: {pred_char}  ({pred_folder})",
                 f"Confidence: {conf*100:.2f}%", "", "Top-k:"]
        lines += [f"  {t['char']} ({t['folder']}): {t['confidence']*100:.1f}%"
                  for t in topk_list]
        ax[1].text(0.05, 0.5, "\n".join(lines), va="center", fontsize=11)
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Predict a single Thai character image")
    p.add_argument("image_path")
    p.add_argument("--model", default="efficientnet_b3",
                   choices=list(OUTPUT_DIRS.keys()))
    p.add_argument("--topk", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    res = test_single_image(args.image_path, args.model, args.topk)
    print(f"\n📷 {res['image']}  ({MODEL_LABELS[res['model']]})")
    print(f"➡️  ทำนาย: {res['predicted_char']}  (folder: {res['predicted_folder']})")
    print(f"🎯 Confidence: {res['confidence']*100:.2f}%")
    print("Top-k:")
    for t in res["topk"]:
        print(f"   {t['char']} ({t['folder']}): {t['confidence']*100:.1f}%")
