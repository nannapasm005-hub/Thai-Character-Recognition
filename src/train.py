"""Training entrypoint for Thai Character Recognition.

ใช้รันได้ทั้งจาก notebook (import train_model) และจาก command line:

    python -m src.train --models all --epochs 20
    python -m src.train --models efficientnet_b3 --epochs 15 --max-per-class 200

มีฟีเจอร์:
  - Stratified train/val/test split (เก็บครบ 72 คลาส)
  - Best model checkpoint ตาม val accuracy
  - Early stopping (monitor val_loss)
  - ReduceLROnPlateau scheduler
  - บันทึก history + test metrics ลง outputs_<model>/
"""
import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ThaiCharDataset, build_samples, stratified_split
from .model import MODEL_LABELS, SUPPORTED_MODELS, create_model, get_device

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = HERE
MAPPING_PATH = os.path.join(HERE, "class_mapping.json")

OUTPUT_DIRS = {
    "resnet50": "outputs_resnet50",
    "efficientnet_b3": "outputs_efficientnet_b3",
    "mobilenet_v3": "outputs_mobilenet_v3",
}


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,
                model_name, device, scheduler=None, patience=4):
    """ฝึกโมเดล 1 ตัว -> คืน (best_model, history)

    - เก็บ best weights ตาม val accuracy สูงสุด
    - early stopping เมื่อ val_loss ไม่ดีขึ้นติดต่อกัน `patience` epochs
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{num_epochs} - {model_name}\n{'='*60}")

        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc.item())
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.float() / len(val_loader.dataset)
        history["val_loss"].append(val_epoch_loss)
        history["val_acc"].append(val_epoch_acc.item())
        print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")

        if scheduler is not None:
            scheduler.step(val_epoch_loss)

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"✅ New best model! Val Acc: {best_acc:.4f}")

        # early stopping ดูจาก val_loss
        if val_epoch_loss < best_val_loss - 1e-4:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping ที่ epoch {epoch+1} "
                      f"(val_loss ไม่ดีขึ้น {patience} epochs)")
                break

    elapsed = time.time() - since
    print(f"\nTraining complete in {elapsed//60:.0f}m {elapsed%60:.0f}s | "
          f"Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history


def evaluate(model, test_loader, class_names, device):
    """ประเมินบน test set -> (accuracy, classification_report dict)"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    uniq = sorted(set(all_labels))
    report = classification_report(
        all_labels, all_preds, labels=uniq,
        target_names=[class_names[i] for i in uniq],
        output_dict=True, zero_division=0,
    )
    return acc, report


def run(model_names, epochs, batch_size, lr, max_per_class, num_workers, seed,
        augment="single"):
    device = get_device()
    print(f"🖥️  Device: {device}")

    samples, class_names, class_to_idx, _ = build_samples(DATA_ROOT, MAPPING_PATH)
    num_classes = len(class_names)
    print(f"📊 {len(samples)} images | {num_classes} classes")

    if max_per_class:
        # subset เร็ว ๆ สำหรับ smoke test: เก็บไม่เกิน N ภาพต่อคลาส
        from collections import defaultdict
        kept, cnt = [], defaultdict(int)
        for s in samples:
            if cnt[s[1]] < max_per_class:
                kept.append(s)
                cnt[s[1]] += 1
        samples = kept
        print(f"⚡ max-per-class={max_per_class} -> {len(samples)} images")

    train_idx, val_idx, test_idx = stratified_split(samples, seed=seed)
    mult = 3 if augment == "triple" else 1
    print(f"📦 train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"(augment={augment}, train samples/epoch = {len(train_idx)*mult})")

    train_ds = ThaiCharDataset(samples, train_idx, mode="train", augment=augment)
    val_ds = ThaiCharDataset(samples, val_idx, mode="val")
    test_ds = ThaiCharDataset(samples, test_idx, mode="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    results = {}
    for name in model_names:
        print(f"\n{'#'*70}\n##  {MODEL_LABELS[name]}\n{'#'*70}")
        out_dir = os.path.join(HERE, OUTPUT_DIRS[name])
        os.makedirs(out_dir, exist_ok=True)

        model = create_model(name, num_classes, pretrained=True, device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                         factor=0.5, patience=2)

        model, history = train_model(model, train_loader, val_loader, criterion,
                                     optimizer, epochs, name, device, scheduler)

        torch.save(model.state_dict(), os.path.join(out_dir, f"{name}_best.pt"))
        with open(os.path.join(out_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(out_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

        if len(test_idx) > 0:
            acc, report = evaluate(model, test_loader, class_names, device)
            with open(os.path.join(out_dir, f"classification_report_{name}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            results[name] = {
                "test_accuracy": acc,
                "macro_f1": report["macro avg"]["f1-score"],
                "best_val_acc": max(history["val_acc"]),
                "epochs_run": len(history["train_loss"]),
            }
            print(f"\n✅ {MODEL_LABELS[name]} Test Accuracy: {acc*100:.2f}%")
        else:
            results[name] = {"best_val_acc": max(history["val_acc"]),
                             "epochs_run": len(history["train_loss"])}

    with open(os.path.join(HERE, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n🎉 เสร็จสิ้น! สรุปผลที่ results_summary.json")
    for name, r in results.items():
        print(f"  {MODEL_LABELS[name]}: {json.dumps(r, ensure_ascii=False)}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Train Thai character recognition models")
    p.add_argument("--models", default="all",
                   help="'all' หรือชื่อโมเดลคั่นด้วย comma (resnet50,efficientnet_b3,mobilenet_v3)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--augment", choices=["single", "triple"], default="single",
                   help="single = 1 ภาพ/รูป (เร็ว), triple = 3 ภาพ/รูป (ดาต้า ×3)")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="จำกัดภาพต่อคลาส (>0 = smoke test เร็ว ๆ)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    names = SUPPORTED_MODELS if args.models == "all" else args.models.split(",")
    for n in names:
        if n not in SUPPORTED_MODELS:
            raise SystemExit(f"Unknown model: {n}")
    run(names, args.epochs, args.batch_size, args.lr,
        args.max_per_class, args.num_workers, args.seed, args.augment)
