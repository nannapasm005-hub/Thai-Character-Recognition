"""เติมตาราง RESULTS_TABLE ใน README.md จาก results_summary.json (รันหลังเทรนเสร็จ)

    python fill_results.py
"""
import json
import os
import re

LABELS = [("resnet50", "ResNet50"),
          ("efficientnet_b3", "EfficientNet-B3"),
          ("mobilenet_v3", "MobileNetV3-Large")]


def main():
    if not os.path.exists("results_summary.json"):
        raise SystemExit("ยังไม่มี results_summary.json — เทรนก่อน")
    s = json.load(open("results_summary.json", encoding="utf-8"))

    rows = ["| Model | Test Accuracy | Macro F1 | Best Val Acc | Epochs |",
            "|---|---|---|---|---|"]
    for key, label in LABELS:
        if key in s:
            r = s[key]
            ta = f"{r.get('test_accuracy', 0)*100:.2f}%"
            f1 = f"{r.get('macro_f1', 0):.4f}"
            bv = f"{r.get('best_val_acc', 0)*100:.2f}%"
            ep = str(r.get("epochs_run", "-"))
        else:
            ta = f1 = bv = ep = "_pending_"
        rows.append(f"| {label} | {ta} | {f1} | {bv} | {ep} |")
    table = "\n".join(rows)

    readme = open("README.md", encoding="utf-8").read()
    new = re.sub(r"<!-- RESULTS_TABLE -->.*?<!-- /RESULTS_TABLE -->",
                 f"<!-- RESULTS_TABLE -->\n{table}\n<!-- /RESULTS_TABLE -->",
                 readme, flags=re.DOTALL)
    open("README.md", "w", encoding="utf-8").write(new)
    print("✅ อัปเดตตารางผลใน README.md แล้ว")
    print(table)


if __name__ == "__main__":
    main()
