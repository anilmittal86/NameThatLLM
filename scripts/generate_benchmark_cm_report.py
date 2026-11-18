#!/usr/bin/env python3
"""
scripts/generate_benchmark_cm_report.py
=======================================
Generates a comprehensive forensic evaluation report and 18x18 confusion matrix
from any local_eval_cache.npz or clean_benchmark_predictions.npz.

Outputs:
1. Model-wise solo balanced accuracy & raw accuracy table.
2. Weighted / optimized ensemble balanced accuracy.
3. Full 18x18 confusion matrix (Markdown format).
4. Per-class performance breakdown and top sister-pair confusion hubs.
5. Family-level aggregated accuracy.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR / "src"))

from namethatllm.config import LABELS, FAMILIES


def generate_cm_report(input_path: Path, output_md: Path, title: str = "Benchmark Evaluation Report") -> str:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = np.load(str(input_path), allow_pickle=True)
    files = set(data.files)

    # 1. Resolve true labels and groups
    if "y_unseen" in files and len(data["y_unseen"]) > 0:
        y_true = np.asarray(data["y_unseen"], dtype=int)
    elif "labels" in files and len(data["labels"]) > 0:
        y_true = np.asarray(data["labels"], dtype=int)
    else:
        raise ValueError("Could not find 'y_unseen' or 'labels' in input file.")

    N = len(y_true)
    num_classes = len(LABELS)

    # 2. Resolve models and unseen log-probabilities
    model_results = []
    blend_lp = None

    if "model_names" in files:
        model_names = [str(m) for m in data["model_names"]]
        learned_weights = data["model_learned_weights"] if "model_learned_weights" in files else np.ones(len(model_names))
        
        acc_weights = 0.0
        blend_lp = np.zeros((N, num_classes), dtype=np.float64)

        for idx, m_name in enumerate(model_names):
            k = f"m_{idx}_unseen_lp"
            if k in files:
                lp = np.asarray(data[k], dtype=np.float64)
                if lp.shape == (N, num_classes):
                    pred = lp.argmax(axis=1)
                    bacc = balanced_accuracy_score(y_true, pred)
                    acc = np.mean(pred == y_true)
                    w = float(learned_weights[idx])
                    model_results.append({
                        "name": m_name,
                        "weight": w,
                        "bacc": bacc,
                        "acc": acc,
                        "lp": lp,
                    })
                    if w > 0:
                        blend_lp += lp * w
                        acc_weights += w

        # Extra streams (e.g. Claude specialist, handwriting, etc.)
        if "num_extra" in files:
            num_ex = int(data["num_extra"][0])
            for ex_idx in range(num_ex):
                ex_name_k = f"extra_{ex_idx}_name"
                ex_lp_k = f"extra_{ex_idx}_unseen_lp"
                ex_w_k = f"extra_{ex_idx}_weight"
                if ex_name_k in files and ex_lp_k in files:
                    ex_name = str(data[ex_name_k][0])
                    ex_lp = np.asarray(data[ex_lp_k], dtype=np.float64)
                    ex_w = float(data[ex_w_k][0]) if ex_w_k in files else 0.0
                    if ex_lp.shape == (N, num_classes):
                        pred = ex_lp.argmax(axis=1)
                        bacc = balanced_accuracy_score(y_true, pred)
                        acc = np.mean(pred == y_true)
                        model_results.append({
                            "name": ex_name,
                            "weight": ex_w,
                            "bacc": bacc,
                            "acc": acc,
                            "lp": ex_lp,
                        })
                        if ex_w > 0:
                            blend_lp += ex_lp * ex_w
                            acc_weights += ex_w

    elif "logits_blend" in files:
        # Format from clean_benchmark_1800_predictions.npz
        blend_lp = np.asarray(data["logits_blend"], dtype=np.float64)
        if "logits_plain" in files:
            p_pred = data["logits_plain"].argmax(axis=1)
            model_results.append({
                "name": "Qwen3-14B Plain Head",
                "weight": 1.0,
                "bacc": balanced_accuracy_score(y_true, p_pred),
                "acc": np.mean(p_pred == y_true),
                "lp": data["logits_plain"],
            })
        if "probs_sinkhorn" in files:
            s_pred = data["probs_sinkhorn"].argmax(axis=1)
            model_results.append({
                "name": "Full Specialist + Sinkhorn",
                "weight": 1.0,
                "bacc": balanced_accuracy_score(y_true, s_pred),
                "acc": np.mean(s_pred == y_true),
                "lp": np.log(np.clip(data["probs_sinkhorn"], 1e-9, 1.0)),
            })

    if blend_lp is None or len(model_results) == 0:
        raise ValueError("Could not extract predictions from input file.")

    ensemble_preds = blend_lp.argmax(axis=1)
    ensemble_bacc = balanced_accuracy_score(y_true, ensemble_preds)
    ensemble_acc = np.mean(ensemble_preds == y_true)

    # 3. Compute Confusion Matrix
    cm = confusion_matrix(y_true, ensemble_preds, labels=list(range(num_classes)))
    class_supports = cm.sum(axis=1)
    class_recalls = np.zeros(num_classes)
    for c in range(num_classes):
        class_recalls[c] = cm[c, c] / max(class_supports[c], 1)

    # 4. Top Confused Pairs (Bilateral)
    bilateral_pairs = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            err_ij = cm[i, j]
            err_ji = cm[j, i]
            tot_err = err_ij + err_ji
            if tot_err > 0:
                bilateral_pairs.append({
                    "class_a": LABELS[i],
                    "class_b": LABELS[j],
                    "a_as_b": err_ij,
                    "b_as_a": err_ji,
                    "total_err": tot_err,
                })
    bilateral_pairs.sort(key=lambda x: x["total_err"], reverse=True)

    # 5. Build Markdown Report
    lines = [
        f"# {title}",
        f"**Source File**: `{input_path.name}`  ",
        f"**Sample Count**: {N:,} samples ({N // num_classes} samples/class)  ",
        f"**Overall Ensemble Balanced Accuracy**: **`{ensemble_bacc:.4%}`**  ",
        f"**Overall Ensemble Top-1 Accuracy**: `{ensemble_acc:.4%}`  ",
        "",
        "---",
        "",
        "## 1. Model-Wise Performance & Weights",
        "",
        "| Model Name | Ensembling Weight | Solo Balanced Acc | Solo Top-1 Acc |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for m in model_results:
        lines.append(f"| **{m['name']}** | `{m['weight']:.4f}` | **`{m['bacc']:.4%}`** | `{m['acc']:.4%}` |")
    lines.append(f"| **Full Ensemble (Blended)** | **`1.0000`** | **`{ensemble_bacc:.4%}`** | **`{ensemble_acc:.4%}`** |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Top-10 Confusion Hubs (Bilateral Error Pairs)",
        "",
        "| Rank | Model A | Model B | A misclassified as B | B misclassified as A | Total Mutual Errors | Hub Family |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :--- |",
    ])

    for r, p in enumerate(bilateral_pairs[:10], start=1):
        # Determine hub family
        fam = "Cross-Family"
        for fam_name, members in FAMILIES.items():
            if LABELS.index(p["class_a"]) in members and LABELS.index(p["class_b"]) in members:
                fam = f"**{fam_name.upper()} Intra-Family**"
                break
        lines.append(
            f"| #{r} | `{p['class_a']}` | `{p['class_b']}` | {p['a_as_b']} | {p['b_as_a']} | **{p['total_err']}** | {fam} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Per-Class Accuracy & Breakdown",
        "",
        "| Class ID | Canonical Model Name | Support | Correct | Recall (Accuracy) | Top Confused Class (Count) |",
        "| :---: | :--- | :---: | :---: | :---: | :--- |",
    ])

    for c in range(num_classes):
        correct = cm[c, c]
        sup = class_supports[c]
        rec = class_recalls[c]
        # Find top confused
        row = cm[c, :].copy()
        row[c] = -1
        top_c = row.argmax()
        top_err = row[top_c]
        conf_str = f"`{LABELS[top_c]}` ({top_err})" if top_err > 0 else "None"
        lines.append(f"| `{c:02d}` | `{LABELS[c]}` | {sup} | {correct} | **`{rec:.2%}`** | {conf_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Full 18x18 Confusion Matrix",
        "",
        "Rows represent True Classes; Columns represent Predicted Classes.",
        "",
    ])

    # Table header with class indices 0..17
    header = "| True \\ Pred | " + " | ".join([f"`{i:02d}`" for i in range(num_classes)]) + " |"
    sep = "| :--- | " + " | ".join([":---:" for _ in range(num_classes)]) + " |"
    lines.append(header)
    lines.append(sep)

    for i in range(num_classes):
        row_vals = []
        for j in range(num_classes):
            val = cm[i, j]
            if i == j:
                row_vals.append(f"**{val}**")
            elif val > 5:
                row_vals.append(f"*{val}*")
            else:
                row_vals.append(str(val))
        lines.append(f"| **`{i:02d}`** `{LABELS[i]}` | " + " | ".join(row_vals) + " |")

    content = "\n".join(lines)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n[Report Generator] Forensic report written to: {output_md}")
    return content


def main():
    parser = argparse.ArgumentParser(description="Generate Confusion Matrix and Model Accuracy Report")
    parser.add_argument("--input-file", type=str, required=True, help="Path to .npz file (local_eval_cache or benchmark predictions)")
    parser.add_argument("--output-md", type=str, default="local_data/benchmark_report.md", help="Path to output markdown report")
    parser.add_argument("--title", type=str, default="Benchmark Evaluation Report", help="Report Title")
    args = parser.parse_args()

    inp = Path(args.input_file)
    out = Path(args.output_md)
    generate_cm_report(inp, out, title=args.title)


if __name__ == "__main__":
    main()
