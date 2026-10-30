"""
========================================================================================
DocGuard: Multimodal Document Forgery & Manipulation Detection Platform
Module 7: Automated Experiment Evaluator & Graph Generator (evaluate_model.py)
----------------------------------------------------------------------------------------
Model Evaluated: Master DocumentFusionNetwork (final_forgery_detector.pth)
Input Dataset:   Held-out Stratified Validation Set (dataset/real & dataset/forged)
Output Artifacts:
  - evaluation/loss_curve.png
  - evaluation/accuracy_curve.png
  - evaluation/confusion_matrix.png
  - evaluation/metrics_summary.json
========================================================================================

Forensic & Academic Defense Rationale:
--------------------------------------
1. Precision vs. Recall in Document Fraud Systems:
   - Precision: TP / (TP + FP)
     Measures the trustworthiness of a positive tampering alert.
     A low precision means high False Positives (FP): innocent claimants or legitimate
     patients have valid reimbursement claims unjustly rejected or flagged for fraud.
   - Recall (Sensitivity): TP / (TP + FN)
     Measures the audit coverage: what fraction of actual fraudulent documents were caught.
     A low recall means high False Negatives (FN): altered invoices or forged checks slip
     past the filter uninspected, causing severe direct financial loss.
   - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
     The harmonic mean that penalizes systems that sacrifice one metric for the other.

2. Presentation-Ready Visualizations:
   Matplotlib & Seaborn visualizations formatted at 300 DPI for thesis reports,
   conference publications, and academic defense slide decks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless rendering without display server
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Import Master Model Architecture & DataLoader
from dataset_loader import get_dataloaders
from phase3_fusion import DocumentFusionNetwork


# =============================================================================
# Submodule 7.1: Model & Dataset Ingestion
# =============================================================================

def load_trained_fusion_model(
    weights_path: Union[str, Path] = "final_forgery_detector.pth",
    device: Optional[torch.device] = None,
) -> Tuple[DocumentFusionNetwork, Dict[str, Any]]:
    """Loads the production multimodal model weights into DocumentFusionNetwork.

    Args:
        weights_path: Path to serialized PyTorch state dictionary.
        device: Target device (CUDA or CPU).

    Returns:
        Tuple: (instantiated model in eval mode, checkpoint metadata dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_file = Path(weights_path).resolve()
    if not weights_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_file}")

    print(f"  [+] Loading production checkpoint from: {weights_file}")
    checkpoint = torch.load(weights_file, map_location="cpu", weights_only=False)

    # Instantiate model without re-loading pretrained backbones from disk
    model = DocumentFusionNetwork(
        ela_weights_path=None,
        rgb_weights_path=None,
        pretrained_backbones=False,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint


# =============================================================================
# Submodule 7.2: Evaluation Engine
# =============================================================================

def evaluate_validation_dataset(
    model: DocumentFusionNetwork,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    decision_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Runs inference across the validation dataset and computes forensic metrics.

    Args:
        model: DocumentFusionNetwork instance in eval mode.
        dataloader: Validation DataLoader emitting synchronized dual-stream batches.
        device: CUDA or CPU device.
        decision_threshold: Decision boundary (default: 0.5).

    Returns:
        Dict: Complete evaluation metrics, raw predictions, and sample breakdowns.
    """
    y_true: List[int] = []
    y_pred_probs: List[float] = []
    sample_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            ela_tensors = batch["ela_tensor"].to(device, non_blocking=True)
            rgb_tensors = batch["rgb_tensor"].to(device, non_blocking=True)
            labels = batch["label"].view(-1).long().tolist()
            file_paths = batch.get("path", ["" for _ in labels])

            outputs = model(ela_tensors, rgb_tensors).view(-1).cpu().tolist()

            for path, lbl, prob in zip(file_paths, labels, outputs):
                pred_binary = 1 if prob >= decision_threshold else 0
                y_true.append(lbl)
                y_pred_probs.append(float(prob))

                sample_records.append({
                    "file_path": str(path),
                    "true_label": "Forged" if lbl == 1 else "Authentic",
                    "predicted_prob": round(float(prob), 4),
                    "predicted_label": "Forged" if pred_binary == 1 else "Authentic",
                    "is_correct": bool(pred_binary == lbl),
                })

    y_true_arr = np.array(y_true, dtype=int)
    y_probs_arr = np.array(y_pred_probs, dtype=float)
    y_pred_binary = (y_probs_arr >= decision_threshold).astype(int)

    # Compute Standard Academic Classification Metrics
    acc = float(accuracy_score(y_true_arr, y_pred_binary))
    prec = float(precision_score(y_true_arr, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_binary, zero_division=0))

    # ROC-AUC calculation (guarding against single-class edge cases)
    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_probs_arr))
    except ValueError:
        roc_auc = 0.5

    # Confusion Matrix Breakdown: [ [TN, FP], [FN, TP] ]
    cm = confusion_matrix(y_true_arr, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # False Positive Rate & False Negative Rate
    fpr = float(fp / max(1, (fp + tn)))
    fnr = float(fn / max(1, (fn + tp)))

    return {
        "metrics": {
            "accuracy": round(acc * 100.0, 2),
            "precision": round(prec * 100.0, 2),
            "recall": round(rec * 100.0, 2),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "false_positive_rate": round(fpr * 100.0, 2),
            "false_negative_rate": round(fnr * 100.0, 2),
            "decision_threshold": decision_threshold,
        },
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
        "total_samples": len(y_true),
        "y_true": y_true,
        "y_pred_probs": y_pred_probs,
        "y_pred_binary": y_pred_binary.tolist(),
        "sample_records": sample_records,
    }


# =============================================================================
# Submodule 7.3: Visualization & Graph Plotting Generators
# =============================================================================

def plot_loss_curves(
    fusion_history_path: Union[str, Path] = "phase3_fusion_history.json",
    ela_history_path: Union[str, Path] = "phase1_ela_history.json",
    rgb_history_path: Union[str, Path] = "phase2_rgb_history.json",
    output_path: Union[str, Path] = "evaluation/loss_curve.png",
) -> Path:
    """Generates comparative loss convergence curves across all 3 training phases."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load Phase 3 Fusion History
    with open(fusion_history_path, "r", encoding="utf-8") as f:
        fusion_hist = json.load(f)
    train_loss = fusion_hist["metrics"]["train_loss"]
    val_loss = fusion_hist["metrics"]["val_loss"]
    epochs = list(range(1, len(train_loss) + 1))

    # Optional Multi-Stream Comparison
    has_ela = Path(ela_history_path).exists()
    has_rgb = Path(rgb_history_path).exists()

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    # Plot Master Fusion Curves
    ax.plot(epochs, train_loss, marker="o", linewidth=2.5, color="#1f77b4", label="Master Fusion: Train Loss")
    ax.plot(epochs, val_loss, marker="s", linewidth=2.5, linestyle="--", color="#ff7f0e", label="Master Fusion: Val Loss")

    # Overlay Stream 1 & Stream 2 Validation if available
    if has_ela:
        with open(ela_history_path, "r", encoding="utf-8") as f:
            raw_ela = json.load(f)
            ela_val = raw_ela.get("metrics", raw_ela)["val_loss"]
            ax.plot(epochs, ela_val, marker="^", linewidth=1.5, linestyle=":", color="#2ca02c", alpha=0.7, label="Stream 1 (ELA) Val Loss")
    if has_rgb:
        with open(rgb_history_path, "r", encoding="utf-8") as f:
            raw_rgb = json.load(f)
            rgb_val = raw_rgb.get("metrics", raw_rgb)["val_loss"]
            ax.plot(epochs, rgb_val, marker="v", linewidth=1.5, linestyle=":", color="#d62728", alpha=0.7, label="Stream 2 (RGB) Val Loss")

    best_idx = int(np.argmin(val_loss))
    ax.scatter(epochs[best_idx], val_loss[best_idx], color="#d62728", s=150, zorder=5, edgecolors="black", label=f"Best Checkpoint (Epoch {epochs[best_idx]})")
    ax.annotate(
        f"Min Val Loss: {val_loss[best_idx]:.4f}",
        xy=(epochs[best_idx], val_loss[best_idx]),
        xytext=(epochs[best_idx] - 0.5, val_loss[best_idx] + 0.008),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=6),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="gray", lw=1),
    )

    ax.set_title("DocGuard: Multimodal Training & Validation Loss Trajectory", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Training Epoch", fontsize=12, labelpad=10)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=12, labelpad=10)
    ax.set_xticks(epochs)
    ax.set_ylim(bottom=min(train_loss) - 0.02, top=max(val_loss) + 0.03)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"  [+] Saved Loss Curve plot to: {out_file}")
    return out_file


def plot_accuracy_curves(
    fusion_history_path: Union[str, Path] = "phase3_fusion_history.json",
    output_path: Union[str, Path] = "evaluation/accuracy_curve.png",
) -> Path:
    """Generates the training and validation accuracy progression plot."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(fusion_history_path, "r", encoding="utf-8") as f:
        fusion_hist = json.load(f)
    train_acc = fusion_hist["metrics"]["train_acc"]
    val_acc = fusion_hist["metrics"]["val_acc"]
    epochs = list(range(1, len(train_acc) + 1))

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    ax.plot(epochs, train_acc, marker="o", linewidth=2.5, color="#2ca02c", label="Training Accuracy (%)")
    ax.plot(epochs, val_acc, marker="s", linewidth=2.5, linestyle="--", color="#1f77b4", label="Validation Accuracy (%)")

    # Annotate final accuracy
    ax.annotate(
        f"Final Train Acc: {train_acc[-1]:.1f}%",
        xy=(epochs[-1], train_acc[-1]),
        xytext=(epochs[-1] - 1.2, train_acc[-1] - 4),
        arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.2, headwidth=6),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e6ffed", ec="#2ca02c", lw=1),
    )

    ax.set_title("DocGuard: Master Multimodal Accuracy Trajectory", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Training Epoch", fontsize=12, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12, labelpad=10)
    ax.set_xticks(epochs)
    ax.set_ylim(40.0, 100.0)
    ax.legend(loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()

    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"  [+] Saved Accuracy Curve plot to: {out_file}")
    return out_file


def plot_confusion_matrix(
    cm_2x2: List[List[int]],
    output_path: Union[str, Path] = "evaluation/confusion_matrix.png",
) -> Path:
    """Generates an annotated, high-resolution forensic confusion matrix heatmap."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cm = np.array(cm_2x2)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    total = np.sum(cm)

    # Rich label annotations with forensic context
    labels = np.array([
        [
            f"True Negative (TN)\n{tn} / {total}\n({tn/total*100:.1f}%)\n[Authentic Verified]",
            f"False Positive (FP)\n{fp} / {total}\n({fp/total*100:.1f}%)\n[Claimant False Alarm]",
        ],
        [
            f"False Negative (FN)\n{fn} / {total}\n({fn/total*100:.1f}%)\n[Undetected Fraud Leak]",
            f"True Positive (TP)\n{tp} / {total}\n({tp/total*100:.1f}%)\n[Forgery Successfully Flagged]",
        ],
    ])

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=1.5,
        linecolor="#dddddd",
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        xticklabels=["Predicted: Authentic", "Predicted: Forged"],
        yticklabels=["Actual: Authentic", "Actual: Forged"],
        ax=ax,
    )

    ax.set_title("DocGuard: Master Forensic Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Model Decision", fontsize=12, labelpad=10)
    ax.set_ylabel("Ground Truth Grounding", fontsize=12, labelpad=10)
    plt.tight_layout()

    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"  [+] Saved Confusion Matrix plot to: {out_file}")
    return out_file


# =============================================================================
# Submodule 7.4: Master Evaluation Pipeline
# =============================================================================

def run_evaluation_pipeline(
    weights_path: str = "final_forgery_detector.pth",
    fusion_history_path: str = "phase3_fusion_history.json",
    dataset_dir: str = "dataset",
    output_dir: str = "evaluation",
    decision_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Coordinates full validation evaluation and artifact generation."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("DocGuard: Running Automated Evaluation & Artifact Generation (Module 7)")
    print("=" * 75)
    print(f"  * Compute Device: {device.type}")

    # 1. Load Model
    model, ckpt = load_trained_fusion_model(weights_path=weights_path, device=device)

    # 2. Ingest Validation Set via DataLoader
    print("  [+] Ingesting held-out validation dataset via dataset_loader...")
    _, val_loader, manifest = get_dataloaders(
        base_dir=dataset_dir,
        batch_size=4,
        num_workers=0,  # Single-process for deterministic evaluation
        test_size=0.20,
    )
    val_count = manifest["val_count"]
    print(f"  [+] Validation Set: {val_count} documents ({manifest['val_distribution']['real']} Authentic, {manifest['val_distribution']['forged']} Forged)")

    # 3. Evaluate Dataset
    print(f"\n[Stage 1/3] Executing Model Inference (Threshold = {decision_threshold})...")
    eval_results = evaluate_validation_dataset(
        model=model,
        dataloader=val_loader,
        device=device,
        decision_threshold=decision_threshold,
    )
    metrics = eval_results["metrics"]

    print("-" * 75)
    print(f"  * Overall Accuracy:         {metrics['accuracy']:.2f}%")
    print(f"  * Precision (True Fraud):   {metrics['precision']:.2f}%")
    print(f"  * Recall (Audit Coverage):  {metrics['recall']:.2f}%")
    print(f"  * F1-Score:                 {metrics['f1_score']:.4f}")
    print(f"  * ROC-AUC Score:            {metrics['roc_auc']:.4f}")
    print(f"  * False Positive Rate:      {metrics['false_positive_rate']:.2f}%")
    print(f"  * False Negative Rate:      {metrics['false_negative_rate']:.2f}%")
    print("-" * 75)

    # 4. Generate Presentation Visualizations
    print("\n[Stage 2/3] Rendering Presentation-Ready Artifacts...")
    loss_img = plot_loss_curves(
        fusion_history_path=fusion_history_path,
        output_path=out_dir / "loss_curve.png",
    )
    acc_img = plot_accuracy_curves(
        fusion_history_path=fusion_history_path,
        output_path=out_dir / "accuracy_curve.png",
    )
    cm_img = plot_confusion_matrix(
        cm_2x2=eval_results["confusion_matrix"]["matrix_2x2"],
        output_path=out_dir / "confusion_matrix.png",
    )

    # 5. Save Structured Metrics Summary JSON
    print("\n[Stage 3/3] Exporting Structured Metrics Summary...")
    metrics_summary_path = out_dir / "metrics_summary.json"
    summary_payload = {
        "evaluation_target": "final_forgery_detector.pth",
        "validation_samples_count": val_count,
        "metrics": metrics,
        "confusion_matrix": eval_results["confusion_matrix"],
        "generated_artifacts": {
            "loss_curve": str(loss_img),
            "accuracy_curve": str(acc_img),
            "confusion_matrix": str(cm_img),
        },
        "sample_level_predictions": eval_results["sample_records"],
    }
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=4)
    print(f"  [+] Saved Metrics Summary to: {metrics_summary_path}")

    print("=" * 75)
    print("SUCCESS: Module 7 (Automated Experiment Evaluator) Completed Successfully!")
    print("=" * 75)

    return summary_payload


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DocGuard Module 7: Automated Experiment Evaluator & Graph Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="final_forgery_detector.pth",
        help="Path to trained master production weights.",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="phase3_fusion_history.json",
        help="Path to training history JSON.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Root path to the document dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Directory where evaluation artifacts and graphs are saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision boundary threshold for classification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation_pipeline(
        weights_path=args.weights,
        fusion_history_path=args.history,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        decision_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
