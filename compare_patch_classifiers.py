#!/usr/bin/env python3
"""
Compare patch-level classifiers: CNN (model.py) vs Autoencoder (model_autoencoder.py).

Runs both models on the same validation set and reports accuracy, precision,
a confusion matrix figure (output/compare_confusion.png), and ROC curves (output/compare_roc.png).

Usage (from project root):
  pip install scikit-learn matplotlib
  python compare_patch_classifiers.py
  python compare_patch_classifiers.py --cnn-checkpoint cnn_model.pth --ae-checkpoint autoencoder_model.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    roc_curve,
)

# Import dataset and split from one place so the split is identical
from model_autoencoder import (
    ANNOTATED_ROOT,
    THRESHOLD_PATH as AE_THRESHOLD_PATH,
    HelicoPatchDataset,
    PatchAutoencoder,
    compute_reconstruction_errors,
    get_transform,
    split_dataset_by_patient,
)
from model import SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
if not OUTPUT_ROOT.exists():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
HELICO_ROOT = PROJECT_ROOT / "HelicoDataSet"
VAL_RATIO = 0.2
SEED = 1


def compute_cnn_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Return (prob_unhealthy, labels) as numpy arrays, same order as loader."""
    model.eval()
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(unhealthy)
            probs_list.extend(probs.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())
    return np.array(probs_list), np.array(labels_list)


def _metrics(labels: np.ndarray, pred: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Returns (accuracy, precision, confusion_matrix)."""
    acc = accuracy_score(labels, pred)
    prec = precision_score(labels, pred, zero_division=0)
    cm = confusion_matrix(labels, pred)
    return acc, prec, cm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CNN and Autoencoder patch classifiers on the same validation set."
    )
    parser.add_argument(
        "--cnn-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "cnn_model.pth",
        help="CNN checkpoint path. Default: cnn_model.pth",
    )
    parser.add_argument(
        "--ae-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "autoencoder_model.pth",
        help="Autoencoder checkpoint path. Default: autoencoder_model.pth",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_ROOT / "compare_confusion.png",
        help="Path for confusion matrix figure. Default: output/compare_confusion.png",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"Validation fraction (patient-level). Default: {VAL_RATIO}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for train/val split. Default: {SEED}",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not HELICO_ROOT.exists():
        raise FileNotFoundError(f"HelicoDataSet not found at {HELICO_ROOT}")
    csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches_stripped.csv"
    if not csv_path.exists():
        csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches.csv"

    transform = get_transform()
    full_dataset = HelicoPatchDataset(
        csv_path=csv_path, images_root=ANNOTATED_ROOT, transform=transform
    )
    _, val_sub = split_dataset_by_patient(
        full_dataset, val_ratio=args.val_ratio, seed=args.seed
    )
    val_loader = DataLoader(val_sub, batch_size=32, shuffle=False, num_workers=2)

    # ---- CNN ----
    if not args.cnn_checkpoint.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {args.cnn_checkpoint}")
    cnn = SimpleCNN(num_classes=2).to(device)
    cnn.load_state_dict(
        torch.load(args.cnn_checkpoint, map_location=device, weights_only=True)
    )
    cnn_probs, labels = compute_cnn_probs(cnn, val_loader, device)
    cnn_pred = (cnn_probs >= 0.5).astype(np.int64)

    # ---- Autoencoder ----
    if not args.ae_checkpoint.exists():
        raise FileNotFoundError(f"AE checkpoint not found: {args.ae_checkpoint}")
    ae = PatchAutoencoder().to(device)
    ae.load_state_dict(
        torch.load(args.ae_checkpoint, map_location=device, weights_only=True)
    )
    ae_aggregation = "mean"
    if AE_THRESHOLD_PATH.exists():
        lines = AE_THRESHOLD_PATH.read_text().strip().splitlines()
        ae_threshold = float(lines[0])
        if len(lines) > 1 and lines[1].strip():
            ae_aggregation = lines[1].strip()
    else:
        ae_threshold = None
    ae_errors, ae_labels = compute_reconstruction_errors(
        ae, val_loader, device, aggregation=ae_aggregation
    )
    if not np.array_equal(labels, ae_labels):
        raise RuntimeError("CNN and AE label order mismatch (should not happen).")
    if ae_threshold is None:
        ae_threshold = float(np.median(ae_errors))
        print(
            f"Warning: {AE_THRESHOLD_PATH} not found; using median error as threshold: {ae_threshold:.6f}"
        )
    ae_pred = (ae_errors > ae_threshold).astype(np.int64)

    # ---- Metrics ----
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    print(f"Validation set: {n} patches ({n_neg} healthy, {n_pos} unhealthy)")

    acc_cnn, prec_cnn, cm_cnn = _metrics(labels, cnn_pred)
    acc_ae, prec_ae, cm_ae = _metrics(labels, ae_pred)

    print(f"\n--- CNN (threshold 0.5) ---")
    print(f"  Accuracy:  {acc_cnn:.4f}")
    print(f"  Precision: {prec_cnn:.4f}")
    print(
        f"  Confusion: TN={cm_cnn[0,0]}, FP={cm_cnn[0,1]}, FN={cm_cnn[1,0]}, TP={cm_cnn[1,1]}"
    )

    print(f"\n--- Autoencoder (threshold {ae_threshold:.4f}) ---")
    print(f"  Accuracy:  {acc_ae:.4f}")
    print(f"  Precision: {prec_ae:.4f}")
    print(
        f"  Confusion: TN={cm_ae[0,0]}, FP={cm_ae[0,1]}, FN={cm_ae[1,0]}, TP={cm_ae[1,1]}"
    )

    # ---- Confusion matrix figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping confusion matrix figure)")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        for ax, cm, title in [
            (ax1, cm_cnn, "CNN"),
            (ax2, cm_ae, "Autoencoder"),
        ]:
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Healthy", "Unhealthy"])
            ax.set_yticklabels(["Healthy", "Unhealthy"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(title)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        fig.tight_layout()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        plt.close(fig)
        print(f"\nSaved confusion matrix figure to {args.out}")

    # ---- ROC curves ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pass
    else:
        # CNN: score = P(unhealthy)
        fpr_cnn, tpr_cnn, _ = roc_curve(labels, cnn_probs, pos_label=1)
        # AE: score = reconstruction error (higher = more likely positive)
        fpr_ae, tpr_ae, _ = roc_curve(labels, ae_errors, pos_label=1)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(fpr_cnn, tpr_cnn, label="CNN")
        ax.plot(fpr_ae, tpr_ae, label="Autoencoder")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves (validation set)")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        fig.tight_layout()
        roc_path = args.out.parent / "compare_roc.png"
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        print(f"Saved ROC figure to {roc_path}")


if __name__ == "__main__":
    main()
