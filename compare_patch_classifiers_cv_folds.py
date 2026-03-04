#!/usr/bin/env python3
"""
Build patch-level confusion matrices and ROC from 5-fold CV (aggregate across folds).

Loads each fold's checkpoint, runs on that fold's validation set, pools (probs, labels)
and (errors, labels), then plots:
  - Confusion matrices (CNN and AE) from pooled predictions.
  - ROC curves (CNN vs AE) from pooled scores.

Outputs (by default to cv/plots/):
  - compare_confusion_cv_folds.png
  - compare_roc_cv_folds.png

Usage (from project root, after running 5-fold training for both models):
  python compare_patch_classifiers_cv_folds.py
  python compare_patch_classifiers_cv_folds.py --cv-root cv --out-dir cv/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve

from model_autoencoder import (
    ANNOTATED_ROOT,
    HelicoPatchDataset,
    PatchAutoencoder,
    compute_reconstruction_errors,
    get_transform,
)
from model_classifier import SimpleCNN
from src.helico.cv_utils import dataset_patient_stratified_kfold_subsets


PROJECT_ROOT = Path(__file__).resolve().parent
HELICO_ROOT = PROJECT_ROOT / "HelicoDataSet"
N_FOLDS = 5
SEED = 42


def _load_ae_threshold(fold_dir: Path) -> tuple[float, str]:
    path = fold_dir / "autoencoder_threshold.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    lines = path.read_text().strip().splitlines()
    thr = float(lines[0])
    agg = lines[1].strip() if len(lines) > 1 and lines[1].strip() else "max_local"
    return thr, agg


def _compute_cnn_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs_list.extend(probs.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())
    return np.array(probs_list), np.array(labels_list)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CNN and AE on pooled 5-fold CV validation sets (confusion + ROC)."
    )
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=PROJECT_ROOT / "cv",
        help="Root containing cv/cnn and cv/autoencoder.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "cv" / "plots",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=N_FOLDS,
        help=f"Number of folds (must match training). Default: {N_FOLDS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (must match training). Default: {SEED}.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches_stripped.csv"
    if not csv_path.exists():
        csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation CSV not found under {HELICO_ROOT}")

    transform = get_transform()
    dataset = HelicoPatchDataset(csv_path=csv_path, images_root=ANNOTATED_ROOT, transform=transform)
    folds = dataset_patient_stratified_kfold_subsets(dataset, n_splits=args.k_folds, seed=args.seed)

    all_cnn_probs: list[np.ndarray] = []
    all_cnn_labels: list[np.ndarray] = []
    all_ae_errors: list[np.ndarray] = []
    all_ae_labels: list[np.ndarray] = []
    ae_thresholds: list[float] = []

    for fold_idx, (_, val_sub) in enumerate(folds, start=1):
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False, num_workers=2)

        # CNN
        cnn_dir = args.cv_root / "cnn" / f"fold_{fold_idx}"
        cnn_ckpt = cnn_dir / "best_model.pth"
        if not cnn_ckpt.exists():
            cnn_ckpt = cnn_dir / "cnn_model.pth"
        if not cnn_ckpt.exists():
            raise FileNotFoundError(f"CNN checkpoint not found in {cnn_dir}")
        cnn = SimpleCNN(num_classes=2).to(device)
        cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device, weights_only=True))
        probs, labels = _compute_cnn_probs(cnn, val_loader, device)
        all_cnn_probs.append(probs)
        all_cnn_labels.append(labels)

        # AE
        ae_dir = args.cv_root / "autoencoder" / f"fold_{fold_idx}"
        ae_ckpt = ae_dir / "autoencoder_model.pth"
        if not ae_ckpt.exists():
            raise FileNotFoundError(f"AE checkpoint not found in {ae_dir}")
        thr, agg = _load_ae_threshold(ae_dir)
        ae_thresholds.append(thr)
        ae = PatchAutoencoder().to(device)
        ae.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=True))
        errors, ae_lab = compute_reconstruction_errors(ae, val_loader, device, aggregation=agg)
        if not np.array_equal(labels, ae_lab):
            raise RuntimeError(f"Fold {fold_idx}: CNN and AE label order mismatch")
        all_ae_errors.append(errors)
        all_ae_labels.append(ae_lab)

    # Pool
    pooled_cnn_probs = np.concatenate(all_cnn_probs)
    pooled_cnn_labels = np.concatenate(all_cnn_labels)
    pooled_ae_errors = np.concatenate(all_ae_errors)
    pooled_ae_labels = np.concatenate(all_ae_labels)
    mean_ae_thr = float(np.mean(ae_thresholds))

    # Predictions for confusion matrices
    cnn_pred = (pooled_cnn_probs >= 0.5).astype(np.int64)
    ae_pred = (pooled_ae_errors > mean_ae_thr).astype(np.int64)

    def _ensure_2x2(cm: np.ndarray) -> np.ndarray:
        if cm.shape == (2, 2):
            return cm
        full = np.zeros((2, 2), dtype=np.float64)
        for i in range(min(2, cm.shape[0])):
            for j in range(min(2, cm.shape[1])):
                full[i, j] = cm[i, j]
        return full

    cm_cnn = _ensure_2x2(confusion_matrix(pooled_cnn_labels, cnn_pred))
    cm_ae = _ensure_2x2(confusion_matrix(pooled_ae_labels, ae_pred))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Figure: confusion matrices (same layout as compare_patch_classifiers)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    for ax, cm, title in [(ax1, cm_cnn, "CNN (5-fold CV)"), (ax2, cm_ae, "Autoencoder (5-fold CV)")]:
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
    conf_path = args.out_dir / "compare_confusion_cv_folds.png"
    fig.savefig(conf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {conf_path}")

    # Figure: ROC from pooled scores
    fpr_cnn, tpr_cnn, _ = roc_curve(pooled_cnn_labels, pooled_cnn_probs, pos_label=1)
    fpr_ae, tpr_ae, _ = roc_curve(pooled_ae_labels, pooled_ae_errors, pos_label=1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(fpr_cnn, tpr_cnn, label="CNN (5-fold CV pooled)")
    ax.plot(fpr_ae, tpr_ae, label="Autoencoder (5-fold CV pooled)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (validation sets pooled across 5 folds)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    fig.tight_layout()
    roc_path = args.out_dir / "compare_roc_cv_folds.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {roc_path}")


if __name__ == "__main__":
    main()
