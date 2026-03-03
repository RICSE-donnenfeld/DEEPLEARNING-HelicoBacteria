#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model_autoencoder import (
    ANNOTATED_ROOT,
    HELICO_ROOT,
    HelicoPatchDataset,
    PatchAutoencoder,
    compute_reconstruction_errors,
    get_transform,
    patient_stratified_kfold_subsets,
    roc_threshold_optimal,
    subset_healthy_only,
)


METRICS_KEYS_COMMON = ["accuracy", "precision", "recall", "specificity", "f1", "val_loss_best"]
METRICS_KEYS_AE_EXTRA = ["threshold", "tpr", "fpr"]


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = max(1, tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def robust_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_fold_metrics(model_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(model_dir.glob("fold_*/metrics.json")):
        payload = json.loads(p.read_text())
        if "fold" not in payload:
            payload["fold"] = float(p.parent.name.split("_")[-1])
        rows.append(payload)
    rows.sort(key=lambda x: x["fold"])
    return rows


def read_threshold_file(path: Path) -> tuple[float, str]:
    lines = path.read_text().strip().splitlines()
    threshold = float(lines[0])
    aggregation = lines[1].strip() if len(lines) > 1 and lines[1].strip() else "max_local"
    return threshold, aggregation


def plot_fold_histogram(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    title: str,
    out_path: Path,
) -> None:
    healthy = errors[labels == 0]
    unhealthy = errors[labels == 1]
    bins = 40

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(healthy, bins=bins, alpha=0.55, density=True, label="healthy", color="tab:blue")
    ax.hist(unhealthy, bins=bins, alpha=0.55, density=True, label="unhealthy", color="tab:orange")
    ax.axvline(threshold, color="tab:red", linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Reconstruction error")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed AE fold diagnostics with histograms and robust stats.")
    parser.add_argument("--cv-root", type=Path, default=Path("cv"), help="Root CV directory.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=42, help="K-fold seed.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--out-dir", type=Path, default=Path("cv/plots"), help="Output directory.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not HELICO_ROOT.exists():
        raise FileNotFoundError(f"HelicoDataSet not found at {HELICO_ROOT}")
    csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches_stripped.csv"
    if not csv_path.exists():
        csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches.csv"

    dataset = HelicoPatchDataset(csv_path=csv_path, images_root=ANNOTATED_ROOT, transform=get_transform())
    folds = patient_stratified_kfold_subsets(dataset, n_splits=args.k_folds, seed=args.seed)

    ae_root = args.cv_root / "autoencoder"
    cnn_root = args.cv_root / "cnn"
    ae_metrics_rows = load_fold_metrics(ae_root)
    cnn_metrics_rows = load_fold_metrics(cnn_root)

    if len(folds) != len(ae_metrics_rows):
        print(
            f"Warning: fold split count ({len(folds)}) != autoencoder metrics count ({len(ae_metrics_rows)})."
        )

    fold_errors: dict[int, np.ndarray] = {}
    fold_labels: dict[int, np.ndarray] = {}
    per_fold_rows: list[dict[str, Any]] = []

    # Evaluate each fold checkpoint on its validation fold to build error distributions.
    for fold_i, (train_sub, val_sub) in enumerate(folds, start=1):
        fold_dir = ae_root / f"fold_{fold_i}"
        ckpt = fold_dir / "autoencoder_model.pth"
        thr_file = fold_dir / "autoencoder_threshold.txt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        if not thr_file.exists():
            raise FileNotFoundError(f"Missing threshold file: {thr_file}")

        threshold, aggregation = read_threshold_file(thr_file)
        healthy_train_count = len(subset_healthy_only(train_sub, dataset))
        train_total = len(train_sub)
        val_total = len(val_sub)

        val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=2)
        model = PatchAutoencoder().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        errors, labels = compute_reconstruction_errors(model, val_loader, device, aggregation=aggregation)
        fold_errors[fold_i] = errors
        fold_labels[fold_i] = labels

        preds_local = (errors > threshold).astype(np.int64)
        local_metrics = binary_metrics(labels.astype(np.int64), preds_local)

        hist_path = out_dir / f"ae_fold_{fold_i}_error_hist.png"
        plot_fold_histogram(
            errors,
            labels,
            threshold,
            title=f"AE fold {fold_i}: error distribution",
            out_path=hist_path,
        )

        # Classwise robust stats for diagnosis.
        healthy_err = errors[labels == 0]
        unhealthy_err = errors[labels == 1]
        row = {
            "fold": fold_i,
            "train_total": train_total,
            "train_healthy_count": healthy_train_count,
            "train_healthy_ratio": (healthy_train_count / max(1, train_total)),
            "val_total": val_total,
            "val_positive_count": int(np.sum(labels == 1)),
            "val_negative_count": int(np.sum(labels == 0)),
            "threshold_local": float(threshold),
            "aggregation": aggregation,
            "healthy_error_median": float(np.median(healthy_err)),
            "healthy_error_iqr": float(np.percentile(healthy_err, 75) - np.percentile(healthy_err, 25)),
            "unhealthy_error_median": float(np.median(unhealthy_err)),
            "unhealthy_error_iqr": float(np.percentile(unhealthy_err, 75) - np.percentile(unhealthy_err, 25)),
        }
        row.update({f"local_{k}": v for k, v in local_metrics.items()})
        per_fold_rows.append(row)

    # Dedicated fold-3 plot alias for convenience.
    if 3 in fold_errors:
        thr3 = float(next(r["threshold_local"] for r in per_fold_rows if r["fold"] == 3))
        plot_fold_histogram(
            fold_errors[3],
            fold_labels[3],
            thr3,
            title="Fold 3 error distribution (healthy vs unhealthy)",
            out_path=out_dir / "fold3_error_distributions.png",
        )

    # Global threshold calibration from pooled out-of-fold validation errors.
    all_errors = np.concatenate([fold_errors[i] for i in sorted(fold_errors)])
    all_labels = np.concatenate([fold_labels[i] for i in sorted(fold_labels)])
    global_threshold, global_tpr, global_fpr = roc_threshold_optimal(all_errors, all_labels, positive_class=1)

    for row in per_fold_rows:
        i = int(row["fold"])
        err = fold_errors[i]
        lab = fold_labels[i].astype(np.int64)
        pred_global = (err > global_threshold).astype(np.int64)
        m_global = binary_metrics(lab, pred_global)
        row["threshold_global"] = float(global_threshold)
        for k, v in m_global.items():
            row[f"global_{k}"] = v

    # Save tabular diagnostics.
    diagnostics_path = out_dir / "ae_fold_diagnostics.json"
    diagnostics_path.write_text(json.dumps(per_fold_rows, indent=2))

    # Healthy train counts chart.
    folds_idx = [int(float(r["fold"])) for r in per_fold_rows]
    healthy_counts = [int(r["train_healthy_count"]) for r in per_fold_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(folds_idx, healthy_counts, color="tab:green", alpha=0.85)
    ax.set_title("AE healthy-train samples per fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Healthy train sample count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ae_healthy_train_counts.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Compare local-threshold vs global-threshold metrics.
    metric_cmp = ["accuracy", "precision", "recall", "specificity", "f1"]
    fig, axs = plt.subplots(1, len(metric_cmp), figsize=(16, 3.8), sharey=False)
    for ax, key in zip(axs, metric_cmp):
        local_vals = [float(r[f"local_{key}"]) for r in per_fold_rows]
        global_vals = [float(r[f"global_{key}"]) for r in per_fold_rows]
        x = np.arange(len(folds_idx))
        w = 0.38
        ax.bar(x - w / 2, local_vals, width=w, label="local thr")
        ax.bar(x + w / 2, global_vals, width=w, label="global thr")
        ax.set_xticks(x, [str(f) for f in folds_idx])
        ax.set_title(key)
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", alpha=0.25)
    axs[0].legend(loc="lower left")
    fig.suptitle("AE per-fold metrics: local threshold vs pooled global threshold", y=1.06)
    fig.tight_layout()
    fig.savefig(out_dir / "ae_local_vs_global_threshold_metrics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Robust stats summary for both models.
    robust_summary: dict[str, Any] = {"cnn": {}, "autoencoder": {}, "global_threshold": {}}
    for key in METRICS_KEYS_COMMON:
        robust_summary["cnn"][key] = robust_stats([float(r[key]) for r in cnn_metrics_rows])
        robust_summary["autoencoder"][key] = robust_stats([float(r[key]) for r in ae_metrics_rows])
    for key in METRICS_KEYS_AE_EXTRA:
        robust_summary["autoencoder"][key] = robust_stats([float(r[key]) for r in ae_metrics_rows])
    robust_summary["global_threshold"] = {
        "threshold": float(global_threshold),
        "tpr": float(global_tpr),
        "fpr": float(global_fpr),
    }

    # Also robust stats for global-threshold performance across folds.
    robust_summary["autoencoder_global_threshold_metrics"] = {
        key: robust_stats([float(r[f"global_{key}"]) for r in per_fold_rows]) for key in metric_cmp
    }

    robust_path = out_dir / "robust_stats_summary.json"
    robust_path.write_text(json.dumps(robust_summary, indent=2))

    # Human-readable report.
    rep_lines: list[str] = []
    rep_lines.append("AE fold diagnostics report")
    rep_lines.append("=========================")
    rep_lines.append("")
    rep_lines.append(f"Pooled global threshold (OOF ROC): {global_threshold:.6f}")
    rep_lines.append(f"Global ROC point: TPR={global_tpr:.4f}, FPR={global_fpr:.4f}")
    rep_lines.append("")
    rep_lines.append("Per-fold health-train counts and local thresholds:")
    for r in per_fold_rows:
        rep_lines.append(
            f"- fold {r['fold']}: train healthy={r['train_healthy_count']}/{r['train_total']} "
            f"({r['train_healthy_ratio']:.3f}), val pos={r['val_positive_count']}, "
            f"val neg={r['val_negative_count']}, local threshold={r['threshold_local']:.6f}, "
            f"local f1={r['local_f1']:.4f}, global f1={r['global_f1']:.4f}"
        )
    rep_lines.append("")
    rep_lines.append("Robust stats (median [Q1, Q3], IQR):")
    for model_name in ("cnn", "autoencoder"):
        rep_lines.append(f"[{model_name}]")
        keys = METRICS_KEYS_COMMON + (METRICS_KEYS_AE_EXTRA if model_name == "autoencoder" else [])
        for key in keys:
            rs = robust_summary[model_name][key]
            rep_lines.append(
                f"  - {key}: median={rs['median']:.4f} "
                f"[{rs['q1']:.4f}, {rs['q3']:.4f}], IQR={rs['iqr']:.4f}, "
                f"mean+-std={rs['mean']:.4f}+-{rs['std']:.4f}"
            )
    report_path = out_dir / "ae_fold_diagnostics_report.txt"
    report_path.write_text("\n".join(rep_lines))

    print(f"Saved diagnostics to {out_dir}")
    for p in sorted(out_dir.glob("ae_*")):
        print(f"- {p}")
    print(f"- {out_dir / 'fold3_error_distributions.png'}")
    print(f"- {robust_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
