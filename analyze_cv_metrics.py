#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS_MAIN = ["accuracy", "precision", "recall", "specificity", "f1"]
MODEL_NAMES = ["cnn", "autoencoder"]


def _load_fold_metrics(model_dir: Path) -> list[dict[str, Any]]:
    fold_files = sorted(model_dir.glob("fold_*/metrics.json"))
    if not fold_files:
        return []
    rows: list[dict[str, Any]] = []
    for f in fold_files:
        payload = json.loads(f.read_text())
        if "fold" not in payload:
            # fallback from folder name if needed
            try:
                payload["fold"] = float(f.parent.name.split("_")[-1])
            except Exception:
                payload["fold"] = float(len(rows) + 1)
        rows.append(payload)
    rows.sort(key=lambda x: x["fold"])
    return rows


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if key in r]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if not values:
        return float("nan")
    m = _mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def _write_interpretation(
    out_path: Path,
    data: dict[str, list[dict[str, Any]]],
) -> None:
    lines: list[str] = []
    lines.append("Cross-validation fold interpretation")
    lines.append("=" * 38)
    lines.append("")

    for model_name in MODEL_NAMES:
        rows = data.get(model_name, [])
        if not rows:
            lines.append(f"[{model_name}] No fold metrics found.")
            lines.append("")
            continue

        f1_values = _series(rows, "f1")
        acc_values = _series(rows, "accuracy")
        rec_values = _series(rows, "recall")
        loss_values = _series(rows, "val_loss_best")

        best_f1_idx = max(range(len(rows)), key=lambda i: rows[i].get("f1", -1.0))
        worst_f1_idx = min(range(len(rows)), key=lambda i: rows[i].get("f1", 999.0))

        lines.append(f"[{model_name}]")
        lines.append(
            f"- F1 mean+-std: {_mean(f1_values):.4f} +- {_std(f1_values):.4f}"
        )
        lines.append(
            f"- Accuracy mean+-std: {_mean(acc_values):.4f} +- {_std(acc_values):.4f}"
        )
        lines.append(
            f"- Recall mean+-std: {_mean(rec_values):.4f} +- {_std(rec_values):.4f}"
        )
        lines.append(
            f"- Val loss mean+-std: {_mean(loss_values):.4f} +- {_std(loss_values):.4f}"
        )
        lines.append(
            f"- Best fold by F1: fold {int(rows[best_f1_idx]['fold'])} "
            f"(F1={rows[best_f1_idx]['f1']:.4f}, acc={rows[best_f1_idx]['accuracy']:.4f})"
        )
        lines.append(
            f"- Worst fold by F1: fold {int(rows[worst_f1_idx]['fold'])} "
            f"(F1={rows[worst_f1_idx]['f1']:.4f}, acc={rows[worst_f1_idx]['accuracy']:.4f})"
        )
        if model_name == "autoencoder":
            thr_values = _series(rows, "threshold")
            lines.append(
                f"- Threshold mean+-std: {_mean(thr_values):.4f} +- {_std(thr_values):.4f}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))


def _plot_fold_lines(out_dir: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False)
    axes = axes.flatten()

    for row_idx, model_name in enumerate(MODEL_NAMES):
        rows = data.get(model_name, [])
        if not rows:
            continue
        folds = [int(r["fold"]) for r in rows]
        for i, metric in enumerate(METRICS_MAIN):
            ax = axes[row_idx * 3 + i % 3]
            # Spread metrics across two rows by reusing axes placement.
            # First model uses row 0 axes, second uses row 1 axes.
            if i >= 3:
                ax = axes[row_idx * 3 + (i - 3)]
            # We instead create per-model dedicated figure below to keep clarity.
        # pass

    plt.close(fig)

    for model_name in MODEL_NAMES:
        rows = data.get(model_name, [])
        if not rows:
            continue
        folds = [int(r["fold"]) for r in rows]
        fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
        axs = axs.flatten()
        for i, metric in enumerate(METRICS_MAIN):
            ax = axs[i]
            vals = _series(rows, metric)
            ax.plot(folds, vals, marker="o")
            ax.set_title(metric)
            ax.set_ylim(0.0, 1.02)
            ax.set_xlabel("Fold")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
        # val_loss in the last subplot
        ax_loss = axs[5]
        vals_loss = _series(rows, "val_loss_best")
        ax_loss.plot(folds, vals_loss, marker="o", color="tab:red")
        ax_loss.set_title("val_loss_best")
        ax_loss.set_xlabel("Fold")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.3)
        fig.suptitle(f"{model_name.upper()} - per-fold metrics", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"{model_name}_fold_lines.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def _plot_model_comparison(out_dir: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    # Mean +- std bar chart for shared metrics
    x = list(range(len(METRICS_MAIN)))
    width = 0.36

    cnn_rows = data.get("cnn", [])
    ae_rows = data.get("autoencoder", [])
    cnn_means = [_mean(_series(cnn_rows, m)) for m in METRICS_MAIN]
    cnn_stds = [_std(_series(cnn_rows, m)) for m in METRICS_MAIN]
    ae_means = [_mean(_series(ae_rows, m)) for m in METRICS_MAIN]
    ae_stds = [_std(_series(ae_rows, m)) for m in METRICS_MAIN]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([v - width / 2 for v in x], cnn_means, width=width, yerr=cnn_stds, capsize=4, label="CNN")
    ax.bar([v + width / 2 for v in x], ae_means, width=width, yerr=ae_stds, capsize=4, label="Autoencoder")
    ax.set_xticks(x, METRICS_MAIN)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("CV metrics: mean +- std across folds")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "models_mean_std_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_patch_metrics_boxplot(out_dir: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    """Box plot of patch-level metrics across 5 folds: CNN vs AE (median/IQR, robust view)."""
    cnn_rows = data.get("cnn", [])
    ae_rows = data.get("autoencoder", [])
    if not cnn_rows or not ae_rows:
        return
    metrics = ["accuracy", "precision", "recall", "specificity", "f1"]
    cnn_vals = {m: _series(cnn_rows, m) for m in metrics}
    ae_vals = {m: _series(ae_rows, m) for m in metrics}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.35
    cnn_boxes = [cnn_vals[m] for m in metrics]
    ae_boxes = [ae_vals[m] for m in metrics]
    bp_cnn = ax.boxplot(
        cnn_boxes,
        positions=x - width / 2,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=True,
    )
    bp_ae = ax.boxplot(
        ae_boxes,
        positions=x + width / 2,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=True,
    )
    for patch in bp_cnn["boxes"]:
        patch.set_facecolor("tab:blue")
        patch.set_alpha(0.6)
    for patch in bp_ae["boxes"]:
        patch.set_facecolor("tab:orange")
        patch.set_alpha(0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Patch-level CV metrics: CNN vs AE (5 folds, box = IQR)")
    ax.legend([bp_cnn["boxes"][0], bp_ae["boxes"][0]], ["CNN", "Autoencoder"], loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "patch_metrics_boxplot_cnn_vs_ae.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_autoencoder_thresholds(out_dir: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    rows = data.get("autoencoder", [])
    if not rows:
        return
    folds = [int(r["fold"]) for r in rows]
    thresholds = _series(rows, "threshold")
    tpr = _series(rows, "tpr")
    fpr = _series(rows, "fpr")

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(folds, thresholds, marker="o", color="tab:purple", label="threshold")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Threshold", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(folds, tpr, marker="s", color="tab:green", label="tpr")
    ax2.plot(folds, fpr, marker="^", color="tab:orange", label="fpr")
    ax2.set_ylabel("Rate", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")
    ax2.set_ylim(0.0, 1.02)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [str(l.get_label()) for l in lines]
    ax1.legend(lines, labels, loc="best")
    ax1.set_title("Autoencoder threshold / TPR / FPR per fold")
    fig.tight_layout()
    fig.savefig(out_dir / "autoencoder_threshold_tpr_fpr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CV fold metrics and create plots.")
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=Path("cv"),
        help="Root folder containing cv/cnn and cv/autoencoder.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cv/plots"),
        help="Output directory for generated plots and interpretation text.",
    )
    args = parser.parse_args()

    data: dict[str, list[dict[str, Any]]] = {}
    for model_name in MODEL_NAMES:
        model_dir = args.cv_root / model_name
        rows = _load_fold_metrics(model_dir)
        data[model_name] = rows

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Generating fold metric lines...")
    _plot_fold_lines(args.out_dir, data)
    print("Generating model comparison bar chart...")
    _plot_model_comparison(args.out_dir, data)
    print("Generating patch metrics boxplot...")
    _plot_patch_metrics_boxplot(args.out_dir, data)
    print("Generating autoencoder thresholds plot...")
    _plot_autoencoder_thresholds(args.out_dir, data)
    print("Writing interpretation text...")
    _write_interpretation(args.out_dir / "interpretation.txt", data)

    print(f"Saved plots and interpretation to {args.out_dir}")
    for p in sorted(args.out_dir.glob("*")):
        print(f"- {p}")


if __name__ == "__main__":
    main()
