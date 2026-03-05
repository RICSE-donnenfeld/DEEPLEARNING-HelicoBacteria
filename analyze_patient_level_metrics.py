#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve  # type: ignore[import-untyped]


METRICS = ["accuracy", "precision", "recall", "specificity", "f1"]


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if key in r]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    m = _mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def _detail_path(patient_root: Path, model: str) -> Path:
    model_dir = patient_root / model
    from_cv = model_dir / "patient_fold_details_from_cv_models.json"
    fallback = model_dir / "patient_fold_details.json"
    if from_cv.exists():
        return from_cv
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Missing patient detail file for '{model}'. "
        f"Expected {from_cv} or {fallback}. Re-run patient_level_pipeline.py first."
    )


def _rows_for_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("split", "")) == split]


def _arrays_from_detail_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    ratios = np.array([float(r["ratio"]) for r in rows], dtype=np.float32)
    if rows and "pred" in rows[0]:
        preds = np.array([int(r["pred"]) for r in rows], dtype=np.int64)
    else:
        tau = float(rows[0]["tau_patient"]) if rows else 0.5
        preds = (ratios > tau).astype(np.int64)
    return labels, ratios, preds


def _binary_metrics_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    specificity = tn / max(1.0, tn + fp)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {"f1": f1, "recall": recall, "specificity": specificity}


def _normalize_density(raw: str) -> str:
    value = raw.strip().upper()
    if value.startswith("NEG"):
        return "NEGATIVE"
    if value.startswith("LOW") or "BAIX" in value:
        return "LOW"
    if value.startswith("HIGH") or "ALT" in value:
        return "HIGH"
    return "UNKNOWN"


def _plot_model_fold_metrics(
    model_name: str,
    cv_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    folds = [int(float(r["fold"])) for r in cv_rows]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    axs = axs.flatten()
    for i, metric in enumerate(METRICS):
        ax = axs[i]
        cv_vals = _series(cv_rows, metric)
        ho_vals = _series(holdout_rows, metric)
        ax.plot(folds, cv_vals, marker="o", label="CV fold")
        ax.plot(folds, ho_vals, marker="s", label="HoldOut per fold")
        ax.set_title(metric)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    ax_tau = axs[5]
    cv_tau = _series(cv_rows, "tau_patient")
    ho_tau = _series(holdout_rows, "tau_patient")
    ax_tau.plot(folds, cv_tau, marker="o", label="CV tau_patient")
    ax_tau.plot(folds, ho_tau, marker="s", label="HoldOut tau_patient")
    ax_tau.set_title("tau_patient")
    ax_tau.set_xlabel("Fold")
    ax_tau.set_ylabel("Threshold")
    ax_tau.grid(alpha=0.3)
    axs[0].legend(loc="lower left")
    fig.suptitle(f"Patient-level fold metrics: {model_name.upper()}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"patient_{model_name}_fold_lines.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_model_comparison(
    cnn_cv_rows: list[dict[str, Any]],
    ae_cv_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    x = list(range(len(METRICS)))
    width = 0.36
    cnn_mean = [_mean(_series(cnn_cv_rows, m)) for m in METRICS]
    cnn_std = [_std(_series(cnn_cv_rows, m)) for m in METRICS]
    ae_mean = [_mean(_series(ae_cv_rows, m)) for m in METRICS]
    ae_std = [_std(_series(ae_cv_rows, m)) for m in METRICS]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], cnn_mean, width=width, yerr=cnn_std, capsize=4, label="CNN")
    ax.bar([i + width / 2 for i in x], ae_mean, width=width, yerr=ae_std, capsize=4, label="AE")
    ax.set_xticks(x, METRICS)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Patient-level CV metrics: CNN vs AE (mean +- std)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "patient_models_cv_mean_std_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_patient_roc_pr(
    cnn_details: list[dict[str, Any]],
    ae_details: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))

    for model_name, rows in [("CNN", cnn_details), ("AE", ae_details)]:
        holdout_rows = _rows_for_split(rows, "holdout")
        y_true, ratios, _ = _arrays_from_detail_rows(holdout_rows)
        fpr, tpr, _ = roc_curve(y_true, ratios)
        precision, recall, _ = precision_recall_curve(y_true, ratios)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(y_true, ratios)

        ax_roc.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={roc_auc:.3f})")
        ax_pr.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={ap:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax_roc.set_title("Patient-level ROC (HoldOut pooled across folds)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(alpha=0.3)
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(out_dir / "patient_roc_cnn_vs_ae_holdout.png", dpi=170, bbox_inches="tight")
    plt.close(fig_roc)

    ax_pr.set_title("Patient-level Precision-Recall (HoldOut pooled across folds)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(alpha=0.3)
    ax_pr.legend(loc="lower left")
    fig_pr.tight_layout()
    fig_pr.savefig(out_dir / "patient_pr_cnn_vs_ae_holdout.png", dpi=170, bbox_inches="tight")
    plt.close(fig_pr)


def _plot_tau_sweep(
    model_name: str,
    detail_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    split_rows = _rows_for_split(detail_rows, "cv_val")
    by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in split_rows:
        by_fold[int(float(row["fold"]))].append(row)
    if not by_fold:
        return

    thresholds = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    f1_curves: list[np.ndarray] = []
    rec_curves: list[np.ndarray] = []
    spec_curves: list[np.ndarray] = []
    tau_by_fold: list[float] = []

    for _, fold_rows in sorted(by_fold.items()):
        y_true, ratios, _ = _arrays_from_detail_rows(fold_rows)
        tau_by_fold.append(float(fold_rows[0]["tau_patient"]))
        f1_vals: list[float] = []
        rec_vals: list[float] = []
        spec_vals: list[float] = []
        for thr in thresholds:
            pred = (ratios > thr).astype(np.int64)
            m = _binary_metrics_arrays(y_true, pred)
            f1_vals.append(m["f1"])
            rec_vals.append(m["recall"])
            spec_vals.append(m["specificity"])
        f1_curves.append(np.array(f1_vals, dtype=np.float32))
        rec_curves.append(np.array(rec_vals, dtype=np.float32))
        spec_curves.append(np.array(spec_vals, dtype=np.float32))

    f1_arr = np.vstack(f1_curves)
    rec_arr = np.vstack(rec_curves)
    spec_arr = np.vstack(spec_curves)

    def _plot_band(ax: plt.Axes, values: np.ndarray, label: str) -> None:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(thresholds, mean, linewidth=2, label=label)
        ax.fill_between(thresholds, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.15)

    fig, ax = plt.subplots(figsize=(9, 6))
    _plot_band(ax, f1_arr, "F1")
    _plot_band(ax, rec_arr, "Recall")
    _plot_band(ax, spec_arr, "Specificity")
    tau_mean = float(np.mean(tau_by_fold))
    ax.axvline(tau_mean, color="black", linestyle="--", linewidth=1.2, label=f"mean tau={tau_mean:.3f}")
    ax.set_title(f"Patient-level threshold sweep on CV val folds: {model_name.upper()}")
    ax.set_xlabel("tau_patient")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"patient_tau_sweep_{model_name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrices(
    model_name: str,
    cv_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    def _mean_cm(rows: list[dict[str, Any]]) -> np.ndarray:
        tn = _mean([float(r["tn"]) for r in rows])
        fp = _mean([float(r["fp"]) for r in rows])
        fn = _mean([float(r["fn"]) for r in rows])
        tp = _mean([float(r["tp"]) for r in rows])
        return np.array([[tn, fp], [fn, tp]], dtype=np.float32)

    cv_cm = _mean_cm(cv_rows)
    ho_cm = _mean_cm(holdout_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, cm, title in [
        (axes[0], cv_cm, "CV mean confusion"),
        (axes[1], ho_cm, "HoldOut-per-fold mean confusion"),
    ]:
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", color="black", fontsize=10)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"Patient-level confusion matrices: {model_name.upper()}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"patient_confusion_{model_name}_cv_holdout.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_ratio_distributions(
    model_name: str,
    detail_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    holdout_rows = _rows_for_split(detail_rows, "holdout")
    y_true, ratios, _ = _arrays_from_detail_rows(holdout_rows)
    neg = ratios[y_true == 0]
    pos = ratios[y_true == 1]
    taus = [float(r["tau_patient"]) for r in holdout_rows]
    tau_mean = float(np.mean(taus)) if taus else 0.5

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0.0, 1.0, 31).tolist()
    ax.hist(neg, bins=bins, alpha=0.55, density=True, label=f"True healthy (n={len(neg)})")
    ax.hist(pos, bins=bins, alpha=0.55, density=True, label=f"True unhealthy (n={len(pos)})")
    ax.axvline(tau_mean, color="black", linestyle="--", linewidth=1.2, label=f"mean tau={tau_mean:.3f}")
    ax.set_title(f"Patient contamination-ratio distributions: {model_name.upper()} (HoldOut)")
    ax.set_xlabel("Contamination ratio")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"patient_ratio_distribution_{model_name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_error_by_burden(
    model_name: str,
    detail_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    holdout_rows = _rows_for_split(detail_rows, "holdout")
    by_burden: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in holdout_rows:
        burden = _normalize_density(str(row.get("density_raw", "UNKNOWN")))
        by_burden[burden].append((int(row["label"]), int(row["pred"])))

    order = [b for b in ["NEGATIVE", "LOW", "HIGH", "UNKNOWN"] if b in by_burden]
    if not order:
        return

    overall_error: list[float] = []
    counts: list[int] = []
    fn_rate_labels: list[str] = []
    fn_rates: list[float] = []
    for burden in order:
        pairs = by_burden[burden]
        n = len(pairs)
        counts.append(n)
        err = sum(int(lbl != pred) for lbl, pred in pairs) / max(1, n)
        overall_error.append(err)
        positives = [(lbl, pred) for lbl, pred in pairs if lbl == 1]
        if positives:
            fn = sum(int(pred == 0) for _, pred in positives)
            fn_rates.append(fn / max(1, len(positives)))
            fn_rate_labels.append(burden)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(order))
    axes[0].bar(x, overall_error, color="#4c78a8")
    axes[0].set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(order, counts)])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Error rate")
    axes[0].set_title("Overall error rate by burden")
    axes[0].grid(axis="y", alpha=0.3)

    if fn_rates:
        x2 = np.arange(len(fn_rates))
        axes[1].bar(x2, fn_rates, color="#f58518")
        axes[1].set_xticks(x2, fn_rate_labels)
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_ylabel("False-negative rate")
        axes[1].set_title("False-negative rate in positive burdens")
        axes[1].grid(axis="y", alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No positive-burden samples available", ha="center", va="center")
        axes[1].set_axis_off()

    fig.suptitle(f"Patient-level error analysis by burden: {model_name.upper()} (HoldOut)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / f"patient_error_by_burden_{model_name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_summary_text(
    cnn_cv_rows: list[dict[str, Any]],
    ae_cv_rows: list[dict[str, Any]],
    cnn_holdout_rows: list[dict[str, Any]],
    ae_holdout_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("Patient-level metrics summary")
    lines.append("============================")
    lines.append("")

    for model_name, cv_rows, ho_rows in [
        ("cnn", cnn_cv_rows, cnn_holdout_rows),
        ("autoencoder", ae_cv_rows, ae_holdout_rows),
    ]:
        lines.append(f"[{model_name}]")
        for metric in METRICS:
            cv_vals = _series(cv_rows, metric)
            ho_vals = _series(ho_rows, metric)
            lines.append(
                f"- {metric}: CV {_mean(cv_vals):.4f}+-{_std(cv_vals):.4f} | "
                f"HoldOut-per-fold {_mean(ho_vals):.4f}+-{_std(ho_vals):.4f}"
            )
        cv_tau = _series(cv_rows, "tau_patient")
        lines.append(f"- tau_patient (CV): {_mean(cv_tau):.4f}+-{_std(cv_tau):.4f}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot patient-level fold metrics and model comparisons.")
    parser.add_argument(
        "--patient-root",
        type=Path,
        default=Path("output/patient_level"),
        help="Root directory containing cnn/ and ae/ patient-level outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/patient_level/plots"),
        help="Output directory for figures and text summary.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cnn_cv_path = args.patient_root / "cnn" / "patient_cv_folds_from_cv_models.json"
    ae_cv_path = args.patient_root / "ae" / "patient_cv_folds_from_cv_models.json"
    cnn_ho_path = args.patient_root / "cnn" / "holdout_metrics_per_fold_from_cv_models.json"
    ae_ho_path = args.patient_root / "ae" / "holdout_metrics_per_fold_from_cv_models.json"
    cnn_detail_path = _detail_path(args.patient_root, "cnn")
    ae_detail_path = _detail_path(args.patient_root, "ae")

    cnn_cv_rows = _load_json(cnn_cv_path)
    ae_cv_rows = _load_json(ae_cv_path)
    cnn_ho_rows = _load_json(cnn_ho_path)
    ae_ho_rows = _load_json(ae_ho_path)
    cnn_detail_rows = _load_json(cnn_detail_path)
    ae_detail_rows = _load_json(ae_detail_path)

    print("Generating fold metrics lines...")
    _plot_model_fold_metrics("cnn", cnn_cv_rows, cnn_ho_rows, out_dir)
    _plot_model_fold_metrics("autoencoder", ae_cv_rows, ae_ho_rows, out_dir)
    
    print("Generating cross-model CV bar comparison...")
    _plot_cross_model_comparison(cnn_cv_rows, ae_cv_rows, out_dir)
    
    print("Generating patient-level ROC and PR curves...")
    _plot_patient_roc_pr(cnn_detail_rows, ae_detail_rows, out_dir)
    
    print("Generating patient-level threshold sweeps...")
    _plot_tau_sweep("cnn", cnn_detail_rows, out_dir)
    _plot_tau_sweep("ae", ae_detail_rows, out_dir)
    
    print("Generating confusion matrices...")
    _plot_confusion_matrices("cnn", cnn_cv_rows, cnn_ho_rows, out_dir)
    _plot_confusion_matrices("ae", ae_cv_rows, ae_ho_rows, out_dir)
    
    print("Generating ratio distributions...")
    _plot_ratio_distributions("cnn", cnn_detail_rows, out_dir)
    _plot_ratio_distributions("ae", ae_detail_rows, out_dir)
    
    print("Generating error by burden plots...")
    _plot_error_by_burden("cnn", cnn_detail_rows, out_dir)
    _plot_error_by_burden("ae", ae_detail_rows, out_dir)
    
    print("Writing text summary...")
    _write_summary_text(
        cnn_cv_rows,
        ae_cv_rows,
        cnn_ho_rows,
        ae_ho_rows,
        out_dir / "patient_metrics_summary.txt",
    )

    print(f"Saved patient-level plots to {out_dir}")
    for p in sorted(out_dir.glob("*")):
        print(f"- {p}")


if __name__ == "__main__":
    main()

