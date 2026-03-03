#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

    cnn_cv_rows = _load_json(cnn_cv_path)
    ae_cv_rows = _load_json(ae_cv_path)
    cnn_ho_rows = _load_json(cnn_ho_path)
    ae_ho_rows = _load_json(ae_ho_path)

    _plot_model_fold_metrics("cnn", cnn_cv_rows, cnn_ho_rows, out_dir)
    _plot_model_fold_metrics("autoencoder", ae_cv_rows, ae_ho_rows, out_dir)
    _plot_cross_model_comparison(cnn_cv_rows, ae_cv_rows, out_dir)
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

