#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore[import-untyped]

from model_autoencoder import (
    ANNOTATED_ROOT,
    CHECKPOINT_PATH as AE_CHECKPOINT_PATH,
    HELICO_ROOT,
    HelicoPatchDataset as AEHelicoPatchDataset,
    PatchAutoencoder,
    THRESHOLD_PATH as AE_THRESHOLD_PATH,
    get_transform,
    reconstruction_error_per_sample,
)
from model_classifier import CHECKPOINT_PATH as CNN_CHECKPOINT_PATH, SimpleCNN


PROJECT_ROOT = Path(__file__).resolve().parent
PATIENT_DIAGNOSIS_CSV = HELICO_ROOT / "PatientDiagnosis.csv"
CV_CROPPED_ROOT = HELICO_ROOT / "CrossValidation" / "Cropped"
HOLDOUT_ROOT = HELICO_ROOT / "HoldOut"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "patient_level"
CV_ROOT_DEFAULT = PROJECT_ROOT / "cv"


@dataclass
class PatientCase:
    patient_id: str
    label: int
    density_raw: str
    patch_paths: list[Path]


class PatchPathDataset(Dataset):
    def __init__(self, patch_paths: list[Path], transform: transforms.Compose):
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.patch_paths[idx]).convert("RGB")
        return self.transform(image)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def _log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _patient_threshold_best_f1(ratios: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(ratios)
    if len(candidates) == 0:
        return 0.5
    # Include extremes so all-negative/all-positive predictions are considered.
    eps = 1e-8
    candidates = np.concatenate(([candidates.min() - eps], candidates, [candidates.max() + eps]))
    best_thr = float(candidates[0])
    best_f1 = -1.0
    for thr in candidates:
        pred = (ratios > thr).astype(np.int64)
        f1 = _binary_metrics(labels, pred)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def _read_patient_labels(csv_path: Path) -> dict[str, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Patient diagnosis file not found: {csv_path}")
    mapping: dict[str, int] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row["CODI"].strip()
            density = row["DENSITAT"].strip().upper()
            label = 0 if density == "NEGATIVA" else 1
            mapping[patient_id] = label
    if not mapping:
        raise RuntimeError(f"No labels found in {csv_path}")
    return mapping


def _read_patient_densities(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Patient diagnosis file not found: {csv_path}")
    mapping: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row["CODI"].strip()
            density = row["DENSITAT"].strip().upper()
            mapping[patient_id] = density
    if not mapping:
        raise RuntimeError(f"No density labels found in {csv_path}")
    return mapping


def _annotated_patient_folds_from_training_setup(
    n_splits: int,
    seed: int,
) -> list[tuple[set[str], set[str]]]:
    """
    Rebuild train/val patient fold assignments used by training scripts,
    based on annotated patch dataset (patient-level stratification).
    """
    csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches_stripped.csv"
    if not csv_path.exists():
        csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches.csv"
    ds = AEHelicoPatchDataset(csv_path=csv_path, images_root=ANNOTATED_ROOT, transform=get_transform())

    unique_patients = sorted(set(ds.patient_ids))
    patient_to_label: dict[str, int] = {}
    for i, pat_id in enumerate(ds.patient_ids):
        lbl = int(ds.samples[i][1])
        patient_to_label[pat_id] = max(patient_to_label.get(pat_id, 0), lbl)
    y = np.array([patient_to_label[p] for p in unique_patients], dtype=np.int64)

    if len(unique_patients) < n_splits:
        raise ValueError(f"Requested {n_splits} folds but only {len(unique_patients)} annotated patients.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[set[str], set[str]]] = []
    for tr_idx, va_idx in skf.split(unique_patients, y):
        tr_pat = {unique_patients[i] for i in tr_idx}
        va_pat = {unique_patients[i] for i in va_idx}
        folds.append((tr_pat, va_pat))
    return folds


def _ae_threshold_and_agg_from_file(path: Path, fallback_agg: str) -> tuple[float, str]:
    lines = path.read_text().strip().splitlines()
    thr = float(lines[0])
    agg = fallback_agg
    if len(lines) > 1 and lines[1].strip():
        agg = lines[1].strip()
    return thr, agg


def _collect_patient_cases(
    cropped_root: Path,
    patient_labels: dict[str, int],
    patient_densities: dict[str, str],
    max_patches_per_patient: int | None = None,
) -> list[PatientCase]:
    if not cropped_root.exists():
        raise FileNotFoundError(f"Cropped root not found: {cropped_root}")

    by_patient: dict[str, list[Path]] = {}
    for folder in sorted(cropped_root.iterdir()):
        if not folder.is_dir():
            continue
        patient_id = folder.name.split("_")[0]
        patch_paths = sorted(folder.glob("*.png"))
        if not patch_paths:
            continue
        if max_patches_per_patient is not None:
            patch_paths = patch_paths[:max_patches_per_patient]
        by_patient.setdefault(patient_id, []).extend(patch_paths)

    cases: list[PatientCase] = []
    for patient_id, patch_paths in sorted(by_patient.items()):
        if patient_id not in patient_labels:
            continue
        if not patch_paths:
            continue
        cases.append(
            PatientCase(
                patient_id=patient_id,
                label=patient_labels[patient_id],
                density_raw=patient_densities.get(patient_id, "UNKNOWN"),
                patch_paths=patch_paths,
            )
        )
    if not cases:
        raise RuntimeError(f"No patient cases found in {cropped_root}")
    return cases


def _predict_patch_scores_cnn(
    model: SimpleCNN,
    patch_paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    ds = PatchPathDataset(patch_paths, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(probs.cpu().tolist())
    return np.array(scores, dtype=np.float32)


def _predict_patch_scores_ae(
    model: PatchAutoencoder,
    patch_paths: list[Path],
    device: torch.device,
    aggregation: str,
    batch_size: int,
) -> np.ndarray:
    ds = PatchPathDataset(patch_paths, transform=get_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            err = reconstruction_error_per_sample(model, images, aggregation=aggregation)
            scores.extend(err.cpu().tolist())
    return np.array(scores, dtype=np.float32)


def _compute_patient_ratios(
    cases: Iterable[PatientCase],
    model_type: str,
    device: torch.device,
    batch_size: int,
    checkpoint_path: Path,
    patch_threshold: float,
    ae_aggregation: str,
    log_every_patients: int = 25,
    log_prefix: str = "inference",
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, list[float]]]:
    cases_list = list(cases)
    labels: list[int] = []
    patient_ids: list[str] = []
    ratios: list[float] = []
    by_patient_scores: dict[str, list[float]] = {}
    _log(
        f"{log_prefix}: scoring {len(cases_list)} patients "
        f"(model={model_type}, patch_threshold={patch_threshold:.6f})"
    )

    if model_type == "cnn":
        cnn_model = SimpleCNN(num_classes=2).to(device)
        cnn_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        for idx, case in enumerate(cases_list, start=1):
            patch_scores = _predict_patch_scores_cnn(
                cnn_model, case.patch_paths, device=device, batch_size=batch_size
            )
            ratio = float((patch_scores > patch_threshold).mean())
            labels.append(case.label)
            patient_ids.append(case.patient_id)
            ratios.append(ratio)
            by_patient_scores[case.patient_id] = patch_scores.tolist()
            if idx % max(1, log_every_patients) == 0 or idx == len(cases_list):
                _log(f"{log_prefix}: processed {idx}/{len(cases_list)} patients")
    else:
        ae_model = PatchAutoencoder().to(device)
        ae_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        for idx, case in enumerate(cases_list, start=1):
            patch_scores = _predict_patch_scores_ae(
                ae_model,
                case.patch_paths,
                device=device,
                aggregation=ae_aggregation,
                batch_size=batch_size,
            )
            ratio = float((patch_scores > patch_threshold).mean())
            labels.append(case.label)
            patient_ids.append(case.patient_id)
            ratios.append(ratio)
            by_patient_scores[case.patient_id] = patch_scores.tolist()
            if idx % max(1, log_every_patients) == 0 or idx == len(cases_list):
                _log(f"{log_prefix}: processed {idx}/{len(cases_list)} patients")

    return (
        np.array(ratios, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        patient_ids,
        by_patient_scores,
    )


def _build_patient_detail_rows(
    *,
    fold_i: int,
    split: str,
    patient_ids: list[str],
    labels: np.ndarray,
    ratios: np.ndarray,
    tau_patient: float,
    patch_threshold: float,
    cases_by_id: dict[str, PatientCase],
) -> list[dict[str, Any]]:
    preds = (ratios > tau_patient).astype(np.int64)
    rows: list[dict[str, Any]] = []
    for idx, patient_id in enumerate(patient_ids):
        case = cases_by_id.get(patient_id)
        density_raw = "UNKNOWN" if case is None else case.density_raw
        rows.append(
            {
                "fold": float(fold_i),
                "split": split,
                "patient_id": patient_id,
                "label": int(labels[idx]),
                "pred": int(preds[idx]),
                "ratio": float(ratios[idx]),
                "tau_patient": float(tau_patient),
                "patch_threshold": float(patch_threshold),
                "density_raw": density_raw,
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patient-level pipeline: aggregate patch predictions per patient, "
            "select patient threshold by CV, and evaluate HoldOut."
        )
    )
    parser.add_argument("--model", choices=("cnn", "ae"), required=True, help="Patch model type.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to model checkpoint. Defaults to cnn_model.pth or autoencoder_model.pth.",
    )
    parser.add_argument(
        "--patch-threshold",
        type=float,
        help=(
            "Patch-level threshold. CNN default is 0.5. "
            "AE default comes from autoencoder_threshold.txt if available."
        ),
    )
    parser.add_argument(
        "--ae-aggregation",
        choices=("mean", "max_local"),
        default="max_local",
        help="AE error aggregation mode.",
    )
    parser.add_argument("--patient-k-folds", type=int, default=5, help="Patient-level CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="CV seed.")
    parser.add_argument("--batch-size", type=int, default=64, help="Patch inference batch size.")
    parser.add_argument(
        "--max-patches-per-patient",
        type=int,
        help="Optional cap of patches per patient for faster experiments.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory. Default: output/patient_level/<model>",
    )
    parser.add_argument(
        "--use-cv-fold-checkpoints",
        action="store_true",
        help=(
            "Use fold checkpoints from cv/<model>/fold_i to compute fold-consistent patient-level metrics. "
            "Recommended when k-fold checkpoints already exist."
        ),
    )
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=CV_ROOT_DEFAULT,
        help="Root directory containing cv/cnn and cv/autoencoder fold checkpoints.",
    )
    parser.add_argument(
        "--log-every-patients",
        type=int,
        default=25,
        help="Print progress every N patients during scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Starting patient-level pipeline (model={args.model})")
    out_dir = args.out_dir if args.out_dir else (OUTPUT_ROOT / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output directory: {out_dir}")

    patient_labels = _read_patient_labels(PATIENT_DIAGNOSIS_CSV)
    patient_densities = _read_patient_densities(PATIENT_DIAGNOSIS_CSV)
    _log(f"Loaded labels for {len(patient_labels)} patients from {PATIENT_DIAGNOSIS_CSV}")
    cv_cases = _collect_patient_cases(
        CV_CROPPED_ROOT,
        patient_labels=patient_labels,
        patient_densities=patient_densities,
        max_patches_per_patient=args.max_patches_per_patient,
    )
    holdout_cases = _collect_patient_cases(
        HOLDOUT_ROOT,
        patient_labels=patient_labels,
        patient_densities=patient_densities,
        max_patches_per_patient=args.max_patches_per_patient,
    )
    _log(f"Collected cases: CV={len(cv_cases)} patients, HoldOut={len(holdout_cases)} patients")

    fold_rows: list[dict[str, float]] = []
    holdout_fold_rows: list[dict[str, float]] = []
    patient_detail_rows: list[dict[str, Any]] = []

    if args.use_cv_fold_checkpoints:
        folds = _annotated_patient_folds_from_training_setup(
            n_splits=args.patient_k_folds,
            seed=args.seed,
        )
        _log(f"Fold-consistent mode enabled with {len(folds)} folds")
        cv_cases_by_id = {c.patient_id: c for c in cv_cases}

        for fold_i, (train_patients, val_patients) in enumerate(folds, start=1):
            _log(
                f"Fold {fold_i}/{len(folds)}: "
                f"train_patients={len(train_patients)} val_patients={len(val_patients)}"
            )
            if args.model == "cnn":
                fold_dir = args.cv_root / "cnn" / f"fold_{fold_i}"
                checkpoint = fold_dir / "best_model.pth"
                if not checkpoint.exists():
                    checkpoint = fold_dir / "cnn_model.pth"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"No CNN checkpoint found for fold {fold_i} in {fold_dir}")
                patch_threshold = 0.5 if args.patch_threshold is None else float(args.patch_threshold)
                ae_aggregation = "n/a"
                _log(
                    f"Fold {fold_i}: checkpoint={checkpoint.name}, "
                    f"patch_threshold={patch_threshold:.6f}"
                )
            else:
                fold_dir = args.cv_root / "autoencoder" / f"fold_{fold_i}"
                checkpoint = fold_dir / "autoencoder_model.pth"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"No AE checkpoint found for fold {fold_i} in {fold_dir}")
                threshold_file = fold_dir / "autoencoder_threshold.txt"
                if args.patch_threshold is not None:
                    patch_threshold = float(args.patch_threshold)
                    ae_aggregation = args.ae_aggregation
                elif threshold_file.exists():
                    patch_threshold, ae_aggregation = _ae_threshold_and_agg_from_file(
                        threshold_file, fallback_agg=args.ae_aggregation
                    )
                else:
                    raise FileNotFoundError(
                        f"No AE threshold provided and missing {threshold_file}. "
                        "Use --patch-threshold."
                    )
                _log(
                    f"Fold {fold_i}: checkpoint={checkpoint.name}, "
                    f"patch_threshold={patch_threshold:.6f}, agg={ae_aggregation}"
                )

            train_cases = [cv_cases_by_id[p] for p in sorted(train_patients) if p in cv_cases_by_id]
            val_cases = [cv_cases_by_id[p] for p in sorted(val_patients) if p in cv_cases_by_id]
            if not train_cases or not val_cases:
                raise RuntimeError(
                    f"Fold {fold_i} has empty train or val patient cases after matching Cropped folders."
                )

            train_ratios, train_labels, _, _ = _compute_patient_ratios(
                train_cases,
                model_type=args.model,
                device=device,
                batch_size=args.batch_size,
                checkpoint_path=checkpoint,
                patch_threshold=float(patch_threshold),
                ae_aggregation=ae_aggregation,
                log_every_patients=args.log_every_patients,
                log_prefix=f"fold{fold_i}/train",
            )
            val_ratios, val_labels, val_patient_ids, _ = _compute_patient_ratios(
                val_cases,
                model_type=args.model,
                device=device,
                batch_size=args.batch_size,
                checkpoint_path=checkpoint,
                patch_threshold=float(patch_threshold),
                ae_aggregation=ae_aggregation,
                log_every_patients=args.log_every_patients,
                log_prefix=f"fold{fold_i}/val",
            )
            tau_patient = _patient_threshold_best_f1(train_ratios, train_labels)
            val_pred = (val_ratios > tau_patient).astype(np.int64)
            val_metrics = _binary_metrics(val_labels, val_pred)
            val_metrics["fold"] = float(fold_i)
            val_metrics["tau_patient"] = float(tau_patient)
            val_metrics["patch_threshold"] = float(patch_threshold)
            fold_rows.append(val_metrics)
            _log(
                f"Fold {fold_i}: CV patient metrics "
                f"acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} "
                f"tau_patient={tau_patient:.6f}"
            )

            # HoldOut eval using the same fold model + fold tau_patient.
            holdout_ratios, holdout_labels, holdout_patient_ids, _ = _compute_patient_ratios(
                holdout_cases,
                model_type=args.model,
                device=device,
                batch_size=args.batch_size,
                checkpoint_path=checkpoint,
                patch_threshold=float(patch_threshold),
                ae_aggregation=ae_aggregation,
                log_every_patients=args.log_every_patients,
                log_prefix=f"fold{fold_i}/holdout",
            )
            holdout_pred = (holdout_ratios > tau_patient).astype(np.int64)
            hm = _binary_metrics(holdout_labels, holdout_pred)
            hm["fold"] = float(fold_i)
            hm["tau_patient"] = float(tau_patient)
            hm["patch_threshold"] = float(patch_threshold)
            holdout_fold_rows.append(hm)
            _log(
                f"Fold {fold_i}: HoldOut patient metrics "
                f"acc={hm['accuracy']:.4f} f1={hm['f1']:.4f}"
            )
            val_cases_by_id = {c.patient_id: c for c in val_cases}
            holdout_cases_by_id = {c.patient_id: c for c in holdout_cases}
            patient_detail_rows.extend(
                _build_patient_detail_rows(
                    fold_i=fold_i,
                    split="cv_val",
                    patient_ids=val_patient_ids,
                    labels=val_labels,
                    ratios=val_ratios,
                    tau_patient=tau_patient,
                    patch_threshold=float(patch_threshold),
                    cases_by_id=val_cases_by_id,
                )
            )
            patient_detail_rows.extend(
                _build_patient_detail_rows(
                    fold_i=fold_i,
                    split="holdout",
                    patient_ids=holdout_patient_ids,
                    labels=holdout_labels,
                    ratios=holdout_ratios,
                    tau_patient=tau_patient,
                    patch_threshold=float(patch_threshold),
                    cases_by_id=holdout_cases_by_id,
                )
            )

        fold_metrics_path = out_dir / "patient_cv_folds_from_cv_models.json"
        holdout_folds_path = out_dir / "holdout_metrics_per_fold_from_cv_models.json"
        patient_detail_path = out_dir / "patient_fold_details_from_cv_models.json"
    else:
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = CNN_CHECKPOINT_PATH if args.model == "cnn" else AE_CHECKPOINT_PATH
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        patch_threshold = args.patch_threshold
        ae_aggregation = args.ae_aggregation
        if args.model == "cnn":
            if patch_threshold is None:
                patch_threshold = 0.5
        else:
            if patch_threshold is None and AE_THRESHOLD_PATH.exists():
                patch_threshold, ae_aggregation = _ae_threshold_and_agg_from_file(
                    AE_THRESHOLD_PATH, fallback_agg=args.ae_aggregation
                )
            if patch_threshold is None:
                raise ValueError(
                    "AE patch threshold not provided and autoencoder_threshold.txt not found. "
                    "Use --patch-threshold."
                )

        cv_ratios, cv_labels, cv_patient_ids, _ = _compute_patient_ratios(
            cv_cases,
            model_type=args.model,
            device=device,
            batch_size=args.batch_size,
            checkpoint_path=checkpoint,
            patch_threshold=float(patch_threshold),
            ae_aggregation=ae_aggregation,
            log_every_patients=args.log_every_patients,
            log_prefix="single/cv",
        )
        holdout_ratios, holdout_labels, holdout_patient_ids, _ = _compute_patient_ratios(
            holdout_cases,
            model_type=args.model,
            device=device,
            batch_size=args.batch_size,
            checkpoint_path=checkpoint,
            patch_threshold=float(patch_threshold),
            ae_aggregation=ae_aggregation,
            log_every_patients=args.log_every_patients,
            log_prefix="single/holdout",
        )

        skf = StratifiedKFold(n_splits=args.patient_k_folds, shuffle=True, random_state=args.seed)
        cv_cases_by_id = {c.patient_id: c for c in cv_cases}
        holdout_cases_by_id = {c.patient_id: c for c in holdout_cases}
        for fold_i, (train_idx, val_idx) in enumerate(skf.split(cv_patient_ids, cv_labels), start=1):
            train_ratios = cv_ratios[train_idx]
            train_labels = cv_labels[train_idx]
            val_ratios = cv_ratios[val_idx]
            val_labels = cv_labels[val_idx]
            val_patient_ids = [cv_patient_ids[int(i)] for i in val_idx.tolist()]
            tau_patient = _patient_threshold_best_f1(train_ratios, train_labels)
            val_pred = (val_ratios > tau_patient).astype(np.int64)
            metrics = _binary_metrics(val_labels, val_pred)
            metrics["fold"] = float(fold_i)
            metrics["tau_patient"] = float(tau_patient)
            metrics["patch_threshold"] = float(patch_threshold)
            fold_rows.append(metrics)

            holdout_pred = (holdout_ratios > tau_patient).astype(np.int64)
            hm = _binary_metrics(holdout_labels, holdout_pred)
            hm["fold"] = float(fold_i)
            hm["tau_patient"] = float(tau_patient)
            hm["patch_threshold"] = float(patch_threshold)
            holdout_fold_rows.append(hm)
            _log(
                f"Fold {fold_i}: CV acc={metrics['accuracy']:.4f}, "
                f"HoldOut acc={hm['accuracy']:.4f}"
            )
            patient_detail_rows.extend(
                _build_patient_detail_rows(
                    fold_i=fold_i,
                    split="cv_val",
                    patient_ids=val_patient_ids,
                    labels=val_labels,
                    ratios=val_ratios,
                    tau_patient=tau_patient,
                    patch_threshold=float(patch_threshold),
                    cases_by_id=cv_cases_by_id,
                )
            )
            patient_detail_rows.extend(
                _build_patient_detail_rows(
                    fold_i=fold_i,
                    split="holdout",
                    patient_ids=holdout_patient_ids,
                    labels=holdout_labels,
                    ratios=holdout_ratios,
                    tau_patient=tau_patient,
                    patch_threshold=float(patch_threshold),
                    cases_by_id=holdout_cases_by_id,
                )
            )

        fold_metrics_path = out_dir / "patient_cv_folds.json"
        holdout_folds_path = out_dir / "holdout_metrics_per_fold.json"
        patient_detail_path = out_dir / "patient_fold_details.json"

    fold_metrics_path.write_text(json.dumps(fold_rows, indent=2))
    holdout_folds_path.write_text(json.dumps(holdout_fold_rows, indent=2))
    patient_detail_path.write_text(json.dumps(patient_detail_rows, indent=2))

    keys = ["accuracy", "precision", "recall", "specificity", "f1", "tau_patient", "patch_threshold"]
    summary: dict[str, Any] = {
        "model": args.model,
        "use_cv_fold_checkpoints": bool(args.use_cv_fold_checkpoints),
        "patient_k_folds": args.patient_k_folds,
        "seed": args.seed,
        "cv": {},
        "holdout_per_fold": {},
    }
    for key in keys:
        cv_vals = [float(row[key]) for row in fold_rows]
        ho_vals = [float(row[key]) for row in holdout_fold_rows]
        summary["cv"][key] = {
            "mean": float(np.mean(cv_vals)),
            "std": float(np.std(cv_vals)),
            "median": float(np.median(cv_vals)),
            "q1": float(np.percentile(cv_vals, 25)),
            "q3": float(np.percentile(cv_vals, 75)),
        }
        summary["holdout_per_fold"][key] = {
            "mean": float(np.mean(ho_vals)),
            "std": float(np.std(ho_vals)),
            "median": float(np.median(ho_vals)),
            "q1": float(np.percentile(ho_vals, 25)),
            "q3": float(np.percentile(ho_vals, 75)),
        }

    summary_path = out_dir / "patient_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    _log(f"Model: {args.model}")
    _log(f"Use CV fold checkpoints: {args.use_cv_fold_checkpoints}")
    _log(f"Saved CV fold metrics to: {fold_metrics_path}")
    _log(f"Saved HoldOut-per-fold metrics to: {holdout_folds_path}")
    _log(f"Saved patient detail rows to: {patient_detail_path}")
    _log(f"Saved summary to: {summary_path}")
    _log(
        "CV mean metrics - "
        f"acc: {summary['cv']['accuracy']['mean']:.4f}, "
        f"prec: {summary['cv']['precision']['mean']:.4f}, "
        f"recall: {summary['cv']['recall']['mean']:.4f}, "
        f"spec: {summary['cv']['specificity']['mean']:.4f}, "
        f"f1: {summary['cv']['f1']['mean']:.4f}"
    )


if __name__ == "__main__":
    main()

