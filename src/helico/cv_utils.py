from __future__ import annotations

from typing import Any
import random

from torch.utils.data import Subset


def dataset_split_by_patient(
    dataset: Any,
    val_ratio: float = 0.2,
    seed: int = 1,
) -> tuple[Subset, Subset]:
    """
    Split at patient level so one patient appears in one split only.
    Expects dataset.patient_ids.
    """
    patient_ids = list(dataset.patient_ids)
    unique_patients = sorted(set(patient_ids))
    if len(unique_patients) < 2:
        raise RuntimeError("Need at least 2 distinct patients for train/val split.")

    rng = random.Random(seed)
    rng.shuffle(unique_patients)

    val_patient_count = max(1, int(len(unique_patients) * val_ratio))
    if val_patient_count >= len(unique_patients):
        val_patient_count = len(unique_patients) - 1

    val_patients = set(unique_patients[:val_patient_count])
    train_patients = set(unique_patients[val_patient_count:])

    train_indices = [i for i, pat in enumerate(patient_ids) if pat in train_patients]
    val_indices = [i for i, pat in enumerate(patient_ids) if pat in val_patients]
    if not train_indices or not val_indices:
        raise RuntimeError("Patient-level split produced an empty train or validation set.")

    print(
        f"Patient-level split: {len(train_patients)} train patients / "
        f"{len(val_patients)} val patients"
    )
    print(
        f"Patient-level split: {len(train_indices)} train patches / "
        f"{len(val_indices)} val patches"
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def dataset_patient_stratified_kfold_subsets(
    dataset: Any,
    n_splits: int,
    seed: int,
) -> list[tuple[Subset, Subset]]:
    """
    Build patient-level folds. Stratification uses patient-level binary label:
    patient is positive if any sample label is positive.
    Expects dataset.patient_ids and dataset.samples where sample[1] is binary label.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for k-fold.")

    patient_ids = list(dataset.patient_ids)
    unique_patients = sorted(set(patient_ids))
    if len(unique_patients) < n_splits:
        raise ValueError(
            f"Cannot use {n_splits} folds with only {len(unique_patients)} patients."
        )

    patient_to_label: dict[str, int] = {}
    for i, pat_id in enumerate(patient_ids):
        label = int(dataset.samples[i][1])
        patient_to_label[pat_id] = max(patient_to_label.get(pat_id, 0), label)

    y_patients = [patient_to_label[pat] for pat in unique_patients]

    try:
        from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "k-fold requires scikit-learn. Install it with: pip install scikit-learn"
        ) from exc

    class_counts = {
        0: sum(1 for y in y_patients if y == 0),
        1: sum(1 for y in y_patients if y == 1),
    }
    use_stratified = min(class_counts.values()) >= n_splits
    if use_stratified:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(unique_patients, y_patients)
    else:
        print(
            "Warning: Not enough patients in one class for stratified folds. "
            "Falling back to non-stratified KFold."
        )
        from sklearn.model_selection import KFold  # type: ignore[import-untyped]

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(unique_patients)

    folds: list[tuple[Subset, Subset]] = []
    for train_patient_idx, val_patient_idx in split_iter:
        train_patients = {unique_patients[i] for i in train_patient_idx}
        val_patients = {unique_patients[i] for i in val_patient_idx}
        train_indices = [i for i, pat in enumerate(patient_ids) if pat in train_patients]
        val_indices = [i for i, pat in enumerate(patient_ids) if pat in val_patients]
        if not train_indices or not val_indices:
            raise RuntimeError("k-fold split produced an empty train or validation set.")
        folds.append((Subset(dataset, train_indices), Subset(dataset, val_indices)))
    return folds


def binary_metrics_from_preds(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    total = max(1, tp + tn + fp + fn)

    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }

