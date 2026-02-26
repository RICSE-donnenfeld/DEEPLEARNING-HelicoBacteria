from __future__ import annotations

"""
Simple CNN to classify HelicoDataSet patches as healthy / unhealthy.

Assumptions:
  - We use `HelicoDataSet/CoordAnnotatedAllPatches*.csv` as patch-level labels.
  - Column `Presence` is -1 for healthy, 1 for unhealthy; 0 is ignored.
  - Images are under:
        HelicoDataSet/CrossValidation/Annotated/{Pat_ID}_{Section_ID}/{Window_ID}.png
    where Window_ID is zero-padded to 5 digits and may have an `_AugX` suffix.

Usage (from project root):
    python train_cnn.py
    python model.py --validate-only --checkpoint cnn_model.pth

This will train a small CNN on the annotated patches and print basic
train/validation accuracy.
"""

from pathlib import Path
from typing import Tuple, Optional
import argparse
import json

import csv
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
HELICO_ROOT = PROJECT_ROOT / "HelicoDataSet"
ANNOTATED_ROOT = HELICO_ROOT / "CrossValidation" / "Annotated"
CHECKPOINT_PATH = PROJECT_ROOT / "cnn_model.pth"
ONNX_EXPORT_PATH = PROJECT_ROOT / "cnn_model.onnx"


class HelicoPatchDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        images_root: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.images_root = images_root
        self.transform = transform
        self.samples: list[Tuple[Path, int]] = []
        self.patient_ids: list[str] = []

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pat_id = row["Pat_ID"]
                section_id = row["Section_ID"]
                window_id = row["Window_ID"]
                presence = int(row["Presence"])
                if presence == 0:
                    continue  # skip ambiguous/unknown patches

                # Map Presence to binary label: 0 = healthy, 1 = unhealthy
                label = 1 if presence == 1 else 0

                subdir = f"{pat_id}_{section_id}"

                if "_" in window_id:
                    base_id, aug = window_id.split("_", 1)
                    filename = f"{int(base_id):05d}_{aug}.png"
                else:
                    filename = f"{int(window_id):05d}.png"

                img_path = images_root / subdir / filename
                if not img_path.exists():
                    continue

                self.samples.append((img_path, label))
                self.patient_ids.append(pat_id)

        if not self.samples:
            raise RuntimeError(
                f"No samples found from {csv_path} under {images_root}. "
                "Check paths and filename conventions."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def predict_image(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
) -> float:
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    # probs[1] is "unhealthy", probs[0] is "healthy"
    return probs[1].item()



def export_onnx(model: nn.Module, output_path: Path, device: torch.device) -> None:
    """Export classifier to ONNX with fixed spatial size and dynamic batch."""
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

# We split by patient to avoid overfitting to a specific patient.
def split_dataset_by_patient(
    dataset: HelicoPatchDataset,
    val_ratio: float = 0.2,
    seed: int = 1,
) -> Tuple[Subset, Subset]:
    """
    Split the dataset at patient-level so that a Pat_ID appears in only one split.
    """
    unique_patients = sorted(set(dataset.patient_ids))
    if len(unique_patients) < 2:
        raise RuntimeError(
            "Need at least 2 distinct patients to create patient-level train/val split."
        )

    rng = random.Random(seed)
    rng.shuffle(unique_patients)

    val_patient_count = max(1, int(len(unique_patients) * val_ratio))
    if val_patient_count >= len(unique_patients):
        val_patient_count = len(unique_patients) - 1

    val_patients = set(unique_patients[:val_patient_count])
    train_patients = set(unique_patients[val_patient_count:])

    train_indices = [
        i for i, pat_id in enumerate(dataset.patient_ids) if pat_id in train_patients
    ]
    val_indices = [
        i for i, pat_id in enumerate(dataset.patient_ids) if pat_id in val_patients
    ]

    if not train_indices or not val_indices:
        raise RuntimeError(
            "Patient-level split produced an empty train or validation set."
        )

    print(
        f"Patient-level split: {len(train_patients)} train patients / "
        f"{len(val_patients)} val patients"
    )
    print(
        f"Patient-level split: {len(train_indices)} train patches / "
        f"{len(val_indices)} val patches"
    )

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def patient_stratified_kfold_subsets(
    dataset: HelicoPatchDataset,
    n_splits: int,
    seed: int,
) -> list[Tuple[Subset, Subset]]:
    """
    Build patient-level folds. Each patient belongs to exactly one validation fold.
    We stratify by patient label (patient positive if any patch is positive).
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for k-fold.")

    unique_patients = sorted(set(dataset.patient_ids))
    if len(unique_patients) < n_splits:
        raise ValueError(
            f"Cannot use {n_splits} folds with only {len(unique_patients)} patients."
        )

    patient_to_label: dict[str, int] = {}
    for i, pat_id in enumerate(dataset.patient_ids):
        label = dataset.samples[i][1]
        patient_to_label[pat_id] = max(patient_to_label.get(pat_id, 0), int(label))

    patient_list = unique_patients
    y_patients = [patient_to_label[pat] for pat in patient_list]

    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:
        raise ImportError(
            "k-fold requires scikit-learn. Install it with: pip install scikit-learn"
        ) from exc

    class_counts = {
        0: sum(1 for y in y_patients if y == 0),
        1: sum(1 for y in y_patients if y == 1),
    }
    use_stratified = min(class_counts.values()) >= n_splits
    if not use_stratified:
        print(
            "Warning: Not enough patients in one class for stratified folds. "
            "Falling back to non-stratified KFold."
        )
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(patient_list)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(patient_list, y_patients)

    folds: list[Tuple[Subset, Subset]] = []
    for train_patient_idx, val_patient_idx in split_iter:
        train_patients = {patient_list[i] for i in train_patient_idx}
        val_patients = {patient_list[i] for i in val_patient_idx}

        train_indices = [
            i for i, pat_id in enumerate(dataset.patient_ids) if pat_id in train_patients
        ]
        val_indices = [
            i for i, pat_id in enumerate(dataset.patient_ids) if pat_id in val_patients
        ]
        if not train_indices or not val_indices:
            raise RuntimeError("k-fold split produced an empty train or validation set.")
        folds.append((Subset(dataset, train_indices), Subset(dataset, val_indices)))

    return folds


def _binary_metrics_from_preds(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN on HelicoDataSet or evaluate a single image with a saved model."
    )
    parser.add_argument(
        "--eval-image",
        type=str,
        help="Path to an image to evaluate using a saved model (skips training).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a model checkpoint (.pth). Defaults to cnn_model.pth in the project root.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Evaluate a checkpoint on the validation split and exit (no training).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (training mode only). Default: 5.",
    )
    parser.add_argument(
        "--export-onnx",
        type=str,
        nargs="?",
        const=str(ONNX_EXPORT_PATH),
        help=(
            "Export checkpoint to ONNX and exit. Optional output path; "
            "defaults to cnn_model.onnx in project root."
        ),
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=1,
        help="Number of patient-level folds for cross-validation (training mode only). Default: 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split or k-fold shuffling. Default: 42.",
    )
    parser.add_argument(
        "--cv-output-dir",
        type=str,
        default=str(PROJECT_ROOT / "cv" / "cnn"),
        help="Directory for k-fold checkpoints/metrics when --k-folds > 1.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.k_folds < 1:
        raise ValueError("--k-folds must be >= 1")
    if args.validate_only and args.eval_image is not None:
        raise ValueError("Use either --validate-only or --eval-image, not both.")
    if args.k_folds > 1 and (
        args.validate_only or args.eval_image is not None or args.export_onnx is not None
    ):
        raise ValueError(
            "--k-folds > 1 is supported only in training mode (no eval/export flags)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.export_onnx is not None:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_PATH
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path}. Train first or specify --checkpoint."
            )
        model = SimpleCNN(num_classes=2).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        onnx_path = Path(args.export_onnx)
        export_onnx(model, onnx_path, device)
        print(f"Exported ONNX model to {onnx_path} (checkpoint: {ckpt_path})")
        return

    # Evaluation-only mode using a saved checkpoint.
    if args.eval_image is not None:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_PATH
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path}. Train first or specify --checkpoint."
            )

        model = SimpleCNN(num_classes=2).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        image_path = Path(args.eval_image)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found at {image_path}")

        prob_unhealthy = predict_image(model, image_path, device)
        label = "unhealthy" if prob_unhealthy >= 0.5 else "healthy"
        print(
            f"Image: {image_path}\n"
            f"  Prob(unhealthy): {prob_unhealthy:.4f}\n"
            f"  Predicted label: {label}"
        )
        return

    # Training mode (original behavior)
    if not HELICO_ROOT.exists():
        raise FileNotFoundError(f"HelicoDataSet folder not found at {HELICO_ROOT}")

    # Prefer stripped CSV if you ran filter_labels.py, fall back otherwise.
    csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches_stripped.csv"
    if not csv_path.exists():
        csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches.csv"

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    full_dataset = HelicoPatchDataset(
        csv_path=csv_path, images_root=ANNOTATED_ROOT, transform=transform
    )

    # Patient-level train/validation split for more reliable evaluation.
    train_dataset, val_dataset = split_dataset_by_patient(
        full_dataset, val_ratio=0.2, seed=args.seed
    )

    def train_single_split(
        train_sub: Subset,
        val_sub: Subset,
        checkpoint_path: Path,
        best_model_path: Path,
    ) -> dict[str, float]:
        batch_size = 32
        train_loader = DataLoader(
            train_sub, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_sub, batch_size=batch_size, shuffle=False, num_workers=2
        )

        model = SimpleCNN(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_val_loss = float("inf")

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

            print(
                f"Epoch {epoch}/{args.epochs} "
                f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
                f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
            )

        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")

        # Evaluate best checkpoint for stable fold metrics.
        best_model = SimpleCNN(num_classes=2).to(device)
        best_model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )
        best_model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = best_model(images)
                preds = outputs.argmax(dim=1).cpu().tolist()
                y_pred.extend(int(p) for p in preds)
                y_true.extend(int(y) for y in labels.tolist())
        metrics = _binary_metrics_from_preds(y_true, y_pred)
        metrics["val_loss_best"] = best_val_loss
        return metrics

    if args.k_folds == 1:
        batch_size = 32
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        # Validation-only mode on the same patient-level split used during training.
        if args.validate_only:
            ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_PATH
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found at {ckpt_path}. Train first or specify --checkpoint."
                )

            model = SimpleCNN(num_classes=2).to(device)
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)

            criterion = nn.CrossEntropyLoss()
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(
                f"Validation only - val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} "
                f"(checkpoint: {ckpt_path})"
            )
            return

        train_single_split(
            train_dataset,
            val_dataset,
            checkpoint_path=CHECKPOINT_PATH,
            best_model_path=PROJECT_ROOT / "best_model.pth",
        )
        return

    # k-fold training mode
    cv_root = Path(args.cv_output_dir)
    cv_root.mkdir(parents=True, exist_ok=True)
    folds = patient_stratified_kfold_subsets(
        full_dataset,
        n_splits=args.k_folds,
        seed=args.seed,
    )
    fold_metrics: list[dict[str, float]] = []
    print(f"Running {args.k_folds}-fold patient-level CV for CNN")

    for fold_idx, (fold_train, fold_val) in enumerate(folds, start=1):
        fold_dir = cv_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Fold {fold_idx}/{args.k_folds} ===")
        metrics = train_single_split(
            fold_train,
            fold_val,
            checkpoint_path=fold_dir / "cnn_model.pth",
            best_model_path=fold_dir / "best_model.pth",
        )
        metrics["fold"] = float(fold_idx)
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        fold_metrics.append(metrics)
        print(
            "Fold metrics - "
            f"acc: {metrics['accuracy']:.4f}, prec: {metrics['precision']:.4f}, "
            f"recall: {metrics['recall']:.4f}, spec: {metrics['specificity']:.4f}, "
            f"f1: {metrics['f1']:.4f}"
        )

    summary: dict[str, dict[str, float] | int] = {
        "k_folds": args.k_folds,
        "seed": args.seed,
        "metrics": {},
    }
    keys = ["accuracy", "precision", "recall", "specificity", "f1", "val_loss_best"]
    for key in keys:
        values = [m[key] for m in fold_metrics]
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        summary["metrics"][key] = {
            "mean": mean,
            "std": var ** 0.5,
        }
    summary_path = cv_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved CV summary to {summary_path}")
    for key in keys:
        row = summary["metrics"][key]
        print(f"{key}: {row['mean']:.4f} +- {row['std']:.4f}")


if __name__ == "__main__":
    main()


