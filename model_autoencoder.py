from __future__ import annotations

"""
Autoencoder for anomaly detection on HelicoDataSet (self-supervised).

Core idea:
  - Train only on "healthy" (majority class) patches.
  - Anomalies (H. pylori) are detected via high reconstruction error.
  - Threshold is chosen from ROC curve: point closest to (FPR=0, TPR=1).

Architecture:
  - Encoder: CNN + pooling → compact bottleneck
  - Decoder: upsampling → reconstructed image
  - Loss: reconstruction error (MSE)

Uses same data as model.py:
  - HelicoDataSet/CoordAnnotatedAllPatches*.csv, Presence: -1=healthy, 1=unhealthy; 0=ignored
  - Images under HelicoDataSet/CrossValidation/Annotated/{Pat_ID}_{Section_ID}/...

Usage:
  python model_autoencoder.py
  python model_autoencoder.py --validate-only --checkpoint autoencoder_model.pth
  python model_autoencoder.py --eval-image path/to/patch.png --checkpoint autoencoder_model.pth
  python model_autoencoder.py --encode-image path/to/patch.png --encode-output latent.npy --checkpoint autoencoder_model.pth
  python model_autoencoder.py --decode-latent latent.npy --decode-output reconstructed.png --checkpoint autoencoder_model.pth
"""

from pathlib import Path
from typing import Tuple, Optional, List
import argparse
import json

import csv
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
HELICO_ROOT = PROJECT_ROOT / "HelicoDataSet"
ANNOTATED_ROOT = HELICO_ROOT / "CrossValidation" / "Annotated"
CHECKPOINT_PATH = PROJECT_ROOT / "autoencoder_model.pth"
THRESHOLD_PATH = PROJECT_ROOT / "autoencoder_threshold.txt"
ONNX_EXPORT_PATH = PROJECT_ROOT / "autoencoder_model.onnx"

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
                    continue  # skip unknown patches

                # 0 = healthy, 1 = unhealthy
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
                f"No samples found from {csv_path} under {images_root}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class Encoder(nn.Module):
    """Deeper CNN: 256→128→64→32→16→8, channels 64→128→256→512→512. Bottleneck 8x8x512."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # 256→128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # 128→64
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # 64→32
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # 32→16
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # 16→8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Decoder(nn.Module):
    """Mirror of Encoder: 8→16→32→64→128→256, 512→512→256→128→64→3."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),  # 8→16
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),   # 16→32
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 32→64
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 64→128
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),     # 128→256
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class PatchAutoencoder(nn.Module):
    """Encoder → Bottleneck (implicit) → Decoder. Reconstructs input; high error = anomaly."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def reconstruction_error(model: nn.Module, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """MSE between input and reconstruction (per sample if reduction='none')."""
    x_hat = model(x)
    return nn.functional.mse_loss(x, x_hat, reduction=reduction)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for images, _ in loader:
        images = images.to(device)
        optimizer.zero_grad()
        loss = reconstruction_error(model, images, reduction="mean")
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
    return running_loss / total


def reconstruction_error_per_sample(
    model: nn.Module,
    x: torch.Tensor,
    aggregation: str = "mean",
) -> torch.Tensor:
    """
    Per-sample reconstruction error (shape (B,)).
    - aggregation "mean": mean MSE over pixels (default; dilutes local anomalies).
    - aggregation "max_local": mean MSE in each grid cell, then max over cells;
      small anomalous regions drive the score (better for focal H. pylori).
    """
    x_hat = model(x)
    err = (x - x_hat).pow(2).mean(dim=1, keepdim=True)  # (B, 1, H, W)
    if aggregation == "mean":
        return err.view(x.size(0), -1).mean(dim=1)
    if aggregation == "max_local":
        # Grid 8x8 → 32x32 cells of 8x8 px; max over cell means
        B, _, H, W = err.shape
        cell = 8
        nc_h, nc_w = H // cell, W // cell
        err = err.view(B, 1, nc_h, cell, nc_w, cell).mean(dim=(3, 5))  # (B, 1, nc_h, nc_w)
        return err.view(B, -1).max(dim=1)[0]
    raise ValueError(f"aggregation must be 'mean' or 'max_local', got {aggregation!r}")


def compute_reconstruction_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    aggregation: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (errors per sample, labels 0=healthy 1=H.pylori). aggregation: 'mean' | 'max_local'."""
    model.eval()
    errors_list: List[float] = []
    labels_list: List[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            err = reconstruction_error_per_sample(model, images, aggregation=aggregation)
            errors_list.extend(err.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())
    return np.array(errors_list), np.array(labels_list)


def roc_threshold_optimal(
    errors: np.ndarray,
    labels: np.ndarray,
    positive_class: int = 1,
) -> Tuple[float, float, float]:
    """
    Find threshold that is closest to (FPR=0, TPR=1).
    Returns (best_threshold, best_tpr, best_fpr).
    Labels: 0 = healthy (negative), 1 = H. pylori (positive).
    """
    # Sort by error descending: higher error → predicted positive
    order = np.argsort(-errors)
    sorted_errors = errors[order]
    sorted_labels = (labels[order] == positive_class).astype(np.float64)

    n_pos = sorted_labels.sum()
    n_neg = len(sorted_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float(np.median(errors)), 0.0, 0.0

    tpr = np.cumsum(sorted_labels) / n_pos
    fpr = np.cumsum(1.0 - sorted_labels) / n_neg

    # Unique thresholds (midpoints between consecutive errors)
    thresh = np.concatenate([[sorted_errors[0] + 1], sorted_errors, [sorted_errors[-1] - 1]])
    thresh = (thresh[:-1] + thresh[1:]) / 2

    # Distance to (0, 1): minimize (FPR)^2 + (1 - TPR)^2
    dist = fpr ** 2 + (1 - tpr) ** 2
    idx = np.argmin(dist)
    return float(thresh[idx]), float(tpr[idx]), float(fpr[idx])


def split_dataset_by_patient(
    dataset: HelicoPatchDataset,
    val_ratio: float = 0.2,
    seed: int = 1,
) -> Tuple[Subset, Subset]:
    """Patient-level split; same as model.py."""
    unique_patients = sorted(set(dataset.patient_ids))
    if len(unique_patients) < 2:
        raise RuntimeError("Need at least 2 distinct patients for train/val split.")

    rng = random.Random(seed)
    rng.shuffle(unique_patients)
    val_patient_count = max(1, int(len(unique_patients) * val_ratio))
    if val_patient_count >= len(unique_patients):
        val_patient_count = len(unique_patients) - 1
    val_patients = set(unique_patients[:val_patient_count])
    train_patients = set(unique_patients[val_patient_count:])

    train_indices = [i for i, pat_id in enumerate(dataset.patient_ids) if pat_id in train_patients]
    val_indices = [i for i, pat_id in enumerate(dataset.patient_ids) if pat_id in val_patients]

    if not train_indices or not val_indices:
        raise RuntimeError("Patient-level split produced empty train or validation set.")

    print(
        f"Patient-level split: {len(train_patients)} train / {len(val_patients)} val patients, "
        f"{len(train_indices)} / {len(val_indices)} patches"
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def patient_stratified_kfold_subsets(
    dataset: HelicoPatchDataset,
    n_splits: int,
    seed: int,
) -> list[Tuple[Subset, Subset]]:
    """
    Build patient-level folds. Stratification uses a patient-level binary label:
    1 if the patient has at least one positive patch, else 0.
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


def _binary_metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def subset_healthy_only(dataset: Subset, base_dataset: HelicoPatchDataset) -> Subset:
    """Restrict to samples with label 0 (healthy)."""
    indices = [i for i in dataset.indices if base_dataset.samples[i][1] == 0]
    return Subset(base_dataset, indices)


def predict_image(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
    threshold: float,
    transform: Optional[transforms.Compose] = None,
    aggregation: str = "mean",
) -> Tuple[float, bool]:
    """Returns (reconstruction_error, is_anomaly). aggregation: 'mean' | 'max_local' (must match threshold)."""
    model.eval()
    if transform is None:
        transform = get_transform()
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        err = reconstruction_error_per_sample(model, x, aggregation=aggregation).item()
    return err, err > threshold


def encode_image_to_latent(
    model: PatchAutoencoder,
    image_path: Path,
    device: torch.device,
    transform: Optional[transforms.Compose] = None,
) -> np.ndarray:
    """Encode one image to latent tensor and return it as numpy array."""
    model.eval()
    if transform is None:
        transform = get_transform()
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model.encode(x)
    return z.squeeze(0).detach().cpu().numpy()


def decode_latent_to_image(
    model: PatchAutoencoder,
    latent: np.ndarray,
    device: torch.device,
) -> Image.Image:
    """Decode latent tensor to a PIL image."""
    model.eval()
    z = torch.from_numpy(latent).float()
    if z.ndim == 3:
        z = z.unsqueeze(0)
    if z.ndim != 4:
        raise ValueError(
            f"Expected latent shape (C,H,W) or (B,C,H,W), got {tuple(z.shape)}"
        )
    z = z.to(device)
    with torch.no_grad():
        x_hat = model.decoder(z)[0].detach().cpu()  # (3,H,W), range approx [-1,1]

    x_hat = ((x_hat + 1.0) / 2.0).clamp(0.0, 1.0)
    arr = (x_hat.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)




def export_onnx(model: nn.Module, output_path: Path, device: torch.device) -> None:
    """Export autoencoder to ONNX with fixed spatial size and dynamic batch."""
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes={"input": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autoencoder anomaly detection: train on healthy only, detect H. pylori by reconstruction error."
    )
    parser.add_argument("--eval-image", type=str, help="Path to image to evaluate (no training).")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path. Default: autoencoder_model.pth")
    parser.add_argument("--validate-only", action="store_true", help="Evaluate checkpoint and report threshold/ROC.")
    parser.add_argument("--encode-image", type=str, help="Path to image to encode into latent (.npy).")
    parser.add_argument("--encode-output", type=str, default="encoded_latent.npy", help="Output path for encoded latent (.npy).")
    parser.add_argument("--decode-latent", type=str, help="Path to latent .npy file to decode into image.")
    parser.add_argument("--decode-output", type=str, default="decoded_image.png", help="Output path for decoded image.")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs (healthy-only). Default: 35.")
    parser.add_argument(
        "--export-onnx",
        type=str,
        nargs="?",
        const=str(ONNX_EXPORT_PATH),
        help=(
            "Export checkpoint to ONNX and exit. Optional output path; "
            "defaults to autoencoder_model.onnx in project root."
        ),
    )
    parser.add_argument(
        "--error-aggregation",
        type=str,
        choices=("mean", "max_local"),
        default="max_local",
        help="Aggregate pixel error: max_local (default, best for focal H. pylori) or mean.",
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
        default=str(PROJECT_ROOT / "cv" / "autoencoder"),
        help="Directory for k-fold checkpoints/thresholds/metrics when --k-folds > 1.",
    )
    args = parser.parse_args()

    selected_modes = [
        args.validate_only,
        args.eval_image is not None,
        args.encode_image is not None,
        args.decode_latent is not None,
        args.export_onnx is not None,
    ]
    if sum(1 for m in selected_modes if m) > 1:
        raise ValueError(
            "Use only one mode at a time: --validate-only, --eval-image, "
            "--encode-image, --decode-latent, or --export-onnx."
        )
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.k_folds < 1:
        raise ValueError("--k-folds must be >= 1.")
    if args.k_folds > 1 and any(selected_modes):
        raise ValueError(
            "--k-folds > 1 is supported only in training mode (no eval/export/validate flags)."
        )
    if args.k_folds > 1 and args.checkpoint is not None:
        raise ValueError(
            "For k-fold, do not pass --checkpoint. Per-fold checkpoints are written under --cv-output-dir."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_PATH

    # ----- ONNX export -----
    if args.export_onnx is not None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model = PatchAutoencoder().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        onnx_path = Path(args.export_onnx)
        export_onnx(model, onnx_path, device)
        print(f"Exported ONNX model to {onnx_path} (checkpoint: {ckpt_path})")
        return

    # ----- Eval single image -----
    if args.eval_image is not None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model = PatchAutoencoder().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        threshold = 0.0
        aggregation = args.error_aggregation
        if THRESHOLD_PATH.exists():
            lines = THRESHOLD_PATH.read_text().strip().splitlines()
            threshold = float(lines[0])
            if len(lines) > 1:
                aggregation = lines[1].strip() or aggregation
        image_path = Path(args.eval_image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        err, is_anomaly = predict_image(model, image_path, device, threshold, aggregation=aggregation)
        label = "H. pylori (anomaly)" if is_anomaly else "healthy"
        print(f"Image: {image_path}\n  Reconstruction error: {err:.6f}\n  Threshold: {threshold:.6f}\n  Predicted: {label}")
        return

    # ----- Encode single image to latent -----
    if args.encode_image is not None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        image_path = Path(args.encode_image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        out_path = Path(args.encode_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model = PatchAutoencoder().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        latent = encode_image_to_latent(model, image_path, device, transform=get_transform())
        np.save(out_path, latent)
        print(
            f"Encoded image: {image_path}\n"
            f"  Latent shape: {latent.shape}\n"
            f"  Saved latent to: {out_path}"
        )
        return

    # ----- Decode latent to image -----
    if args.decode_latent is not None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        latent_path = Path(args.decode_latent)
        if not latent_path.exists():
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        out_path = Path(args.decode_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model = PatchAutoencoder().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        latent = np.load(latent_path)
        decoded = decode_latent_to_image(model, latent, device)
        decoded.save(out_path)
        print(
            f"Decoded latent: {latent_path}\n"
            f"  Input latent shape: {latent.shape}\n"
            f"  Saved image to: {out_path}"
        )
        return

    # ----- Load data -----
    if not HELICO_ROOT.exists():
        raise FileNotFoundError(f"HelicoDataSet not found at {HELICO_ROOT}")

    csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches_stripped.csv"
    if not csv_path.exists():
        csv_path = HELICO_ROOT / "CoordAnnotatedAllPatches.csv"

    transform = get_transform()
    full_dataset = HelicoPatchDataset(csv_path=csv_path, images_root=ANNOTATED_ROOT, transform=transform)
    train_sub, val_sub = split_dataset_by_patient(full_dataset, val_ratio=0.2, seed=args.seed)

    # ----- Validate-only: load checkpoint, compute ROC, print threshold -----
    if args.validate_only:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False, num_workers=2)
        model = PatchAutoencoder().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        agg = args.error_aggregation
        errors, labels = compute_reconstruction_errors(model, val_loader, device, aggregation=agg)
        threshold, tpr, fpr = roc_threshold_optimal(errors, labels, positive_class=1)
        THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
        THRESHOLD_PATH.write_text(f"{threshold}\n{agg}\n")
        pred = (errors > threshold).astype(np.int64)
        acc = (pred == labels).mean()
        print(
            f"Validation (checkpoint: {ckpt_path}, aggregation: {agg})\n"
            f"  Optimal threshold (closest to FPR=0, TPR=1): {threshold:.6f}\n"
            f"  TPR: {tpr:.4f}, FPR: {fpr:.4f}\n"
            f"  Accuracy at this threshold: {acc:.4f}\n"
            f"  Threshold saved to {THRESHOLD_PATH}"
        )
        return

    def train_single_split(
        train_subset: Subset,
        val_subset: Subset,
        checkpoint_out: Path,
        threshold_out: Path,
    ) -> dict[str, float]:
        train_healthy = subset_healthy_only(train_subset, full_dataset)
        n_healthy = len(train_healthy)
        if n_healthy == 0:
            raise RuntimeError(
                "No healthy training samples in this fold. "
                "Try fewer folds or a different seed."
            )
        print(f"Training on healthy patches only: {n_healthy} samples")

        batch_size = 32
        train_loader = DataLoader(train_healthy, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

        model = PatchAutoencoder().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_val_loss = float("inf")
        agg = args.error_aggregation

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            model.eval()
            val_errors, val_labels = compute_reconstruction_errors(
                model, val_loader, device, aggregation=agg
            )
            val_loss = float(np.mean(val_errors))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_out)
            threshold, tpr, fpr = roc_threshold_optimal(val_errors, val_labels, positive_class=1)
            pred = (val_errors > threshold).astype(np.int64)
            val_acc = float((pred == val_labels).mean())
            print(
                f"Epoch {epoch}/{args.epochs} train_loss: {train_loss:.6f} val_loss: {val_loss:.6f} "
                f"threshold: {threshold:.4f} TPR: {tpr:.3f} FPR: {fpr:.3f} val_acc: {val_acc:.4f}"
            )

        model.load_state_dict(torch.load(checkpoint_out, map_location=device, weights_only=True))
        val_errors, val_labels = compute_reconstruction_errors(
            model, val_loader, device, aggregation=agg
        )
        threshold, tpr, fpr = roc_threshold_optimal(val_errors, val_labels, positive_class=1)
        threshold_out.parent.mkdir(parents=True, exist_ok=True)
        threshold_out.write_text(f"{threshold}\n{agg}\n")
        pred = (val_errors > threshold).astype(np.int64)
        metrics = _binary_metrics_from_preds(val_labels.astype(np.int64), pred)
        metrics["val_loss_best"] = best_val_loss
        metrics["threshold"] = threshold
        metrics["tpr"] = tpr
        metrics["fpr"] = fpr
        print(f"Saved best checkpoint (by val_loss) to {checkpoint_out}")
        print(
            f"Optimal threshold (closest to FPR=0, TPR=1): {threshold:.6f} "
            f"(aggregation: {agg}, saved to {threshold_out})"
        )
        return metrics

    if args.k_folds == 1:
        # ----- Training (single split) -----
        train_single_split(
            train_sub,
            val_sub,
            checkpoint_out=ckpt_path,
            threshold_out=THRESHOLD_PATH,
        )
        return

    # ----- Training (k-fold) -----
    cv_root = Path(args.cv_output_dir)
    cv_root.mkdir(parents=True, exist_ok=True)
    folds = patient_stratified_kfold_subsets(
        full_dataset,
        n_splits=args.k_folds,
        seed=args.seed,
    )
    fold_metrics: list[dict[str, float]] = []
    print(f"Running {args.k_folds}-fold patient-level CV for autoencoder")

    for fold_idx, (fold_train, fold_val) in enumerate(folds, start=1):
        fold_dir = cv_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Fold {fold_idx}/{args.k_folds} ===")
        metrics = train_single_split(
            fold_train,
            fold_val,
            checkpoint_out=fold_dir / "autoencoder_model.pth",
            threshold_out=fold_dir / "autoencoder_threshold.txt",
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
    keys = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "val_loss_best",
        "threshold",
        "tpr",
        "fpr",
    ]
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
