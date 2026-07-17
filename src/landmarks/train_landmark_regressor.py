from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

LANDMARK_68_FLIP_ORDER = np.asarray(
    [
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
        27, 28, 29, 30,
        35, 34, 33, 32, 31,
        45, 44, 43, 42, 47, 46,
        39, 38, 37, 36, 41, 40,
        54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
        64, 63, 62, 61, 60, 67, 66, 65,
    ],
    dtype=np.int64,
)


@dataclass
class Sample:
    image_path: Path
    pts_path: Path


def parse_pts(path: Path) -> np.ndarray:
    points = []
    inside = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if text == "{":
            inside = True
            continue
        if text == "}":
            break
        if inside and text:
            x, y = text.split()[:2]
            points.append([float(x), float(y)])
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Invalid pts file: {path}")
    return points


def find_image_for_pts(pts_path: Path) -> Path | None:
    stem = pts_path.with_suffix("")
    for suffix in IMAGE_SUFFIXES:
        candidate = stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def collect_samples(root: Path) -> list[Sample]:
    samples = []
    for pts_path in sorted(root.rglob("*.pts")):
        image_path = find_image_for_pts(pts_path)
        if image_path is not None:
            samples.append(Sample(image_path=image_path, pts_path=pts_path))
    if not samples:
        raise FileNotFoundError(f"No image/.pts pairs found under {root}")
    return samples


def landmark_crop_box(points: np.ndarray, image_width: int, image_height: int, margin: float = 0.25) -> tuple[int, int, int, int]:
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    box_w = max(float(x_max - x_min), 1.0)
    box_h = max(float(y_max - y_min), 1.0)
    side_margin = max(box_w, box_h) * margin
    left = int(np.floor(max(0.0, x_min - side_margin)))
    top = int(np.floor(max(0.0, y_min - side_margin)))
    right = int(np.ceil(min(float(image_width), x_max + side_margin)))
    bottom = int(np.ceil(min(float(image_height), y_max + side_margin)))
    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)
    return left, top, right, bottom


def crop_image_and_points(image: np.ndarray, points: np.ndarray, margin: float = 0.25) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    left, top, right, bottom = landmark_crop_box(points, w, h, margin=margin)
    cropped = image[top:bottom, left:right]
    cropped_points = points.copy()
    cropped_points[:, 0] -= left
    cropped_points[:, 1] -= top
    return cropped, cropped_points, (left, top, right, bottom)


def horizontal_flip(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flipped = np.ascontiguousarray(image[:, ::-1])
    h, w = image.shape[:2]
    flipped_points = points.copy()
    flipped_points[:, 0] = (w - 1) - flipped_points[:, 0]
    if len(flipped_points) == 68:
        flipped_points = flipped_points[LANDMARK_68_FLIP_ORDER]
    return flipped, flipped_points


def rotate_image_and_points(image: np.ndarray, points: np.ndarray, angle_deg: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, scale)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points, ones], axis=1)
    rotated_points = homo @ matrix.T.astype(np.float32)
    return rotated, rotated_points.astype(np.float32)


def jitter_color(image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    adjusted = image.astype(np.float32) * contrast + brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


class PtsLandmarkDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_size: int = 128,
        crop_to_landmarks: bool = True,
        crop_margin: float = 0.25,
        augment: bool = False,
        sample_indices: list[int] | None = None,
    ):
        self.root = root
        self.image_size = image_size
        self.crop_to_landmarks = crop_to_landmarks
        self.crop_margin = crop_margin
        self.augment = augment
        all_samples = collect_samples(root)
        if sample_indices is None:
            self.samples = all_samples
        else:
            self.samples = [all_samples[i] for i in sample_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(sample.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = parse_pts(sample.pts_path)

        if self.augment:
            if random.random() < 0.5:
                image, points = horizontal_flip(image, points)
            if random.random() < 0.8:
                angle = random.uniform(-15.0, 15.0)
                scale = random.uniform(0.92, 1.08)
                image, points = rotate_image_and_points(image, points, angle, scale)

        if self.crop_to_landmarks:
            image, points, _crop_box = crop_image_and_points(image, points, margin=self.crop_margin)

        if self.augment and random.random() < 0.8:
            brightness = random.uniform(-22.0, 22.0)
            contrast = random.uniform(0.82, 1.18)
            image = jitter_color(image, brightness, contrast)

        h, w = image.shape[:2]
        points_norm = points.copy()
        points_norm[:, 0] = np.clip(points_norm[:, 0] / max(w - 1, 1), 0.0, 1.0)
        points_norm[:, 1] = np.clip(points_norm[:, 1] / max(h - 1, 1), 0.0, 1.0)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(points_norm.reshape(-1)), torch.tensor([w, h], dtype=torch.float32)


class SmallLandmarkCNN(nn.Module):
    def __init__(self, num_points: int = 68):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, num_points * 2), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.features(x))


def nme_68(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_pts = pred.view(pred.shape[0], -1, 2)
    tgt_pts = target.view(target.shape[0], -1, 2)
    errors = torch.linalg.norm(pred_pts - tgt_pts, dim=-1).mean(dim=-1)
    if pred_pts.shape[1] >= 46:
        left_eye = tgt_pts[:, 36]
        right_eye = tgt_pts[:, 45]
        norm = torch.linalg.norm(left_eye - right_eye, dim=-1).clamp_min(1e-6)
    else:
        norm = torch.ones_like(errors)
    return (errors / norm).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small coordinate-regression landmark baseline on 300W-style .pts data.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/300W"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/landmark_cnn_300w.pt"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/landmark_300w_training_result.txt"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--crop-to-landmarks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-margin", type=float, default=0.25)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    all_samples = collect_samples(args.root)
    indices = list(range(len(all_samples)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(len(indices) * args.val_ratio)))
    val_indices = sorted(indices[:val_size])
    train_indices = sorted(indices[val_size:])

    train_set = PtsLandmarkDataset(
        args.root,
        image_size=args.image_size,
        crop_to_landmarks=args.crop_to_landmarks,
        crop_margin=args.crop_margin,
        augment=args.augment,
        sample_indices=train_indices,
    )
    val_set = PtsLandmarkDataset(
        args.root,
        image_size=args.image_size,
        crop_to_landmarks=args.crop_to_landmarks,
        crop_margin=args.crop_margin,
        augment=False,
        sample_indices=val_indices,
    )
    first_points = parse_pts(all_samples[0].pts_path).shape[0]
    model = SmallLandmarkCNN(num_points=first_points).to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    history = []
    best_state = None
    best_row = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for images, targets, _sizes in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        val_nmes = []
        with torch.no_grad():
            for images, targets, _sizes in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                val_losses.append(float(criterion(preds, targets).cpu()))
                val_nmes.append(float(nme_68(preds.cpu(), targets.cpu())))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "val_nme": float(np.mean(val_nmes)),
        }
        history.append(row)
        if best_row is None or row["val_nme"] < best_row["val_nme"]:
            best_row = row.copy()
            best_state = copy.deepcopy(model.state_dict())
        print(row)

    if best_state is None or best_row is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "num_points": first_points,
            "image_size": args.image_size,
            "history": history,
            "best_epoch": best_row["epoch"],
            "best_val_nme": best_row["val_nme"],
            "crop_to_landmarks": args.crop_to_landmarks,
            "crop_margin": args.crop_margin,
            "augment": args.augment,
            "train_indices": train_indices,
            "val_indices": val_indices,
        },
        args.output,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "300W Landmark CNN Training Result",
        "=" * 50,
        f"Root: {args.root}",
        f"Samples: {len(all_samples)}",
        f"Train/val: {len(train_set)}/{len(val_set)}",
        f"Points: {first_points}",
        f"Device: {device}",
        f"Crop to landmarks: {args.crop_to_landmarks}",
        f"Crop margin: {args.crop_margin}",
        f"Augment train set: {args.augment}",
        f"Best epoch: {best_row['epoch']}",
        f"Best val NME: {best_row['val_nme']:.4f}",
        f"Checkpoint: {args.output}",
        "",
        "History:",
    ]
    for row in history:
        marker = " *best" if row["epoch"] == best_row["epoch"] else ""
        lines.append(f"- epoch {row['epoch']}: train_loss={row['train_loss']:.6f}, val_loss={row['val_loss']:.6f}, val_nme={row['val_nme']:.4f}{marker}")
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
