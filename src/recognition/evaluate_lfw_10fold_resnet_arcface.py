"""Evaluate a project-trained ResNet ArcFace checkpoint on LFW 6000 pairs/10-fold."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


@dataclass
class PairRecord:
    fold: int
    path1: str
    path2: str
    same: bool


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-5)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNetEmbedding(nn.Module):
    def __init__(self, layers: list[int], embedding_dim: int):
        super().__init__()
        self.in_channels = 64
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.PReLU(64),
        )
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512, eps=1e-5),
            nn.Dropout(p=0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_dim),
            nn.BatchNorm1d(embedding_dim, eps=1e-5),
        )

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [IBasicBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.input_layer(images)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        embeddings = self.output_layer(features)
        return F.normalize(embeddings, p=2, dim=1)


class ConvBNPReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNPReLU(channels, channels, kernel_size=3, stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DepthwiseDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNPReLU(in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileFaceNetEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNPReLU(3, 64, kernel_size=3, stride=2),
            ConvBNPReLU(64, 64, kernel_size=3, stride=1, groups=64),
            DepthwiseResidual(64),
            DepthwiseResidual(64),
            DepthwiseDownsample(64, 128),
            DepthwiseResidual(128),
            DepthwiseResidual(128),
            DepthwiseResidual(128),
            DepthwiseResidual(128),
            DepthwiseDownsample(128, 128),
            DepthwiseResidual(128),
            DepthwiseResidual(128),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.PReLU(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.features(images)
        return F.normalize(embeddings, p=2, dim=1)

class ResNetEmbedding(nn.Module):
    def __init__(self, backbone_name: str, embedding_dim: int):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "mobilefacenet":
            self.model = MobileFaceNetEmbedding(embedding_dim)
        elif backbone_name == "iresnet50":
            self.model = IResNetEmbedding([3, 4, 14, 3], embedding_dim)
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.model = nn.Sequential(backbone, nn.Linear(in_features, embedding_dim), nn.BatchNorm1d(embedding_dim))
        elif backbone_name == "resnet18":
            backbone = models.resnet18(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.model = nn.Sequential(backbone, nn.Linear(in_features, embedding_dim), nn.BatchNorm1d(embedding_dim))
        else:
            raise ValueError(backbone_name)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(images)
        return F.normalize(embeddings, p=2, dim=1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lfw-home", type=Path, default=Path("data/raw/sklearn/lfw_home"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-thresholds", type=int, default=1000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-report", type=Path, required=True)
    return parser.parse_args()


def lfw_image_path(image_dir: Path, person: str, index: str) -> Path:
    return image_dir / person / f"{person}_{int(index):04d}.jpg"


def parse_pairs(pairs_path: Path, image_dir: Path) -> list[PairRecord]:
    lines = [line.strip() for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = lines[0].split()
    if len(header) == 2:
        fold_size = int(header[1]) * 2
        expected = int(header[0]) * fold_size
        pair_lines = lines[1:]
    else:
        fold_size = 600
        expected = len(lines)
        pair_lines = lines
    records: list[PairRecord] = []
    for i, line in enumerate(pair_lines):
        parts = line.split()
        fold = i // fold_size
        if len(parts) == 3:
            person, idx1, idx2 = parts
            records.append(PairRecord(fold, str(lfw_image_path(image_dir, person, idx1)), str(lfw_image_path(image_dir, person, idx2)), True))
        elif len(parts) == 4:
            person1, idx1, person2, idx2 = parts
            records.append(PairRecord(fold, str(lfw_image_path(image_dir, person1, idx1)), str(lfw_image_path(image_dir, person2, idx2)), False))
        else:
            raise ValueError(f"Bad pair line: {line}")
    if len(records) != expected:
        raise ValueError(f"Expected {expected} pairs, got {len(records)}")
    return records


def preprocess(path: Path, image_size: int) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.5, 0.5, 0.5], dtype=np.float32)) / np.array([0.5, 0.5, 0.5], dtype=np.float32)
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


def load_model(checkpoint_path: Path, device: str) -> tuple[ResNetEmbedding, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ResNetEmbedding(str(checkpoint.get("backbone", "resnet50")), int(checkpoint.get("embedding_dim", 256))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def extract_embeddings(model: nn.Module, paths: list[Path], image_size: int, batch_size: int, device: str) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            batch = torch.stack([preprocess(path, image_size) for path in batch_paths]).to(device)
            values = model(batch).detach().cpu().numpy().astype(np.float32)
            values /= np.linalg.norm(values, axis=1, keepdims=True) + 1e-12
            for path, value in zip(batch_paths, values):
                embeddings[str(path)] = value
            done = min(start + batch_size, len(paths))
            if done == len(paths) or done % (batch_size * 10) == 0:
                print(f"embedded {done}/{len(paths)} images, elapsed={time.perf_counter() - started:.1f}s")
    return embeddings


def choose_threshold(scores: np.ndarray, labels: np.ndarray, num_thresholds: int) -> tuple[float, float]:
    thresholds = np.linspace(float(scores.min()), float(scores.max()), num_thresholds)
    best_threshold = float(thresholds[0])
    best_accuracy = -1.0
    for threshold in thresholds:
        accuracy = float(((scores >= threshold) == labels).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def evaluate(records: list[PairRecord], embeddings: dict[str, np.ndarray], num_thresholds: int) -> dict:
    scores = np.array([float(np.dot(embeddings[r.path1], embeddings[r.path2])) for r in records], dtype=np.float32)
    labels = np.array([r.same for r in records], dtype=bool)
    folds = np.array([r.fold for r in records], dtype=np.int32)
    fold_results = []
    total_correct = 0
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        test_mask = folds == fold
        threshold, train_acc = choose_threshold(scores[train_mask], labels[train_mask], num_thresholds)
        preds = scores[test_mask] >= threshold
        correct = int((preds == labels[test_mask]).sum())
        total = int(test_mask.sum())
        total_correct += correct
        fold_results.append({"fold": fold + 1, "threshold": threshold, "train_accuracy": train_acc, "test_accuracy": correct / total, "correct": correct, "total": total})
    fold_acc = np.array([row["test_accuracy"] for row in fold_results], dtype=np.float64)
    return {
        "protocol": "LFW pairs.txt 6000 pairs / 10-fold",
        "mean_accuracy": float(fold_acc.mean()),
        "std_accuracy": float(fold_acc.std(ddof=1)),
        "total_correct": total_correct,
        "num_pairs": len(records),
        "fold_results": fold_results,
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model, checkpoint = load_model(args.checkpoint, device)
    image_dir = args.lfw_home / "lfw_funneled"
    records = parse_pairs(args.lfw_home / "pairs.txt", image_dir)
    unique_paths = sorted({Path(r.path1) for r in records} | {Path(r.path2) for r in records})
    embeddings = extract_embeddings(model, unique_paths, int(checkpoint.get("image_size", 112)), args.batch_size, device)
    result = evaluate(records, embeddings, args.num_thresholds)
    result.update({
        "checkpoint": str(args.checkpoint),
        "device": device,
        "backbone": checkpoint.get("backbone"),
        "embedding_dim": checkpoint.get("embedding_dim"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_best_val_acc": checkpoint.get("best_val_acc"),
        "dataset": checkpoint.get("dataset"),
    })
    lines = [
        "LFW 10-Fold Verification - ResNet ArcFace Checkpoint",
        "=" * 64,
        f"Checkpoint: {args.checkpoint}",
        f"Dataset: {result['dataset']}",
        f"Device: {device}",
        f"Backbone: {result['backbone']}",
        f"Embedding dim: {result['embedding_dim']}",
        f"Checkpoint best epoch: {result['checkpoint_best_epoch']}",
        f"Checkpoint best val acc: {result['checkpoint_best_val_acc']}",
        f"Protocol: {result['protocol']}",
        f"Mean accuracy: {result['mean_accuracy']:.4f}",
        f"Std accuracy: {result['std_accuracy']:.4f}",
        f"Total correct: {result['total_correct']}/{result['num_pairs']}",
        "",
        "Fold results:",
    ]
    for row in result["fold_results"]:
        lines.append("  fold {fold:02d}: acc={test_accuracy:.4f}, train_acc={train_accuracy:.4f}, threshold={threshold:.4f}, correct={correct}/{total}".format(**row))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.json_report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()


