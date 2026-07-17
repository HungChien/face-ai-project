from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_LFW_ROOT = Path("data/raw/sklearn/lfw_home/lfw_funneled")


@dataclass
class FaceSample:
    image_path: Path
    label: int
    identity: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet18 + ArcFace smoke baseline on LFW identities.")
    parser.add_argument("--root", type=Path, default=DEFAULT_LFW_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-identities", type=int, default=30)
    parser.add_argument("--min-images", type=int, default=5)
    parser.add_argument("--max-images-per-identity", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/resnet18_arcface_lfw_smoke_best.pt"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/resnet18_arcface_lfw_smoke_result.txt"))
    parser.add_argument("--history-json", type=Path, default=Path("outputs/reports/resnet18_arcface_lfw_smoke_history.json"))
    parser.add_argument("--curve", type=Path, default=Path("outputs/images/resnet18_arcface_lfw_smoke_curves.jpg"))
    return parser.parse_args()


def list_identity_images(root: Path) -> dict[str, list[Path]]:
    identities: dict[str, list[Path]] = {}
    for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(p for p in identity_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            identities[identity_dir.name] = images
    if not identities:
        raise FileNotFoundError(f"No identity image folders found under {root}")
    return identities


def build_samples(args: argparse.Namespace) -> tuple[list[FaceSample], list[FaceSample], list[str], dict[str, int]]:
    rng = random.Random(args.seed)
    identity_images = list_identity_images(args.root)
    eligible = [(name, paths) for name, paths in identity_images.items() if len(paths) >= args.min_images]
    eligible.sort(key=lambda item: (-len(item[1]), item[0]))
    selected = eligible[: args.num_identities]
    if len(selected) < 2:
        raise ValueError("Need at least two identities for ArcFace classification.")

    train_samples: list[FaceSample] = []
    val_samples: list[FaceSample] = []
    identity_names = [name for name, _paths in selected]
    label_map = {name: idx for idx, name in enumerate(identity_names)}
    counts = {}

    for name, paths in selected:
        paths = paths.copy()
        rng.shuffle(paths)
        paths = sorted(paths[: args.max_images_per_identity])
        val_count = max(1, int(round(len(paths) * args.val_ratio)))
        if len(paths) - val_count < 1:
            val_count = len(paths) - 1
        val_paths = paths[:val_count]
        train_paths = paths[val_count:]
        label = label_map[name]
        train_samples.extend(FaceSample(path, label, name) for path in train_paths)
        val_samples.extend(FaceSample(path, label, name) for path in val_paths)
        counts[name] = len(paths)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples, identity_names, counts


class LfwIdentityDataset(Dataset):
    def __init__(self, samples: list[FaceSample], image_size: int, augment: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(sample.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
        if self.augment and random.random() < 0.7:
            image = image.astype(np.float32)
            contrast = random.uniform(0.85, 1.15)
            brightness = random.uniform(-16.0, 16.0)
            image = np.clip(image * contrast + brightness, 0, 255).astype(np.uint8)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.5, 0.5, 0.5], dtype=np.float32)) / np.array([0.5, 0.5, 0.5], dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image), torch.tensor(sample.label, dtype=torch.long)


class ResNet18Embedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 32.0, margin: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = float(np.cos(margin))
        self.sin_m = float(np.sin(margin))
        self.th = float(np.cos(np.pi - margin))
        self.mm = float(np.sin(np.pi - margin) * margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1.0, 1.0)
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().detach().cpu())


def run_epoch(model, head, loader, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    head.train(training)
    losses = []
    accuracies = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(training):
            embeddings = model(images)
            train_logits = head(embeddings, labels)
            loss = F.cross_entropy(train_logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            eval_logits = head(embeddings, None)
        losses.append(float(loss.detach().cpu()))
        accuracies.append(accuracy_from_logits(eval_logits.detach(), labels.detach()))
    return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(accuracies))}


def save_curve(history: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("Classification accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    train_samples, val_samples, identity_names, identity_counts = build_samples(args)
    train_loader = DataLoader(LfwIdentityDataset(train_samples, args.image_size, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(LfwIdentityDataset(val_samples, args.image_size, augment=False), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ResNet18Embedding(args.embedding_dim).to(device)
    head = ArcFaceHead(args.embedding_dim, len(identity_names)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)

    history = []
    best_state = None
    best_row = None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, head, train_loader, device, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, head, val_loader, device, optimizer=None)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
        }
        history.append(row)
        if best_row is None or row["val_acc"] > best_row["val_acc"]:
            best_row = row.copy()
            best_state = {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            }
        print(row)

    elapsed = time.perf_counter() - start
    if best_state is None or best_row is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **best_state,
            "backbone": "resnet18",
            "loss": "ArcFace",
            "embedding_dim": args.embedding_dim,
            "num_classes": len(identity_names),
            "identity_names": identity_names,
            "identity_counts": identity_counts,
            "image_size": args.image_size,
            "best_epoch": best_row["epoch"],
            "best_val_acc": best_row["val_acc"],
            "history": history,
        },
        args.output,
    )
    save_curve(history, args.curve)

    args.history_json.parent.mkdir(parents=True, exist_ok=True)
    args.history_json.write_text(json.dumps({"history": history, "best": best_row}, indent=2), encoding="utf-8")

    lines = [
        "ResNet18 + ArcFace LFW Smoke Baseline",
        "=" * 50,
        f"Root: {args.root}",
        f"Device: {device}",
        f"Backbone: ResNet18",
        f"Loss/head: ArcFace scale=32 margin=0.5",
        f"Embedding dim: {args.embedding_dim}",
        f"Identities: {len(identity_names)}",
        f"Train/val images: {len(train_samples)}/{len(val_samples)}",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch_size}",
        f"Best epoch: {best_row['epoch']}",
        f"Best val accuracy: {best_row['val_acc']:.4f}",
        f"Checkpoint: {args.output}",
        f"Curve: {args.curve}",
        f"History JSON: {args.history_json}",
        f"Elapsed seconds: {elapsed:.3f}",
        "",
        "History:",
    ]
    for row in history:
        marker = " *best" if row["epoch"] == best_row["epoch"] else ""
        lines.append(
            f"- epoch {row['epoch']}: train_loss={row['train_loss']:.4f}, train_acc={row['train_acc']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, val_acc={row['val_acc']:.4f}{marker}"
        )
    lines.extend(["", "Selected identities:"])
    for name in identity_names:
        lines.append(f"- {name}: {identity_counts[name]} images used")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
