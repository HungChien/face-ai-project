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
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models


@dataclass
class FaceSample:
    image_path: Path
    label: int
    identity: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet ArcFace baseline on a face identity dataset.")
    parser.add_argument("--dataset-format", choices=["celeba", "folder"], default="celeba")
    parser.add_argument("--data-root", type=Path, default=None, help="Folder dataset root. Expected layout: <data-root>/<identity>/*.jpg.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--celeba-root", type=Path, default=Path("data/raw/celeba"))
    parser.add_argument("--backbone", choices=["resnet18", "resnet50", "iresnet50", "mobilefacenet"], default="resnet50")
    parser.add_argument("--pretrained", choices=["none", "imagenet"], default="none")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--sampler", choices=["random", "pk"], default="random")
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--images-per-identity", type=int, default=4)
    parser.add_argument("--strong-augment", action="store_true")
    parser.add_argument("--erase-prob", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-identities", type=int, default=200)
    parser.add_argument("--min-train-images", type=int, default=5)
    parser.add_argument("--max-train-images-per-identity", type=int, default=30)
    parser.add_argument("--max-val-images-per-identity", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["random", "official"], default="random")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epoch-cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/resnet50_arcface_celeba_subset_best.pt"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/resnet50_arcface_celeba_subset_result.txt"))
    parser.add_argument("--history-json", type=Path, default=Path("outputs/reports/resnet50_arcface_celeba_subset_history.json"))
    parser.add_argument("--curve", type=Path, default=Path("outputs/images/resnet50_arcface_celeba_subset_curves.jpg"))
    return parser.parse_args()


def read_mapping(path: Path) -> dict[str, str]:
    mapping = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1]
    return mapping


def build_celeba_samples(args: argparse.Namespace) -> tuple[list[FaceSample], list[FaceSample], list[str], dict[str, dict[str, int]]]:
    image_dir = args.celeba_root / "img_align_celeba"
    identity_map = read_mapping(args.celeba_root / "identity_CelebA.txt")
    partition_map = read_mapping(args.celeba_root / "list_eval_partition.txt")
    by_identity: dict[str, dict[str, list[Path]]] = {}

    for file_name, identity in identity_map.items():
        path = image_dir / file_name
        if not path.exists():
            continue
        partition = partition_map.get(file_name, "0")
        bucket = "train" if partition == "0" else "val" if partition == "1" else "test"
        by_identity.setdefault(identity, {"train": [], "val": [], "test": [], "all": []})[bucket].append(path)
        by_identity[identity]["all"].append(path)

    if args.split_mode == "official":
        eligible = [
            (identity, splits)
            for identity, splits in by_identity.items()
            if len(splits["train"]) >= args.min_train_images and len(splits["val"]) >= 1
        ]
        eligible.sort(key=lambda item: (-len(item[1]["train"]), item[0]))
    else:
        min_total = max(args.min_train_images + 1, int(np.ceil(args.min_train_images / max(1.0 - args.val_ratio, 1e-6))))
        eligible = [
            (identity, splits)
            for identity, splits in by_identity.items()
            if len(splits["all"]) >= min_total
        ]
        eligible.sort(key=lambda item: (-len(item[1]["all"]), item[0]))

    selected = eligible[: args.num_identities]
    if len(selected) < 2:
        raise ValueError("Need at least two eligible identities for the selected split mode.")

    rng = random.Random(args.seed)
    identity_names = [identity for identity, _splits in selected]
    label_map = {identity: idx for idx, identity in enumerate(identity_names)}
    train_samples: list[FaceSample] = []
    val_samples: list[FaceSample] = []
    counts: dict[str, dict[str, int]] = {}

    for identity, splits in selected:
        if args.split_mode == "official":
            train_paths = splits["train"].copy()
            val_paths = splits["val"].copy()
            rng.shuffle(train_paths)
            rng.shuffle(val_paths)
        else:
            paths = splits["all"].copy()
            rng.shuffle(paths)
            max_total = args.max_train_images_per_identity + args.max_val_images_per_identity
            paths = paths[:max_total]
            val_count = max(1, int(round(len(paths) * args.val_ratio)))
            val_count = min(val_count, args.max_val_images_per_identity, len(paths) - 1)
            train_paths = paths[val_count:]
            val_paths = paths[:val_count]
        train_paths = sorted(train_paths[: args.max_train_images_per_identity])
        val_paths = sorted(val_paths[: args.max_val_images_per_identity])
        label = label_map[identity]
        train_samples.extend(FaceSample(path, label, identity) for path in train_paths)
        val_samples.extend(FaceSample(path, label, identity) for path in val_paths)
        counts[identity] = {"train": len(train_paths), "val": len(val_paths), "all": len(splits["all"]), "official_train": len(splits["train"]), "official_val": len(splits["val"])}

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples, identity_names, counts




def list_identity_images(identity_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in identity_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def build_folder_samples(args: argparse.Namespace) -> tuple[list[FaceSample], list[FaceSample], list[str], dict[str, dict[str, int]]]:
    if args.data_root is None:
        raise ValueError("--data-root is required when --dataset-format folder is used.")
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")

    identity_dirs = sorted(path for path in args.data_root.iterdir() if path.is_dir())
    by_identity = []
    min_total = max(args.min_train_images + 1, int(np.ceil(args.min_train_images / max(1.0 - args.val_ratio, 1e-6))))
    for identity_dir in identity_dirs:
        images = list_identity_images(identity_dir)
        if len(images) >= min_total:
            by_identity.append((identity_dir.name, images))
    by_identity.sort(key=lambda item: (-len(item[1]), item[0]))

    selected = by_identity[: args.num_identities]
    if len(selected) < 2:
        raise ValueError(
            "Need at least two eligible identities. "
            "Expected folder layout: <data-root>/<identity>/*.jpg with enough images per identity."
        )

    rng = random.Random(args.seed)
    identity_names = [identity for identity, _paths in selected]
    label_map = {identity: idx for idx, identity in enumerate(identity_names)}
    train_samples: list[FaceSample] = []
    val_samples: list[FaceSample] = []
    counts: dict[str, dict[str, int]] = {}

    for identity, paths in selected:
        paths = paths.copy()
        rng.shuffle(paths)
        max_total = args.max_train_images_per_identity + args.max_val_images_per_identity
        paths = paths[:max_total]
        val_count = max(1, int(round(len(paths) * args.val_ratio)))
        val_count = min(val_count, args.max_val_images_per_identity, len(paths) - 1)
        val_paths = sorted(paths[:val_count])
        train_paths = sorted(paths[val_count : val_count + args.max_train_images_per_identity])
        label = label_map[identity]
        train_samples.extend(FaceSample(path, label, identity) for path in train_paths)
        val_samples.extend(FaceSample(path, label, identity) for path in val_paths)
        counts[identity] = {"train": len(train_paths), "val": len(val_paths), "all": len(paths)}

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples, identity_names, counts


def build_samples(args: argparse.Namespace) -> tuple[list[FaceSample], list[FaceSample], list[str], dict[str, dict[str, int]]]:
    if args.dataset_format == "celeba":
        return build_celeba_samples(args)
    if args.dataset_format == "folder":
        return build_folder_samples(args)
    raise ValueError(args.dataset_format)

class FaceIdentityDataset(Dataset):
    def __init__(self, samples: list[FaceSample], image_size: int, augment: bool, strong_augment: bool = False, erase_prob: float = 0.0):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.strong_augment = strong_augment
        self.erase_prob = erase_prob

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
            image = np.clip(image * random.uniform(0.85, 1.15) + random.uniform(-16, 16), 0, 255).astype(np.uint8)
        if self.augment and self.strong_augment and random.random() < 0.25:
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=random.uniform(0.1, 1.0))
        if self.augment and self.strong_augment and random.random() < 0.20:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.repeat(gray[:, :, None], 3, axis=2)
        if self.augment and self.erase_prob > 0 and random.random() < self.erase_prob:
            h, w = image.shape[:2]
            erase_h = random.randint(max(4, h // 10), max(5, h // 4))
            erase_w = random.randint(max(4, w // 10), max(5, w // 4))
            top = random.randint(0, max(0, h - erase_h))
            left = random.randint(0, max(0, w - erase_w))
            image[top : top + erase_h, left : left + erase_w] = np.asarray([127, 127, 127], dtype=np.uint8)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.5, 0.5, 0.5], dtype=np.float32)) / np.array([0.5, 0.5, 0.5], dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image), torch.tensor(sample.label, dtype=torch.long)


class PKBatchSampler(Sampler[list[int]]):
    def __init__(self, samples: list[FaceSample], identities_per_batch: int, images_per_identity: int, batches_per_epoch: int, seed: int):
        self.identities_per_batch = identities_per_batch
        self.images_per_identity = images_per_identity
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        by_label: dict[int, list[int]] = {}
        for index, sample in enumerate(samples):
            by_label.setdefault(sample.label, []).append(index)
        self.by_label = by_label
        self.labels = sorted(by_label)
        if len(self.labels) < identities_per_batch:
            raise ValueError("PK sampler requires at least identities_per_batch labels.")

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            labels = rng.sample(self.labels, self.identities_per_batch)
            batch = []
            for label in labels:
                candidates = self.by_label[label]
                if len(candidates) >= self.images_per_identity:
                    batch.extend(rng.sample(candidates, self.images_per_identity))
                else:
                    batch.extend(rng.choice(candidates) for _ in range(self.images_per_identity))
            rng.shuffle(batch)
            yield batch

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
        self._initialize_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [IBasicBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.1)
                nn.init.constant_(module.bias, 0)

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
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.features(images)
        return F.normalize(embeddings, p=2, dim=1)

class ResNetEmbedding(nn.Module):
    def __init__(self, backbone_name: str, embedding_dim: int, pretrained: str = "none"):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "mobilefacenet":
            if pretrained != "none":
                print("Warning: mobilefacenet does not use torchvision ImageNet weights; using random initialization.")
            self.model = MobileFaceNetEmbedding(embedding_dim)
        elif backbone_name == "iresnet50":
            if pretrained != "none":
                print("Warning: iresnet50 does not use torchvision ImageNet weights; using random initialization.")
            self.model = IResNetEmbedding([3, 4, 14, 3], embedding_dim)
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained == "imagenet" else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.model = nn.Sequential(
                backbone,
                nn.Linear(in_features, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained == "imagenet" else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.model = nn.Sequential(
                backbone,
                nn.Linear(in_features, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
        else:
            raise ValueError(backbone_name)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(images)
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
        return ((one_hot * phi) + ((1.0 - one_hot) * cosine)) * self.scale


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().detach().cpu())



def learning_rate_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    if args.scheduler == "none":
        return args.lr
    if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
        return args.lr * epoch / args.warmup_epochs
    if args.scheduler == "cosine":
        span = max(1, args.epochs - args.warmup_epochs)
        progress = min(1.0, max(0.0, (epoch - args.warmup_epochs - 1) / span))
        cosine = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
        return args.min_lr + (args.lr - args.min_lr) * cosine
    raise ValueError(args.scheduler)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
def run_epoch(model: nn.Module, head: ArcFaceHead, loader: DataLoader, device: str, optimizer=None) -> dict[str, float]:
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
        accuracies.append(batch_accuracy(eval_logits.detach(), labels.detach()))
    return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(accuracies))}


def save_curve(history: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("Accuracy")
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

    train_samples, val_samples, identity_names, counts = build_samples(args)
    dataset_name = args.dataset_name or ("CelebA identity subset" if args.dataset_format == "celeba" else args.data_root.name)
    train_dataset = FaceIdentityDataset(train_samples, args.image_size, True, strong_augment=args.strong_augment, erase_prob=args.erase_prob)
    val_dataset = FaceIdentityDataset(val_samples, args.image_size, False)
    pk_sampler = None
    if args.sampler == "pk":
        pk_batch_size = args.identities_per_batch * args.images_per_identity
        batches_per_epoch = max(1, len(train_samples) // pk_batch_size)
        pk_sampler = PKBatchSampler(train_samples, args.identities_per_batch, args.images_per_identity, batches_per_epoch, args.seed)
        train_loader = DataLoader(train_dataset, batch_sampler=pk_sampler, num_workers=args.num_workers)
        effective_batch_size = pk_batch_size
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        effective_batch_size = args.batch_size
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = ResNetEmbedding(args.backbone, args.embedding_dim, args.pretrained).to(device)
    head = ArcFaceHead(args.embedding_dim, len(identity_names)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    best_row = None
    best_state = None
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if pk_sampler is not None:
            pk_sampler.set_epoch(epoch)
        current_lr = learning_rate_for_epoch(args, epoch)
        set_optimizer_lr(optimizer, current_lr)
        train_metrics = run_epoch(model, head, train_loader, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, head, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "lr": current_lr,
        }
        history.append(row)
        if best_row is None or row["val_acc"] > best_row["val_acc"]:
            best_row = row.copy()
            best_state = {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                **best_state,
                "backbone": args.backbone,
                "loss": "ArcFace",
                "pretrained": args.pretrained,
                "embedding_dim": args.embedding_dim,
                "num_classes": len(identity_names),
                "identity_names": identity_names,
                "identity_counts": counts,
                "image_size": args.image_size,
                "best_epoch": best_row["epoch"],
                "best_val_acc": best_row["val_acc"],
                "history": history,
                "dataset": dataset_name,
                "dataset_format": args.dataset_format,
                "data_root": str(args.data_root) if args.data_root is not None else str(args.celeba_root),
                "optimizer": "AdamW",
                "weight_decay": args.weight_decay,
                "scheduler": args.scheduler,
                "warmup_epochs": args.warmup_epochs,
                "min_lr": args.min_lr,
                "sampler": args.sampler,
                "identities_per_batch": args.identities_per_batch,
                "images_per_identity": args.images_per_identity,
                "effective_batch_size": effective_batch_size,
                "strong_augment": args.strong_augment,
                "erase_prob": args.erase_prob,
                "partial_checkpoint": True,
            }, args.output)
            args.history_json.parent.mkdir(parents=True, exist_ok=True)
            args.history_json.write_text(json.dumps({"history": history, "best": best_row, "identity_counts": counts, "partial": True}, indent=2), encoding="utf-8")
        print(row)
        if args.epoch_cooldown_seconds > 0 and epoch < args.epochs:
            time.sleep(args.epoch_cooldown_seconds)

    if best_state is None or best_row is None:
        raise RuntimeError("No best checkpoint was produced.")
    elapsed = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **best_state,
        "backbone": args.backbone,
        "loss": "ArcFace",
        "pretrained": args.pretrained,
        "embedding_dim": args.embedding_dim,
        "num_classes": len(identity_names),
        "identity_names": identity_names,
        "identity_counts": counts,
        "image_size": args.image_size,
        "best_epoch": best_row["epoch"],
        "best_val_acc": best_row["val_acc"],
        "history": history,
        "dataset": dataset_name,
        "dataset_format": args.dataset_format,
        "data_root": str(args.data_root) if args.data_root is not None else str(args.celeba_root),
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "sampler": args.sampler,
        "identities_per_batch": args.identities_per_batch,
        "images_per_identity": args.images_per_identity,
        "effective_batch_size": effective_batch_size,
        "strong_augment": args.strong_augment,
        "erase_prob": args.erase_prob,
    }, args.output)
    save_curve(history, args.curve)
    args.history_json.parent.mkdir(parents=True, exist_ok=True)
    args.history_json.write_text(json.dumps({"history": history, "best": best_row, "identity_counts": counts}, indent=2), encoding="utf-8")

    lines = [
        "ResNet ArcFace Face Identity Training",
        "=" * 50,
        f"Dataset: {dataset_name}",
        f"Dataset format: {args.dataset_format}",
        f"Data root: {args.data_root if args.data_root is not None else args.celeba_root}",
        f"Device: {device}",
        f"Backbone: {args.backbone}",
        f"Pretrained: {args.pretrained}",
        f"Loss/head: ArcFace scale=32 margin=0.5",
        f"Embedding dim: {args.embedding_dim}",
        f"Identities: {len(identity_names)}",
        f"Split mode: {args.split_mode}",
        f"Train/val images: {len(train_samples)}/{len(val_samples)}",
        f"Epochs: {args.epochs}",
        f"Batch size: {effective_batch_size}",
        f"Sampler: {args.sampler}",
        f"Scheduler: {args.scheduler}",
        f"Warmup epochs: {args.warmup_epochs}",
        f"Base/min LR: {args.lr}/{args.min_lr}",
        f"Weight decay: {args.weight_decay}",
        f"Strong augment: {args.strong_augment}",
        f"Erase prob: {args.erase_prob}",
        f"Epoch cooldown seconds: {args.epoch_cooldown_seconds}",
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
        lines.append(f"- epoch {row['epoch']}: lr={row.get('lr', args.lr):.6g}, train_loss={row['train_loss']:.4f}, train_acc={row['train_acc']:.4f}, val_loss={row['val_loss']:.4f}, val_acc={row['val_acc']:.4f}{marker}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()




















