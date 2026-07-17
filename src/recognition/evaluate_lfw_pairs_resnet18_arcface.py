from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


LFW_HOME = Path("data/raw/sklearn/lfw_home")
IMAGE_ROOT = LFW_HOME / "lfw_funneled"


@dataclass
class LfwPair:
    pair_type: str
    name_a: str
    index_a: int
    name_b: str
    index_b: int
    path_a: Path
    path_b: Path
    exists: bool


class ResNetEmbedding(nn.Module):
    def __init__(self, backbone_name: str, embedding_dim: int):
        super().__init__()
        if backbone_name == "resnet50":
            backbone = models.resnet50(weights=None)
        elif backbone_name == "resnet18":
            backbone = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ResNet ArcFace checkpoint on LFW pair verification.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/resnet18_arcface_lfw_smoke10_best.pt"))
    parser.add_argument("--train-pairs", default="pairsDevTrain.txt")
    parser.add_argument("--test-pairs", default="pairsDevTest.txt")
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/resnet18_arcface_lfw_pair_verification_result.txt"))
    parser.add_argument("--json-report", type=Path, default=Path("outputs/reports/resnet18_arcface_lfw_pair_verification_result.json"))
    return parser.parse_args()


def lfw_image_path(name: str, index: int) -> Path:
    return IMAGE_ROOT / name / f"{name}_{index:04d}.jpg"


def parse_pair_line(line: str) -> LfwPair | None:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) == 3:
        name = parts[0]
        index_a = int(parts[1])
        index_b = int(parts[2])
        path_a = lfw_image_path(name, index_a)
        path_b = lfw_image_path(name, index_b)
        return LfwPair("positive", name, index_a, name, index_b, path_a, path_b, path_a.exists() and path_b.exists())
    if len(parts) == 4:
        name_a = parts[0]
        index_a = int(parts[1])
        name_b = parts[2]
        index_b = int(parts[3])
        path_a = lfw_image_path(name_a, index_a)
        path_b = lfw_image_path(name_b, index_b)
        return LfwPair("negative", name_a, index_a, name_b, index_b, path_a, path_b, path_a.exists() and path_b.exists())
    return None


def load_pairs(path: Path) -> tuple[str, list[LfwPair]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].strip()
    pairs = [pair for line in lines[1:] if (pair := parse_pair_line(line)) is not None]
    return header, pairs


def pair_label(pair: LfwPair) -> int:
    return 1 if pair.pair_type == "positive" else 0


def preprocess_image(path: Path, image_size: int) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.5, 0.5, 0.5], dtype=np.float32)) / np.array([0.5, 0.5, 0.5], dtype=np.float32)
    image = np.transpose(image, (2, 0, 1))
    return image


def load_model(checkpoint_path: Path, device: str) -> tuple[ResNetEmbedding, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = str(checkpoint.get("backbone", "resnet18"))
    model = ResNetEmbedding(backbone, int(checkpoint.get("embedding_dim", 128))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def unique_existing_paths(pairs: list[LfwPair]) -> list[Path]:
    paths = set()
    for pair in pairs:
        if pair.exists:
            paths.add(pair.path_a)
            paths.add(pair.path_b)
    return sorted(paths)


def extract_embeddings(model: ResNetEmbedding, image_paths: list[Path], image_size: int, batch_size: int, device: str) -> dict[str, np.ndarray | None]:
    cache: dict[str, np.ndarray | None] = {}
    tensors = []
    keys = []
    with torch.no_grad():
        for path in image_paths:
            image = preprocess_image(path, image_size)
            if image is None:
                cache[str(path)] = None
                continue
            tensors.append(torch.from_numpy(image))
            keys.append(str(path))
            if len(tensors) >= batch_size:
                batch = torch.stack(tensors, dim=0).to(device)
                embeddings = model(batch).cpu().numpy().astype(np.float32)
                for key, emb in zip(keys, embeddings):
                    cache[key] = emb / max(float(np.linalg.norm(emb)), 1e-12)
                tensors.clear()
                keys.clear()
        if tensors:
            batch = torch.stack(tensors, dim=0).to(device)
            embeddings = model(batch).cpu().numpy().astype(np.float32)
            for key, emb in zip(keys, embeddings):
                cache[key] = emb / max(float(np.linalg.norm(emb)), 1e-12)
    return cache


def score_pairs(pairs: list[LfwPair], embedding_cache: dict[str, np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, int]:
    scores = []
    labels = []
    skipped = 0
    for pair in pairs:
        if not pair.exists:
            skipped += 1
            continue
        emb_a = embedding_cache.get(str(pair.path_a))
        emb_b = embedding_cache.get(str(pair.path_b))
        if emb_a is None or emb_b is None:
            skipped += 1
            continue
        scores.append(float(np.dot(emb_a, emb_b)))
        labels.append(pair_label(pair))
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32), skipped


def select_best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(scores) == 0:
        return 0.0, 0.0
    thresholds = np.linspace(float(scores.min()), float(scores.max()), 800)
    best_threshold = float(thresholds[0])
    best_accuracy = -1.0
    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int32)
        accuracy = float((preds == labels).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def evaluate_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(np.int32)
    accuracy = float((preds == labels).mean()) if len(labels) else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "correct": int((preds == labels).sum()),
        "total": int(len(labels)),
        "true_positive": int(((preds == 1) & (labels == 1)).sum()),
        "true_negative": int(((preds == 0) & (labels == 0)).sum()),
        "false_positive": int(((preds == 1) & (labels == 0)).sum()),
        "false_negative": int(((preds == 0) & (labels == 1)).sum()),
    }


def score_summary(scores: np.ndarray, labels: np.ndarray) -> dict:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    return {
        "positive_mean": round(float(pos.mean()), 4) if len(pos) else None,
        "positive_std": round(float(pos.std()), 4) if len(pos) else None,
        "negative_mean": round(float(neg.mean()), 4) if len(neg) else None,
        "negative_std": round(float(neg.std()), 4) if len(neg) else None,
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    start = time.perf_counter()

    train_header, train_pairs = load_pairs(LFW_HOME / args.train_pairs)
    test_header, test_pairs = load_pairs(LFW_HOME / args.test_pairs)
    if args.max_train_pairs > 0:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_test_pairs > 0:
        test_pairs = test_pairs[: args.max_test_pairs]

    model, checkpoint = load_model(args.checkpoint, device)
    image_size = int(checkpoint.get("image_size", 112))
    all_paths = unique_existing_paths(train_pairs + test_pairs)
    embedding_cache = extract_embeddings(model, all_paths, image_size, args.batch_size, device)

    train_scores, train_labels, train_skipped = score_pairs(train_pairs, embedding_cache)
    threshold, train_acc = select_best_threshold(train_scores, train_labels)
    test_scores, test_labels, test_skipped = score_pairs(test_pairs, embedding_cache)
    test_metrics = evaluate_at_threshold(test_scores, test_labels, threshold)
    elapsed = time.perf_counter() - start

    summary = {
        "task": "LFW pair verification with trained ResNet ArcFace checkpoint",
        "checkpoint": str(args.checkpoint),
        "device": device,
        "backbone": checkpoint.get("backbone", "resnet18"),
        "embedding_dim": int(checkpoint.get("embedding_dim", 128)),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_best_val_acc": checkpoint.get("best_val_acc"),
        "train_pair_file": args.train_pairs,
        "train_pair_header": train_header,
        "test_pair_file": args.test_pairs,
        "test_pair_header": test_header,
        "threshold_selected_on_train": round(threshold, 4),
        "train_accuracy_at_selected_threshold": round(train_acc, 4),
        "test_metrics": test_metrics,
        "train_score_summary": score_summary(train_scores, train_labels),
        "test_score_summary": score_summary(test_scores, test_labels),
        "train_pairs_scored": int(len(train_labels)),
        "test_pairs_scored": int(len(test_labels)),
        "train_pairs_skipped": int(train_skipped),
        "test_pairs_skipped": int(test_skipped),
        "unique_embeddings_cached": len(embedding_cache),
        "elapsed_seconds": round(elapsed, 3),
    }

    lines = [
        "LFW Pair Verification - ResNet ArcFace Checkpoint",
        "=" * 64,
        f"Checkpoint: {args.checkpoint}",
        f"Device: {device}",
        f"Backbone: {summary['backbone']}",
        f"Embedding dim: {summary['embedding_dim']}",
        f"Checkpoint best epoch: {summary['checkpoint_best_epoch']}",
        f"Checkpoint best val acc: {summary['checkpoint_best_val_acc']}",
        f"Train pair file: {args.train_pairs} ({train_header})",
        f"Test pair file: {args.test_pairs} ({test_header})",
        f"Train pairs scored: {len(train_labels)}",
        f"Test pairs scored: {len(test_labels)}",
        f"Train pairs skipped: {train_skipped}",
        f"Test pairs skipped: {test_skipped}",
        f"Selected threshold: {threshold:.4f}",
        f"Train accuracy at selected threshold: {train_acc:.4f}",
        f"Test accuracy: {test_metrics['accuracy']:.4f}",
        f"Correct / total: {test_metrics['correct']} / {test_metrics['total']}",
        f"TP: {test_metrics['true_positive']}",
        f"TN: {test_metrics['true_negative']}",
        f"FP: {test_metrics['false_positive']}",
        f"FN: {test_metrics['false_negative']}",
        f"Train score summary: {summary['train_score_summary']}",
        f"Test score summary: {summary['test_score_summary']}",
        f"Unique embeddings cached: {len(embedding_cache)}",
        f"Elapsed seconds: {elapsed:.3f}",
        "",
        "Interpretation:",
        "- This evaluates the project-trained smoke checkpoint, not a pretrained recognition model.",
        "- Threshold is selected on pairsDevTrain and reported on pairsDevTest.",
        "- LFW funneled images are used directly as aligned inputs.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
