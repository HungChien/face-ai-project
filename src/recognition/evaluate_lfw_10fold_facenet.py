"""Evaluate facenet-pytorch VGGFace2 InceptionResnetV1 on LFW 6000 pairs.

This script follows the official LFW pairs.txt 10-fold protocol:
for each fold, thresholds are selected on the other nine folds and evaluated
on the held-out fold.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image


@dataclass
class PairRecord:
    fold: int
    path1: str
    path2: str
    same: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lfw-home",
        type=Path,
        default=Path("data/raw/sklearn/lfw_home"),
        help="Directory containing pairs.txt and lfw_funneled.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="LFW image directory. Defaults to <lfw-home>/lfw_funneled.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-thresholds", type=int, default=1000)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/embeddings/facenet_vggface2_lfw_embeddings.npz"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/reports/facenet_vggface2_lfw_10fold_result.txt"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("outputs/reports/facenet_vggface2_lfw_10fold_result.json"),
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def lfw_image_path(image_dir: Path, person: str, index: str) -> Path:
    return image_dir / person / f"{person}_{int(index):04d}.jpg"


def parse_pairs(pairs_path: Path, image_dir: Path) -> list[PairRecord]:
    lines = [line.strip() for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = lines[0].split()
    if len(header) == 2:
        num_folds, pairs_per_type = int(header[0]), int(header[1])
        fold_size = pairs_per_type * 2
        pair_lines = lines[1:]
    else:
        num_folds, fold_size = 10, 600
        pair_lines = lines

    records: list[PairRecord] = []
    for i, line in enumerate(pair_lines):
        parts = line.split()
        fold = i // fold_size
        if len(parts) == 3:
            person, idx1, idx2 = parts
            path1 = lfw_image_path(image_dir, person, idx1)
            path2 = lfw_image_path(image_dir, person, idx2)
            same = True
        elif len(parts) == 4:
            person1, idx1, person2, idx2 = parts
            path1 = lfw_image_path(image_dir, person1, idx1)
            path2 = lfw_image_path(image_dir, person2, idx2)
            same = False
        else:
            raise ValueError(f"Bad LFW pair line: {line}")
        records.append(PairRecord(fold=fold, path1=str(path1), path2=str(path2), same=same))

    expected = num_folds * fold_size
    if len(records) != expected:
        raise ValueError(f"Expected {expected} pairs from header, got {len(records)}")
    return records


def preprocess_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((160, 160), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - 127.5) / 128.0


def compute_embeddings(
    model: InceptionResnetV1,
    paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    total = len(paths)
    started = time.time()
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_paths = paths[start : start + batch_size]
            images = torch.stack([preprocess_image(path) for path in batch_paths]).to(device)
            batch_embeddings = model(images).detach().cpu().numpy()
            batch_embeddings /= np.linalg.norm(batch_embeddings, axis=1, keepdims=True) + 1e-12
            for path, embedding in zip(batch_paths, batch_embeddings):
                embeddings[str(path)] = embedding.astype(np.float32)
            done = min(start + batch_size, total)
            if done == total or done % (batch_size * 10) == 0:
                elapsed = time.time() - started
                print(f"embedded {done}/{total} images, elapsed={elapsed:.1f}s")
    return embeddings


def load_or_compute_embeddings(
    cache_path: Path,
    unique_paths: list[Path],
    model: InceptionResnetV1,
    batch_size: int,
    device: torch.device,
    force_recompute: bool,
) -> dict[str, np.ndarray]:
    expected_keys = {str(path) for path in unique_paths}
    if cache_path.exists() and not force_recompute:
        cached = np.load(cache_path, allow_pickle=False)
        if set(cached.files) == expected_keys:
            return {key: cached[key] for key in cached.files}
        print("embedding cache does not match pairs.txt; recomputing")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = compute_embeddings(model, unique_paths, batch_size, device)
    np.savez_compressed(cache_path, **embeddings)
    return embeddings


def choose_threshold(scores: np.ndarray, labels: np.ndarray, num_thresholds: int) -> tuple[float, float]:
    thresholds = np.linspace(float(scores.min()), float(scores.max()), num_thresholds)
    best_acc = -1.0
    best_threshold = float(thresholds[0])
    for threshold in thresholds:
        acc = float(((scores >= threshold) == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return best_threshold, best_acc


def evaluate_10fold(records: list[PairRecord], embeddings: dict[str, np.ndarray], num_thresholds: int) -> dict:
    scores = np.array(
        [float(np.dot(embeddings[record.path1], embeddings[record.path2])) for record in records],
        dtype=np.float32,
    )
    labels = np.array([record.same for record in records], dtype=bool)
    folds = np.array([record.fold for record in records], dtype=np.int32)

    fold_results = []
    total_correct = 0
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        test_mask = folds == fold
        threshold, train_acc = choose_threshold(scores[train_mask], labels[train_mask], num_thresholds)
        predictions = scores[test_mask] >= threshold
        correct = int((predictions == labels[test_mask]).sum())
        total = int(test_mask.sum())
        acc = correct / total
        total_correct += correct
        fold_results.append(
            {
                "fold": int(fold + 1),
                "threshold": threshold,
                "train_accuracy": train_acc,
                "test_accuracy": acc,
                "correct": correct,
                "total": total,
            }
        )

    fold_acc = np.array([item["test_accuracy"] for item in fold_results], dtype=np.float64)
    return {
        "protocol": "LFW pairs.txt 6000 pairs / 10-fold",
        "model": "facenet-pytorch InceptionResnetV1 pretrained=vggface2",
        "num_pairs": len(records),
        "num_folds": len(fold_results),
        "mean_accuracy": float(fold_acc.mean()),
        "std_accuracy": float(fold_acc.std(ddof=1)),
        "total_correct": int(total_correct),
        "fold_results": fold_results,
    }


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir or (args.lfw_home / "lfw_funneled")
    pairs_path = args.lfw_home / "pairs.txt"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing pairs.txt: {pairs_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing LFW image directory: {image_dir}")

    records = parse_pairs(pairs_path, image_dir)
    unique_paths = sorted({Path(record.path1) for record in records} | {Path(record.path2) for record in records})
    missing = [path for path in unique_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} LFW images, first missing: {missing[0]}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device}")
    print(f"pairs={len(records)}, unique_images={len(unique_paths)}")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    embeddings = load_or_compute_embeddings(
        args.cache,
        unique_paths,
        model,
        args.batch_size,
        device,
        args.force_recompute,
    )
    result = evaluate_10fold(records, embeddings, args.num_thresholds)
    result["device"] = str(device)
    result["batch_size"] = args.batch_size
    result["embedding_cache"] = str(args.cache)
    result["lfw_home"] = str(args.lfw_home)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Facenet-pytorch VGGFace2 LFW 10-fold Verification",
        "=" * 56,
        f"Protocol: {result['protocol']}",
        f"Model: {result['model']}",
        f"Device: {result['device']}",
        f"Pairs: {result['num_pairs']}",
        f"Mean accuracy: {result['mean_accuracy']:.4f}",
        f"Std accuracy: {result['std_accuracy']:.4f}",
        f"Total correct: {result['total_correct']}/{result['num_pairs']}",
        "",
        "Fold results:",
    ]
    for item in result["fold_results"]:
        lines.append(
            "  fold {fold:02d}: acc={test_accuracy:.4f}, train_acc={train_accuracy:.4f}, "
            "threshold={threshold:.4f}, correct={correct}/{total}".format(**item)
        )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved report: {args.report}")
    print(f"saved json: {args.json}")


if __name__ == "__main__":
    main()
