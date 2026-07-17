from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


LFW_HOME = Path("data/raw/sklearn/lfw_home")
IMAGE_ROOT = LFW_HOME / "lfw_funneled"
DEFAULT_MODEL = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"


@dataclass
class LfwPair:
    pair_type: str
    path_a: Path
    path_b: Path
    exists: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ONNX LFW verification with InsightFace W600K R50.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--variant", choices=["full", "center-crop", "center-crop-flip"], default="full")
    parser.add_argument("--crop-ratio", type=float, default=0.72)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-pairs", default="pairsDevTrain.txt")
    parser.add_argument("--test-pairs", default="pairsDevTest.txt")
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/insightface_w600k_r50_lfw_batch_result.txt"))
    parser.add_argument("--json-report", type=Path, default=Path("outputs/reports/insightface_w600k_r50_lfw_batch_result.json"))
    return parser.parse_args()


def lfw_image_path(name: str, index: int) -> Path:
    return IMAGE_ROOT / name / f"{name}_{index:04d}.jpg"


def parse_pair_line(line: str) -> LfwPair | None:
    parts = line.strip().split()
    if len(parts) == 3:
        name, a, b = parts[0], int(parts[1]), int(parts[2])
        pa, pb = lfw_image_path(name, a), lfw_image_path(name, b)
        return LfwPair("positive", pa, pb, pa.exists() and pb.exists())
    if len(parts) == 4:
        na, ia, nb, ib = parts[0], int(parts[1]), parts[2], int(parts[3])
        pa, pb = lfw_image_path(na, ia), lfw_image_path(nb, ib)
        return LfwPair("negative", pa, pb, pa.exists() and pb.exists())
    return None


def load_pairs(path: Path) -> tuple[str, list[LfwPair]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0].strip(), [pair for line in lines[1:] if (pair := parse_pair_line(line)) is not None]


def unique_paths(pairs: list[LfwPair]) -> list[Path]:
    seen = set()
    paths = []
    for pair in pairs:
        if not pair.exists:
            continue
        for path in (pair.path_a, pair.path_b):
            key = str(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def center_crop(image: np.ndarray, crop_ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(32, min(h, w, int(round(min(h, w) * crop_ratio))))
    x0, y0 = (w - side) // 2, (h - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def load_image(path: Path, variant: str, crop_ratio: float) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    if variant in {"center-crop", "center-crop-flip"}:
        image = center_crop(image, crop_ratio)
    return image


def make_session(model: Path, provider: str):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model), providers=providers), providers


def normalize_rows(array: np.ndarray) -> np.ndarray:
    return array / np.linalg.norm(array, axis=1, keepdims=True).clip(min=1e-12)


def infer_batch(session, input_name: str, output_name: str, images: list[np.ndarray]) -> np.ndarray:
    blob = cv2.dnn.blobFromImages(images, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    return normalize_rows(np.asarray(session.run([output_name], {input_name: blob})[0], dtype=np.float32))


def extract_embeddings(session, paths: list[Path], variant: str, crop_ratio: float, batch_size: int) -> dict[str, np.ndarray | None]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    cache: dict[str, np.ndarray | None] = {}
    images: list[np.ndarray] = []
    keys: list[str] = []

    def flush() -> None:
        nonlocal images, keys
        if not images:
            return
        embeddings = infer_batch(session, input_name, output_name, images)
        if variant == "center-crop-flip":
            flipped = [np.ascontiguousarray(image[:, ::-1]) for image in images]
            embeddings = normalize_rows(embeddings + infer_batch(session, input_name, output_name, flipped))
        for key, embedding in zip(keys, embeddings):
            cache[key] = embedding
        images, keys = [], []

    for index, path in enumerate(paths, start=1):
        image = load_image(path, variant, crop_ratio)
        if image is None:
            cache[str(path)] = None
            continue
        images.append(image)
        keys.append(str(path))
        if len(images) >= batch_size:
            flush()
        if index % 1000 == 0:
            print(f"Embedded {index}/{len(paths)} images", flush=True)
    flush()
    return cache


def score_pairs(pairs: list[LfwPair], cache: dict[str, np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, int]:
    scores, labels = [], []
    skipped = 0
    for pair in pairs:
        if not pair.exists:
            skipped += 1
            continue
        ea, eb = cache.get(str(pair.path_a)), cache.get(str(pair.path_b))
        if ea is None or eb is None:
            skipped += 1
            continue
        scores.append(float(np.dot(ea, eb)))
        labels.append(1 if pair.pair_type == "positive" else 0)
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32), skipped


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_t, best_acc = float(scores.min()), -1.0
    for threshold in np.linspace(float(scores.min()), float(scores.max()), 1000):
        pred = (scores >= threshold).astype(np.int32)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_t, best_acc = float(threshold), acc
    return best_t, best_acc


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int32)
    return {
        "accuracy": round(float((pred == labels).mean()), 4),
        "correct": int((pred == labels).sum()),
        "total": int(len(labels)),
        "tp": int(((pred == 1) & (labels == 1)).sum()),
        "tn": int(((pred == 0) & (labels == 0)).sum()),
        "fp": int(((pred == 1) & (labels == 0)).sum()),
        "fn": int(((pred == 0) & (labels == 1)).sum()),
    }


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    train_header, train_pairs = load_pairs(LFW_HOME / args.train_pairs)
    test_header, test_pairs = load_pairs(LFW_HOME / args.test_pairs)
    session, providers = make_session(args.model, args.provider)
    cache = extract_embeddings(session, unique_paths(train_pairs + test_pairs), args.variant, args.crop_ratio, args.batch_size)
    train_scores, train_labels, train_skipped = score_pairs(train_pairs, cache)
    threshold, train_acc = select_threshold(train_scores, train_labels)
    test_scores, test_labels, test_skipped = score_pairs(test_pairs, cache)
    test_metrics = evaluate(test_scores, test_labels, threshold)
    elapsed = time.perf_counter() - start
    summary = {
        "model": str(args.model),
        "providers": providers,
        "variant": args.variant,
        "crop_ratio": args.crop_ratio,
        "train_pair_file": args.train_pairs,
        "test_pair_file": args.test_pairs,
        "threshold": round(threshold, 4),
        "train_accuracy": round(train_acc, 4),
        "test_metrics": test_metrics,
        "train_pairs_scored": int(len(train_labels)),
        "test_pairs_scored": int(len(test_labels)),
        "train_pairs_skipped": int(train_skipped),
        "test_pairs_skipped": int(test_skipped),
        "unique_embeddings_cached": len(cache),
        "elapsed_seconds": round(elapsed, 3),
    }
    lines = [
        "LFW Verification - InsightFace W600K R50 Batch ONNX",
        "=" * 60,
        f"Model: {args.model}",
        f"Providers: {providers}",
        f"Variant: {args.variant}",
        f"Crop ratio: {args.crop_ratio}",
        f"Train pairs: {args.train_pairs} ({train_header})",
        f"Test pairs: {args.test_pairs} ({test_header})",
        f"Train pairs scored: {len(train_labels)}",
        f"Test pairs scored: {len(test_labels)}",
        f"Selected threshold: {threshold:.4f}",
        f"Train accuracy: {train_acc:.4f}",
        f"Test accuracy: {test_metrics['accuracy']:.4f}",
        f"Correct / total: {test_metrics['correct']} / {test_metrics['total']}",
        f"TP: {test_metrics['tp']}",
        f"TN: {test_metrics['tn']}",
        f"FP: {test_metrics['fp']}",
        f"FN: {test_metrics['fn']}",
        f"Unique embeddings cached: {len(cache)}",
        f"Elapsed seconds: {elapsed:.3f}",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    args.json_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
