from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate InsightFace recognition backbone on LFW funneled aligned images with preprocessing variants.")
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--variant", choices=["full", "center-crop", "center-crop-flip"], default="center-crop-flip")
    parser.add_argument("--crop-ratio", type=float, default=0.72)
    parser.add_argument("--train-pairs", default="pairsDevTrain.txt")
    parser.add_argument("--test-pairs", default="pairsDevTest.txt")
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/insightface_buffalo_l_lfw_aligned_variants_result.txt"))
    parser.add_argument("--json-report", type=Path, default=Path("outputs/reports/insightface_buffalo_l_lfw_aligned_variants_result.json"))
    return parser.parse_args()


def lfw_image_path(name: str, index: int) -> Path:
    return IMAGE_ROOT / name / f"{name}_{index:04d}.jpg"


def parse_pair_line(line: str) -> LfwPair | None:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) == 3:
        name = parts[0]
        a = int(parts[1])
        b = int(parts[2])
        pa = lfw_image_path(name, a)
        pb = lfw_image_path(name, b)
        return LfwPair("positive", name, a, name, b, pa, pb, pa.exists() and pb.exists())
    if len(parts) == 4:
        na = parts[0]
        ia = int(parts[1])
        nb = parts[2]
        ib = int(parts[3])
        pa = lfw_image_path(na, ia)
        pb = lfw_image_path(nb, ib)
        return LfwPair("negative", na, ia, nb, ib, pa, pb, pa.exists() and pb.exists())
    return None


def load_pairs(path: Path) -> tuple[str, list[LfwPair]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0].strip(), [pair for line in lines[1:] if (pair := parse_pair_line(line)) is not None]


def load_app(model_name: str, provider: str):
    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=0 if provider == "cuda" else -1, det_size=(320, 320))
    return app, providers


def center_crop(image: np.ndarray, crop_ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    side = int(round(min(h, w) * crop_ratio))
    side = max(32, min(side, h, w))
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


def normalize(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    return embedding / max(float(np.linalg.norm(embedding)), 1e-12)


def extract_embedding(recognizer, path: Path, variant: str, crop_ratio: float) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    if variant in {"center-crop", "center-crop-flip"}:
        image = center_crop(image, crop_ratio)
    emb = normalize(np.asarray(recognizer.get_feat(image)[0], dtype=np.float32))
    if variant == "center-crop-flip":
        flipped = np.ascontiguousarray(image[:, ::-1])
        emb_flip = normalize(np.asarray(recognizer.get_feat(flipped)[0], dtype=np.float32))
        emb = normalize(emb + emb_flip)
    return emb


def pair_label(pair: LfwPair) -> int:
    return 1 if pair.pair_type == "positive" else 0


def score_pairs(recognizer, pairs: list[LfwPair], cache: dict[str, np.ndarray | None], variant: str, crop_ratio: float):
    scores = []
    labels = []
    skipped = 0
    for pair in pairs:
        if not pair.exists:
            skipped += 1
            continue
        for path in [pair.path_a, pair.path_b]:
            key = str(path)
            if key not in cache:
                cache[key] = extract_embedding(recognizer, path, variant, crop_ratio)
        ea = cache[str(pair.path_a)]
        eb = cache[str(pair.path_b)]
        if ea is None or eb is None:
            skipped += 1
            continue
        scores.append(float(np.dot(ea, eb)))
        labels.append(pair_label(pair))
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32), skipped


def select_threshold(scores: np.ndarray, labels: np.ndarray):
    thresholds = np.linspace(float(scores.min()), float(scores.max()), 1000)
    best_t = float(thresholds[0])
    best_acc = -1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float):
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
    app, providers = load_app(args.model_name, args.provider)
    recognizer = app.models["recognition"]
    cache: dict[str, np.ndarray | None] = {}
    train_scores, train_labels, train_skipped = score_pairs(recognizer, train_pairs, cache, args.variant, args.crop_ratio)
    threshold, train_acc = select_threshold(train_scores, train_labels)
    test_scores, test_labels, test_skipped = score_pairs(recognizer, test_pairs, cache, args.variant, args.crop_ratio)
    test_metrics = metrics(test_scores, test_labels, threshold)
    elapsed = time.perf_counter() - start
    summary = {
        "model_name": args.model_name,
        "providers": providers,
        "variant": args.variant,
        "crop_ratio": args.crop_ratio,
        "train_header": train_header,
        "test_header": test_header,
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
        "LFW Verification - InsightFace Aligned Variant",
        "=" * 56,
        f"Model: {args.model_name}",
        f"Providers: {providers}",
        f"Variant: {args.variant}",
        f"Crop ratio: {args.crop_ratio}",
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
