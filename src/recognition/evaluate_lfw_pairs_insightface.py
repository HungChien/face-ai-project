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
    parser = argparse.ArgumentParser(description="Evaluate InsightFace face-domain pretrained backbone on LFW pairs.")
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--ctx-id", type=int, default=0)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--embedding-mode", choices=["aligned", "detected"], default="detected")
    parser.add_argument("--train-pairs", default="pairsDevTrain.txt")
    parser.add_argument("--test-pairs", default="pairsDevTest.txt")
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/insightface_buffalo_l_lfw_gpu_result.txt"))
    parser.add_argument("--json-report", type=Path, default=Path("outputs/reports/insightface_buffalo_l_lfw_gpu_result.json"))
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


def load_app(args: argparse.Namespace):
    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.provider == "cuda" else ["CPUExecutionProvider"]
    ctx_id = args.ctx_id if args.provider == "cuda" else -1
    app = FaceAnalysis(name=args.model_name, providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size))
    return app, providers


def normalize_embedding(embedding: np.ndarray) -> np.ndarray | None:
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm <= 0:
        return None
    return embedding / norm


def extract_embedding(app, image_path: Path, embedding_mode: str) -> np.ndarray | None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None
    if embedding_mode == "aligned":
        recognizer = app.models["recognition"]
        embedding = np.asarray(recognizer.get_feat(image_bgr)[0], dtype=np.float32)
        return normalize_embedding(embedding)
    faces = app.get(image_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
    return normalize_embedding(np.asarray(face.normed_embedding, dtype=np.float32))


def pair_label(pair: LfwPair) -> int:
    return 1 if pair.pair_type == "positive" else 0


def score_pairs(app, pairs: list[LfwPair], embedding_cache: dict[str, np.ndarray | None], embedding_mode: str) -> tuple[np.ndarray, np.ndarray, int]:
    scores = []
    labels = []
    skipped = 0
    valid_pairs = [pair for pair in pairs if pair.exists]
    for index, pair in enumerate(valid_pairs, start=1):
        for path in [pair.path_a, pair.path_b]:
            key = str(path)
            if key not in embedding_cache:
                embedding_cache[key] = extract_embedding(app, path, embedding_mode)
        emb_a = embedding_cache[str(pair.path_a)]
        emb_b = embedding_cache[str(pair.path_b)]
        if emb_a is None or emb_b is None:
            skipped += 1
            continue
        scores.append(float(np.dot(emb_a, emb_b)))
        labels.append(pair_label(pair))
        if index % 100 == 0:
            print(f"Scored {index}/{len(valid_pairs)} pairs, cache={len(embedding_cache)}, skipped={skipped}")
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32), skipped


def select_best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(scores) == 0:
        return 0.0, 0.0
    thresholds = np.linspace(float(scores.min()), float(scores.max()), 1000)
    best_threshold = float(thresholds[0])
    best_accuracy = -1.0
    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int32)
        accuracy = float((preds == labels).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(np.int32)
    return {
        "accuracy": round(float((preds == labels).mean()), 4),
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
    start = time.perf_counter()
    train_header, train_pairs = load_pairs(LFW_HOME / args.train_pairs)
    test_header, test_pairs = load_pairs(LFW_HOME / args.test_pairs)
    if args.max_train_pairs > 0:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_test_pairs > 0:
        test_pairs = test_pairs[: args.max_test_pairs]

    app, providers = load_app(args)
    embedding_cache: dict[str, np.ndarray | None] = {}
    train_scores, train_labels, train_skipped = score_pairs(app, train_pairs, embedding_cache, args.embedding_mode)
    threshold, train_acc = select_best_threshold(train_scores, train_labels)
    test_scores, test_labels, test_skipped = score_pairs(app, test_pairs, embedding_cache, args.embedding_mode)
    test_metrics = evaluate(test_scores, test_labels, threshold)
    elapsed = time.perf_counter() - start

    summary = {
        "task": "LFW face-domain pretrained backbone verification",
        "model_library": "InsightFace",
        "model_name": args.model_name,
        "providers": providers,
        "embedding_mode": args.embedding_mode,
        "det_size": args.det_size,
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
        "LFW Verification - Face-Domain Pretrained Backbone",
        "=" * 60,
        f"Model library: InsightFace",
        f"Model pack: {args.model_name}",
        f"Providers: {providers}",
        f"Embedding mode: {args.embedding_mode}",
        f"Detection size: {args.det_size}",
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
        "Result note:",
        "- This is a face-domain pretrained ArcFace-style backbone evaluation.",
        "- Threshold is selected on pairsDevTrain and reported on pairsDevTest.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
