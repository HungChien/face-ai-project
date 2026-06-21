import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


LFW_HOME = Path("data/raw/sklearn/lfw_home")
IMAGE_ROOT = LFW_HOME / "lfw_funneled"
OUTPUT_IMAGE_DIR = Path("outputs/images")
REPORT_DIR = Path("outputs/reports")
RECOGNITION_REPORT_TXT = REPORT_DIR / "lfw_recognition_verification_result.txt"
RECOGNITION_REPORT_JSON = REPORT_DIR / "lfw_recognition_verification_result.json"


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
    parser = argparse.ArgumentParser(
        description="Explore LFW pair files and optionally evaluate pretrained face recognition."
    )
    parser.add_argument(
        "--run-recognition",
        action="store_true",
        help="Run InsightFace pretrained embedding verification on LFW pairs.",
    )
    parser.add_argument(
        "--train-pairs",
        default="pairsDevTrain.txt",
        help="Pair file used to select the cosine-similarity threshold.",
    )
    parser.add_argument(
        "--test-pairs",
        default="pairsDevTest.txt",
        help="Pair file used to report final verification accuracy.",
    )
    parser.add_argument("--model-name", default="buffalo_s", help="InsightFace model pack name.")
    parser.add_argument("--det-size", type=int, default=640, help="InsightFace detector input size.")
    parser.add_argument(
        "--max-train-pairs",
        type=int,
        default=0,
        help="Optional debug limit for train pairs. 0 means all pairs.",
    )
    parser.add_argument(
        "--max-test-pairs",
        type=int,
        default=0,
        help="Optional debug limit for test pairs. 0 means all pairs.",
    )
    parser.add_argument(
        "--embedding-mode",
        choices=["aligned", "detected"],
        default="aligned",
        help="Use aligned LFW funneled images directly, or run face detection before embedding.",
    )
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
        return LfwPair(
            "positive",
            name,
            index_a,
            name,
            index_b,
            path_a,
            path_b,
            path_a.exists() and path_b.exists(),
        )

    if len(parts) == 4:
        name_a = parts[0]
        index_a = int(parts[1])
        name_b = parts[2]
        index_b = int(parts[3])
        path_a = lfw_image_path(name_a, index_a)
        path_b = lfw_image_path(name_b, index_b)
        return LfwPair(
            "negative",
            name_a,
            index_a,
            name_b,
            index_b,
            path_a,
            path_b,
            path_a.exists() and path_b.exists(),
        )

    return None


def load_pairs(path: Path) -> tuple[str, list[LfwPair]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].strip()
    pairs = [pair for line in lines[1:] if (pair := parse_pair_line(line)) is not None]
    return header, pairs


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_pair_grid(pairs: list[LfwPair], output_path: Path) -> None:
    positive_pairs = [pair for pair in pairs if pair.pair_type == "positive" and pair.exists][:3]
    negative_pairs = [pair for pair in pairs if pair.pair_type == "negative" and pair.exists][:3]
    selected_pairs = positive_pairs + negative_pairs

    fig, axes = plt.subplots(len(selected_pairs), 2, figsize=(7, 12))
    for row_index, pair in enumerate(selected_pairs):
        left_axis = axes[row_index, 0]
        right_axis = axes[row_index, 1]
        left_axis.imshow(read_image(pair.path_a))
        right_axis.imshow(read_image(pair.path_b))
        left_axis.axis("off")
        right_axis.axis("off")
        label = "same identity" if pair.pair_type == "positive" else "different identities"
        left_axis.set_title(f"{pair.name_a} #{pair.index_a}", fontsize=8)
        right_axis.set_title(f"{pair.name_b} #{pair.index_b}\n{label}", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_opencv_pair_grid(pairs: list[LfwPair], output_path: Path) -> None:
    positive_pairs = [pair for pair in pairs if pair.pair_type == "positive" and pair.exists][:3]
    negative_pairs = [pair for pair in pairs if pair.pair_type == "negative" and pair.exists][:3]
    selected_pairs = positive_pairs + negative_pairs
    tile = 150
    label_h = 34
    row_h = tile + label_h
    canvas = np.full((len(selected_pairs) * row_h, tile * 2, 3), 255, dtype=np.uint8)

    for row_index, pair in enumerate(selected_pairs):
        y0 = row_index * row_h
        for col_index, path in enumerate([pair.path_a, pair.path_b]):
            image = cv2.imread(str(path))
            image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
            x0 = col_index * tile
            canvas[y0 : y0 + tile, x0 : x0 + tile] = image

        color = (40, 150, 40) if pair.pair_type == "positive" else (40, 40, 220)
        label = "positive same" if pair.pair_type == "positive" else "negative different"
        cv2.putText(canvas, label, (6, y0 + tile + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.line(canvas, (tile, y0), (tile, y0 + tile), color, 2)

    cv2.imwrite(str(output_path), canvas)


def summarize_pairs(file_name: str, header: str, pairs: list[LfwPair]) -> dict:
    positive_count = sum(pair.pair_type == "positive" for pair in pairs)
    negative_count = sum(pair.pair_type == "negative" for pair in pairs)
    missing_count = sum(not pair.exists for pair in pairs)
    identities = {pair.name_a for pair in pairs} | {pair.name_b for pair in pairs}

    return {
        "file": file_name,
        "header": header,
        "pair_count": len(pairs),
        "positive_pairs": positive_count,
        "negative_pairs": negative_count,
        "unique_identities": len(identities),
        "missing_image_pairs": missing_count,
    }


def pair_label(pair: LfwPair) -> int:
    return 1 if pair.pair_type == "positive" else 0


def load_insightface_app(model_name: str, det_size: int):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def extract_embedding(app, image_path: Path, embedding_mode: str) -> np.ndarray | None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None

    if embedding_mode == "aligned":
        recognizer = app.models["recognition"]
        embedding = np.asarray(recognizer.get_feat(image_bgr)[0], dtype=np.float32)
    else:
        faces = app.get(image_bgr)
        if not faces:
            return None
        face = max(
            faces,
            key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])),
        )
        embedding = np.asarray(face.normed_embedding, dtype=np.float32)

    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm


def pair_similarity(app, pair: LfwPair, embedding_cache: dict[str, np.ndarray | None], embedding_mode: str) -> float | None:
    key_a = str(pair.path_a)
    key_b = str(pair.path_b)
    if key_a not in embedding_cache:
        embedding_cache[key_a] = extract_embedding(app, pair.path_a, embedding_mode)
    if key_b not in embedding_cache:
        embedding_cache[key_b] = extract_embedding(app, pair.path_b, embedding_mode)

    emb_a = embedding_cache[key_a]
    emb_b = embedding_cache[key_b]
    if emb_a is None or emb_b is None:
        return None
    return float(np.dot(emb_a, emb_b))


def score_pairs(app, pairs: list[LfwPair], embedding_cache: dict[str, np.ndarray | None], embedding_mode: str) -> tuple[np.ndarray, np.ndarray, int]:
    scores: list[float] = []
    labels: list[int] = []
    skipped = 0
    valid_pairs = [pair for pair in pairs if pair.exists]

    for index, pair in enumerate(valid_pairs, start=1):
        score = pair_similarity(app, pair, embedding_cache, embedding_mode)
        if score is None:
            skipped += 1
            continue
        scores.append(score)
        labels.append(pair_label(pair))
        if index % 100 == 0:
            print(f"Scored {index}/{len(valid_pairs)} LFW pairs...")

    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32), skipped


def select_best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(scores) == 0:
        return 0.0, 0.0
    thresholds = np.linspace(float(scores.min()), float(scores.max()), 400)
    best_threshold = float(thresholds[0])
    best_accuracy = -1.0

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(np.int32)
        accuracy = float((predictions == labels).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def evaluate_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int32)
    accuracy = float((predictions == labels).mean()) if len(labels) else 0.0
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    return {
        "accuracy": round(accuracy, 4),
        "correct": int((predictions == labels).sum()),
        "total": int(len(labels)),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def run_lfw_recognition_verification(args: argparse.Namespace) -> None:
    train_header, train_pairs = load_pairs(LFW_HOME / args.train_pairs)
    test_header, test_pairs = load_pairs(LFW_HOME / args.test_pairs)
    if args.max_train_pairs > 0:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_test_pairs > 0:
        test_pairs = test_pairs[: args.max_test_pairs]

    start_time = time.perf_counter()
    app = load_insightface_app(args.model_name, args.det_size)
    embedding_cache: dict[str, np.ndarray | None] = {}

    train_scores, train_labels, train_skipped = score_pairs(app, train_pairs, embedding_cache, args.embedding_mode)
    threshold, train_accuracy = select_best_threshold(train_scores, train_labels)
    test_scores, test_labels, test_skipped = score_pairs(app, test_pairs, embedding_cache, args.embedding_mode)
    test_metrics = evaluate_at_threshold(test_scores, test_labels, threshold)
    elapsed = time.perf_counter() - start_time

    summary = {
        "task": "LFW pretrained face recognition verification",
        "model_library": "InsightFace",
        "model_name": args.model_name,
        "provider": "CPUExecutionProvider",
        "embedding_mode": args.embedding_mode,
        "train_pair_file": args.train_pairs,
        "train_pair_header": train_header,
        "test_pair_file": args.test_pairs,
        "test_pair_header": test_header,
        "threshold_selected_on_train": round(threshold, 4),
        "train_accuracy_at_selected_threshold": round(train_accuracy, 4),
        "test_metrics": test_metrics,
        "train_pairs_scored": int(len(train_labels)),
        "test_pairs_scored": int(len(test_labels)),
        "train_pairs_skipped": int(train_skipped),
        "test_pairs_skipped": int(test_skipped),
        "unique_embeddings_cached": len(embedding_cache),
        "elapsed_seconds": round(elapsed, 3),
        "preprocessing": [
            "Read LFW funneled images from the official pair files.",
            "Use LFW funneled aligned crops directly for embedding by default.",
            "Use the pretrained recognition model to extract normalized embeddings.",
            "Compute cosine similarity for each pair.",
            "Select threshold on pairsDevTrain and report accuracy on pairsDevTest.",
        ],
    }

    report_lines = [
        "LFW Pretrained Face Recognition Verification",
        "=" * 50,
        "Model library: InsightFace",
        f"Model pack: {args.model_name}",
        "Provider: CPUExecutionProvider",
        f"Embedding mode: {args.embedding_mode}",
        f"Train pair file: {args.train_pairs} ({train_header})",
        f"Test pair file: {args.test_pairs} ({test_header})",
        f"Train pairs scored: {len(train_labels)}",
        f"Test pairs scored: {len(test_labels)}",
        f"Train pairs skipped: {train_skipped}",
        f"Test pairs skipped: {test_skipped}",
        f"Selected threshold: {threshold:.4f}",
        f"Train accuracy at selected threshold: {train_accuracy:.4f}",
        f"Test accuracy: {test_metrics['accuracy']:.4f}",
        f"Correct / total: {test_metrics['correct']} / {test_metrics['total']}",
        f"TP: {test_metrics['true_positive']}",
        f"TN: {test_metrics['true_negative']}",
        f"FP: {test_metrics['false_positive']}",
        f"FN: {test_metrics['false_negative']}",
        f"Unique embeddings cached: {len(embedding_cache)}",
        f"Elapsed seconds: {elapsed:.3f}",
        "",
        "Preprocessing and evaluation:",
        "- Read LFW funneled images from the official pair files.",
        "- Use LFW funneled aligned crops directly for embedding by default.",
        "- Extract normalized embeddings using the pretrained recognition model.",
        "- Compute cosine similarity for each pair.",
        "- Select threshold on pairsDevTrain and report accuracy on pairsDevTest.",
        "",
        "Result: Successful",
    ]

    RECOGNITION_REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")
    RECOGNITION_REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nRecognition report saved to: {RECOGNITION_REPORT_TXT}")
    print(f"Recognition JSON saved to: {RECOGNITION_REPORT_JSON}")


def main() -> None:
    args = parse_args()
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pair_files = ["pairs.txt", "pairsDevTrain.txt", "pairsDevTest.txt"]
    summaries = []
    all_pairs: list[LfwPair] = []

    for file_name in pair_files:
        path = LFW_HOME / file_name
        header, pairs = load_pairs(path)
        summaries.append(summarize_pairs(file_name, header, pairs))
        all_pairs.extend(pairs)

    pair_grid_path = OUTPUT_IMAGE_DIR / "lfw_pair_examples.jpg"
    opencv_pair_grid_path = OUTPUT_IMAGE_DIR / "lfw_opencv_pair_examples.jpg"
    report_path = REPORT_DIR / "lfw_pair_exploration_result.txt"
    json_path = REPORT_DIR / "lfw_pair_exploration_result.json"

    save_pair_grid(all_pairs, pair_grid_path)
    save_opencv_pair_grid(all_pairs, opencv_pair_grid_path)

    report_lines = [
        "LFW Pair Structure Exploration",
        "=" * 50,
        f"LFW home: {LFW_HOME}",
        f"Image root: {IMAGE_ROOT}",
        "",
        "Pair format:",
        "- Positive pair line: <name> <image_index_1> <image_index_2>",
        "- Negative pair line: <name_1> <image_index_1> <name_2> <image_index_2>",
        "- Image path rule: <name>/<name>_<4-digit-index>.jpg",
        "",
        "Pair files:",
    ]

    for summary in summaries:
        report_lines.extend(
            [
                f"- {summary['file']}:",
                f"  header: {summary['header']}",
                f"  pair count: {summary['pair_count']}",
                f"  positive pairs: {summary['positive_pairs']}",
                f"  negative pairs: {summary['negative_pairs']}",
                f"  unique identities: {summary['unique_identities']}",
                f"  missing image pairs: {summary['missing_image_pairs']}",
            ]
        )

    report_lines.extend(
        [
            "",
            "Visualization tools:",
            "- Matplotlib: pair image grid with titles.",
            "- OpenCV: pair image grid with positive/negative labels and separators.",
            "",
            "Annotation standard:",
            "- LFW does not provide dense landmark annotations in pairs.txt.",
            "- The pair files define a verification label: same identity or different identities.",
            "- Identity names are directory names; image indices map to zero-padded file names.",
            "",
            "Preprocessing method for face verification:",
            "- Detect face region if raw images are used.",
            "- Align faces using eye or five-point landmarks before embedding extraction.",
            "- Resize aligned crops to the recognition model input size, commonly 112x112.",
            "- Normalize pixel values according to the backbone model requirement.",
            "- Build positive pairs from the same identity and negative pairs from different identities.",
            "- Keep train/test pair files separate to avoid evaluation leakage.",
            "",
            f"Matplotlib pair examples: {pair_grid_path}",
            f"OpenCV pair examples: {opencv_pair_grid_path}",
            "",
            "Result: Successful",
        ]
    )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "summaries": summaries,
                "matplotlib_pair_examples": str(pair_grid_path),
                "opencv_pair_examples": str(opencv_pair_grid_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print(f"\nJSON report saved to: {json_path}")

    if args.run_recognition:
        print("\n" + "-" * 50)
        run_lfw_recognition_verification(args)


if __name__ == "__main__":
    main()

