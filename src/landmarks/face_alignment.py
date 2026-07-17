from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGE = Path("data/samples/face_test.jpg")
DEFAULT_LANDMARK_JSON = Path("outputs/reports/face_landmark_mediapipe_result.json")
DEFAULT_OUTPUT_DIR = Path("outputs/landmarks/alignment")

FIVE_POINT_ORDER = ["left_eye", "right_eye", "nose_tip", "left_mouth", "right_mouth"]

# ArcFace-style 112x112 five-point template.
TEMPLATE_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align a face using five facial landmarks.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--landmarks-json", type=Path, default=DEFAULT_LANDMARK_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-size", type=int, default=112)
    return parser.parse_args()


def load_five_points(landmark_json: Path) -> np.ndarray:
    data = json.loads(landmark_json.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No landmark results found in {landmark_json}")

    successful = [item for item in results if item.get("status") == "successful"]
    if not successful:
        raise ValueError(f"No successful landmark result found in {landmark_json}")

    points_dict = successful[0]["five_point_landmarks"]
    points = np.asarray([points_dict[name] for name in FIVE_POINT_ORDER], dtype=np.float32)
    return points


def estimate_alignment(src_points: np.ndarray, output_size: int) -> np.ndarray:
    dst = TEMPLATE_112.copy()
    if output_size != 112:
        dst *= output_size / 112.0
    matrix, inliers = cv2.estimateAffinePartial2D(src_points, dst, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("Failed to estimate affine transform from landmarks.")
    return matrix.astype(np.float32)


def draw_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    colors = [(0, 220, 255), (0, 220, 255), (80, 255, 80), (255, 160, 80), (255, 160, 80)]
    for name, point, color in zip(FIVE_POINT_ORDER, points, colors):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(canvas, (x, y), 8, color, -1)
        cv2.putText(canvas, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return canvas


def make_side_by_side(original: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    preview_h = 360
    scale = preview_h / original.shape[0]
    preview_w = max(1, int(round(original.shape[1] * scale)))
    original_preview = cv2.resize(original, (preview_w, preview_h))
    aligned_preview = cv2.resize(aligned, (preview_h, preview_h))
    gap = np.full((preview_h, 24, 3), 255, dtype=np.uint8)
    return np.concatenate([original_preview, gap, aligned_preview], axis=1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    src_points = load_five_points(args.landmarks_json)
    matrix = estimate_alignment(src_points, args.output_size)
    aligned = cv2.warpAffine(image, matrix, (args.output_size, args.output_size), flags=cv2.INTER_LINEAR)

    original_with_points = draw_points(image, src_points)
    comparison = make_side_by_side(original_with_points, aligned)

    aligned_path = args.output_dir / "face_test_aligned_112.jpg"
    comparison_path = args.output_dir / "face_test_alignment_comparison.jpg"
    report_path = Path("outputs/reports/face_alignment_result.txt")
    json_path = Path("outputs/reports/face_alignment_result.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(aligned_path), aligned)
    cv2.imwrite(str(comparison_path), comparison)

    summary = {
        "task": "five-point face alignment",
        "image": str(args.image),
        "landmarks_json": str(args.landmarks_json),
        "point_order": FIVE_POINT_ORDER,
        "source_points": src_points.round(2).tolist(),
        "template_points": TEMPLATE_112.round(2).tolist(),
        "affine_matrix": matrix.round(6).tolist(),
        "output_size": [args.output_size, args.output_size],
        "aligned_image": str(aligned_path),
        "comparison_image": str(comparison_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "Five-point Face Alignment Result",
        "=" * 50,
        f"Input image: {args.image}",
        f"Landmark JSON: {args.landmarks_json}",
        f"Output aligned image: {aligned_path}",
        f"Comparison image: {comparison_path}",
        f"Output size: {args.output_size}x{args.output_size}",
        "",
        "Source points:",
    ]
    for name, point in zip(FIVE_POINT_ORDER, src_points):
        lines.append(f"- {name}: [{point[0]:.1f}, {point[1]:.1f}]")
    lines.extend(["", "Affine matrix:", str(matrix)])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()