import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from torchvision.datasets import CelebA


DATA_ROOT = Path("data/raw")
REPORT_DIR = Path("outputs/reports")
OUTPUT_IMAGE_DIR = Path("outputs/images")
DEFAULT_MEDIAPIPE_MODEL = Path("models/checkpoints/face_landmarker.task")
LANDMARK_NAMES = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
MEDIAPIPE_FACE_MESH_INDICES = {
    "left_eye": 468,
    "right_eye": 473,
    "nose": 1,
    "left_mouth": 61,
    "right_mouth": 291,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 100 random CelebA images for face box and landmark quality."
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--mediapipe-model", type=Path, default=DEFAULT_MEDIAPIPE_MODEL)
    parser.add_argument(
        "--skip-mediapipe",
        action="store_true",
        help="Only evaluate annotation-derived boxes and OpenCV detection.",
    )
    parser.add_argument(
        "--mediapipe-crop-mode",
        choices=["full-image", "derived-box"],
        default="full-image",
        help="Run MediaPipe on the full aligned image or on the landmark-derived crop.",
    )
    parser.add_argument(
        "--save-failure-grid",
        action="store_true",
        help="Save a compact grid of OpenCV detection failures for manual inspection.",
    )
    return parser.parse_args()


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, width, height = box.astype(np.float32)
    return np.array([x, y, x + width, y + height], dtype=np.float32)


def xyxy_to_xywh(box: np.ndarray) -> list[float]:
    return [
        float(box[0]),
        float(box[1]),
        float(box[2] - box[0]),
        float(box[3] - box[1]),
    ]


def landmark_bbox(points: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    image_width, image_height = image_size
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    pad_x = width * 0.9
    pad_top = height * 1.35
    pad_bottom = height * 0.9

    x1 = max(0.0, min_x - pad_x)
    y1 = max(0.0, min_y - pad_top)
    x2 = min(float(image_width), max_x + pad_x)
    y2 = min(float(image_height), max_y + pad_bottom)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def expand_box(box: np.ndarray, image_size: tuple[int, int], margin: float = 0.18) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    pad_x = width * margin
    pad_y = height * margin
    return (
        max(0, int(math.floor(x1 - pad_x))),
        max(0, int(math.floor(y1 - pad_y))),
        min(image_width, int(math.ceil(x2 + pad_x))),
        min(image_height, int(math.ceil(y2 + pad_y))),
    )


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def summarize(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "p10": None,
            "p90": None,
            "max": None,
        }
    array = np.array(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": round(float(array.mean()), 4),
        "median": round(float(np.median(array)), 4),
        "min": round(float(array.min()), 4),
        "p10": round(float(np.percentile(array, 10)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "max": round(float(array.max()), 4),
    }


def detect_opencv_faces(image_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return [xywh_to_xyxy(np.array(face, dtype=np.float32)) for face in faces]


def load_mediapipe_landmarker(model_path: Path):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def run_mediapipe_on_crop(
    mp_module,
    landmarker,
    image_rgb: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> dict | None:
    x1, y1, x2, y2 = crop_box
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=crop)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    crop_h, crop_w = crop.shape[:2]
    landmarks = result.face_landmarks[0]
    points = []
    for name in LANDMARK_NAMES:
        landmark = landmarks[MEDIAPIPE_FACE_MESH_INDICES[name]]
        points.append([x1 + landmark.x * crop_w, y1 + landmark.y * crop_h])
    return {
        "points": np.array(points, dtype=np.float32),
        "dense_landmark_count": len(landmarks),
    }


def run_mediapipe_on_image(mp_module, landmarker, image_rgb: np.ndarray) -> dict | None:
    mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    image_h, image_w = image_rgb.shape[:2]
    landmarks = result.face_landmarks[0]
    points = []
    for name in LANDMARK_NAMES:
        landmark = landmarks[MEDIAPIPE_FACE_MESH_INDICES[name]]
        points.append([landmark.x * image_w, landmark.y * image_h])
    return {
        "points": np.array(points, dtype=np.float32),
        "dense_landmark_count": len(landmarks),
    }


def normalized_mean_error(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    interocular = np.linalg.norm(gt_points[0] - gt_points[1])
    if interocular <= 0:
        return float("nan")
    distances = np.linalg.norm(pred_points - gt_points, axis=1)
    return float(distances.mean() / interocular)


def save_failure_grid(dataset: CelebA, failures: list[dict], output_path: Path) -> None:
    if not failures:
        return

    sample = failures[: min(12, len(failures))]
    tile_w, tile_h = 178, 218
    cols = 4
    rows = int(math.ceil(len(sample) / cols))
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 255, dtype=np.uint8)

    for grid_index, failure in enumerate(sample):
        image, target = dataset[failure["dataset_index"]]
        _attr, bbox, landmarks = target
        image_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        gt_box = xywh_to_xyxy(np.asarray(bbox, dtype=np.float32))
        gt_points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        derived_box = landmark_bbox(gt_points, image.size)

        for box, color in [(gt_box, (80, 80, 240)), (derived_box, (32, 163, 158))]:
            x1, y1, x2, y2 = [int(round(value)) for value in box]
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 1)

        row = grid_index // cols
        col = grid_index % cols
        y0 = row * tile_h
        x0 = col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = image_bgr

    cv2.imwrite(str(output_path), canvas)


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    dataset = CelebA(
        root=str(args.data_root),
        split="all",
        target_type=["attr", "bbox", "landmarks"],
        download=False,
    )

    rng = np.random.default_rng(args.seed)
    sample_size = min(args.sample_size, len(dataset))
    indices = rng.choice(len(dataset), size=sample_size, replace=False)

    use_mediapipe = not args.skip_mediapipe and args.mediapipe_model.exists()
    mp_module = None
    landmarker = None
    if use_mediapipe:
        mp_module, landmarker = load_mediapipe_landmarker(args.mediapipe_model)

    derived_box_ious = []
    opencv_best_ious = []
    mediapipe_nmes = []
    mediapipe_dense_counts = []
    detection_failures = []
    landmark_failures = []
    sample_records = []

    for sample_order, dataset_index in enumerate(indices.tolist(), start=1):
        image, target = dataset[int(dataset_index)]
        _attr, bbox, landmarks = target
        image_rgb = np.asarray(image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        gt_box = xywh_to_xyxy(np.asarray(bbox, dtype=np.float32))
        gt_points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        derived_box = landmark_bbox(gt_points, image.size)
        derived_iou = box_iou(gt_box, derived_box)
        derived_box_ious.append(derived_iou)

        detected_boxes = detect_opencv_faces(image_bgr)
        best_detection_iou = 0.0
        if detected_boxes:
            best_detection_iou = max(box_iou(box, derived_box) for box in detected_boxes)
        opencv_best_ious.append(best_detection_iou)
        detection_failed = best_detection_iou < 0.3
        if detection_failed:
            detection_failures.append(
                {
                    "sample_order": sample_order,
                    "dataset_index": int(dataset_index),
                    "best_iou": round(best_detection_iou, 4),
                }
            )

        mediapipe_status = "skipped"
        mediapipe_nme = None
        if use_mediapipe and landmarker is not None:
            if args.mediapipe_crop_mode == "full-image":
                result = run_mediapipe_on_image(mp_module, landmarker, image_rgb)
            else:
                crop_box = expand_box(derived_box, image.size)
                result = run_mediapipe_on_crop(mp_module, landmarker, image_rgb, crop_box)
            if result is None:
                mediapipe_status = "failed"
                landmark_failures.append(
                    {
                        "sample_order": sample_order,
                        "dataset_index": int(dataset_index),
                    }
                )
            else:
                mediapipe_status = "successful"
                mediapipe_dense_counts.append(result["dense_landmark_count"])
                mediapipe_nme = normalized_mean_error(result["points"], gt_points)
                mediapipe_nmes.append(mediapipe_nme)

        sample_records.append(
            {
                "sample_order": sample_order,
                "dataset_index": int(dataset_index),
                "derived_box_vs_original_bbox_iou": round(derived_iou, 4),
                "opencv_detection_best_iou_vs_derived_box": round(best_detection_iou, 4),
                "opencv_detection_count": len(detected_boxes),
                "opencv_detection_failed_iou_lt_0_3": detection_failed,
                "mediapipe_status": mediapipe_status,
                "mediapipe_nme": None if mediapipe_nme is None else round(mediapipe_nme, 4),
            }
        )

    if landmarker is not None:
        landmarker.close()

    failure_grid_path = OUTPUT_IMAGE_DIR / "celeba_100_opencv_detection_failures.jpg"
    if args.save_failure_grid:
        save_failure_grid(dataset, detection_failures, failure_grid_path)

    elapsed = time.perf_counter() - start_time
    detection_failure_count = len(detection_failures)
    landmark_failure_count = len(landmark_failures)
    mediapipe_attempted = use_mediapipe
    mediapipe_scored = len(mediapipe_nmes)

    summary = {
        "task": "CelebA 100-image detection and landmark evaluation",
        "dataset": "CelebA",
        "data_root": str(args.data_root),
        "sample_size": sample_size,
        "seed": args.seed,
        "sampling": "random without replacement",
        "image_type": "CelebA img_align_celeba aligned images",
        "bbox_reference": "CelebA original bbox annotation",
        "landmark_reference": "CelebA five-point aligned landmarks",
        "derived_box_method": (
            "min/max of five landmarks with padding: x=0.9w, top=1.35h, bottom=0.9h"
        ),
        "derived_box_vs_original_bbox_iou": summarize(derived_box_ious),
        "derived_box_iou_ge_0_5": int(sum(value >= 0.5 for value in derived_box_ious)),
        "derived_box_iou_ge_0_3": int(sum(value >= 0.3 for value in derived_box_ious)),
        "opencv_detector": "haarcascade_frontalface_default.xml",
        "opencv_detection_reference": "best IoU against landmark-derived box",
        "opencv_detection_failure_rule": "best IoU < 0.3",
        "opencv_detection_failure_count": detection_failure_count,
        "opencv_detection_failure_rate": round(detection_failure_count / sample_size, 4),
        "opencv_best_iou_vs_derived_box": summarize(opencv_best_ious),
        "mediapipe_attempted": mediapipe_attempted,
        "mediapipe_model": str(args.mediapipe_model),
        "mediapipe_crop_mode": args.mediapipe_crop_mode,
        "mediapipe_landmark_failure_count": landmark_failure_count if mediapipe_attempted else None,
        "mediapipe_landmark_failure_rate": (
            round(landmark_failure_count / sample_size, 4) if mediapipe_attempted else None
        ),
        "mediapipe_nme_reference": "CelebA five landmarks, normalized by inter-ocular distance",
        "mediapipe_nme": summarize(mediapipe_nmes),
        "mediapipe_dense_landmark_counts": sorted(set(mediapipe_dense_counts)),
        "failure_grid": str(failure_grid_path) if args.save_failure_grid else None,
        "elapsed_seconds": round(elapsed, 3),
        "samples": sample_records,
    }

    json_path = REPORT_DIR / "celeba_100_evaluation_result.json"
    txt_path = REPORT_DIR / "celeba_100_evaluation_result.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "CelebA 100-Image Detection And Landmark Evaluation",
        "=" * 60,
        f"Data root: {args.data_root}",
        f"Sample size: {sample_size}",
        f"Random seed: {args.seed}",
        "",
        "Box consistency:",
        "- Reference: CelebA original bbox annotation.",
        "- Compared box: five-point landmark-derived face box.",
        f"- Mean IoU: {summary['derived_box_vs_original_bbox_iou']['mean']}",
        f"- Median IoU: {summary['derived_box_vs_original_bbox_iou']['median']}",
        f"- Min / P10 / P90: {summary['derived_box_vs_original_bbox_iou']['min']} / {summary['derived_box_vs_original_bbox_iou']['p10']} / {summary['derived_box_vs_original_bbox_iou']['p90']}",
        f"- IoU >= 0.5: {summary['derived_box_iou_ge_0_5']} / {sample_size}",
        f"- IoU >= 0.3: {summary['derived_box_iou_ge_0_3']} / {sample_size}",
        "",
        "OpenCV detection batch check:",
        "- Detector: haarcascade_frontalface_default.xml",
        "- Failure rule: best detection IoU against landmark-derived box < 0.3",
        f"- Failure count: {detection_failure_count} / {sample_size}",
        f"- Failure rate: {summary['opencv_detection_failure_rate']}",
        f"- Mean best IoU: {summary['opencv_best_iou_vs_derived_box']['mean']}",
        f"- Median best IoU: {summary['opencv_best_iou_vs_derived_box']['median']}",
        "",
        "MediaPipe landmark check:",
    ]

    if mediapipe_attempted:
        report_lines.extend(
            [
                f"- Model file: {args.mediapipe_model}",
                f"- Crop mode: {args.mediapipe_crop_mode}",
                "- Reference: CelebA five-point landmarks.",
                "- Error metric: NME normalized by inter-ocular distance.",
                f"- Landmark failure count: {landmark_failure_count} / {sample_size}",
                f"- Landmark failure rate: {summary['mediapipe_landmark_failure_rate']}",
                f"- Mean NME: {summary['mediapipe_nme']['mean']}",
                f"- Median NME: {summary['mediapipe_nme']['median']}",
                f"- Min / P10 / P90: {summary['mediapipe_nme']['min']} / {summary['mediapipe_nme']['p10']} / {summary['mediapipe_nme']['p90']}",
                f"- Dense landmark counts: {summary['mediapipe_dense_landmark_counts']}",
            ]
        )
    else:
        report_lines.extend(
            [
                "- Skipped.",
                f"- Reason: model not found or --skip-mediapipe was used ({args.mediapipe_model}).",
            ]
        )

    report_lines.extend(
        [
            "",
            "Interpretation notes:",
            "- LFW is not used for landmark error here because this project does not have LFW landmark ground truth.",
            "- CelebA is the better source for this check because it provides bbox and five-point landmarks.",
            "- The original CelebA bbox and img_align_celeba images can be coordinate-shifted; IoU should be treated as an annotation-consistency check, not a detector benchmark.",
            "- OpenCV Haar is a weak baseline. A production detector should be evaluated separately with RetinaFace, InsightFace detection, or MMDetection on the same sample protocol.",
            "",
            f"JSON report: {json_path}",
            f"Elapsed seconds: {elapsed:.3f}",
            "",
            "Result: Successful",
        ]
    )

    txt_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()


