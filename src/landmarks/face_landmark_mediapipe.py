import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python import vision


DEFAULT_IMAGE = Path("data/samples/face_test.jpg")
DEFAULT_DETECTION_JSON = Path("outputs/mmdetection_face_detection/preds/face_test.json")
DEFAULT_MODEL = Path("models/checkpoints/face_landmarker.task")

FACE_MESH_INDICES = {
    "left_eye": 33,
    "right_eye": 263,
    "nose_tip": 1,
    "left_mouth": 61,
    "right_mouth": 291,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe face landmark detection on MMDetection face boxes."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path.")
    parser.add_argument(
        "--detections",
        type=Path,
        default=DEFAULT_DETECTION_JSON,
        help="MMDetection prediction JSON path.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.25,
        help="Minimum MMDetection face score used before landmark detection.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.18,
        help="Relative padding added around the detection box before landmark inference.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="MediaPipe FaceLandmarker .task model path.",
    )
    return parser.parse_args()


def load_detection_boxes(prediction_path: Path, score_threshold: float) -> list[dict]:
    data = json.loads(prediction_path.read_text(encoding="utf-8"))
    bboxes = data.get("bboxes", [])
    scores = data.get("scores", [])
    labels = data.get("labels", [])

    detections = []
    for bbox, score, label in zip(bboxes, scores, labels):
        if float(score) < score_threshold:
            continue
        detections.append(
            {
                "bbox_xyxy": [float(value) for value in bbox],
                "score": float(score),
                "label": int(label),
            }
        )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections


def expand_box(
    bbox: list[float],
    image_width: int,
    image_height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    pad_x = width * margin
    pad_y = height * margin
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(image_width, int(round(x2 + pad_x)))
    y2 = min(image_height, int(round(y2 + pad_y)))
    return x1, y1, x2, y2


def draw_landmarks(
    image: np.ndarray,
    dense_points: list[tuple[int, int]],
    five_points: dict[str, tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (32, 163, 158), 4)

    for point in dense_points:
        cv2.circle(image, point, 1, (60, 180, 255), -1)

    for name, point in five_points.items():
        cv2.circle(image, point, 10, (80, 80, 255), -1)
        cv2.putText(
            image,
            name,
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def run_landmarks(
    image_bgr: np.ndarray,
    detections: list[dict],
    margin: float,
    model_path: Path,
) -> list[dict]:
    image_height, image_width = image_bgr.shape[:2]
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )

    landmark_results = []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for detection_index, detection in enumerate(detections, start=1):
            crop_box = expand_box(
                detection["bbox_xyxy"],
                image_width=image_width,
                image_height=image_height,
                margin=margin,
            )
            x1, y1, x2, y2 = crop_box
            crop_bgr = image_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            result = landmarker.detect(mp_image)
            if not result.face_landmarks:
                landmark_results.append(
                    {
                        "detection_index": detection_index,
                        "detection_score": round(detection["score"], 4),
                        "bbox_xyxy": [round(value, 2) for value in detection["bbox_xyxy"]],
                        "crop_box_xyxy": list(crop_box),
                        "landmark_count": 0,
                        "status": "no_landmarks_detected",
                    }
                )
                continue

            face_landmarks = result.face_landmarks[0]
            crop_h, crop_w = crop_bgr.shape[:2]
            dense_points = []
            for landmark in face_landmarks:
                px = int(round(x1 + landmark.x * crop_w))
                py = int(round(y1 + landmark.y * crop_h))
                dense_points.append((px, py))

            five_points = {}
            for name, landmark_index in FACE_MESH_INDICES.items():
                landmark = face_landmarks[landmark_index]
                px = int(round(x1 + landmark.x * crop_w))
                py = int(round(y1 + landmark.y * crop_h))
                five_points[name] = (px, py)

            landmark_results.append(
                {
                    "detection_index": detection_index,
                    "detection_score": round(detection["score"], 4),
                    "bbox_xyxy": [round(value, 2) for value in detection["bbox_xyxy"]],
                    "crop_box_xyxy": list(crop_box),
                    "landmark_count": len(dense_points),
                    "five_point_landmarks": {
                        name: [int(point[0]), int(point[1])]
                        for name, point in five_points.items()
                    },
                    "status": "successful",
                }
            )
            draw_landmarks(image_bgr, dense_points, five_points, crop_box)
    return landmark_results


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.detections.exists():
        raise FileNotFoundError(f"MMDetection prediction JSON not found: {args.detections}")
    if not args.model.exists():
        raise FileNotFoundError(f"MediaPipe FaceLandmarker model not found: {args.model}")

    output_image_dir = Path("outputs/landmarks")
    report_dir = Path("outputs/reports")
    output_image_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    detections = load_detection_boxes(args.detections, args.score_thr)
    start_time = time.perf_counter()
    landmark_results = run_landmarks(image_bgr, detections, args.margin, args.model)
    elapsed = time.perf_counter() - start_time

    output_image_path = output_image_dir / "face_test_mediapipe_landmarks.jpg"
    report_txt_path = report_dir / "face_landmark_mediapipe_result.txt"
    report_json_path = report_dir / "face_landmark_mediapipe_result.json"
    cv2.imwrite(str(output_image_path), image_bgr)

    successful_faces = sum(item["status"] == "successful" for item in landmark_results)
    summary = {
        "task": "Face landmark detection",
        "model": "MediaPipe FaceLandmarker",
        "model_path": str(args.model),
        "input_image": str(args.image),
        "mmdetection_predictions": str(args.detections),
        "score_threshold": args.score_thr,
        "crop_margin": args.margin,
        "detections_used": len(detections),
        "successful_faces": successful_faces,
        "elapsed_seconds": round(elapsed, 3),
        "output_image": str(output_image_path),
        "results": landmark_results,
        "note": (
            "The MMDetection face box is used as the region of interest. "
            "MediaPipe FaceLandmarker predicts dense landmarks inside the crop, "
            "then coordinates are mapped back to the original image."
        ),
    }
    report_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "Face Landmark Detection Result",
        "=" * 50,
        "Model: MediaPipe FaceLandmarker",
        f"Model file: {args.model}",
        f"Input image: {args.image}",
        f"MMDetection prediction JSON: {args.detections}",
        f"Score threshold: {args.score_thr}",
        f"Crop margin: {args.margin}",
        f"Detections used: {len(detections)}",
        f"Successful faces: {successful_faces}",
        f"Elapsed seconds: {elapsed:.3f}",
        f"Output image: {output_image_path}",
        "",
        "Five-point landmarks:",
    ]
    for item in landmark_results:
        report_lines.append(f"- Detection #{item['detection_index']}: {item['status']}")
        if item["status"] == "successful":
            for name, point in item["five_point_landmarks"].items():
                report_lines.append(f"  - {name}: {point}")
            report_lines.append(f"  - dense landmark count: {item['landmark_count']}")

    report_lines.extend(
        [
            "",
            "Pipeline:",
            "- Load MMDetection face detection result.",
            "- Expand the face box and crop the region of interest.",
            "- Run MediaPipe FaceLandmarker on the crop.",
            "- Map crop-level landmarks back to original image coordinates.",
            "- Draw dense landmarks and five key points on the test image.",
            "",
            "Result: Successful" if successful_faces else "Result: Failed",
        ]
    )
    report_txt_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nJSON report saved to: {report_json_path}")


if __name__ == "__main__":
    main()

