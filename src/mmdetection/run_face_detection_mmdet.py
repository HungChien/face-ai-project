import argparse
import json
import time
from pathlib import Path

import mmcv
import mmdet
import mmengine
import numpy as np
import torch
from mmdet.apis import DetInferencer


DEFAULT_IMAGE = Path("data/samples/face_test.jpg")
DEFAULT_MODEL = "grounding_dino_swin-t_pretrain_obj365_goldg"
DEFAULT_TEXT_PROMPT = "face"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMDetection face detection on a provided image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Input image path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="MMDetection model alias or config path.",
    )
    parser.add_argument(
        "--text-prompt",
        default=DEFAULT_TEXT_PROMPT,
        help="Open-vocabulary prompt used for face detection.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.25,
        help="Score threshold for visualization and counting.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device. Use cpu for the ml-mmdet environment.",
    )
    return parser.parse_args()


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value


def extract_detections(result: dict, score_threshold: float) -> list[dict]:
    predictions = result.get("predictions", [])
    if not predictions:
        return []

    first_prediction = predictions[0]
    bboxes = first_prediction.get("bboxes", [])
    scores = first_prediction.get("scores", [])
    labels = first_prediction.get("labels", [])
    label_names = first_prediction.get("label_names", [])

    detections = []
    for index, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
        if score < score_threshold:
            continue
        label_name = ""
        if index < len(label_names):
            label_name = label_names[index]
        detections.append(
            {
                "bbox_xyxy": [round(float(item), 2) for item in bbox],
                "score": round(float(score), 4),
                "label": int(label),
                "label_name": label_name,
            }
        )
    return detections


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    output_dir = Path("outputs/mmdetection_face_detection")
    report_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    inferencer = DetInferencer(model=args.model, device=args.device)
    result = inferencer(
        str(args.image),
        out_dir=str(output_dir),
        pred_score_thr=args.score_thr,
        no_save_pred=False,
        no_save_vis=False,
        texts=args.text_prompt,
        custom_entities=True,
    )
    elapsed = time.perf_counter() - start_time

    detections = extract_detections(result, args.score_thr)
    visual_path = output_dir / "vis" / args.image.name
    prediction_path = output_dir / "preds" / f"{args.image.stem}.json"

    summary = {
        "task": "MMDetection face detection",
        "model": args.model,
        "model_role": "MMDetection open-vocabulary detector with face text prompt",
        "text_prompt": args.text_prompt,
        "device": args.device,
        "input_image": str(args.image),
        "score_threshold": args.score_thr,
        "elapsed_seconds": round(elapsed, 3),
        "detection_count": len(detections),
        "detections": detections,
        "visualization_path": str(visual_path),
        "prediction_path": str(prediction_path),
        "wider_face_note": (
            "MMDetection includes WIDER FACE configs, but the official config "
            "directory does not provide an indexed pretrained checkpoint. "
            "GroundingDINO is used here to complete MMDetection-based face "
            "detection on the provided image."
        ),
        "torch_version": torch.__version__,
        "mmcv_version": mmcv.__version__,
        "mmengine_version": mmengine.__version__,
        "mmdet_version": mmdet.__version__,
    }

    report_json_path = report_dir / "mmdetection_face_detection_result.json"
    report_txt_path = report_dir / "mmdetection_face_detection_result.txt"
    report_json_path.write_text(
        json.dumps(to_jsonable(summary), indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "MMDetection Face Detection Result",
        "=" * 50,
        f"Model: {args.model}",
        "Model role: MMDetection open-vocabulary detector with face prompt",
        f"Text prompt: {args.text_prompt}",
        f"Device: {args.device}",
        f"Input image: {args.image}",
        f"Score threshold: {args.score_thr}",
        f"Elapsed seconds: {elapsed:.3f}",
        f"Detected faces above threshold: {len(detections)}",
        f"Visualization: {visual_path}",
        f"Prediction JSON: {prediction_path}",
        "",
        "Detections:",
    ]
    if detections:
        for index, detection in enumerate(detections, start=1):
            report_lines.append(
                f"- #{index}: bbox={detection['bbox_xyxy']}, "
                f"score={detection['score']}, label={detection['label_name']}"
            )
    else:
        report_lines.append("- No face detected above the score threshold.")

    report_lines.extend(
        [
            "",
            "WIDER FACE config note:",
            "- MMDetection includes WIDER FACE face-detection configs.",
            "- The official config directory does not provide an indexed pretrained checkpoint.",
            "- This run uses GroundingDINO in MMDetection with prompt 'face' to complete image-level face detection.",
            "",
            f"PyTorch version: {torch.__version__}",
            f"MMCV version: {mmcv.__version__}",
            f"MMEngine version: {mmengine.__version__}",
            f"MMDetection version: {mmdet.__version__}",
            "",
            "Result: Successful",
        ]
    )
    report_txt_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nReport saved to: {report_txt_path}")
    print(f"JSON report saved to: {report_json_path}")


if __name__ == "__main__":
    main()
