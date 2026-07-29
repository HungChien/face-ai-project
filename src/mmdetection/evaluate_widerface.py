from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from mmengine.config import Config


DEFAULT_CONFIG = Path("configs/mmdetection/widerface_retinanet_r50_fpn.py")
DEFAULT_WORK_DIR = Path("outputs/mmdetection_widerface/retinanet_r50_fpn")
DEFAULT_REPORT = Path("outputs/reports/widerface_retinanet_eval_result.txt")
DEFAULT_VIS_DIR = Path("outputs/images/widerface_debug_checkpoint")


def _jsonable_detection(bbox, score: float, label: int) -> dict:
    return {
        "bbox_xyxy": [round(float(item), 2) for item in bbox],
        "score": round(float(score), 4),
        "label": int(label),
    }


def visualize_checkpoint(
    config: Path,
    checkpoint: Path,
    images: list[Path],
    out_dir: Path,
    report: Path,
    score_thr: float,
    max_vis: int,
    device: str,
) -> None:
    from mmdet.apis import inference_detector, init_detector

    out_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    model = init_detector(str(config), str(checkpoint), device=device)
    summaries = []

    for image_path in images:
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        result = inference_detector(model, str(image_path))
        pred = result.pred_instances
        bboxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()

        detections = [
            _jsonable_detection(bbox, score, label)
            for bbox, score, label in zip(bboxes, scores, labels)
            if float(score) >= score_thr
        ]
        drawn_detections = detections[:max_vis]

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        for detection in drawn_detections:
            x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox_xyxy"]]
            score = detection["score"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(
                image,
                f"face {score:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

        output_path = out_dir / f"{image_path.stem}_debug_det.jpg"
        cv2.imwrite(str(output_path), image)
        summaries.append(
            {
                "input_image": str(image_path),
                "visualization": str(output_path),
                "score_threshold": score_thr,
                "detection_count": len(detections),
                "drawn_detection_count": len(drawn_detections),
                "detections": detections,
            }
        )

    json_report = report.with_suffix(".json")
    json_report.write_text(
        json.dumps(
            {
                "module": "WIDER FACE debug checkpoint visualization",
                "config": str(config),
                "checkpoint": str(checkpoint),
                "device": device,
                "score_threshold": score_thr,
                "max_visualized_detections_per_image": max_vis,
                "images": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "WIDER FACE Debug Checkpoint Visualization",
        "=" * 50,
        f"Config: {config}",
        f"Checkpoint: {checkpoint}",
        f"Device: {device}",
        f"Score threshold: {score_thr}",
        f"Max visualized detections per image: {max_vis}",
        f"Output dir: {out_dir}",
        f"JSON report: {json_report}",
        "",
        "Images:",
    ]
    for item in summaries:
        lines.append(
            f"- {item['input_image']} -> {item['visualization']} "
            f"({item['detection_count']} detections, "
            f"{item['drawn_detection_count']} drawn)"
        )
        if not item["detections"]:
            lines.append("  No detections above threshold.")
        else:
            for index, detection in enumerate(item["detections"][:10], start=1):
                lines.append(
                    f"  #{index}: bbox={detection['bbox_xyxy']}, "
                    f"score={detection['score']}"
                )

    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate or visualize the WIDER FACE RetinaNet baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a trained checkpoint. Defaults to work-dir/latest.pth.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--visualize-images",
        type=Path,
        nargs="*",
        default=None,
        help="Run checkpoint inference on images and save bbox visualizations.")
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=DEFAULT_VIS_DIR,
        help="Directory for checkpoint visualization images.")
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--max-vis", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        import torch

        torch.cuda.is_available = lambda: False  # type: ignore[method-assign]

    checkpoint = args.checkpoint or (args.work_dir / "latest.pth")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Train the model first.")

    if args.visualize_images is not None:
        visualize_checkpoint(
            config=args.config,
            checkpoint=checkpoint,
            images=args.visualize_images,
            out_dir=args.vis_dir,
            report=args.report,
            score_thr=args.score_thr,
            max_vis=args.max_vis,
            device=args.device,
        )
        return

    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(args.work_dir)
    cfg.load_from = str(checkpoint)
    if args.device == "cpu":
        cfg.device = "cpu"

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    lines = [
        "WIDER FACE RetinaNet Evaluation",
        "=" * 50,
        f"Config: {args.config}",
        f"Checkpoint: {checkpoint}",
        "",
        "Metrics:",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport saved to: {args.report}")


if __name__ == "__main__":
    main()