from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_landmark_regressor import PtsLandmarkDataset, SmallLandmarkCNN, crop_image_and_points, parse_pts
from visualize_landmark_regressor import load_rgb, preprocess, sample_nme


ARCFACE_112_TEMPLATE = np.asarray(
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
    parser = argparse.ArgumentParser(description="Predict 68 landmarks, derive five points, and align faces to 112x112.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/300W"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/landmark_cnn_300w_cropped.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/landmarks/alignment_300w_model"))
    parser.add_argument("--grid-output", type=Path, default=Path("outputs/landmarks/alignment_300w_model/landmark_model_alignment_grid.jpg"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/landmark_300w_alignment_result.txt"))
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def landmarks_68_to_five(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 68:
        raise ValueError("Expected 68 landmarks to derive five alignment points.")
    left_eye = points[36:42].mean(axis=0)
    right_eye = points[42:48].mean(axis=0)
    nose_tip = points[30]
    left_mouth = points[48]
    right_mouth = points[54]
    return np.asarray([left_eye, right_eye, nose_tip, left_mouth, right_mouth], dtype=np.float32)


def predict_landmarks(
    model: SmallLandmarkCNN,
    image: np.ndarray,
    gt_points: np.ndarray,
    image_size: int,
    crop_to_landmarks: bool,
    crop_margin: float,
    device: str,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    model_image = image
    crop_box = None
    if crop_to_landmarks:
        model_image, _crop_gt_points, crop_box = crop_image_and_points(image, gt_points, margin=crop_margin)

    h, w = model_image.shape[:2]
    tensor = preprocess(model_image, image_size).to(device)
    with torch.no_grad():
        pred_norm = model(tensor).cpu().numpy().reshape(-1, 2)
    pred_points = pred_norm.copy()
    pred_points[:, 0] *= max(w - 1, 1)
    pred_points[:, 1] *= max(h - 1, 1)
    if crop_box is not None:
        left, top, _right, _bottom = crop_box
        pred_points[:, 0] += left
        pred_points[:, 1] += top
    return pred_points.astype(np.float32), crop_box


def align_face(image: np.ndarray, five_points: np.ndarray, output_size: int = 112) -> tuple[np.ndarray, np.ndarray]:
    matrix, _inliers = cv2.estimateAffinePartial2D(five_points, ARCFACE_112_TEMPLATE, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("Failed to estimate affine alignment matrix.")
    aligned = cv2.warpAffine(image, matrix, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return aligned, matrix


def draw_source_panel(ax, image: np.ndarray, pred_points: np.ndarray, five_points: np.ndarray, crop_box, title: str) -> None:
    ax.imshow(image)
    if crop_box is not None:
        left, top, right, bottom = crop_box
        ax.add_patch(patches.Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor="#2a9d8f", linewidth=1.2))
    ax.scatter(pred_points[:, 0], pred_points[:, 1], s=5, c="#e5484d", linewidths=0, label="68 pred")
    ax.scatter(five_points[:, 0], five_points[:, 1], s=24, c="#2f6fed", linewidths=0, label="5 pts")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = int(checkpoint.get("image_size", 128))
    num_points = int(checkpoint.get("num_points", 68))
    crop_to_landmarks = bool(checkpoint.get("crop_to_landmarks", False))
    crop_margin = float(checkpoint.get("crop_margin", 0.25))

    model = SmallLandmarkCNN(num_points=num_points).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = PtsLandmarkDataset(args.root, image_size=image_size, crop_to_landmarks=crop_to_landmarks, crop_margin=crop_margin)
    rng = random.Random(args.seed)
    sample_indices = sorted(rng.sample(range(len(dataset.samples)), min(args.num_samples, len(dataset.samples))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = len(sample_indices)
    fig, axes = plt.subplots(rows, 2, figsize=(8.0, rows * 3.1), squeeze=False)
    report_rows = []

    for row_index, sample_index in enumerate(sample_indices):
        sample = dataset.samples[sample_index]
        image = load_rgb(sample.image_path)
        gt_points = parse_pts(sample.pts_path)
        pred_points, crop_box = predict_landmarks(model, image, gt_points, image_size, crop_to_landmarks, crop_margin, device)
        five_points = landmarks_68_to_five(pred_points)
        aligned, matrix = align_face(image, five_points)
        nme = sample_nme(pred_points, gt_points)

        stem = sample.image_path.stem
        aligned_path = args.output_dir / f"{stem}_aligned_112.jpg"
        cv2.imwrite(str(aligned_path), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))

        draw_source_panel(axes[row_index][0], image, pred_points, five_points, crop_box, f"{sample.image_path.name}\nNME={nme:.3f}")
        axes[row_index][1].imshow(aligned)
        axes[row_index][1].set_title("Aligned 112x112", fontsize=9)
        axes[row_index][1].axis("off")

        report_rows.append((sample.image_path.name, nme, aligned_path, matrix, five_points))

    fig.tight_layout()
    args.grid_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.grid_output, dpi=180)
    plt.close(fig)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "300W Model Landmark Alignment Result",
        "=" * 50,
        f"Root: {args.root}",
        f"Checkpoint: {args.checkpoint}",
        f"Device: {device}",
        f"Crop to landmarks: {crop_to_landmarks}",
        f"Crop margin: {crop_margin}",
        f"Samples aligned: {len(report_rows)}",
        f"Mean NME: {float(np.mean([row[1] for row in report_rows])):.4f}",
        f"Grid image: {args.grid_output}",
        f"Output dir: {args.output_dir}",
        "",
        "Sample details:",
    ]
    for name, nme, aligned_path, matrix, five_points in report_rows:
        lines.append(f"- {name}: NME={nme:.4f}, aligned={aligned_path}")
        lines.append(f"  five_points={np.round(five_points, 2).tolist()}")
        lines.append(f"  affine={np.round(matrix, 5).tolist()}")
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
