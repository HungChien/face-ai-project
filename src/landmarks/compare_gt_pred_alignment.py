from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from align_with_landmark_model import ARCFACE_112_TEMPLATE, align_face, landmarks_68_to_five
from train_landmark_regressor import SmallLandmarkCNN, collect_samples, crop_image_and_points, parse_pts
from visualize_landmark_regressor import load_rgb, preprocess, sample_nme


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GT-five-point alignment against predicted-five-point alignment on 300W.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/300W"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/landmark_cnn_300w_aug30_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/landmarks/alignment_compare_300w"))
    parser.add_argument("--grid-output", type=Path, default=Path("outputs/landmarks/alignment_compare_300w/gt_vs_pred_alignment_grid.jpg"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/landmark_300w_alignment_compare_result.txt"))
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--split", choices=["val", "all"], default="val")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


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


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points.astype(np.float32), ones], axis=1)
    return homo @ matrix.T.astype(np.float32)


def template_residual(five_points: np.ndarray, matrix: np.ndarray) -> float:
    aligned_points = transform_points(five_points, matrix)
    return float(np.linalg.norm(aligned_points - ARCFACE_112_TEMPLATE, axis=1).mean())


def five_point_delta(gt_five: np.ndarray, pred_five: np.ndarray) -> float:
    eye_dist = np.linalg.norm(gt_five[0] - gt_five[1])
    return float(np.linalg.norm(pred_five - gt_five, axis=1).mean() / max(eye_dist, 1e-6))


def draw_source(ax, image: np.ndarray, gt_five: np.ndarray, pred_five: np.ndarray, crop_box, title: str) -> None:
    ax.imshow(image)
    if crop_box is not None:
        left, top, right, bottom = crop_box
        ax.add_patch(patches.Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor="#2a9d8f", linewidth=1.0))
    ax.scatter(gt_five[:, 0], gt_five[:, 1], s=28, c="#24b47e", linewidths=0, label="GT 5")
    ax.scatter(pred_five[:, 0], pred_five[:, 1], s=28, c="#e5484d", linewidths=0, label="Pred 5")
    for gt, pred in zip(gt_five, pred_five):
        ax.plot([gt[0], pred[0]], [gt[1], pred[1]], color="#f5a524", linewidth=0.8, alpha=0.75)
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
    val_indices = checkpoint.get("val_indices")

    model = SmallLandmarkCNN(num_points=num_points).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    samples = collect_samples(args.root)
    candidate_indices = list(range(len(samples)))
    if args.split == "val" and val_indices:
        candidate_indices = [int(i) for i in val_indices]
    rng = random.Random(args.seed)
    chosen_indices = sorted(rng.sample(candidate_indices, min(args.num_samples, len(candidate_indices))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(chosen_indices), 3, figsize=(11.2, len(chosen_indices) * 3.2), squeeze=False)
    rows = []

    for row_i, index in enumerate(chosen_indices):
        sample = samples[index]
        image = load_rgb(sample.image_path)
        gt_points = parse_pts(sample.pts_path)
        pred_points, crop_box = predict_landmarks(model, image, gt_points, image_size, crop_to_landmarks, crop_margin, device)

        gt_five = landmarks_68_to_five(gt_points)
        pred_five = landmarks_68_to_five(pred_points)
        gt_aligned, gt_matrix = align_face(image, gt_five)
        pred_aligned, pred_matrix = align_face(image, pred_five)

        pred_nme = sample_nme(pred_points, gt_points)
        delta = five_point_delta(gt_five, pred_five)
        gt_template_error = template_residual(gt_five, gt_matrix)
        pred_template_error = template_residual(pred_five, pred_matrix)

        stem = sample.image_path.stem
        gt_path = args.output_dir / f"{stem}_gt_aligned_112.jpg"
        pred_path = args.output_dir / f"{stem}_pred_aligned_112.jpg"
        cv2.imwrite(str(gt_path), cv2.cvtColor(gt_aligned, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(pred_path), cv2.cvtColor(pred_aligned, cv2.COLOR_RGB2BGR))

        draw_source(axes[row_i][0], image, gt_five, pred_five, crop_box, f"{sample.image_path.name}\nNME={pred_nme:.3f}, 5ptΔ={delta:.3f}")
        axes[row_i][1].imshow(gt_aligned)
        axes[row_i][1].set_title(f"GT five alignment\nres={gt_template_error:.2f}px", fontsize=9)
        axes[row_i][1].axis("off")
        axes[row_i][2].imshow(pred_aligned)
        axes[row_i][2].set_title(f"Pred five alignment\nres={pred_template_error:.2f}px", fontsize=9)
        axes[row_i][2].axis("off")

        rows.append(
            {
                "name": sample.image_path.name,
                "pred_nme": pred_nme,
                "five_delta": delta,
                "gt_template_error": gt_template_error,
                "pred_template_error": pred_template_error,
                "gt_path": gt_path,
                "pred_path": pred_path,
                "gt_five": gt_five,
                "pred_five": pred_five,
            }
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    args.grid_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.grid_output, dpi=180)
    plt.close(fig)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "300W GT Five-Point vs Pred Five-Point Alignment Compare",
        "=" * 64,
        f"Root: {args.root}",
        f"Checkpoint: {args.checkpoint}",
        f"Device: {device}",
        f"Split: {args.split}",
        f"Crop to landmarks: {crop_to_landmarks}",
        f"Crop margin: {crop_margin}",
        f"Samples compared: {len(rows)}",
        f"Mean pred NME: {float(np.mean([r['pred_nme'] for r in rows])):.4f}",
        f"Mean normalized five-point delta: {float(np.mean([r['five_delta'] for r in rows])):.4f}",
        f"Mean GT template residual px: {float(np.mean([r['gt_template_error'] for r in rows])):.4f}",
        f"Mean Pred template residual px: {float(np.mean([r['pred_template_error'] for r in rows])):.4f}",
        f"Grid image: {args.grid_output}",
        f"Output dir: {args.output_dir}",
        "",
        "Sample details:",
    ]
    for row in rows:
        lines.append(
            f"- {row['name']}: pred_nme={row['pred_nme']:.4f}, five_delta={row['five_delta']:.4f}, "
            f"gt_res={row['gt_template_error']:.2f}px, pred_res={row['pred_template_error']:.2f}px"
        )
        lines.append(f"  gt_aligned={row['gt_path']}")
        lines.append(f"  pred_aligned={row['pred_path']}")
        lines.append(f"  gt_five={np.round(row['gt_five'], 2).tolist()}")
        lines.append(f"  pred_five={np.round(row['pred_five'], 2).tolist()}")
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
