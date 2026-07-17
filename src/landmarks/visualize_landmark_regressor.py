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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize 300W landmark regressor predictions against .pts ground truth.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/300W"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/landmark_cnn_300w.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/landmarks/landmark_cnn_300w_predictions.jpg"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/landmark_300w_visualization_result.txt"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess(image: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32) / 255.0
    resized = np.transpose(resized, (2, 0, 1))
    return torch.from_numpy(resized).unsqueeze(0)


def sample_nme(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    errors = np.linalg.norm(pred_points - gt_points, axis=1).mean()
    if gt_points.shape[0] >= 46:
        norm = np.linalg.norm(gt_points[36] - gt_points[45])
    else:
        norm = max(gt_points[:, 0].max() - gt_points[:, 0].min(), 1.0)
    return float(errors / max(norm, 1e-6))


def draw_points(
    ax,
    image: np.ndarray,
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    crop_box: tuple[int, int, int, int] | None,
    title: str,
) -> None:
    ax.imshow(image)
    if crop_box is not None:
        left, top, right, bottom = crop_box
        rect = patches.Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor="#2a9d8f", linewidth=1.2)
        ax.add_patch(rect)
    ax.scatter(gt_points[:, 0], gt_points[:, 1], s=8, c="#24b47e", label="GT", linewidths=0)
    ax.scatter(pred_points[:, 0], pred_points[:, 1], s=8, c="#e5484d", label="Pred", linewidths=0)
    for gt, pred in zip(gt_points[::4], pred_points[::4]):
        ax.plot([gt[0], pred[0]], [gt[1], pred[1]], color="#f5a524", linewidth=0.5, alpha=0.55)
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

    rows = int(np.ceil(len(sample_indices) / 4))
    cols = min(4, len(sample_indices))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0), squeeze=False)
    nmes = []

    with torch.no_grad():
        for axis, index in zip(axes.ravel(), sample_indices):
            sample = dataset.samples[index]
            image = load_rgb(sample.image_path)
            gt_points = parse_pts(sample.pts_path)
            model_image = image
            crop_box = None
            if crop_to_landmarks:
                model_image, _crop_gt_points, crop_box = crop_image_and_points(image, gt_points, margin=crop_margin)

            h, w = model_image.shape[:2]
            tensor = preprocess(model_image, image_size).to(device)
            pred_norm = model(tensor).cpu().numpy().reshape(-1, 2)
            pred_points = pred_norm.copy()
            pred_points[:, 0] *= max(w - 1, 1)
            pred_points[:, 1] *= max(h - 1, 1)
            if crop_box is not None:
                left, top, _right, _bottom = crop_box
                pred_points[:, 0] += left
                pred_points[:, 1] += top

            nme = sample_nme(pred_points, gt_points)
            nmes.append(nme)
            draw_points(axis, image, gt_points, pred_points, crop_box, f"{sample.image_path.name}\nNME={nme:.3f}")

    for axis in axes.ravel()[len(sample_indices):]:
        axis.axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "300W Landmark Prediction Visualization",
        "=" * 50,
        f"Root: {args.root}",
        f"Checkpoint: {args.checkpoint}",
        f"Device: {device}",
        f"Crop to landmarks: {crop_to_landmarks}",
        f"Crop margin: {crop_margin}",
        f"Samples visualized: {len(sample_indices)}",
        f"Mean visualized NME: {float(np.mean(nmes)):.4f}",
        f"Output image: {args.output}",
        "",
        "Sample details:",
    ]
    for index, nme in zip(sample_indices, nmes):
        lines.append(f"- {dataset.samples[index].image_path.name}: NME={nme:.4f}")
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
