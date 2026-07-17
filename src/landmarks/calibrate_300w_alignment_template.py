from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from align_with_landmark_model import ARCFACE_112_TEMPLATE, landmarks_68_to_five
from train_landmark_regressor import collect_samples, parse_pts
from visualize_landmark_regressor import load_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a 300W-specific five-point alignment template and compare it with ArcFace template.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/300W"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/landmark_cnn_300w_aug30_best.pt"))
    parser.add_argument("--template-json", type=Path, default=Path("outputs/reports/landmark_300w_calibrated_template.json"))
    parser.add_argument("--template-plot", type=Path, default=Path("outputs/landmarks/calibrated_template_300w/template_comparison.jpg"))
    parser.add_argument("--grid-output", type=Path, default=Path("outputs/landmarks/calibrated_template_300w/arcface_vs_300w_template_alignment.jpg"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/landmark_300w_template_calibration_result.txt"))
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def affine_transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([points.astype(np.float32), ones], axis=1)
    return homo @ matrix.T.astype(np.float32)


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.estimateAffinePartial2D(source.astype(np.float32), target.astype(np.float32), method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("Failed to estimate similarity transform.")
    return matrix.astype(np.float32)


def template_residual(five_points: np.ndarray, template: np.ndarray) -> float:
    matrix = estimate_similarity(five_points, template)
    aligned = affine_transform_points(five_points, matrix)
    return float(np.linalg.norm(aligned - template, axis=1).mean())


def align_face_to_template(image: np.ndarray, five_points: np.ndarray, template: np.ndarray, output_size: int = 112) -> tuple[np.ndarray, np.ndarray]:
    matrix = estimate_similarity(five_points, template)
    aligned = cv2.warpAffine(image, matrix, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return aligned, matrix


def canonicalize_template(template: np.ndarray) -> np.ndarray:
    matrix = estimate_similarity(template, ARCFACE_112_TEMPLATE)
    return affine_transform_points(template, matrix).astype(np.float32)


def calibrate_template(five_shapes: list[np.ndarray], iterations: int) -> tuple[np.ndarray, list[float]]:
    template = ARCFACE_112_TEMPLATE.copy().astype(np.float32)
    history = []
    for _ in range(iterations):
        aligned_shapes = []
        for shape in five_shapes:
            matrix = estimate_similarity(shape, template)
            aligned_shapes.append(affine_transform_points(shape, matrix))
        new_template = np.mean(np.stack(aligned_shapes, axis=0), axis=0).astype(np.float32)
        new_template = canonicalize_template(new_template)
        delta = float(np.linalg.norm(new_template - template, axis=1).mean())
        history.append(delta)
        template = new_template
    return template, history


def split_indices(count: int, val_ratio: float = 0.15, seed: int = 42) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(count * val_ratio)))
    val_indices = sorted(indices[:val_size])
    train_indices = sorted(indices[val_size:])
    return train_indices, val_indices


def draw_template_plot(arcface_template: np.ndarray, calibrated_template: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(arcface_template[:, 0], arcface_template[:, 1], s=48, c="#2f6fed", label="ArcFace")
    ax.scatter(calibrated_template[:, 0], calibrated_template[:, 1], s=48, c="#e5484d", label="300W calibrated")
    for i, (a, c) in enumerate(zip(arcface_template, calibrated_template)):
        ax.plot([a[0], c[0]], [a[1], c[1]], color="#f5a524", linewidth=1.0)
        ax.text(c[0] + 0.8, c[1] + 0.8, str(i), fontsize=8)
    ax.set_xlim(20, 90)
    ax.set_ylim(105, 35)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="lower center")
    ax.set_title("Five-point template comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    samples = collect_samples(args.root)
    checkpoint = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint.exists() else {}
    train_indices = checkpoint.get("train_indices")
    val_indices = checkpoint.get("val_indices")
    if not train_indices or not val_indices:
        train_indices, val_indices = split_indices(len(samples))
    train_indices = [int(i) for i in train_indices]
    val_indices = [int(i) for i in val_indices]

    all_five = [landmarks_68_to_five(parse_pts(sample.pts_path)) for sample in samples]
    train_five = [all_five[i] for i in train_indices]
    val_five = [all_five[i] for i in val_indices]
    calibrated_template, history = calibrate_template(train_five, args.iterations)

    train_arc_res = [template_residual(shape, ARCFACE_112_TEMPLATE) for shape in train_five]
    train_cal_res = [template_residual(shape, calibrated_template) for shape in train_five]
    val_arc_res = [template_residual(shape, ARCFACE_112_TEMPLATE) for shape in val_five]
    val_cal_res = [template_residual(shape, calibrated_template) for shape in val_five]

    args.template_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "template_name": "300w_calibrated_112x112",
        "source": "300W GT 68-point annotations converted to five points",
        "point_order": ["left_eye_center", "right_eye_center", "nose_tip", "left_mouth_corner", "right_mouth_corner"],
        "arcface_template": np.round(ARCFACE_112_TEMPLATE, 4).tolist(),
        "calibrated_template": np.round(calibrated_template, 4).tolist(),
        "iterations": args.iterations,
        "history_mean_point_delta": history,
        "train_count": len(train_five),
        "val_count": len(val_five),
    }
    args.template_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    draw_template_plot(ARCFACE_112_TEMPLATE, calibrated_template, args.template_plot)

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(val_indices, min(args.num_samples, len(val_indices))))
    fig, axes = plt.subplots(len(chosen), 3, figsize=(11.2, len(chosen) * 3.1), squeeze=False)
    sample_rows = []
    output_dir = args.grid_output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for row_i, index in enumerate(chosen):
        sample = samples[index]
        image = load_rgb(sample.image_path)
        five = all_five[index]
        arc_aligned, _arc_matrix = align_face_to_template(image, five, ARCFACE_112_TEMPLATE)
        cal_aligned, _cal_matrix = align_face_to_template(image, five, calibrated_template)
        arc_res = template_residual(five, ARCFACE_112_TEMPLATE)
        cal_res = template_residual(five, calibrated_template)

        arc_path = output_dir / f"{sample.image_path.stem}_arcface_aligned_112.jpg"
        cal_path = output_dir / f"{sample.image_path.stem}_300w_template_aligned_112.jpg"
        cv2.imwrite(str(arc_path), cv2.cvtColor(arc_aligned, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(cal_path), cv2.cvtColor(cal_aligned, cv2.COLOR_RGB2BGR))

        axes[row_i][0].imshow(image)
        axes[row_i][0].scatter(five[:, 0], five[:, 1], s=22, c="#24b47e", linewidths=0)
        axes[row_i][0].set_title(sample.image_path.name, fontsize=9)
        axes[row_i][0].axis("off")
        axes[row_i][1].imshow(arc_aligned)
        axes[row_i][1].set_title(f"ArcFace template\nres={arc_res:.2f}px", fontsize=9)
        axes[row_i][1].axis("off")
        axes[row_i][2].imshow(cal_aligned)
        axes[row_i][2].set_title(f"300W calibrated\nres={cal_res:.2f}px", fontsize=9)
        axes[row_i][2].axis("off")
        sample_rows.append((sample.image_path.name, arc_res, cal_res, arc_path, cal_path))

    fig.tight_layout()
    fig.savefig(args.grid_output, dpi=180)
    plt.close(fig)

    lines = [
        "300W Five-Point Template Calibration Result",
        "=" * 56,
        f"Root: {args.root}",
        f"Checkpoint split source: {args.checkpoint}",
        f"Train shapes: {len(train_five)}",
        f"Val shapes: {len(val_five)}",
        f"Iterations: {args.iterations}",
        f"Template JSON: {args.template_json}",
        f"Template plot: {args.template_plot}",
        f"Grid image: {args.grid_output}",
        "",
        "Mean residual px after similarity fit:",
        f"- train ArcFace: {float(np.mean(train_arc_res)):.4f}",
        f"- train 300W calibrated: {float(np.mean(train_cal_res)):.4f}",
        f"- train improvement: {float(np.mean(train_arc_res) - np.mean(train_cal_res)):.4f}",
        f"- val ArcFace: {float(np.mean(val_arc_res)):.4f}",
        f"- val 300W calibrated: {float(np.mean(val_cal_res)):.4f}",
        f"- val improvement: {float(np.mean(val_arc_res) - np.mean(val_cal_res)):.4f}",
        "",
        "ArcFace template:",
        str(np.round(ARCFACE_112_TEMPLATE, 4).tolist()),
        "",
        "300W calibrated template:",
        str(np.round(calibrated_template, 4).tolist()),
        "",
        "Sample details:",
    ]
    for name, arc_res, cal_res, arc_path, cal_path in sample_rows:
        lines.append(f"- {name}: arcface_res={arc_res:.4f}px, calibrated_res={cal_res:.4f}px")
        lines.append(f"  arcface_aligned={arc_path}")
        lines.append(f"  calibrated_aligned={cal_path}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
