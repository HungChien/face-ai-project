from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LANDMARKS_DIR = PROJECT_ROOT / "src" / "landmarks"
if str(LANDMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(LANDMARKS_DIR))

from train_landmark_regressor import SmallLandmarkCNN, crop_image_and_points, landmark_crop_box, parse_pts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run landmark-driven traditional face effects baseline.")
    parser.add_argument("--image", type=Path, default=None, help="Input face image. If omitted, a 300W sample is used.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/landmark_cnn_300w_aug30_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/effects/images"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/effects_baseline_result.txt"))
    parser.add_argument("--json-report", type=Path, default=Path("outputs/reports/effects_baseline_result.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_image() -> Path:
    candidates = [
        Path("data/raw/300W/300W/01_Indoor/indoor_016.png"),
        Path("data/raw/300W/300W/01_Indoor/indoor_008.png"),
        Path("data/raw/300W/300W/02_Outdoor/outdoor_004.png"),
    ]
    for path in candidates:
        if path.exists():
            return path
    found = sorted(Path("data/raw/300W").rglob("*.png"))
    if found:
        return found[0]
    raise FileNotFoundError("No default 300W image found. Please pass --image.")


def sidecar_pts(image_path: Path) -> Path | None:
    pts = image_path.with_suffix(".pts")
    return pts if pts.exists() else None


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess(image: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32) / 255.0
    resized = np.transpose(resized, (2, 0, 1))
    return torch.from_numpy(resized).unsqueeze(0)


def detect_face_box(image_rgb: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        h, w = image_rgb.shape[:2]
        side = int(min(h, w) * 0.82)
        left = (w - side) // 2
        top = (h - side) // 2
        return left, top, left + side, top + side
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    margin = int(max(w, h) * 0.22)
    ih, iw = image_rgb.shape[:2]
    return max(0, x - margin), max(0, y - margin), min(iw, x + w + margin), min(ih, y + h + margin)


def predict_landmarks(image_rgb: np.ndarray, image_path: Path, checkpoint_path: Path, device: str) -> tuple[np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    image_size = int(checkpoint.get("image_size", 128))
    num_points = int(checkpoint.get("num_points", 68))
    crop_margin = float(checkpoint.get("crop_margin", 0.25))

    model = SmallLandmarkCNN(num_points=num_points).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pts_path = sidecar_pts(image_path)
    crop_source = "haar_or_center"
    if pts_path is not None:
        gt_points = parse_pts(pts_path)
        crop_image, _crop_points, crop_box = crop_image_and_points(image_rgb, gt_points, margin=crop_margin)
        crop_source = "sidecar_pts_for_crop"
    else:
        crop_box = detect_face_box(image_rgb)
        left, top, right, bottom = crop_box
        crop_image = image_rgb[top:bottom, left:right]

    h, w = crop_image.shape[:2]
    tensor = preprocess(crop_image, image_size).to(device)
    with torch.no_grad():
        pred_norm = model(tensor).detach().cpu().numpy().reshape(-1, 2)
    points = pred_norm.copy()
    points[:, 0] *= max(w - 1, 1)
    points[:, 1] *= max(h - 1, 1)
    left, top, _right, _bottom = crop_box
    points[:, 0] += left
    points[:, 1] += top
    points[:, 0] = np.clip(points[:, 0], 0, image_rgb.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image_rgb.shape[0] - 1)

    meta = {
        "checkpoint": str(checkpoint_path),
        "image_size": image_size,
        "num_points": num_points,
        "crop_source": crop_source,
        "crop_box": [int(v) for v in crop_box],
        "has_sidecar_pts": pts_path is not None,
    }
    return points.astype(np.float32), meta


def feather_mask(shape: tuple[int, int], polygons: list[np.ndarray], blur: int = 31) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    for poly in polygons:
        cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 1.0)
    if blur > 0:
        if blur % 2 == 0:
            blur += 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return np.clip(mask[..., None], 0.0, 1.0)


def face_hull(points: np.ndarray) -> np.ndarray:
    return cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)


def blend_with_mask(base: np.ndarray, effect: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base.astype(np.float32) * (1.0 - mask) + effect.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def beauty_effect(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    smooth = cv2.bilateralFilter(bgr, d=0, sigmaColor=42, sigmaSpace=9)
    smooth = cv2.cvtColor(smooth, cv2.COLOR_BGR2RGB)
    bright = np.clip(smooth.astype(np.float32) * 1.08 + 7.0, 0, 255).astype(np.uint8)
    mask = feather_mask(image.shape[:2], [face_hull(points)], blur=45)
    return blend_with_mask(image, bright, mask * 0.72)


def overlay_polygon(image: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int], alpha: float, blur: int) -> np.ndarray:
    color_layer = np.zeros_like(image, dtype=np.uint8)
    color_layer[:, :] = np.array(color, dtype=np.uint8)
    mask = feather_mask(image.shape[:2], [polygon], blur=blur) * alpha
    return blend_with_mask(image, color_layer, mask)


def makeup_effect(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    out = image.copy()
    if len(points) >= 68:
        lips_outer = points[48:60]
        lips_inner = points[60:68]
        out = overlay_polygon(out, lips_outer, (188, 52, 86), alpha=0.42, blur=13)
        out = overlay_polygon(out, lips_inner, (230, 145, 160), alpha=0.20, blur=9)
        left_eye = points[[36, 37, 38, 39, 40, 41]]
        right_eye = points[[42, 43, 44, 45, 46, 47]]
        out = overlay_polygon(out, left_eye + np.array([0, -3], dtype=np.float32), (190, 95, 150), alpha=0.22, blur=17)
        out = overlay_polygon(out, right_eye + np.array([0, -3], dtype=np.float32), (190, 95, 150), alpha=0.22, blur=17)
        for idxs in ([2, 3, 4, 31], [12, 13, 14, 35]):
            cheek = points[idxs].mean(axis=0)
            radius = max(8, int(np.linalg.norm(points[0] - points[16]) * 0.08))
            yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
            mask2d = ((xx - cheek[0]) ** 2 + (yy - cheek[1]) ** 2 <= radius ** 2).astype(np.float32)
            mask = cv2.GaussianBlur(mask2d, (radius * 2 + 1, radius * 2 + 1), 0)[..., None] * 0.32
            color = np.zeros_like(out, dtype=np.uint8)
            color[:] = (236, 118, 134)
            out = blend_with_mask(out, color, mask)
    return out


def draw_transparent_line(image: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], thickness: int, alpha: float) -> np.ndarray:
    overlay = image.copy()
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def sticker_effect(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    out = image.copy()
    if len(points) >= 68:
        left_eye = points[36:42]
        right_eye = points[42:48]
        le = left_eye.mean(axis=0)
        re = right_eye.mean(axis=0)
        eye_dist = float(np.linalg.norm(re - le))
        radius_x = max(10, int(eye_dist * 0.30))
        radius_y = max(6, int(eye_dist * 0.17))
        color = (35, 45, 55)
        lens = (115, 210, 220)
        overlay = out.copy()
        cv2.ellipse(overlay, tuple(np.round(le).astype(int)), (radius_x, radius_y), 0, 0, 360, lens, -1, cv2.LINE_AA)
        cv2.ellipse(overlay, tuple(np.round(re).astype(int)), (radius_x, radius_y), 0, 0, 360, lens, -1, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, 0.28, out, 0.72, 0)
        cv2.ellipse(out, tuple(np.round(le).astype(int)), (radius_x, radius_y), 0, 0, 360, color, 3, cv2.LINE_AA)
        cv2.ellipse(out, tuple(np.round(re).astype(int)), (radius_x, radius_y), 0, 0, 360, color, 3, cv2.LINE_AA)
        out = draw_transparent_line(out, tuple(np.round(le + [radius_x, 0]).astype(int)), tuple(np.round(re - [radius_x, 0]).astype(int)), color, 3, 0.95)
        brow_mid = (points[21] + points[22]) / 2
        star_center = tuple(np.round(brow_mid + np.array([0, -eye_dist * 0.38])).astype(int))
        star_radius = max(8, int(eye_dist * 0.10))
        cv2.circle(out, star_center, star_radius, (255, 205, 70), -1, cv2.LINE_AA)
        cv2.circle(out, star_center, star_radius + 3, (255, 150, 60), 2, cv2.LINE_AA)
    return out


def landmark_overlay(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    out = image.copy()
    for idx, (x, y) in enumerate(points):
        color = (244, 84, 94) if idx in {36, 39, 42, 45, 30, 48, 54} else (36, 170, 126)
        cv2.circle(out, (int(round(x)), int(round(y))), 2, color, -1, cv2.LINE_AA)
    return out


def save_grid(images: list[tuple[str, np.ndarray]], output: Path) -> None:
    cols = 2
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.4), squeeze=False)
    for ax, (title, image) in zip(axes.ravel(), images):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes.ravel()[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    started = time.perf_counter()
    image_path = args.image or default_image()
    image = read_rgb(image_path)
    points, landmark_meta = predict_landmarks(image, image_path, args.checkpoint, device)

    outputs = {
        "original": image,
        "landmarks": landmark_overlay(image, points),
        "beauty": beauty_effect(image, points),
        "makeup": makeup_effect(image, points),
        "ar_sticker": sticker_effect(image, points),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, output_image in outputs.items():
        path = args.output_dir / f"{image_path.stem}_{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        written[name] = str(path)

    grid_path = args.output_dir / f"{image_path.stem}_effects_grid.jpg"
    save_grid(
        [
            ("Original", outputs["original"]),
            ("Predicted landmarks", outputs["landmarks"]),
            ("Beauty: smoothing + brightening", outputs["beauty"]),
            ("Makeup: lipstick + blush + eye shadow", outputs["makeup"]),
            ("AR sticker: glasses + forehead badge", outputs["ar_sticker"]),
        ],
        grid_path,
    )

    elapsed = time.perf_counter() - started
    result = {
        "input_image": str(image_path),
        "device": device,
        "elapsed_seconds": elapsed,
        "landmark_meta": landmark_meta,
        "effects": {
            "beauty": "bilateral smoothing, face-mask blending, mild brightening",
            "makeup": "landmark polygons for lips/eyes plus cheek color masks",
            "ar_sticker": "eye-center glasses and forehead badge anchored by 68 landmarks",
        },
        "outputs": {**written, "grid": str(grid_path)},
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Landmark-driven Traditional Face Effects Baseline",
        "=" * 58,
        f"Input image: {image_path}",
        f"Checkpoint: {args.checkpoint}",
        f"Device: {device}",
        f"Elapsed seconds: {elapsed:.3f}",
        f"Landmark crop source: {landmark_meta['crop_source']}",
        f"Crop box: {landmark_meta['crop_box']}",
        "",
        "Implemented effects:",
        "- Beauty: bilateral smoothing + face mask brightening.",
        "- Makeup: lipstick, cheek blush, and eye shadow using 68-point regions.",
        "- AR sticker: glasses and forehead badge anchored to eye/brow landmarks.",
        "",
        "Output files:",
    ]
    for key, path in result["outputs"].items():
        lines.append(f"- {key}: {path}")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
