import argparse
import json
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.experiments.python import vision
from mediapipe.experiments.python.core.base_options import BaseOptions


DEFAULT_IMAGE = Path("outputs/effects/images/indoor_016_original.jpg")
DEFAULT_MODEL = Path("models/checkpoints/face_landmarker.task")
DEFAULT_OUTPUT_VIDEO = Path("outputs/effects/videos/dynamic_face_effects_demo.mp4")
DEFAULT_CONTACT_SHEET = Path("outputs/effects/videos/dynamic_face_effects_contact_sheet.jpg")
DEFAULT_REPORT = Path("outputs/reports/dynamic_effects_demo_result.txt")
DEFAULT_JSON_REPORT = Path("outputs/reports/dynamic_effects_demo_result.json")

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_CHEEK = [50, 101, 118, 205]
RIGHT_CHEEK = [280, 330, 347, 425]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-time landmark-driven dynamic face effects demo.")
    parser.add_argument("--input-video", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-video", type=Path, default=DEFAULT_OUTPUT_VIDEO)
    parser.add_argument("--contact-sheet", type=Path, default=DEFAULT_CONTACT_SHEET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--detect-width", type=int, default=360)
    parser.add_argument("--detect-every", type=int, default=1, help="Run face landmark detection every N frames and track with the previous smoothed landmarks between detections.")
    parser.add_argument("--ema", type=float, default=0.72)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_source_image(path: Path, width: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    scale = width / float(w)
    return cv2.resize(image, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def synthesize_motion_frame(image: np.ndarray, index: int, total: int) -> np.ndarray:
    h, w = image.shape[:2]
    phase = 2.0 * math.pi * index / max(total, 1)
    angle = 2.8 * math.sin(phase)
    scale = 1.0 + 0.018 * math.sin(phase * 1.7)
    tx = 13.0 * math.sin(phase * 0.8)
    ty = 6.0 * math.cos(phase * 1.2)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[:, 2] += [tx, ty]
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def read_video_frames(path: Path, width: int, max_frames: int) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = width / float(w)
        frame = cv2.resize(frame, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from {path}")
    return frames, fps


def create_landmarker(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_points(landmarker, frame_bgr: np.ndarray, detect_width: int) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    if detect_width > 0 and w > detect_width:
        scale = detect_width / float(w)
        detect_frame = cv2.resize(frame_bgr, (detect_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        detect_frame = frame_bgr

    frame_rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    dh, dw = detect_frame.shape[:2]
    landmarks = result.face_landmarks[0]
    points = np.array([[point.x * dw, point.y * dh] for point in landmarks], dtype=np.float32)
    return points / max(scale, 1e-6)


def polygon_mask(shape: tuple[int, int], points: np.ndarray, blur: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    cv2.fillConvexPoly(mask, np.round(points).astype(np.int32), 1.0)
    if blur > 0:
        if blur % 2 == 0:
            blur += 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return np.clip(mask[..., None], 0.0, 1.0)


def blend(base: np.ndarray, effect: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base.astype(np.float32) * (1.0 - mask) + effect.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_beauty(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    smooth = cv2.bilateralFilter(frame, d=0, sigmaColor=36, sigmaSpace=9)
    bright = np.clip(smooth.astype(np.float32) * 1.05 + 5.0, 0, 255).astype(np.uint8)
    hull = cv2.convexHull(points[FACE_OVAL].astype(np.float32)).reshape(-1, 2)
    mask = polygon_mask(frame.shape[:2], hull, blur=51) * 0.58
    return blend(frame, bright, mask)


def apply_makeup(frame: np.ndarray, points: np.ndarray, phase: float) -> np.ndarray:
    out = frame.copy()
    lip_color = np.zeros_like(out)
    lip_color[:] = (82, 42, 188)
    lips = points[LIPS_OUTER]
    lip_alpha = 0.34 + 0.08 * (0.5 + 0.5 * math.sin(phase * 1.5))
    out = blend(out, lip_color, polygon_mask(out.shape[:2], lips, blur=15) * lip_alpha)

    blush_color = np.zeros_like(out)
    blush_color[:] = (128, 105, 236)
    for cheek_ids in [LEFT_CHEEK, RIGHT_CHEEK]:
        cheek_center = points[cheek_ids].mean(axis=0)
        eye_dist = np.linalg.norm(points[33] - points[263])
        radius = max(18, int(eye_dist * 0.13))
        yy, xx = np.ogrid[:out.shape[0], :out.shape[1]]
        mask2d = ((xx - cheek_center[0]) ** 2 + (yy - cheek_center[1]) ** 2 <= radius ** 2).astype(np.float32)
        mask = cv2.GaussianBlur(mask2d, (radius * 2 + 1, radius * 2 + 1), 0)[..., None] * 0.20
        out = blend(out, blush_color, mask)
    return out


def draw_hat(frame: np.ndarray, points: np.ndarray, phase: float) -> None:
    left = points[234]
    right = points[454]
    top = points[10]
    eye_dist = float(np.linalg.norm(points[33] - points[263]))
    width = int(max(90, np.linalg.norm(right - left) * 1.12))
    height = int(width * 0.30)
    center = top + np.array([0.0, -eye_dist * (0.36 + 0.04 * math.sin(phase * 2.0))], dtype=np.float32)
    x1, y1 = np.round(center + [-width / 2, -height / 2]).astype(int)
    x2, y2 = np.round(center + [width / 2, height / 2]).astype(int)
    brim_y = y2 - int(height * 0.12)
    cv2.ellipse(frame, ((x1 + x2) // 2, brim_y), (width // 2, max(8, height // 5)), 0, 0, 360, (32, 36, 48), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1 + width // 5, y1), (x2 - width // 5, brim_y), (38, 48, 74), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1 + width // 5, brim_y - max(4, height // 8)), (x2 - width // 5, brim_y), (210, 174, 72), -1, cv2.LINE_AA)


def draw_glasses(frame: np.ndarray, points: np.ndarray, phase: float) -> None:
    le = points[LEFT_EYE].mean(axis=0)
    re = points[RIGHT_EYE].mean(axis=0)
    eye_dist = float(np.linalg.norm(re - le))
    rx = max(18, int(eye_dist * 0.30))
    ry = max(10, int(eye_dist * 0.17))
    lens_alpha = 0.20 + 0.08 * (0.5 + 0.5 * math.sin(phase * 2.4))
    overlay = frame.copy()
    for center in [le, re]:
        cv2.ellipse(overlay, tuple(np.round(center).astype(int)), (rx, ry), 0, 0, 360, (224, 210, 110), -1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, lens_alpha, frame, 1.0 - lens_alpha, 0)
    for center in [le, re]:
        cv2.ellipse(frame, tuple(np.round(center).astype(int)), (rx, ry), 0, 0, 360, (24, 28, 36), 3, cv2.LINE_AA)
    cv2.line(frame, tuple(np.round(le + [rx, 0]).astype(int)), tuple(np.round(re - [rx, 0]).astype(int)), (24, 28, 36), 3, cv2.LINE_AA)
    glint = le + np.array([-rx * 0.28, -ry * 0.28], dtype=np.float32)
    cv2.circle(frame, tuple(np.round(glint).astype(int)), max(3, rx // 8), (255, 255, 255), -1, cv2.LINE_AA)


def draw_tracking_overlay(frame: np.ndarray, points: np.ndarray, frame_index: int, fps: float) -> None:
    for idx in [1, 10, 33, 61, 152, 263, 291]:
        cv2.circle(frame, tuple(np.round(points[idx]).astype(int)), 3, (0, 250, 180), -1, cv2.LINE_AA)
    cv2.putText(frame, f"Dynamic face effects | frame {frame_index:03d} | {fps:.1f} FPS", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 220), 2, cv2.LINE_AA)


def apply_dynamic_effects(frame: np.ndarray, points: np.ndarray, frame_index: int, fps: float) -> np.ndarray:
    phase = 2.0 * math.pi * frame_index / max(fps * 2.0, 1.0)
    out = apply_beauty(frame, points)
    out = apply_makeup(out, points, phase)
    draw_hat(out, points, phase)
    draw_glasses(out, points, phase)
    draw_tracking_overlay(out, points, frame_index, fps)
    return out


def write_contact_sheet(frames: list[np.ndarray], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    selected = np.linspace(0, len(frames) - 1, 8).astype(int)
    thumbs = []
    for idx in selected:
        thumb = cv2.resize(frames[idx], (320, int(frames[idx].shape[0] * 320 / frames[idx].shape[1])), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"frame {idx}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 220), 2, cv2.LINE_AA)
        thumbs.append(thumb)
    row1 = np.concatenate(thumbs[:4], axis=1)
    row2 = np.concatenate(thumbs[4:], axis=1)
    sheet = np.concatenate([row1, row2], axis=0)
    cv2.imwrite(str(output_path), sheet)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_video.parent)
    ensure_dir(args.report.parent)
    ensure_dir(args.json_report.parent)

    if args.input_video:
        source_frames, source_fps = read_video_frames(args.input_video, args.width, args.frames)
        source = str(args.input_video)
        fps = float(source_fps)
    else:
        source_image = load_source_image(args.image, args.width)
        source_frames = [synthesize_motion_frame(source_image, i, args.frames) for i in range(args.frames)]
        source = str(args.image)
        fps = float(args.fps)

    h, w = source_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {args.output_video}")

    processed_frames = []
    detected = 0
    reused_previous = 0
    previous_points: np.ndarray | None = None
    started = time.perf_counter()
    with create_landmarker(args.model) as landmarker:
        for frame_index, frame in enumerate(source_frames):
            should_detect = previous_points is None or frame_index % max(args.detect_every, 1) == 0
            points = detect_points(landmarker, frame, args.detect_width) if should_detect else None
            if points is None:
                if previous_points is None:
                    output = frame.copy()
                    cv2.putText(output, "No face detected", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    points = previous_points
                    reused_previous += 1
                    output = apply_dynamic_effects(frame, points, frame_index, fps)
            else:
                detected += 1
                if previous_points is not None:
                    points = args.ema * previous_points + (1.0 - args.ema) * points
                previous_points = points
                output = apply_dynamic_effects(frame, points, frame_index, fps)
            writer.write(output)
            processed_frames.append(output)
    writer.release()
    elapsed = time.perf_counter() - started
    processing_fps = len(source_frames) / max(elapsed, 1e-6)
    write_contact_sheet(processed_frames, args.contact_sheet)

    summary = {
        "method": "real-time MediaPipe FaceLandmarker tracking with dynamic AR sticker and beauty/makeup effects",
        "source": source,
        "input_video": str(args.input_video) if args.input_video else None,
        "landmarker_model": str(args.model),
        "frames": len(source_frames),
        "video_fps": fps,
        "frame_width": w,
        "detection_width": args.detect_width,
        "detect_every": args.detect_every,
        "processing_seconds": elapsed,
        "processing_fps": processing_fps,
        "detected_frames": detected,
        "detection_rate": detected / max(len(source_frames), 1),
        "reused_previous_landmarks": reused_previous,
        "effects": ["dynamic glasses", "animated hat", "skin smoothing", "brightening", "lipstick", "blush", "tracking overlay"],
        "outputs": {
            "video": str(args.output_video),
            "contact_sheet": str(args.contact_sheet),
        },
    }
    args.json_report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "Dynamic Face Effects Demo",
        "=" * 32,
        f"Source: {source}",
        f"Landmarker: {args.model}",
        f"Frames: {len(source_frames)}",
        f"Video FPS: {fps:.2f}",
        f"Frame width: {w}",
        f"Detection width: {args.detect_width}",
        f"Detect every: {args.detect_every} frame(s)",
        f"Processing seconds: {elapsed:.3f}",
        f"Processing FPS: {processing_fps:.2f}",
        f"Detected frames: {detected} / {len(source_frames)}",
        f"Detection rate: {summary['detection_rate']:.4f}",
        f"Reused previous landmarks: {reused_previous}",
        "",
        "Implemented dynamic effects:",
        "- Real-time dense landmark tracking with EMA smoothing.",
        "- Dynamic AR glasses with animated lens highlight.",
        "- Animated hat anchored to forehead/face width.",
        "- Beauty: bilateral smoothing and mild brightening inside face mask.",
        "- Makeup: lipstick and cheek blush driven by facial landmarks.",
        "",
        "Outputs:",
        f"- video: {args.output_video}",
        f"- contact_sheet: {args.contact_sheet}",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()



