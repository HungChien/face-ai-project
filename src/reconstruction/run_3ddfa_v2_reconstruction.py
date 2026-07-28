import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_IMAGE = Path("outputs/effects/images/indoor_016_original.jpg")
DEFAULT_REPO_ROOT = Path("third_party/3DDFA_V2")
DEFAULT_CONFIG = Path("configs/mb1_120x120.yml")
DEFAULT_OUTPUT_DIR = Path("outputs/3d_reconstruction/3ddfa_v2")
DEFAULT_REPORT = Path("outputs/reports/face_3d_reconstruction_3ddfa_v2_result.txt")
DEFAULT_JSON_REPORT = Path("outputs/reports/face_3d_reconstruction_3ddfa_v2_result.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official 3DDFA_V2 single-image 3D face reconstruction."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_repo_paths(repo_root: Path, config_path: Path) -> tuple[Path, Path]:
    repo_root = repo_root.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"3DDFA_V2 repo root not found: {repo_root}")
    if not (repo_root / "TDDFA.py").exists():
        raise FileNotFoundError(f"TDDFA.py not found under 3DDFA_V2 repo root: {repo_root}")
    if not (repo_root / "FaceBoxes").exists():
        raise FileNotFoundError(f"FaceBoxes package not found under 3DDFA_V2 repo root: {repo_root}")

    config = config_path if config_path.is_absolute() else repo_root / config_path
    if not config.exists():
        raise FileNotFoundError(f"3DDFA_V2 config file not found: {config}")
    return repo_root, config


def import_3ddfa(repo_root: Path) -> dict:
    sys.path.insert(0, str(repo_root))
    return {
        "faceboxes": importlib.import_module("FaceBoxes"),
        "tddfa": importlib.import_module("TDDFA"),
        "serialization": importlib.import_module("utils.serialization"),
    }


def load_config(config_path: Path, repo_root: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for key in ("checkpoint_fp", "bfm_fp"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(repo_root / cfg[key])
    return cfg


def vertices_to_array(vertices: object) -> np.ndarray:
    array = np.asarray(vertices, dtype=np.float32)
    if array.shape[0] == 3:
        return array.T
    if array.shape[-1] == 3:
        return array.reshape(-1, 3)
    raise ValueError(f"Unsupported vertex shape: {array.shape}")


def draw_projected_mesh(image_bgr: np.ndarray, vertices: np.ndarray, triangles: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    overlay = image_bgr.copy()
    projected = vertices[:, :2].astype(np.int32)
    step = max(1, len(triangles) // 5500)
    for tri in triangles[::step]:
        if np.max(tri) >= len(projected):
            continue
        points = projected[tri].reshape(-1, 1, 2)
        cv2.polylines(overlay, [points], True, (0, 220, 255), 1, cv2.LINE_AA)
    alpha = 0.55
    rendered = cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)
    cv2.putText(rendered, "3DDFA_V2 dense reconstruction", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 220), 2)
    cv2.imwrite(str(output_path), rendered)


def draw_pose_proxy(image_bgr: np.ndarray, vertices: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output = image_bgr.copy()
    xy = vertices[:, :2]
    x1, y1 = np.floor(xy.min(axis=0)).astype(int)
    x2, y2 = np.ceil(xy.max(axis=0)).astype(int)
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 220), 2)
    cv2.line(output, tuple(center.astype(int)), tuple((center + [width * 0.28, 0]).astype(int)), (0, 0, 255), 3)
    cv2.line(output, tuple(center.astype(int)), tuple((center + [0, -height * 0.28]).astype(int)), (0, 255, 0), 3)
    cv2.line(output, tuple(center.astype(int)), tuple((center + [width * 0.18, height * 0.18]).astype(int)), (255, 0, 0), 3)
    cv2.putText(output, "2D projected pose proxy", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 220), 2)
    cv2.imwrite(str(output_path), output)


def save_multiview(vertices: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    views = [("front", 15, -90), ("left", 8, -10), ("right", 8, -170), ("top", 80, -90)]
    fig = plt.figure(figsize=(16, 4))
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    radius = max(np.ptp(centered, axis=0).max() / 2.0, 1.0)
    for index, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        ax.scatter(centered[:, 0], centered[:, 2], -centered[:, 1], s=0.35, c=centered[:, 2], cmap="viridis")
        ax.set_title(title)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(-radius, radius)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_fallback_ply(path: Path, vertices: np.ndarray, triangles: np.ndarray, image_height: int) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(triangles)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for x, y, z in vertices:
            handle.write(f"{x:.3f} {image_height - y:.3f} {z:.3f}\n")
        for idx1, idx2, idx3 in triangles:
            handle.write(f"3 {int(idx3)} {int(idx2)} {int(idx1)}\n")


def write_fallback_obj(path: Path, vertices: np.ndarray, triangles: np.ndarray, image_height: int) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for x, y, z in vertices:
            handle.write(f"v {x:.3f} {image_height - y:.3f} {z:.3f}\n")
        for idx1, idx2, idx3 in triangles:
            handle.write(f"f {int(idx3) + 1} {int(idx2) + 1} {int(idx1) + 1}\n")


def main() -> None:
    args = parse_args()
    repo_root, config_path = resolve_repo_paths(args.repo_root, args.config)
    if args.check_only:
        print(f"3DDFA_V2 repo: {repo_root}")
        print(f"Config: {config_path}")
        print("Check passed.")
        return

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)

    image_dir = args.output_dir / "images"
    mesh_dir = args.output_dir / "meshes"
    ensure_dir(image_dir)
    ensure_dir(mesh_dir)
    ensure_dir(args.report.parent)
    ensure_dir(args.json_report.parent)

    modules = import_3ddfa(repo_root)
    FaceBoxes = modules["faceboxes"].FaceBoxes
    TDDFA = modules["tddfa"].TDDFA

    cfg = load_config(config_path, repo_root)
    tddfa = TDDFA(gpu_mode=args.device == "cuda", **cfg)
    face_boxes = FaceBoxes()

    boxes = face_boxes(image)
    if len(boxes) == 0:
        raise RuntimeError("3DDFA_V2 FaceBoxes did not detect a face.")
    param_lst, roi_box_lst = tddfa(image, boxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=args.dense)
    vertices = vertices_to_array(ver_lst[0])
    triangles = np.asarray(tddfa.tri, dtype=np.int32)

    stem = args.image.stem
    render_path = image_dir / f"{stem}_3ddfa_v2_projected_mesh.jpg"
    pose_path = image_dir / f"{stem}_3ddfa_v2_pose_proxy.jpg"
    multiview_path = image_dir / f"{stem}_3ddfa_v2_multiview.jpg"
    obj_path = mesh_dir / f"{stem}_3ddfa_v2.obj"
    ply_path = mesh_dir / f"{stem}_3ddfa_v2.ply"

    draw_projected_mesh(image, vertices, triangles, render_path)
    draw_pose_proxy(image, vertices, pose_path)
    save_multiview(vertices, multiview_path)
    write_fallback_obj(obj_path, vertices, triangles, image.shape[0])
    write_fallback_ply(ply_path, vertices, triangles, image.shape[0])

    summary = {
        "method": "3DDFA_V2 single-image 3D face reconstruction",
        "repo_root": str(repo_root),
        "config": str(config_path),
        "input_image": str(args.image),
        "device": args.device,
        "dense": args.dense,
        "detected_faces": len(boxes),
        "vertices": int(vertices.shape[0]),
        "triangles": int(triangles.shape[0]),
        "visualization_backend": "OpenCV/Matplotlib fallback visualization; official TDDFA reconstruction is used",
        "outputs": {
            "projected_mesh": str(render_path),
            "pose_proxy": str(pose_path),
            "multiview": str(multiview_path),
            "obj_mesh": str(obj_path),
            "ply_mesh": str(ply_path),
        },
    }
    args.json_report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "3DDFA_V2 3D Face Reconstruction",
        "=" * 40,
        f"Input image: {args.image}",
        f"3DDFA_V2 repo: {repo_root}",
        f"Config: {config_path}",
        f"Device: {args.device}",
        f"Dense reconstruction: {args.dense}",
        f"Detected faces: {len(boxes)}",
        f"Vertices: {summary['vertices']}",
        f"Triangles: {summary['triangles']}",
        "Visualization: OpenCV/Matplotlib fallback, official 3DDFA_V2 TDDFA reconstruction core",
        "",
        "Outputs:",
    ]
    for name, path in summary["outputs"].items():
        lines.append(f"- {name}: {path}")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()


