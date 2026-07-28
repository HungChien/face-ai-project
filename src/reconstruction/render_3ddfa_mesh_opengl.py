import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh


DEFAULT_MESH = Path("outputs/3d_reconstruction/3ddfa_v2/meshes/indoor_016_original_3ddfa_v2.obj")
DEFAULT_OUTPUT_DIR = Path("outputs/3d_reconstruction/3ddfa_v2/opengl")
DEFAULT_REPORT = Path("outputs/reports/face_3d_reconstruction_opengl_render_result.txt")
DEFAULT_JSON_REPORT = Path("outputs/reports/face_3d_reconstruction_opengl_render_result.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 3DDFA_v2 face mesh with OpenGL via pyrender.")
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]])


def rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]])


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    scale = np.max(mesh.extents)
    if scale <= 0:
        raise ValueError("Mesh has invalid extents.")
    mesh.apply_scale(2.2 / scale)
    return mesh


def camera_pose(distance: float = 3.2) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, distance]
    return pose


def render_view(mesh: trimesh.Trimesh, transform: np.ndarray, width: int, height: int) -> np.ndarray:
    scene = pyrender.Scene(bg_color=[245, 247, 250, 255], ambient_light=[0.35, 0.35, 0.35])
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.65,
        baseColorFactor=[0.78, 0.58, 0.48, 1.0],
    )
    render_mesh = mesh.copy()
    render_mesh.apply_transform(transform)
    scene.add(pyrender.Mesh.from_trimesh(render_mesh, material=material, smooth=True))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    scene.add(camera, pose=camera_pose())
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0.0, -1.5, 3.0]
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.8), pose=light_pose)
    fill_pose = np.eye(4)
    fill_pose[:3, 3] = [-2.0, 1.5, 2.0]
    scene.add(pyrender.PointLight(color=np.ones(3), intensity=22.0), pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, _depth = renderer.render(scene)
    renderer.delete()
    return color


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 56), (245, 247, 250), -1)
    cv2.putText(output, title, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 32, 44), 2, cv2.LINE_AA)
    return output


def make_grid(images: list[np.ndarray], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    first_row = np.concatenate(images[:3], axis=1)
    second_row = np.concatenate(images[3:], axis=1)
    grid = np.concatenate([first_row, second_row], axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    if not args.mesh.exists():
        raise FileNotFoundError(args.mesh)
    ensure_dir(args.output_dir)
    ensure_dir(args.report.parent)
    ensure_dir(args.json_report.parent)

    raw_mesh = trimesh.load_mesh(args.mesh, process=False)
    if not isinstance(raw_mesh, trimesh.Trimesh):
        raw_mesh = raw_mesh.dump(concatenate=True)
    mesh = normalize_mesh(raw_mesh)

    views = [
        ("front", np.eye(4)),
        ("left", rotation_y(-55)),
        ("right", rotation_y(55)),
        ("top", rotation_x(-55)),
        ("bottom", rotation_x(45)),
        ("three_quarter", rotation_y(35) @ rotation_x(-12)),
    ]
    image_outputs = {}
    titled_images = []
    for name, transform in views:
        rendered = render_view(mesh, transform, args.width, args.height)
        rendered = add_title(rendered, name)
        output_path = args.output_dir / f"{args.mesh.stem}_opengl_{name}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        image_outputs[name] = str(output_path)
        titled_images.append(rendered)

    grid_path = args.output_dir / f"{args.mesh.stem}_opengl_multiview_grid.jpg"
    make_grid(titled_images, grid_path)

    summary = {
        "method": "OpenGL offscreen rendering with pyrender",
        "mesh": str(args.mesh),
        "vertices": int(len(raw_mesh.vertices)),
        "faces": int(len(raw_mesh.faces)),
        "viewport": [args.width, args.height],
        "views": list(image_outputs.keys()),
        "outputs": {
            "views": image_outputs,
            "grid": str(grid_path),
        },
    }
    args.json_report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "OpenGL Rendering for 3DDFA_v2 Face Mesh",
        "=" * 44,
        f"Mesh: {args.mesh}",
        f"Renderer: pyrender OpenGL OffscreenRenderer",
        f"Vertices: {summary['vertices']}",
        f"Faces: {summary['faces']}",
        f"Viewport: {args.width} x {args.height}",
        "",
        "Outputs:",
    ]
    for name, path in image_outputs.items():
        lines.append(f"- {name}: {path}")
    lines.append(f"- grid: {grid_path}")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
