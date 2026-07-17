"""Prepare a 112x112 aligned MS1M folder dataset for ArcFace training.

The InsightFace `faces_emore` / MS1M-ArcFace RecordIO images are already
ArcFace-style aligned 112x112 faces. After converting RecordIO to identity
folders, this script validates the images, preserves the identity folder
layout, and writes a clean processed dataset under data/processed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MS1M aligned 112x112 identity folders.")
    parser.add_argument("--input-root", type=Path, default=Path("data/raw/ms-celeb-1m-subset"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/ms1m-aligned-112"))
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-if-already-112", action="store_true", default=True)
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/ms1m_aligned_112_preprocess_result.txt"))
    parser.add_argument("--json-report", type=Path, default=Path("outputs/reports/ms1m_aligned_112_preprocess_result.json"))
    parser.add_argument("--preview", type=Path, default=Path("outputs/images/ms1m_aligned_112_preview.jpg"))
    parser.add_argument("--preview-identities", type=int, default=8)
    parser.add_argument("--preview-images-per-identity", type=int, default=4)
    return parser.parse_args()


def list_images(identity_dir: Path) -> list[Path]:
    return sorted(path for path in identity_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def write_image(src: Path, dst: Path, image_size: int) -> tuple[bool, str, tuple[int, int] | None]:
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        return False, "unreadable", None
    h, w = image.shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if (h, w) == (image_size, image_size):
        shutil.copy2(src, dst)
        return True, "copied", (h, w)
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), resized)
    return True, "resized", (h, w)


def save_preview(output_root: Path, preview_path: Path, identities: int, images_per_identity: int) -> int:
    identity_dirs = sorted(path for path in output_root.iterdir() if path.is_dir())[:identities]
    if not identity_dirs:
        return 0
    rows = []
    shown = 0
    for identity_dir in identity_dirs:
        image_paths = list_images(identity_dir)[:images_per_identity]
        row_images = []
        for path in image_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            row_images.append(image)
            shown += 1
        if row_images:
            while len(row_images) < images_per_identity:
                row_images.append(np.full_like(row_images[0], 255))
            rows.append(np.concatenate(row_images, axis=1))
    if not rows:
        return 0
    grid = np.concatenate(rows, axis=0)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(images_per_identity * 1.8, len(rows) * 1.8))
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(preview_path, dpi=180)
    plt.close()
    return shown


def main() -> None:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output root is not empty; pass --overwrite to rebuild: {args.output_root}")

    started = time.perf_counter()
    prepare_output_root(args.output_root, args.overwrite)
    identity_dirs = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    copied = 0
    resized = 0
    unreadable = 0
    original_shapes: dict[str, int] = {}
    identity_counts: dict[str, int] = {}

    for identity_index, identity_dir in enumerate(identity_dirs, 1):
        output_identity_dir = args.output_root / identity_dir.name
        count = 0
        for image_path in list_images(identity_dir):
            output_path = output_identity_dir / image_path.name
            ok, action, shape = write_image(image_path, output_path, args.image_size)
            if not ok:
                unreadable += 1
                continue
            count += 1
            if action == "copied":
                copied += 1
            elif action == "resized":
                resized += 1
            if shape is not None:
                original_shapes[f"{shape[0]}x{shape[1]}"] = original_shapes.get(f"{shape[0]}x{shape[1]}", 0) + 1
        if count:
            identity_counts[identity_dir.name] = count
        if identity_index % 100 == 0:
            print(f"processed identities {identity_index}/{len(identity_dirs)}, images={copied + resized}")

    preview_count = save_preview(args.output_root, args.preview, args.preview_identities, args.preview_images_per_identity)
    elapsed = time.perf_counter() - started
    summary = {
        "task": "prepare MS1M aligned 112x112 folder dataset",
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "image_size": args.image_size,
        "input_identities": len(identity_dirs),
        "output_identities": len(identity_counts),
        "output_images": int(sum(identity_counts.values())),
        "copied_images": copied,
        "resized_images": resized,
        "unreadable_images": unreadable,
        "original_shapes": original_shapes,
        "min_images_per_identity": min(identity_counts.values()) if identity_counts else 0,
        "max_images_per_identity": max(identity_counts.values()) if identity_counts else 0,
        "preview": str(args.preview),
        "preview_images": preview_count,
        "elapsed_seconds": round(elapsed, 3),
    }
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "MS1M Aligned 112x112 Preprocess Result",
        "=" * 56,
        f"Input root: {args.input_root}",
        f"Output root: {args.output_root}",
        f"Input identities: {summary['input_identities']}",
        f"Output identities: {summary['output_identities']}",
        f"Output images: {summary['output_images']}",
        f"Copied images: {copied}",
        f"Resized images: {resized}",
        f"Unreadable images: {unreadable}",
        f"Original shapes: {original_shapes}",
        f"Images per identity: min={summary['min_images_per_identity']}, max={summary['max_images_per_identity']}",
        f"Preview: {args.preview}",
        f"Elapsed seconds: {elapsed:.3f}",
    ]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
