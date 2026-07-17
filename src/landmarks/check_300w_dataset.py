from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ROOT = Path("data/raw/300W")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a 300W-style landmark dataset with .pts annotations.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/landmark_300w_dataset_check_result.txt"))
    return parser.parse_args()


def count_pts_points(path: Path) -> int | None:
    try:
        inside = False
        count = 0
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            text = line.strip()
            if text == "{":
                inside = True
                continue
            if text == "}":
                break
            if inside and text:
                parts = text.split()
                if len(parts) >= 2:
                    count += 1
        return count
    except OSError:
        return None


def main() -> None:
    args = parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)

    if not args.root.exists():
        lines = [
            "300W Dataset Check",
            "=" * 50,
            f"Root: {args.root}",
            "Status: missing",
            "Expected: put 300W or COFW data under data/raw/300W or pass --root.",
        ]
        args.report.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return

    images = [p for p in args.root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    pts_files = sorted(args.root.rglob("*.pts"))
    point_counts = {}
    invalid = []
    for pts in pts_files:
        count = count_pts_points(pts)
        point_counts[count] = point_counts.get(count, 0) + 1
        if count not in {68, 29}:
            invalid.append((pts, count))

    lines = [
        "300W Dataset Check",
        "=" * 50,
        f"Root: {args.root}",
        f"Image files: {len(images)}",
        f"PTS annotation files: {len(pts_files)}",
        f"Point-count distribution: {point_counts}",
        f"Invalid/unexpected annotation count: {len(invalid)}",
        "",
        "Status: ready" if pts_files else "Status: no .pts annotations found",
    ]
    if invalid[:10]:
        lines.append("")
        lines.append("Examples with unexpected point counts:")
        for path, count in invalid[:10]:
            lines.append(f"- {path}: {count}")

    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()