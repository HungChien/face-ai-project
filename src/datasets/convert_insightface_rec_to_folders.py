"""Convert an InsightFace/MXNet RecordIO face dataset into identity folders.

Expected input:
    <rec-root>/train.rec
    <rec-root>/train.idx

Expected output:
    <output-root>/identity_000000/00000001.jpg
    <output-root>/identity_000000/00000002.jpg
    <output-root>/identity_000001/00000001.jpg

This is useful for converting cleaned MS-Celeb-1M variants such as
MS1M-ArcFace/MS1MV2 into the folder layout consumed by
src/recognition/train_arcface_celeba_subset.py --dataset-format folder.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert InsightFace RecordIO to identity folders.")
    parser.add_argument("--rec-root", type=Path, required=True, help="Directory containing train.rec and train.idx.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output folder dataset root.")
    parser.add_argument("--max-identities", type=int, default=1000, help="Keep top-K identities by image count. 0 means all.")
    parser.add_argument("--min-images-per-identity", type=int, default=5)
    parser.add_argument("--max-images-per-identity", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Only scan labels and write summary; do not export images.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/reports/ms1m_recordio_conversion_summary.json"),
    )
    return parser.parse_args()


def require_mxnet():
    try:
        import mxnet as mx  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "mxnet is required to read InsightFace RecordIO files. "
            "Install it in a separate conversion environment if ml-gpu cannot install it cleanly, "
            "then rerun this script."
        ) from exc
    return mx


def open_recordio(mx, rec_root: Path):
    rec_path = rec_root / "train.rec"
    idx_path = rec_root / "train.idx"
    if not rec_path.exists() or not idx_path.exists():
        raise FileNotFoundError(f"Expected train.rec and train.idx under {rec_root}")
    reader = mx.recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), "r")
    header0, _ = mx.recordio.unpack(reader.read_idx(0))
    if header0.flag > 0:
        image_indices = list(range(1, int(header0.label[0])))
    else:
        image_indices = list(reader.keys)
    return reader, image_indices


def get_label(mx, reader, idx: int) -> int:
    header, _img = mx.recordio.unpack(reader.read_idx(idx))
    label = header.label
    if isinstance(label, (list, tuple)):
        return int(label[0])
    try:
        return int(label[0])
    except TypeError:
        return int(label)


def scan_labels(mx, reader, image_indices: list[int]) -> Counter:
    counts: Counter = Counter()
    for n, idx in enumerate(image_indices, 1):
        counts[get_label(mx, reader, idx)] += 1
        if n % 100000 == 0:
            print(f"scanned {n}/{len(image_indices)} images")
    return counts


def selected_labels(counts: Counter, max_identities: int, min_images: int) -> list[int]:
    labels = [(label, count) for label, count in counts.items() if count >= min_images]
    labels.sort(key=lambda item: (-item[1], item[0]))
    if max_identities > 0:
        labels = labels[:max_identities]
    return [label for label, _count in labels]


def export_images(
    mx,
    reader,
    image_indices: list[int],
    labels_to_keep: set[int],
    output_root: Path,
    max_images_per_identity: int,
) -> dict[int, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    exported: dict[int, int] = defaultdict(int)
    for n, idx in enumerate(image_indices, 1):
        packed = reader.read_idx(idx)
        header, image_bytes = mx.recordio.unpack(packed)
        label = header.label
        if not isinstance(label, (int, float)):
            label = label[0]
        label = int(label)
        if label not in labels_to_keep:
            continue
        if exported[label] >= max_images_per_identity:
            continue
        identity_dir = output_root / f"identity_{label:06d}"
        identity_dir.mkdir(parents=True, exist_ok=True)
        exported[label] += 1
        image_path = identity_dir / f"{exported[label]:08d}.jpg"
        image_path.write_bytes(image_bytes)
        if n % 100000 == 0:
            print(f"processed {n}/{len(image_indices)} images, exported={sum(exported.values())}")
    return dict(exported)


def main() -> None:
    args = parse_args()
    mx = require_mxnet()
    reader, image_indices = open_recordio(mx, args.rec_root)
    counts = scan_labels(mx, reader, image_indices)
    labels = selected_labels(counts, args.max_identities, args.min_images_per_identity)
    exported = {}
    if not args.dry_run:
        exported = export_images(
            mx,
            reader,
            image_indices,
            set(labels),
            args.output_root,
            args.max_images_per_identity,
        )

    summary = {
        "rec_root": str(args.rec_root),
        "output_root": str(args.output_root),
        "total_images_in_recordio": len(image_indices),
        "total_identities_in_recordio": len(counts),
        "selected_identities": len(labels),
        "min_images_per_identity": args.min_images_per_identity,
        "max_images_per_identity": args.max_images_per_identity,
        "max_identities": args.max_identities,
        "dry_run": args.dry_run,
        "top_10_identity_counts": counts.most_common(10),
        "exported_images": int(sum(exported.values())) if exported else 0,
        "exported_identities": len(exported),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
