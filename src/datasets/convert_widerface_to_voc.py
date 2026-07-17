from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def parse_annotation_file(path: Path) -> list[tuple[str, list[list[int]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items: list[tuple[str, list[list[int]]]] = []
    i = 0
    while i < len(lines):
        image_rel = lines[i].strip()
        i += 1
        if not image_rel:
            continue
        face_count = int(lines[i].strip())
        i += 1
        boxes: list[list[int]] = []
        annotation_lines = max(face_count, 1)
        for line_idx in range(annotation_lines):
            fields = [int(float(x)) for x in lines[i].split()]
            i += 1
            if line_idx >= face_count:
                continue
            x, y, w, h = fields[:4]
            invalid = fields[9] if len(fields) > 9 else 0
            if invalid == 1 or w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
        items.append((image_rel, boxes))
    return items


def image_size(image_path: Path) -> tuple[int, int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to read image sizes.") from exc

    with Image.open(image_path) as img:
        width, height = img.size
        channels = len(img.getbands())
    return width, height, channels


def make_xml(folder: str, filename: str, width: int, height: int,
             channels: int, boxes: list[list[int]]) -> ET.ElementTree:
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(channels)

    for xmin, ymin, xmax, ymax in boxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "face"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        box = ET.SubElement(obj, "bndbox")
        ET.SubElement(box, "xmin").text = str(xmin)
        ET.SubElement(box, "ymin").text = str(ymin)
        ET.SubElement(box, "xmax").text = str(xmax)
        ET.SubElement(box, "ymax").text = str(ymax)

    return ET.ElementTree(annotation)


def convert_split(annotation_file: Path, source_image_root: Path,
                  output_root: Path, output_split: str,
                  copy_images: bool) -> dict[str, int]:
    items = parse_annotation_file(annotation_file)
    split_root = output_root / output_split
    ann_root = split_root / "Annotations"
    image_root = split_root / "images"
    ids: list[str] = []

    skipped_missing = 0
    skipped_empty = 0

    for image_rel, boxes in items:
        if not boxes:
            skipped_empty += 1
            continue

        src_img = source_image_root / image_rel
        if not src_img.exists():
            skipped_missing += 1
            continue

        img_id = str(Path(image_rel).with_suffix("")).replace("\\", "/")
        event_dir = Path(image_rel).parent.as_posix()
        filename = Path(image_rel).name
        folder = "images"

        width, height, channels = image_size(src_img)
        xml = make_xml(folder, filename, width, height, channels, boxes)

        xml_path = ann_root / Path(image_rel).with_suffix(".xml")
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml.write(xml_path, encoding="utf-8", xml_declaration=True)

        if copy_images:
            dst_img = image_root / image_rel
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

        ids.append(img_id)

    id_file = output_root / ("train.txt" if output_split == "WIDER_train" else "val.txt")
    id_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return {
        "annotation_items": len(items),
        "converted_images": len(ids),
        "skipped_missing_images": skipped_missing,
        "skipped_empty_annotations": skipped_empty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official WIDER FACE txt annotations to Pascal VOC XML.")
    parser.add_argument("--wider-root", type=Path, required=True,
                        help="Root containing WIDER_train, WIDER_val, and wider_face_split.")
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/raw/WIDERFace"))
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images into the converted output root.")
    args = parser.parse_args()

    split_root = args.wider_root / "wider_face_split"
    train_ann = split_root / "wider_face_train_bbx_gt.txt"
    val_ann = split_root / "wider_face_val_bbx_gt.txt"
    train_images = args.wider_root / "WIDER_train" / "images"
    val_images = args.wider_root / "WIDER_val" / "images"

    required = [train_ann, val_ann, train_images, val_images]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing WIDER FACE inputs:\n" + "\n".join(missing))

    args.output_root.mkdir(parents=True, exist_ok=True)
    train_stats = convert_split(
        train_ann, train_images, args.output_root, "WIDER_train",
        args.copy_images)
    val_stats = convert_split(
        val_ann, val_images, args.output_root, "WIDER_val",
        args.copy_images)

    report = [
        "WIDER FACE VOC Conversion",
        "=" * 50,
        f"Source root: {args.wider_root}",
        f"Output root: {args.output_root}",
        f"Copy images: {args.copy_images}",
        "",
        "Train:",
    ]
    report.extend(f"- {key}: {value}" for key, value in train_stats.items())
    report.append("")
    report.append("Val:")
    report.extend(f"- {key}: {value}" for key, value in val_stats.items())

    report_path = Path("outputs/reports/widerface_voc_conversion_result.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()


