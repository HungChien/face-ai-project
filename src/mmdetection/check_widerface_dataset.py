from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


DEFAULT_ROOT = Path("data/raw/WIDERFace")
DEFAULT_REPORT = Path("outputs/reports/widerface_dataset_check_result.txt")


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def validate_xml(xml_path: Path) -> tuple[bool, str]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:  # noqa: BLE001
        return False, f"XML parse error: {exc}"

    size = root.find("size")
    if size is None:
        return False, "missing <size>"
    if size.findtext("width") is None or size.findtext("height") is None:
        return False, "missing width or height"
    if root.findtext("folder") is None:
        return False, "missing <folder>"
    if root.find("object") is None:
        return False, "no object annotations"
    return True, "ok"


def check_split(root: Path, split: str) -> dict[str, object]:
    id_file = root / f"{split}.txt"
    prefix = root / ("WIDER_train" if split == "train" else "WIDER_val")
    ann_dir = prefix / "Annotations"

    ids = read_ids(id_file)
    missing_xml: list[str] = []
    invalid_xml: list[str] = []

    for img_id in ids[:200]:
        xml_path = ann_dir / f"{img_id}.xml"
        if not xml_path.exists():
            missing_xml.append(str(xml_path))
            continue
        ok, reason = validate_xml(xml_path)
        if not ok:
            invalid_xml.append(f"{xml_path}: {reason}")

    return {
        "split": split,
        "id_file": str(id_file),
        "prefix": str(prefix),
        "annotation_dir": str(ann_dir),
        "id_file_exists": id_file.exists(),
        "prefix_exists": prefix.exists(),
        "annotation_dir_exists": ann_dir.exists(),
        "image_id_count": len(ids),
        "sample_checked": min(len(ids), 200),
        "missing_xml_count": len(missing_xml),
        "invalid_xml_count": len(invalid_xml),
        "missing_xml_examples": missing_xml[:5],
        "invalid_xml_examples": invalid_xml[:5],
    }


def format_result(root: Path, train: dict[str, object],
                  val: dict[str, object]) -> str:
    lines = [
        "WIDER FACE Dataset Check",
        "=" * 50,
        f"Expected root: {root}",
        "",
        "Expected MMDetection format:",
        "- train.txt and val.txt at dataset root",
        "- WIDER_train/Annotations/*.xml",
        "- WIDER_val/Annotations/*.xml",
        "- XML files in Pascal VOC style",
        "",
    ]
    for item in (train, val):
        lines.extend([
            f"{item['split']} split:",
            f"- id file: {item['id_file']}",
            f"- id file exists: {item['id_file_exists']}",
            f"- data prefix: {item['prefix']}",
            f"- data prefix exists: {item['prefix_exists']}",
            f"- annotation dir exists: {item['annotation_dir_exists']}",
            f"- image id count: {item['image_id_count']}",
            f"- XML samples checked: {item['sample_checked']}",
            f"- missing XML count in checked samples: {item['missing_xml_count']}",
            f"- invalid XML count in checked samples: {item['invalid_xml_count']}",
        ])
        missing = item["missing_xml_examples"]
        invalid = item["invalid_xml_examples"]
        if missing:
            lines.append("- missing XML examples:")
            lines.extend(f"  - {path}" for path in missing)
        if invalid:
            lines.append("- invalid XML examples:")
            lines.extend(f"  - {path}" for path in invalid)
        lines.append("")

    ok = all([
        train["id_file_exists"], train["prefix_exists"],
        train["annotation_dir_exists"], train["image_id_count"],
        train["missing_xml_count"] == 0, train["invalid_xml_count"] == 0,
        val["id_file_exists"], val["prefix_exists"],
        val["annotation_dir_exists"], val["image_id_count"],
        val["missing_xml_count"] == 0, val["invalid_xml_count"] == 0,
    ])
    lines.append(f"Result: {'Ready' if ok else 'Not ready'}")
    if not ok:
        lines.extend([
            "",
            "Next step:",
            "- Download WIDER FACE train/val images and annotations.",
            "- Convert official WIDER FACE txt annotations to Pascal VOC XML.",
            "- Place converted files under data/raw/WIDERFace.",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether WIDER FACE is ready for MMDetection.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    train = check_split(args.data_root, "train")
    val = check_split(args.data_root, "val")
    result = format_result(args.data_root, train, val)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(result, encoding="utf-8")
    print(result)
    print(f"\nReport saved to: {args.report}")


if __name__ == "__main__":
    main()
