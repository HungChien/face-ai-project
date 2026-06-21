import json
import time
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CelebA


DATA_ROOT = Path("data/raw")
OUTPUT_IMAGE_DIR = Path("outputs/images")
REPORT_DIR = Path("outputs/reports")


def landmark_bbox(points: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    image_width, image_height = image_size
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    pad_x = width * 0.9
    pad_top = height * 1.35
    pad_bottom = height * 0.9

    x1 = max(0.0, min_x - pad_x)
    y1 = max(0.0, min_y - pad_top)
    x2 = min(float(image_width), max_x + pad_x)
    y2 = min(float(image_height), max_y + pad_bottom)
    return x1, y1, x2 - x1, y2 - y1


def save_celeba_annotation_grid(dataset: CelebA, output_path: Path) -> None:
    sample_indices = list(range(min(8, len(dataset))))
    cols = 4
    rows = int(np.ceil(len(sample_indices) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 8))
    axes = np.atleast_1d(axes).ravel()
    attr_names = [name for name in dataset.attr_names if name]

    for axis, index in zip(axes, sample_indices):
        image, target = dataset[index]
        attr, _bbox, landmarks = target
        axis.imshow(image)
        axis.axis("off")

        points = landmarks.numpy().astype(np.float32).reshape(-1, 2)
        x, y, width, height = landmark_bbox(points, image.size)
        rect = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=1.5,
            edgecolor="#20a39e",
            facecolor="none",
        )
        axis.add_patch(rect)
        axis.scatter(points[:, 0], points[:, 1], s=12, c="#ef5b5b")

        positive_attrs = [
            attr_names[attr_index]
            for attr_index, value in enumerate(attr)
            if int(value) == 1
        ][:3]
        axis.set_title(", ".join(positive_attrs) or "no positive attrs", fontsize=8)

    for axis in axes[len(sample_indices) :]:
        axis.axis("off")

    fig.suptitle(
        "CelebA aligned images: five landmarks plus landmark-derived face boxes",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_opencv_annotation_grid(dataset: CelebA, output_path: Path) -> None:
    sample_indices = list(range(min(8, len(dataset))))
    attr_names = [name for name in dataset.attr_names if name]
    tile_w, tile_h = 220, 220
    label_h = 38
    cols = 4
    rows = int(np.ceil(len(sample_indices) / cols))
    canvas = np.full((rows * (tile_h + label_h), cols * tile_w, 3), 255, dtype=np.uint8)

    for grid_index, dataset_index in enumerate(sample_indices):
        image, target = dataset[dataset_index]
        attr, _bbox, landmarks = target
        image_rgb = np.asarray(image.convert("RGB"))
        original_w, original_h = image.size
        scale_x = tile_w / original_w
        scale_y = tile_h / original_h
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

        points = landmarks.numpy().astype(np.float32).reshape(-1, 2)
        x, y, width, height = landmark_bbox(points, image.size)
        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + width) * scale_x))
        y2 = int(round((y + height) * scale_y))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (32, 163, 158), 2)

        for point in points:
            px = int(round(point[0] * scale_x))
            py = int(round(point[1] * scale_y))
            cv2.circle(image_bgr, (px, py), 3, (91, 91, 239), -1)

        row = grid_index // cols
        col = grid_index % cols
        y0 = row * (tile_h + label_h)
        x0 = col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = image_bgr
        positive_attrs = [
            attr_names[attr_index]
            for attr_index, value in enumerate(attr)
            if int(value) == 1
        ][:2]
        label = ", ".join(positive_attrs)[:28] or "no positive attrs"
        cv2.putText(
            canvas,
            label,
            (x0 + 4, y0 + tile_h + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def save_attribute_distribution(dataset: CelebA, output_path: Path) -> list[dict]:
    attr_names = [name for name in dataset.attr_names if name]
    attr_array = dataset.attr.numpy()
    positive_counts = (attr_array == 1).sum(axis=0)
    positive_rates = positive_counts / len(dataset)

    order = np.argsort(positive_rates)[::-1]
    top_indices = order[:20]
    names = [attr_names[index] for index in top_indices]
    rates = [positive_rates[index] for index in top_indices]

    fig, axis = plt.subplots(figsize=(12, 7))
    axis.barh(names[::-1], rates[::-1], color="#4f7cac")
    axis.set_xlabel("Positive rate")
    axis.set_title("CelebA top attributes by positive rate")
    axis.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return [
        {
            "attribute": attr_names[index],
            "positive_count": int(positive_counts[index]),
            "positive_rate": round(float(positive_rates[index]), 4),
        }
        for index in top_indices
    ]


def summarize_bbox(dataset: CelebA) -> dict:
    bbox = dataset.bbox.numpy()
    widths = bbox[:, 2]
    heights = bbox[:, 3]
    areas = widths * heights
    return {
        "mean_width": round(float(widths.mean()), 3),
        "mean_height": round(float(heights.mean()), 3),
        "median_width": round(float(np.median(widths)), 3),
        "median_height": round(float(np.median(heights)), 3),
        "mean_area": round(float(areas.mean()), 3),
    }


def summarize_landmarks(dataset: CelebA) -> dict:
    landmarks = dataset.landmarks_align.numpy().reshape(-1, 5, 2)
    names = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
    summary = {}
    for index, name in enumerate(names):
        points = landmarks[:, index, :]
        summary[name] = {
            "mean_x": round(float(points[:, 0].mean()), 3),
            "mean_y": round(float(points[:, 1].mean()), 3),
        }
    return summary


def main() -> None:
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    dataset = CelebA(
        root=str(DATA_ROOT),
        split="all",
        target_type=["attr", "bbox", "landmarks"],
        download=True,
    )
    elapsed = time.perf_counter() - start_time

    annotation_grid_path = OUTPUT_IMAGE_DIR / "celeba_annotation_examples.jpg"
    opencv_annotation_grid_path = OUTPUT_IMAGE_DIR / "celeba_opencv_annotation_examples.jpg"
    attr_distribution_path = OUTPUT_IMAGE_DIR / "celeba_attribute_distribution.jpg"
    report_path = REPORT_DIR / "celeba_exploration_result.txt"
    json_path = REPORT_DIR / "celeba_exploration_result.json"

    save_celeba_annotation_grid(dataset, annotation_grid_path)
    save_opencv_annotation_grid(dataset, opencv_annotation_grid_path)
    top_attributes = save_attribute_distribution(dataset, attr_distribution_path)
    bbox_summary = summarize_bbox(dataset)
    landmark_summary = summarize_landmarks(dataset)

    bbox_note = (
        "torchvision downloads img_align_celeba aligned images. The bbox annotation "
        "is kept for dataset-format study, but direct drawing can be offset if the "
        "bbox and the aligned image are not in the same coordinate system. The "
        "annotation grids therefore use landmark-derived face boxes for visualization."
    )

    summary = {
        "dataset": "CelebA",
        "data_root": str(DATA_ROOT),
        "image_count": len(dataset),
        "attribute_count": int(dataset.attr.shape[1]),
        "bbox_format": "[x, y, width, height]",
        "bbox_note": bbox_note,
        "landmark_format": "[left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y]",
        "bbox_summary": bbox_summary,
        "landmark_summary": landmark_summary,
        "top_20_attributes": top_attributes,
        "matplotlib_annotation_grid": str(annotation_grid_path),
        "opencv_annotation_grid": str(opencv_annotation_grid_path),
        "matplotlib_attribute_distribution": str(attr_distribution_path),
        "elapsed_seconds": round(elapsed, 3),
    }

    report_lines = [
        "CelebA Dataset Exploration",
        "=" * 50,
        f"Data root: {DATA_ROOT}",
        f"Image count: {len(dataset)}",
        f"Attribute count: {dataset.attr.shape[1]}",
        "",
        "Visualization tools:",
        "- Matplotlib: annotation examples and attribute distribution chart.",
        "- OpenCV: image-level landmark and face-box visualization.",
        "",
        "Annotation format:",
        "- bbox: [x, y, width, height]",
        "- landmarks: left eye, right eye, nose, left mouth, right mouth; each point is [x, y]",
        "- attributes: 40 binary labels, encoded as positive or negative facial attributes",
        "",
        "Coordinate-system note:",
        f"- {bbox_note}",
        "",
        "BBox summary:",
        f"- Mean width: {bbox_summary['mean_width']}",
        f"- Mean height: {bbox_summary['mean_height']}",
        f"- Median width: {bbox_summary['median_width']}",
        f"- Median height: {bbox_summary['median_height']}",
        f"- Mean area: {bbox_summary['mean_area']}",
        "",
        "Top 10 attributes by positive rate:",
    ]

    for item in top_attributes[:10]:
        report_lines.append(
            f"- {item['attribute']}: {item['positive_count']} ({item['positive_rate']})"
        )

    report_lines.extend(
        [
            "",
            "Preprocessing notes:",
            "- Detection task: use bbox only after confirming it matches the image coordinate system.",
            "- Landmark task: use five-point landmarks for face alignment.",
            "- Attribute task: resize images consistently and keep binary labels aligned with image filenames.",
            "- Recognition preprocessing: for img_align_celeba, prefer five-point landmarks for alignment/cropping visualization.",
            "- Keep original annotations unchanged; write processed crops to data/processed if needed.",
            "",
            f"Matplotlib annotation examples: {annotation_grid_path}",
            f"OpenCV annotation examples: {opencv_annotation_grid_path}",
            f"Attribute distribution: {attr_distribution_path}",
            f"Elapsed seconds: {elapsed:.3f}",
            "",
            "Result: Successful",
        ]
    )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nJSON report saved to: {json_path}")


if __name__ == "__main__":
    main()
