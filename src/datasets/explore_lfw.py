import json
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people


DATA_HOME = Path("data/raw/sklearn")
OUTPUT_IMAGE_DIR = Path("outputs/images")
REPORT_DIR = Path("outputs/reports")


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def save_sample_grid(images: np.ndarray, names: np.ndarray, output_path: Path) -> None:
    sample_count = min(12, len(images))
    cols = 4
    rows = int(np.ceil(sample_count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = np.atleast_1d(axes).ravel()

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= sample_count:
            continue
        image = to_uint8_rgb(images[index])
        axis.imshow(image)
        axis.set_title(names[index], fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_opencv_sample_grid(images: np.ndarray, names: np.ndarray, output_path: Path) -> None:
    sample_count = min(12, len(images))
    tile_h, tile_w = 150, 120
    label_h = 32
    cols = 4
    rows = int(np.ceil(sample_count / cols))
    canvas = np.full((rows * (tile_h + label_h), cols * tile_w, 3), 255, dtype=np.uint8)

    for index in range(sample_count):
        row = index // cols
        col = index % cols
        image = to_uint8_rgb(images[index])
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        y0 = row * (tile_h + label_h)
        x0 = col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = image_bgr
        label = str(names[index])[:18]
        cv2.putText(
            canvas,
            label,
            (x0 + 4, y0 + tile_h + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def save_identity_distribution(name_counts: Counter, output_path: Path) -> None:
    most_common = name_counts.most_common(20)
    names = [item[0] for item in most_common]
    counts = [item[1] for item in most_common]

    fig, axis = plt.subplots(figsize=(12, 6))
    axis.barh(names[::-1], counts[::-1], color="#2f6f8f")
    axis.set_xlabel("Image count")
    axis.set_title("LFW top identities by image count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    dataset = fetch_lfw_people(
        data_home=str(DATA_HOME),
        color=True,
        resize=0.5,
        min_faces_per_person=1,
        funneled=True,
        download_if_missing=True,
    )
    elapsed = time.perf_counter() - start_time

    images = dataset.images
    targets = dataset.target
    target_names = dataset.target_names
    names = target_names[targets]
    name_counts = Counter(names.tolist())

    image_count = len(images)
    identity_count = len(target_names)
    image_shape = tuple(images[0].shape)
    counts = np.array(list(name_counts.values()))

    sample_grid_path = OUTPUT_IMAGE_DIR / "lfw_sample_grid.jpg"
    opencv_sample_grid_path = OUTPUT_IMAGE_DIR / "lfw_opencv_sample_grid.jpg"
    distribution_path = OUTPUT_IMAGE_DIR / "lfw_identity_distribution.jpg"
    report_path = REPORT_DIR / "lfw_exploration_result.txt"
    json_path = REPORT_DIR / "lfw_exploration_result.json"

    save_sample_grid(images, names, sample_grid_path)
    save_opencv_sample_grid(images, names, opencv_sample_grid_path)
    save_identity_distribution(name_counts, distribution_path)

    summary = {
        "dataset": "LFW people",
        "data_home": str(DATA_HOME),
        "image_count": int(image_count),
        "identity_count": int(identity_count),
        "image_shape": image_shape,
        "min_images_per_identity": int(counts.min()),
        "max_images_per_identity": int(counts.max()),
        "mean_images_per_identity": round(float(counts.mean()), 3),
        "median_images_per_identity": round(float(np.median(counts)), 3),
        "identities_with_one_image": int(np.sum(counts == 1)),
        "top_10_identities": [
            {"name": name, "image_count": int(count)}
            for name, count in name_counts.most_common(10)
        ],
        "matplotlib_sample_grid": str(sample_grid_path),
        "opencv_sample_grid": str(opencv_sample_grid_path),
        "identity_distribution": str(distribution_path),
        "elapsed_seconds": round(elapsed, 3),
    }

    report_lines = [
        "LFW Dataset Exploration",
        "=" * 50,
        f"Data home: {DATA_HOME}",
        f"Image count: {image_count}",
        f"Identity count: {identity_count}",
        f"Image shape: {image_shape}",
        f"Min images per identity: {counts.min()}",
        f"Max images per identity: {counts.max()}",
        f"Mean images per identity: {counts.mean():.3f}",
        f"Median images per identity: {np.median(counts):.3f}",
        f"Identities with one image: {int(np.sum(counts == 1))}",
        "",
        "Top 10 identities:",
    ]

    for name, count in name_counts.most_common(10):
        report_lines.append(f"- {name}: {count}")

    report_lines.extend(
        [
            "",
            "Visualization tools:",
            "- Matplotlib: sample grid and identity distribution chart.",
            "- OpenCV: image-level sample grid with text labels.",
            "",
            "Preprocessing notes:",
            "- LFW is identity-labeled and suitable for face verification tasks.",
            "- Images should be detected, aligned, resized, and normalized before recognition.",
            "- Pair-based evaluation should build positive pairs from the same identity and negative pairs from different identities.",
            "- Identities with only one image are useful for gallery statistics but cannot form positive verification pairs alone.",
            "",
            f"Matplotlib sample grid: {sample_grid_path}",
            f"OpenCV sample grid: {opencv_sample_grid_path}",
            f"Identity distribution: {distribution_path}",
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
