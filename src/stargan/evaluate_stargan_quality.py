from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision import models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate StarGAN generated image quality with FID and IS.")
    parser.add_argument("--grid", type=Path, default=Path("server_downloads/stargan_refine/stargan_celeba_attr5_refine/epoch_030.jpg"))
    parser.add_argument("--ncols", type=int, default=8)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inception-weights", choices=["auto", "none"], default="auto")
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/stargan_celeba_attr5_refine_quality_eval.txt"))
    parser.add_argument("--json-report", type=Path, default=Path("outputs/reports/stargan_celeba_attr5_refine_quality_eval.json"))
    parser.add_argument("--plot", type=Path, default=Path("outputs/stargan/curves/stargan_celeba_attr5_refine_quality_summary.jpg"))
    return parser.parse_args()


def split_torchvision_grid(grid_path: Path, ncols: int, padding: int) -> tuple[list[Image.Image], list[Image.Image], dict]:
    image = Image.open(grid_path).convert("RGB")
    width, height = image.size
    tile_w = (width - padding * (ncols + 1)) // ncols
    nrows = (height - padding) // (tile_w + padding)
    tile_h = (height - padding * (nrows + 1)) // nrows
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError(f"Cannot infer tiles from grid size {image.size}")

    tiles: list[Image.Image] = []
    for row in range(nrows):
        for col in range(ncols):
            x0 = padding + col * (tile_w + padding)
            y0 = padding + row * (tile_h + padding)
            tiles.append(image.crop((x0, y0, x0 + tile_w, y0 + tile_h)))

    real = tiles[:ncols]
    generated = tiles[ncols:]
    meta = {
        "grid_size": [width, height],
        "ncols": ncols,
        "nrows": nrows,
        "tile_size": [tile_w, tile_h],
        "real_count": len(real),
        "generated_count": len(generated),
    }
    return real, generated, meta


def load_inception(device: torch.device) -> torch.nn.Module:
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, transform_input=False)
    model.eval().to(device)
    return model


def collect_inception_stats(images: list[Image.Image], model: torch.nn.Module, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured.append(output.flatten(1).detach())

    handle = model.avgpool.register_forward_hook(hook)
    try:
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            batch = torch.stack([transform(image) for image in batch_images]).to(device)
            with torch.no_grad():
                captured.clear()
                out = model(batch)
                if isinstance(out, tuple):
                    out = out[0]
                features.append(captured[0].cpu().numpy())
                logits.append(out.detach().cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(features, axis=0), np.concatenate(logits, axis=0)


def frechet_distance(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(generated_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(generated_features, rowvar=False)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + eps) @ (sigma2 + eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def inception_score(logits: np.ndarray, splits: int = 5) -> tuple[float, float]:
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    split_count = min(splits, len(probs))
    scores = []
    for split in np.array_split(probs, split_count):
        py = np.mean(split, axis=0, keepdims=True)
        kl = split * (np.log(split + 1e-12) - np.log(py + 1e-12))
        scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return float(np.mean(scores)), float(np.std(scores))


def pixel_metrics(real: list[Image.Image], generated: list[Image.Image], ncols: int) -> dict:
    real_arrays = [np.asarray(image).astype(np.float32) / 255.0 for image in real]
    generated_arrays = [np.asarray(image).astype(np.float32) / 255.0 for image in generated]
    l1_values = []
    source_l2_values = []
    for index, generated_array in enumerate(generated_arrays):
        source = real_arrays[index % ncols]
        l1_values.append(float(np.mean(np.abs(generated_array - source))))
        source_l2_values.append(float(np.sqrt(np.mean((generated_array - source) ** 2))))
    flattened = np.stack([array.reshape(-1) for array in generated_arrays], axis=0)
    diversity = float(np.mean(np.std(flattened, axis=0)))
    return {
        "mean_l1_to_source": float(np.mean(l1_values)),
        "mean_rmse_to_source": float(np.mean(source_l2_values)),
        "pixel_diversity_std": diversity,
    }


def save_summary_plot(result: dict, output_path: Path) -> None:
    names = ["mean_l1_to_source", "mean_rmse_to_source", "pixel_diversity_std"]
    values = [result["auxiliary_metrics"][name] for name in names]
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(names, values, color=["#4f7cac", "#f08a5d", "#6a994e"])
    axis.set_title("StarGAN Quality Auxiliary Metrics")
    axis.set_ylabel("Value")
    axis.tick_params(axis="x", rotation=15)
    if result.get("fid") is not None and result.get("inception_score_mean") is not None:
        axis.text(
            0.02,
            0.95,
            f"FID={result['fid']:.3f}, IS={result['inception_score_mean']:.3f}",
            transform=axis.transAxes,
            va="top",
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    real, generated, grid_meta = split_torchvision_grid(args.grid, args.ncols, args.padding)

    result = {
        "module": "StarGAN generated image quality evaluation",
        "grid": str(args.grid),
        "device": str(device),
        "grid_meta": grid_meta,
        "fid": None,
        "inception_score_mean": None,
        "inception_score_std": None,
        "metric_note": "",
        "auxiliary_metrics": pixel_metrics(real, generated, args.ncols),
    }

    if args.inception_weights == "auto":
        try:
            model = load_inception(device)
            real_features, _real_logits = collect_inception_stats(real, model, device, args.batch_size)
            generated_features, generated_logits = collect_inception_stats(generated, model, device, args.batch_size)
            result["fid"] = frechet_distance(real_features, generated_features)
            is_mean, is_std = inception_score(generated_logits)
            result["inception_score_mean"] = is_mean
            result["inception_score_std"] = is_std
            result["metric_note"] = (
                "FID/IS computed with torchvision InceptionV3 ImageNet weights. "
                "The real set is the source row from the StarGAN sample grid; generated set is all edited rows."
            )
        except Exception as exc:  # noqa: BLE001
            result["metric_note"] = (
                "Official Inception FID/IS were not computed because pretrained InceptionV3 weights "
                f"could not be loaded locally: {type(exc).__name__}: {exc}. "
                "Auxiliary source-distance/diversity metrics were still computed."
            )
    else:
        result["metric_note"] = "Official Inception FID/IS disabled; auxiliary source-distance/diversity metrics computed."

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    save_summary_plot(result, args.plot)
    args.json_report.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "StarGAN Generated Image Quality Evaluation",
        "=" * 60,
        f"Grid: {args.grid}",
        f"Device: {device}",
        f"Real images: {grid_meta['real_count']}",
        f"Generated images: {grid_meta['generated_count']}",
        f"FID: {result['fid'] if result['fid'] is not None else 'unavailable'}",
        f"Inception Score: {result['inception_score_mean'] if result['inception_score_mean'] is not None else 'unavailable'}",
        f"Inception Score std: {result['inception_score_std'] if result['inception_score_std'] is not None else 'unavailable'}",
        "",
        "Auxiliary metrics:",
    ]
    for key, value in result["auxiliary_metrics"].items():
        lines.append(f"- {key}: {value:.6f}")
    lines.extend(["", f"Metric note: {result['metric_note']}", f"JSON report: {args.json_report}", f"Plot: {args.plot}"])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
