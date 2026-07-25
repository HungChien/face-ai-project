from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from torch import nn

from src.recognition.evaluate_lfw_10fold_resnet_arcface import ResNetEmbedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamically quantize a project-trained ArcFace embedding model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/resnet50_ms1m10000_reproduce_server_best.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/checkpoints/resnet50_ms1m10000_reproduce_server_dynamic_quantized.pt"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/resnet50_ms1m10000_dynamic_quantization_report.txt"))
    parser.add_argument("--json-report", type=Path, default=Path("outputs/reports/resnet50_ms1m10000_dynamic_quantization_report.json"))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def build_model(checkpoint: dict, device: str = "cpu") -> nn.Module:
    backbone = str(checkpoint.get("backbone", "resnet50"))
    embedding_dim = int(checkpoint.get("embedding_dim", 512))
    model = ResNetEmbedding(backbone, embedding_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def benchmark(model: nn.Module, sample: torch.Tensor, warmup_iters: int, benchmark_iters: int) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(sample)
        started = time.perf_counter()
        for _ in range(benchmark_iters):
            _ = model(sample)
        elapsed = time.perf_counter() - started
    per_batch_ms = elapsed / benchmark_iters * 1000
    per_image_ms = per_batch_ms / sample.shape[0]
    return {"elapsed_seconds": elapsed, "per_batch_ms": per_batch_ms, "per_image_ms": per_image_ms}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_size = int(args.image_size or checkpoint.get("image_size", 112))
    fp32_model = build_model(checkpoint, "cpu")

    quantized_model = torch.quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model.eval()

    sample = torch.randn(args.batch_size, 3, image_size, image_size)
    with torch.no_grad():
        fp32_output = fp32_model(sample)
        quantized_output = quantized_model(sample)
    max_abs_diff = float((fp32_output - quantized_output).abs().max().item())
    mean_abs_diff = float((fp32_output - quantized_output).abs().mean().item())
    cosine_similarity = float(torch.nn.functional.cosine_similarity(fp32_output, quantized_output, dim=1).mean().item())

    fp32_speed = benchmark(fp32_model, sample, args.warmup_iters, args.benchmark_iters)
    quantized_speed = benchmark(quantized_model, sample, args.warmup_iters, args.benchmark_iters)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": quantized_model,
            "source_checkpoint": str(args.checkpoint),
            "backbone": checkpoint.get("backbone"),
            "embedding_dim": checkpoint.get("embedding_dim"),
            "image_size": image_size,
            "quantization": "torch.quantization.quantize_dynamic({nn.Linear}, dtype=torch.qint8)",
        },
        args.output,
    )

    fp32_size_mb = file_size_mb(args.checkpoint)
    quantized_size_mb = file_size_mb(args.output)
    size_reduction = 1.0 - quantized_size_mb / fp32_size_mb
    speedup = fp32_speed["per_image_ms"] / quantized_speed["per_image_ms"] if quantized_speed["per_image_ms"] > 0 else 0.0

    result = {
        "source_checkpoint": str(args.checkpoint),
        "quantized_model": str(args.output),
        "backbone": checkpoint.get("backbone"),
        "embedding_dim": checkpoint.get("embedding_dim"),
        "image_size": image_size,
        "batch_size": args.batch_size,
        "warmup_iters": args.warmup_iters,
        "benchmark_iters": args.benchmark_iters,
        "fp32_size_mb": fp32_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "size_reduction": size_reduction,
        "fp32_speed": fp32_speed,
        "quantized_speed": quantized_speed,
        "speedup": speedup,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "mean_cosine_similarity": cosine_similarity,
    }

    lines = [
        "ArcFace Dynamic Quantization Report",
        "=" * 48,
        f"Source checkpoint: {args.checkpoint}",
        f"Quantized model: {args.output}",
        f"Backbone: {result['backbone']}",
        f"Embedding dim: {result['embedding_dim']}",
        f"Image size: {image_size}",
        f"Batch size: {args.batch_size}",
        "",
        "Model size:",
        f"  FP32 checkpoint: {fp32_size_mb:.2f} MB",
        f"  Dynamic quantized: {quantized_size_mb:.2f} MB",
        f"  Size reduction: {size_reduction * 100:.2f}%",
        "",
        "CPU inference benchmark:",
        f"  FP32: {fp32_speed['per_image_ms']:.4f} ms/image ({fp32_speed['per_batch_ms']:.4f} ms/batch)",
        f"  Dynamic quantized: {quantized_speed['per_image_ms']:.4f} ms/image ({quantized_speed['per_batch_ms']:.4f} ms/batch)",
        f"  Speedup: {speedup:.3f}x",
        "",
        "Output consistency on random input:",
        f"  Mean cosine similarity: {cosine_similarity:.6f}",
        f"  Mean abs diff: {mean_abs_diff:.6f}",
        f"  Max abs diff: {max_abs_diff:.6f}",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.json_report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

