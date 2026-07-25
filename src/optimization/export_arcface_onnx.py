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

from src.recognition.evaluate_lfw_10fold_resnet_arcface import ResNetEmbedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a project-trained ArcFace embedding model to ONNX and test ONNXRuntime inference.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/resnet50_ms1m10000_reproduce_server_best.pt"))
    parser.add_argument("--onnx-output", type=Path, default=Path("models/exported/resnet50_ms1m10000_reproduce_server.onnx"))
    parser.add_argument("--report", type=Path, default=Path("outputs/reports/resnet50_ms1m10000_onnx_export_report.txt"))
    parser.add_argument("--json-report", type=Path, default=Path("outputs/reports/resnet50_ms1m10000_onnx_export_report.json"))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def build_model(checkpoint: dict) -> torch.nn.Module:
    backbone = str(checkpoint.get("backbone", "resnet50"))
    embedding_dim = int(checkpoint.get("embedding_dim", 512))
    model = ResNetEmbedding(backbone, embedding_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def benchmark_torch(model: torch.nn.Module, sample: torch.Tensor, warmup_iters: int, benchmark_iters: int) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(sample)
        started = time.perf_counter()
        for _ in range(benchmark_iters):
            _ = model(sample)
        elapsed = time.perf_counter() - started
    per_batch_ms = elapsed / benchmark_iters * 1000
    return {"elapsed_seconds": elapsed, "per_batch_ms": per_batch_ms, "per_image_ms": per_batch_ms / sample.shape[0]}


def benchmark_onnx(session, sample_np: np.ndarray, warmup_iters: int, benchmark_iters: int) -> dict[str, float]:
    input_name = session.get_inputs()[0].name
    for _ in range(warmup_iters):
        _ = session.run(None, {input_name: sample_np})
    started = time.perf_counter()
    for _ in range(benchmark_iters):
        _ = session.run(None, {input_name: sample_np})
    elapsed = time.perf_counter() - started
    per_batch_ms = elapsed / benchmark_iters * 1000
    return {"elapsed_seconds": elapsed, "per_batch_ms": per_batch_ms, "per_image_ms": per_batch_ms / sample_np.shape[0]}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_size = int(args.image_size or checkpoint.get("image_size", 112))
    model = build_model(checkpoint)
    sample = torch.randn(args.batch_size, 3, image_size, image_size)

    args.onnx_output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        args.onnx_output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
        dynamo=False,
    )

    torch_speed = benchmark_torch(model, sample, args.warmup_iters, args.benchmark_iters)
    with torch.no_grad():
        torch_output = model(sample).detach().cpu().numpy()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required for validation. Install with: pip install onnxruntime") from exc

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.onnx_output), providers=providers)
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: sample.numpy().astype(np.float32)})[0]
    onnx_speed = benchmark_onnx(session, sample.numpy().astype(np.float32), args.warmup_iters, args.benchmark_iters)

    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    mean_abs_diff = float(np.mean(np.abs(torch_output - onnx_output)))
    cosine_similarity = float(np.mean(np.sum(torch_output * onnx_output, axis=1) / ((np.linalg.norm(torch_output, axis=1) * np.linalg.norm(onnx_output, axis=1)) + 1e-12)))
    speedup = torch_speed["per_image_ms"] / onnx_speed["per_image_ms"] if onnx_speed["per_image_ms"] > 0 else 0.0

    result = {
        "source_checkpoint": str(args.checkpoint),
        "onnx_model": str(args.onnx_output),
        "backbone": checkpoint.get("backbone"),
        "embedding_dim": checkpoint.get("embedding_dim"),
        "image_size": image_size,
        "batch_size": args.batch_size,
        "opset": args.opset,
        "onnx_size_mb": file_size_mb(args.onnx_output),
        "torch_speed": torch_speed,
        "onnx_speed": onnx_speed,
        "speedup": speedup,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "mean_cosine_similarity": cosine_similarity,
        "providers": providers,
    }

    lines = [
        "ArcFace ONNX Export Report",
        "=" * 48,
        f"Source checkpoint: {args.checkpoint}",
        f"ONNX model: {args.onnx_output}",
        f"Backbone: {result['backbone']}",
        f"Embedding dim: {result['embedding_dim']}",
        f"Image size: {image_size}",
        f"Batch size: {args.batch_size}",
        f"Opset: {args.opset}",
        f"ONNX size: {result['onnx_size_mb']:.2f} MB",
        "",
        "CPU inference benchmark:",
        f"  PyTorch: {torch_speed['per_image_ms']:.4f} ms/image ({torch_speed['per_batch_ms']:.4f} ms/batch)",
        f"  ONNXRuntime: {onnx_speed['per_image_ms']:.4f} ms/image ({onnx_speed['per_batch_ms']:.4f} ms/batch)",
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



