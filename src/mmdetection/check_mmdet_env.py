import platform
import sys
from pathlib import Path

import mmcv
import mmdet
import mmengine
import torch


def main() -> None:
    report_lines = [
        "MMDetection Environment Check",
        "=" * 50,
        f"Operating system: {platform.platform()}",
        f"Python executable: {sys.executable}",
        f"Python version: {sys.version}",
        f"PyTorch version: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]

    if torch.cuda.is_available():
        report_lines.extend(
            [
                f"CUDA version: {torch.version.cuda}",
                f"GPU name: {torch.cuda.get_device_name(0)}",
            ]
        )
    else:
        report_lines.extend(
            [
                "CUDA version: N/A",
                "GPU name: N/A",
            ]
        )

    report_lines.extend(
        [
            f"MMEngine version: {mmengine.__version__}",
            f"MMCV version: {mmcv.__version__}",
            f"MMDetection version: {mmdet.__version__}",
            "",
            "Result: Successful",
        ]
    )

    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "mmdetection_env_result.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
