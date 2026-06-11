import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def main():
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA version: N/A")
        print("GPU name: N/A")

    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")

    x = torch.randn(2, 3)
    print("\nTorch tensor test:")
    print(x)

    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "env_check_result.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Environment Check\n")
        f.write("=" * 50 + "\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")

        if torch.cuda.is_available():
            f.write(f"CUDA version: {torch.version.cuda}\n")
            f.write(f"GPU name: {torch.cuda.get_device_name(0)}\n")
        else:
            f.write("CUDA version: N/A\n")
            f.write("GPU name: N/A\n")

        f.write(f"OpenCV version: {cv2.__version__}\n")
        f.write(f"NumPy version: {np.__version__}\n")

    print(f"\nEnvironment report saved to: {report_path}")


if __name__ == "__main__":
    main()