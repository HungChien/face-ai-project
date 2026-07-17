from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.config import Config


DEFAULT_CONFIG = Path("configs/mmdetection/widerface_retinanet_r50_fpn.py")
DEFAULT_WORK_DIR = Path("outputs/mmdetection_widerface/retinanet_r50_fpn")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the WIDER FACE RetinaNet baseline with MMDetection.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the work directory.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="Use cpu for smoke tests when the local CUDA stack is incompatible.")
    args = parser.parse_args()

    if args.device == "cpu":
        import torch

        torch.cuda.is_available = lambda: False  # type: ignore[method-assign]

    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(args.work_dir)
    cfg.resume = args.resume
    if args.device == "cpu":
        cfg.device = "cpu"

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()