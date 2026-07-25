from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from train_stargan_celeba import DEFAULT_ATTRS, Generator, denorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CelebA StarGAN attribute editing with a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--source-attrs", nargs="+", type=int, default=None)
    parser.add_argument("--selected-attrs", nargs="+", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/stargan/images/sample_stargan_edit.jpg"))
    parser.add_argument("--json-report", type=Path, default=Path("outputs/reports/stargan_sample_result.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    selected_attrs = args.selected_attrs or checkpoint.get("selected_attrs", DEFAULT_ATTRS)
    image_size = int(checkpoint.get("image_size", 128))
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    generator = Generator(len(selected_attrs)).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    transform = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(args.image).convert("RGB")
    source = transform(image).unsqueeze(0).to(device)
    if args.source_attrs is None:
        source_attrs = torch.zeros(1, len(selected_attrs), device=device)
    else:
        source_attrs = torch.tensor(args.source_attrs, dtype=torch.float32, device=device).view(1, -1)
        if source_attrs.size(1) != len(selected_attrs):
            raise ValueError("--source-attrs length must match selected attrs")

    outputs = [source.cpu()]
    edit_names = ["original"]
    with torch.no_grad():
        for attr_index, attr_name in enumerate(selected_attrs):
            target = source_attrs.clone()
            target[:, attr_index] = 1.0 - target[:, attr_index]
            edited = generator(source, target)
            outputs.append(edited.cpu())
            edit_names.append(f"toggle_{attr_name}")

    grid = make_grid(torch.cat(outputs, dim=0), nrow=len(outputs), padding=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(denorm(grid), args.output)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "image": str(args.image),
                "selected_attrs": selected_attrs,
                "edits": edit_names,
                "output": str(args.output),
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved StarGAN edit grid to: {args.output}")


if __name__ == "__main__":
    main()
