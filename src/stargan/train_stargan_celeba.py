from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import make_grid, save_image


DEFAULT_ATTRS = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact StarGAN baseline on CelebA.")
    parser.add_argument("--celeba-root", type=Path, default=Path("/root/autodl-pub/CelebA"))
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--attr-path", type=Path, default=None)
    parser.add_argument("--selected-attrs", nargs="+", default=DEFAULT_ATTRS)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=20000)
    parser.add_argument("--sequential-subset", action="store_true", help="Use the first max-images samples instead of a seeded random subset.")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs to run in this invocation. With --resume, this means additional epochs.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--g-lr", type=float, default=1e-4)
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay-start-epoch", type=int, default=0, help="Start linear LR decay after this absolute epoch; 0 disables decay.")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-rec", type=float, default=10.0)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stargan"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/checkpoints/stargan"))
    parser.add_argument("--run-name", default="stargan_celeba_baseline")
    parser.add_argument("--resume", type=Path, default=None, help="Resume generator/discriminator and optimizer states from a checkpoint.")
    return parser.parse_args()


def resolve_celeba_paths(root: Path, image_dir: Path | None, attr_path: Path | None) -> tuple[Path, Path]:
    if image_dir is not None and attr_path is not None:
        return image_dir, attr_path

    image_candidates = [
        root / "img_align_celeba",
        root / "img_align_celeba_png",
        root / "Img" / "img_align_celeba",
        root / "Img" / "img_align_celeba_png",
        root / "CelebA" / "img_align_celeba",
        root / "CelebA" / "img_align_celeba_png",
        root / "celeba" / "img_align_celeba",
        root / "celeba" / "img_align_celeba_png",
        root / "images",
    ]
    attr_candidates = [
        root / "list_attr_celeba.txt",
        root / "Anno" / "list_attr_celeba.txt",
        root / "CelebA" / "list_attr_celeba.txt",
        root / "celeba" / "list_attr_celeba.txt",
    ]

    resolved_image_dir = image_dir
    if resolved_image_dir is None:
        for candidate in image_candidates:
            if candidate.exists():
                resolved_image_dir = candidate
                break
    resolved_attr_path = attr_path
    if resolved_attr_path is None:
        for candidate in attr_candidates:
            if candidate.exists():
                resolved_attr_path = candidate
                break

    if resolved_image_dir is None:
        found = sorted(root.rglob("img_align_celeba")) or sorted(root.rglob("img_align_celeba_png"))
        if found:
            resolved_image_dir = found[0]
    if resolved_attr_path is None:
        found = sorted(root.rglob("list_attr_celeba.txt"))
        if found:
            resolved_attr_path = found[0]

    if resolved_image_dir is None or not resolved_image_dir.exists():
        raise FileNotFoundError(f"CelebA image directory not found under {root}")
    if resolved_attr_path is None or not resolved_attr_path.exists():
        raise FileNotFoundError(f"CelebA attr file not found under {root}")
    return resolved_image_dir, resolved_attr_path


class CelebAStarGANDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        attr_path: Path,
        selected_attrs: list[str],
        image_size: int,
    ) -> None:
        self.image_dir = image_dir
        self.selected_attrs = selected_attrs
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        lines = attr_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 3:
            raise ValueError(f"Invalid CelebA attr file: {attr_path}")
        attr_names = lines[1].split()
        missing = [name for name in selected_attrs if name not in attr_names]
        if missing:
            raise ValueError(f"Selected attrs not found in CelebA annotations: {missing}")
        selected_indices = [attr_names.index(name) for name in selected_attrs]

        self.records: list[tuple[str, list[float]]] = []
        for line in lines[2:]:
            parts = line.split()
            filename = parts[0]
            values = [1.0 if int(parts[index + 1]) == 1 else 0.0 for index in selected_indices]
            image_path = self.resolve_image_path(filename)
            if image_path.exists():
                self.records.append((image_path.name, values))

        if not self.records:
            raise RuntimeError(f"No CelebA images matched attr file in {image_dir}")

    def resolve_image_path(self, filename: str) -> Path:
        image_path = self.image_dir / filename
        if image_path.exists():
            return image_path
        stem = Path(filename).stem
        for suffix in [".png", ".jpg", ".jpeg", ".webp"]:
            candidate = self.image_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        return image_path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        filename, attr = self.records[index]
        image = Image.open(self.image_dir / filename).convert("RGB")
        return self.transform(image), torch.tensor(attr, dtype=torch.float32), filename


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.main(x)


class Generator(nn.Module):
    def __init__(self, attr_dim: int, conv_dim: int = 64, repeat_num: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(3 + attr_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
        ]
        curr_dim = conv_dim
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(curr_dim * 2, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            curr_dim *= 2
        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr_dim))
        for _ in range(2):
            layers.extend(
                [
                    nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(curr_dim // 2, affine=True),
                    nn.ReLU(inplace=True),
                ]
            )
            curr_dim //= 2
        layers.extend([nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3), nn.Tanh()])
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, target_attr: torch.Tensor) -> torch.Tensor:
        attrs = target_attr.view(target_attr.size(0), target_attr.size(1), 1, 1)
        attrs = attrs.repeat(1, 1, x.size(2), x.size(3))
        return self.main(torch.cat([x, attrs], dim=1))


class Discriminator(nn.Module):
    def __init__(self, image_size: int, attr_dim: int, conv_dim: int = 64, repeat_num: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.01)]
        curr_dim = conv_dim
        for _ in range(1, repeat_num):
            layers.extend(
                [
                    nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.01),
                ]
            )
            curr_dim *= 2
        kernel_size = max(1, image_size // (2**repeat_num))
        self.main = nn.Sequential(*layers)
        self.src_head = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.cls_head = nn.Conv2d(curr_dim, attr_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.main(x)
        src = self.src_head(features)
        cls = self.cls_head(features).view(x.size(0), -1)
        return src, cls


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


def make_target_attrs(source_attr: torch.Tensor, selected_attrs: list[str]) -> torch.Tensor:
    target = source_attr.clone()
    hair_indices = [
        index
        for index, name in enumerate(selected_attrs)
        if name in {"Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"}
    ]
    non_hair_indices = [index for index in range(target.size(1)) if index not in hair_indices]

    for row in target:
        edit_hair = hair_indices and (not non_hair_indices or random.random() < 0.5)
        if edit_hair:
            chosen = random.choice(hair_indices)
            row[hair_indices] = 0.0
            row[chosen] = 1.0
        else:
            attr_index = random.choice(non_hair_indices or list(range(row.numel())))
            row[attr_index] = 1.0 - row[attr_index]
    return target


def adjust_learning_rate(optimizer: torch.optim.Optimizer, initial_lr: float, epoch: int, end_epoch: int, decay_start_epoch: int, min_lr: float) -> float:
    if decay_start_epoch <= 0 or epoch < decay_start_epoch or end_epoch <= decay_start_epoch:
        lr = initial_lr
    else:
        progress = min(1.0, (epoch - decay_start_epoch) / max(1, end_epoch - decay_start_epoch))
        lr = initial_lr - progress * (initial_lr - min_lr)
    lr = max(lr, min_lr)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def gradient_penalty(discriminator: Discriminator, real: torch.Tensor, fake: torch.Tensor, device: torch.device) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    mixed = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    out, _ = discriminator(mixed)
    grad = torch.autograd.grad(
        outputs=out,
        inputs=mixed,
        grad_outputs=torch.ones_like(out),
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(grad.size(0), -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def save_samples(generator: Generator, fixed_images: torch.Tensor, fixed_attrs: torch.Tensor, selected_attrs: list[str], output_path: Path) -> None:
    generator.eval()
    device = next(generator.parameters()).device
    with torch.no_grad():
        rows = [fixed_images.cpu()]
        for attr_index in range(len(selected_attrs)):
            target = fixed_attrs.clone()
            target[:, attr_index] = 1.0 - target[:, attr_index]
            rows.append(generator(fixed_images.to(device), target.to(device)).cpu())
    grid = make_grid(torch.cat(rows, dim=0), nrow=fixed_images.size(0), padding=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denorm(grid), output_path)
    generator.train()


def plot_history(history: list[dict], output_path: Path) -> None:
    if not history:
        return
    steps = [item["step"] for item in history]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for axis, key, title in [
        (axes[0], "d_loss", "Discriminator loss"),
        (axes[1], "g_loss", "Generator loss"),
        (axes[2], "d_cls", "D attr classification"),
        (axes[3], "g_rec", "Reconstruction loss"),
    ]:
        axis.plot(steps, [item[key] for item in history], linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("step")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    image_dir, attr_path = resolve_celeba_paths(args.celeba_root, args.image_dir, args.attr_path)
    dataset = CelebAStarGANDataset(image_dir, attr_path, args.selected_attrs, args.image_size)
    if args.max_images and args.max_images < len(dataset):
        if args.sequential_subset:
            indices = list(range(args.max_images))
        else:
            rng = np.random.default_rng(args.seed)
            indices = rng.permutation(len(dataset))[: args.max_images].tolist()
        dataset_for_loader: Dataset = Subset(dataset, indices)
    else:
        dataset_for_loader = dataset

    loader = DataLoader(
        dataset_for_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    sample_dir = args.output_dir / "images" / args.run_name
    report_dir = Path("outputs/reports")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    generator = Generator(len(args.selected_attrs)).to(device)
    discriminator = Discriminator(args.image_size, len(args.selected_attrs)).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2))

    fixed_images, fixed_attrs, _ = next(iter(loader))
    fixed_images = fixed_images[: min(8, fixed_images.size(0))].to(device)
    fixed_attrs = fixed_attrs[: fixed_images.size(0)].to(device)

    history: list[dict] = []
    started = time.perf_counter()
    total_steps = 0
    best_g_loss = float("inf")
    start_epoch = 1
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(resume_checkpoint["generator"])
        discriminator.load_state_dict(resume_checkpoint["discriminator"])
        if "g_optimizer" in resume_checkpoint:
            g_optimizer.load_state_dict(resume_checkpoint["g_optimizer"])
        if "d_optimizer" in resume_checkpoint:
            d_optimizer.load_state_dict(resume_checkpoint["d_optimizer"])
        total_steps = int(resume_checkpoint.get("step", 0))
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_g_loss = float(resume_checkpoint.get("best_g_loss", best_g_loss))
        print(f"Resumed from {args.resume}: start_epoch={start_epoch}, total_steps={total_steps}", flush=True)
    latest_checkpoint = args.checkpoint_dir / f"{args.run_name}_latest.pt"
    best_checkpoint = args.checkpoint_dir / f"{args.run_name}_best.pt"

    end_epoch = start_epoch + args.num_epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        current_g_lr = adjust_learning_rate(g_optimizer, args.g_lr, epoch, end_epoch, args.lr_decay_start_epoch, args.min_lr)
        current_d_lr = adjust_learning_rate(d_optimizer, args.d_lr, epoch, end_epoch, args.lr_decay_start_epoch, args.min_lr)
        for real_images, real_attrs, _filenames in loader:
            total_steps += 1
            real_images = real_images.to(device, non_blocking=True)
            real_attrs = real_attrs.to(device, non_blocking=True)
            target_attrs = make_target_attrs(real_attrs, args.selected_attrs).to(device)

            out_src, out_cls = discriminator(real_images)
            d_loss_real = -out_src.mean()
            d_loss_cls = classification_loss(out_cls, real_attrs)

            fake_images = generator(real_images, target_attrs)
            out_src_fake, _ = discriminator(fake_images.detach())
            d_loss_fake = out_src_fake.mean()
            d_loss_gp = gradient_penalty(discriminator, real_images, fake_images.detach(), device)
            d_loss = d_loss_real + d_loss_fake + args.lambda_cls * d_loss_cls + args.lambda_gp * d_loss_gp

            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()

            g_loss = torch.tensor(0.0, device=device)
            g_loss_fake = torch.tensor(0.0, device=device)
            g_loss_cls = torch.tensor(0.0, device=device)
            g_loss_rec = torch.tensor(0.0, device=device)
            if total_steps % args.n_critic == 0:
                fake_images = generator(real_images, target_attrs)
                out_src_fake, out_cls_fake = discriminator(fake_images)
                g_loss_fake = -out_src_fake.mean()
                g_loss_cls = classification_loss(out_cls_fake, target_attrs)
                rec_images = generator(fake_images, real_attrs)
                g_loss_rec = torch.mean(torch.abs(real_images - rec_images))
                g_loss = g_loss_fake + args.lambda_cls * g_loss_cls + args.lambda_rec * g_loss_rec

                g_optimizer.zero_grad(set_to_none=True)
                g_loss.backward()
                g_optimizer.step()

            if total_steps % 20 == 0:
                record = {
                    "epoch": epoch,
                    "step": total_steps,
                    "d_loss": float(d_loss.item()),
                    "d_cls": float(d_loss_cls.item()),
                    "d_gp": float(d_loss_gp.item()),
                    "g_loss": float(g_loss.item()),
                    "g_fake": float(g_loss_fake.item()),
                    "g_cls": float(g_loss_cls.item()),
                    "g_rec": float(g_loss_rec.item()),
                    "g_lr": float(current_g_lr),
                    "d_lr": float(current_d_lr),
                }
                history.append(record)
                print(record, flush=True)

            if total_steps % args.sample_every == 0:
                save_samples(
                    generator,
                    fixed_images,
                    fixed_attrs,
                    args.selected_attrs,
                    sample_dir / f"step_{total_steps:07d}.jpg",
                )

        checkpoint = {
            "epoch": epoch,
            "step": total_steps,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "selected_attrs": args.selected_attrs,
            "image_size": args.image_size,
            "run_name": args.run_name,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "best_g_loss": best_g_loss,
        }
        torch.save(checkpoint, latest_checkpoint)
        if history and history[-1]["g_loss"] < best_g_loss:
            best_g_loss = history[-1]["g_loss"]
            torch.save(checkpoint, best_checkpoint)
        if epoch % args.checkpoint_every == 0:
            torch.save(checkpoint, args.checkpoint_dir / f"{args.run_name}_epoch{epoch:03d}.pt")
        save_samples(generator, fixed_images, fixed_attrs, args.selected_attrs, sample_dir / f"epoch_{epoch:03d}.jpg")

    elapsed = time.perf_counter() - started
    curve_path = args.output_dir / "curves" / f"{args.run_name}_loss_curves.jpg"
    plot_history(history, curve_path)
    history_path = report_dir / f"{args.run_name}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    report = {
        "module": "CelebA StarGAN attribute editing baseline",
        "run_name": args.run_name,
        "celeba_root": str(args.celeba_root),
        "image_dir": str(image_dir),
        "attr_path": str(attr_path),
        "selected_attrs": args.selected_attrs,
        "dataset_images_total": len(dataset),
        "dataset_images_used": len(dataset_for_loader),
        "image_size": args.image_size,
        "epochs_this_run": args.num_epochs,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "resume": str(args.resume) if args.resume else None,
        "batch_size": args.batch_size,
        "device": str(device),
        "total_steps": total_steps,
        "elapsed_seconds": round(elapsed, 3),
        "latest_checkpoint": str(latest_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "sample_dir": str(sample_dir),
        "curve": str(curve_path),
        "history_json": str(history_path),
    }
    report_path = report_dir / f"{args.run_name}_result.txt"
    json_path = report_dir / f"{args.run_name}_result.json"
    lines = [
        "CelebA StarGAN Attribute Editing Baseline",
        "=" * 60,
        f"Run name: {args.run_name}",
        f"Image dir: {image_dir}",
        f"Attr path: {attr_path}",
        f"Selected attrs: {', '.join(args.selected_attrs)}",
        f"Images used: {len(dataset_for_loader)} / {len(dataset)}",
        f"Epochs this run: {args.num_epochs}",
        f"Epoch range: {start_epoch}-{end_epoch}",
        f"Resume: {args.resume if args.resume else 'none'}",
        f"Batch size: {args.batch_size}",
        f"Device: {device}",
        f"Total steps: {total_steps}",
        f"Elapsed seconds: {elapsed:.3f}",
        "",
        "Outputs:",
        f"- Latest checkpoint: {latest_checkpoint}",
        f"- Best checkpoint: {best_checkpoint}",
        f"- Samples: {sample_dir}",
        f"- Curves: {curve_path}",
        f"- History JSON: {history_path}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()




