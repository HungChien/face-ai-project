# CelebA StarGAN Training

This note records the server-side commands for task 7.2: training a StarGAN baseline on CelebA for facial attribute editing.

## Dataset

The script expects the standard CelebA structure:

```text
CelebA/
  img_align_celeba/
    000001.jpg
    ...
  list_attr_celeba.txt
```

It also searches common public-data layouts such as:

```text
/root/autodl-pub/CelebA/Img/img_align_celeba
/root/autodl-pub/CelebA/Anno/list_attr_celeba.txt
```

Check the server dataset path first:

```bash
find /root/autodl-pub/CelebA -maxdepth 4 -iname "img_align_celeba" -o -iname "list_attr_celeba.txt"
```

If the paths are non-standard, pass them explicitly with `--image-dir` and `--attr-path`.

## Smoke Test

Use this first to verify dataloader, CUDA, checkpoint saving, and sample-image generation:

```bash
cd /root/autodl-tmp/face-ai-project

python -u src/stargan/train_stargan_celeba.py \
  --celeba-root /root/autodl-pub/CelebA \
  --selected-attrs Black_Hair Blond_Hair Brown_Hair Male Young \
  --image-size 128 \
  --max-images 1000 \
  --num-epochs 1 \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda \
  --sample-every 50 \
  --run-name stargan_celeba_smoke \
  2>&1 | tee outputs/reports/stargan_celeba_smoke_console.log
```

Expected outputs:

```text
models/checkpoints/stargan/stargan_celeba_smoke_latest.pt
models/checkpoints/stargan/stargan_celeba_smoke_best.pt
outputs/stargan/images/stargan_celeba_smoke/
outputs/stargan/curves/stargan_celeba_smoke_loss_curves.jpg
outputs/reports/stargan_celeba_smoke_result.txt
outputs/reports/stargan_celeba_smoke_result.json
```

## Baseline Training

After the smoke test succeeds, run a larger baseline:

```bash
python -u src/stargan/train_stargan_celeba.py \
  --celeba-root /root/autodl-pub/CelebA \
  --selected-attrs Black_Hair Blond_Hair Brown_Hair Male Young \
  --image-size 128 \
  --max-images 50000 \
  --num-epochs 10 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda \
  --sample-every 500 \
  --checkpoint-every 1 \
  --run-name stargan_celeba_attr5_baseline \
  2>&1 | tee outputs/reports/stargan_celeba_attr5_baseline_console.log
```

For a full-data run, set `--max-images 0`.

## Sampling

Use a trained generator to edit one CelebA image:

```bash
python src/stargan/sample_stargan_celeba.py \
  --checkpoint models/checkpoints/stargan/stargan_celeba_attr5_baseline_best.pt \
  --image /root/autodl-pub/CelebA/img_align_celeba/000001.jpg \
  --device cuda \
  --output outputs/stargan/images/stargan_celeba_attr5_sample.jpg
```

If the public dataset uses a different image path, replace `--image` with an existing image from the server.

## Attributes

Default selected attributes:

```text
Black_Hair, Blond_Hair, Brown_Hair, Male, Young
```

These cover the requested hair color, gender, and age editing directions.

## Continued Quality Run

If the baseline already produced a usable checkpoint but the edited images are still blurry or contain artifacts, continue from the best checkpoint with a larger random subset and more epochs:

```bash
python -u src/stargan/train_stargan_celeba.py \
  --image-dir /root/autodl-tmp/celeba/img_align_celeba_png \
  --attr-path /root/autodl-pub/CelebA/Anno/list_attr_celeba.txt \
  --selected-attrs Black_Hair Blond_Hair Brown_Hair Male Young \
  --image-size 128 \
  --max-images 100000 \
  --num-epochs 20 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda \
  --sample-every 500 \
  --checkpoint-every 1 \
  --g-lr 5e-5 \
  --d-lr 5e-5 \
  --lr-decay-start-epoch 20 \
  --min-lr 1e-6 \
  --resume models/checkpoints/stargan/stargan_celeba_attr5_baseline_latest.pt \
  --run-name stargan_celeba_attr5_refine \
  2>&1 | tee outputs/reports/stargan_celeba_attr5_refine_console.log
```

This run keeps the aligned PNG dataset, resumes the generator/discriminator weights, samples a broader random subset, uses mutually exclusive hair-color targets, and decays the learning rate late in training.

