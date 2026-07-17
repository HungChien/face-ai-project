# MS-Celeb-1M Cleaned Subset Preparation

## Recommended Source

Use the cleaned MS-Celeb-1M variant listed by the InsightFace project:

- Dataset: MS1M-ArcFace
- Scale: 85K identities / 5.8M images
- Source page: https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_
- Google Drive link listed by InsightFace: https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view?usp=sharing

This dataset is commonly used as a cleaned MS-Celeb-1M training set for ArcFace-style face recognition. Use it only for research/learning, keep it out of Git, and cite the dataset/source in reports.

## Expected Raw Layout

After downloading and extracting, InsightFace datasets usually contain RecordIO files:

```text
data/raw/ms1m-arcface-recordio/
  train.rec
  train.idx
  property
```

The project training script expects identity folders instead:

```text
data/raw/ms-celeb-1m-subset/
  identity_000001/
    00000001.jpg
    00000002.jpg
  identity_000002/
    00000001.jpg
```

## Convert RecordIO To Identity Folders

The conversion script is:

```text
src/datasets/convert_insightface_rec_to_folders.py
```

It reads `train.rec/train.idx`, groups images by label, then exports a manageable subset.

First scan the dataset without exporting:

```powershell
D:\Anaconda3\envs\ml-gpu\python.exe src\datasets\convert_insightface_rec_to_folders.py `
  --rec-root data\raw\ms1m-arcface-recordio `
  --output-root data\raw\ms-celeb-1m-subset `
  --max-identities 1000 `
  --min-images-per-identity 5 `
  --max-images-per-identity 30 `
  --dry-run `
  --summary outputs\reports\ms1m_recordio_dry_run_summary.json
```

Then export the subset:

```powershell
D:\Anaconda3\envs\ml-gpu\python.exe src\datasets\convert_insightface_rec_to_folders.py `
  --rec-root data\raw\ms1m-arcface-recordio `
  --output-root data\raw\ms-celeb-1m-subset `
  --max-identities 1000 `
  --min-images-per-identity 5 `
  --max-images-per-identity 30 `
  --summary outputs\reports\ms1m_recordio_conversion_summary.json
```

If `mxnet` is missing, install it in a separate conversion environment or in `ml-gpu` only if dependency resolution is clean. The training itself does not require `mxnet`; only RecordIO conversion does.

## Train ResNet50 + ArcFace On The Converted Subset

Run a small GPU smoke test first:

```powershell
D:\Anaconda3\envs\ml-gpu\python.exe src\recognition\train_arcface_celeba_subset.py `
  --dataset-format folder `
  --data-root data\raw\ms-celeb-1m-subset `
  --dataset-name MS-Celeb-1M-subset `
  --backbone resnet50 `
  --pretrained imagenet `
  --num-identities 100 `
  --min-train-images 5 `
  --max-train-images-per-identity 20 `
  --max-val-images-per-identity 5 `
  --epochs 2 `
  --batch-size 64 `
  --device cuda
```

Then run the larger baseline:

```powershell
D:\Anaconda3\envs\ml-gpu\python.exe src\recognition\train_arcface_celeba_subset.py `
  --dataset-format folder `
  --data-root data\raw\ms-celeb-1m-subset `
  --dataset-name MS-Celeb-1M-subset `
  --backbone resnet50 `
  --pretrained imagenet `
  --num-identities 1000 `
  --min-train-images 5 `
  --max-train-images-per-identity 30 `
  --max-val-images-per-identity 10 `
  --epochs 20 `
  --batch-size 64 `
  --device cuda `
  --output models\checkpoints\resnet50_imagenet_arcface_ms1m_subset_best.pt `
  --report outputs\reports\resnet50_imagenet_arcface_ms1m_subset_result.txt `
  --history-json outputs\reports\resnet50_imagenet_arcface_ms1m_subset_history.json `
  --curve outputs\images\resnet50_imagenet_arcface_ms1m_subset_curves.jpg
```

## Git Policy

Do not commit downloaded datasets or converted image folders:

```text
data/raw/*
data/processed/*
```

Only commit scripts, reports, and small visualizations.

## Local Conversion Result

Local raw dataset:

```text
data/raw/ms1m-arcface-recordio/faces_emore
```

RecordIO files:

```text
train.rec
train.idx
property
```

Dry-run scan:

```text
total_images_in_recordio: 5,822,653
total_identities_in_recordio: 85,742
selected_identities: 1,000
```

Exported subset:

```text
output_root: data/raw/ms-celeb-1m-subset
exported_identities: 1,000
exported_images: 30,000
max_images_per_identity: 30
min_images_per_identity: 5
```

Conversion reports:

```text
outputs/reports/ms1m_recordio_dry_run_summary.json
outputs/reports/ms1m_recordio_conversion_summary.json
```

GPU smoke training on the converted subset:

```text
Dataset: MS-Celeb-1M-subset-smoke
Device: cuda
Backbone: resnet50
Pretrained: imagenet
Identities: 100
Train/val images: 2000/500
Epochs: 2
Best val accuracy: 0.1831
```

Smoke outputs:

```text
models/checkpoints/resnet50_imagenet_arcface_ms1m_subset_smoke_best.pt
outputs/reports/resnet50_imagenet_arcface_ms1m_subset_smoke_result.txt
outputs/reports/resnet50_imagenet_arcface_ms1m_subset_smoke_history.json
outputs/images/resnet50_imagenet_arcface_ms1m_subset_smoke_curves.jpg
```
