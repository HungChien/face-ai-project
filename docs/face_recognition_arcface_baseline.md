# Face Recognition ArcFace Baseline

This note records the ArcFace smoke baseline for face recognition model training.

## Goal

Build a minimal trainable face recognition pipeline before scaling to ResNet50 and larger identity datasets:

```text
LFW identity subset -> ResNet18 embedding -> ArcFace head -> identity classification
```

This is a smoke baseline, not the final target model.

## Main Script

```text
src/recognition/train_resnet18_arcface_smoke.py
```

## Run Command

```powershell
conda activate ml-gpu
python src\recognition\train_resnet18_arcface_smoke.py --epochs 20 --num-identities 10 --max-images-per-identity 20 --batch-size 16 --device auto --output models\checkpoints\resnet18_arcface_lfw_smoke10_best.pt --report outputs\reports\resnet18_arcface_lfw_smoke10_result.txt --history-json outputs\reports\resnet18_arcface_lfw_smoke10_history.json --curve outputs\images\resnet18_arcface_lfw_smoke10_curves.jpg
```

## Current Smoke Result

```text
Dataset: LFW funneled identity subset
Backbone: ResNet18, randomly initialized
Head/loss: ArcFace, scale=32, margin=0.5
Embedding dim: 128
Identities: 10
Train/val images: 160/40
Epochs: 20
Best epoch: 13
Best val accuracy: 0.7292
Checkpoint: models/checkpoints/resnet18_arcface_lfw_smoke10_best.pt
Curve: outputs/images/resnet18_arcface_lfw_smoke10_curves.jpg
```

## Interpretation

The smoke baseline verifies that the project can train a face recognition model with an ArcFace objective on GPU. Because this run uses only 10 identities and a randomly initialized ResNet18, it should be treated as a pipeline check rather than a final recognition system.

The earlier 30-identity, 5-epoch run was too hard for a from-scratch smoke setting and produced near-zero validation accuracy. The 10-identity, 20-epoch run is the useful baseline to report.

## Next Step

Use this script as the base for:

```text
1. ResNet50 backbone
2. larger identity subset from CelebA identity labels or cleaned MS-Celeb subset
3. LFW pair verification using learned embeddings
4. loss/accuracy curve comparison against pretrained InsightFace baseline
```

## LFW Pair Verification With Smoke Checkpoint

The script `src/recognition/evaluate_lfw_pairs_resnet18_arcface.py` evaluates the trained ResNet18 + ArcFace smoke checkpoint on the official LFW pair protocol.

Run command:

```powershell
conda activate ml-gpu
python src\recognition\evaluate_lfw_pairs_resnet18_arcface.py --checkpoint models\checkpoints\resnet18_arcface_lfw_smoke10_best.pt --device auto
```

Current result:

```text
Train pair file: pairsDevTrain.txt (1100)
Test pair file: pairsDevTest.txt (500)
Train pairs scored: 2200
Test pairs scored: 1000
Selected threshold: 0.3325
Train accuracy at selected threshold: 0.6436
Test accuracy: 0.6200
Correct / total: 620 / 1000
TP: 302
TN: 318
FP: 182
FN: 198
Unique embeddings cached: 4992
```

Comparison:

```text
InsightFace pretrained LFW accuracy: 0.9440
Project ResNet18 + ArcFace smoke accuracy: 0.6200
```

This is expected for a randomly initialized ResNet18 trained on only 10 LFW identities. The result is useful because it verifies the full training-to-verification pipeline. Accuracy should improve by using a larger identity dataset, longer training, ResNet50, stronger augmentation, and a proper train/validation protocol that avoids evaluating on identities unseen by the smoke subset.

## ResNet50 + ArcFace CelebA200 Training

The script `src/recognition/train_arcface_celeba_subset.py` trains a larger ArcFace baseline on a CelebA identity subset. This is the first non-smoke training run for ArcFace training workflow.

Training command:

```powershell
conda activate ml-gpu
python src\recognition\train_arcface_celeba_subset.py --backbone resnet50 --epochs 10 --num-identities 200 --max-train-images-per-identity 30 --max-val-images-per-identity 10 --batch-size 64 --device auto --split-mode random --output models\checkpoints\resnet50_arcface_celeba200_best.pt --report outputs\reports\resnet50_arcface_celeba200_result.txt --history-json outputs\reports\resnet50_arcface_celeba200_history.json --curve outputs\images\resnet50_arcface_celeba200_curves.jpg
```

Training result:

```text
Dataset: CelebA identity subset
Backbone: ResNet50
Head/loss: ArcFace
Embedding dim: 256
Identities: 200
Split mode: random per identity
Train/val images: 4833/1205
Epochs: 10
Best epoch: 10
Best val accuracy: 0.1426
Checkpoint: models/checkpoints/resnet50_arcface_celeba200_best.pt
Curve: outputs/images/resnet50_arcface_celeba200_curves.jpg
```

LFW pair verification command:

```powershell
conda activate ml-gpu
python src\recognition\evaluate_lfw_pairs_resnet18_arcface.py --checkpoint models\checkpoints\resnet50_arcface_celeba200_best.pt --device auto --report outputs\reports\resnet50_arcface_celeba200_lfw_pair_verification_result.txt --json-report outputs\reports\resnet50_arcface_celeba200_lfw_pair_verification_result.json
```

LFW result:

```text
Train accuracy at selected threshold: 0.6232
Test accuracy: 0.6380
Correct / total: 638 / 1000
TP: 275
TN: 363
FP: 137
FN: 225
```

Interpretation:

The model is learning identity-discriminative features, but 10 epochs on 200 CelebA identities is still far from the 98.5% LFW target. It is useful as the first full training baseline because it uses ResNet50, ArcFace, a larger identity set, train/validation curves, checkpointing, and LFW verification.

## ResNet50 + ArcFace CelebA1000 E20 Training

Expanded training run with more identities and epochs.

Training command:

```powershell
conda activate ml-gpu
python src\recognition\train_arcface_celeba_subset.py --backbone resnet50 --epochs 20 --num-identities 1000 --max-train-images-per-identity 30 --max-val-images-per-identity 10 --batch-size 128 --device auto --split-mode random --output models\checkpoints\resnet50_arcface_celeba1000_e20_best.pt --report outputs\reports\resnet50_arcface_celeba1000_e20_result.txt --history-json outputs\reports\resnet50_arcface_celeba1000_e20_history.json --curve outputs\images\resnet50_arcface_celeba1000_e20_curves.jpg
```

Training result:

```text
Dataset: CelebA identity subset
Backbone: ResNet50
Head/loss: ArcFace
Embedding dim: 256
Identities: 1000
Train/val images: 24033/6005
Epochs: 20
Best epoch: 17
Best val accuracy: 0.0185
Checkpoint: models/checkpoints/resnet50_arcface_celeba1000_e20_best.pt
Curve: outputs/images/resnet50_arcface_celeba1000_e20_curves.jpg
```

LFW pair verification result:

```text
Selected threshold: 0.9900
Train accuracy at selected threshold: 0.6405
Test accuracy: 0.6640
Correct / total: 664 / 1000
TP: 297
TN: 367
FP: 133
FN: 203
```

Comparison:

```text
ResNet18 LFW10 smoke LFW accuracy: 0.6200
ResNet50 CelebA200 E10 LFW accuracy: 0.6380
ResNet50 CelebA1000 E20 LFW accuracy: 0.6640
InsightFace pretrained LFW accuracy: 0.9440
```

Interpretation:

Increasing identities and epochs improves LFW verification, but the gain is still modest. The positive and negative pair cosine distributions remain heavily overlapped, indicating that randomly initialized ResNet50 still needs substantially more training, better data, or pretrained initialization to approach the 98.5% target.

## ImageNet-Initialized ResNet50 + ArcFace CelebA1000 E20

Training command:

```powershell
conda activate ml-gpu
python src\recognition\train_arcface_celeba_subset.py --backbone resnet50 --pretrained imagenet --epochs 20 --num-identities 1000 --max-train-images-per-identity 30 --max-val-images-per-identity 10 --batch-size 128 --device auto --split-mode random --output models\checkpoints\resnet50_imagenet_arcface_celeba1000_e20_best.pt --report outputs\reports\resnet50_imagenet_arcface_celeba1000_e20_result.txt --history-json outputs\reports\resnet50_imagenet_arcface_celeba1000_e20_history.json --curve outputs\images\resnet50_imagenet_arcface_celeba1000_e20_curves.jpg
```

Training result:

```text
Backbone: ResNet50
Pretrained: ImageNet
Head/loss: ArcFace
Embedding dim: 256
Identities: 1000
Train/val images: 24033/6005
Epochs: 20
Best epoch: 11
Best val accuracy: 0.7083
Checkpoint: models/checkpoints/resnet50_imagenet_arcface_celeba1000_e20_best.pt
```

LFW pair verification result:

```text
Selected threshold: 0.1954
Train accuracy at selected threshold: 0.7995
Test accuracy: 0.8080
Correct / total: 808 / 1000
TP: 395
TN: 413
FP: 87
FN: 105
```

Updated comparison:

```text
ResNet18 LFW10 smoke LFW accuracy: 0.6200
ResNet50 CelebA200 E10 random init LFW accuracy: 0.6380
ResNet50 CelebA1000 E20 random init LFW accuracy: 0.6640
ResNet50 CelebA1000 E20 ImageNet init LFW accuracy: 0.8080
InsightFace pretrained LFW accuracy: 0.9440
```

Interpretation:

ImageNet initialization is the strongest improvement so far. Under the same CelebA1000/E20 setting, validation classification accuracy improves from 0.0185 to 0.7083, and LFW verification improves from 0.6640 to 0.8080. The remaining gap to the 98.5% target likely requires face-domain pretraining, cleaner/larger identity data, stronger training schedules, and a stricter face alignment/normalization pipeline.

## Facenet-PyTorch VGGFace2 LFW 10-Fold Evaluation

Independent environment: `ml-face` at `D:\Anaconda3\envs\ml-face\python.exe`.

Script: `src/recognition/evaluate_lfw_10fold_facenet.py`

Command:

```bash
D:\Anaconda3\envs\ml-face\python.exe src\recognition\evaluate_lfw_10fold_facenet.py --batch-size 64
```

Protocol: LFW official `pairs.txt`, 6000 pairs, 10-fold cross validation. Each fold uses the other nine folds for threshold selection and evaluates on the held-out fold.

Result:

```text
Model: facenet-pytorch InceptionResnetV1 pretrained=vggface2
Device: CPU
Pairs: 6000
Unique images: 7701
Mean accuracy: 0.9677
Std accuracy: 0.0070
Total correct: 5806/6000
```

Outputs:

```text
outputs/reports/facenet_vggface2_lfw_10fold_result.txt
outputs/reports/facenet_vggface2_lfw_10fold_result.json
outputs/embeddings/facenet_vggface2_lfw_embeddings.npz
```

This is a real full-protocol LFW result and is more comparable to the 98.5% target than the earlier 1000-pair DevTest checks. It should be reported as a face-domain pretrained reference baseline, not as the final self-trained ResNet50 + ArcFace model. The current gap to the 98.5% target is 1.73 percentage points.

## ResNet50 + ArcFace MS-Celeb-1M Subset E20 Training

A cleaned MS-Celeb-1M subset was prepared from InsightFace `faces_emore` / MS1M-ArcFace RecordIO and exported to identity folders.

Dataset:

```text
data/raw/ms-celeb-1m-subset
identities: 1000
exported images: 30000
train/val images used by training: 24000/6000
```

Training command:

```bash
D:\Anaconda3\envs\ml-gpu\python.exe src\recognition\train_arcface_celeba_subset.py --dataset-format folder --data-root data\raw\ms-celeb-1m-subset --dataset-name MS-Celeb-1M-subset --backbone resnet50 --pretrained imagenet --num-identities 1000 --min-train-images 5 --max-train-images-per-identity 30 --max-val-images-per-identity 10 --epochs 20 --batch-size 64 --device cuda
```

Training result:

```text
Device: cuda
Backbone: resnet50
Pretrained: imagenet
Embedding dim: 256
Epochs: 20
Best epoch: 10
Best val accuracy: 0.7665
Elapsed seconds: 579.437
```

Outputs:

```text
models/checkpoints/resnet50_imagenet_arcface_ms1m_subset_best.pt
outputs/reports/resnet50_imagenet_arcface_ms1m_subset_result.txt
outputs/reports/resnet50_imagenet_arcface_ms1m_subset_history.json
outputs/images/resnet50_imagenet_arcface_ms1m_subset_curves.jpg
```

LFW pair DevTest result:

```text
Train threshold file: pairsDevTrain.txt
Test file: pairsDevTest.txt
Test accuracy: 0.7210
Correct / total: 721 / 1000
```

LFW official 6000 pairs / 10-fold result:

```text
Mean accuracy: 0.6975
Std accuracy: 0.0128
Total correct: 4185/6000
```

10-fold outputs:

```text
src/recognition/evaluate_lfw_10fold_resnet_arcface.py
outputs/reports/resnet50_imagenet_arcface_ms1m_subset_lfw_10fold_result.txt
outputs/reports/resnet50_imagenet_arcface_ms1m_subset_lfw_10fold_result.json
```

Interpretation: the model fits the selected MS1M training identities, but LFW verification remains far below the 98.5% target. The likely bottlenecks are the small 1000-identity/30-image subset, direct resize instead of a unified detect-align training pipeline, 256-d embedding, no learning-rate schedule, and no hard-sample mining or larger training duration.

## MS1M Aligned-112 Preprocessing And Baseline

The downloaded InsightFace `faces_emore` / MS1M-ArcFace images were verified to already be ArcFace-style aligned 112x112 faces. A processed training folder was generated to make the alignment/preprocessing stage explicit and reproducible.

Preprocessing script:

```text
src/datasets/prepare_ms1m_aligned_112.py
```

Preprocessing result:

```text
Input root: data/raw/ms-celeb-1m-subset
Output root: data/processed/ms1m-aligned-112
Input identities: 1000
Output identities: 1000
Output images: 30000
Copied images: 30000
Resized images: 0
Unreadable images: 0
Original shapes: {'112x112': 30000}
```

Preprocessing outputs:

```text
outputs/reports/ms1m_aligned_112_preprocess_result.txt
outputs/reports/ms1m_aligned_112_preprocess_result.json
outputs/images/ms1m_aligned_112_preview.jpg
```

Aligned baseline training:

```text
Dataset: MS1M-aligned-112
Device: cuda
Backbone: resnet50
Pretrained: imagenet
Embedding dim: 256
Identities: 1000
Train/val images: 24000/6000
Epochs: 20
Best epoch: 10
Best val accuracy: 0.7595
Elapsed seconds: 575.980
```

LFW official 6000 pairs / 10-fold result:

```text
Mean accuracy: 0.7075
Std accuracy: 0.0092
Total correct: 4245/6000
```

Compared with the previous raw-folder MS1M baseline, LFW 10-fold accuracy improved from 0.6975 to 0.7075. The small gain confirms the data was already mostly aligned; further improvement likely requires larger identity coverage, 512-d embeddings, a learning-rate schedule, stronger face recognition training strategy, and matching LFW alignment preprocessing.

## MS1M Aligned-112 Embedding-512 Baseline

A second aligned MS1M baseline was trained with the same data and settings as the 256-d run, changing only the embedding dimension from 256 to 512.

Training result:

```text
Dataset: MS1M-aligned-112-emb512
Device: cuda
Backbone: resnet50
Pretrained: imagenet
Embedding dim: 512
Identities: 1000
Train/val images: 24000/6000
Epochs: 20
Best epoch: 9
Best val accuracy: 0.7746
Elapsed seconds: 585.744
```

LFW official 6000 pairs / 10-fold result:

```text
Mean accuracy: 0.7027
Std accuracy: 0.0197
Total correct: 4216/6000
```

Comparison with aligned 256-d baseline:

```text
256-d val accuracy: 0.7595
512-d val accuracy: 0.7746
256-d LFW 10-fold: 0.7075
512-d LFW 10-fold: 0.7027
```

Interpretation: increasing the embedding dimension improves identity classification validation accuracy on the selected MS1M subset, but it does not improve LFW verification in this setting. The current bottleneck is more likely data scale, optimization strategy, and train/eval alignment consistency than embedding capacity alone.

## MS1M Aligned-112 5000-Identity Embedding-512 Baseline

The MS1M-ArcFace subset was expanded from 1000 identities to 5000 identities while keeping at most 30 images per identity. The model kept the aligned-112 input, ResNet50 backbone, ImageNet initialization, ArcFace head, and 512-d embedding.

Dataset conversion result:

```text
Raw output root: data/raw/ms-celeb-1m-subset-5000
Processed output root: data/processed/ms1m-aligned-112-5000
Identities: 5000
Images: 150000
Original shapes: {'112x112': 150000}
Unreadable images: 0
```

Training result:

```text
Dataset: MS1M-aligned-112-5000-emb512
Device: cuda
Backbone: resnet50
Pretrained: imagenet
Embedding dim: 512
Identities: 5000
Train/val images: 120000/30000
Epochs: 20
Best epoch: 6
Best val accuracy: 0.9497
Elapsed seconds: 2937.454
```

LFW official 6000 pairs / 10-fold result:

```text
Mean accuracy: 0.7407
Std accuracy: 0.0228
Total correct: 4444/6000
```

Comparison:

```text
1000 identities, 512-d: val=0.7746, LFW=0.7027
5000 identities, 512-d: val=0.9497, LFW=0.7407
```

Interpretation: increasing identity coverage substantially improves both subset validation accuracy and LFW verification. However, the LFW result is still far below the 98.5% target, so the next bottlenecks are likely training strategy, larger class coverage, batch sampling, learning-rate scheduling, stronger data augmentation, and consistent LFW aligned preprocessing.

## MS1M 5000-Identity Strategy Experiments

Two training-strategy experiments were run after the 5000-identity embedding-512 baseline.

### PK sampler + strong augmentation + cosine LR

Configuration:

```text
Dataset: MS1M-aligned-112-5000-emb512-strategy
Sampler: pk
P/K: 16 identities x 4 images
Scheduler: cosine
Warmup epochs: 2
Base/min LR: 0.0005 / 0.00001
Weight decay: 0.0005
Strong augment: true
Erase prob: 0.1
```

Result:

```text
Best epoch: 20
Best val accuracy: 0.0005
```

Interpretation: this setting failed to converge for the current ArcFace classification implementation. It should not be used as the main training recipe without further debugging. The likely causes are an overly aggressive combination of PK sampling, augmentation, and LR/weight decay for the present softmax-classification setup.

### Random sampler + cosine LR + light random erasing

Configuration:

```text
Dataset: MS1M-aligned-112-5000-emb512-cosine
Sampler: random
Scheduler: cosine
Warmup epochs: 2
Base/min LR: 0.0003 / 0.00001
Weight decay: 0.0001
Strong augment: false
Erase prob: 0.05
```

Training result:

```text
Best epoch: 19
Best val accuracy: 0.9578
Elapsed seconds: 11410.176
```

LFW official 6000 pairs / 10-fold result:

```text
Mean accuracy: 0.6970
Std accuracy: 0.0225
Total correct: 4182/6000
```

Comparison:

```text
5000 identities baseline: val=0.9497, LFW=0.7407
5000 identities cosine:   val=0.9578, LFW=0.6970
```

Interpretation: cosine LR improves closed-set validation accuracy on the selected MS1M identities, but it hurts LFW verification. This confirms that the current train/val split over the same selected identities is not a reliable proxy for open-set face verification. The next useful direction is not just better classification accuracy; it is improving verification-oriented generalization through larger identity coverage, train/eval alignment consistency, and a stronger ArcFace-style backbone/training recipe.

## MS1M Aligned-112 10000-Identity Embedding-512 Baseline

The baseline random-sampling recipe was kept unchanged while expanding identity coverage from 5000 to 10000 identities.

Dataset conversion result:

```text
Raw output root: data/raw/ms-celeb-1m-subset-10000
Processed output root: data/processed/ms1m-aligned-112-10000
Identities: 10000
Images: 300000
Original shapes: {'112x112': 300000}
Unreadable images: 0
```

Training result:

```text
Dataset: MS1M-aligned-112-10000-emb512
Device: cuda
Backbone: resnet50
Pretrained: imagenet
Embedding dim: 512
Sampler: random
Scheduler: none
Identities: 10000
Train/val images: 240000/60000
Epochs: 20
Best epoch: 6
Best val accuracy: 0.9753
Elapsed seconds: 7062.392
```

LFW official 6000 pairs / 10-fold result:

```text
Mean accuracy: 0.7668
Std accuracy: 0.0134
Total correct: 4601/6000
```

Comparison:

```text
5000 identities baseline:  val=0.9497, LFW=0.7407
10000 identities baseline: val=0.9753, LFW=0.7668
```

Interpretation: increasing identity coverage remains the most reliable improvement so far. Both closed-set validation accuracy and open-set LFW verification improve. The result is still far below 98.5%, so future work should use a larger portion of MS1M, stronger ArcFace-compatible backbone/training settings, and a verification-oriented validation protocol.
