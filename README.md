# Face AI Project

A computer vision internship project for face detection, facial landmark localization, face recognition verification, and later facial effects/deployment work.

## Project Status

The repository is currently organized around Phase 1 deliverables: environment setup, framework verification, dataset exploration, face detection, landmark localization, and pretrained LFW verification.

| Module | Status | Main Evidence |
| --- | --- | --- |
| Git/GitHub workflow | Completed | Git history and GitHub remote |
| Conda/Python environments | Completed | `docs/env_setup.md`, `outputs/reports/env_check_result.txt` |
| Docker Hello World | Completed | `Dockerfile`, `src/hello_docker.py`, `outputs/reports/docker_hello_result.txt` |
| OpenCV image processing | Completed | `src/opencv_demo.py`, `outputs/images/gray.jpg`, `outputs/images/edges.jpg` |
| Jupyter experiment | Completed | `notebooks/01_environment_and_opencv_demo.ipynb`, `outputs/reports/jupyter_demo_result.txt` |
| PyTorch training smoke test | Completed | `src/pytorch_demo.py`, `outputs/reports/pytorch_demo_result.txt` |
| OpenCV Haar face baseline | Completed | `src/detection/face_detect_opencv.py`, `outputs/reports/face_detection_baseline_result.txt` |
| MMDetection environment check | Completed | `src/mmdetection/check_mmdet_env.py`, `outputs/reports/mmdetection_env_result.txt` |
| MMDetection face detection | Completed | `src/mmdetection/run_face_detection_mmdet.py`, `outputs/reports/mmdetection_face_detection_result.txt` |
| Face landmark localization | Completed | `src/landmarks/face_landmark_mediapipe.py`, `outputs/reports/face_landmark_mediapipe_result.txt` |
| LFW dataset exploration | Completed | `src/datasets/explore_lfw.py`, `outputs/reports/lfw_exploration_result.txt` |
| LFW pair + verification exploration | Completed | `src/datasets/explore_lfw_pairs.py`, `outputs/reports/lfw_pair_exploration_result.txt` |
| LFW pretrained recognition accuracy | Completed | `outputs/reports/lfw_recognition_verification_result.txt` |
| CelebA dataset exploration | Completed | `src/datasets/explore_celeba.py`, `outputs/reports/celeba_exploration_result.txt` |
| CelebA 100-image evaluation | Completed | `src/evaluation/evaluate_celeba_100.py`, `outputs/reports/celeba_100_evaluation_result.txt` |
| MS-Celeb-1M | Deferred | Not downloaded because of scale, licensing, and cleaning concerns |
| Phase 2 Week 3 setup | Started | `docs/week3_plan.md`, `docs/face_detection_algorithms.md`, `configs/mmdetection/widerface_retinanet_r50_fpn.py`, `configs/mmdetection/widerface_retinanet_r50_fpn_debug.py`, `configs/mmdetection/widerface_retinanet_r50_fpn_debug_infer.py`, `outputs/reports/widerface_debug_train_result.txt`, `outputs/reports/widerface_debug_checkpoint_visualization_result.txt` |

## Environment Layout

This project uses two Conda environments because the RTX 5080 GPU requires a newer PyTorch stack than the Windows MMCV build supports.

```text
ml-gpu
  Main project environment.
  Used for OpenCV, PyTorch, datasets, MediaPipe landmarks, and InsightFace verification.

ml-mmdet
  MMDetection/MMCV environment.
  Used for MMDetection CPU inference and OpenMMLab verification.
```

Dependency records:

```text
requirements-gpu.txt
requirements-mmdet.txt
```

Detailed setup notes:

```text
docs/env_setup.md
```

## Project Structure

```text
face-ai-project/
  data/
    raw/                 local datasets, ignored by Git
    processed/           derived data, ignored by Git
    samples/             small authorized sample images
  docs/
    env_setup.md
    docker_setup.md
    jupyter_usage.md
    opencv_demo.md
    pytorch_demo.md
    face_detection_baseline.md
    phase1_report.md
  models/
    checkpoints/         local model files, ignored by Git
    onnx/                exported ONNX files, ignored by Git
    quantized/           quantized models, ignored by Git
  notebooks/
    01_environment_and_opencv_demo.ipynb
  outputs/
    images/              dataset/demo visualizations
    landmarks/           landmark visualizations
    reports/             text and JSON experiment reports
    videos/              later video outputs
  src/
    datasets/            LFW and CelebA exploration/evaluation
    detection/           OpenCV face baseline
    landmarks/           face landmark localization
    mmdetection/         MMDetection environment and face detection
    check_env.py
    hello_docker.py
    opencv_demo.py
    pytorch_demo.py
```


## Phase 2 Week 3

Week 3 starts the face detection training part of Phase 2.

Main files:

```text
docs/week3_plan.md
docs/face_detection_algorithms.md
configs/mmdetection/widerface_retinanet_r50_fpn.py
configs/mmdetection/widerface_retinanet_r50_fpn_smoke.py
src/datasets/convert_widerface_to_voc.py
src/mmdetection/check_widerface_dataset.py
src/mmdetection/train_widerface.py
src/mmdetection/evaluate_widerface.py
```

WIDER FACE is not committed to Git. After downloading and converting it locally, run:

```powershell
conda activate ml-mmdet
python src\mmdetection\check_widerface_dataset.py
python src\mmdetection\train_widerface.py
python src\mmdetection\evaluate_widerface.py
```

For a quick CPU-only training smoke test:

```powershell
$env:PYTHONUTF8='1'
python src\mmdetection\train_widerface.py --config configs\mmdetection\widerface_retinanet_r50_fpn_smoke.py --work-dir outputs\mmdetection_widerface\retinanet_r50_fpn_smoke --device cpu
```

For a longer CPU debug training run:

```powershell
$env:PYTHONUTF8='1'
python src\mmdetection\train_widerface.py --config configs\mmdetection\widerface_retinanet_r50_fpn_debug.py --work-dir outputs\mmdetection_widerface\retinanet_r50_fpn_debug --device cpu
```

For debug checkpoint visualization:

```powershell
$env:PYTHONUTF8='1'
python src\mmdetection\evaluate_widerface.py --config configs\mmdetection\widerface_retinanet_r50_fpn_debug_infer.py --checkpoint outputs\mmdetection_widerface\retinanet_r50_fpn_debug\epoch_1.pth --visualize-images data\samples\face_test.jpg data\raw\WIDERFace\WIDER_val\images\0--Parade\0_Parade_marchingband_1_1004.jpg --vis-dir outputs\images\widerface_debug_checkpoint --report outputs\reports\widerface_debug_checkpoint_visualization_result.txt --score-thr 0.0 --max-vis 20 --device cpu
```


## Phase 2 Task 4

Face landmark detection and alignment files:

```text
docs/landmark_detection_alignment.md
docs/300w_download.md
src/landmarks/check_300w_dataset.py
src/landmarks/train_landmark_regressor.py
src/landmarks/face_alignment.py
src/landmarks/align_with_landmark_model.py
src/landmarks/compare_gt_pred_alignment.py
src/landmarks/calibrate_300w_alignment_template.py
outputs/reports/landmark_300w_dataset_check_result.txt
outputs/reports/face_alignment_result.txt
outputs/landmarks/alignment/face_test_aligned_112.jpg
outputs/landmarks/alignment/face_test_alignment_comparison.jpg
outputs/landmarks/alignment_300w_model/landmark_model_alignment_grid.jpg
outputs/landmarks/alignment_compare_300w/gt_vs_pred_alignment_grid.jpg
outputs/landmarks/calibrated_template_300w/arcface_vs_300w_template_alignment.jpg
```

Run alignment demo:

```powershell
conda activate ml-gpu
python src\landmarks\face_alignment.py
```




Run 300W five-point template calibration:

```powershell
conda activate ml-gpu
python src\landmarks\calibrate_300w_alignment_template.py --checkpoint models\checkpoints\landmark_cnn_300w_aug30_best.pt --iterations 20 --num-samples 6
```
Run GT-vs-predicted five-point alignment comparison:

```powershell
conda activate ml-gpu
python src\landmarks\compare_gt_pred_alignment.py --checkpoint models\checkpoints\landmark_cnn_300w_aug30_best.pt --num-samples 6 --split val --device auto
```
Run 300W model based alignment demo:

```powershell
conda activate ml-gpu
python src\landmarks\align_with_landmark_model.py --checkpoint models\checkpoints\landmark_cnn_300w_cropped.pt --num-samples 6 --device auto
```
Check 300W-style landmark data:

```powershell
conda activate ml-gpu
python src\landmarks\check_300w_dataset.py --root data\raw\300W
```

Download/place official 300W split archives:

```text
data\raw\300W_OFFICIAL\300w.zip.001
data\raw\300W_OFFICIAL\300w.zip.002
data\raw\300W_OFFICIAL\300w.zip.003
data\raw\300W_OFFICIAL\300w.zip.004
```

Then extract and verify:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\extract_300w.ps1
```

Train the landmark baseline after placing 300W/COFW-style `.pts` annotations:

```powershell
conda activate ml-gpu
python src\landmarks\train_landmark_regressor.py --root data\raw\300W --epochs 5 --device auto
```


## Phase 2 Task 5

Face recognition ArcFace smoke baseline:

```text
docs/face_recognition_arcface_baseline.md
src/recognition/train_resnet18_arcface_smoke.py
src/recognition/evaluate_lfw_pairs_resnet18_arcface.py
src/recognition/train_arcface_celeba_subset.py
models/checkpoints/resnet18_arcface_lfw_smoke10_best.pt
outputs/reports/resnet18_arcface_lfw_smoke10_result.txt
outputs/images/resnet18_arcface_lfw_smoke10_curves.jpg
outputs/reports/resnet18_arcface_lfw_pair_verification_result.txt
models/checkpoints/resnet50_arcface_celeba200_best.pt
outputs/reports/resnet50_arcface_celeba200_result.txt
outputs/reports/resnet50_arcface_celeba200_lfw_pair_verification_result.txt
models/checkpoints/resnet50_arcface_celeba1000_e20_best.pt
outputs/reports/resnet50_arcface_celeba1000_e20_result.txt
outputs/reports/resnet50_arcface_celeba1000_e20_lfw_pair_verification_result.txt
models/checkpoints/resnet50_imagenet_arcface_celeba1000_e20_best.pt
outputs/reports/resnet50_imagenet_arcface_celeba1000_e20_result.txt
outputs/reports/resnet50_imagenet_arcface_celeba1000_e20_lfw_pair_verification_result.txt
```

Run ResNet18 + ArcFace smoke training:

```powershell
conda activate ml-gpu
python src\recognition\train_resnet18_arcface_smoke.py --epochs 20 --num-identities 10 --max-images-per-identity 20 --batch-size 16 --device auto --output models\checkpoints\resnet18_arcface_lfw_smoke10_best.pt --report outputs\reports\resnet18_arcface_lfw_smoke10_result.txt --history-json outputs\reports\resnet18_arcface_lfw_smoke10_history.json --curve outputs\images\resnet18_arcface_lfw_smoke10_curves.jpg
```





Run ImageNet-initialized ResNet50 + ArcFace CelebA1000 E20 training:

```powershell
conda activate ml-gpu
python src\recognition\train_arcface_celeba_subset.py --backbone resnet50 --pretrained imagenet --epochs 20 --num-identities 1000 --max-train-images-per-identity 30 --max-val-images-per-identity 10 --batch-size 128 --device auto --split-mode random --output models\checkpoints\resnet50_imagenet_arcface_celeba1000_e20_best.pt --report outputs\reports\resnet50_imagenet_arcface_celeba1000_e20_result.txt --history-json outputs\reports\resnet50_imagenet_arcface_celeba1000_e20_history.json --curve outputs\images\resnet50_imagenet_arcface_celeba1000_e20_curves.jpg
```

Current ImageNet initialized result:

```text
Best val accuracy: 0.7083
LFW pair test accuracy: 0.8080
```
Run expanded ResNet50 + ArcFace CelebA1000 E20 training:

```powershell
conda activate ml-gpu
python src\recognition\train_arcface_celeba_subset.py --backbone resnet50 --epochs 20 --num-identities 1000 --max-train-images-per-identity 30 --max-val-images-per-identity 10 --batch-size 128 --device auto --split-mode random --output models\checkpoints\resnet50_arcface_celeba1000_e20_best.pt --report outputs\reports\resnet50_arcface_celeba1000_e20_result.txt --history-json outputs\reports\resnet50_arcface_celeba1000_e20_history.json --curve outputs\images\resnet50_arcface_celeba1000_e20_curves.jpg
```

Current expanded training result:

```text
Identities: 1000
Train/val images: 24033/6005
Best val accuracy: 0.0185
LFW pair test accuracy: 0.6640
```
Run ResNet50 + ArcFace CelebA200 training:

```powershell
conda activate ml-gpu
python src\recognition\train_arcface_celeba_subset.py --backbone resnet50 --epochs 10 --num-identities 200 --max-train-images-per-identity 30 --max-val-images-per-identity 10 --batch-size 64 --device auto --split-mode random --output models\checkpoints\resnet50_arcface_celeba200_best.pt --report outputs\reports\resnet50_arcface_celeba200_result.txt --history-json outputs\reports\resnet50_arcface_celeba200_history.json --curve outputs\images\resnet50_arcface_celeba200_curves.jpg
```

Current ResNet50 CelebA200 result:

```text
Identities: 200
Train/val images: 4833/1205
Best val accuracy: 0.1426
LFW pair test accuracy: 0.6380
```
Run LFW pair verification with the smoke checkpoint:

```powershell
conda activate ml-gpu
python src\recognition\evaluate_lfw_pairs_resnet18_arcface.py --checkpoint models\checkpoints\resnet18_arcface_lfw_smoke10_best.pt --device auto
```

Current pair verification result:

```text
Train accuracy at selected threshold: 0.6436
Test accuracy: 0.6200
Correct / total: 620 / 1000
```
Current smoke result:

```text
Backbone: ResNet18
Head/loss: ArcFace
Identities: 10
Train/val images: 160/40
Best epoch: 13
Best val accuracy: 0.7292
```

## Reproduce Phase 1

Run main project checks in `ml-gpu`:

```powershell
conda activate ml-gpu
python src\check_env.py
python src\opencv_demo.py
python src\pytorch_demo.py
python src\detection\face_detect_opencv.py
python src\datasets\explore_lfw.py
python src\datasets\explore_lfw_pairs.py
python src\datasets\explore_lfw_pairs.py --run-recognition --model-name buffalo_l
python src\datasets\explore_celeba.py
python src\landmarks\face_landmark_mediapipe.py
python src\evaluation\evaluate_celeba_100.py --save-failure-grid
```

Run MMDetection checks in `ml-mmdet`:

```powershell
conda activate ml-mmdet
python src\mmdetection\check_mmdet_env.py
python src\mmdetection\run_face_detection_mmdet.py
```

Run Docker Hello World:

```powershell
docker build -t face-ai-hello .
docker run --rm face-ai-hello
```

## Key Phase 1 Results

LFW dataset:

```text
Images: 13,233
Identities: 5,749
Pair protocol: pairs.txt, pairsDevTrain.txt, pairsDevTest.txt
```

CelebA dataset:

```text
Images: 202,599
Attributes: 40
Annotations: bbox, five landmarks, identity, split, binary attributes
```

MMDetection face detection:

```text
Model: grounding_dino_swin-t_pretrain_obj365_goldg
Prompt: face
Detected faces above threshold: 1
```

MediaPipe face landmarks:

```text
Model: MediaPipe FaceLandmarker
Successful faces: 1
Dense landmarks: 478
Five points: left eye, right eye, nose tip, left mouth, right mouth
```

LFW pretrained recognition verification:

```text
Model: InsightFace buffalo_l
Train protocol: pairsDevTrain.txt threshold selection
Test protocol: pairsDevTest.txt evaluation
Test accuracy: 0.9440
Correct / total: 944 / 1000
```

CelebA 100-image evaluation:

```text
Sample: 100 random CelebA aligned images, seed 42
Five-point derived box vs original bbox mean IoU: 0.0871
IoU >= 0.5: 4 / 100
OpenCV Haar detection failure rate: 0.05
MediaPipe landmark failure rate: 0.01
MediaPipe mean NME: 0.0440
```

## Data And Model Policy

Full face datasets and model weights are not committed to Git. They are kept locally under ignored folders such as `data/raw/` and `models/checkpoints/`.

The repository should track source code, lightweight documentation, small authorized samples, and compact reports/visualizations only.

## Current Limitations

- MMDetection runs on CPU in `ml-mmdet` because the compatible Windows MMCV build uses an older PyTorch/CUDA stack.
- The MMDetection WIDER FACE config is available, but OpenMMLab does not provide an indexed pretrained WIDER FACE checkpoint for direct `mim download` use in this setup.
- LFW verification currently uses LFW funneled aligned images directly for embedding extraction. This is suitable as a Phase 1 pretrained baseline, but later stages should implement a complete detect-align-recognize pipeline.
- MS-Celeb-1M is deferred because of dataset scale, licensing, privacy, and cleaning requirements.

## Repository

```text
https://github.com/HungChien/face-ai-project
```


### Facenet-PyTorch VGGFace2 LFW 10-Fold Reference

Independent environment:

```powershell
D:\Anaconda3\envs\ml-face\python.exe
```

Run the full official LFW 6000 pairs / 10-fold verification:

```powershell
D:\Anaconda3\envs\ml-face\python.exe src\recognition\evaluate_lfw_10fold_facenet.py --batch-size 64
```

Result:

```text
Model: facenet-pytorch InceptionResnetV1 pretrained=vggface2
Protocol: LFW pairs.txt 6000 pairs / 10-fold
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

This is a face-domain pretrained reference baseline. It is useful for measuring the gap to the 98.5% target, but it is not the same as the self-trained ResNet50 + ArcFace deliverable.

### ResNet50 + ArcFace MS-Celeb-1M Subset Training

Formal GPU run on the converted cleaned MS-Celeb-1M subset:

```text
Dataset: MS-Celeb-1M-subset
Identities: 1000
Train/val images: 24000/6000
Backbone: ResNet50
Initialization: ImageNet pretrained
Loss/head: ArcFace
Device: cuda
Epochs: 20
Best epoch: 10
Best val accuracy: 0.7665
```

LFW verification:

```text
pairsDevTest accuracy: 0.7210
Official LFW 6000 pairs / 10-fold mean accuracy: 0.6975 ± 0.0128
Correct: 4185/6000
```

Main outputs:

```text
models/checkpoints/resnet50_imagenet_arcface_ms1m_subset_best.pt
outputs/reports/resnet50_imagenet_arcface_ms1m_subset_result.txt
outputs/reports/resnet50_imagenet_arcface_ms1m_subset_lfw_10fold_result.txt
outputs/images/resnet50_imagenet_arcface_ms1m_subset_curves.jpg
```

### MS1M Aligned-112 Baseline

The MS1M-ArcFace subset was standardized into an explicit processed aligned folder:

```text
data/processed/ms1m-aligned-112
```

Preprocessing result:

```text
Identities: 1000
Images: 30000
Original shapes: 112x112 for all images
Copied: 30000
Resized: 0
Unreadable: 0
```

Aligned baseline result:

```text
Backbone: ResNet50
Initialization: ImageNet pretrained
Loss/head: ArcFace
Device: cuda
Train/val images: 24000/6000
Best val accuracy: 0.7595
LFW 6000 pairs / 10-fold: 0.7075 ± 0.0092
Correct: 4245/6000
```

Main outputs:

```text
src/datasets/prepare_ms1m_aligned_112.py
outputs/reports/ms1m_aligned_112_preprocess_result.txt
outputs/images/ms1m_aligned_112_preview.jpg
outputs/reports/resnet50_imagenet_arcface_ms1m_aligned112_result.txt
outputs/reports/resnet50_imagenet_arcface_ms1m_aligned112_lfw_10fold_result.txt
outputs/images/resnet50_imagenet_arcface_ms1m_aligned112_curves.jpg
```

### MS1M Aligned-112 Embedding-512 Baseline

Same aligned MS1M subset and ResNet50 + ArcFace setup, with `embedding_dim=512`:

```text
Best val accuracy: 0.7746
LFW 6000 pairs / 10-fold: 0.7027 ± 0.0197
Correct: 4216/6000
```

Comparison:

```text
256-d aligned baseline: val=0.7595, LFW=0.7075
512-d aligned baseline: val=0.7746, LFW=0.7027
```

The 512-d embedding improves identity classification on the selected MS1M subset, but does not improve LFW verification under the current data scale and training recipe.

### MS1M Aligned-112 5000-Identity Embedding-512 Baseline

The MS1M-ArcFace subset was expanded to 5000 identities:

```text
Processed root: data/processed/ms1m-aligned-112-5000
Identities: 5000
Images: 150000
Backbone: ResNet50
Initialization: ImageNet pretrained
Loss/head: ArcFace
Embedding dim: 512
Device: cuda
Train/val images: 120000/30000
Best val accuracy: 0.9497
```

LFW verification:

```text
Official LFW 6000 pairs / 10-fold: 0.7407 ± 0.0228
Correct: 4444/6000
```

Comparison:

```text
1000 identities, 512-d: val=0.7746, LFW=0.7027
5000 identities, 512-d: val=0.9497, LFW=0.7407
```

Main outputs:

```text
outputs/reports/ms1m_recordio_conversion_5000_summary.json
outputs/reports/ms1m_aligned_112_5000_preprocess_result.txt
outputs/reports/resnet50_imagenet_arcface_ms1m_aligned112_5000_emb512_result.txt
outputs/reports/resnet50_imagenet_arcface_ms1m_aligned112_5000_emb512_lfw_10fold_result.txt
outputs/images/resnet50_imagenet_arcface_ms1m_aligned112_5000_emb512_curves.jpg
```

### MS1M 5000-Identity Training Strategy Experiments

Additional strategy experiments were run on the 5000-identity, 512-d setup:

```text
PK sampler + strong augment + cosine LR:
best val accuracy = 0.0005  # failed to converge

Random sampler + cosine LR + light erasing:
best val accuracy = 0.9578
LFW 6000 pairs / 10-fold = 0.6970 ± 0.0225
```

Comparison with the previous 5000-identity baseline:

```text
Baseline: val=0.9497, LFW=0.7407
Cosine:   val=0.9578, LFW=0.6970
```

Conclusion: higher closed-set validation accuracy on selected MS1M identities does not necessarily improve open-set LFW verification. The current next bottleneck is verification generalization, not merely classification accuracy.

### MS1M Aligned-112 10000-Identity Embedding-512 Baseline

The random-sampling baseline was expanded from 5000 to 10000 identities:

```text
Processed root: data/processed/ms1m-aligned-112-10000
Identities: 10000
Images: 300000
Backbone: ResNet50
Initialization: ImageNet pretrained
Loss/head: ArcFace
Embedding dim: 512
Sampler: random
Scheduler: none
Device: cuda
Train/val images: 240000/60000
Best val accuracy: 0.9753
```

LFW verification:

```text
Official LFW 6000 pairs / 10-fold: 0.7668 ± 0.0134
Correct: 4601/6000
```

Comparison:

```text
5000 identities baseline:  val=0.9497, LFW=0.7407
10000 identities baseline: val=0.9753, LFW=0.7668
```

Main outputs:

```text
outputs/reports/ms1m_recordio_conversion_10000_summary.json
outputs/reports/ms1m_aligned_112_10000_preprocess_result.txt
outputs/reports/resnet50_imagenet_arcface_ms1m_aligned112_10000_emb512_result.txt
outputs/reports/resnet50_imagenet_arcface_ms1m_aligned112_10000_emb512_lfw_10fold_result.txt
outputs/images/resnet50_imagenet_arcface_ms1m_aligned112_10000_emb512_curves.jpg
```
