# Face AI Project

An end-to-end face AI internship project covering dataset analysis, face detection,
landmark localization and alignment, face recognition, model optimization, attribute
editing, 3D reconstruction, rendering, and dynamic effects.

## Final Demo

Generate and open the unified result dashboard:

```powershell
conda activate ml-gpu
python src/app/final_demo.py --serve --open
```

The command builds `outputs/final_demo/index.html` and serves it locally. The demo is
artifact-driven: it indexes the real reports, images, meshes, videos, checkpoints, and
source entrypoints produced by this repository.

## Completed Modules

| Module | Main implementation | Representative result |
|---|---|---|
| Dataset exploration | LFW and CelebA format, labels, pairs, OpenCV/Matplotlib visualization | Dataset reports and sample grids |
| Face detection | MMDetection RetinaNet on WIDER FACE | epoch-1 mAP/AP50: 0.328 |
| Landmarks and alignment | 300W 68-point regressor, five-point extraction, affine alignment | Best validation NME: 0.1721 |
| Face recognition | ResNet50 + ArcFace on aligned MS1M subset | Validation accuracy: 97.94%; LFW 10-fold: 76.98% |
| Reference recognition | VGGFace2-pretrained FaceNet | LFW 10-fold: 96.77% |
| Optimization | PyTorch dynamic quantization and ONNX export | 20.21% smaller; 1.174x / 1.170x speedup |
| Attribute editing | StarGAN on CelebA | FID 66.4039; IS 2.1776 +/- 0.1392 |
| 3D face reconstruction | Official 3DDFA_V2 plus OpenGL rendering | 38,365 vertices, 76,073 triangles, six views |
| Dynamic effects | Detection, landmark tracking, temporal smoothing, OpenCV composition | 96-frame MP4 at 24 FPS |

The recognition accuracy above is reported honestly: the self-trained model completes
the requested training and evaluation pipeline but does not reach the aspirational
98.5% LFW target. The repository keeps the experiments that explain this gap.

## Repository Layout

```text
configs/              Reproducible MMDetection configurations
docs/                 Environment, algorithm, and execution notes
models/               Final local checkpoints and exported models (ignored by Git)
outputs/               Representative images, reports, meshes, videos, and demo
scripts/               Small setup and dataset utilities
src/
  app/                 Unified final demo
  datasets/            LFW, CelebA, WIDER FACE, and MS1M preparation
  mmdetection/         WIDER FACE training, inference, and evaluation
  landmarks/           300W training, visualization, and alignment
  recognition/         ArcFace training and LFW evaluation
  optimization/        Quantization and ONNX export/inference
  stargan/             CelebA StarGAN training and sampling
  reconstruction/      3DDFA_V2 reconstruction and OpenGL rendering
  effects/             Static and dynamic effects
third_party/           Local external repositories; see third_party/README.md
```

## Environments

- `ml-mmdet`: CPU-compatible MMEngine/MMCV/MMDetection workflow.
- `ml-gpu`: local GPU experiments and landmark/recognition utilities.
- AutoDL RTX 4090: full ArcFace and StarGAN training.

Environment notes are in `docs/env_setup.md` and `docs/stargan_celeba_training.md`.

## Core Reproduction Commands

```powershell
# Unified dashboard
python src/app/final_demo.py

# WIDER FACE evaluation
python src/mmdetection/evaluate_widerface.py --help

# 300W landmark training and alignment
python src/landmarks/train_landmark_regressor.py --help
python src/landmarks/align_with_landmark_model.py --help

# ArcFace training and LFW verification
python src/recognition/train_arcface_celeba_subset.py --help
python src/recognition/evaluate_lfw_10fold_resnet_arcface.py --help

# Optimization
python src/optimization/quantize_arcface_model.py --help
python src/optimization/export_arcface_onnx.py --help

# StarGAN
python src/stargan/train_stargan_celeba.py --help
python src/stargan/sample_stargan_celeba.py --help

# 3D reconstruction and dynamic effects
python src/reconstruction/run_3ddfa_v2_demo.py --help
python src/reconstruction/render_3ddfa_opengl.py --help
python src/effects/run_dynamic_effects_demo.py --help
```

## Data and Model Policy

Datasets and large model files are intentionally excluded from Git. Expected local
locations include `data/raw`, `data/processed`, `models/checkpoints`, and
`third_party/3DDFA_V2`. Text/JSON metrics and representative visual results remain in
`outputs/` so the project can be reviewed without rerunning every training job.

The final Chinese technical report is maintained outside the repository in the
internship `docs` directory as both Word and PDF.
