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
| MS-Celeb-1M | Deferred | Not downloaded because of scale, licensing, and cleaning concerns |

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
