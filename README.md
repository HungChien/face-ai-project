# Face AI Project

A computer vision internship project focused on face detection, facial landmark detection, face alignment, face recognition, facial effects, and model deployment.

## 1. Project Overview

This project aims to build a complete face AI workflow, covering both algorithm research and engineering practice.

The main technical modules include:

* Face detection
* Facial landmark detection
* Face alignment
* Face recognition and verification
* Facial attribute editing
* Real-time facial effects
* Model optimization and deployment

The project is implemented mainly with Python, PyTorch, OpenCV, OpenMMLab, ONNX, and Docker.

## 2. Project Objectives

The main objectives are:

1. Build a reproducible computer vision development environment.
2. Learn the standard workflow of OpenMMLab and MMDetection.
3. Explore standard face datasets such as LFW and CelebA.
4. Implement face detection, landmark detection, and alignment.
5. Build a face recognition model based on ResNet and ArcFace.
6. Evaluate face verification performance on LFW.
7. Study model optimization, ONNX conversion, and deployment.
8. Implement facial effects and real-time visual applications.

## 3. Current Progress

| Module                               | Status    | Evidence                              |
| ------------------------------------ | --------- | ------------------------------------- |
| GitHub repository setup              | Completed | Repository structure and Git history  |
| Git version control workflow         | Completed | Multiple commits and pushes           |
| Conda and Python environment         | Completed | `docs/env_setup.md`                   |
| PyTorch and CUDA verification        | Completed | `src/check_env.py`                    |
| OpenCV environment verification      | Completed | `src/check_env.py`                    |
| OpenCV image processing demo         | Completed | `src/opencv_demo.py`                  |
| PyTorch training demo                | Completed | `src/pytorch_demo.py`                 |
| Haar Cascade face detection baseline | Completed | `src/detection/face_detect_opencv.py` |
| Docker Hello World                   | Completed | `Dockerfile`, `docs/docker_setup.md`  |
| Jupyter Notebook experiment          | Pending   | Planned                               |
| MMDetection environment setup        | Pending   | Planned                               |
| MMDetection inference demo           | Pending   | Planned                               |
| MMDetection training smoke test      | Pending   | Planned                               |
| LFW dataset exploration              | Pending   | Planned                               |
| CelebA dataset exploration           | Pending   | Planned                               |
| MS-Celeb-1M study notes              | Pending   | Planned                               |
| Facial landmark detection            | Pending   | Planned                               |
| Face alignment                       | Pending   | Planned                               |
| Face recognition and verification    | Pending   | Planned                               |

## 4. Verified Environment

The current local development environment has been verified with the following configuration:

```text
Operating system: Windows
Conda environment: face_ai
Python: 3.10.20
PyTorch: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 2060
OpenCV: 4.13.0
NumPy: 2.2.6
```

Detailed environment setup instructions are available in:

```text
docs/env_setup.md
```

## 5. Project Structure

```text
face-ai-project/
├── configs/
│   └── mmdetection/
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── docs/
│   ├── env_setup.md
│   ├── docker_setup.md
│   ├── opencv_demo.md
│   ├── pytorch_demo.md
│   └── face_detection_baseline.md
├── models/
│   ├── checkpoints/
│   ├── onnx/
│   └── quantized/
├── notebooks/
├── outputs/
│   ├── images/
│   ├── reports/
│   └── videos/
├── src/
│   ├── alignment/
│   ├── app/
│   ├── deployment/
│   ├── detection/
│   ├── effects/
│   ├── landmark/
│   ├── recognition/
│   ├── check_env.py
│   ├── hello_docker.py
│   ├── opencv_demo.py
│   └── pytorch_demo.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

Some directories are reserved for later project stages and may currently contain only placeholder files.

## 6. Quick Start

### 6.1 Clone the repository

```bash
git clone https://github.com/HungChien/face-ai-project.git
cd face-ai-project
```

### 6.2 Create the Conda environment

```bash
conda create -n face_ai python=3.10 -y
conda activate face_ai
```

### 6.3 Install project dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The PyTorch installation depends on the local CUDA environment.

The current development machine uses PyTorch with CUDA 12.1. Other machines should install the appropriate PyTorch version according to their operating system, GPU, and CUDA configuration.

### 6.4 Verify the environment

```bash
python src/check_env.py
```

The script checks:

* Python version
* PyTorch version
* CUDA availability
* CUDA version
* GPU device name
* OpenCV version
* NumPy version

The generated report is stored under:

```text
outputs/reports/env_check_result.txt
```

## 7. Docker Hello World

The project includes a minimal Docker example to verify the basic Docker workflow.

The workflow covers:

```text
Python script
→ Dockerfile
→ Docker image
→ Docker container
→ Program output
```

### 7.1 Build the Docker image

Run the following command from the repository root:

```bash
docker build -t face-ai-hello .
```

### 7.2 Run the Docker container

```bash
docker run --rm face-ai-hello
```

Expected output:

```text
Hello from Docker!
The face-ai-project container is running successfully.
Python version: 3.10.x
Platform: Linux x86_64
```

The exact Python version and architecture may vary.

Docker-related files:

```text
Dockerfile
.dockerignore
src/hello_docker.py
docs/docker_setup.md
outputs/reports/docker_hello_result.txt
```

The current Docker image is intentionally minimal and does not include PyTorch, OpenCV, CUDA, MMDetection, datasets, or model checkpoints.

Its purpose is to verify image building and container execution.

## 8. OpenCV Image Processing Demo

The OpenCV demo verifies basic image loading and processing operations.

Script:

```text
src/opencv_demo.py
```

Processing workflow:

```text
Input image
→ Grayscale conversion
→ Gaussian blur
→ Canny edge detection
→ Output saving
```

Run:

```bash
python src/opencv_demo.py
```

The script generates image outputs under:

```text
outputs/images/
```

Documentation:

```text
docs/opencv_demo.md
```

## 9. PyTorch Training Demo

The PyTorch demo verifies that the project can complete a basic neural network training loop on CPU or GPU.

Script:

```text
src/pytorch_demo.py
```

Training workflow:

```text
Synthetic data generation
→ Model construction
→ Forward propagation
→ Loss computation
→ Backward propagation
→ Parameter update
→ Result logging
```

Run:

```bash
python src/pytorch_demo.py
```

The current demo uses randomly generated binary classification data.

Its purpose is to verify the training pipeline rather than achieve high classification accuracy.

Training report:

```text
outputs/reports/pytorch_demo_result.txt
```

Documentation:

```text
docs/pytorch_demo.md
```

## 10. Face Detection Baseline

The first face detection baseline uses OpenCV Haar Cascade.

Script:

```text
src/detection/face_detect_opencv.py
```

Detection workflow:

```text
Input image
→ Grayscale conversion
→ Haar Cascade detection
→ Bounding-box drawing
→ Result saving
→ Report generation
```

Run:

```bash
python src/detection/face_detect_opencv.py
```

The current baseline successfully detects the main frontal face in the test image.

However, it also produces false-positive detections in complex background and clothing regions.

This result demonstrates the limitations of traditional feature-based face detection and provides a baseline for comparison with modern deep learning detectors such as:

* MTCNN
* RetinaFace
* MMDetection-based detectors

Documentation:

```text
docs/face_detection_baseline.md
```

## 11. Jupyter Notebook Plan

The next environment and tool-learning task is to add a Jupyter Notebook experiment.

Planned file:

```text
notebooks/01_environment_and_opencv_demo.ipynb
```

The notebook will include:

1. Importing PyTorch, OpenCV, NumPy, and Matplotlib.
2. Printing environment and CUDA information.
3. Reading a sample image.
4. Converting the image from BGR to RGB.
5. Generating a grayscale image.
6. Performing edge detection.
7. Displaying results with Matplotlib.

## 12. MMDetection Plan

The project will use MMDetection to learn the standardized OpenMMLab computer vision workflow.

Planned tasks:

1. Install and verify MMEngine, MMCV, and MMDetection.
2. Record framework and dependency versions.
3. Run an official pretrained-model inference demo.
4. Learn the role of config files and checkpoints.
5. Run a small training smoke test.
6. Save training logs and inference results.
7. Prepare for face detection experiments on WIDER FACE.

Planned files:

```text
configs/mmdetection/
src/mmdetection_demo/check_mmdet_env.py
src/mmdetection_demo/inference_demo.py
docs/mmdetection_study.md
outputs/reports/mmdetection_env_result.txt
outputs/reports/mmdetection_inference_result.txt
outputs/reports/mmdetection_training_smoke_test.txt
```

## 13. Dataset Exploration Plan

The project will explore standard face datasets without committing complete datasets to GitHub.

### 13.1 LFW

Planned exploration:

* Dataset directory structure
* Number of identities
* Number of images
* Images per identity
* Face verification pair format
* Same-person and different-person pairs
* Image preprocessing
* Face verification evaluation

### 13.2 CelebA

Planned exploration:

* Image file structure
* Attribute annotation format
* Identity annotations
* Landmark annotations
* Training, validation, and test partitions
* Attribute distribution
* Cropping, resizing, and normalization

### 13.3 MS-Celeb-1M

Planned study topics:

* Large-scale identity classification
* Identity-to-label mapping
* Long-tail class distribution
* Data cleaning
* ArcFace-style training
* Storage and compliance considerations
* Use of cleaned subsets or alternative datasets

### 13.4 WIDER FACE

Planned exploration:

* Face-detection annotation format
* Face bounding boxes
* Easy, medium, and hard subsets
* Small-face detection challenges
* MMDetection data configuration
* Comparison with the Haar Cascade baseline

## 14. Data and Privacy Policy

Face datasets and human images require careful handling.

This repository follows these principles:

1. Full raw datasets are not committed to GitHub.
2. Full processed datasets are not committed to GitHub.
3. Unauthorized human face images are not committed.
4. Sensitive or private images are not committed.
5. Model checkpoints and large binary files are not committed.
6. Only source code, documentation, configuration files, lightweight reports, and safe sample outputs are stored.

Local dataset directories:

```text
data/raw/
data/processed/
```

Small authorized or synthetic sample files may be stored under:

```text
data/samples/
```

Dataset licenses and source-code licenses must be reviewed separately.

## 15. Model and Large-File Policy

The following file types are generally excluded from Git tracking:

```text
*.pth
*.pt
*.ckpt
*.onnx
*.engine
*.tflite
*.mp4
*.avi
*.mov
*.zip
*.tar
*.tar.gz
```

Large models and datasets should be stored locally or obtained through documented download instructions.

## 16. Phase 1 Requirements

The first project phase focuses on environment preparation, tool learning, framework experience, and dataset exploration.

### Completed

* Conda and Python environment setup
* PyTorch installation and CUDA verification
* OpenCV installation and image-processing demo
* GitHub repository setup
* Git commit and push workflow
* PyTorch training demo
* Haar Cascade face detection baseline
* Docker Desktop setup
* Docker image creation
* Docker Hello World container execution
* Basic environment and experiment documentation

### Remaining

* Jupyter Notebook experiment
* MMDetection installation and environment verification
* MMDetection pretrained-model inference
* MMDetection training smoke test
* LFW dataset exploration
* CelebA dataset exploration
* MS-Celeb-1M study notes
* Phase 1 summary report

## 17. Phase 1 Completion Criteria

Phase 1 will be considered complete when the repository contains:

```text
Environment setup documentation
GitHub repository and Git history
Docker Hello World example
OpenCV image processing program
Jupyter Notebook experiment
MMDetection environment verification
MMDetection inference result
MMDetection training smoke test
LFW dataset exploration report
CelebA dataset exploration report
MS-Celeb-1M study notes
Phase 1 summary report
```

## 18. Roadmap

The planned development workflow is:

```text
Environment setup
→ Docker and Jupyter verification
→ MMDetection framework learning
→ Dataset exploration
→ Face detection
→ Facial landmark detection
→ Face alignment
→ Face embedding extraction
→ Face verification
→ ArcFace training
→ Model optimization
→ ONNX deployment
→ Facial effects
→ System integration
```

## 19. Known Limitations

Current limitations include:

* The Haar Cascade detector produces false positives in complex scenes.
* The PyTorch demo uses synthetic random data.
* The Docker image currently includes only a minimal Python runtime.
* MMDetection has not yet been integrated.
* Jupyter Notebook experiments have not yet been committed.
* Dataset exploration has not yet been completed.
* Facial landmark detection has not yet been implemented.
* Face alignment has not yet been implemented.
* Face recognition and verification have not yet been implemented.

## 20. Documentation

Current documentation:

```text
docs/env_setup.md
docs/docker_setup.md
docs/opencv_demo.md
docs/pytorch_demo.md
docs/face_detection_baseline.md
```

Planned documentation:

```text
docs/jupyter_usage.md
docs/mmdetection_study.md
docs/dataset_usage_policy.md
docs/dataset_exploration.md
docs/phase1_report.md
```

## 21. Reproducibility

To reproduce the current completed experiments:

```bash
conda activate face_ai
python src/check_env.py
python src/opencv_demo.py
python src/pytorch_demo.py
python src/detection/face_detect_opencv.py
docker build -t face-ai-hello .
docker run --rm face-ai-hello
```

Input images must be placed under `data/samples/` and must comply with the repository data and privacy policy.

## 22. Repository

GitHub repository:

```text
https://github.com/HungChien/face-ai-project
```

## 23. License and Dataset Notice

This repository does not redistribute CelebA, LFW, MS-Celeb-1M, WIDER FACE, or other face datasets.

Users are responsible for reviewing and complying with the license, access rules, privacy requirements, and usage restrictions of each dataset and pretrained model.
