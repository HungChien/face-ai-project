# Face AI Project

A computer vision internship project focused on face detection, facial landmark detection, face alignment, face recognition, facial effects, and model deployment.

## Project Objectives

The project aims to build a complete face AI workflow covering:

* Face detection
* Facial landmark detection
* Face alignment
* Face recognition and verification
* Real-time facial effects
* Model optimization and deployment

## Current Progress

| Module                          | Status    | Evidence                              |
| ------------------------------- | --------- | ------------------------------------- |
| Repository setup                | Completed | Project structure and Git history     |
| Conda environment               | Completed | `docs/env_setup.md`                   |
| PyTorch and CUDA verification   | Completed | `src/check_env.py`                    |
| OpenCV image processing demo    | Completed | `src/opencv_demo.py`                  |
| PyTorch training demo           | Completed | `src/pytorch_demo.py`                 |
| Haar Cascade face baseline      | Completed | `src/detection/face_detect_opencv.py` |
| Docker Hello World              | Pending   | Not started                           |
| Jupyter Notebook experiment     | Pending   | Not started                           |
| MMDetection setup               | Pending   | Not started                           |
| MMDetection inference           | Pending   | Not started                           |
| MMDetection training smoke test | Pending   | Not started                           |
| LFW dataset exploration         | Pending   | Not started                           |
| CelebA dataset exploration      | Pending   | Not started                           |
| MS-Celeb-1M study notes         | Pending   | Not started                           |

## Environment

The verified local development environment includes:

```text
Python: 3.10.20
PyTorch: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 2060
OpenCV: 4.13.0
NumPy: 2.2.6
```

For detailed setup instructions, see:

```text
docs/env_setup.md
```

## Project Structure

```text
face-ai-project/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── docs/
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
│   └── recognition/
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/HungChien/face-ai-project.git
cd face-ai-project
```

### 2. Create the Conda environment

```bash
conda create -n face_ai python=3.10 -y
conda activate face_ai
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The current PyTorch build uses CUDA 12.1. On another machine, install the PyTorch build appropriate for that machine before running GPU experiments.

### 4. Verify the environment

```bash
python src/check_env.py
```

### 5. Run the OpenCV demo

Place a safe sample image at:

```text
data/samples/test.jpg
```

Then run:

```bash
python src/opencv_demo.py
```

### 6. Run the PyTorch demo

```bash
python src/pytorch_demo.py
```

### 7. Run the face detection baseline

Place an authorized test image at:

```text
data/samples/face_test.jpg
```

Then run:

```bash
python src/detection/face_detect_opencv.py
```

## Completed Experiments

### OpenCV Image Processing Demo

The demo performs:

```text
Image loading
→ Grayscale conversion
→ Gaussian blur
→ Canny edge detection
→ Result saving
```

Documentation:

```text
docs/opencv_demo.md
```

### PyTorch Training Demo

The demo verifies:

```text
Tensor creation
→ Forward propagation
→ Loss computation
→ Backward propagation
→ Optimizer update
→ Checkpoint saving
```

Documentation:

```text
docs/pytorch_demo.md
```

### Face Detection Baseline

The first face detection baseline uses OpenCV Haar Cascade.

The baseline successfully detected the main frontal face, but also produced false positives in complex background and clothing regions. This result demonstrates the limitations of traditional feature-based detection and provides a reference for later comparison with MTCNN or RetinaFace.

Documentation:

```text
docs/face_detection_baseline.md
```

## Data and Privacy Policy

Large face datasets and unauthorized human face images are not committed to this repository.

The repository only stores:

* Source code
* Configuration files
* Documentation
* Lightweight reports
* Safe and authorized sample outputs

Raw and processed datasets should be stored locally under:

```text
data/raw/
data/processed/
```

## Planned Phase 1 Work

The remaining Phase 1 tasks are:

1. Create and run a Docker Hello World example
2. Add a Jupyter Notebook experiment
3. Install and verify MMDetection
4. Run an MMDetection inference demo
5. Run an MMDetection training smoke test
6. Explore the LFW dataset
7. Explore the CelebA dataset
8. Prepare MS-Celeb-1M study notes
9. Complete the Phase 1 report

## Known Limitations

* The current Haar Cascade detector produces false positives in complex scenes.
* The PyTorch demo uses synthetic random data and is intended only for pipeline verification.
* Facial landmark detection and face alignment have not yet been implemented.
* The current repository does not yet include MMDetection, Docker, or dataset exploration experiments.

## Documentation

Available documentation includes:

```text
docs/env_setup.md
docs/opencv_demo.md
docs/pytorch_demo.md
docs/face_detection_baseline.md
```

## License and Dataset Notice

Source code licensing and dataset licensing are separate concerns.

Users must review and comply with the license and usage terms of every dataset before downloading or using it. This repository does not redistribute CelebA, LFW, MS-Celeb-1M, WIDER FACE, or other face datasets.
