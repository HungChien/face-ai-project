# Environment Setup

## 1. Objective

The objective of this document is to record the environment setup process for the `face-ai-project`.

This document explains how to create the Conda environment, install the required dependencies, verify GPU availability, and run the environment check script.

The environment setup is required before running later project modules, including:

```text
OpenCV image processing demo
PyTorch training demo
Face detection baseline
Facial landmark detection
Face alignment
Face recognition
Facial effects
Model optimization and deployment
```

## 2. Repository Context

In this document, `<project-root>` refers to the root directory of this repository.

Example:

```text
<project-root>/
├── configs/
├── data/
├── docs/
├── models/
├── notebooks/
├── outputs/
├── src/
├── README.md
└── requirements.txt
```

## 3. Conda Environment

This project uses Conda to manage the Python environment.

Environment name:

```text
ml-gpu
```

Create the environment:

```bash
conda create -n ml-gpu python=3.10 -y
```

Activate the environment:

```bash
conda activate ml-gpu
```

## 4. Python Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install basic scientific computing and machine learning packages:

```bash
pip install numpy pandas matplotlib tqdm scikit-learn
```

Install OpenCV:

```bash
pip install opencv-python
```

Install Jupyter Notebook:

```bash
pip install jupyter notebook
```

Install ONNX and ONNX Runtime:

```bash
pip install onnx onnxruntime
```

Install GPU-enabled PyTorch with CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For machines without an NVIDIA GPU, install the CPU version of PyTorch instead:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 5. Dependency Record

After installing dependencies, generate `requirements.txt` from the activated environment:

```bash
python -m pip freeze > requirements.txt
```

The generated dependency file records the installed package versions.

Key installed packages include:

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
opencv-python==4.13.0.92
onnx==1.21.0
onnxruntime==1.23.2
numpy==2.2.6
pandas==2.3.3
matplotlib==3.10.9
scikit-learn==1.7.2
jupyter==1.1.1
```

## 6. Environment Check Script

The environment check script is located at:

```text
src/check_env.py
```

Run the script from `<project-root>`:

```bash
python src/check_env.py
```

On Windows, this command can also be written as:

```bash
python src\check_env.py
```

The script generates the following report:

```text
outputs/reports/env_check_result.txt
```

## 7. Local Verification Result

The local environment check result is:

```text
Environment Check
==================================================
Python version: 3.10.20 | packaged by Anaconda, Inc. | (main, Jun 11 2026, 15:13:20) [MSC v.1942 64 bit (AMD64)]
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
GPU name: NVIDIA GeForce RTX 2060
OpenCV version: 4.13.0
NumPy version: 2.2.6
```

## 8. Result Explanation

The environment check confirms that:

```text
1. Python 3.10 is installed and available in the Conda environment.
2. PyTorch is installed successfully.
3. CUDA is available.
4. PyTorch can detect the local NVIDIA GeForce RTX 2060 GPU.
5. OpenCV is installed successfully.
6. NumPy is installed successfully.
```

The result shows that GPU acceleration is available for later deep learning tasks.

## 9. Notes

If the environment needs to be used again, activate the Conda environment first:

```bash
conda activate ml-gpu
```

Then enter the project root directory:

```bash
cd face-ai-project
```

Run the environment check script:

```bash
python src/check_env.py
```

If new dependencies are installed later, update `requirements.txt`:

```bash
python -m pip freeze > requirements.txt
```

## 10. Conclusion

The project environment has been successfully configured.

PyTorch can detect and use the local NVIDIA GeForce RTX 2060 GPU through CUDA 12.1. OpenCV, NumPy, ONNX, ONNX Runtime, and other basic dependencies are also installed successfully.

The environment is ready for Week 1 tasks and later computer vision experiments.