# Environment Setup

## 1. Project Information

Project name:

```text
face-ai-project
```

Project goal:

```text
This project is for an intelligent vision AI internship project. The main tasks include face detection, facial landmark detection, face alignment, face recognition, real-time facial effects, and model optimization and deployment.
```

## 2. Repository Setup

Clone the repository:

```bash
git clone <repository-url>
cd face-ai-project
```

If the repository has already been cloned, enter the project root directory:

```bash
cd face-ai-project
```

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
face_ai
```

Create the environment:

```bash
conda create -n face_ai python=3.10 -y
```

Activate the environment:

```bash
conda activate face_ai
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

## 5. Environment Check Script

Environment check script path:

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

The script generates the following environment check report:

```text
outputs/reports/env_check_result.txt
```

## 6. Local Verification Result

The local environment check result is:

```text
Python version: 3.10.20 | packaged by Anaconda, Inc.
PyTorch version: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
GPU name: NVIDIA GeForce RTX 2060
OpenCV version: 4.13.0
NumPy version: 2.2.6
```

## 7. Conclusion

The Python environment has been successfully configured.

On the local development machine, PyTorch can detect and use the NVIDIA GeForce RTX 2060 GPU through CUDA 12.1. OpenCV, NumPy, ONNX, and ONNX Runtime are also installed successfully.

This environment is ready for Week 1 tasks, including:

- OpenCV image processing demo
- PyTorch basic test
- Face detection baseline
- Future face recognition and facial effects experiments

## 8. Notes

If the environment needs to be used again, activate the Conda environment first:

```bash
conda activate face_ai
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