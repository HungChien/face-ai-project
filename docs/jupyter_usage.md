# Jupyter Notebook Experiment

## 1. Objective

The objective of this task is to learn how to use Jupyter Notebook for interactive computer vision experiments.

The experiment verifies the Python environment and demonstrates a basic OpenCV image processing workflow.

## 2. Notebook

The main notebook is:

```text
notebooks/01_environment_and_opencv_demo.ipynb
```

## 3. Environment

The notebook uses the Conda environment:

```text
face_ai
```

The environment is registered as a Jupyter kernel with:

```bash
python -m ipykernel install --user --name face_ai --display-name "Python (face_ai)"
```

## 4. Start Jupyter Notebook

Activate the environment:

```bash
conda activate face_ai
```

Start Jupyter Notebook from the repository root:

```bash
jupyter notebook
```

Select the following kernel:

```text
Python (face_ai)
```

## 5. Experiment Workflow

The notebook performs the following steps:

```text
Environment verification
→ Image loading
→ BGR-to-RGB conversion
→ Grayscale conversion
→ Gaussian blur
→ Canny edge detection
→ Matplotlib visualization
→ Output saving
→ Report generation
```

## 6. Input

The notebook expects an input image at:

```text
data/samples/test.jpg
```

Only safe, authorized, licensed, or synthetic images should be used.

## 7. Outputs

The generated image outputs are:

```text
outputs/images/jupyter_gray.jpg
outputs/images/jupyter_edges.jpg
```

The experiment report is:

```text
outputs/reports/jupyter_demo_result.txt
```

## 8. Verification

The notebook verifies:

* Python version
* PyTorch version
* CUDA availability
* CUDA version
* GPU device name
* OpenCV version
* NumPy version
* Matplotlib version
* Image loading
* Image processing
* Image visualization
* Output saving

## 9. Result

The Jupyter Notebook experiment was completed successfully.

The experiment confirms that the project environment supports interactive computer vision development with PyTorch, OpenCV, NumPy, and Matplotlib.

## 10. Limitations

This notebook is a basic environment and image processing demonstration.

It does not currently include model training, MMDetection, face landmark detection, face alignment, face recognition, or dataset exploration.
