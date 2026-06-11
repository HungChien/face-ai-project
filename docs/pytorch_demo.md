# PyTorch Demo

## 1. Objective

The objective of this demo is to verify that PyTorch can successfully run a complete minimal training workflow in the project environment.

This demo confirms that the environment supports:

- Tensor creation
- Model definition
- Forward propagation
- Loss computation
- Backward propagation
- Parameter update
- Model checkpoint saving

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

## 3. Script Path

The PyTorch demo script is located at:

```text
src/pytorch_demo.py
```

## 4. Run Command

Activate the Conda environment:

```bash
conda activate face_ai
```

Run the script from `<project-root>`:

```bash
python src/pytorch_demo.py
```

On Windows, this command can also be written as:

```bash
python src\pytorch_demo.py
```

## 5. Demo Design

The script creates a small synthetic classification task.

Input data:

```text
Number of samples: 200
Input dimension: 10
Number of classes: 2
```

Model structure:

```text
Linear(10 -> 32)
ReLU
Linear(32 -> 2)
```

Training configuration:

```text
Loss function: CrossEntropyLoss
Optimizer: Adam
Learning rate: 0.001
Epochs: 10
```

## 6. Output Files

The script generates the following local files:

```text
models/checkpoints/pytorch_demo.pth
outputs/reports/pytorch_demo_result.txt
```

Note:

```text
The model checkpoint is generated locally and is not required to be committed to GitHub.
```

## 7. Expected Terminal Output

A successful run should print messages similar to:

```text
Epoch [1/10] Loss: 0.7048 Accuracy: 0.5150
Epoch [2/10] Loss: 0.7011 Accuracy: 0.5250
...
Epoch [10/10] Loss: 0.6753 Accuracy: 0.5950

PyTorch demo completed successfully.
Model checkpoint saved to: models/checkpoints/pytorch_demo.pth
Training report saved to: outputs/reports/pytorch_demo_result.txt
```

The exact loss and accuracy values may differ because the input data is randomly generated.

## 8. Result Explanation

This demo verifies that PyTorch can perform a complete training loop:

```text
Input tensor
→ Forward propagation
→ Loss computation
→ Backward propagation
→ Optimizer step
→ Model checkpoint saving
```

If CUDA is available, the script automatically uses GPU. Otherwise, it uses CPU.

## 9. Conclusion

The PyTorch demo was completed successfully if the script generated the training report and saved the model checkpoint locally.

This confirms that the project environment is ready for future deep learning tasks, including face detection, facial landmark detection, face recognition, model optimization, and deployment.