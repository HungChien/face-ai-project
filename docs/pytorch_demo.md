# PyTorch Demo

## 1. Objective

The objective of this demo is to verify that the project environment can successfully run a complete minimal PyTorch training workflow.

This demo is part of the Week 1 setup and baseline verification tasks. It confirms that the local environment supports basic deep learning operations before moving on to face detection, facial landmark detection, face alignment, face recognition, and facial effects.

This demo verifies the following PyTorch workflow:

```text
Tensor creation
→ Model definition
→ Forward propagation
→ Loss computation
→ Backward propagation
→ Parameter update
→ Training log generation
→ Model checkpoint saving
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

## 3. Environment Context

The PyTorch demo was executed in the configured Conda environment:

```text
Environment name: face_ai
```

The local environment supports CUDA and uses the following GPU:

```text
GPU name: NVIDIA GeForce RTX 2060
CUDA version: 12.1
PyTorch version: 2.5.1+cu121
```

This confirms that the PyTorch demo can run with GPU acceleration.

## 4. Script Path

The PyTorch demo script is located at:

```text
src/pytorch_demo.py
```

## 5. Run Command

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

## 6. Method

This demo uses a small synthetic binary classification task.

The data is randomly generated instead of being loaded from real images. This keeps the demo simple and focuses only on verifying the PyTorch training pipeline.

The data configuration is:

```text
Number of samples: 200
Input dimension: 10
Number of classes: 2
```

The model is a simple fully connected neural network:

```text
Linear(10 → 32)
→ ReLU
→ Linear(32 → 2)
```

The training configuration is:

```text
Loss function: CrossEntropyLoss
Optimizer: Adam
Learning rate: 0.001
Number of epochs: 10
Device: cuda
```

## 7. Output Files

The script generates the following files:

```text
models/checkpoints/pytorch_demo.pth
outputs/reports/pytorch_demo_result.txt
```

The model checkpoint is generated locally. In this project, model checkpoint files are usually not required to be committed to GitHub because real model files may become large in later stages.

The training report can be committed because it is a lightweight text file and records the experiment result.

## 8. Experiment Result

The actual training result is:

```text
PyTorch Demo Training Result
==================================================
Device: cuda
Number of samples: 200
Input dimension: 10
Number of classes: 2
Number of epochs: 10
Learning rate: 0.001

Epoch [1/10] Loss: 0.6851 Accuracy: 0.5450
Epoch [2/10] Loss: 0.6836 Accuracy: 0.5650
Epoch [3/10] Loss: 0.6821 Accuracy: 0.5700
Epoch [4/10] Loss: 0.6806 Accuracy: 0.5700
Epoch [5/10] Loss: 0.6792 Accuracy: 0.5700
Epoch [6/10] Loss: 0.6778 Accuracy: 0.5700
Epoch [7/10] Loss: 0.6765 Accuracy: 0.5750
Epoch [8/10] Loss: 0.6751 Accuracy: 0.5650
Epoch [9/10] Loss: 0.6738 Accuracy: 0.5700
Epoch [10/10] Loss: 0.6726 Accuracy: 0.5800
```

## 9. Result Summary

The result summary is:

```text
Device used: cuda
Initial loss: 0.6851
Final loss: 0.6726
Initial accuracy: 0.5450
Final accuracy: 0.5800
Loss trend: decreasing
Accuracy trend: slightly increasing
```

The model successfully completed all 10 training epochs on GPU.

The loss decreased from `0.6851` to `0.6726`, which shows that the optimizer was able to update the model parameters and reduce the training objective.

The accuracy increased from `0.5450` to `0.5800`, which indicates that the model learned some weak patterns from the synthetic data.

## 10. Result Explanation

This demo is not designed to achieve high accuracy. The dataset is randomly generated, so there is no strong real-world pattern for the model to learn.

The main purpose of this experiment is to verify that the following operations work correctly:

```text
1. PyTorch can create tensors successfully.
2. The model can be moved to CUDA.
3. Forward propagation works correctly.
4. CrossEntropyLoss can compute the training loss.
5. Backward propagation can compute gradients.
6. The Adam optimizer can update model parameters.
7. Training logs can be printed and saved.
8. Model checkpoints can be saved locally.
```

The successful completion of the training loop confirms that the PyTorch environment is ready for more complex deep learning tasks.

## 11. Error Analysis

The final accuracy is only `0.5800`, which is expected for this demo.

The main reasons are:

```text
1. The input data is randomly generated.
2. The labels are also randomly generated.
3. There is no meaningful image structure or semantic pattern in the dataset.
4. The model is intentionally small.
5. The training only runs for 10 epochs.
```

Therefore, the accuracy should not be interpreted as a model performance benchmark.

Instead, this result should be interpreted as an environment and pipeline verification result.

## 12. Limitations

This PyTorch demo has several limitations:

```text
1. It does not use real image data.
2. It does not evaluate on a validation set.
3. It does not measure generalization performance.
4. It does not include data loading with Dataset and DataLoader.
5. It does not include image preprocessing.
6. It does not include a convolutional neural network.
7. It is not related to face recognition accuracy.
```

This demo is only used for early-stage verification.

Later stages will use real computer vision datasets and more suitable model architectures.

## 13. Next Step

After this PyTorch demo, the project can move to computer vision and face-related baselines.

The next tasks include:

```text
1. OpenCV face detection baseline
2. Stronger face detection with MTCNN or RetinaFace
3. Facial landmark detection
4. Face alignment
5. Face embedding extraction
6. Face verification using cosine similarity
7. Model optimization and deployment
```

For future deep learning experiments, the project should gradually introduce:

```text
1. Real image datasets
2. PyTorch Dataset and DataLoader
3. CNN-based models
4. Training and validation split
5. Evaluation metrics
6. Checkpoint management
7. GPU memory monitoring
```

## 14. Conclusion

The PyTorch demo was completed successfully.

The experiment confirmed that the environment can run a complete minimal training workflow on GPU:

```text
Synthetic input data
→ Neural network forward pass
→ Loss computation
→ Backward propagation
→ Optimizer update
→ Training report generation
→ Model checkpoint saving
```

The model ran on `cuda`, which confirms that PyTorch can use the local NVIDIA GeForce RTX 2060 GPU.

Although the accuracy is not high, this is expected because the task uses randomly generated data. The important result is that the loss decreased during training and the full PyTorch training pipeline worked correctly.

Overall, this demo is considered successful because it verifies that the project environment is ready for later deep learning tasks, including face detection, facial landmark detection, face recognition, and facial effects.