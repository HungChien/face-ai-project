# Docker Hello World

## 1. Objective

The objective of this task is to learn the basic Docker workflow and verify that the project can run a Python program inside a container.

The task covers:

```text
Dockerfile creation
→ Docker image build
→ Container execution
→ Output verification
```

## 2. Repository Files

The Docker Hello World example uses the following files:

```text
Dockerfile
.dockerignore
src/hello_docker.py
outputs/reports/docker_hello_result.txt
```

## 3. Python Script

The Docker container runs:

```text
src/hello_docker.py
```

The script prints a Hello World message, the Python version, and the container platform information.

## 4. Dockerfile

The Dockerfile uses the following base image:

```text
python:3.10-slim
```

The container working directory is:

```text
/app
```

The Python script is copied into the container and executed when the container starts.

## 5. Build Command

Run the following command from the repository root:

```bash
docker build -t face-ai-hello .
```

This command creates a Docker image named:

```text
face-ai-hello
```

## 6. Run Command

Run the container with:

```bash
docker run --rm face-ai-hello
```

The `--rm` option automatically removes the container after execution.

## 7. Expected Output

A successful run should produce output similar to:

```text
Hello from Docker!
The face-ai-project container is running successfully.
Python version: 3.10.x
Platform: Linux x86_64
```

The exact Python version and platform architecture may vary.

## 8. Verification

The Docker image can be checked with:

```bash
docker images
```

The image list should include:

```text
face-ai-hello
```

## 9. Result

The Docker image was built successfully, and the Python Hello World program ran successfully inside the container.

This confirms that the basic Docker workflow is working:

```text
Source code
→ Dockerfile
→ Docker image
→ Docker container
→ Program output
```

## 10. Limitations

This Docker image is intentionally minimal.

It does not currently include:

```text
PyTorch
OpenCV
CUDA
MMDetection
Project datasets
Model checkpoints
```

The purpose of this task is only to verify the basic Docker workflow.

A more complete project image can be created later after the main dependencies and framework versions are finalized.