# Face Detection Baseline

## 1. Objective

The objective of this baseline is to verify that the project environment can perform basic face detection on an input image.

This baseline is the first step toward a complete face recognition pipeline.

A complete face recognition system usually includes:

```text
Face detection
→ Facial landmark detection
→ Face alignment
→ Face feature extraction
→ Face similarity comparison
```

This demo only focuses on the first step: face detection.

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

## 3. Input Image

The input image is placed at:

```text
data/samples/face_test.jpg
```

The image used in this experiment contains one clear frontal human face in the foreground, with a complex outdoor city background.

To avoid copyright and portrait-rights issues, the input image should be either:

```text
1. A self-taken image
2. An authorized image
3. A properly licensed public sample image
4. A synthetic face image with a suitable license
```

Unauthorized human face images downloaded from the internet should not be used or committed to the repository.

## 4. Script Path

The face detection baseline script is located at:

```text
src/detection/face_detect_opencv.py
```

## 5. Method

This baseline uses OpenCV Haar Cascade for frontal face detection.

The processing pipeline is:

```text
Read input image
→ Convert image to grayscale
→ Load OpenCV Haar Cascade detector
→ Detect face bounding boxes
→ Draw bounding boxes
→ Save output image and report
```

The detector used in this baseline is:

```text
haarcascade_frontalface_default.xml
```

## 6. Run Command

Activate the Conda environment:

```bash
conda activate ml-gpu
```

Run the script from `<project-root>`:

```bash
python src/detection/face_detect_opencv.py
```

On Windows, this command can also be written as:

```bash
python src\detection\face_detect_opencv.py
```

## 7. Output Files

The script generates the following files:

```text
outputs/images/face_detect_opencv.jpg
outputs/reports/face_detection_baseline_result.txt
```

## 8. Experiment Result

In this experiment, the detector produced 7 bounding boxes in total.

The main foreground face was successfully detected. However, several additional false positive boxes were also detected in the background and clothing regions.

Result summary:

```text
Total detected boxes: 7
True positive faces: 1
False positives: 6
```

The output image shows that:

```text
1. The main frontal face was detected correctly.
2. Several small regions in the background were incorrectly detected as faces.
3. One region on the clothing pattern was also incorrectly detected as a face.
```

## 9. Result Explanation

The output image contains green bounding boxes around all detected regions.

The largest bounding box corresponds to the actual face in the foreground. The smaller boxes are false positives caused by background patterns, building textures, clothing logos, and local high-contrast regions.

Each bounding box is represented as:

```text
x, y, width, height
```

where `(x, y)` is the top-left corner of the detected region.

## 10. Error Analysis

The baseline result shows that OpenCV Haar Cascade can detect a clear frontal face, but it is not robust enough for complex real-world scenes.

The false positives are mainly caused by the following factors:

```text
1. The input image has a complex city background.
2. The background contains many small high-contrast structures.
3. The clothing contains logos, text, and local patterns.
4. Haar Cascade is a traditional feature-based method and may confuse face-like textures with real faces.
5. The original detection parameters allow relatively small candidate regions, which increases the probability of false positives.
```

This is a typical limitation of traditional face detection methods.

## 11. Limitations

This baseline uses a traditional OpenCV Haar Cascade detector.

Its limitations include:

```text
1. Lower accuracy than modern deep learning-based face detectors
2. Weak performance under complex backgrounds
3. Weak performance on side faces
4. Weak performance under occlusion
5. Weak performance under large pose variation
6. Sensitivity to high-contrast non-face textures
7. Higher false positive rate in cluttered scenes
```

Therefore, this baseline should only be used for early-stage pipeline verification.

It is not suitable as the final face detection solution for this project.

## 12. Possible Parameter Tuning

To reduce false positives, stricter detection parameters can be used.

Initial baseline parameters:

```python
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)
```

Possible improved parameters:

```python
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=8,
    minSize=(100, 100),
)
```

A larger `minSize` can filter out very small false positive regions, and a larger `minNeighbors` value can make the detector more conservative.

However, parameter tuning only provides limited improvement. For more reliable results, a modern deep learning-based detector should be used.

## 13. Next Step

The next step is to replace this traditional baseline with a stronger face detector.

Candidate methods include:

```text
1. MTCNN
2. RetinaFace
3. InsightFace detection module
4. Other deep learning-based face detectors
```

Compared with Haar Cascade, these methods usually provide better robustness under:

```text
1. Complex background
2. Large pose variation
3. Partial occlusion
4. Different lighting conditions
5. Small faces
```

This baseline result provides a useful reference point for future comparison.

## 14. Conclusion

The OpenCV Haar Cascade face detection baseline was successfully implemented and tested.

The baseline successfully detected the main frontal face, which confirms that the project can complete the basic face detection pipeline:

```text
Input image
→ Face detection
→ Bounding box visualization
→ Result image saving
→ Detection report generation
```

At the same time, the experiment also revealed multiple false positives in the background and clothing regions. This demonstrates the limitation of traditional Haar Cascade detection and motivates the use of stronger deep learning-based detectors in later stages.

Overall, this baseline is considered successful because it verifies the detection workflow and provides a clear starting point for future improvement.