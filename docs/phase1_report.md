# Phase 1 Report: Face Recognition Basics and Dataset Exploration

This report summarizes the Phase 1 work against the internship project plan and detailed task plan.

## Scope

Phase 1 focuses on face recognition basics, dataset exploration, face detection, face landmark localization, and a pretrained LFW verification baseline.

## Environment

Two environments are used to keep dependency conflicts clear:

| Environment | Purpose | Key packages |
| --- | --- | --- |
| `ml-gpu` | General computer vision, datasets, landmarks, recognition | PyTorch CUDA, OpenCV, Matplotlib, MediaPipe, InsightFace |
| `ml-mmdet` | MMDetection/MMCV experiments | MMEngine, MMCV, MMDetection |

The split avoids forcing MMCV-compatible PyTorch versions onto the newer GPU/recognition stack.

## Completed Tasks

| Requirement | Status | Evidence |
| --- | --- | --- |
| Task 2.1: Learn face detection, alignment, recognition, verification, identification | Completed through implementation notes and scripts | `README.md`, dataset and recognition scripts |
| Task 2.2: Explore CelebA and LFW with OpenCV and Matplotlib | Completed | `src/datasets/explore_lfw.py`, `src/datasets/explore_lfw_pairs.py`, `src/datasets/explore_celeba.py` |
| Task 2.3: Run MMDetection face detection on provided image | Completed | `src/mmdetection/run_face_detection_mmdet.py`, `outputs/mmdetection_face_detection/` |
| Task 2.4: Run face landmark localization on detected face | Completed | `src/landmarks/face_landmark_mediapipe.py`, `outputs/landmarks/` |
| LFW pretrained face recognition verification accuracy | Completed | `src/datasets/explore_lfw_pairs.py`, `outputs/reports/lfw_recognition_verification_result.txt` |

## Dataset Exploration Results

### LFW

- Images: 13,233
- Identities: 5,749
- Main format: one folder per identity, image files under each identity folder
- Pair protocol: same-person and different-person verification pairs
- Visual outputs:
  - `outputs/images/lfw_sample_grid.jpg`
  - `outputs/images/lfw_opencv_sample_grid.jpg`
  - `outputs/images/lfw_pair_examples.jpg`
  - `outputs/images/lfw_opencv_pair_examples.jpg`
  - `outputs/images/lfw_identity_distribution.jpg`
- Report outputs:
  - `outputs/reports/lfw_exploration_result.txt`
  - `outputs/reports/lfw_pair_exploration_result.txt`

### CelebA

- Images: 202,599
- Attribute labels: 40 binary attributes per image
- Extra annotations: bounding boxes and five facial landmarks
- Main format: flat image directory plus separate annotation text files
- Visual outputs:
  - `outputs/images/celeba_annotation_examples.jpg`
  - `outputs/images/celeba_opencv_annotation_examples.jpg`
  - `outputs/images/celeba_attribute_distribution.jpg`
- Report output:
  - `outputs/reports/celeba_exploration_result.txt`

## Detection Result

MMDetection was used on the provided face test image.

- Script: `src/mmdetection/run_face_detection_mmdet.py`
- Input: `data/samples/face_test.jpg`
- Output image: `outputs/mmdetection_face_detection/vis/face_test.jpg`
- Prediction JSON: `outputs/mmdetection_face_detection/preds/face_test.json`
- Report: `outputs/reports/mmdetection_face_detection_result.txt`
- Detected faces: 1

## Landmark Result

MediaPipe Face Landmarker was used to localize dense landmarks on the detected face.

- Script: `src/landmarks/face_landmark_mediapipe.py`
- Output image: `outputs/landmarks/face_test_mediapipe_landmarks.jpg`
- Report: `outputs/reports/face_landmark_mediapipe_result.txt`
- Detected faces: 1
- Landmark count: 478 dense landmarks
- Five-point summary generated from dense landmarks: left eye, right eye, nose tip, left mouth, right mouth

## LFW Recognition Verification Result

A pretrained InsightFace model was used for LFW verification on the official pair protocol.

- Script: `src/datasets/explore_lfw_pairs.py`
- Recognition report: `outputs/reports/lfw_recognition_verification_result.txt`
- Train pairs: 2,200
- Test pairs: 1,000
- Threshold selected on train pairs: 0.2288
- Train accuracy: 0.9482
- Test accuracy: 0.9440
- Correct test pairs: 944 / 1,000
- Confusion counts: TP 466, TN 478, FP 22, FN 34

## Deferred Or Limited Items

- MS-Celeb-1M is deferred because it is large and has licensing/cleanliness concerns. LFW and CelebA already cover the required Phase 1 exploration targets.
- The MMDetection face detector is kept in a separate `ml-mmdet` environment because MMCV has tighter version constraints than the newer GPU recognition stack.
- The project currently uses MMDetection for the required face detection task, and InsightFace/MediaPipe for recognition and landmarks. This keeps Phase 1 practical while still matching the required deliverables.

## Clean Project Layout

- Dataset exploration: `src/datasets/`
- MMDetection detection: `src/mmdetection/`
- Landmark localization: `src/landmarks/`
- Simpler OpenCV baseline: `src/detection/`
- Reports: `outputs/reports/`
- Visual evidence: `outputs/images/`, `outputs/mmdetection_face_detection/`, `outputs/landmarks/`

## Recommended Next Step

After Phase 1 is accepted, move into the next stage by selecting a stable recognition baseline and designing a repeatable evaluation workflow. Keep new experiments grouped by task instead of adding loose scripts at the project root.
