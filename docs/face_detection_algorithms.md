# Face Detection Algorithms: MTCNN, RetinaFace, and MMDetection Baseline

## 1. Detection Problem

Face detection predicts whether an image contains faces and returns bounding boxes for each face. A robust detector should handle:

- small faces
- side faces and large pose changes
- partial occlusion
- strong lighting changes
- crowded scenes
- low-resolution or blurry faces

For this project, detection is the first stage of the detect-align-recognize pipeline.

```text
image -> face detection -> landmark localization -> alignment -> recognition
```

## 2. MTCNN

MTCNN is a cascaded face detector. It uses three stages:

| Stage | Role |
| --- | --- |
| P-Net | Proposes candidate face boxes quickly at multiple scales |
| R-Net | Refines candidates and removes false positives |
| O-Net | Produces final boxes and facial landmarks |

Key ideas:

- image pyramid for multi-scale faces
- cascade design to reject easy background regions early
- joint detection and landmark prediction
- non-maximum suppression between stages

Strengths:

- simple and widely used
- returns both boxes and five facial landmarks
- good for small projects and preprocessing pipelines

Limitations:

- weaker than modern dense detectors on hard WIDER FACE cases
- cascade can be slower when many candidates exist
- performance drops on heavy occlusion and extreme poses

## 3. RetinaFace

RetinaFace is a single-stage face detector designed specifically for face detection. It is usually stronger than generic object detectors on hard face cases.

Key ideas:

- dense anchor-based prediction
- feature pyramid for multi-scale faces
- face classification and bounding-box regression
- facial landmark regression
- optional extra supervision such as dense face localization in the original work

Strengths:

- strong performance on WIDER FACE
- robust for small faces and hard subsets
- naturally supports landmarks for later alignment

Limitations:

- training setup is more specialized than a generic detector
- high accuracy usually depends on good pretrained weights and carefully tuned anchors

## 4. RetinaNet as MMDetection Baseline

MMDetection includes a WIDER FACE RetinaNet R50-FPN config. It is not as face-specialized as RetinaFace, but it is a clean first training baseline because:

- the config is already available in MMDetection
- the model has a standard detection training and evaluation pipeline
- it can produce a validation mAP report
- it helps verify dataset conversion, dataloading, training, and evaluation before moving to stronger detectors

This project uses RetinaNet R50-FPN as the first WIDER FACE training baseline.

## 5. Evaluation Metrics

The validation summary should include:

| Metric | Meaning |
| --- | --- |
| Precision | Among predicted faces, how many are correct |
| Recall | Among ground-truth faces, how many are detected |
| mAP | Mean average precision across confidence thresholds |

MMDetection's built-in WIDER FACE config uses `VOCMetric` with `mAP` in 11-point mode.

## 6. Practical Plan

The immediate detector training path is:

1. Prepare WIDER FACE in Pascal VOC XML format.
2. Verify dataset paths and annotation files.
3. Train RetinaNet R50-FPN with MMDetection.
4. Evaluate validation mAP.
5. Visualize detections on test images.
6. Compare the trained detector with the GroundingDINO face-prompt result.
