# Face Landmark Detection and Alignment

## 4.1 Common Landmark Algorithms

Face landmark detection predicts semantic facial keypoints such as eye corners, nose tip, mouth corners, jawline, and eyebrow points. The output can be sparse, such as 5 points for alignment, or dense, such as 68, 98, or 468+ points.

### HRNet

HRNet keeps high-resolution feature maps through the network and repeatedly fuses multi-resolution branches. It is strong for landmark localization because keypoint coordinates require spatial precision.

Typical use:
- 68-point or 98-point landmark localization
- heatmap regression
- high-accuracy evaluation on 300W, WFLW, AFLW, COFW

Strengths:
- accurate on small facial structures
- preserves spatial details better than pure downsample-then-upsample networks

Weaknesses:
- heavier than simple CNN regressors
- training and inference cost is higher

### SAN

SAN, or Style Aggregated Network, improves landmark localization under appearance variation by aggregating style-related representations. It is useful when faces have pose, lighting, expression, or image-style changes.

Typical use:
- robust landmark detection under domain variation
- 300W-style benchmark evaluation

Strengths:
- better robustness to visual style changes
- useful for difficult images

Weaknesses:
- more complex training pipeline
- less convenient as a first baseline

### Heatmap Regression vs Coordinate Regression

Heatmap regression predicts a probability map for each landmark. The final point is obtained from the heatmap peak. This is common in HRNet-style models.

Coordinate regression directly predicts x/y values. It is simpler and lighter, but usually less accurate for precise landmark localization.

For this project:
- Use a simple coordinate-regression CNN as a first trainable baseline.
- Use NME as the main metric.
- Use five-point landmarks for face alignment.
- Later, replace the baseline with HRNet/MMPose if higher accuracy is required.

## 4.2 Dataset Plan: 300W or COFW

The landmark pipeline supports supervised training on 300W or COFW. These datasets are not currently present in `data/raw`, so this project provides dataset-check and training entry scripts first.

Expected 300W-style structure:

```text
data/raw/300W/
  images_or_subsets...
  *.jpg / *.png
  *.pts
```

A `.pts` file usually contains 68 landmark points:

```text
version: 1
n_points: 68
{
x y
...
}
```

## 4.3 Alignment

Face alignment estimates an affine/similarity transform from source landmarks to a canonical template. For recognition pipelines, five points are enough:

```text
left eye, right eye, nose tip, left mouth, right mouth
```

The script `src/landmarks/face_alignment.py` reads five-point landmarks and writes aligned 112x112 face crops.

## Metric: NME

Normalized Mean Error is:

```text
NME = mean Euclidean landmark error / normalization distance
```

Common normalization choices:
- inter-ocular distance for 300W 68-point landmarks
- bounding-box size for some difficult landmark datasets

This project defaults to inter-ocular distance using the outer eye-corner indices when 68 points are available.
## 300W Model Based Alignment

The script `src/landmarks/align_with_landmark_model.py` uses the cropped 300W landmark CNN checkpoint to predict 68 landmarks, derives five alignment points, and aligns faces to the ArcFace-style 112x112 template.

Five-point derivation from 68 landmarks:

- left eye: mean of points 36-41
- right eye: mean of points 42-47
- nose tip: point 30
- left mouth corner: point 48
- right mouth corner: point 54

Current small-scale result:

```text
Checkpoint: models/checkpoints/landmark_cnn_300w_cropped.pt
Samples aligned: 6
Mean NME: 0.2350
Grid image: outputs/landmarks/alignment_300w_model/landmark_model_alignment_grid.jpg
Report: outputs/reports/landmark_300w_alignment_result.txt
```

## GT vs Predicted Five-Point Alignment Compare

The script `src/landmarks/compare_gt_pred_alignment.py` compares the alignment upper bound from GT 68-point annotations against alignment from predicted 68-point landmarks.

Current validation-sample result:

```text
Checkpoint: models/checkpoints/landmark_cnn_300w_aug30_best.pt
Samples compared: 6
Mean pred NME: 0.1625
Mean normalized five-point delta: 0.2027
Mean GT template residual px: 3.8860
Mean Pred template residual px: 2.6075
Grid image: outputs/landmarks/alignment_compare_300w/gt_vs_pred_alignment_grid.jpg
Report: outputs/reports/landmark_300w_alignment_compare_result.txt
```

This comparison shows that five-point alignment quality is affected by both predicted landmark error and the mismatch between 300W landmark-derived five points and the ArcFace-style 112x112 template, especially for large-pose outdoor images.

## 300W Five-Point Template Calibration

The script `src/landmarks/calibrate_300w_alignment_template.py` calibrates a 300W-specific five-point alignment template from GT 68-point annotations. It uses the training split from `models/checkpoints/landmark_cnn_300w_aug30_best.pt`, derives five points for each sample, iteratively aligns them to the current template, and averages the aligned shapes.

Current result:

```text
Train shapes: 510
Val shapes: 90
Iterations: 20
Val ArcFace residual: 3.9948 px
Val 300W calibrated residual: 3.7051 px
Val improvement: 0.2897 px
Template JSON: outputs/reports/landmark_300w_calibrated_template.json
Template plot: outputs/landmarks/calibrated_template_300w/template_comparison.jpg
Grid image: outputs/landmarks/calibrated_template_300w/arcface_vs_300w_template_alignment.jpg
```

Calibrated 112x112 template:

```text
[[38.3673, 52.7174], [73.4310, 52.5792], [56.1849, 70.9255], [39.4389, 91.8267], [72.7087, 91.4551]]
```

The calibrated template reduces average validation residual, but the improvement is moderate. This indicates that 300W-to-ArcFace template mismatch exists, while large-pose and expression variation still require a stronger landmark model or pose-aware alignment strategy.
