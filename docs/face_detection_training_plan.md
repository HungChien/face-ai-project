# Face Detection Training Plan

## Goal

This note organizes the face detection training workflow: prepare a reproducible WIDER FACE pipeline with MMDetection, validate it with a small sanity check, and keep the path open for longer detector runs.

## System Context

The broader recognition stack includes:

- robust face detection under pose, occlusion, and lighting changes
- face landmark localization and alignment
- ResNet + ArcFace recognition training
- model optimization, ONNX export, and deployment preparation

This plan keeps the detector work focused on algorithm study, WIDER FACE training, and validation metrics.

## Scope

| Experiment | Status | Output |
| --- | --- | --- |
| Detector algorithm survey | Started | `docs/face_detection_algorithms.md` |
| MMDetection WIDER FACE pipeline | Started | `configs/mmdetection/widerface_retinanet_r50_fpn.py`, `src/datasets/convert_widerface_to_voc.py`, `src/mmdetection/check_widerface_dataset.py`, `src/mmdetection/train_widerface.py` |
| WIDER FACE validation metrics | Pending | mAP metrics after dataset and training are ready |

## Dataset Format

MMDetection 3.3.0 includes `WIDERFaceDataset`, but it expects WIDER FACE in Pascal VOC style:

```text
data/raw/WIDERFace/
  train.txt
  val.txt
  WIDER_train/
    Annotations/
    images/
  WIDER_val/
    Annotations/
    images/
```

The official WIDER FACE release uses text annotations, so a conversion step is needed before training with MMDetection's built-in `WIDERFaceDataset`.

## Recommended Order

1. Download WIDER FACE train/val images and annotations locally.
2. Convert WIDER FACE annotations to Pascal VOC XML format.

```powershell
conda activate ml-gpu
python src\datasets\convert_widerface_to_voc.py --wider-root data\raw\WIDER_FACE_OFFICIAL --copy-images
```

3. Run:

```powershell
conda activate ml-mmdet
python src\mmdetection\check_widerface_dataset.py
```

4. If the dataset check passes, run training:

```powershell
conda activate ml-mmdet
python src\mmdetection\train_widerface.py
```

5. Evaluate on validation set:

```powershell
conda activate ml-mmdet
python src\mmdetection\evaluate_widerface.py
```

6. Save precision, recall, and mAP into `outputs/reports/widerface_retinanet_eval_result.txt`.

## Notes

- The first training baseline uses RetinaNet R50-FPN because MMDetection already provides a WIDER FACE config for it.
- RetinaFace is more face-specific and should be studied as the next stronger detector, but a clean MMDetection RetinaNet baseline is a practical first artifact.
- WIDER FACE data and trained checkpoints should stay local and should not be committed to Git.

