# Week 3 Plan: Face Detection Training Preparation

## Goal

Week 3 starts Phase 2, focusing on face detection model training. The practical target for this week is to prepare a reproducible WIDER FACE training pipeline with MMDetection, then run a small sanity check before full training.

## Phase 2 Context

The full Phase 2 target is to build a stronger face recognition system that includes:

- robust face detection under pose, occlusion, and lighting changes
- face landmark localization and alignment
- ResNet + ArcFace recognition training
- model optimization, ONNX export, and deployment preparation

Week 3 should not try to finish all Phase 2 tasks at once. The recommended scope is Task 3.1 to Task 3.3.

## Week 3 Scope

| Task | Status | Output |
| --- | --- | --- |
| Task 3.1: Study MTCNN and RetinaFace | Started | `docs/face_detection_algorithms.md` |
| Task 3.2: Prepare MMDetection WIDER FACE training | Started | `configs/mmdetection/widerface_retinanet_r50_fpn.py`, `src/datasets/convert_widerface_to_voc.py`, `src/mmdetection/check_widerface_dataset.py`, `src/mmdetection/train_widerface.py` |
| Task 3.3: Evaluate on WIDER FACE validation set | Pending | mAP report after dataset and training are ready |

## Dataset Requirement

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

The official WIDER FACE release uses text annotations, so a conversion step is required before training with MMDetection's built-in `WIDERFaceDataset`.

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

6. Write precision, recall, and mAP into `outputs/reports/widerface_retinanet_eval_result.txt`.

## Notes

- The first training baseline uses RetinaNet R50-FPN because MMDetection already provides a WIDER FACE config for it.
- RetinaFace is more face-specific and should be studied as the next stronger detector, but a clean MMDetection RetinaNet baseline is a practical first deliverable.
- WIDER FACE data and trained checkpoints should stay local and should not be committed to Git.

