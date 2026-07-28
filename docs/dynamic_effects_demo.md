# Dynamic Face Effects Demo

This module covers the dynamic effects part of Phase 3.

## Method

The demo implements a frame-level face effects pipeline:

1. Run MediaPipe FaceLandmarker on key frames.
2. Smooth landmarks with exponential moving average.
3. Reuse the previous smoothed landmarks between key frames for lightweight tracking.
4. Add dynamic AR stickers anchored to face landmarks.
5. Add beauty and makeup effects with landmark-driven masks.
6. Export an MP4 demo video and a contact sheet.

Implemented effects:

- Dynamic glasses with animated lens highlight.
- Animated hat anchored to forehead and face width.
- Beauty: face-mask smoothing and mild brightening.
- Makeup: lipstick and cheek blush.

## Run

```powershell
D:\Anaconda3\envs\ml-gpu\python.exe src\effects\run_dynamic_effects_demo.py `
  --image outputs\effects\images\indoor_016_original.jpg `
  --model models\checkpoints\face_landmarker.task `
  --frames 96 `
  --fps 24 `
  --width 360 `
  --detect-width 180 `
  --detect-every 4 `
  --output-video outputs\effects\videos\dynamic_face_effects_demo.mp4 `
  --contact-sheet outputs\effects\videos\dynamic_face_effects_contact_sheet.jpg `
  --report outputs\reports\dynamic_effects_demo_result.txt `
  --json-report outputs\reports\dynamic_effects_demo_result.json
```

## Current Result

- Frames: 96
- Output video FPS: 24
- Frame size: 360 x 530
- Landmark detection width: 180
- Detection interval: every 4 frames
- Detected key frames: 24 / 96
- Reused smoothed landmarks: 72 / 96
- Processing speed: 6.66 FPS on local CPU/TFLite FaceLandmarker path

## Outputs

- Code: `src/effects/run_dynamic_effects_demo.py`
- Demo video: `outputs/effects/videos/dynamic_face_effects_demo.mp4`
- Contact sheet: `outputs/effects/videos/dynamic_face_effects_contact_sheet.jpg`
- Text report: `outputs/reports/dynamic_effects_demo_result.txt`
- JSON report: `outputs/reports/dynamic_effects_demo_result.json`

## Notes

The output MP4 plays at 24 FPS. The measured processing FPS is the local offline processing speed, not the playback frame rate. Further real-time optimization can use GPU/TFLite acceleration, lower preview resolution, optical-flow landmark propagation, or C++/mobile inference deployment.

