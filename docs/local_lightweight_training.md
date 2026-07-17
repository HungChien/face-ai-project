# Local Lightweight Face Recognition Training

This local profile is designed for laptop training where temperature and sustained CPU/GPU load matter more than maximum speed.

## What Changed

The heavy experiments used:

```text
ResNet50 / IResNet50
5000-10000 identities
30 train images per identity
batch size 64
20 epochs
no cooldown
```

The local profile uses:

```text
MobileFaceNet
2000 identities
12 train images per identity
3 val images per identity
batch size 24
12 epochs
num_workers 0
60 seconds cooldown between epochs
```

This reduces GPU computation, CPU image loading pressure, and sustained thermal load. It is intended as a local development baseline, not the final high-accuracy model.

## Run

From PowerShell:

```powershell
cd F:\Internship\Bytedance\face-ai-project
.\scripts\run_local_mobilefacenet_light.ps1
```

The script trains the local MobileFaceNet baseline and then runs LFW 6000 pairs / 10-fold verification.

## Outputs

```text
models/checkpoints/mobilefacenet_ms1m_local_light_best.pt
outputs/reports/mobilefacenet_ms1m_local_light_result.txt
outputs/reports/mobilefacenet_ms1m_local_light_history.json
outputs/reports/mobilefacenet_ms1m_local_light_console_log.txt
outputs/reports/mobilefacenet_ms1m_local_light_lfw_10fold_result.txt
outputs/images/mobilefacenet_ms1m_local_light_curves.jpg
```

## If Temperature Is Still Too High

Reduce these in order:

```text
--batch-size 24 -> 16
--num-identities 2000 -> 1000
--max-train-images-per-identity 12 -> 8
--epochs 12 -> 8
--epoch-cooldown-seconds 60 -> 120
```

The safest very-light profile is:

```text
1000 identities
8 train images per identity
batch size 16
8 epochs
120 seconds cooldown
```

## If You Want More Accuracy Later

Keep this local profile for smoke tests and development. Run the heavier ResNet50/IResNet50 experiments on a rented GPU server.
