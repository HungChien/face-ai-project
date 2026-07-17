# 300W Dataset Download and Placement

This project uses 300W for Phase 2 facial landmark detection and alignment.

## Official Source

Official page: https://ibug.doc.ic.ac.uk/resources/300-W/

The 300W release is provided as four split zip parts:

- `300w.zip.001`
- `300w.zip.002`
- `300w.zip.003`
- `300w.zip.004`

The official site requires a download form with name, email, and affiliation. Use this dataset only for research/education according to the license notes on the official page.

## Placement

Put the four downloaded files here:

```text
data/raw/300W_OFFICIAL/
  300w.zip.001
  300w.zip.002
  300w.zip.003
  300w.zip.004
```

Extracted data should be placed under:

```text
data/raw/300W/
```

After placement, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\extract_300w.ps1
```

Then verify:

```powershell
D:\Anaconda3\envs\ml-gpu\python.exe src\landmarks\check_300w_dataset.py
```

Expected verification: image files and `.pts` landmark files are detected, usually with 68 landmark points per annotation.
