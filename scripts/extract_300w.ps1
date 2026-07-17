param(
    [string]$ArchiveDir = "data\raw\300W_OFFICIAL",
    [string]$OutputDir = "data\raw\300W",
    [string]$PythonExe = "D:\Anaconda3\envs\ml-gpu\python.exe"
)

$ErrorActionPreference = "Stop"

$requiredParts = @(
    "300w.zip.001",
    "300w.zip.002",
    "300w.zip.003",
    "300w.zip.004"
)

$missing = @()
foreach ($part in $requiredParts) {
    $path = Join-Path $ArchiveDir $part
    if (-not (Test-Path -LiteralPath $path)) {
        $missing += $path
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Missing 300W split archive files:" -ForegroundColor Yellow
    foreach ($path in $missing) {
        Write-Host "  $path"
    }
    Write-Host ""
    Write-Host "Download the four parts from the official 300W page and place them under $ArchiveDir."
    exit 1
}

$sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
$sevenZipPath = $null
if ($sevenZip) {
    $sevenZipPath = $sevenZip.Source
    if (-not $sevenZipPath) {
        $sevenZipPath = $sevenZip.Path
    }
}
if (-not $sevenZipPath) {
    $candidatePaths = @(
        "C:\Program Files\7-Zip\7z.exe",
        "C:\Program Files (x86)\7-Zip\7z.exe"
    )
    foreach ($candidate in $candidatePaths) {
        if (Test-Path -LiteralPath $candidate) {
            $sevenZipPath = $candidate
            break
        }
    }
}

if (-not $sevenZipPath) {
    Write-Host "7-Zip was not found. Install it first, then rerun this script." -ForegroundColor Yellow
    Write-Host "Suggested command: winget install 7zip.7zip"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$firstPart = Join-Path $ArchiveDir "300w.zip.001"
& $sevenZipPath x $firstPart "-o$OutputDir" -y

if ($LASTEXITCODE -ne 0) {
    throw "7-Zip extraction failed with exit code $LASTEXITCODE"
}

if (Test-Path -LiteralPath $PythonExe) {
    & $PythonExe src\landmarks\check_300w_dataset.py --root $OutputDir
} else {
    Write-Host "Python executable not found: $PythonExe" -ForegroundColor Yellow
    Write-Host "Run dataset check manually after activating the correct environment."
}
