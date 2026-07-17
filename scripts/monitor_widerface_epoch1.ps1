param(
    [int]$TrainPid = 33080,
    [string]$Workspace = 'F:\Internship\Bytedance\face-ai-project',
    [int]$MaxCpuTempC = 90,
    [int]$PollSeconds = 30
)

Set-Location $Workspace
$checkpoint = Join-Path $Workspace 'outputs\mmdetection_widerface\retinanet_r50_fpn_cpu_full\epoch_1.pth'
$monitorLog = Join-Path $Workspace 'outputs\reports\widerface_epoch1_monitor.log'

function Write-MonitorLog([string]$Message) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -LiteralPath $monitorLog -Value "[$stamp] $Message"
}

function Get-CpuTempC {
    try {
        $temps = Get-CimInstance -Namespace root\wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop
        $values = @()
        foreach ($temp in $temps) {
            if ($null -ne $temp.CurrentTemperature -and $temp.CurrentTemperature -gt 0) {
                $values += [math]::Round(($temp.CurrentTemperature / 10) - 273.15, 1)
            }
        }
        if ($values.Count -gt 0) {
            return ($values | Measure-Object -Maximum).Maximum
        }
    } catch {
        return $null
    }
    return $null
}

Write-MonitorLog "Monitor started. TrainPid=$TrainPid; checkpoint=$checkpoint; MaxCpuTempC=$MaxCpuTempC; PollSeconds=$PollSeconds"

while ($true) {
    $proc = Get-Process -Id $TrainPid -ErrorAction SilentlyContinue
    $tempC = Get-CpuTempC
    $tempText = if ($null -eq $tempC) { 'unavailable' } else { "$tempC C" }

    if ($null -eq $proc) {
        Write-MonitorLog "Training process is no longer running. Temperature=$tempText. Monitor exiting."
        break
    }

    if (Test-Path -LiteralPath $checkpoint) {
        Write-MonitorLog "epoch_1 checkpoint detected. Stopping training PID $TrainPid. Temperature=$tempText."
        Stop-Process -Id $TrainPid -Force
        Write-MonitorLog "Training stopped after epoch 1 checkpoint."
        break
    }

    if ($null -ne $tempC -and $tempC -ge $MaxCpuTempC) {
        Write-MonitorLog "CPU temperature $tempC C >= threshold $MaxCpuTempC C. Stopping training PID $TrainPid."
        Stop-Process -Id $TrainPid -Force
        Write-MonitorLog "Training stopped due to high temperature."
        break
    }

    Write-MonitorLog "Training running. CPUSeconds=$([math]::Round($proc.CPU, 1)); Temperature=$tempText; checkpoint_exists=False"
    Start-Sleep -Seconds $PollSeconds
}