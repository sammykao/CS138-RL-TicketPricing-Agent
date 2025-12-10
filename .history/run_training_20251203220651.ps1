# Quick script to run visualization (same as run_visualization.ps1)
# Usage: .\run_training.ps1 [-Checkpoint PATH] [-StepDelay MS]

param(
    [string]$Checkpoint = $null,
    [int]$StepDelay = 2
)

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

$args = @()
if ($Checkpoint) {
    $args += "-Checkpoint", $Checkpoint
}
$args += "-StepDelay", $StepDelay

# Run visualization
& .\run_visualization.ps1 @args

