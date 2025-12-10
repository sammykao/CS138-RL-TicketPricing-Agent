# Setup and run RL Ticket Pricing Visualization
# Usage: 
#   Episode mode:  .\run_visualization.ps1
#   Training mode: .\run_visualization.ps1 -Mode training
#   With checkpoint: .\run_visualization.ps1 -Checkpoint "path\to\checkpoint.pt" -Mode training

param(
    [string]$Checkpoint = $null,
    [int]$StepDelay = 25,
    [float]$DemandScale = 1
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "Setting up RL Ticket Pricing Visualization..." -ForegroundColor Green

# Check if uv is installed
try {
    $null = Get-Command uv -ErrorAction Stop
} catch {
    Write-Host "Warning: uv is not installed. Installing uv..." -ForegroundColor Yellow
    pip install uv
}

# Get script directory and navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check if learning_environment directory exists
$learningEnvPath = Join-Path $scriptPath "learning_environment"
if (-not (Test-Path $learningEnvPath)) {
    Write-Host "Error: learning_environment directory not found!" -ForegroundColor Red
    exit 1
}

# Sync dependencies (creates/updates .venv)
Write-Host "Syncing dependencies with uv..." -ForegroundColor Green
uv sync

# Build argument list for visualization script
$visualizationArgs = @()

if ($Checkpoint) {
    $visualizationArgs += "--checkpoint", $Checkpoint
}

$visualizationArgs += "--step-delay", $StepDelay
$visualizationArgs += "--demand-scale", $DemandScale

# Run visualization
Write-Host "Starting visualization..." -ForegroundColor Green
Write-Host "Controls: SPACE=Step | R=Reset | T=Toggle Mode | ESC=Quit" -ForegroundColor Yellow
Write-Host ""

Set-Location $learningEnvPath
uv run python visualization/run_visualization.py $visualizationArgs

