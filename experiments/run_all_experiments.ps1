<#
.SYNOPSIS
Batch script to run all GNN-Bandit experiments for Q1 journal publication.

.DESCRIPTION
This script iterates over the required OBD datasets (obd-all, obd-men, obd-women)
and runs the Main, Ablation, Sensitivity, and Cold-Start experiments for 5 random seeds (0 to 4).
This is designed to be run on a powerful machine to collect all the data needed for the paper.

.EXAMPLE
.\run_all_experiments.ps1
#>

$ErrorActionPreference = "Stop"

# Configuration
$Datasets = @("obd-all", "obd-men", "obd-women", "criteo")
$Seeds = "0,1,2,3,4"
$OutputDir = "experiments/results"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " GNN-Bandit Batch Experiment Runner" -ForegroundColor Cyan
Write-Host " Datasets: $($Datasets -join ', ')" -ForegroundColor Cyan
Write-Host " Seeds: $Seeds" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Ensure we are in the project root (where the script is meant to be run from, or one level up)
# If this script is inside experiments/, we should cd to the parent directory first
$scriptPath = $MyInvocation.MyCommand.Path
$parentDir = Split-Path (Split-Path $scriptPath)
Set-Location $parentDir

Write-Host "Running from: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

foreach ($dataset in $Datasets) {
    Write-Host ">>> Starting full suite for dataset: $dataset" -ForegroundColor Green

    # 1. Main Experiment
    Write-Host "    [1/4] Running Main Experiment..." -ForegroundColor Yellow
    python experiments/run_main.py --dataset $dataset --seeds $Seeds --output $OutputDir
    if ($LASTEXITCODE -ne 0) { throw "run_main.py failed for $dataset" }

    # 2. Ablation Study
    Write-Host "    [2/4] Running Ablation Study..." -ForegroundColor Yellow
    python experiments/run_ablation.py --dataset $dataset --seeds $Seeds --output $OutputDir
    if ($LASTEXITCODE -ne 0) { throw "run_ablation.py failed for $dataset" }

    # 3. Sensitivity Analysis
    Write-Host "    [3/4] Running Sensitivity Analysis..." -ForegroundColor Yellow
    python experiments/run_sensitivity.py --dataset $dataset --seeds $Seeds --output $OutputDir
    if ($LASTEXITCODE -ne 0) { throw "run_sensitivity.py failed for $dataset" }

    # 4. Cold-Start Analysis
    Write-Host "    [4/4] Running Cold-Start Analysis..." -ForegroundColor Yellow
    python experiments/run_cold_start.py --dataset $dataset --seeds $Seeds --output $OutputDir
    if ($LASTEXITCODE -ne 0) { throw "run_cold_start.py failed for $dataset" }

    Write-Host ">>> Completed full suite for dataset: $dataset`n" -ForegroundColor Green
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ALL EXPERIMENTS COMPLETE!" -ForegroundColor Cyan
Write-Host " Results are saved in: $OutputDir" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
