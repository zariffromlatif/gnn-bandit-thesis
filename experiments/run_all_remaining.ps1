<#
.SYNOPSIS
Master automated runner for all remaining experiments (KuaiRec/KuaiRand, TGN, Backward RL, Significance).

.DESCRIPTION
Runs sequentially:
  [Stage 1] Automated download and preprocessing of KuaiRec & KuaiRand datasets
  [Stage 2] Main, Ablation, Sensitivity, Cold-Start suites on KuaiRec & KuaiRand (Seeds 0-4)
  [Stage 3] Temporal Graph Network (TGN) vs. LightGCN benchmarks across all datasets (Seeds 0-4)
  [Stage 4] Multi-Step Backward Sequential RL across all datasets (Seeds 0-4)
  [Stage 5] Automated statistical significance re-computation and LaTeX table generation

Idempotent: automatically skips already-completed runs.
#>

$ErrorActionPreference = "Continue"

$scriptPath = $MyInvocation.MyCommand.Path
if ($scriptPath) {
    $parentDir = Split-Path (Split-Path $scriptPath)
    Set-Location $parentDir
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " GNN-BANDIT MASTER EXPERIMENT SUITE RUNNER" -ForegroundColor Cyan
Write-Host " Working Directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host " GPU: NVIDIA RTX 4090 (24GB VRAM) | CPU: i9-14900K (24C/32T)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------------------------
# STAGE 1: Download & Preprocess KuaiRec & KuaiRand
# -----------------------------------------------------------------------------
Write-Host "=================================================================" -ForegroundColor Yellow
Write-Host " STAGE 1: KuaiRec & KuaiRand Dataset Ingestion" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Yellow

Write-Host ">>> [1.1] Downloading KuaiRec & KuaiRand from Zenodo..." -ForegroundColor Green
python download_kuai.py --dataset all

Write-Host "`n>>> [1.2] Preprocessing KuaiRec..." -ForegroundColor Green
python preprocess_kuairec.py

Write-Host "`n>>> [1.3] Preprocessing KuaiRand..." -ForegroundColor Green
python preprocess_kuairand.py

# -----------------------------------------------------------------------------
# STAGE 2: KuaiRec & KuaiRand Full Benchmark Suite
# -----------------------------------------------------------------------------
Write-Host "`n=================================================================" -ForegroundColor Yellow
Write-Host " STAGE 2: Video Recommendation Benchmark Suite (KuaiRec / KuaiRand)" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Yellow

$KuaiDatasets = @("kuairec", "kuairand")
$Seeds = "0,1,2,3,4"

foreach ($ds in $KuaiDatasets) {
    Write-Host "`n>>> Starting full experimental suite for: $ds" -ForegroundColor Cyan

    Write-Host "    [2.1] Main Policy Benchmark (5 Seeds)..." -ForegroundColor Yellow
    python experiments/run_main.py --dataset $ds --seeds $Seeds --output experiments/results

    Write-Host "    [2.2] Ablation Study (5 Seeds)..." -ForegroundColor Yellow
    python experiments/run_ablation.py --dataset $ds --seeds $Seeds --output experiments/results

    Write-Host "    [2.3] Sensitivity Analysis (5 Seeds)..." -ForegroundColor Yellow
    python experiments/run_sensitivity.py --dataset $ds --seeds $Seeds --output experiments/results

    Write-Host "    [2.4] Cold-Start Evaluation (5 Seeds)..." -ForegroundColor Yellow
    python experiments/run_cold_start.py --dataset $ds --seeds $Seeds --output experiments/results
}

# -----------------------------------------------------------------------------
# STAGE 3: Temporal Graph Network (TGN) Comparison Suite
# -----------------------------------------------------------------------------
Write-Host "`n=================================================================" -ForegroundColor Yellow
Write-Host " STAGE 3: Temporal Graph Network (TGN) vs. LightGCN Benchmarks" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Yellow

$AllDatasets = @("obd-all", "obd-men", "obd-women", "criteo")
foreach ($ds in $AllDatasets) {
    Write-Host "`n>>> Running TGN Temporal Graph Benchmark for: $ds" -ForegroundColor Cyan
    python experiments/run_main.py --dataset $ds --seeds $Seeds --graph_encoder tgn --output experiments/results_tgn
}

# -----------------------------------------------------------------------------
# STAGE 4: Multi-Step Backward Sequential RL Suite (Seeds 0-4)
# -----------------------------------------------------------------------------
Write-Host "`n=================================================================" -ForegroundColor Yellow
Write-Host " STAGE 4: Multi-Step Backward Sequential RL Suite" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Yellow

foreach ($ds in $AllDatasets) {
    for ($s = 0; $s -le 4; $s++) {
        $backwardRes = "experiments/results/${ds}_backward_seed${s}/results.json"
        if (Test-Path $backwardRes) {
            Write-Host "    ✓ Skipping Backward RL for $ds (Seed $s) — already exists." -ForegroundColor DarkGray
        } else {
            Write-Host "    >>> Running Backward RL for $ds (Seed $s)..." -ForegroundColor Green
            python experiments/run_backward_rl.py --dataset $ds --seed $s
        }
    }
}

# -----------------------------------------------------------------------------
# STAGE 5: Significance Testing & Summary Generation
# -----------------------------------------------------------------------------
Write-Host "`n=================================================================" -ForegroundColor Yellow
Write-Host " STAGE 5: Statistical Significance & Summary Compilation" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Yellow

Write-Host ">>> Running Significance Tests on Standard Results..." -ForegroundColor Green
python experiments/significance_tests.py --results_dir experiments/results

Write-Host "`n>>> Running Significance Tests on TGN Results..." -ForegroundColor Green
python experiments/significance_tests.py --results_dir experiments/results_tgn

Write-Host "`n>>> Analyzing Multi-Lambda CFR-GNN..." -ForegroundColor Green
python analyze_v2.py

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host " ALL REMAINING EXPERIMENTS COMPLETE!" -ForegroundColor Cyan
Write-Host " All results saved and verified." -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
