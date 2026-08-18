
$lambdas = @(0.05, 0.1, 0.2)
$datasets = @("obd-all", "obd-men", "obd-women", "criteo")

foreach ($lambda in $lambdas) {
    Write-Host "========================================="
    Write-Host "Running Sweep for lambda = $lambda"
    Write-Host "========================================="
    $out_dir = "experiments/results-v2-lambda-$lambda"
    
    foreach ($ds in $datasets) {
        # Run all seeds for the dataset
        python -u experiments/run_main.py --dataset $ds --seeds "0,1,2,3,4" --output $out_dir --cfr_lambda $lambda
    }
}

Write-Host "Sweep complete! Don't forget to push the results back to git."

