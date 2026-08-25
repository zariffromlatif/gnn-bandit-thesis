
import json
import glob
import numpy as np

lambdas = ["0.05", "0.1", "0.2"]
datasets = ["obd-all", "obd-men", "obd-women", "criteo"]
models = ["GNN-Bandit", "CQL", "DecisionTransformer"]

print(f"{'Dataset':<12} | {'Lambda':<6} | {'Model':<20} | {'DR Mean':<10} | {'DR Std':<10} | {'CV (%)':<10}")
print("-" * 80)

for ds in datasets:
    for lam in lambdas:
        results = {m: [] for m in models}
        files = glob.glob(f"experiments/results-v2-lambda-{lam}/{ds}/results_seed*.json")
        for f in files:
            with open(f, "r") as fp:
                data = json.load(fp)
                for m in models:
                    if m in data.get("ope_results", {}):
                        results[m].append(data["ope_results"][m]["DR"]["value"])
        
        for m in models:
            if len(results[m]) > 0:
                mean_dr = np.mean(results[m])
                std_dr = np.std(results[m])
                cv = (std_dr / mean_dr) * 100 if mean_dr != 0 else 0
                print(f"{ds:<12} | {lam:<6} | {m:<20} | {mean_dr:.6f}   | {std_dr:.6f}   | {cv:.2f}%")
        print("-" * 80)

