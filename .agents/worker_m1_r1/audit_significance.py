"""
Statistical and Experimental Suite Audit Script for GNN-Bandit.
Genuine parsing and computation of statistical metrics across all seeds and experimental folders.
"""

import json
import glob
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

def load_results_by_dir(res_dir):
    """
    Loads all results_seed*.json from a directory.
    Returns: dict[dataset][seed][model][estimator] -> dict of stats (value, std, ci_lower, ci_upper, etc.)
    """
    data = defaultdict(lambda: defaultdict(dict))
    files = sorted(glob.glob(f"{res_dir}/**/results_seed*.json", recursive=True))
    for f in files:
        with open(f, "r") as fp:
            d = json.load(fp)
        ds = d.get("dataset")
        seed = d.get("seed")
        ope = d.get("ope_results", {})
        data[ds][seed] = ope
    return data

def run_significance_suite(dataset_data, primary_model="GNN-Bandit", estimator="DR"):
    """
    For a single dataset across seeds, compares primary_model to all baselines on given estimator.
    Returns a dataframe of comparison metrics.
    """
    seeds = sorted(dataset_data.keys())
    # Extract values per model
    model_values = defaultdict(list)
    for s in seeds:
        ope = dataset_data[s]
        for m, ests in ope.items():
            if estimator in ests and "value" in ests[estimator]:
                model_values[m].append(ests[estimator]["value"])
    
    if primary_model not in model_values:
        return None
    
    prim_vals = np.array(model_values[primary_model])
    n_seeds = len(prim_vals)
    prim_mean = np.mean(prim_vals)
    prim_std = np.std(prim_vals, ddof=1) if n_seeds > 1 else 0.0
    
    rows = []
    # Primary model row
    rows.append({
        "Model": primary_model,
        "Mean": prim_mean,
        "Std": prim_std,
        "Values": prim_vals.tolist(),
        "Lift (%)": 0.0,
        "Diff Mean": 0.0,
        "t-stat": np.nan,
        "t-pval": np.nan,
        "t-sig": "-",
        "W-stat": np.nan,
        "W-pval": np.nan,
        "W-sig": "-",
        "Is_Primary": True
    })
    
    for m, vals in model_values.items():
        if m == primary_model:
            continue
        vals_arr = np.array(vals)
        if len(vals_arr) != n_seeds:
            continue
        m_mean = np.mean(vals_arr)
        m_std = np.std(vals_arr, ddof=1) if n_seeds > 1 else 0.0
        diff = prim_vals - vals_arr
        diff_mean = prim_mean - m_mean
        lift = (diff_mean / abs(m_mean)) * 100.0 if m_mean != 0 else 0.0
        
        # Paired t-test
        if np.allclose(prim_vals, vals_arr):
            t_stat, t_pval = 0.0, 1.0
        else:
            t_stat, t_pval = stats.ttest_rel(prim_vals, vals_arr)
            
        # Wilcoxon signed-rank test
        if np.allclose(prim_vals, vals_arr):
            w_stat, w_pval = 0.0, 1.0
        else:
            try:
                # zero_method='wilcox' or 'pratt'
                res_w = stats.wilcoxon(prim_vals, vals_arr)
                w_stat, w_pval = res_w.statistic, res_w.pvalue
            except Exception as e:
                w_stat, w_pval = np.nan, np.nan
                
        def sig_marker(p):
            if np.isnan(p):
                return "N/A"
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            else:
                return "ns"
                
        rows.append({
            "Model": m,
            "Mean": m_mean,
            "Std": m_std,
            "Values": vals_arr.tolist(),
            "Lift (%)": lift,
            "Diff Mean": diff_mean,
            "t-stat": t_stat,
            "t-pval": t_pval,
            "t-sig": sig_marker(t_pval),
            "W-stat": w_stat,
            "W-pval": w_pval,
            "W-sig": sig_marker(w_pval),
            "Is_Primary": False
        })
        
    df = pd.DataFrame(rows)
    # sort baselines by Mean desc
    prim_df = df[df["Is_Primary"]]
    base_df = df[~df["Is_Primary"]].sort_values(by="Mean", ascending=False)
    return pd.concat([prim_df, base_df], ignore_index=True)

if __name__ == "__main__":
    for rdir in ["experiments/results-v2-lambda-0.05", "experiments/results-v2-lambda-0.1", "experiments/results-v2-lambda-0.2", "experiments/results"]:
        if not os.path.exists(rdir):
            continue
        print(f"\n=======================================================")
        print(f"DIRECTORY: {rdir}")
        print(f"=======================================================")
        res = load_results_by_dir(rdir)
        for ds in sorted(res.keys()):
            df = run_significance_suite(res[ds], primary_model="GNN-Bandit", estimator="DR")
            if df is not None:
                print(f"\n--- DATASET: {ds.upper()} (DR Estimator) ---")
                print(df[["Model", "Mean", "Std", "Lift (%)", "t-stat", "t-pval", "t-sig", "W-stat", "W-pval", "W-sig"]].to_string(index=False))
