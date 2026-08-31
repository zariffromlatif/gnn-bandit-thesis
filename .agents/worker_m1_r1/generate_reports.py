"""
Comprehensive Statistical & Margin Data Generator.
Calculates all statistics, p-values, t-stats, W-stats, margins, sensitivities,
and generates structured Markdown report files.
"""

import os
import sys
import json
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

def compute_detailed_stats(res_dir, primary_model="GNN-Bandit", estimators=["DR", "SNIPW", "IPW", "DM"]):
    """
    Computes comprehensive statistics for each dataset in res_dir.
    Returns: dict[dataset][estimator] -> DataFrame with full metrics
    """
    files = sorted(glob.glob(f"{res_dir}/**/results_seed*.json", recursive=True))
    dataset_records = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for f in files:
        with open(f, "r") as fp:
            d = json.load(fp)
        ds = d.get("dataset")
        seed = d.get("seed")
        ope = d.get("ope_results", {})
        for m, est_dict in ope.items():
            for est_name, m_vals in est_dict.items():
                if isinstance(m_vals, dict) and "value" in m_vals:
                    dataset_records[ds][est_name][m].append(m_vals["value"])
    
    results = defaultdict(dict)
    for ds, est_data in dataset_records.items():
        for est_name in estimators:
            if est_name not in est_data:
                continue
            model_vals = est_data[est_name]
            if primary_model not in model_vals:
                continue
            
            prim_scores = np.array(model_vals[primary_model])
            n_seeds = len(prim_scores)
            prim_mean = np.mean(prim_scores)
            prim_std = np.std(prim_scores, ddof=1) if n_seeds > 1 else 0.0
            
            rows = []
            # Primary model
            rows.append({
                "Model": primary_model,
                "Mean": prim_mean,
                "Std": prim_std,
                "Diff": 0.0,
                "Lift_pct": 0.0,
                "t_stat": np.nan,
                "t_pval": np.nan,
                "t_sig": "-",
                "W_stat": np.nan,
                "W_pval": np.nan,
                "W_sig": "-",
                "Seeds": prim_scores.tolist(),
                "Rank": 1
            })
            
            for m, scores in model_vals.items():
                if m == primary_model:
                    continue
                scores_arr = np.array(scores)
                if len(scores_arr) != n_seeds:
                    continue
                m_mean = np.mean(scores_arr)
                m_std = np.std(scores_arr, ddof=1) if n_seeds > 1 else 0.0
                diff = prim_mean - m_mean
                lift = (diff / abs(m_mean)) * 100.0 if m_mean != 0 else 0.0
                
                # Paired t-test
                if np.allclose(prim_scores, scores_arr):
                    t_stat, t_pval = 0.0, 1.0
                else:
                    t_stat, t_pval = stats.ttest_rel(prim_scores, scores_arr)
                
                # Wilcoxon signed-rank test
                if np.allclose(prim_scores, scores_arr):
                    w_stat, w_pval = 0.0, 1.0
                else:
                    try:
                        res_w = stats.wilcoxon(prim_scores, scores_arr)
                        w_stat, w_pval = res_w.statistic, res_w.pvalue
                    except Exception:
                        w_stat, w_pval = np.nan, np.nan
                
                def get_sig(p):
                    if np.isnan(p): return "N/A"
                    if p < 0.001: return "***"
                    elif p < 0.01: return "**"
                    elif p < 0.05: return "*"
                    else: return "ns"
                
                rows.append({
                    "Model": m,
                    "Mean": m_mean,
                    "Std": m_std,
                    "Diff": diff,
                    "Lift_pct": lift,
                    "t_stat": t_stat,
                    "t_pval": t_pval,
                    "t_sig": get_sig(t_pval),
                    "W_stat": w_stat,
                    "W_pval": w_pval,
                    "W_sig": get_sig(w_pval),
                    "Seeds": scores_arr.tolist(),
                    "Rank": 0
                })
            
            df = pd.DataFrame(rows)
            # Re-rank by mean descending
            df = df.sort_values(by="Mean", ascending=False).reset_index(drop=True)
            df["Rank"] = df.index + 1
            results[ds][est_name] = df
            
    return results

if __name__ == "__main__":
    v2_stats = compute_detailed_stats("experiments/results-v2-lambda-0.05")
    print("Computed v2-lambda-0.05 stats for datasets:", list(v2_stats.keys()))
