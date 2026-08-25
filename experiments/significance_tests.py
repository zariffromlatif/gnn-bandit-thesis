"""
Statistical significance testing for thesis benchmark results.

Computes:
  - Paired Student's t-test (two-sided)
  - Wilcoxon signed-rank test
  - Significance markers: *** (p < 0.001), ** (p < 0.01), * (p < 0.05), ns (not significant)
  - Publication-ready LaTeX and Markdown comparison tables for KBS submission.

Usage:
    python experiments/significance_tests.py [--results_dir experiments/results-v2-lambda-0.05]
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def format_p_val(p: float) -> str:
    if p < 0.001:
        return f"{p:.1e} (***)"
    elif p < 0.01:
        return f"{p:.4f} (**)"
    elif p < 0.05:
        return f"{p:.4f} (*)"
    else:
        return f"{p:.4f} (ns)"


def run_tests(results_dir: str):
    root = Path(results_dir)
    if not root.exists():
        print(f"Directory {results_dir} not found!")
        return

    # Find all dataset subfolders
    datasets = [d.name for d in root.iterdir() if d.is_dir()]
    print(f"Found {len(datasets)} datasets in {results_dir}: {datasets}\n")

    for ds in sorted(datasets):
        ds_dir = root / ds
        files = sorted(ds_dir.glob("results_seed*.json"))
        if not files:
            continue

        # Extract metrics across seeds for all models
        # model -> list of DR values across seeds
        dr_values = defaultdict(list)

        for f in files:
            with open(f) as fp:
                data = json.load(fp)
            ope = data.get("ope_results", {})
            for model_name, metrics in ope.items():
                if "DR" in metrics and "value" in metrics["DR"]:
                    dr_values[model_name].append(metrics["DR"]["value"])

        if "GNN-Bandit" not in dr_values:
            print(f"[{ds}] GNN-Bandit not found in results. Skipping.")
            continue

        gnn_scores = np.array(dr_values["GNN-Bandit"])
        n_seeds = len(gnn_scores)
        gnn_mean = gnn_scores.mean()
        gnn_std = gnn_scores.std()

        print(f"{'='*80}")
        print(f"DATASET: {ds.upper()} ({n_seeds} seeds)")
        print(f"GNN-Bandit Mean DR: {gnn_mean:.6f} ± {gnn_std:.6f}")
        print(f"{'='*80}")
        print(f"{'Baseline':<22} | {'Mean DR ± Std':<22} | {'Lift vs Baseline':<16} | {'t-test p-value':<18} | {'Wilcoxon p-value':<18}")
        print(f"{'-'*22}-|-{'-'*22}-|-{'-'*16}-|-{'-'*18}-|-{'-'*18}")

        # Sort baselines by mean score descending
        sorted_baselines = sorted(
            [m for m in dr_values.keys() if m != "GNN-Bandit"],
            key=lambda m: np.mean(dr_values[m]),
            reverse=True
        )

        for model in sorted_baselines:
            scores = np.array(dr_values[model])
            if len(scores) != n_seeds:
                continue

            m_mean = scores.mean()
            m_std = scores.std()
            lift = (gnn_mean - m_mean) / max(abs(m_mean), 1e-9) * 100.0

            # Paired t-test
            if np.allclose(gnn_scores, scores):
                p_ttest = 1.0
                p_wilcox = 1.0
            else:
                _, p_ttest = stats.ttest_rel(gnn_scores, scores)
                try:
                    res_w = stats.wilcoxon(gnn_scores, scores)
                    p_wilcox = res_w.pvalue
                except Exception:
                    p_wilcox = float("nan")

            ttest_str = format_p_val(p_ttest)
            wilcox_str = format_p_val(p_wilcox) if not np.isnan(p_wilcox) else "N/A"

            print(f"{model:<22} | {m_mean:.6f} ± {m_std:.6f} | {lift:+7.2f}%         | {ttest_str:<18} | {wilcox_str:<18}")

        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="experiments/results-v2-lambda-0.05",
                        help="Path to results directory containing dataset folders.")
    args = parser.parse_args()
    run_tests(args.results_dir)
