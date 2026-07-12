# %% [markdown]
# # Multi-Campaign & Cross-Dataset Comparison
#
# **Key thesis claim:** The GNN-Bandit framework generalises across
# different campaigns (OBD-all, OBD-men, OBD-women) and even across
# a completely different dataset (Criteo Uplift v2.1, binary treatment).
#
# This notebook:
#   1. Loads result JSONs from `experiments/results/`
#   2. Builds comparison tables and grouped bar charts
#   3. Computes relative improvement over baselines
#   4. Handles multi-seed aggregation (mean +/- std) if available
#
# Produces **Figures 7-8 and Tables 2-3** in the paper.

# %% — Setup
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (10, 6),
})
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %% — Discover available results
DATASET_NAMES = ["obd-all", "obd-men", "obd-women", "criteo"]
DATASET_DISPLAY = {
    "obd-all":   "OBD-All",
    "obd-men":   "OBD-Men",
    "obd-women": "OBD-Women",
    "criteo":    "Criteo Uplift",
}

# Scan for result files
available = {}
for ds_name in DATASET_NAMES:
    ds_dir = RESULTS_DIR / ds_name
    if ds_dir.exists():
        files = sorted(ds_dir.glob("results_seed*.json"))
        if files:
            available[ds_name] = files
            print(f"  {ds_name}: {len(files)} seed(s) found")

if not available:
    print("\nNo result files found in experiments/results/")
    print("Run the experiments first:")
    print("  python experiments/run_main.py --dataset all --seeds 0,1,2,3,4")
    raise SystemExit(1)

print(f"\nTotal: {len(available)} dataset(s) with results")

# %% — Load and aggregate results
def load_results(json_files: list) -> dict:
    """Load result JSONs and aggregate across seeds."""
    all_seeds = []
    for f in json_files:
        with open(f) as fh:
            all_seeds.append(json.load(fh))
    return all_seeds

def extract_dr_table(all_seeds: list) -> pd.DataFrame:
    """
    Extract DR values from all seeds into a DataFrame.
    Returns: DataFrame with columns [Method, DR_mean, DR_std, DR_ci_lower, DR_ci_upper, n_seeds]
    """
    rows = []
    method_seed_vals = {}  # method -> list of DR values across seeds

    for seed_data in all_seeds:
        ope = seed_data.get("ope_results", {})
        for method, estimators in ope.items():
            dr = estimators.get("DR", {})
            val = dr.get("value")
            if val is not None:
                method_seed_vals.setdefault(method, []).append(val)

    for method, vals in method_seed_vals.items():
        arr = np.array(vals)
        rows.append({
            "Method": method,
            "DR_mean": float(arr.mean()),
            "DR_std": float(arr.std()) if len(arr) > 1 else 0.0,
            "DR_min": float(arr.min()),
            "DR_max": float(arr.max()),
            "n_seeds": len(arr),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("DR_mean", ascending=False).reset_index(drop=True)
    return df


all_tables = {}
for ds_name, files in available.items():
    seeds = load_results(files)
    tbl = extract_dr_table(seeds)
    all_tables[ds_name] = tbl
    print(f"\n{'='*50}")
    print(f"{DATASET_DISPLAY.get(ds_name, ds_name)} — DR Results")
    print(f"{'='*50}")
    print(tbl.to_string(index=False))

# %% — Figure: Grouped bar chart — DR by method across datasets
# Only include datasets that have results
plot_datasets = [ds for ds in DATASET_NAMES if ds in all_tables]
n_datasets = len(plot_datasets)

if n_datasets == 0:
    print("No data to plot.")
else:
    # Collect all methods across all datasets for consistent ordering
    method_order_candidates = [
        "GNN-Bandit", "MF-Bandit", "Greedy-GNN", "DQN",
        "BTS", "Uplift-Only", "Random",
    ]
    all_methods_seen = set()
    for tbl in all_tables.values():
        all_methods_seen.update(tbl["Method"].tolist())
    method_order = [m for m in method_order_candidates if m in all_methods_seen]
    # Add any methods not in the predefined order
    for m in sorted(all_methods_seen):
        if m not in method_order:
            method_order.append(m)

    n_methods = len(method_order)

    METHOD_COLORS = {
        "GNN-Bandit":  "#2ecc71",
        "MF-Bandit":   "#1abc9c",
        "Greedy-GNN":  "#3498db",
        "DQN":         "#e74c3c",
        "BTS":         "#f39c12",
        "Uplift-Only": "#9b59b6",
        "Random":      "#95a5a6",
    }
    # Fallback color for unknown methods
    _fallback_colors = ["#e67e22", "#2c3e50", "#d35400", "#c0392b"]

    fig, ax = plt.subplots(figsize=(max(10, n_datasets * 3), 6))

    x = np.arange(n_datasets)
    bar_width = 0.8 / n_methods

    for i, method in enumerate(method_order):
        vals = []
        errs = []
        for ds in plot_datasets:
            tbl = all_tables[ds]
            row = tbl[tbl["Method"] == method]
            if len(row) > 0:
                vals.append(row["DR_mean"].values[0])
                errs.append(row["DR_std"].values[0])
            else:
                vals.append(0)
                errs.append(0)

        offset = (i - n_methods / 2 + 0.5) * bar_width
        color = METHOD_COLORS.get(method, _fallback_colors[i % len(_fallback_colors)])
        ax.bar(
            x + offset, vals, bar_width,
            yerr=errs if any(e > 0 for e in errs) else None,
            capsize=2, label=method, color=color,
            edgecolor="black", linewidth=0.4,
            error_kw={"linewidth": 0.8},
        )

    ax.set_xlabel("Dataset / Campaign")
    ax.set_ylabel("DR Policy Value Estimate")
    ax.set_title("Multi-Campaign Comparison: DR Reward by Method")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_DISPLAY.get(ds, ds) for ds in plot_datasets])
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    sns.despine()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "multi_campaign_dr_comparison.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "multi_campaign_dr_comparison.png", bbox_inches="tight")
    plt.show()
    print(f"Saved: {FIG_DIR / 'multi_campaign_dr_comparison.pdf'}")

# %% — Figure: Relative improvement of GNN-Bandit over each baseline
if n_datasets > 0:
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5),
                             squeeze=False)

    for col, ds in enumerate(plot_datasets):
        ax = axes[0, col]
        tbl = all_tables[ds]

        gnn_row = tbl[tbl["Method"] == "GNN-Bandit"]
        if len(gnn_row) == 0:
            ax.set_title(DATASET_DISPLAY.get(ds, ds) + "\n(no GNN-Bandit result)")
            continue

        gnn_dr = gnn_row["DR_mean"].values[0]
        baselines = tbl[tbl["Method"] != "GNN-Bandit"].copy()

        if len(baselines) == 0:
            continue

        # Relative improvement
        baselines["rel_improvement"] = np.where(
            np.abs(baselines["DR_mean"]) > 1e-8,
            (gnn_dr - baselines["DR_mean"]) / np.abs(baselines["DR_mean"]) * 100,
            0.0,
        )
        baselines = baselines.sort_values("rel_improvement", ascending=True)

        colors = ["#2ecc71" if v >= 0 else "#e74c3c"
                  for v in baselines["rel_improvement"]]
        bars = ax.barh(baselines["Method"], baselines["rel_improvement"],
                       color=colors, edgecolor="black", linewidth=0.4)
        for bar, val in zip(bars, baselines["rel_improvement"]):
            x_pos = bar.get_width() + (1 if val >= 0 else -1)
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f}%", va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=9, fontweight="bold")

        ax.set_xlabel("Relative DR Improvement (%)")
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.axvline(x=0, color="black", linewidth=0.8)
        sns.despine(ax=ax)

    plt.suptitle("GNN-Bandit Improvement Over Baselines", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "relative_improvement_all_datasets.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "relative_improvement_all_datasets.png", bbox_inches="tight")
    plt.show()
    print(f"Saved: {FIG_DIR / 'relative_improvement_all_datasets.pdf'}")

# %% — Table: Full OPE comparison (all estimators)
print("\n" + "=" * 80)
print("FULL OPE TABLE (All Estimators)")
print("=" * 80)

for ds_name in plot_datasets:
    seeds = load_results(available[ds_name])
    print(f"\n--- {DATASET_DISPLAY.get(ds_name, ds_name)} ---")
    print(f"{'Method':<20s} {'IPW':>12s} {'SNIPW':>12s} {'DM':>12s} {'DR':>12s}")
    print("-" * 68)

    # Aggregate across seeds
    method_estimates = {}  # method -> {estimator -> [values]}
    for seed_data in seeds:
        ope = seed_data.get("ope_results", {})
        for method, estimators in ope.items():
            if method not in method_estimates:
                method_estimates[method] = {}
            for est_name, est_data in estimators.items():
                method_estimates[method].setdefault(est_name, []).append(
                    est_data.get("value", 0))

    for method in method_order:
        if method not in method_estimates:
            continue
        est = method_estimates[method]
        ipw_val  = np.mean(est.get("IPW", [0]))
        snipw_val = np.mean(est.get("SNIPW", [0]))
        dm_val   = np.mean(est.get("DM", [0]))
        dr_val   = np.mean(est.get("DR", [0]))
        print(f"{method:<20s} {ipw_val:>12.6f} {snipw_val:>12.6f} "
              f"{dm_val:>12.6f} {dr_val:>12.6f}")

# %% — Sleeping Dogs comparison across campaigns
print("\n" + "=" * 80)
print("SLEEPING DOGS ANALYSIS")
print("=" * 80)

sd_data_all = {}
for ds_name in plot_datasets:
    seeds = load_results(available[ds_name])
    for seed_data in seeds:
        sd = seed_data.get("sleeping_dogs", {})
        if sd:
            sd_data_all.setdefault(ds_name, []).append(sd)

if sd_data_all:
    for ds_name, sd_seeds in sd_data_all.items():
        print(f"\n--- {DATASET_DISPLAY.get(ds_name, ds_name)} ---")
        for sd in sd_seeds:
            for policy_name, stats in sd.items():
                n_sd = stats.get("n_sleeping_dog", 0)
                n_per = stats.get("n_persuadable", 0)
                prob_sd = stats.get("avg_max_prob_sleeping_dog", 0)
                prob_per = stats.get("avg_max_prob_persuadable", 0)
                print(f"  {policy_name:<20s}  "
                      f"Sleeping Dogs: {n_sd:>4d} (avg prob {prob_sd:.4f})  "
                      f"Persuadables: {n_per:>4d} (avg prob {prob_per:.4f})")

    # Figure: Sleeping Dogs intervention probabilities
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = []
    for ds_name, sd_seeds in sd_data_all.items():
        for sd in sd_seeds:
            for policy_name, stats in sd.items():
                plot_data.append({
                    "Dataset": DATASET_DISPLAY.get(ds_name, ds_name),
                    "Policy": policy_name,
                    "Sleeping Dog Prob": stats.get("avg_max_prob_sleeping_dog", 0),
                    "Persuadable Prob": stats.get("avg_max_prob_persuadable", 0),
                })
    if plot_data:
        sd_df = pd.DataFrame(plot_data)
        sd_melted = sd_df.melt(
            id_vars=["Dataset", "Policy"],
            value_vars=["Sleeping Dog Prob", "Persuadable Prob"],
            var_name="Segment", value_name="Avg Max Action Prob",
        )
        sns.barplot(data=sd_melted, x="Policy", y="Avg Max Action Prob",
                    hue="Segment", ax=ax,
                    palette={"Sleeping Dog Prob": "#e74c3c",
                             "Persuadable Prob": "#2ecc71"})
        ax.set_title("Sleeping Dogs Detection:\nIntervention Probability by User Segment")
        ax.set_ylabel("Average Max Action Probability")
        ax.legend(title="User Segment")
        plt.xticks(rotation=15)
        sns.despine()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "sleeping_dogs_comparison.pdf", bbox_inches="tight")
        plt.savefig(FIG_DIR / "sleeping_dogs_comparison.png", bbox_inches="tight")
        plt.show()
        print(f"Saved: {FIG_DIR / 'sleeping_dogs_comparison.pdf'}")
else:
    print("  No sleeping dogs data found in results.")

# %% — Figure: Campaign-specific heatmap
if len(plot_datasets) > 1 and len(method_order) > 0:
    # Build a heatmap matrix: rows = methods, cols = datasets
    heatmap_data = np.full((len(method_order), len(plot_datasets)), np.nan)
    for j, ds in enumerate(plot_datasets):
        tbl = all_tables[ds]
        for i, method in enumerate(method_order):
            row = tbl[tbl["Method"] == method]
            if len(row) > 0:
                heatmap_data[i, j] = row["DR_mean"].values[0]

    fig, ax = plt.subplots(figsize=(max(6, len(plot_datasets) * 2), len(method_order) * 0.6 + 2))
    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(plot_datasets)))
    ax.set_xticklabels([DATASET_DISPLAY.get(ds, ds) for ds in plot_datasets])
    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels(method_order)

    # Annotate cells
    for i in range(len(method_order)):
        for j in range(len(plot_datasets)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.003 else "black"
                ax.text(j, i, f"{val:.5f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    ax.set_title("DR Policy Value Heatmap Across Campaigns")
    plt.colorbar(im, ax=ax, label="DR Value", shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dr_heatmap_campaigns.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "dr_heatmap_campaigns.png", bbox_inches="tight")
    plt.show()
    print(f"Saved: {FIG_DIR / 'dr_heatmap_campaigns.pdf'}")

# %% — Export LaTeX tables for the paper
print("\n" + "=" * 80)
print("LATEX TABLE (copy-paste into paper)")
print("=" * 80)

for ds_name in plot_datasets:
    tbl = all_tables[ds_name]
    display_name = DATASET_DISPLAY.get(ds_name, ds_name)
    n_seeds = tbl["n_seeds"].max() if len(tbl) > 0 else 1

    print(f"\n% --- {display_name} ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(f"\\caption{{OPE Results on {display_name} "
          f"({'averaged over ' + str(n_seeds) + ' seeds' if n_seeds > 1 else 'seed 0'})"
          r"}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")

    if n_seeds > 1:
        print(r"Method & DR (mean $\pm$ std) & Relative $\Delta$ \\")
    else:
        print(r"Method & DR Value & Relative $\Delta$ \\")

    print(r"\midrule")

    gnn_row = tbl[tbl["Method"] == "GNN-Bandit"]
    gnn_dr = gnn_row["DR_mean"].values[0] if len(gnn_row) > 0 else 0

    for _, row in tbl.iterrows():
        method = row["Method"]
        dr_mean = row["DR_mean"]
        dr_std = row["DR_std"]

        if abs(gnn_dr) > 1e-8 and method != "GNN-Bandit":
            rel = (gnn_dr - dr_mean) / abs(gnn_dr) * 100
            delta_str = f"{rel:+.1f}\\%"
        elif method == "GNN-Bandit":
            delta_str = "---"
        else:
            delta_str = "N/A"

        bold_start = r"\textbf{" if method == "GNN-Bandit" else ""
        bold_end = "}" if method == "GNN-Bandit" else ""

        if n_seeds > 1:
            print(f"  {bold_start}{method}{bold_end} "
                  f"& ${dr_mean:.6f} \\pm {dr_std:.6f}$ & {delta_str} \\\\")
        else:
            print(f"  {bold_start}{method}{bold_end} "
                  f"& {dr_mean:.6f} & {delta_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

# %% — Ablation results loader (if available)
print("\n" + "=" * 80)
print("ABLATION RESULTS")
print("=" * 80)

for ds_name in plot_datasets:
    ds_dir = RESULTS_DIR / ds_name
    ablation_files = sorted(ds_dir.glob("ablation_seed*.json"))
    if not ablation_files:
        print(f"\n  {ds_name}: No ablation results found.")
        continue

    print(f"\n--- {DATASET_DISPLAY.get(ds_name, ds_name)} ---")
    for af in ablation_files:
        with open(af) as fh:
            abl = json.load(fh)
        print(f"  {af.name}:")
        print(f"  {'Variant':<30s} {'DR':>12s} {'95% CI':>25s}")
        print(f"  {'-'*67}")
        for variant, estimators in abl.items():
            dr = estimators.get("DR", {})
            val = dr.get("value", "N/A")
            ci_lo = dr.get("ci_lower", "")
            ci_hi = dr.get("ci_upper", "")
            if isinstance(val, (int, float)):
                print(f"  {variant:<30s} {val:>12.6f} "
                      f"[{ci_lo:.6f}, {ci_hi:.6f}]")
            else:
                print(f"  {variant:<30s} {str(val):>12s}")

# %% — Sensitivity results loader (if available)
print("\n" + "=" * 80)
print("SENSITIVITY RESULTS")
print("=" * 80)

for ds_name in plot_datasets:
    ds_dir = RESULTS_DIR / ds_name
    sens_files = sorted(ds_dir.glob("sensitivity_seed*.json"))
    if not sens_files:
        print(f"\n  {ds_name}: No sensitivity results found.")
        continue

    print(f"\n--- {DATASET_DISPLAY.get(ds_name, ds_name)} ---")
    for sf in sens_files:
        with open(sf) as fh:
            sens = json.load(fh)
        for sweep_name, sweep_results in sens.items():
            print(f"\n  Sweep: {sweep_name}")
            for param_val, metrics in sweep_results.items():
                dr = metrics.get("DR_value", "N/A")
                if isinstance(dr, (int, float)):
                    print(f"    {param_val:>8s} -> DR = {dr:.6f}")
                else:
                    print(f"    {param_val:>8s} -> DR = {dr}")

    # Plot sensitivity curves
    with open(sens_files[0]) as fh:
        sens = json.load(fh)

    n_sweeps = len(sens)
    if n_sweeps > 0:
        fig, axes = plt.subplots(1, n_sweeps, figsize=(5 * n_sweeps, 4))
        if n_sweeps == 1:
            axes = [axes]

        for idx, (sweep_name, sweep_results) in enumerate(sens.items()):
            ax = axes[idx]
            x_vals = []
            y_vals = []
            y_lo = []
            y_hi = []
            for param_val, metrics in sweep_results.items():
                dr = metrics.get("DR_value")
                if dr is not None and not (isinstance(dr, float) and np.isnan(dr)):
                    x_vals.append(float(param_val))
                    y_vals.append(dr)
                    y_lo.append(metrics.get("DR_ci_lower", dr))
                    y_hi.append(metrics.get("DR_ci_upper", dr))

            if x_vals:
                ax.plot(x_vals, y_vals, "o-", color="#2ecc71", linewidth=2,
                        markersize=7, markeredgecolor="black", markeredgewidth=0.5)
                ax.fill_between(x_vals, y_lo, y_hi, alpha=0.2, color="#2ecc71")
            ax.set_xlabel(sweep_name.replace("_", " ").title())
            ax.set_ylabel("DR Value")
            ax.set_title(f"Sensitivity: {sweep_name}")
            sns.despine(ax=ax)

        plt.suptitle(f"Hyperparameter Sensitivity ({DATASET_DISPLAY.get(ds_name, ds_name)})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"sensitivity_{ds_name}.pdf", bbox_inches="tight")
        plt.savefig(FIG_DIR / f"sensitivity_{ds_name}.png", bbox_inches="tight")
        plt.show()
        print(f"Saved: {FIG_DIR / f'sensitivity_{ds_name}.pdf'}")

# %% — Summary
print("\n" + "=" * 80)
print("ALL FIGURES GENERATED")
print("=" * 80)
print(f"\nOutput directory: {FIG_DIR}")
for f in sorted(FIG_DIR.glob("*")):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<45s} ({size_kb:.1f} KB)")
print("\nDone.")
