"""
Publication-Quality Plot Generator for GNN-Bandit Q1 Journal Submission
Generates high-resolution vector PDF and raster PNG figures:
  1. Fig 1: Cross-Dataset Policy Return Comparison (OBD-All, Men, Women, KuaiRec, KuaiRand)
  2. Fig 2: Component-wise Ablation Breakdown (% Degradation without GNN / BCQ)
  3. Fig 3: Cold-Start Generalization Lift (Degree-0 User Regimes)
  4. Fig 4: Architectural Benchmark: LightGCN vs. Temporal Graph Network (TGN)
  5. Fig 5: Lagrangian Multiplier Pareto Frontier (Lambda-CFR vs Policy Return & CV%)
  6. Fig 6: Risk-Averse CVaR Tail Safety-Performance Trade-off Curve
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.autolayout": True,
})

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Figure 1: Cross-Dataset Benchmark (GNN-Bandit vs Top Baselines)
# -----------------------------------------------------------------------------
def plot_fig1_cross_dataset():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    datasets = ["OBD-All", "OBD-Men", "OBD-Women", "KuaiRec", "KuaiRand"]
    gnn_bandit = [0.008404, 0.010317, 0.010086, 0.112655, 0.453466]
    cql = [0.006706, 0.008829, 0.008599, 0.107662, 0.478223]
    greedy_gnn = [0.005956, 0.008856, 0.008053, 0.112085, 0.450351]
    bts_logging = [0.004050, 0.006044, 0.005662, 0.070021, 0.399527]
    
    # Normalize by BTS Logging Policy to show Relative Value Ratio
    ratio_gnn = [g / b for g, b in zip(gnn_bandit, bts_logging)]
    ratio_cql = [c / b for c, b in zip(cql, bts_logging)]
    ratio_greedy = [gr / b for gr, b in zip(greedy_gnn, bts_logging)]
    
    x = np.arange(len(datasets))
    width = 0.26
    
    rects1 = ax.bar(x - width, ratio_gnn, width, label="GNN-Bandit (Ours)", color="#1f77b4", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x, ratio_cql, width, label="CQL (Offline RL)", color="#ff7f0e", edgecolor="black", linewidth=0.8)
    rects3 = ax.bar(x + width, ratio_greedy, width, label="Greedy-GNN (Heuristic)", color="#2ca02c", edgecolor="black", linewidth=0.8)
    
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Logging Baseline (BTS = 1.0×)")
    
    ax.set_ylabel("Normalized Policy Value (× Logging BTS)")
    ax.set_title("Cross-Dataset Empirical Superiority across Diverse Domains")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray", loc="upper right")
    ax.set_ylim(0.8, 2.3)
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}×",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight="bold", color="#1f77b4")
                    
    plt.savefig(FIG_DIR / "fig1_cross_dataset_benchmark.pdf")
    plt.savefig(FIG_DIR / "fig1_cross_dataset_benchmark.png", dpi=300)
    plt.close()
    print("[OK] Saved Figure 1: Cross-Dataset Benchmark")

# -----------------------------------------------------------------------------
# Figure 2: Component-wise Ablation Breakdown
# -----------------------------------------------------------------------------
def plot_fig2_ablation():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    
    datasets = ["OBD-All", "OBD-Men", "OBD-Women", "Criteo"]
    deg_gnn = [41.71, 32.11, 31.31, 6.15]
    deg_bcq = [51.11, 41.25, 46.78, 6.04]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, deg_gnn, width, label="Drop without Graph (No-Graph)", color="#d62728", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, deg_bcq, width, label="Drop without Constraint (No-BCQ)", color="#9467bd", edgecolor="black", linewidth=0.8)
    
    ax.set_ylabel("Relative Policy Performance Collapse (%)")
    ax.set_title("Component-Wise Ablation: Synergistic Collapse without Graph or BCQ")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray")
    ax.set_ylim(0, 60)
    
    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f"-{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight="bold")
                    
    plt.savefig(FIG_DIR / "fig2_ablation_breakdown.pdf")
    plt.savefig(FIG_DIR / "fig2_ablation_breakdown.png", dpi=300)
    plt.close()
    print("[OK] Saved Figure 2: Ablation Breakdown")

# -----------------------------------------------------------------------------
# Figure 3: Cold-Start Generalization
# -----------------------------------------------------------------------------
def plot_fig3_cold_start():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    
    models = ["GNN-Bandit\n(Ours)", "Greedy-GNN", "CQL", "MF-Bandit", "NeuralUCB", "BTS\n(Logging)"]
    scores = [0.012080, 0.011096, 0.010615, 0.008456, 0.006992, 0.007721]
    stds =   [0.000686, 0.000051, 0.000045, 0.000123, 0.000052, 0.000157]
    
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#7f7f7f", "#bcbd22", "#17becf"]
    
    bars = ax.bar(models, scores, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.8)
    
    # Add lift annotation over MF-Bandit
    ax.annotate("+42.86% Lift\nover MF", xy=(0, 0.012080), xytext=(0, 0.0135),
                ha='center', va='bottom', fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
                
    ax.set_ylabel("Doubly Robust Policy Return (DR)")
    ax.set_title("Cold-Start Performance on 205 Zero-Degree Users (OBD-Men)")
    ax.set_ylim(0, 0.016)
    
    plt.savefig(FIG_DIR / "fig3_cold_start_comparison.pdf")
    plt.savefig(FIG_DIR / "fig3_cold_start_comparison.png", dpi=300)
    plt.close()
    print("[OK] Saved Figure 3: Cold-Start Comparison")

# -----------------------------------------------------------------------------
# Figure 4: LightGCN vs. Temporal Graph Network (TGN)
# -----------------------------------------------------------------------------
def plot_fig4_tgn():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    
    datasets = ["OBD-All", "OBD-Men", "OBD-Women", "Criteo"]
    lightgcn = [0.008404, 0.010317, 0.010086, 0.002726]
    tgn =      [0.007354, 0.010181, 0.009098, 0.002712]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, lightgcn, width, label="LightGCN (Parameter-Free Spectral)", color="#1f77b4", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, tgn, width, label="TGN (Continuous Temporal Memory)", color="#e377c2", edgecolor="black", linewidth=0.8)
    
    ax.set_ylabel("Doubly Robust Policy Value (DR)")
    ax.set_title("Architectural Evaluation: Spectral Laplacians vs. Recurrent Memory")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray")
    
    plt.savefig(FIG_DIR / "fig4_tgn_vs_lightgcn.pdf")
    plt.savefig(FIG_DIR / "fig4_tgn_vs_lightgcn.png", dpi=300)
    plt.close()
    print("[OK] Saved Figure 4: TGN vs LightGCN")

# -----------------------------------------------------------------------------
# Figure 5: Lagrangian Multiplier Pareto Frontier (Lambda-CFR)
# -----------------------------------------------------------------------------
def plot_fig5_lagrangian():
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    
    lambdas = [0.05, 0.10, 0.20]
    obd_all_dr = [0.008501, 0.008404, 0.008120]
    obd_all_cv = [2.07, 1.18, 2.59]
    
    color = '#1f77b4'
    ax1.set_xlabel(r"Lagrangian Multiplier $\lambda_{\mathrm{CFR}}$ (Counterfactual Variance Bound)", fontweight="bold")
    ax1.set_ylabel("Expected Policy Value (DR)", color=color, fontweight="bold")
    line1 = ax1.plot(lambdas, obd_all_dr, marker='o', color=color, linewidth=2.2, label="Policy Return (DR)")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel("Coefficient of Variation (CV %)", color=color, fontweight="bold")
    line2 = ax2.plot(lambdas, obd_all_cv, marker='s', linestyle='--', color=color, linewidth=2.0, label="Variance Risk (CV%)")
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.set_title(r"Lagrangian Pareto Frontier: Policy Value vs. Counterfactual Variance")
    plt.savefig(FIG_DIR / "fig5_lagrangian_pareto_frontier.pdf")
    plt.savefig(FIG_DIR / "fig5_lagrangian_pareto_frontier.png", dpi=300)
    plt.close()
    print("[OK] Saved Figure 5: Lagrangian Pareto Frontier")

# -----------------------------------------------------------------------------
# Figure 6: CVaR Tail Risk Safety Curve
# -----------------------------------------------------------------------------
def plot_fig6_cvar():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    
    alpha = [0.05, 0.10, 0.25, 0.50, 1.00]
    obd_all = [0.007640, 0.008429, 0.008895, 0.009044, 0.009993]
    obd_women = [0.009632, 0.009862, 0.009603, 0.010300, 0.010563]
    
    ax.plot(alpha, obd_all, marker='o', linewidth=2.2, label="OBD-All", color="#1f77b4")
    ax.plot(alpha, obd_women, marker='s', linewidth=2.2, label="OBD-Women", color="#ff7f0e")
    
    ax.axvline(0.10, color="gray", linestyle=":", linewidth=1.5, label=r"Operational Setting ($\alpha=0.10$)")
    
    ax.set_xlabel(r"CVaR Risk Level $\alpha$ (Tail Fraction Optimized)", fontweight="bold")
    ax.set_ylabel("Doubly Robust Policy Return (DR)")
    ax.set_title(r"Risk-Averse Safety Trade-off Curve Across Quantile Cutoffs $\alpha$")
    ax.legend(frameon=True, facecolor="white", edgecolor="lightgray", loc="lower right")
    
    plt.savefig(FIG_DIR / "fig6_cvar_safety_curve.pdf")
    plt.savefig(FIG_DIR / "fig6_cvar_safety_curve.png", dpi=300)
    plt.close()
    print("[OK] Saved Figure 6: CVaR Safety Curve")

if __name__ == "__main__":
    print(f"Generating publication figures in {FIG_DIR.resolve()} ...")
    plot_fig1_cross_dataset()
    plot_fig2_ablation()
    plot_fig3_cold_start()
    plot_fig4_tgn()
    plot_fig5_lagrangian()
    plot_fig6_cvar()
    print("\n[SUCCESS] All 6 publication figures generated successfully (PDF + PNG)!")
