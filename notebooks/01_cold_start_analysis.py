# %% [markdown]
# # Cold-Start Analysis
#
# **Key thesis claim:** LightGCN propagation helps cold-start users
# disproportionately. Users with few/no interactions in training data
# rely entirely on neighbour-propagated treatment-effect signals,
# and the GNN framework recovers meaningful policy quality for them.
#
# OBD has ~42.6% cold-start users (users with very few positive
# interactions), making this analysis critical for publication.
#
# Produces **Figures 5-6** in the paper.

# %% — Setup
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_dataset
from src.utils.metrics import RewardModel, evaluate_policy
from src.graph.lightgcn import LightGCN
from src.agent.bcq import BCQAgent
from src.baselines.policies import DQNPolicy

# Reproducibility
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# Publication-quality plot settings
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (8, 5),
})
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %%  — Load data
print("Loading OBD-all ...")
dataset = load_dataset("obd-all", root=str(ROOT))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  {dataset.n_users} users, {dataset.n_items} items, device={device}")

# %% — User activity profiling
# Count positive interactions per user in training data
train = dataset.train
pos_mask = train.rewards > 0

# Total interactions per user (all) and positive interactions
user_total_counts = np.bincount(train.user_ids, minlength=dataset.n_users)
user_pos_counts = np.bincount(train.user_ids[pos_mask], minlength=dataset.n_users)

print(f"\nUser activity distribution:")
print(f"  Total users:           {dataset.n_users}")
print(f"  Users with 0 clicks:   {(user_pos_counts == 0).sum()} "
      f"({(user_pos_counts == 0).mean()*100:.1f}%)")
print(f"  Users with 1-3 clicks: {((user_pos_counts >= 1) & (user_pos_counts <= 3)).sum()}")
print(f"  Users with 4+ clicks:  {(user_pos_counts >= 4).sum()}")

# Define cold-start bins
# Bin 0: 0 positive interactions (truly cold)
# Bin 1: 1-3 positive interactions (sparse)
# Bin 2: 4-10 positive interactions (moderate)
# Bin 3: 11+ positive interactions (warm)
def assign_bin(count):
    if count == 0:
        return 0
    elif count <= 3:
        return 1
    elif count <= 10:
        return 2
    else:
        return 3

user_bins = np.array([assign_bin(c) for c in user_pos_counts])
BIN_LABELS = ["0 clicks\n(cold-start)", "1-3 clicks\n(sparse)",
              "4-10 clicks\n(moderate)", "11+ clicks\n(warm)"]

# %% — Figure: User activity histogram
fig, ax = plt.subplots(figsize=(7, 4))
bin_counts = [int((user_bins == b).sum()) for b in range(4)]
bars = ax.bar(BIN_LABELS, bin_counts, color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"],
              edgecolor="black", linewidth=0.5)
for bar, count in zip(bars, bin_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f"{count}\n({count/dataset.n_users*100:.1f}%)",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Number of Users")
ax.set_title("User Activity Distribution in OBD Training Data")
ax.set_ylim(0, max(bin_counts) * 1.25)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "cold_start_user_distribution.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "cold_start_user_distribution.png", bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'cold_start_user_distribution.pdf'}")

# %% — Import pipeline helpers
sys.path.insert(0, str(ROOT / "experiments"))
from run_main import DEFAULT_CONFIG, train_lightgcn, build_states, train_reward_model

config = DEFAULT_CONFIG.copy()

# %% — Train LightGCN
print("\nTraining LightGCN ...")
gcn_model = train_lightgcn(dataset, config, device, SEED)

# Build augmented states
states_train = build_states(
    dataset.train.contexts, dataset.train.user_ids, gcn_model, device)
states_test = build_states(
    dataset.test.contexts, dataset.test.user_ids, gcn_model, device)

print(f"  State dim: {states_train.shape[1]} "
      f"({dataset.context_dim} context + {config['gcn_embed_dim']} GNN)")

# %% — Train reward model (needed for DR evaluation)
print("\nTraining Reward Model ...")
rm_gnn = RewardModel(
    states_train.shape[1], dataset.n_items, device=str(device))
rm_gnn.fit(states_train, train.actions,
           train.rewards.astype(np.float32), n_epochs=config["rm_epochs"])

rm_ctx = RewardModel(
    dataset.context_dim, dataset.n_items, device=str(device))
rm_ctx.fit(train.contexts, train.actions,
           train.rewards.astype(np.float32), n_epochs=config["rm_epochs"])

# %% — Train GNN-Bandit (full model)
print("\nTraining GNN-Bandit ...")
agent_gnn = BCQAgent(
    state_dim=states_train.shape[1],
    n_actions=dataset.n_items,
    hidden=config["bcq_hidden"],
    threshold_ratio=config["bcq_threshold_ratio"],
    min_actions=config["bcq_min_actions"],
    device=str(device),
)
agent_gnn.train(states_train, train.actions,
                train.rewards.astype(np.float32),
                n_epochs_bc=config["bcq_epochs_bc"],
                n_epochs_q=config["bcq_epochs_q"])

# %% — Train No-Graph BCQ (ablation — context only, no GNN embeddings)
print("\nTraining Context-Only BCQ (no GNN) ...")
agent_ctx = BCQAgent(
    state_dim=dataset.context_dim,
    n_actions=dataset.n_items,
    hidden=config["bcq_hidden"],
    threshold_ratio=config["bcq_threshold_ratio"],
    min_actions=config["bcq_min_actions"],
    device=str(device),
)
agent_ctx.train(train.contexts, train.actions,
                train.rewards.astype(np.float32),
                n_epochs_bc=config["bcq_epochs_bc"],
                n_epochs_q=config["bcq_epochs_q"])

# %% — Train DQN baseline (no constraint, no graph)
print("\nTraining DQN baseline ...")
dqn = DQNPolicy(state_dim=dataset.context_dim, n_actions=dataset.n_items,
                device=str(device))
dqn.train(train.contexts, train.actions,
          train.rewards.astype(np.float32), n_epochs=config["dqn_epochs"])

# %% — Per-bin evaluation
test = dataset.test
test_user_bins = np.array([assign_bin(user_pos_counts[uid]) for uid in test.user_ids])

# Predictions
probs_gnn = agent_gnn.action_probabilities(states_test)
probs_ctx = agent_ctx.action_probabilities(test.contexts)
probs_dqn = dqn.action_probabilities(test.contexts)

rm_preds_gnn = rm_gnn.predict(states_test)
rm_preds_ctx = rm_ctx.predict(test.contexts)

methods = {
    "GNN-Bandit (Ours)": (probs_gnn, rm_preds_gnn),
    "No-Graph BCQ":      (probs_ctx, rm_preds_ctx),
    "DQN (no constraint)": (probs_dqn, rm_preds_ctx),
}

print("\n" + "=" * 70)
print("PER-BIN COLD-START EVALUATION (DR Estimator)")
print("=" * 70)

results_table = []

for method_name, (probs, rm_preds) in methods.items():
    for bin_idx in range(4):
        mask = test_user_bins == bin_idx
        n_samples = mask.sum()
        if n_samples < 10:
            continue

        ope = evaluate_policy(
            probs[mask],
            test.rewards[mask].astype(np.float32),
            test.propensities[mask],
            test.actions[mask],
            dataset.n_items,
            rm_preds[mask],
            label="",
        )
        dr = ope.get("DR")
        results_table.append({
            "Method": method_name,
            "Bin": bin_idx,
            "Bin_Label": BIN_LABELS[bin_idx].replace("\n", " "),
            "N": int(n_samples),
            "DR_value": dr.value if dr else np.nan,
            "DR_ci_lower": dr.ci_lower if dr else np.nan,
            "DR_ci_upper": dr.ci_upper if dr else np.nan,
        })

df = pd.DataFrame(results_table)
print("\n" + df.to_string(index=False))

# %% — Figure: Cold-start performance comparison (main figure)
fig, ax = plt.subplots(figsize=(9, 5.5))

# Colors for methods
COLORS = {
    "GNN-Bandit (Ours)":   "#2ecc71",
    "No-Graph BCQ":        "#3498db",
    "DQN (no constraint)": "#e74c3c",
}

n_bins = 4
n_methods = len(methods)
bar_width = 0.22
x = np.arange(n_bins)

for i, method_name in enumerate(methods.keys()):
    mdf = df[df["Method"] == method_name]
    vals = []
    errs_low = []
    errs_high = []
    for b in range(n_bins):
        row = mdf[mdf["Bin"] == b]
        if len(row) > 0:
            v = row["DR_value"].values[0]
            lo = row["DR_ci_lower"].values[0]
            hi = row["DR_ci_upper"].values[0]
            vals.append(v)
            errs_low.append(v - lo)
            errs_high.append(hi - v)
        else:
            vals.append(0)
            errs_low.append(0)
            errs_high.append(0)

    offset = (i - n_methods / 2 + 0.5) * bar_width
    bars = ax.bar(
        x + offset, vals, bar_width,
        yerr=[errs_low, errs_high],
        capsize=3, label=method_name,
        color=COLORS[method_name], edgecolor="black", linewidth=0.5,
        error_kw={"linewidth": 1},
    )

ax.set_xlabel("User Activity Bin (training positive interactions)")
ax.set_ylabel("DR Policy Value Estimate")
ax.set_title("Cold-Start Analysis: GNN Propagation Helps Low-Activity Users")
ax.set_xticks(x)
ax.set_xticklabels(BIN_LABELS, fontsize=10)
ax.legend(loc="upper left", framealpha=0.9)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "cold_start_dr_by_bin.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "cold_start_dr_by_bin.png", bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'cold_start_dr_by_bin.pdf'}")

# %% — Figure: Relative improvement of GNN over No-Graph
fig, ax = plt.subplots(figsize=(7, 4.5))

gnn_vals = df[df["Method"] == "GNN-Bandit (Ours)"].sort_values("Bin")["DR_value"].values
ctx_vals = df[df["Method"] == "No-Graph BCQ"].sort_values("Bin")["DR_value"].values

# Relative improvement: (GNN - NoGraph) / |NoGraph|
# Handle near-zero baselines carefully
rel_improvement = np.where(
    np.abs(ctx_vals) > 1e-8,
    (gnn_vals - ctx_vals) / np.abs(ctx_vals) * 100,
    0.0,
)

colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in rel_improvement]
bars = ax.bar(BIN_LABELS, rel_improvement, color=colors,
              edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, rel_improvement):
    y_pos = bar.get_height() + (2 if val >= 0 else -4)
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
            f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top",
            fontsize=11, fontweight="bold")

ax.set_ylabel("Relative DR Improvement (%)")
ax.set_title("GNN Benefit Over Context-Only BCQ by User Activity")
ax.axhline(y=0, color="black", linewidth=0.8)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "cold_start_relative_improvement.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "cold_start_relative_improvement.png", bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'cold_start_relative_improvement.pdf'}")

# %% — Summary statistics
print("\n" + "=" * 70)
print("COLD-START SUMMARY")
print("=" * 70)

cold_mask = test_user_bins == 0
warm_mask = test_user_bins == 3

for method_name, (probs, rm_preds) in methods.items():
    cold_ope = evaluate_policy(
        probs[cold_mask], test.rewards[cold_mask].astype(np.float32),
        test.propensities[cold_mask], test.actions[cold_mask],
        dataset.n_items, rm_preds[cold_mask], label="",
    )
    warm_ope = evaluate_policy(
        probs[warm_mask], test.rewards[warm_mask].astype(np.float32),
        test.propensities[warm_mask], test.actions[warm_mask],
        dataset.n_items, rm_preds[warm_mask], label="",
    )
    dr_cold = cold_ope.get("DR")
    dr_warm = warm_ope.get("DR")
    print(f"\n{method_name}:")
    if dr_cold:
        print(f"  Cold-start (0 clicks): DR = {dr_cold.value:.6f} "
              f"[{dr_cold.ci_lower:.6f}, {dr_cold.ci_upper:.6f}]")
    if dr_warm:
        print(f"  Warm (11+ clicks):     DR = {dr_warm.value:.6f} "
              f"[{dr_warm.ci_lower:.6f}, {dr_warm.ci_upper:.6f}]")

# %% — Save raw results
results_df = df.copy()
results_df.to_csv(FIG_DIR / "cold_start_results.csv", index=False)
print(f"\nRaw results saved to {FIG_DIR / 'cold_start_results.csv'}")
print("\nDone.")
