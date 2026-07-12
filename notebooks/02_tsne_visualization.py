# %% [markdown]
# # t-SNE Embedding Visualization
#
# **Key thesis claim:** LightGCN graph convolution propagates treatment-effect
# signals through the user-item graph.  After L layers, embeddings encode
# *causal response patterns* of the local neighbourhood.
#
# This notebook visualises:
#   1. Initial embeddings E^{(0)} (before propagation) -- random/uninformative
#   2. Propagated embeddings E^{(L)} (after L=3 layers) -- clustered by uplift
#   3. Concatenated state vectors (context + GNN) used by BCQ
#
# Colour coding follows the 4 uplift quadrants:
#   - Persuadable  (positive uplift, low baseline)  -> GREEN  = intervene
#   - Sure Thing   (positive uplift, high baseline)  -> BLUE   = save budget
#   - Lost Cause   (negative uplift, low baseline)   -> GREY   = don't bother
#   - Sleeping Dog  (negative uplift, high baseline)  -> RED    = DO NOT TOUCH
#
# Produces **Figures 3-4** in the paper.

# %% — Setup
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_dataset
from src.graph.lightgcn import LightGCN

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (8, 6),
})
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %% — Load data
print("Loading OBD-all ...")
dataset = load_dataset("obd-all", root=str(ROOT))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  {dataset.n_users} users, {dataset.n_items} items")

# %% — Load uplift estimates for colouring
uplift_path = dataset.uplift_df_path
if uplift_path and uplift_path.exists():
    uplift_df = pd.read_csv(uplift_path)
    print(f"  Loaded {len(uplift_df)} uplift estimates")
else:
    uplift_df = None
    print("  WARNING: No uplift data found. Colours will be by user activity.")

# Build per-user mean uplift and baseline response
uplift_table = np.zeros((dataset.n_users, dataset.n_items), dtype=np.float32)
if uplift_df is not None:
    for _, row in uplift_df.iterrows():
        uid = int(row["user_id"])
        iid = int(row["item_id"])
        if uid < dataset.n_users and iid < dataset.n_items:
            uplift_table[uid, iid] = row["uplift"]

user_mean_uplift = uplift_table.mean(axis=1)  # (n_users,)

# Baseline response: mean reward when NOT treated (approximate with overall mean)
train = dataset.train
user_reward_sums = np.zeros(dataset.n_users, dtype=np.float32)
user_reward_counts = np.zeros(dataset.n_users, dtype=np.float32)
for uid, r in zip(train.user_ids, train.rewards):
    user_reward_sums[uid] += r
    user_reward_counts[uid] += 1
user_baseline = np.where(
    user_reward_counts > 0,
    user_reward_sums / user_reward_counts,
    0.0,
)

# %% — Assign uplift quadrants
# Threshold: uplift > 0 and baseline > median
baseline_median = np.median(user_baseline[user_baseline > 0]) if (user_baseline > 0).any() else 0.01

def assign_quadrant(uplift, baseline, uplift_thresh=0.0, base_thresh=None):
    if base_thresh is None:
        base_thresh = baseline_median
    if uplift > uplift_thresh and baseline <= base_thresh:
        return 0  # Persuadable
    elif uplift > uplift_thresh and baseline > base_thresh:
        return 1  # Sure Thing
    elif uplift <= uplift_thresh and baseline <= base_thresh:
        return 2  # Lost Cause
    else:
        return 3  # Sleeping Dog

user_quadrants = np.array([
    assign_quadrant(user_mean_uplift[u], user_baseline[u])
    for u in range(dataset.n_users)
])

QUADRANT_NAMES  = ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]
QUADRANT_COLORS = ["#2ecc71",     "#3498db",    "#95a5a6",    "#e74c3c"]

quad_counts = [int((user_quadrants == q).sum()) for q in range(4)]
print("\nUplift quadrant distribution:")
for q in range(4):
    print(f"  {QUADRANT_NAMES[q]:<15s}: {quad_counts[q]:>4d} users "
          f"({quad_counts[q]/dataset.n_users*100:.1f}%)")

# %% — Train LightGCN
sys.path.insert(0, str(ROOT / "experiments"))
from run_main import DEFAULT_CONFIG, train_lightgcn
config = DEFAULT_CONFIG.copy()

print("\nTraining LightGCN ...")
gcn_model = train_lightgcn(dataset, config, device, SEED)

# %% — Extract embeddings at different stages
gcn_model.eval()
with torch.no_grad():
    # Layer 0: initial embeddings (before any propagation)
    E0 = gcn_model.embedding.weight.cpu().numpy()  # (n_nodes, K)
    E0_users = E0[:dataset.n_users]                  # (n_users, K)
    E0_items = E0[dataset.n_users:]                   # (n_items, K)

    # Final embeddings: after L layers of propagation + mean pooling
    all_emb = gcn_model.forward().cpu().numpy()       # (n_nodes, K)
    EL_users = all_emb[:dataset.n_users]               # (n_users, K)
    EL_items = all_emb[dataset.n_users:]                # (n_items, K)

print(f"\nEmbedding shapes:")
print(f"  Initial E0 users: {E0_users.shape}")
print(f"  Final   EL users: {EL_users.shape}")

# %% — t-SNE on user embeddings
print("\nRunning t-SNE on initial embeddings ...")
tsne_0 = TSNE(n_components=2, perplexity=min(30, dataset.n_users - 1),
              random_state=SEED, n_iter=1000)
emb_2d_init = tsne_0.fit_transform(E0_users)

print("Running t-SNE on propagated embeddings ...")
tsne_L = TSNE(n_components=2, perplexity=min(30, dataset.n_users - 1),
              random_state=SEED, n_iter=1000)
emb_2d_prop = tsne_L.fit_transform(EL_users)

# %% — Figure: Side-by-side t-SNE (MAIN FIGURE)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for q in range(4):
    mask = user_quadrants == q
    ax1.scatter(emb_2d_init[mask, 0], emb_2d_init[mask, 1],
                c=QUADRANT_COLORS[q], label=QUADRANT_NAMES[q],
                s=25, alpha=0.7, edgecolors="white", linewidths=0.3)
    ax2.scatter(emb_2d_prop[mask, 0], emb_2d_prop[mask, 1],
                c=QUADRANT_COLORS[q], label=QUADRANT_NAMES[q],
                s=25, alpha=0.7, edgecolors="white", linewidths=0.3)

ax1.set_title(r"(a) Initial Embeddings $E^{(0)}$" + "\n(Before Graph Propagation)")
ax2.set_title(r"(b) Propagated Embeddings $\bar{E}$" + f"\n(After {config['gcn_n_layers']} GCN Layers)")

for ax in (ax1, ax2):
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(loc="upper right", framealpha=0.9, markerscale=1.5)
    sns.despine(ax=ax)

plt.suptitle("LightGCN Embedding Space: Treatment-Effect Signal Propagation",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "tsne_before_after_gcn.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "tsne_before_after_gcn.png", bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'tsne_before_after_gcn.pdf'}")

# %% — Figure: Propagated embeddings only (detailed, larger)
fig, ax = plt.subplots(figsize=(8, 7))

for q in range(4):
    mask = user_quadrants == q
    ax.scatter(emb_2d_prop[mask, 0], emb_2d_prop[mask, 1],
               c=QUADRANT_COLORS[q], label=f"{QUADRANT_NAMES[q]} (n={mask.sum()})",
               s=40, alpha=0.75, edgecolors="white", linewidths=0.4)

ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.set_title("User Embeddings After Graph Propagation\n"
             "Coloured by Uplift Quadrant")
ax.legend(loc="upper right", framealpha=0.9, markerscale=1.5,
          title="Uplift Quadrant", title_fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "tsne_propagated_detailed.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "tsne_propagated_detailed.png", bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'tsne_propagated_detailed.pdf'}")

# %% — Figure: Item embeddings with user-item relationships
print("\nRunning t-SNE on all node embeddings (users + items) ...")
all_node_emb = np.vstack([EL_users, EL_items])  # (n_nodes, K)
node_types = np.array(
    ["user"] * dataset.n_users + ["item"] * dataset.n_items)

tsne_all = TSNE(n_components=2, perplexity=min(30, len(all_node_emb) - 1),
                random_state=SEED, n_iter=1000)
all_2d = tsne_all.fit_transform(all_node_emb)

fig, ax = plt.subplots(figsize=(9, 7))

# Plot items first (background)
item_mask = node_types == "item"
ax.scatter(all_2d[item_mask, 0], all_2d[item_mask, 1],
           c="#f39c12", marker="s", s=50, alpha=0.8,
           edgecolors="black", linewidths=0.5, label=f"Items (n={dataset.n_items})",
           zorder=2)

# Plot users coloured by quadrant
for q in range(4):
    user_q_mask = np.zeros(len(all_node_emb), dtype=bool)
    user_indices = np.where(user_quadrants == q)[0]  # indices in user space
    user_q_mask[user_indices] = True  # indices in all-node space (users are first)
    if user_q_mask.any():
        ax.scatter(all_2d[user_q_mask, 0], all_2d[user_q_mask, 1],
                   c=QUADRANT_COLORS[q], marker="o", s=25, alpha=0.7,
                   edgecolors="white", linewidths=0.3,
                   label=f"Users: {QUADRANT_NAMES[q]}", zorder=3)

ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.set_title("Joint User-Item Embedding Space (After Graph Propagation)")
ax.legend(loc="upper right", framealpha=0.9, markerscale=1.5, fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "tsne_user_item_joint.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "tsne_user_item_joint.png", bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'tsne_user_item_joint.pdf'}")

# %% — Embedding quality: inter-/intra-quadrant distance
from sklearn.metrics import silhouette_score

sil_init = silhouette_score(E0_users, user_quadrants,
                            metric="cosine", sample_size=min(500, dataset.n_users),
                            random_state=SEED)
sil_prop = silhouette_score(EL_users, user_quadrants,
                            metric="cosine", sample_size=min(500, dataset.n_users),
                            random_state=SEED)

print(f"\nSilhouette Score (cosine, by uplift quadrant):")
print(f"  Initial embeddings E(0):    {sil_init:.4f}")
print(f"  Propagated embeddings E(L): {sil_prop:.4f}")
print(f"  Improvement:                {sil_prop - sil_init:+.4f}")

# %% — Figure: Silhouette comparison bar
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(
    [r"Initial $E^{(0)}$", r"Propagated $\bar{E}$"],
    [sil_init, sil_prop],
    color=["#95a5a6", "#2ecc71"], edgecolor="black", linewidth=0.5, width=0.5,
)
for bar, val in zip(bars, [sil_init, sil_prop]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Silhouette Score (cosine)")
ax.set_title("Embedding Cluster Quality\nBy Uplift Quadrant")
ax.set_ylim(bottom=min(0, sil_init - 0.05))
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "silhouette_comparison.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "silhouette_comparison.png", bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'silhouette_comparison.pdf'}")

# %% — Figure: Embedding norm distribution by quadrant
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

for q in range(4):
    mask = user_quadrants == q
    norms_init = np.linalg.norm(E0_users[mask], axis=1)
    norms_prop = np.linalg.norm(EL_users[mask], axis=1)
    ax1.hist(norms_init, bins=20, alpha=0.5, color=QUADRANT_COLORS[q],
             label=QUADRANT_NAMES[q], density=True)
    ax2.hist(norms_prop, bins=20, alpha=0.5, color=QUADRANT_COLORS[q],
             label=QUADRANT_NAMES[q], density=True)

ax1.set_title(r"(a) Initial $E^{(0)}$ Norms")
ax2.set_title(r"(b) Propagated $\bar{E}$ Norms")
for ax in (ax1, ax2):
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
plt.suptitle("Embedding Norm Distributions by Uplift Quadrant", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "embedding_norms_by_quadrant.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "embedding_norms_by_quadrant.png", bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'embedding_norms_by_quadrant.pdf'}")

# %% — Save numeric results
results = {
    "silhouette_initial": float(sil_init),
    "silhouette_propagated": float(sil_prop),
    "silhouette_improvement": float(sil_prop - sil_init),
    "quadrant_counts": {QUADRANT_NAMES[q]: quad_counts[q] for q in range(4)},
    "n_users": dataset.n_users,
    "n_items": dataset.n_items,
    "embed_dim": config["gcn_embed_dim"],
    "n_layers": config["gcn_n_layers"],
}
import json
with open(FIG_DIR / "tsne_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {FIG_DIR / 'tsne_results.json'}")
print("Done.")
