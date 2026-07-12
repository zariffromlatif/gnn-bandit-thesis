"""
Criteo Uplift v2.1 Preprocessing — Tier 3 Evaluation
=====================================================
Transforms the Criteo Uplift Prediction dataset into formats compatible
with the Graph-Enhanced Causal RL framework.

Dataset: ~25.3M rows, 12 continuous features (f0-f11), binary treatment,
         two outcomes (conversion, visit), and exposure indicator.

Output (in data/processed_criteo/):
  - context_train.npz, context_val.npz, context_test.npz
        BCQ states = [f0..f11, treatment] (13-dim per row)
  - graph_user_knn.npz
        k-NN user similarity graph from f0-f11 features
        (built on a SAMPLED subset for computational feasibility,
         then extended to full dataset via nearest-centroid mapping)
  - cluster_assignments.npy
        User cluster IDs (each cluster = a "user node" in the graph)
  - uplift_estimates.csv
        Per-cluster uplift (treatment effect on conversion & visit)
  - stats.json
        Dataset statistics for paper tables

Design choices:
  - 25M individual user rows → graph with 25M nodes is infeasible.
    Solution: cluster users into ~5K-10K segments via MiniBatchKMeans,
    build k-NN graph over cluster centroids. Each user inherits its
    cluster's graph embedding. This mirrors the OBD's segment-level
    graph while having a legitimate clustering justification.
  - Random 80/10/10 split (no temporal dimension in Criteo).
  - Propensity is known (≈0.846 treatment rate in v2.1, or use
    the exposure-based subset for cleaner causal estimation).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = Path(__file__).parent / "data" / "criteo-uplift" / "criteo-uplift-v2.1.csv"
OUT       = Path(__file__).parent / "data" / "processed_criteo"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [f"f{i}" for i in range(12)]
N_CLUSTERS   = 5000     # user segments for the graph
KNN_K        = 15       # neighbours in cluster-level graph
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
CHUNKSIZE    = 1_000_000
RANDOM_SEED  = 42

# ============================================================================
# Helpers
# ============================================================================
def build_knn_adj(features: np.ndarray, k: int) -> csr_matrix:
    """Build a symmetric k-NN adjacency from a feature matrix."""
    sim = cosine_similarity(features)
    np.fill_diagonal(sim, 0)
    n = sim.shape[0]
    rows, cols, vals = [], [], []
    for i in range(n):
        topk = np.argsort(sim[i])[-k:]
        for j in topk:
            if sim[i, j] > 0:
                rows.append(i); cols.append(j); vals.append(sim[i, j])
                rows.append(j); cols.append(i); vals.append(sim[i, j])
    adj = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    adj.eliminate_zeros()
    return adj


# ============================================================================
# STEP 1: Load data
# ============================================================================
print("=" * 70)
print("CRITEO UPLIFT v2.1 PREPROCESSING")
print("=" * 70)

print("\nSTEP 1: Loading data ...")
# Columns: f0-f11, treatment, conversion, visit, exposure
df = pd.read_csv(DATA_PATH)
n_total = len(df)
print(f"  Total rows: {n_total:,}")
print(f"  Columns: {list(df.columns)}")
print(f"  Treatment rate: {df['treatment'].mean():.4f}")
print(f"  Conversion rate: {df['conversion'].mean():.6f}")
print(f"  Visit rate: {df['visit'].mean():.4f}")
if "exposure" in df.columns:
    print(f"  Exposure rate: {df['exposure'].mean():.4f}")

# ============================================================================
# STEP 2: Feature scaling
# ============================================================================
print("\nSTEP 2: Scaling features ...")
scaler = StandardScaler()
df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS]).astype(np.float32)

# Save scaler params
scaler_params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
}

# ============================================================================
# STEP 3: Cluster users for graph construction
# ============================================================================
print(f"\nSTEP 3: Clustering users into {N_CLUSTERS} segments ...")
features = df[FEATURE_COLS].values

kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_SEED,
    batch_size=10_000,
    n_init=3,
    verbose=1,
)
df["cluster_id"] = kmeans.fit_predict(features)
centroids = kmeans.cluster_centers_  # (N_CLUSTERS, 12)

np.save(OUT / "cluster_centroids.npy", centroids)
np.save(OUT / "cluster_assignments.npy", df["cluster_id"].values.astype(np.int32))

# Cluster size distribution
cluster_sizes = df["cluster_id"].value_counts()
print(f"  Cluster size: mean={cluster_sizes.mean():.0f}, "
      f"min={cluster_sizes.min()}, max={cluster_sizes.max()}")

# ============================================================================
# STEP 4: Build cluster-level k-NN graph
# ============================================================================
print(f"\nSTEP 4: Building cluster-level k-NN graph (k={KNN_K}) ...")
A_uu = build_knn_adj(centroids, k=KNN_K)
save_npz(OUT / "graph_user_knn.npz", A_uu)
print(f"  Graph: {A_uu.shape}, nnz={A_uu.nnz}, "
      f"density={A_uu.nnz / (N_CLUSTERS ** 2):.6f}")

# ============================================================================
# STEP 5: Per-cluster uplift estimates
# ============================================================================
print("\nSTEP 5: Computing per-cluster uplift ...")

cluster_stats = df.groupby(["cluster_id", "treatment"]).agg(
    n=("conversion", "size"),
    conversions=("conversion", "sum"),
    visits=("visit", "sum"),
).reset_index()

# Pivot to get treatment=0 and treatment=1 side by side
treated   = cluster_stats[cluster_stats["treatment"] == 1].set_index("cluster_id")
control   = cluster_stats[cluster_stats["treatment"] == 0].set_index("cluster_id")

uplift = pd.DataFrame(index=range(N_CLUSTERS))
uplift["n_treated"]     = treated["n"].reindex(uplift.index, fill_value=0)
uplift["n_control"]     = control["n"].reindex(uplift.index, fill_value=0)
uplift["conv_treated"]  = treated["conversions"].reindex(uplift.index, fill_value=0)
uplift["conv_control"]  = control["conversions"].reindex(uplift.index, fill_value=0)
uplift["visit_treated"] = treated["visits"].reindex(uplift.index, fill_value=0)
uplift["visit_control"] = control["visits"].reindex(uplift.index, fill_value=0)

uplift["ctr_treated"]  = uplift["conv_treated"] / uplift["n_treated"].clip(lower=1)
uplift["ctr_control"]  = uplift["conv_control"] / uplift["n_control"].clip(lower=1)
uplift["uplift_conv"]  = uplift["ctr_treated"] - uplift["ctr_control"]

uplift["vtr_treated"]  = uplift["visit_treated"] / uplift["n_treated"].clip(lower=1)
uplift["vtr_control"]  = uplift["visit_control"] / uplift["n_control"].clip(lower=1)
uplift["uplift_visit"] = uplift["vtr_treated"] - uplift["vtr_control"]

uplift.to_csv(OUT / "uplift_estimates.csv", index_label="cluster_id")

print(f"  Clusters with positive conv uplift: "
      f"{(uplift['uplift_conv'] > 0).sum()} / {N_CLUSTERS}")
print(f"  Clusters with negative conv uplift (Sleeping Dogs): "
      f"{(uplift['uplift_conv'] < 0).sum()} / {N_CLUSTERS}")
print(f"  Mean conversion uplift: {uplift['uplift_conv'].mean():.6f}")
print(f"  Mean visit uplift:      {uplift['uplift_visit'].mean():.6f}")

del cluster_stats, treated, control, uplift

# ============================================================================
# STEP 6: Random train/val/test split + context vectors
# ============================================================================
print(f"\nSTEP 6: Creating random splits ({TRAIN_RATIO}/{VAL_RATIO}/"
      f"{1-TRAIN_RATIO-VAL_RATIO}) ...")

rng = np.random.RandomState(RANDOM_SEED)
perm = rng.permutation(n_total)
t1 = int(n_total * TRAIN_RATIO)
t2 = int(n_total * (TRAIN_RATIO + VAL_RATIO))

split_indices = {
    "train": perm[:t1],
    "val":   perm[t1:t2],
    "test":  perm[t2:],
}

# Context = [f0..f11] (12-dim; treatment is the ACTION, not part of state)
context_cols = FEATURE_COLS

split_stats = {}
for name, idx in split_indices.items():
    sub = df.iloc[idx]
    ctx = sub[context_cols].values.astype(np.float32)

    np.savez_compressed(
        OUT / f"context_{name}.npz",
        contexts=ctx,
        cluster_id=sub["cluster_id"].values.astype(np.int32),
        treatment=sub["treatment"].values.astype(np.int8),
        conversion=sub["conversion"].values.astype(np.int8),
        visit=sub["visit"].values.astype(np.int8),
        exposure=sub["exposure"].values.astype(np.int8) if "exposure" in sub.columns else np.zeros(len(sub), dtype=np.int8),
    )

    n_t = sub["treatment"].sum()
    n_c = len(sub) - n_t
    conv_rate = sub["conversion"].mean()

    split_stats[name] = {
        "rows":       len(sub),
        "treated":    int(n_t),
        "control":    int(n_c),
        "conv_rate":  float(f"{conv_rate:.6f}"),
        "visit_rate": float(f"{sub['visit'].mean():.6f}"),
    }
    print(f"  {name:5s}: {len(sub):>10,} rows | "
          f"treated={n_t:,} control={n_c:,} | "
          f"conv={conv_rate:.4f}")

# ============================================================================
# STEP 7: Save statistics
# ============================================================================
stats = {
    "dataset":        "criteo-uplift-v2.1",
    "n_total":        n_total,
    "n_features":     len(FEATURE_COLS),
    "n_clusters":     N_CLUSTERS,
    "knn_k":          KNN_K,
    "graph_nnz":      int(A_uu.nnz),
    "treatment_rate": float(f"{df['treatment'].mean():.4f}"),
    "conversion_rate": float(f"{df['conversion'].mean():.6f}"),
    "context_dim":    len(context_cols),
    "splits":         split_stats,
    "scaler":         scaler_params,
}
with open(OUT / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'=' * 70}")
print("CRITEO PREPROCESSING COMPLETE")
print(f"Output: {OUT.resolve()}")
for fp in sorted(OUT.iterdir()):
    kb = fp.stat().st_size / 1024
    unit = "KB"
    if kb > 1024:
        kb /= 1024
        unit = "MB"
    if kb > 1024:
        kb /= 1024
        unit = "GB"
    print(f"  {fp.name:<40s} {kb:>8.1f} {unit}")
print("=" * 70)
