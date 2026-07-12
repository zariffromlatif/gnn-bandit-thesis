"""
Preprocessing pipeline for GNN-Bandit (Graph-Enhanced Causal Reinforcement Learning
for Proactive Customer Retention).

Transforms Open Bandit Dataset (OBD) into:
  1. Encoded user/item ID tables
  2. User-Intervention Bipartite Graph (edge list + adjacency)
  3. Uplift/treatment effect estimates per user-item pair
  4. Temporal train/val/test splits ready for LightGCN + BCQ

Memory strategy: the BTS file has ~12M rows and ~85 columns of float64 which
exhausts RAM when loaded naively. We read it in chunks and retain only the
columns needed for each step.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, hstack, vstack

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent / "open_bandit_dataset"
OUT  = Path(__file__).parent / "processed"
OUT.mkdir(exist_ok=True)

CAMPAIGN = "all"          # "all" | "men" | "women"
RANDOM_CSV = ROOT / "random" / CAMPAIGN / f"{CAMPAIGN}.csv"
BTS_CSV    = ROOT / "bts"    / CAMPAIGN / f"{CAMPAIGN}.csv"
ITEM_CTX   = ROOT / "random" / CAMPAIGN / "item_context.csv"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

USER_FEAT_COLS = ["user_feature_0", "user_feature_1",
                  "user_feature_2",  "user_feature_3"]

# Columns we actually need (skip the 80 affinity columns during initial passes
# to save RAM; they are included only in the final split files for BTS-random).
AFFINITY_COLS = [f"user-item_affinity_{i}" for i in range(80)]
CORE_COLS     = (["timestamp", "item_id", "position", "click",
                  "propensity_score"] + USER_FEAT_COLS)
ALL_COLS      = CORE_COLS + AFFINITY_COLS

CHUNKSIZE = 500_000   # rows per chunk for BTS file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def user_key_series(df: pd.DataFrame) -> pd.Series:
    """Combine 4 user feature hashes into a single string key."""
    return df[USER_FEAT_COLS].astype(str).agg("-".join, axis=1)


def _parse_ts(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    return df


def load_csv_light(path: Path, policy: str) -> pd.DataFrame:
    """Load only CORE_COLS (no affinity columns) for memory efficiency."""
    wanted = set(CORE_COLS)
    df = pd.read_csv(path, index_col=False, usecols=lambda c: c in wanted)
    _parse_ts(df)
    df["policy"] = policy
    df["user_key"] = user_key_series(df)
    return df


def load_csv_chunked_light(path: Path, policy: str) -> pd.DataFrame:
    """Read a large CSV in chunks, keeping only CORE_COLS."""
    wanted = set(CORE_COLS)
    chunks = []
    for chunk in pd.read_csv(path, index_col=False,
                              usecols=lambda c: c in wanted,
                              chunksize=CHUNKSIZE):
        _parse_ts(chunk)
        chunk["policy"]   = policy
        chunk["user_key"] = user_key_series(chunk)
        chunks.append(chunk)
        print(f"    ... loaded {sum(len(c) for c in chunks):>10,} rows", end="\r")
    print()
    return pd.concat(chunks, ignore_index=True)

# ---------------------------------------------------------------------------
# STEP 1: Load core data (no affinity columns yet)
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading data (core columns only) ...")

df_random = load_csv_light(RANDOM_CSV, "random")
print(f"  random   | {len(df_random):>10,} rows | click rate: {df_random['click'].mean():.4f}")

print("  bts      | reading in chunks ...")
df_bts = load_csv_chunked_light(BTS_CSV, "bts")
print(f"  bts      | {len(df_bts):>10,} rows | click rate: {df_bts['click'].mean():.4f}")

df_items = pd.read_csv(ITEM_CTX, index_col=0)

# ---------------------------------------------------------------------------
# STEP 2: Encode user IDs
# ---------------------------------------------------------------------------
print("\nSTEP 2: Encoding user IDs ...")

all_user_keys = pd.concat([df_random["user_key"], df_bts["user_key"]]).unique()
user2id = {k: i for i, k in enumerate(all_user_keys)}
n_users = len(user2id)
n_items = int(df_items["item_id"].max()) + 1  # item_id is 0-indexed

df_random["user_id"] = df_random["user_key"].map(user2id).astype(np.int32)
df_bts["user_id"]    = df_bts["user_key"].map(user2id).astype(np.int32)

print(f"  Unique users : {n_users:,}")
print(f"  Unique items : {n_items}")

with open(OUT / "user2id.pkl", "wb") as f:
    pickle.dump(user2id, f)
pd.DataFrame({"user_key": list(user2id.keys()),
              "user_id":  list(user2id.values())}).to_csv(
    OUT / "user_id_mapping.csv", index=False)

# ---------------------------------------------------------------------------
# STEP 3: Bipartite graph from clicks
# ---------------------------------------------------------------------------
print("\nSTEP 3: Building User-Intervention Bipartite Graph ...")

clicks_r = df_random[df_random["click"] == 1][["user_id", "item_id"]]
clicks_b = df_bts[df_bts["click"] == 1][["user_id", "item_id"]]
clicks   = pd.concat([clicks_r, clicks_b]).drop_duplicates()
print(f"  Positive interaction edges (deduplicated): {len(clicks):,}")

clicks.to_csv(OUT / "bipartite_edges.csv", index=False)

rows = clicks["user_id"].values.astype(np.int32)
cols = clicks["item_id"].values.astype(np.int32)
R    = csr_matrix((np.ones(len(clicks), dtype=np.float32),
                   (rows, cols)), shape=(n_users, n_items))
save_npz(OUT / "user_item_adj.npz", R)
print(f"  Adjacency shape: {R.shape}  density: {R.nnz / (n_users * n_items):.6f}")

# LightGCN symmetric block adjacency
zero_uu = csr_matrix((n_users, n_users), dtype=np.float32)
zero_ii = csr_matrix((n_items, n_items), dtype=np.float32)
A = vstack([hstack([zero_uu, R]), hstack([R.T, zero_ii])]).tocsr()
save_npz(OUT / "lightgcn_adj.npz", A)
print(f"  LightGCN block adj shape: {A.shape}")

user_degrees = np.asarray(R.sum(axis=1)).flatten()
item_degrees = np.asarray(R.sum(axis=0)).flatten()
np.save(OUT / "user_degrees.npy", user_degrees)
np.save(OUT / "item_degrees.npy", item_degrees)
print(f"  Mean user degree: {user_degrees.mean():.3f}  "
      f"Cold-start users (degree=0): {(user_degrees == 0).sum():,}")

# ---------------------------------------------------------------------------
# STEP 4: Uplift / treatment effect
# ---------------------------------------------------------------------------
print("\nSTEP 4: Computing uplift (treatment effects) ...")

def click_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["user_key", "item_id"])["click"]
              .agg(clicks="sum", impressions="count")
              .reset_index())

stats_r = click_stats(df_random)
stats_b = click_stats(df_bts)

uplift_df = stats_r.merge(stats_b, on=["user_key", "item_id"],
                           suffixes=("_random", "_bts"))
uplift_df["ctr_random"] = uplift_df["clicks_random"] / uplift_df["impressions_random"]
uplift_df["ctr_bts"]    = uplift_df["clicks_bts"]    / uplift_df["impressions_bts"]
uplift_df["uplift"]     = uplift_df["ctr_bts"] - uplift_df["ctr_random"]
uplift_df["user_id"]    = uplift_df["user_key"].map(user2id).astype("Int32")

uplift_df[["user_id", "item_id", "ctr_random", "ctr_bts",
           "uplift", "impressions_random", "impressions_bts"]].to_csv(
    OUT / "uplift_estimates.csv", index=False)

print(f"  Pairs with uplift estimate : {len(uplift_df):,}")
print(f"  Mean uplift                : {uplift_df['uplift'].mean():.4f}")
print(f"  Positive uplift (BTS>Rand) : {(uplift_df['uplift'] > 0).sum():,}")

# Free memory before the final split step
del stats_r, stats_b, uplift_df, clicks_r, clicks_b, clicks

# ---------------------------------------------------------------------------
# STEP 5: Temporal train / val / test split  (include affinity columns)
# ---------------------------------------------------------------------------
print("\nSTEP 5: Creating temporal splits (with affinity columns) ...")

# Combine lightweight frames to determine temporal boundaries
df_combined_light = pd.concat([df_random, df_bts], ignore_index=True)
df_combined_light.sort_values("timestamp", inplace=True)
n = len(df_combined_light)
t_train_idx = int(n * TRAIN_RATIO)
t_val_idx   = int(n * (TRAIN_RATIO + VAL_RATIO))

ts_train_end = df_combined_light["timestamp"].iloc[t_train_idx]
ts_val_end   = df_combined_light["timestamp"].iloc[t_val_idx]
del df_combined_light

print(f"  Train boundary : < {ts_train_end}")
print(f"  Val   boundary : < {ts_val_end}")

# Now write the split CSVs by streaming the full data (with affinities)
# We handle the random file (small) fully in memory, BTS in chunks.
SAVE_COLS = (["user_id", "item_id", "position", "click",
              "propensity_score", "policy", "timestamp"]
             + USER_FEAT_COLS + AFFINITY_COLS)

train_path = OUT / "train.csv"
val_path   = OUT / "val.csv"
test_path  = OUT / "test.csv"

# --- random (fits in memory) ---
df_r_full = pd.read_csv(RANDOM_CSV, index_col=0)
_parse_ts(df_r_full)
df_r_full["policy"]   = "random"
df_r_full["user_id"]  = user_key_series(df_r_full).map(user2id).astype(np.int32)
for path, mask in [
        (train_path, df_r_full["timestamp"] <  ts_train_end),
        (val_path,   (df_r_full["timestamp"] >= ts_train_end) & (df_r_full["timestamp"] < ts_val_end)),
        (test_path,  df_r_full["timestamp"] >= ts_val_end)]:
    sub = df_r_full.loc[mask, [c for c in SAVE_COLS if c in df_r_full.columns]]
    sub.to_csv(path, index=False, mode="w", header=True)
    print(f"  random -> {path.name}: {len(sub):,} rows")
del df_r_full

# --- bts (chunked) ---
for chunk in pd.read_csv(BTS_CSV, index_col=0, chunksize=CHUNKSIZE):
    _parse_ts(chunk)
    chunk["policy"]  = "bts"
    chunk["user_id"] = user_key_series(chunk).map(user2id).astype(np.int32)
    for path, mask in [
            (train_path, chunk["timestamp"] <  ts_train_end),
            (val_path,   (chunk["timestamp"] >= ts_train_end) & (chunk["timestamp"] < ts_val_end)),
            (test_path,  chunk["timestamp"] >= ts_val_end)]:
        sub = chunk.loc[mask, [c for c in SAVE_COLS if c in chunk.columns]]
        if len(sub):
            sub.to_csv(path, index=False, mode="a", header=False)

# Print final row counts
for label, path in [("Train", train_path), ("Val", val_path), ("Test", test_path)]:
    # Fast row count without loading
    count = sum(1 for _ in open(path)) - 1
    print(f"  {label:5s}: {count:>10,} rows  ->  {path.name}")

# ---------------------------------------------------------------------------
# STEP 6: Item features
# ---------------------------------------------------------------------------
print("\nSTEP 6: Saving item feature matrix ...")
for col in ["item_feature_1", "item_feature_2", "item_feature_3"]:
    df_items[col] = pd.Categorical(df_items[col]).codes.astype(np.float32)
df_items.sort_values("item_id", inplace=True)
feat_matrix = df_items[["item_feature_0", "item_feature_1",
                         "item_feature_2", "item_feature_3"]].values.astype(np.float32)
np.save(OUT / "item_features.npy", feat_matrix)
df_items.to_csv(OUT / "item_features.csv", index=False)
print(f"  Item feature matrix shape: {feat_matrix.shape}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print(f"Output: {OUT.resolve()}")
print()
for f in sorted(OUT.iterdir()):
    kb = f.stat().st_size / 1024
    print(f"  {f.name:<35s}  {kb:>10.1f} KB")
print("=" * 60)
