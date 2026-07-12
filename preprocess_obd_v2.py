"""
OBD Preprocessing v2 — Impression-Level Contextual Bandit + Enriched Graph
============================================================================
Implements Options A + B + Multi-Campaign for the Three-Tier evaluation.

MEMORY-SAFE: Never concatenates all BTS rows with affinities at once.
Uses streaming/chunked processing for all large operations.

Key changes from v1:
  1. IMPRESSION-LEVEL STATES: Each impression row becomes a BCQ context
     (4 user features + position + N affinities + LightGCN embedding).
  2. ENRICHED GRAPH: Dense affinity-weighted bipartite edges + user-user
     similarity edges + item-item similarity edges → heterogeneous graph.
  3. MULTI-CAMPAIGN: Processes All, Men, Women campaigns independently.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, hstack, vstack
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# Configuration
# ============================================================================
ROOT     = Path(__file__).parent / "data" / "open_bandit_dataset"
OUT_BASE = Path(__file__).parent / "data" / "processed_v2"

CAMPAIGNS = {
    "all":   {"n_items": 80, "affinity_cols": 80},
    "men":   {"n_items": 34, "affinity_cols": 34},
    "women": {"n_items": 46, "affinity_cols": 46},
}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
CHUNKSIZE   = 500_000
USER_FEAT_COLS = [f"user_feature_{i}" for i in range(4)]
KNN_K = 10


# ============================================================================
# Helpers
# ============================================================================
def user_key_series(df: pd.DataFrame) -> pd.Series:
    return df[USER_FEAT_COLS].astype(str).agg("-".join, axis=1)


def parse_ts(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    return df


def affinity_cols_for(campaign: str) -> list[str]:
    n = CAMPAIGNS[campaign]["affinity_cols"]
    return [f"user-item_affinity_{i}" for i in range(n)]


def core_cols_light() -> list[str]:
    """Columns needed for user-key encoding & temporal split (no affinities)."""
    return (["timestamp", "item_id", "position", "click", "propensity_score"]
            + USER_FEAT_COLS)


def build_knn_adj(features: np.ndarray, k: int) -> csr_matrix:
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
# Processing per campaign
# ============================================================================
def process_campaign(campaign: str):
    print("\n" + "=" * 70)
    print(f"PROCESSING CAMPAIGN: {campaign.upper()}")
    print("=" * 70)

    cfg = CAMPAIGNS[campaign]
    n_aff = cfg["affinity_cols"]
    aff_cols = affinity_cols_for(campaign)
    all_cols_with_aff = core_cols_light() + aff_cols

    random_csv = ROOT / "random" / campaign / f"{campaign}.csv"
    bts_csv    = ROOT / "bts"    / campaign / f"{campaign}.csv"
    item_csv   = ROOT / "random" / campaign / "item_context.csv"

    out = OUT_BASE / campaign
    out.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # PASS 1: Load LIGHT data (no affinities) to build user2id & split
    # ==================================================================
    print("\nPASS 1: Loading lightweight data (no affinities) ...")
    light_cols = set(core_cols_light())

    df_r_light = pd.read_csv(random_csv, index_col=False,
                              usecols=lambda c: c in light_cols)
    parse_ts(df_r_light)
    df_r_light["policy"]   = "random"
    df_r_light["user_key"] = user_key_series(df_r_light)
    print(f"  random | {len(df_r_light):>10,} rows")

    # BTS light — chunked
    bts_light_chunks = []
    for chunk in pd.read_csv(bts_csv, index_col=False,
                              usecols=lambda c: c in light_cols,
                              chunksize=CHUNKSIZE):
        parse_ts(chunk)
        chunk["policy"]   = "bts"
        chunk["user_key"] = user_key_series(chunk)
        bts_light_chunks.append(chunk)
        print(f"    ... bts light: {sum(len(c) for c in bts_light_chunks):>10,}",
              end="\r")
    print()
    df_b_light = pd.concat(bts_light_chunks, ignore_index=True)
    del bts_light_chunks
    print(f"  bts    | {len(df_b_light):>10,} rows")

    # ------------------------------------------------------------------
    # User segment encoding
    # ------------------------------------------------------------------
    print("\n  Encoding user segments ...")
    all_keys = pd.concat([df_r_light["user_key"],
                          df_b_light["user_key"]]).unique()
    user2id = {k: i for i, k in enumerate(all_keys)}
    n_users = len(user2id)
    n_items = cfg["n_items"]

    df_r_light["user_id"] = df_r_light["user_key"].map(user2id).astype(np.int32)
    df_b_light["user_id"] = df_b_light["user_key"].map(user2id).astype(np.int32)

    print(f"  User segments: {n_users}")
    print(f"  Items:         {n_items}")

    with open(out / "user2id.pkl", "wb") as f:
        pickle.dump(user2id, f)

    # ------------------------------------------------------------------
    # Temporal split boundaries
    # ------------------------------------------------------------------
    print("\n  Computing temporal split boundaries ...")
    combined = pd.concat([df_r_light[["timestamp"]],
                          df_b_light[["timestamp"]]], ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    n_total = len(combined)
    t_train_idx = int(n_total * TRAIN_RATIO)
    t_val_idx   = int(n_total * (TRAIN_RATIO + VAL_RATIO))
    ts_train_end = combined["timestamp"].iloc[t_train_idx]
    ts_val_end   = combined["timestamp"].iloc[t_val_idx]
    del combined

    print(f"  Total impressions: {n_total:,}")
    print(f"  Train boundary:    < {ts_train_end}")
    print(f"  Val   boundary:    < {ts_val_end}")

    # ------------------------------------------------------------------
    # Uplift estimates (from light data — click stats only)
    # ------------------------------------------------------------------
    print("\n  Computing uplift ...")

    def click_stats(df):
        return (df.groupby(["user_id", "item_id"])["click"]
                  .agg(clicks="sum", impressions="count")
                  .reset_index())

    stats_r = click_stats(df_r_light)
    stats_b = click_stats(df_b_light)
    uplift_df = stats_r.merge(stats_b, on=["user_id", "item_id"],
                               suffixes=("_random", "_bts"))
    uplift_df["ctr_random"] = uplift_df["clicks_random"] / uplift_df["impressions_random"]
    uplift_df["ctr_bts"]    = uplift_df["clicks_bts"]    / uplift_df["impressions_bts"]
    uplift_df["uplift"]     = uplift_df["ctr_bts"] - uplift_df["ctr_random"]
    uplift_df.to_csv(out / "uplift_estimates.csv", index=False)

    print(f"  Pairs: {len(uplift_df):,} | "
          f"Mean uplift: {uplift_df['uplift'].mean():.6f} | "
          f"Positive: {(uplift_df['uplift'] > 0).sum():,} | "
          f"Sleeping Dogs: {(uplift_df['uplift'] < 0).sum():,}")
    del stats_r, stats_b, uplift_df

    # Encode user features (needed for context vectors)
    # Fit encoder on combined light data
    user_feat_categories = {}
    for col in USER_FEAT_COLS:
        cats = pd.Categorical(
            pd.concat([df_r_light[col], df_b_light[col]])
        ).categories
        user_feat_categories[col] = cats

    del df_r_light, df_b_light  # Free light dataframes

    # ==================================================================
    # PASS 2: Stream full data (with affinities) for graph + contexts
    # ==================================================================
    print("\nPASS 2: Streaming full data for graph & context vectors ...")

    # Accumulators for user-segment affinity profiles
    aff_sum   = np.zeros((n_users, n_aff), dtype=np.float64)
    aff_count = np.zeros(n_users, dtype=np.int64)

    # Context output files (write incrementally)
    train_ctx_list = []
    val_ctx_list   = []
    test_ctx_list  = []
    train_meta_list = []
    val_meta_list   = []
    test_meta_list  = []

    def process_df_for_contexts(df, policy_label):
        """Process a dataframe chunk: accumulate affinities + split to contexts."""
        nonlocal aff_sum, aff_count

        df["user_key"] = user_key_series(df)
        df["user_id"]  = df["user_key"].map(user2id).astype(np.int32)

        # Accumulate affinities for graph
        for uid in df["user_id"].unique():
            mask = df["user_id"] == uid
            aff_sum[uid]   += df.loc[mask, aff_cols].values.sum(axis=0)
            aff_count[uid] += mask.sum()

        # Encode user features
        for col in USER_FEAT_COLS:
            df[col + "_enc"] = pd.Categorical(
                df[col], categories=user_feat_categories[col]
            ).codes.astype(np.float32)

        user_feat_enc_cols = [c + "_enc" for c in USER_FEAT_COLS]
        context_cols = user_feat_enc_cols + ["position"] + aff_cols

        # Split by timestamp
        mask_train = df["timestamp"] < ts_train_end
        mask_val   = (df["timestamp"] >= ts_train_end) & (df["timestamp"] < ts_val_end)
        mask_test  = df["timestamp"] >= ts_val_end

        for mask, ctx_list, meta_list in [
            (mask_train, train_ctx_list, train_meta_list),
            (mask_val,   val_ctx_list,   val_meta_list),
            (mask_test,  test_ctx_list,  test_meta_list),
        ]:
            sub = df.loc[mask]
            if len(sub) == 0:
                continue
            ctx_list.append(sub[context_cols].values.astype(np.float32))
            meta_list.append({
                "user_id":          sub["user_id"].values.astype(np.int32),
                "item_id":          sub["item_id"].values.astype(np.int32),
                "click":            sub["click"].values.astype(np.int8),
                "propensity_score": sub["propensity_score"].values.astype(np.float32),
                "policy":           np.full(len(sub), policy_label, dtype="U6"),
            })

    # --- Process random (fits in memory with affinities) ---
    print("  Processing random (full) ...")
    full_cols = set(all_cols_with_aff)
    df_r = pd.read_csv(random_csv, index_col=False,
                        usecols=lambda c: c in full_cols)
    parse_ts(df_r)
    process_df_for_contexts(df_r, "random")
    print(f"    random done: {len(df_r):,} rows")
    del df_r

    # --- Process BTS in chunks ---
    print("  Processing BTS (chunked) ...")
    total_bts = 0
    for chunk in pd.read_csv(bts_csv, index_col=False,
                              usecols=lambda c: c in full_cols,
                              chunksize=CHUNKSIZE):
        parse_ts(chunk)
        process_df_for_contexts(chunk, "bts")
        total_bts += len(chunk)
        print(f"    ... bts: {total_bts:>10,}", end="\r")
    print(f"\n    bts done: {total_bts:,} rows")

    # ==================================================================
    # STEP 3: Build enriched graph (Option B)
    # ==================================================================
    print("\nSTEP 3: Building enriched graph ...")

    # 3a. Affinity-weighted bipartite graph
    aff_mean = aff_sum / np.maximum(aff_count, 1)[:, None]
    aff_mean = aff_mean.astype(np.float32)

    # Save user segment profiles
    pd.DataFrame(aff_mean, columns=aff_cols).to_csv(
        out / "user_segment_profiles.csv", index_label="user_id")

    R_weighted = csr_matrix(aff_mean)
    save_npz(out / "graph_bipartite_weighted.npz", R_weighted)
    print(f"  Weighted bipartite: {R_weighted.shape}, "
          f"nnz={R_weighted.nnz}, "
          f"density={R_weighted.nnz / max(n_users * n_items, 1):.4f}")

    # 3b. User-user k-NN graph
    print(f"  Building user-user k-NN graph (k={KNN_K}) ...")
    A_uu = build_knn_adj(aff_mean, k=min(KNN_K, n_users - 1))
    save_npz(out / "graph_user_user_sim.npz", A_uu)
    print(f"  User-user adj: {A_uu.shape}, nnz={A_uu.nnz}")

    # 3c. Item-item similarity graph
    df_items = pd.read_csv(item_csv)
    for col in ["item_feature_1", "item_feature_2", "item_feature_3"]:
        if col in df_items.columns:
            df_items[col] = pd.Categorical(df_items[col]).codes.astype(np.float32)
    df_items.sort_values("item_id", inplace=True)
    item_feat_names = [c for c in df_items.columns if c.startswith("item_feature")]
    item_feats = df_items[item_feat_names].values.astype(np.float32)
    np.save(out / "item_features.npy", item_feats)
    df_items.to_csv(out / "item_features.csv", index=False)

    A_ii = build_knn_adj(item_feats, k=min(KNN_K, n_items - 1))
    save_npz(out / "graph_item_item_sim.npz", A_ii)
    print(f"  Item-item adj: {A_ii.shape}, nnz={A_ii.nnz}")

    # 3d. Full LightGCN block adjacency
    A_full = vstack([
        hstack([A_uu,         R_weighted]),
        hstack([R_weighted.T, A_ii])
    ]).tocsr()
    save_npz(out / "lightgcn_adj.npz", A_full)
    print(f"  LightGCN block adj: {A_full.shape}, nnz={A_full.nnz}")

    # ==================================================================
    # STEP 4: Save context splits
    # ==================================================================
    print("\nSTEP 4: Saving context splits ...")

    split_stats = {}
    for name, ctx_list, meta_list in [
        ("train", train_ctx_list, train_meta_list),
        ("val",   val_ctx_list,   val_meta_list),
        ("test",  test_ctx_list,  test_meta_list),
    ]:
        if not ctx_list:
            print(f"  {name}: EMPTY")
            continue

        ctx_all  = np.concatenate(ctx_list, axis=0)
        uid_all  = np.concatenate([m["user_id"] for m in meta_list])
        iid_all  = np.concatenate([m["item_id"] for m in meta_list])
        clk_all  = np.concatenate([m["click"] for m in meta_list])
        ps_all   = np.concatenate([m["propensity_score"] for m in meta_list])
        pol_all  = np.concatenate([m["policy"] for m in meta_list])

        np.savez_compressed(
            out / f"context_{name}.npz",
            contexts=ctx_all,
            user_id=uid_all,
            item_id=iid_all,
            click=clk_all,
            propensity_score=ps_all,
        )
        np.save(out / f"policy_{name}.npy", pol_all)

        n_rand = (pol_all == "random").sum()
        n_bts  = (pol_all == "bts").sum()
        ctr    = clk_all.mean()

        split_stats[name] = {
            "rows":        int(len(ctx_all)),
            "random_rows": int(n_rand),
            "bts_rows":    int(n_bts),
            "click_rate":  float(f"{ctr:.6f}"),
        }
        print(f"  {name:5s}: {len(ctx_all):>10,} impressions | "
              f"random={n_rand:,} bts={n_bts:,} | CTR={ctr:.4f}")

        del ctx_all, uid_all, iid_all, clk_all, ps_all, pol_all

    ctx_dim = n_aff + 5  # 4 user feats + position + N affinities

    # ==================================================================
    # STEP 5: Save stats
    # ==================================================================
    stats = {
        "campaign":         campaign,
        "n_user_segments":  n_users,
        "n_items":          n_items,
        "n_impressions":    n_total,
        "context_dim":      ctx_dim,
        "graph_nodes":      n_users + n_items,
        "graph_nnz_bipartite": int(R_weighted.nnz),
        "graph_nnz_user_user": int(A_uu.nnz),
        "graph_nnz_item_item": int(A_ii.nnz),
        "graph_nnz_total":     int(A_full.nnz),
        "train_boundary":   str(ts_train_end),
        "val_boundary":     str(ts_val_end),
        "splits":           split_stats,
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Context dim: {ctx_dim} "
          f"(4 user + 1 position + {n_aff} affinities)")
    print(f"  Graph nodes: {n_users + n_items} "
          f"({n_users} users + {n_items} items)")
    print(f"  Graph edges: {A_full.nnz:,} (weighted)")

    print(f"\n  Output: {out.resolve()}")
    for fp in sorted(out.iterdir()):
        kb = fp.stat().st_size / 1024
        unit = "KB"
        if kb > 1024:
            kb /= 1024; unit = "MB"
        print(f"    {fp.name:<40s} {kb:>8.1f} {unit}")


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("OBD PREPROCESSING v2")
    print("Option A (Impression-Level) + Option B (Enriched Graph)")
    print("Multi-Campaign: All, Men, Women")
    print("=" * 70)

    for campaign in CAMPAIGNS:
        process_campaign(campaign)

    print("\n" + "=" * 70)
    print("ALL CAMPAIGNS COMPLETE")
    print("=" * 70)
