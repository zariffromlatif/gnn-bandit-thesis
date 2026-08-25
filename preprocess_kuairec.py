"""
Preprocessing script for KuaiRec short-video recommendation dataset.

Downloads (if needed) and preprocesses KuaiRec into the same format
as OBD/Criteo for the GNN-Bandit pipeline:
  - BanditSplit containers (contexts, actions, rewards, propensities, user_ids)
  - LightGCN adjacency matrix
  - Video category clustering (K-Means on category tag features)
  - Ground-truth reward matrix for exact policy evaluation

Usage:
    python preprocess_kuairec.py [--n_clusters 50] [--reward_threshold 2.0]
"""

import argparse
import ast
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, vstack, hstack, eye
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[0]
RAW_DIR = ROOT / "data" / "kuairec"
OUT_DIR = ROOT / "data" / "processed_kuairec"


def load_raw_data():
    """Load raw KuaiRec CSV files."""
    print("Loading raw KuaiRec data ...")
    
    small = pd.read_csv(RAW_DIR / "data" / "small_matrix.csv")
    big = pd.read_csv(RAW_DIR / "data" / "big_matrix.csv")
    users = pd.read_csv(RAW_DIR / "data" / "user_features.csv")
    items_cat = pd.read_csv(RAW_DIR / "data" / "item_categories.csv")
    items_daily = pd.read_csv(RAW_DIR / "data" / "item_daily_features.csv")
    
    # Standardize column names (v1 used 'photo_id')
    for df in [small, big]:
        if 'photo_id' in df.columns and 'video_id' not in df.columns:
            df.rename(columns={'photo_id': 'video_id'}, inplace=True)
    
    print(f"  small_matrix: {len(small):,} rows ({small['user_id'].nunique()} users x {small['video_id'].nunique()} items)")
    print(f"  big_matrix:   {len(big):,} rows ({big['user_id'].nunique()} users x {big['video_id'].nunique()} items)")
    print(f"  user_features: {len(users):,} users, {users.shape[1]} columns")
    print(f"  item_categories: {len(items_cat):,} items")
    
    return small, big, users, items_cat, items_daily


def build_user_features(users_df: pd.DataFrame, valid_user_ids: set) -> tuple:
    """Extract numerical user context features."""
    # Filter to valid users only
    users = users_df[users_df['user_id'].isin(valid_user_ids)].copy()
    users = users.sort_values('user_id').reset_index(drop=True)
    
    # Numerical features
    num_cols = ['follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days']
    # Binary features
    bin_cols = ['is_lowactive_period', 'is_live_streamer', 'is_video_author']
    # Onehot features (encrypted categorical)
    onehot_cols = [c for c in users.columns if c.startswith('onehot_feat')]
    
    feature_cols = num_cols + bin_cols + onehot_cols
    features = users[feature_cols].fillna(0).values.astype(np.float32)
    
    # Normalize numerical columns
    for i in range(len(num_cols)):
        col = features[:, i]
        mean, std = col.mean(), col.std()
        if std > 0:
            features[:, i] = (col - mean) / std
    
    user_id_list = users['user_id'].values
    user2id = {uid: idx for idx, uid in enumerate(user_id_list)}
    
    print(f"  User features: {features.shape} ({len(feature_cols)} dims)")
    return features, user2id, user_id_list


def build_video_clusters(items_cat: pd.DataFrame, n_clusters: int = 50) -> tuple:
    """Cluster videos into K categories using multi-hot tag features + K-Means."""
    # Parse category tag lists
    items_cat = items_cat.copy()
    items_cat['feat_list'] = items_cat['feat'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    
    # Multi-hot encode category tags
    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform(items_cat['feat_list'])
    print(f"  Video tag matrix: {tag_matrix.shape} ({len(mlb.classes_)} unique tags)")
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tag_matrix)
    
    video2cluster = dict(zip(items_cat['video_id'].values, cluster_labels))
    
    # Cluster centroids as "item features" for LightGCN
    centroids = kmeans.cluster_centers_.astype(np.float32)
    
    print(f"  Clustered {len(items_cat)} videos into {n_clusters} categories")
    cluster_sizes = pd.Series(cluster_labels).value_counts()
    print(f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"median={cluster_sizes.median():.0f}")
    
    return video2cluster, centroids, cluster_labels


def build_ground_truth_matrix(small_df: pd.DataFrame, user2id: dict,
                               video2cluster: dict, n_clusters: int,
                               reward_threshold: float) -> np.ndarray:
    """
    Build the ground-truth reward matrix from KuaiRec's fully-observed small_matrix.
    
    Returns:
        gt_matrix: (n_users, n_clusters) mean binary reward per user-cluster pair
    """
    n_users = len(user2id)
    gt_matrix = np.zeros((n_users, n_clusters), dtype=np.float32)
    gt_counts = np.zeros((n_users, n_clusters), dtype=np.float32)
    
    for _, row in small_df.iterrows():
        uid = row['user_id']
        vid = row['video_id']
        if uid not in user2id or vid not in video2cluster:
            continue
        uidx = user2id[uid]
        cidx = video2cluster[vid]
        reward = 1.0 if row['watch_ratio'] >= reward_threshold else 0.0
        gt_matrix[uidx, cidx] += reward
        gt_counts[uidx, cidx] += 1.0
    
    # Average reward per cluster
    mask = gt_counts > 0
    gt_matrix[mask] /= gt_counts[mask]
    
    coverage = mask.mean()
    print(f"  Ground-truth matrix: ({n_users}, {n_clusters}), coverage: {coverage:.2%}")
    return gt_matrix


def build_interaction_data(big_df: pd.DataFrame, user2id: dict,
                            video2cluster: dict, user_features: np.ndarray,
                            n_clusters: int, reward_threshold: float):
    """
    Convert big_matrix interactions into bandit-format data.
    
    Treatment/action = video cluster assignment
    Reward = binary engagement (watch_ratio >= threshold)
    Propensity = uniform over clusters (approximation for unlogged policy)
    """
    records = []
    for _, row in big_df.iterrows():
        uid = row['user_id']
        vid = row['video_id']
        if uid not in user2id or vid not in video2cluster:
            continue
        records.append({
            'user_idx': user2id[uid],
            'cluster': video2cluster[vid],
            'reward': 1.0 if row['watch_ratio'] >= reward_threshold else 0.0,
            'timestamp': row.get('timestamp', 0),
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    contexts = user_features[df['user_idx'].values]
    actions = df['cluster'].values.astype(np.int32)
    rewards = df['reward'].values.astype(np.float32)
    user_ids = df['user_idx'].values.astype(np.int32)
    
    # Propensity: approximate as uniform over clusters
    propensities = np.full(len(df), 1.0 / n_clusters, dtype=np.float32)
    
    print(f"  Interaction data: {len(df):,} rows, {df['user_idx'].nunique()} users, "
          f"{df['cluster'].nunique()} clusters, click_rate: {rewards.mean():.4f}")
    
    return contexts, actions, rewards, propensities, user_ids, df['timestamp'].values


def build_lightgcn_adj(user_ids: np.ndarray, actions: np.ndarray,
                        n_users: int, n_clusters: int) -> csr_matrix:
    """Build bipartite User-Cluster adjacency for LightGCN."""
    # Count interactions per (user, cluster) pair
    from collections import Counter
    edge_counts = Counter(zip(user_ids, actions))
    
    rows, cols, vals = [], [], []
    for (u, c), count in edge_counts.items():
        rows.append(u)
        cols.append(c)
        vals.append(float(count))
    
    # User-Cluster interaction matrix R: (n_users, n_clusters)
    R = csr_matrix((vals, (rows, cols)), shape=(n_users, n_clusters))
    
    # Binarize (presence of interaction)
    R.data[:] = 1.0
    
    # Build symmetric block adjacency: [[0, R], [R^T, 0]]
    n_nodes = n_users + n_clusters
    zero_uu = csr_matrix((n_users, n_users))
    zero_cc = csr_matrix((n_clusters, n_clusters))
    adj = vstack([
        hstack([zero_uu, R]),
        hstack([R.T, zero_cc]),
    ]).tocsr()
    
    print(f"  LightGCN adjacency: ({n_nodes}, {n_nodes}), nnz: {adj.nnz}")
    return adj


def main(n_clusters: int = 50, reward_threshold: float = 2.0,
         train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Full preprocessing pipeline for KuaiRec."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load raw data
    small, big, users, items_cat, items_daily = load_raw_data()
    
    # 2. Build video clusters
    video2cluster, centroids, _ = build_video_clusters(items_cat, n_clusters)
    
    # 3. Get valid user IDs (users in small_matrix = ground-truth users)
    gt_user_ids = set(small['user_id'].unique())
    # Also include users from big_matrix for training
    all_user_ids = gt_user_ids | set(big['user_id'].unique())
    
    # 4. Build user features
    user_features, user2id, _ = build_user_features(users, all_user_ids)
    
    # 5. Build ground-truth matrix (from small_matrix)
    gt_matrix = build_ground_truth_matrix(small, user2id, video2cluster,
                                           n_clusters, reward_threshold)
    
    # 6. Build interaction data (from big_matrix)
    contexts, actions, rewards, propensities, user_ids, timestamps = \
        build_interaction_data(big, user2id, video2cluster, user_features,
                               n_clusters, reward_threshold)
    
    # 7. Chronological train/val/test split
    N = len(contexts)
    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))
    
    splits = {
        'train': slice(0, train_end),
        'val': slice(train_end, val_end),
        'test': slice(val_end, N),
    }
    
    for split_name, sl in splits.items():
        np.savez_compressed(
            OUT_DIR / f"context_{split_name}.npz",
            contexts=contexts[sl],
            item_id=actions[sl],
            click=rewards[sl],
            propensity_score=propensities[sl],
            user_id=user_ids[sl],
            timestamps=timestamps[sl],
        )
        print(f"  Saved {split_name}: {sl.stop - sl.start:,} rows")
    
    # 8. Build LightGCN adjacency (from training data only)
    train_sl = splits['train']
    adj = build_lightgcn_adj(user_ids[train_sl], actions[train_sl],
                              len(user2id), n_clusters)
    save_npz(OUT_DIR / "lightgcn_adj.npz", adj)
    
    # 9. Save item features (cluster centroids)
    np.save(OUT_DIR / "item_features.npy", centroids)
    
    # 10. Save ground-truth matrix
    np.save(OUT_DIR / "ground_truth_matrix.npy", gt_matrix)
    
    # 11. Save user2id mapping
    with open(OUT_DIR / "user2id.pkl", "wb") as f:
        pickle.dump(user2id, f)
    
    # 12. Save stats
    stats = {
        "dataset": "kuairec",
        "n_user_segments": len(user2id),
        "n_items": n_clusters,
        "n_impressions": N,
        "context_dim": contexts.shape[1],
        "graph_nodes": len(user2id) + n_clusters,
        "n_video_clusters": n_clusters,
        "reward_threshold": reward_threshold,
        "gt_users": len(gt_user_ids),
        "gt_items": small['video_id'].nunique(),
        "gt_density": gt_matrix.astype(bool).mean(),
        "splits": {
            name: {
                "rows": sl.stop - sl.start,
                "click_rate": float(rewards[sl].mean()),
            }
            for name, sl in splits.items()
        },
    }
    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ KuaiRec preprocessing complete! Output: {OUT_DIR}")
    print(f"  Users: {len(user2id)}, Clusters: {n_clusters}, "
          f"Interactions: {N:,}, Context dim: {contexts.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--reward_threshold", type=float, default=2.0,
                        help="watch_ratio threshold for binary engagement")
    args = parser.parse_args()
    main(n_clusters=args.n_clusters, reward_threshold=args.reward_threshold)
