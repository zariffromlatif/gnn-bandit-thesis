"""
Preprocessing script for KuaiRand short-video recommendation dataset.

Downloads (if needed) and preprocesses KuaiRand-Pure into the same format
as OBD/Criteo for the GNN-Bandit pipeline:
  - BanditSplit containers (contexts, actions, rewards, propensities, user_ids)
  - LightGCN adjacency matrix
  - Video category clustering (K-Means on tag features)
  - Separate random-exposure subset for unbiased OPE

Usage:
    python preprocess_kuairand.py [--n_clusters 50] [--variant pure]
"""

import argparse
import ast
import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, vstack, hstack
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[0]
RAW_DIR = ROOT / "data" / "kuairand"
OUT_DIR = ROOT / "data" / "processed_kuairand"


def _find_file(base_dir: Path, filename: str) -> Path:
    """Find a file either directly, in data/, or anywhere recursively in base_dir."""
    candidates = [
        base_dir / filename,
        base_dir / "data" / filename,
        base_dir / "KuaiRand-Pure" / "data" / filename,
        base_dir / "KuaiRand-Pure" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    # Recursive search
    matches = list(base_dir.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find '{filename}' anywhere inside {base_dir}")


def load_raw_data(variant: str = "pure"):
    """Load raw KuaiRand CSV files."""
    print("Loading raw KuaiRand data ...")

    # User features
    users_path = _find_file(RAW_DIR, "user_features.csv")
    users = pd.read_csv(users_path)
    print(f"  Found KuaiRand files in: {users_path.parent}")
    print(f"  Users: {len(users):,}")
    
    # Video features
    basic_path = _find_file(RAW_DIR, f"video_features_basic_{variant}.csv")
    stat_path = _find_file(RAW_DIR, f"video_features_statistic_{variant}.csv")
    
    videos_basic = pd.read_csv(basic_path)
    videos_stat = pd.read_csv(stat_path)
    videos = pd.merge(videos_basic, videos_stat, on="video_id", how="left")
    print(f"  Videos: {len(videos):,}")
    
    # Interaction logs
    rand_path = _find_file(RAW_DIR, f"log_random_4_22_to_5_08_{variant}.csv")
    std_path = _find_file(RAW_DIR, f"log_standard_4_22_to_5_08_{variant}.csv")

    log_random = pd.read_csv(rand_path)
    log_random["is_random"] = 1
    
    log_standard = pd.read_csv(std_path)
    log_standard["is_random"] = 0
    
    # Also load prior standard logs if available
    try:
        prior_path = _find_file(RAW_DIR, f"log_standard_4_08_to_4_21_{variant}.csv")
        log_prior = pd.read_csv(prior_path)
        log_prior["is_random"] = 0
        log_standard = pd.concat([log_prior, log_standard], ignore_index=True)
    except FileNotFoundError:
        pass
    
    print(f"  Random exposure logs: {len(log_random):,}")
    print(f"  Standard logs: {len(log_standard):,}")
    
    return users, videos, log_random, log_standard


def build_user_features(users_df: pd.DataFrame, valid_user_ids: set) -> tuple:
    """Extract numerical user context features."""
    users = users_df[users_df['user_id'].isin(valid_user_ids)].copy()
    users = users.sort_values('user_id').reset_index(drop=True)
    
    # Binary features
    bin_cols = [c for c in users.columns if c.startswith('is_')]
    # Onehot categorical features (encrypted)
    onehot_cols = [c for c in users.columns if c.startswith('onehot_feat')]
    
    feature_cols = bin_cols + onehot_cols
    features = users[feature_cols].fillna(0).values.astype(np.float32)
    
    user_id_list = users['user_id'].values
    user2id = {uid: idx for idx, uid in enumerate(user_id_list)}
    
    print(f"  User features: {features.shape}")
    return features, user2id, user_id_list


def build_video_clusters(videos_df: pd.DataFrame, n_clusters: int = 50) -> tuple:
    """Cluster videos into K categories using tag features + K-Means."""
    videos = videos_df.copy()
    
    # Parse tag column: comma-separated integers
    def parse_tags(tag_str):
        if pd.isna(tag_str) or tag_str == '':
            return []
        try:
            # Handle both "13,44,12" and "[13, 44, 12]" formats
            tag_str = str(tag_str).strip()
            if tag_str.startswith('['):
                return [int(x) for x in ast.literal_eval(tag_str)]
            return [int(x.strip()) for x in tag_str.split(',') if x.strip()]
        except (ValueError, SyntaxError):
            return []
    
    videos['tag_list'] = videos['tag'].apply(parse_tags)
    
    # Multi-hot encode tags
    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform(videos['tag_list'])
    print(f"  Video tag matrix: {tag_matrix.shape} ({len(mlb.classes_)} unique tags)")
    
    # Also include numerical video features for richer clustering
    stat_cols = [c for c in videos.columns if c in [
        'play_cnt', 'play_user_num', 'complete_play_cnt', 'valid_play_cnt',
        'long_view_cnt', 'short_view_cnt', 'like_cnt', 'comment_cnt',
        'share_cnt', 'download_cnt', 'video_duration'
    ]]
    
    if stat_cols:
        stat_features = videos[stat_cols].fillna(0).values.astype(np.float32)
        # Normalize
        for i in range(stat_features.shape[1]):
            col = stat_features[:, i]
            mean, std = col.mean(), col.std()
            if std > 0:
                stat_features[:, i] = (col - mean) / std
        cluster_input = np.hstack([tag_matrix.astype(np.float32), stat_features])
    else:
        cluster_input = tag_matrix.astype(np.float32)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_input)
    
    video2cluster = dict(zip(videos['video_id'].values, cluster_labels))
    centroids = kmeans.cluster_centers_.astype(np.float32)
    
    # Use only the tag portion of centroids as "item features"
    tag_centroids = centroids[:, :tag_matrix.shape[1]]
    
    print(f"  Clustered {len(videos)} videos into {n_clusters} categories")
    cluster_sizes = pd.Series(cluster_labels).value_counts()
    print(f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"median={cluster_sizes.median():.0f}")
    
    return video2cluster, tag_centroids, cluster_labels


def build_interaction_data(log_df: pd.DataFrame, user2id: dict,
                            video2cluster: dict, user_features: np.ndarray,
                            n_clusters: int, is_random: bool = False):
    """Convert interaction logs into bandit-format data."""
    records = []
    for _, row in log_df.iterrows():
        uid = row['user_id']
        vid = row['video_id']
        if uid not in user2id or vid not in video2cluster:
            continue
        # Use is_click as primary reward signal
        reward = float(row.get('is_click', 0))
        records.append({
            'user_idx': user2id[uid],
            'cluster': video2cluster[vid],
            'reward': reward,
            'timestamp': row.get('time_ms', 0),
        })
    
    if not records:
        raise ValueError("No valid interactions found!")
    
    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    contexts = user_features[df['user_idx'].values]
    actions = df['cluster'].values.astype(np.int32)
    rewards = df['reward'].values.astype(np.float32)
    user_ids = df['user_idx'].values.astype(np.int32)
    timestamps = df['timestamp'].values
    
    # Propensity
    if is_random:
        # Random exposure: uniform over the candidate pool
        propensities = np.full(len(df), 1.0 / n_clusters, dtype=np.float32)
    else:
        # Standard logs: estimate propensity from observed action frequencies
        action_counts = Counter(actions)
        total = len(actions)
        prop_per_action = {a: max(c / total, 1e-6) for a, c in action_counts.items()}
        propensities = np.array([prop_per_action[a] for a in actions], dtype=np.float32)
    
    source = "random" if is_random else "standard"
    print(f"  [{source}] Interactions: {len(df):,}, users: {df['user_idx'].nunique()}, "
          f"clusters: {df['cluster'].nunique()}, click_rate: {rewards.mean():.4f}")
    
    return contexts, actions, rewards, propensities, user_ids, timestamps


def build_lightgcn_adj(user_ids: np.ndarray, actions: np.ndarray,
                        n_users: int, n_clusters: int) -> csr_matrix:
    """Build bipartite User-Cluster adjacency for LightGCN."""
    edge_counts = Counter(zip(user_ids, actions))
    
    rows, cols, vals = [], [], []
    for (u, c), count in edge_counts.items():
        rows.append(u)
        cols.append(c)
        vals.append(1.0)
    
    R = csr_matrix((vals, (rows, cols)), shape=(n_users, n_clusters))
    
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


def main(n_clusters: int = 50, variant: str = "pure",
         train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Full preprocessing pipeline for KuaiRand."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load raw data
    users, videos, log_random, log_standard = load_raw_data(variant)
    
    # 2. Build video clusters
    video2cluster, centroids, _ = build_video_clusters(videos, n_clusters)
    
    # 3. Get valid user IDs (present in both feature table and logs)
    log_user_ids = set(log_standard['user_id'].unique()) | set(log_random['user_id'].unique())
    feature_user_ids = set(users['user_id'].unique())
    valid_user_ids = log_user_ids & feature_user_ids
    
    # 4. Build user features
    user_features, user2id, _ = build_user_features(users, valid_user_ids)
    
    # 5. Build interaction data from STANDARD logs (for training)
    ctx_std, act_std, rew_std, prop_std, uid_std, ts_std = \
        build_interaction_data(log_standard, user2id, video2cluster,
                               user_features, n_clusters, is_random=False)
    
    # 6. Build interaction data from RANDOM logs (for unbiased OPE)
    ctx_rnd, act_rnd, rew_rnd, prop_rnd, uid_rnd, ts_rnd = \
        build_interaction_data(log_random, user2id, video2cluster,
                               user_features, n_clusters, is_random=True)
    
    # 7. Chronological train/val/test split on standard logs
    N = len(ctx_std)
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
            contexts=ctx_std[sl],
            item_id=act_std[sl],
            click=rew_std[sl],
            propensity_score=prop_std[sl],
            user_id=uid_std[sl],
            timestamps=ts_std[sl],
        )
        print(f"  Saved {split_name}: {sl.stop - sl.start:,} rows")
    
    # Save random exposure data separately (for unbiased evaluation)
    np.savez_compressed(
        OUT_DIR / "context_random_exposure.npz",
        contexts=ctx_rnd,
        item_id=act_rnd,
        click=rew_rnd,
        propensity_score=prop_rnd,
        user_id=uid_rnd,
        timestamps=ts_rnd,
    )
    print(f"  Saved random exposure: {len(ctx_rnd):,} rows")
    
    # 8. Build LightGCN adjacency (from training data only)
    train_sl = splits['train']
    adj = build_lightgcn_adj(uid_std[train_sl], act_std[train_sl],
                              len(user2id), n_clusters)
    save_npz(OUT_DIR / "lightgcn_adj.npz", adj)
    
    # 9. Save item features (cluster centroids)
    np.save(OUT_DIR / "item_features.npy", centroids)
    
    # 10. Save user2id mapping
    with open(OUT_DIR / "user2id.pkl", "wb") as f:
        pickle.dump(user2id, f)
    
    # 11. Save stats
    stats = {
        "dataset": "kuairand",
        "variant": variant,
        "n_user_segments": len(user2id),
        "n_items": n_clusters,
        "n_impressions": N,
        "n_random_impressions": len(ctx_rnd),
        "context_dim": ctx_std.shape[1],
        "graph_nodes": len(user2id) + n_clusters,
        "n_video_clusters": n_clusters,
        "n_raw_videos": len(videos),
        "splits": {
            name: {
                "rows": sl.stop - sl.start,
                "click_rate": float(rew_std[sl].mean()),
            }
            for name, sl in splits.items()
        },
    }
    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ KuaiRand preprocessing complete! Output: {OUT_DIR}")
    print(f"  Users: {len(user2id)}, Clusters: {n_clusters}, "
          f"Standard interactions: {N:,}, Random interactions: {len(ctx_rnd):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--variant", type=str, default="pure",
                        choices=["pure", "1k", "27k"])
    args = parser.parse_args()
    main(n_clusters=args.n_clusters, variant=args.variant)
