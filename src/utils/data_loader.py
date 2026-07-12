"""
Data loaders for OBD (Open Bandit Dataset) and Criteo Uplift datasets.

Provides a unified interface so that LightGCN, BCQ, and OPE modules can
consume either dataset without branching logic.
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import load_npz, csr_matrix


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class BanditSplit:
    """One train / val / test split for a contextual-bandit dataset."""
    contexts:        np.ndarray          # (N, D)  state features
    actions:         np.ndarray          # (N,)    action taken (item_id or treatment)
    rewards:         np.ndarray          # (N,)    observed reward (click / conversion)
    propensities:    np.ndarray          # (N,)    pi_0(a|x) logging-policy probability
    user_ids:        np.ndarray          # (N,)    user / cluster id
    policy_labels:   Optional[np.ndarray] = None   # (N,)  "random" / "bts" (OBD only)


@dataclass
class BanditDataset:
    """Full dataset ready for the GNN-Bandit pipeline."""
    name:            str
    train:           BanditSplit
    val:             BanditSplit
    test:            BanditSplit
    adj:             csr_matrix           # LightGCN block adjacency
    n_users:         int
    n_items:         int                  # number of actions
    n_nodes:         int                  # graph node count (users + items, or clusters)
    context_dim:     int
    stats:           dict = field(default_factory=dict)
    # Optional extras
    item_features:   Optional[np.ndarray] = None   # (n_items, F)
    uplift_df_path:  Optional[Path] = None
    user2id:         Optional[dict] = None


# ============================================================================
# OBD loader (v2 — impression-level, enriched graph)
# ============================================================================

def load_obd(campaign: str = "all",
             data_dir: str = "data/processed_v2",
             root: Optional[str] = None) -> BanditDataset:
    """
    Load one OBD campaign processed by ``preprocess_obd_v2.py``.

    Parameters
    ----------
    campaign : str
        One of "all", "men", "women".
    data_dir : str
        Relative (to *root*) or absolute path to the processed_v2 folder.
    root : str, optional
        Project root.  Defaults to the repo root (two levels up from this file).

    Returns
    -------
    BanditDataset
    """
    if root is None:
        root = Path(__file__).resolve().parents[2]
    base = Path(root) / data_dir / campaign

    # Stats
    with open(base / "stats.json") as f:
        stats = json.load(f)

    n_users = stats["n_user_segments"]
    n_items = stats["n_items"]

    # Graph adjacency (full block matrix for LightGCN)
    adj = load_npz(base / "lightgcn_adj.npz")

    # user2id mapping
    with open(base / "user2id.pkl", "rb") as f:
        user2id = pickle.load(f)

    # Item features
    item_features = np.load(base / "item_features.npy")

    # Load splits
    def _load_split(split_name: str) -> BanditSplit:
        data = np.load(base / f"context_{split_name}.npz")
        pol  = np.load(base / f"policy_{split_name}.npy")
        return BanditSplit(
            contexts=data["contexts"],          # (N, 85) for "all"
            actions=data["item_id"],             # item shown
            rewards=data["click"],               # 0 or 1
            propensities=data["propensity_score"],
            user_ids=data["user_id"],
            policy_labels=pol,
        )

    train = _load_split("train")
    val   = _load_split("val")
    test  = _load_split("test")

    return BanditDataset(
        name=f"OBD-{campaign}",
        train=train, val=val, test=test,
        adj=adj,
        n_users=n_users,
        n_items=n_items,
        n_nodes=n_users + n_items,
        context_dim=stats["context_dim"],
        stats=stats,
        item_features=item_features,
        uplift_df_path=base / "uplift_estimates.csv",
        user2id=user2id,
    )


# ============================================================================
# Criteo Uplift loader
# ============================================================================

def load_criteo(data_dir: str = "data/processed_criteo",
                root: Optional[str] = None) -> BanditDataset:
    """
    Load the Criteo Uplift v2.1 dataset processed by ``preprocess_criteo.py``.

    The Criteo dataset is a binary treatment setting (treat / control) rather
    than multi-arm bandit, so ``n_items = 2`` (action 0 = control, 1 = treat).
    The graph is a cluster-level k-NN adjacency built over 5 000 user segments.
    """
    if root is None:
        root = Path(__file__).resolve().parents[2]
    base = Path(root) / data_dir

    with open(base / "stats.json") as f:
        stats = json.load(f)

    n_clusters = stats["n_clusters"]
    n_actions  = 2  # treat / control

    # k-NN graph over cluster centroids
    adj = load_npz(base / "graph_user_knn.npz")

    # Cluster centroids can serve as "item features" in a loose sense
    centroids = np.load(base / "cluster_centroids.npy")

    def _load_split(split_name: str) -> BanditSplit:
        data = np.load(base / f"context_{split_name}.npz")
        treatment = data["treatment"]                     # 0 or 1
        conversion = data["conversion"]                   # 0 or 1

        # Propensity: Criteo is an RCT with ~85 % treatment rate
        treatment_rate = stats["treatment_rate"]
        propensities = np.where(
            treatment == 1,
            treatment_rate,
            1.0 - treatment_rate,
        ).astype(np.float32)

        return BanditSplit(
            contexts=data["contexts"],
            actions=treatment.astype(np.int32),
            rewards=conversion.astype(np.float32),
            propensities=propensities,
            user_ids=data["cluster_id"],
        )

    train = _load_split("train")
    val   = _load_split("val")
    test  = _load_split("test")

    return BanditDataset(
        name="Criteo-Uplift",
        train=train, val=val, test=test,
        adj=adj,
        n_users=n_clusters,
        n_items=n_actions,
        n_nodes=n_clusters,        # no separate item nodes in the graph
        context_dim=stats["context_dim"],
        stats=stats,
        item_features=centroids,   # cluster centroids
        uplift_df_path=base / "uplift_estimates.csv",
    )


# ============================================================================
# Convenience: load by name
# ============================================================================

def load_dataset(name: str, **kwargs) -> BanditDataset:
    """
    Load a dataset by name.

    Examples
    --------
    >>> ds = load_dataset("obd-all")
    >>> ds = load_dataset("obd-men")
    >>> ds = load_dataset("criteo")
    """
    name = name.lower().strip()
    if name.startswith("obd"):
        campaign = name.split("-")[1] if "-" in name else "all"
        return load_obd(campaign=campaign, **kwargs)
    elif name.startswith("criteo"):
        return load_criteo(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name!r}. "
                         f"Choose from: obd-all, obd-men, obd-women, criteo")
