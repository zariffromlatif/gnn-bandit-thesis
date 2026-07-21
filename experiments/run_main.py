"""
Main experiment: GNN-Bandit vs. all baselines on OBD and Criteo.

Runs the full pipeline:
  1. Load data
  2. Train LightGCN (graph embeddings)
  3. Train reward model (for DM / DR estimators)
  4. Train GNN-Bandit (BCQ with GNN embeddings)
  5. Train all baselines
  6. Evaluate all policies with OPE (IPW, SNIPW, DM, DR)
  7. Run Sleeping Dogs analysis
  8. Save results

Usage
-----
    python experiments/run_main.py --dataset obd-all --seed 0
    python experiments/run_main.py --dataset criteo --seed 0
    python experiments/run_main.py --dataset all --seeds 0,1,2,3,4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.lightgcn import LightGCN
from src.agent.bcq import BCQAgent
from src.causal.cate_estimator import CATEEstimator
from src.baselines.policies import (
    RandomPolicy, BTSPolicy, DQNPolicy,
    MFBanditPolicy, GreedyGNNPolicy, UpliftPolicy,
    LinUCBPolicy, NeuralUCBPolicy, CQLPolicy, IQLPolicy,
)
from src.ope.estimators import evaluate_all
from src.utils.data_loader import load_dataset
from src.utils.metrics import RewardModel, evaluate_policy, sleeping_dogs_analysis


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # LightGCN
    "gcn_embed_dim":  64,
    "gcn_n_layers":   3,
    "gcn_lr":         1e-3,
    "gcn_epochs":     100,
    "gcn_batch_size": 16384,
    "gcn_reg":        1e-4,

    # BCQ
    "bcq_hidden":          256,
    "bcq_n_hidden":        2,
    "bcq_threshold_ratio": 0.3,     # tau = 0.3 / n_actions (adaptive)
    "bcq_min_actions":     5,       # safety floor: at least 5 actions survive
    "bcq_lr":              1e-3,
    "bcq_epochs_bc":       30,
    "bcq_epochs_q":        100,     # more training for better Q-value discrimination
    "bcq_batch_size":      16384,
    "bcq_temperature":     0.1,     # concentrated policy (matches Greedy-GNN scale)
    "bcq_hybrid_weight":   1.0,     # weight for GNN dot-product in hybrid scoring
    "bcq_num_quantiles":   32,      # number of quantiles for Distributional RL
    "bcq_cvar_alpha":      0.10,    # risk aversion parameter (0.10 = optimize for worst 10%)

    # CATE estimator
    "cate_hidden":        128,
    "cate_epochs":        50,
    "cate_uplift_weight": 0.5,     # blend: (1-w)*reward + w*cate

    # Reward model
    "rm_hidden":   128,
    "rm_epochs":   30,

    # OPE
    "ope_clip":  100.0,

    # Baselines
    "dqn_epochs":  50,
    "mf_epochs":   30,
    "cql_alpha":   1.0,            # CQL conservatism weight
    "iql_expectile": 0.7,          # IQL expectile parameter
}


# ============================================================================
# Pipeline steps
# ============================================================================

def train_lightgcn(dataset, config, device, seed):
    """Step 1: Train LightGCN and produce user embeddings."""
    print("\n" + "=" * 60)
    print("STEP 1: Training LightGCN")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LightGCN(
        n_nodes=dataset.n_nodes,
        embed_dim=config["gcn_embed_dim"],
        n_layers=config["gcn_n_layers"],
        adj=dataset.adj,
        n_users=dataset.n_users,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["gcn_lr"])
    rng = np.random.RandomState(seed)

    if "Criteo" in dataset.name:
        # For Criteo (user-user graph), train on graph edges (link prediction)
        adj_coo = dataset.adj.tocoo()
        train_users = adj_coo.row
        train_items = adj_coo.col
        n_items_for_neg = dataset.n_users
    else:
        # Training data: positive interactions from train split
        train = dataset.train
        pos_mask = train.rewards > 0
        train_users = train.user_ids[pos_mask]
        train_items = train.actions[pos_mask]

        if len(train_users) == 0:
            print("  WARNING: No positive interactions in training data!")
            print("  Using all interactions instead.")
            train_users = train.user_ids
            train_items = train.actions
        n_items_for_neg = dataset.n_items

    print(f"  Positive interactions for BPR: {len(train_users):,}")
    print(f"  Graph nodes: {dataset.n_nodes}, Embed dim: {config['gcn_embed_dim']}")

    for epoch in range(config["gcn_epochs"]):
        model.train()
        perm = rng.permutation(len(train_users))
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, len(train_users), config["gcn_batch_size"]):
            idx = perm[start:start + config["gcn_batch_size"]]
            u = torch.LongTensor(train_users[idx]).to(device)
            p = torch.LongTensor(train_items[idx]).to(device)
            neg = LightGCN.sample_negatives(
                train_users[idx], train_items[idx], n_items_for_neg, rng
            )
            n = torch.LongTensor(neg).to(device)

            loss = model.bpr_loss(u, p, n, reg_weight=config["gcn_reg"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}  BPR loss: {total_loss/n_batches:.4f}")

    model.eval()
    print("  LightGCN training complete.")
    return model


def build_states(contexts, user_ids, gcn_model, device):
    """Concatenate context features with GNN user embeddings → full state."""
    user_emb = gcn_model.encode_users(user_ids)   # (N, K)
    return np.hstack([contexts, user_emb]).astype(np.float32)


def train_reward_model(dataset, states_train, config, device):
    """Step 2: Train reward model for DM / DR estimators."""
    print("\n" + "=" * 60)
    print("STEP 2: Training Reward Model")
    print("=" * 60)

    rm = RewardModel(
        state_dim=states_train.shape[1],
        n_actions=dataset.n_items,
        hidden=config["rm_hidden"],
        device=str(device),
    )
    rm.fit(
        states_train,
        dataset.train.actions,
        dataset.train.rewards.astype(np.float32),
        n_epochs=config["rm_epochs"],
    )
    return rm


def train_cate_model(dataset, states_train, config, device):
    """Step 2b: Train CATE estimator for treatment effect estimation."""
    print("\n" + "=" * 60)
    print("STEP 2b: Training CATE Estimator")
    print("=" * 60)

    cate = CATEEstimator(
        state_dim=states_train.shape[1],
        n_actions=dataset.n_items,
        hidden=config["cate_hidden"],
        device=str(device),
    )

    # Prefer precomputed uplift table (from randomised data)
    if dataset.uplift_df_path and dataset.uplift_df_path.exists():
        import pandas as pd
        df = pd.read_csv(dataset.uplift_df_path)
        uplift_table = np.zeros(
            (dataset.n_users, dataset.n_items), dtype=np.float32)
        if "Criteo" in dataset.name:
            for _, row in df.iterrows():
                uid = int(row["cluster_id"])
                if uid < dataset.n_users:
                    uplift_table[uid, 1] = float(row["uplift_conv"])
        else:
            for _, row in df.iterrows():
                uid = int(row["user_id"])
                iid = int(row["item_id"])
                if uid < dataset.n_users and iid < dataset.n_items:
                    uplift_table[uid, iid] = float(row["uplift"])

        print(f"  Using precomputed uplift table: {uplift_table.shape}")
        print(f"  Non-zero entries: {(uplift_table != 0).sum():,}")
        cate.fit_from_uplift_table(
            states_train, dataset.train.user_ids,
            uplift_table, n_epochs=config["cate_epochs"],
        )
    else:
        # Fall back to outcome-based estimation (e.g. Criteo)
        print("  No uplift table — training from outcomes")
        cate.fit_from_outcomes(
            states_train, dataset.train.actions,
            dataset.train.rewards.astype(np.float32),
            n_epochs=config["cate_epochs"],
        )

    # Print segmentation summary
    seg = cate.segment_users(
        states_train, dataset.train.user_ids)
    print(f"  User segments: {seg['segment_counts']}")

    return cate


def train_gnn_bandit(dataset, states_train, config, device, seed,
                     gcn_model=None, cate_model=None):
    """Step 3: Train the GNN-Bandit (BCQ with CATE-weighted rewards)."""
    print("\n" + "=" * 60)
    print("STEP 3: Training GNN-Bandit (BCQ)")
    print("=" * 60)

    torch.manual_seed(seed)

    # Extract item embeddings for hybrid scoring
    item_emb = None
    if gcn_model is not None and dataset.n_nodes != dataset.n_users:
        with torch.no_grad():
            item_emb = gcn_model.get_item_embeddings().cpu().numpy()
        print(f"  Hybrid scoring enabled: item_emb {item_emb.shape}")
    else:
        print("  Hybrid scoring disabled (no explicit item nodes in graph)")

    # Compute uplift-weighted rewards if CATE model available
    raw_rewards = dataset.train.rewards.astype(np.float32)
    if cate_model is not None:
        uplift_weight = config.get("cate_uplift_weight", 0.5)
        train_rewards = cate_model.uplift_weighted_rewards(
            states_train, dataset.train.actions, raw_rewards,
            uplift_weight=uplift_weight,
        )
        print(f"  Uplift-weighted rewards: weight={uplift_weight:.2f}, "
              f"mean={train_rewards.mean():.6f} (raw={raw_rewards.mean():.6f})")
    else:
        train_rewards = raw_rewards

    agent = BCQAgent(
        state_dim=states_train.shape[1],
        n_actions=dataset.n_items,
        hidden=config["bcq_hidden"],
        n_hidden=config["bcq_n_hidden"],
        threshold_ratio=config["bcq_threshold_ratio"],
        min_actions=config["bcq_min_actions"],
        lr=config["bcq_lr"],
        temperature=config["bcq_temperature"],
        item_embeddings=item_emb,
        gnn_embed_dim=config["gcn_embed_dim"],
        hybrid_weight=config.get("bcq_hybrid_weight", 1.0),
        num_quantiles=config.get("bcq_num_quantiles", 32),
        cvar_alpha=config.get("bcq_cvar_alpha", 0.10),
        device=str(device),
    )
    agent.train(
        states_train,
        dataset.train.actions,
        train_rewards,
        n_epochs_bc=config["bcq_epochs_bc"],
        n_epochs_q=config["bcq_epochs_q"],
        batch_size=config["bcq_batch_size"],
    )
    return agent


def train_baselines(dataset, states_train, gcn_model, config, device, seed):
    """Step 4: Train all baseline policies."""
    print("\n" + "=" * 60)
    print("STEP 4: Training Baselines")
    print("=" * 60)

    torch.manual_seed(seed)
    baselines = {}

    # 1. Random
    baselines["Random"] = RandomPolicy(dataset.n_items)
    print("  [ 1/10] Random policy ready.")

    # 2. BTS (Thompson Sampling)
    baselines["BTS"] = BTSPolicy(
        dataset.n_items,
        default_propensity=1.0 / dataset.n_items,
    )
    print("  [ 2/10] BTS policy ready.")

    # 3. LinUCB (standard contextual bandit)
    print("  [ 3/10] Training LinUCB ...")
    linucb = LinUCBPolicy(
        state_dim=states_train.shape[1],
        n_actions=dataset.n_items,
    )
    linucb.train(
        states_train,
        dataset.train.actions,
        dataset.train.rewards.astype(np.float32),
    )
    baselines["LinUCB"] = linucb

    # 4. NeuralUCB (neural contextual bandit)
    print("  [ 4/10] Training NeuralUCB ...")
    nucb = NeuralUCBPolicy(
        state_dim=states_train.shape[1],
        n_actions=dataset.n_items,
        device=str(device),
    )
    nucb.train(
        states_train,
        dataset.train.actions,
        dataset.train.rewards.astype(np.float32),
        n_epochs=config["dqn_epochs"],
    )
    baselines["NeuralUCB"] = nucb

    # 5. DQN (no batch constraint)
    print("  [ 5/10] Training DQN ...")
    dqn = DQNPolicy(
        state_dim=states_train.shape[1],
        n_actions=dataset.n_items,
        device=str(device),
    )
    dqn.train(
        states_train,
        dataset.train.actions,
        dataset.train.rewards.astype(np.float32),
        n_epochs=config["dqn_epochs"],
        verbose=True,
    )
    baselines["DQN"] = dqn

    # 6. CQL (conservative Q-learning)
    print("  [ 6/10] Training CQL ...")
    cql = CQLPolicy(
        state_dim=states_train.shape[1],
        n_actions=dataset.n_items,
        cql_alpha=config.get("cql_alpha", 1.0),
        device=str(device),
    )
    cql.train(
        states_train,
        dataset.train.actions,
        dataset.train.rewards.astype(np.float32),
        n_epochs=config["dqn_epochs"],
    )
    baselines["CQL"] = cql

    # 7. IQL (implicit Q-learning)
    print("  [ 7/10] Training IQL ...")
    iql = IQLPolicy(
        state_dim=states_train.shape[1],
        n_actions=dataset.n_items,
        expectile=config.get("iql_expectile", 0.7),
        device=str(device),
    )
    iql.train(
        states_train,
        dataset.train.actions,
        dataset.train.rewards.astype(np.float32),
        n_epochs=config["dqn_epochs"],
    )
    baselines["IQL"] = iql

    # 8. MF + Bandit (no graph propagation)
    print("  [ 8/10] Training MF-Bandit ...")
    mf = MFBanditPolicy(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embed_dim=config["gcn_embed_dim"],
        state_dim=dataset.context_dim,
        device=str(device),
    )
    pos_mask = dataset.train.rewards > 0
    if pos_mask.any():
        mf.train_mf(
            dataset.train.user_ids[pos_mask],
            dataset.train.actions[pos_mask],
            n_epochs=config["mf_epochs"],
        )
    mf.train_policy(
        dataset.train.contexts,
        dataset.train.user_ids,
        dataset.train.actions,
        dataset.train.rewards.astype(np.float32),
    )
    baselines["MF-Bandit"] = mf

    # 9. Greedy GNN (no RL)
    baselines["Greedy-GNN"] = GreedyGNNPolicy(dataset.n_items)
    print("  [ 9/10] Greedy-GNN policy ready.")

    # 10. Uplift-only
    print("  [10/10] Loading Uplift policy ...")
    uplift_pol = UpliftPolicy(dataset.n_users, dataset.n_items)
    if dataset.uplift_df_path and dataset.uplift_df_path.exists():
        uplift_pol.load_uplift(str(dataset.uplift_df_path))
    baselines["Uplift-Only"] = uplift_pol

    return baselines


def evaluate_all_policies(
    dataset, states_test, gcn_model, agent, baselines,
    reward_model, config, device,
):
    """Step 5: Evaluate everything with OPE."""
    print("\n" + "=" * 60)
    print("STEP 5: Off-Policy Evaluation")
    print("=" * 60)

    test = dataset.test
    rm_preds = reward_model.predict(states_test)   # (N, A)
    all_results = {}

    # --- GNN-Bandit (ours) ---
    probs = agent.action_probabilities(states_test)
    all_results["GNN-Bandit"] = evaluate_policy(
        probs, test.rewards.astype(np.float32), test.propensities,
        test.actions, dataset.n_items, rm_preds,
        clip=config["ope_clip"], label="GNN-Bandit (Ours)",
    )

    # --- Baselines ---
    for name, policy in baselines.items():
        if name == "BTS":
            probs = policy.action_probabilities(
                states_test,
                logged_actions=test.actions,
                logged_propensities=test.propensities,
            )
        elif name == "Greedy-GNN":
            if dataset.n_nodes == dataset.n_users:
                probs = np.ones((len(test.user_ids), dataset.n_items)) / dataset.n_items
            else:
                user_emb = gcn_model.encode_users(test.user_ids)
                with torch.no_grad():
                    item_emb = gcn_model.get_item_embeddings().cpu().numpy()
                probs = policy.action_probabilities(user_emb, item_emb)
        elif name == "MF-Bandit":
            probs = policy.action_probabilities(
                test.contexts, user_ids=test.user_ids,
            )
        elif name == "Uplift-Only":
            probs = policy.action_probabilities(
                states_test, user_ids=test.user_ids,
            )
        else:
            probs = policy.action_probabilities(states_test)

        all_results[name] = evaluate_policy(
            probs, test.rewards.astype(np.float32), test.propensities,
            test.actions, dataset.n_items, rm_preds,
            clip=config["ope_clip"], label=name,
        )

    return all_results


def run_sleeping_dogs(dataset, states_test, agent, baselines, gcn_model):
    """Step 6: Sleeping Dogs analysis."""
    print("\n" + "=" * 60)
    print("STEP 6: Sleeping Dogs Analysis")
    print("=" * 60)

    if dataset.uplift_df_path is None or not dataset.uplift_df_path.exists():
        print("  Skipping — no uplift data available.")
        return {}

    # Build uplift table
    import pandas as pd
    df = pd.read_csv(dataset.uplift_df_path)
    uplift_table = np.zeros((dataset.n_users, dataset.n_items), dtype=np.float32)
    
    if "Criteo" in dataset.name or "cluster_id" in df.columns:
        for _, row in df.iterrows():
            uid = int(row["cluster_id"])
            if uid < dataset.n_users:
                uplift_table[uid, 1] = float(row["uplift_conv"])
    else:
        for _, row in df.iterrows():
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            if uid < dataset.n_users and iid < dataset.n_items:
                uplift_table[uid, iid] = float(row["uplift"])

    test = dataset.test
    results = {}

    # GNN-Bandit
    probs = agent.action_probabilities(states_test)
    results["GNN-Bandit"] = sleeping_dogs_analysis(
        probs, test.user_ids, uplift_table, dataset.n_items,
    )

    # Random baseline
    probs_rand = baselines["Random"].action_probabilities(states_test)
    results["Random"] = sleeping_dogs_analysis(
        probs_rand, test.user_ids, uplift_table, dataset.n_items,
    )

    # Print comparison
    for name, res in results.items():
        print(f"\n  {name}:")
        print(f"    Sleeping Dogs:  {res['n_sleeping_dog']:,} users | "
              f"avg intervention prob: {res['avg_max_prob_sleeping_dog']:.4f}")
        print(f"    Persuadables:   {res['n_persuadable']:,} users | "
              f"avg intervention prob: {res['avg_max_prob_persuadable']:.4f}")

    return results


# ============================================================================
# Main pipeline
# ============================================================================

def run_experiment(dataset_name: str, seed: int, config: dict,
                   output_dir: str = "experiments/results"):
    out_dir = Path(ROOT) / output_dir / dataset_name
    result_file = out_dir / f"results_seed{seed}.json"
    if result_file.exists():
        print(f"\n============================================================\nSkipping Seed {seed} for {dataset_name}: {result_file.name} already exists.\n============================================================\n")
        return None, None
    """Run the full GNN-Bandit experiment pipeline."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Seed: {seed}")

    # Seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 0. Load data
    print("\nLoading dataset ...")
    dataset = load_dataset(dataset_name, root=str(ROOT))
    print(f"  {dataset.name}: {dataset.n_users} users, "
          f"{dataset.n_items} actions, {dataset.context_dim}-dim context")
    print(f"  Train: {len(dataset.train.contexts):,} | "
          f"Val: {len(dataset.val.contexts):,} | "
          f"Test: {len(dataset.test.contexts):,}")

    # 1. Train LightGCN
    gcn_model = train_lightgcn(dataset, config, device, seed)

    # Build augmented states (context + GNN embedding)
    print("\n  Building augmented states ...")
    states_train = build_states(
        dataset.train.contexts, dataset.train.user_ids, gcn_model, device)
    states_val = build_states(
        dataset.val.contexts, dataset.val.user_ids, gcn_model, device)
    states_test = build_states(
        dataset.test.contexts, dataset.test.user_ids, gcn_model, device)
    print(f"  State dim: {states_train.shape[1]} "
          f"({dataset.context_dim} context + {config['gcn_embed_dim']} GNN)")

    # 2. Train reward model
    reward_model = train_reward_model(dataset, states_train, config, device)

    # 2b. Train CATE estimator (causal uplift estimation)
    cate_model = train_cate_model(dataset, states_train, config, device)

    # 3. Train GNN-Bandit (with CATE-weighted rewards + hybrid scoring)
    agent = train_gnn_bandit(dataset, states_train, config, device, seed,
                             gcn_model=gcn_model, cate_model=cate_model)

    # 4. Train baselines
    baselines = train_baselines(
        dataset, states_train, gcn_model, config, device, seed)

    # 5. Evaluate
    ope_results = evaluate_all_policies(
        dataset, states_test, gcn_model, agent, baselines,
        reward_model, config, device,
    )

    # 6. Sleeping Dogs
    sd_results = run_sleeping_dogs(
        dataset, states_test, agent, baselines, gcn_model)

    # 7. Save results
    out_dir = Path(ROOT) / output_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Serialise OPE results
    ope_serialised = {}
    for method, estimators in ope_results.items():
        ope_serialised[method] = {
            est_name: {
                "value": res.value,
                "std":   res.std,
                "ci_lower": res.ci_lower,
                "ci_upper": res.ci_upper,
                "n": res.n,
            }
            for est_name, res in estimators.items()
        }

    import gc
    del gcn_model, reward_model, cate_model, agent, baselines
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result_file = out_dir / f"results_seed{seed}.json"
    with open(result_file, "w") as f:
        json.dump({
            "dataset":          dataset_name,
            "seed":             seed,
            "config":           config,
            "ope_results":      ope_serialised,
            "sleeping_dogs":    sd_results,
            "elapsed_seconds":  time.time() - start_time,
        }, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT COMPLETE  ({time.time() - start_time:.1f}s)")
    print(f"Results saved to: {result_file}")
    print(f"{'=' * 60}")

    return ope_results, sd_results


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GNN-Bandit main experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="obd-all",
                        choices=["obd-all", "obd-men", "obd-women",
                                 "criteo", "all"],
                        help="Dataset to evaluate on.")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma-separated random seeds (e.g. '0,1,2,3,4').")
    parser.add_argument("--output", type=str, default="experiments/results",
                        help="Output directory for results.")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    datasets = (["obd-all", "obd-men", "obd-women", "criteo"]
                if args.dataset == "all" else [args.dataset])

    config = DEFAULT_CONFIG.copy()

    for ds in datasets:
        for seed in seeds:
            print(f"\n{'#' * 60}")
            print(f"# Dataset: {ds}  |  Seed: {seed}")
            print(f"{'#' * 60}")
            run_experiment(ds, seed, config, args.output)


if __name__ == "__main__":
    main()
