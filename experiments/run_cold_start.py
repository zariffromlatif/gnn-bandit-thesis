"""
Cold-start analysis: Evaluate GNN-Bandit and baselines exclusively on cold-start users.

Users with degree 0 in the LightGCN adjacency matrix are considered cold-start.
This script filters the test set to only these users and runs the standard evaluation.

Usage
-----
    python experiments/run_cold_start.py --dataset obd-all --seed 0
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_dataset
from run_main import (
    DEFAULT_CONFIG, train_lightgcn, build_states,
    train_reward_model, train_cate_model, train_gnn_bandit,
    train_baselines, evaluate_all_policies
)


def run_cold_start(dataset_name: str, seed: int, config: dict, output_dir: str = "experiments/results"):
    out_dir = Path(ROOT) / output_dir / dataset_name
    result_file = out_dir / f"coldstart_seed{seed}.json"
    if result_file.exists():
        print(f"\n============================================================\nSkipping Seed {seed} for {dataset_name}: {result_file.name} already exists.\n============================================================\n")
        return None, None
    """Run cold-start analysis."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\nCOLD-START ANALYSIS — {dataset_name} (seed={seed})")
    print("=" * 60)

    dataset = load_dataset(dataset_name, root=str(ROOT))

    # ------------------------------------------------------------------
    # Identify Cold-Start Users
    # ------------------------------------------------------------------
    # The adjacency matrix is a block matrix of size (n_users + n_items, n_users + n_items)
    # The first n_users rows contain edges for users.
    degree = np.asarray(dataset.adj[:dataset.n_users].sum(axis=1)).flatten()
    cold_start_users = np.where(degree == 0)[0]
    
    print(f"  Total users:       {dataset.n_users:,}")
    print(f"  Cold-start users:  {len(cold_start_users):,} ({(len(cold_start_users)/dataset.n_users)*100:.1f}%)")

    # ------------------------------------------------------------------
    # Train Models (on full training set)
    # ------------------------------------------------------------------
    gcn_model = train_lightgcn(dataset, config, device, seed)
    
    states_train = build_states(
        dataset.train.contexts, dataset.train.user_ids, gcn_model, device)
    states_test = build_states(
        dataset.test.contexts, dataset.test.user_ids, gcn_model, device)

    reward_model = train_reward_model(dataset, states_train, config, device)
    cate_model = train_cate_model(dataset, states_train, config, device)
    
    agent = train_gnn_bandit(
        dataset, states_train, config, device, seed,
        gcn_model=gcn_model, cate_model=cate_model
    )
    
    baselines = train_baselines(
        dataset, states_train, gcn_model, config, device, seed
    )

    # ------------------------------------------------------------------
    # Filter Test Set to Cold-Start Users Only
    # ------------------------------------------------------------------
    test = dataset.test
    mask = np.isin(test.user_ids, cold_start_users)
    
    print("\n" + "=" * 60)
    print("FILTERING TEST SET FOR COLD-START EVALUATION")
    print("=" * 60)
    print(f"  Total test samples:       {len(test.user_ids):,}")
    print(f"  Cold-start test samples:  {mask.sum():,} ({(mask.sum()/len(test.user_ids))*100:.1f}%)")
    
    if mask.sum() == 0:
        print("  WARNING: No cold-start users found in the test set. Exiting.")
        return

    test_cs = copy.copy(test)
    test_cs.contexts = test_cs.contexts[mask]
    test_cs.actions = test_cs.actions[mask]
    test_cs.rewards = test_cs.rewards[mask]
    test_cs.propensities = test_cs.propensities[mask]
    test_cs.user_ids = test_cs.user_ids[mask]
    if test_cs.policy_labels is not None:
        test_cs.policy_labels = test_cs.policy_labels[mask]

    dataset_cs = copy.copy(dataset)
    dataset_cs.test = test_cs

    states_test_cs = states_test[mask]

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results = evaluate_all_policies(
        dataset_cs, states_test_cs, gcn_model, agent, baselines,
        reward_model, config, device
    )

    # ------------------------------------------------------------------
    # Save Results
    # ------------------------------------------------------------------
    out_dir = Path(ROOT) / output_dir / f"{dataset_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_cold_start.json"

    json_ready = {}
    for pol_name, metrics in results.items():
        json_ready[pol_name] = {
            k: {"value": v.value, "ci_lower": v.ci_lower, "ci_upper": v.ci_upper}
            for k, v in metrics.items()
        }

    with open(out_path, "w") as f:
        json.dump(json_ready, f, indent=2)

    print(f"\nSaved cold-start results to: {out_path}")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-Bandit cold-start analysis")
    parser.add_argument(
        "--dataset", type=str, default="obd-all",
        choices=["obd-all", "obd-men", "obd-women", "criteo", "all"],
        help="Dataset to evaluate on. (default: obd-all)"
    )
    parser.add_argument(
        "--seeds", type=str, default="0",
        help="Comma-separated random seeds (e.g. '0,1,2,3,4'). (default: 0)"
    )
    parser.add_argument(
        "--output", type=str, default="experiments/results",
        help="Output directory for results. (default: experiments/results)"
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    datasets = ["obd-all", "obd-men", "obd-women"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        for s in seeds:
            run_cold_start(ds, s, DEFAULT_CONFIG, args.output)
