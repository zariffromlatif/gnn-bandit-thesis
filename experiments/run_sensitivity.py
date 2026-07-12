"""
Sensitivity analysis: sweep key hyperparameters and report DR reward.

Sweeps:
  1. LightGCN embedding dimension:  [16, 32, 64, 128]
  2. LightGCN number of layers:     [1, 2, 3, 4]
  3. BCQ constraint threshold tau:  [0.01, 0.03, 0.05, 0.10, 0.20]

Usage
-----
    python experiments/run_sensitivity.py --dataset obd-all --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_dataset
from src.utils.metrics import RewardModel, evaluate_policy

from run_main import (
    DEFAULT_CONFIG, train_lightgcn, build_states,
    train_reward_model, train_cate_model, train_gnn_bandit,
)


def sweep_one(
    dataset, config, device, seed,
    param_name: str, param_values: list,
    config_key: str,
):
    """Sweep a single hyperparameter, holding everything else fixed."""
    results = {}
    for val in param_values:
        print(f"\n{'~' * 50}")
        print(f"  {param_name} = {val}")
        print(f"{'~' * 50}")

        cfg = config.copy()
        cfg[config_key] = val

        torch.manual_seed(seed)
        np.random.seed(seed)

        gcn = train_lightgcn(dataset, cfg, device, seed)
        s_train = build_states(
            dataset.train.contexts, dataset.train.user_ids, gcn, device)
        s_test = build_states(
            dataset.test.contexts, dataset.test.user_ids, gcn, device)

        rm = train_reward_model(dataset, s_train, cfg, device)
        cate = train_cate_model(dataset, s_train, cfg, device)
        agent = train_gnn_bandit(dataset, s_train, cfg, device, seed,
                                 gcn_model=gcn, cate_model=cate)

        probs = agent.action_probabilities(s_test)
        rm_preds = rm.predict(s_test)
        test = dataset.test

        ope = evaluate_policy(
            probs, test.rewards.astype(np.float32), test.propensities,
            test.actions, dataset.n_items, rm_preds,
            label=f"{param_name}={val}",
        )

        dr = ope.get("DR")
        results[val] = {
            "DR_value": dr.value if dr else None,
            "DR_ci_lower": dr.ci_lower if dr else None,
            "DR_ci_upper": dr.ci_upper if dr else None,
        }

    return results


def run_sensitivity(dataset_name: str, seed: int, config: dict,
                    output_dir: str = "experiments/results"):
    """Run all sensitivity sweeps."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nSENSITIVITY ANALYSIS — {dataset_name} (seed={seed})")
    print("=" * 60)

    dataset = load_dataset(dataset_name, root=str(ROOT))
    all_results = {}

    # Sweep 1: Embedding dimension
    print("\n" + "#" * 60)
    print("SWEEP 1: Embedding Dimension")
    print("#" * 60)
    all_results["embed_dim"] = sweep_one(
        dataset, config, device, seed,
        param_name="embed_dim",
        param_values=[16, 32, 64, 128],
        config_key="gcn_embed_dim",
    )

    # Sweep 2: Number of GNN layers
    print("\n" + "#" * 60)
    print("SWEEP 2: GNN Layers")
    print("#" * 60)
    all_results["n_layers"] = sweep_one(
        dataset, config, device, seed,
        param_name="n_layers",
        param_values=[1, 2, 3, 4],
        config_key="gcn_n_layers",
    )

    # Sweep 3: BCQ threshold ratio (tau = ratio / n_actions)
    print("\n" + "#" * 60)
    print("SWEEP 3: BCQ Threshold Ratio")
    print("#" * 60)
    all_results["bcq_threshold_ratio"] = sweep_one(
        dataset, config, device, seed,
        param_name="threshold_ratio",
        param_values=[0.1, 0.3, 0.5, 1.0, 2.0],
        config_key="bcq_threshold_ratio",
    )

    # Sweep 4: Risk-Aversion (CVaR Alpha)
    print("\n" + "#" * 60)
    print("SWEEP 4: CVaR Alpha (Risk-Aversion)")
    print("#" * 60)
    all_results["cvar_alpha"] = sweep_one(
        dataset, config, device, seed,
        param_name="cvar_alpha",
        param_values=[0.05, 0.10, 0.25, 0.50, 1.0],
        config_key="bcq_cvar_alpha",
    )

    # Save
    out_dir = Path(ROOT) / output_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy/int keys to strings for JSON
    serialisable = {}
    for sweep_name, sweep_results in all_results.items():
        serialisable[sweep_name] = {
            str(k): v for k, v in sweep_results.items()
        }

    with open(out_dir / f"sensitivity_seed{seed}.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"SENSITIVITY ANALYSIS COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    # Summary
    for sweep_name, sweep_results in all_results.items():
        print(f"\n{sweep_name}:")
        for val, res in sweep_results.items():
            dr = res.get("DR_value", "N/A")
            if isinstance(dr, float):
                print(f"  {val:>8} → DR = {dr:.6f}")
            else:
                print(f"  {val:>8} → DR = {dr}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="GNN-Bandit sensitivity analysis")
    parser.add_argument("--dataset", type=str, default="obd-all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="experiments/results")
    args = parser.parse_args()

    run_sensitivity(args.dataset, args.seed, DEFAULT_CONFIG, args.output)


if __name__ == "__main__":
    main()
