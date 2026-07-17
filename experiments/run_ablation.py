"""
Ablation study: Remove one component at a time, measure DR reward drop.

Ablation variants:
  1. Full GNN-Bandit           → main result (from run_main.py)
  2. No-Graph (MF + BCQ)       → proves LightGCN adds value
  3. No-Constraint (GNN + DQN) → proves BCQ constraint is necessary
  4. No-GNN-No-Constraint      → raw context + DQN (minimal model)
  5. IPW-only evaluation       → proves DR is better than IPW alone

Usage
-----
    python experiments/run_ablation.py --dataset obd-all --seed 0
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

from src.graph.lightgcn import LightGCN
from src.agent.bcq import BCQAgent
from src.baselines.policies import DQNPolicy
from src.utils.data_loader import load_dataset
from src.utils.metrics import RewardModel, evaluate_policy

from run_main import (
    DEFAULT_CONFIG, train_lightgcn, build_states,
    train_reward_model, train_cate_model, train_gnn_bandit,
)


def run_ablation(dataset_name: str, seed: int, config: dict,
                 output_dir: str = "experiments/results"):
    out_dir = Path(ROOT) / output_dir / dataset_name
    result_file = out_dir / f"ablation_seed{seed}.json"
    if result_file.exists():
        print(f"\n============================================================\nSkipping Seed {seed} for {dataset_name}: {result_file.name} already exists.\n============================================================\n")
        return None, None
    """Run all ablation variants."""
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\nABLATION STUDY — {dataset_name} (seed={seed})")
    print("=" * 60)

    dataset = load_dataset(dataset_name, root=str(ROOT))

    # --- Full model (for reference) ---
    gcn_model = train_lightgcn(dataset, config, device, seed)
    states_train = build_states(
        dataset.train.contexts, dataset.train.user_ids, gcn_model, device)
    states_test = build_states(
        dataset.test.contexts, dataset.test.user_ids, gcn_model, device)

    reward_model = train_reward_model(dataset, states_train, config, device)
    cate_model = train_cate_model(dataset, states_train, config, device)
    rm_preds = reward_model.predict(states_test)

    test = dataset.test
    results = {}

    # ------------------------------------------------------------------
    # Variant 1: Full GNN-Bandit (with CATE)
    # ------------------------------------------------------------------
    print("\n--- Ablation 1: Full GNN-Bandit ---")
    agent_full = train_gnn_bandit(dataset, states_train, config, device, seed,
                                  gcn_model=gcn_model, cate_model=cate_model)
    probs = agent_full.action_probabilities(states_test)
    results["Full GNN-Bandit"] = evaluate_policy(
        probs, test.rewards.astype(np.float32), test.propensities,
        test.actions, dataset.n_items, rm_preds,
        label="Full GNN-Bandit",
    )

    # ------------------------------------------------------------------
    # Variant 2: No Graph (raw context → BCQ, no GNN embeddings)
    # ------------------------------------------------------------------
    print("\n--- Ablation 2: No Graph (Context-only BCQ) ---")
    agent_no_graph = BCQAgent(
        state_dim=dataset.context_dim,
        n_actions=dataset.n_items,
        hidden=config["bcq_hidden"],
        threshold_ratio=config["bcq_threshold_ratio"],
        min_actions=config["bcq_min_actions"],
        num_quantiles=config.get("bcq_num_quantiles", 32),
        cvar_alpha=config.get("bcq_cvar_alpha", 0.10),
        device=str(device),
    )
    agent_no_graph.train(
        dataset.train.contexts,   # no GNN embedding
        dataset.train.actions,
        dataset.train.rewards.astype(np.float32),
        n_epochs_bc=config["bcq_epochs_bc"],
        n_epochs_q=config["bcq_epochs_q"],
    )
    # Need a reward model for context-only states too
    rm_no_graph = RewardModel(
        dataset.context_dim, dataset.n_items, device=str(device))
    rm_no_graph.fit(
        dataset.train.contexts, dataset.train.actions,
        dataset.train.rewards.astype(np.float32), n_epochs=config["rm_epochs"],
        verbose=False,
    )
    probs_ng = agent_no_graph.action_probabilities(dataset.test.contexts)
    rm_preds_ng = rm_no_graph.predict(dataset.test.contexts)
    results["No-Graph (BCQ only)"] = evaluate_policy(
        probs_ng, test.rewards.astype(np.float32), test.propensities,
        test.actions, dataset.n_items, rm_preds_ng,
        label="No-Graph (BCQ only)",
    )

    # ------------------------------------------------------------------
    # Variant 3: No Constraint (GNN + DQN, no BCQ threshold)
    # ------------------------------------------------------------------
    print("\n--- Ablation 3: No Constraint (GNN + DQN) ---")
    dqn_gnn = DQNPolicy(
        state_dim=states_train.shape[1],
        n_actions=dataset.n_items,
        device=str(device),
    )
    dqn_gnn.train(
        states_train, dataset.train.actions,
        dataset.train.rewards.astype(np.float32),
        n_epochs=config["dqn_epochs"],
    )
    probs_dqn = dqn_gnn.action_probabilities(states_test)
    results["No-Constraint (GNN+DQN)"] = evaluate_policy(
        probs_dqn, test.rewards.astype(np.float32), test.propensities,
        test.actions, dataset.n_items, rm_preds,
        label="No-Constraint (GNN+DQN)",
    )

    # ------------------------------------------------------------------
    # Variant 4: No GNN + No Constraint (Context → DQN)
    # ------------------------------------------------------------------
    print("\n--- Ablation 4: Minimal (Context + DQN) ---")
    dqn_ctx = DQNPolicy(
        state_dim=dataset.context_dim,
        n_actions=dataset.n_items,
        device=str(device),
    )
    dqn_ctx.train(
        dataset.train.contexts, dataset.train.actions,
        dataset.train.rewards.astype(np.float32),
        n_epochs=config["dqn_epochs"],
    )
    probs_min = dqn_ctx.action_probabilities(dataset.test.contexts)
    results["Minimal (Context+DQN)"] = evaluate_policy(
        probs_min, test.rewards.astype(np.float32), test.propensities,
        test.actions, dataset.n_items, rm_preds_ng,
        label="Minimal (Context+DQN)",
    )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY (DR Estimate)")
    print("=" * 60)
    print(f"{'Variant':<30s} {'DR Value':>12s} {'95% CI':>25s}")
    print("-" * 67)
    for variant, ope in results.items():
        if "DR" in ope:
            dr = ope["DR"]
            print(f"{variant:<30s} {dr.value:>12.6f} "
                  f"[{dr.ci_lower:.6f}, {dr.ci_upper:.6f}]")
        else:
            ipw = ope.get("IPW")
            if ipw:
                print(f"{variant:<30s} {ipw.value:>12.6f} (IPW)")

    # Save
    out_dir = Path(ROOT) / output_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    serialised = {}
    for variant, ope in results.items():
        serialised[variant] = {
            name: {"value": r.value, "std": r.std,
                   "ci_lower": r.ci_lower, "ci_upper": r.ci_upper}
            for name, r in ope.items()
        }
    with open(out_dir / f"ablation_seed{seed}.json", "w") as f:
        json.dump(serialised, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nAblation complete ({elapsed:.1f}s)")
    return results


def main():
    parser = argparse.ArgumentParser(description="GNN-Bandit ablation study")
    parser.add_argument("--dataset", type=str, default="obd-all")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated random seeds (e.g. '0,1,2,3,4').")
    parser.add_argument("--output", type=str, default="experiments/results")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    
    for seed in seeds:
        run_ablation(args.dataset, seed, DEFAULT_CONFIG, args.output)


if __name__ == "__main__":
    main()
