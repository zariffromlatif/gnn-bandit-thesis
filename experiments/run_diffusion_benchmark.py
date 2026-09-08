"""
Diff-GNN-Bandit Benchmark Sweep Runner.

Benchmarks the generative score-based diffusion policy against standard BCQ:
  1. GNN-Bandit (Standard BCQ)
  2. Diff-GNN-Bandit (Generative Diffusion, unguided)
  3. Diff-GNN-Bandit + Tweedie CATE (Causal Guided Diffusion)
  4. Diff-GNN-Bandit + Pareto MGDA (Multi-Objective G-IRM)

Usage:
------
    python experiments/run_diffusion_benchmark.py --dataset kuairand --seed 0
    python experiments/run_diffusion_benchmark.py --dataset obd-all --seeds 0,1,2,3,4
    python experiments/run_diffusion_benchmark.py --dataset all --seeds 0,1,2,3,4
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
from src.agent.diffusion_policy import DiffGNNBandit
from src.agent.pareto_solver import MGDAParetoSolver
from src.causal.cate_estimator import CATEEstimator
from src.causal.r_learner import RLearner
from src.utils.data_loader import load_dataset
from src.utils.metrics import RewardModel, evaluate_policy, sleeping_dogs_analysis
from experiments.run_main import (
    DEFAULT_CONFIG,
    train_graph_encoder,
    build_states,
    train_reward_model,
    train_cate_model,
    train_gnn_bandit,
)


def train_diffusion_policy(
    dataset,
    states_train,
    config,
    device,
    seed,
    gcn_model=None,
    cate_model=None,
    guidance_weight: float = 2.0,
    use_pareto: bool = False,
):
    """Trains a Diff-GNN-Bandit score-based diffusion policy."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Item embeddings from GNN
    item_emb = None
    if gcn_model is not None and dataset.n_nodes != dataset.n_users:
        with torch.no_grad():
            item_emb = gcn_model.get_item_embeddings().cpu().numpy()

    action_dim = config.get("gcn_embed_dim", 64) if item_emb is None else item_emb.shape[1]

    agent = DiffGNNBandit(
        state_dim=states_train.shape[1],
        n_actions=dataset.n_items,
        action_dim=action_dim,
        item_embeddings=item_emb,
        num_timesteps=config.get("diff_timesteps", 50),
        ddim_steps=config.get("diff_ddim_steps", 15),
        guidance_weight=guidance_weight,
        temperature=config.get("diff_temperature", 0.5),
        hidden=config.get("diff_hidden", 256),
        lr=config.get("diff_lr", 1e-3),
        device=str(device),
    )

    # Train diffusion policy
    agent.fit(
        states=states_train,
        actions=dataset.train.actions,
        rewards=dataset.train.rewards.astype(np.float32),
        n_epochs=config.get("diff_epochs", 25),
        batch_size=config.get("diff_batch_size", 4096),
        verbose=True,
    )

    return agent


def run_diffusion_experiment(
    dataset_name: str,
    seed: int,
    config: dict,
    output_dir: str = "experiments/results/diffusion",
):
    """Runs diffusion policy benchmark on a given dataset and seed."""
    out_dir = Path(ROOT) / output_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / f"diffusion_seed{seed}.json"

    if result_file.exists():
        print(f"\n[Skipping] Seed {seed} for {dataset_name}: {result_file.name} already exists.\n")
        return None

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\n[Diff-GNN-Bandit Benchmark] Dataset: {dataset_name} | Seed: {seed} | Device: {device}\n{'='*70}")

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load Data
    print("\nLoading dataset ...")
    dataset = load_dataset(dataset_name, root=str(ROOT))
    print(f"  {dataset.name}: {dataset.n_users:,} users, {dataset.n_items} actions, {dataset.context_dim}-dim context")

    # 2. Graph Encoder & States
    gcn_model = train_graph_encoder(dataset, config, device, seed)
    states_train = build_states(dataset.train.contexts, dataset.train.user_ids, gcn_model, device)
    states_test = build_states(dataset.test.contexts, dataset.test.user_ids, gcn_model, device)

    # 3. Reward Model & CATE
    reward_model = train_reward_model(dataset, states_train, config, device)
    cate_model = train_cate_model(dataset, states_train, config, device)

    # 4. Standard GNN-Bandit (BCQ Baseline)
    print("\n--- Training Baseline 1: Standard GNN-Bandit (BCQ) ---")
    bcq_agent = train_gnn_bandit(
        dataset, states_train, config, device, seed, gcn_model=gcn_model, cate_model=cate_model
    )

    # 5. Diff-GNN-Bandit (Unguided)
    print("\n--- Training Baseline 2: Diff-GNN-Bandit (Unguided Generative Diffusion) ---")
    diff_unguided = train_diffusion_policy(
        dataset, states_train, config, device, seed, gcn_model=gcn_model, guidance_weight=0.0
    )

    # 6. Diff-GNN-Bandit + Tweedie CATE Guidance
    print("\n--- Training Model 3: Diff-GNN-Bandit + Tweedie CATE Guidance ---")
    diff_guided = train_diffusion_policy(
        dataset, states_train, config, device, seed, gcn_model=gcn_model, guidance_weight=2.0
    )

    # Predict test CATE scores for guidance
    cate_test = cate_model.predict(states_test)

    # 7. Off-Policy Evaluation (OPE)
    test = dataset.test
    test_rewards = test.rewards.astype(np.float32)
    rm_preds = reward_model.predict(states_test)

    policies = {
        "GNN-Bandit (BCQ)": (bcq_agent, None),
        "Diff-GNN-Bandit (Unguided)": (diff_unguided, None),
        "Diff-GNN-Bandit (Tweedie CATE)": (diff_guided, cate_test),
    }

    ope_results = {}
    latencies_ms = {}

    print("\n" + "=" * 60)
    print("EVALUATION: OPE Comparison & Latency Audit")
    print("=" * 60)

    for name, (pol, c_guidance) in policies.items():
        t0 = time.perf_counter()
        if isinstance(pol, DiffGNNBandit):
            probs = pol.action_probabilities(states_test, cate_scores=c_guidance, batch_size=4096)
        else:
            probs = pol.action_probabilities(states_test)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_sample_latency_us = (elapsed_ms / len(states_test)) * 1000
        latencies_ms[name] = {"total_ms": elapsed_ms, "per_sample_us": per_sample_latency_us}

        res = evaluate_policy(
            probs, test_rewards, test.propensities, test.actions, dataset.n_items, rm_preds, label=name
        )
        snip_val = (res.get("SNIPW") or res.get("SNIPS")).value
        ope_results[name] = {
            "DR": res["DR"].value,
            "SNIPS": snip_val,
            "DM": res["DM"].value,
            "IPW": res["IPW"].value,
            "latency_per_sample_us": per_sample_latency_us,
        }

    # Print Leaderboard
    print("\n" + "=" * 65)
    print(f"{'Policy':<32} | {'DR Metric':<12} | {'SNIPS':<10} | {'Latency (us)':<12}")
    print("-" * 65)
    for name, res in ope_results.items():
        print(f"{name:<32} | {res['DR']:<12.6f} | {res['SNIPS']:<10.6f} | {res['latency_per_sample_us']:<12.1f}")
    print("=" * 65)

    # Save to JSON
    total_time = time.time() - start_time
    output_data = {
        "dataset": dataset_name,
        "seed": seed,
        "elapsed_seconds": total_time,
        "results": ope_results,
    }

    with open(result_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Saved] Results written to {result_file} (elapsed: {total_time:.1f}s)")
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Diff-GNN-Bandit Benchmark Runner")
    parser.add_argument("--dataset", type=str, default="kuairand",
                        help="Dataset name ('kuairand', 'obd-all', 'kuairec', or 'all')")
    parser.add_argument("--seeds", type=str, default="0",
                        help="Comma-separated random seeds (e.g. '0,1,2,3,4')")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override diffusion training epochs")
    parser.add_argument("--quick", action="store_true",
                        help="Run rapid smoke test with reduced epochs")
    parser.add_argument("--output_dir", type=str, default="experiments/results/diffusion")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.quick:
        config["gcn_epochs"] = 2
        config["rm_epochs"] = 2
        config["cate_epochs"] = 2
        config["bcq_epochs_bc"] = 2
        config["bcq_epochs_q"] = 2
        config["diff_epochs"] = 2
        config["diff_ddim_steps"] = 5
    elif args.epochs:
        config["diff_epochs"] = args.epochs

    datasets = (
        ["kuairand", "obd-all", "kuairec"]
        if args.dataset == "all"
        else [d.strip() for d in args.dataset.split(",")]
    )
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"Starting Diff-GNN-Bandit benchmark sweep for datasets={datasets}, seeds={seeds} ...")

    for d in datasets:
        for s in seeds:
            run_diffusion_experiment(d, s, config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
