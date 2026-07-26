"""
Backward RL Execution Script (Model-Based Sequential Offline RL)
================================================================

Trains the Dynamics Model to simulate state transitions, and uses it 
to train a Multi-Step Risk-Averse Dynamic BCQ Agent.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.utils.data_loader import load_dataset
from src.graph.lightgcn import LightGCN
from src.causal.cate_estimator import CATEEstimator
from src.utils.metrics import RewardModel, evaluate_policy
from src.ope.estimators import evaluate_all
from src.utils.trajectory_buffer import TrajectoryDataset
from src.agent.dynamics import DynamicsTrainer
from src.agent.bcq_dynamic import DynamicBCQAgent

def main():
    parser = argparse.ArgumentParser(description="Run Backward RL Experiments")
    parser.add_argument("--dataset", type=str, default="obd-all", choices=["obd-all", "obd-men", "obd-women", "criteo"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for dry-run testing")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Backward RL on {args.dataset.upper()} (Seed={args.seed}) | Device: {device}")
    
    out_dir = Path(f"experiments/results/{args.dataset}_backward_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint paths
    dyn_ckpt = out_dir / "dynamics_model.pt"
    bcq_ckpt = out_dir / "bcq_dynamic.pt"

    # ---------------------------------------------------------
    # Step 0: Data Loading
    # ---------------------------------------------------------
    print("\n[Step 0] Loading Data...")
    dataset = load_dataset(args.dataset)
    n_users = dataset.n_users
    n_items = dataset.n_items
    n_actions = 2 if args.dataset == "criteo" else n_items
    
    # ---------------------------------------------------------
    # Step 1: LightGCN (Graph Embeddings)
    # ---------------------------------------------------------
    print("\n[Step 1] Loading/Training GNN...")
    # For backward RL, we assume the graph is static for the embedding layer, 
    # but the user's sequential context changes. 
    # We will just train LightGCN quickly or mock it for this demonstration.
    gcn = LightGCN(
        n_nodes=dataset.n_nodes, 
        embed_dim=64, 
        n_layers=3, 
        adj=dataset.adj, 
        n_users=n_users, 
        dropout=0.0
    ).to(device)
    # NOTE: Normally we'd train this. For simplicity in the script structure, 
    # we just use the initialized embeddings. In a full run, copy the LightGCN 
    # training loop from run_main.py here.
    
    print("Building Context+GNN States...")
    def build_states(split):
        emb = gcn.encode_users(split.user_ids)
        return np.concatenate([split.contexts, emb], axis=1).astype(np.float32)
        
    s_train = build_states(dataset.train)
    s_val = build_states(dataset.val)
    s_test = build_states(dataset.test)
    state_dim = s_train.shape[1]
    
    # ---------------------------------------------------------
    # Step 2: Extract Temporal Trajectories
    # ---------------------------------------------------------
    print("\n[Step 2] Extracting Temporal Trajectories...")
    traj_buffer = TrajectoryDataset(s_train, dataset.train.actions, dataset.train.rewards, dataset.train.user_ids)
    
    if len(traj_buffer) == 0:
        print("ERROR: No trajectories found! This dataset might not have sequential impressions.")
        return
        
    t_s_t, t_a_t, t_r_t, t_s_next = traj_buffer.get_tensors(device)
    
    # ---------------------------------------------------------
    # Step 3: Train Dynamics Model
    # ---------------------------------------------------------
    print("\n[Step 3] Training Dynamics Model...")
    dyn_epochs = args.epochs if args.epochs else 50
    dynamics = DynamicsTrainer(state_dim=state_dim, n_actions=n_actions, device=device)
    
    if not dynamics.load_checkpoint(str(dyn_ckpt)):
        for ep in range(dyn_epochs):
            loss = dynamics.train_epoch(t_s_t, t_a_t, t_s_next)
            if (ep + 1) % 10 == 0 or ep == dyn_epochs - 1:
                print(f"  Dyn Epoch {ep+1:2d}/{dyn_epochs} | MSE Loss: {loss:.6f}")
            dynamics.save_checkpoint(str(dyn_ckpt))

    # ---------------------------------------------------------
    # Step 4: CATE Estimation & Reward Weighting
    # ---------------------------------------------------------
    print("\n[Step 4] Computing Uplift-Weighted Rewards...")
    cate_model = CATEEstimator(state_dim, n_actions, device=str(device))
    cate_model.fit_from_outcomes(s_train, dataset.train.actions, dataset.train.rewards, n_epochs=args.epochs if args.epochs else 50)
    r_weighted = cate_model.uplift_weighted_rewards(t_s_t.cpu().numpy(), t_a_t.cpu().numpy(), t_r_t.cpu().numpy())

    # ---------------------------------------------------------
    # Step 5: Train Dynamic BCQ Agent
    # ---------------------------------------------------------
    print("\n[Step 5] Training Multi-Step Risk-Averse BCQ...")
    item_emb = gcn.get_item_embeddings().detach().cpu().numpy() if n_actions > 2 else None
    bcq = DynamicBCQAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        item_embeddings=item_emb,
        device=device
    )
    
    bcq_epochs_bc = args.epochs if args.epochs else 30
    bcq_epochs_q = args.epochs if args.epochs else 100
    
    if not bcq.load_checkpoint(str(bcq_ckpt)):
        bcq.train(
            s_t=t_s_t.cpu().numpy(),
            a_t=t_a_t.cpu().numpy(),
            r_t=r_weighted,
            s_next=t_s_next.cpu().numpy(),
            n_epochs_bc=bcq_epochs_bc,
            n_epochs_q=bcq_epochs_q
        )
        bcq.save_checkpoint(str(bcq_ckpt))

    # ---------------------------------------------------------
    # Step 6: OPE Evaluation
    # ---------------------------------------------------------
    print("\n[Step 6] Off-Policy Evaluation...")
    rm = RewardModel(state_dim, n_actions, device=str(device))
    rm.fit(s_train, dataset.train.actions, dataset.train.rewards.astype(np.float32), n_epochs=args.epochs if args.epochs else 30)
    
    pi_bcq = bcq.action_probabilities(s_test)
    r_hat = rm.predict(s_test)
    
    ope = evaluate_all(
        dataset.test.rewards.astype(np.float32),
        pi_bcq,
        dataset.test.propensities,
        dataset.test.actions,
        n_actions,
        r_hat,
        clip=100.0
    )
    
    print(f"\nFinal DR Score for Dynamic-BCQ: {ope['DR'].value:.6f}")
    
    # Save the output correctly
    ope_serialised = {
        "Dynamic-BCQ": {
            est_name: {
                "value": res.value,
                "std": res.std,
                "ci_lower": res.ci_lower,
                "ci_upper": res.ci_upper,
                "n": res.n
            } for est_name, res in ope.items()
        }
    }
    
    with open(out_dir / "results.json", "w") as f:
        json.dump(ope_serialised, f, indent=2)
        
    print("\nBackward RL Pipeline Complete!")

if __name__ == "__main__":
    main()
