import sys
from pathlib import Path
ROOT = Path("e:/T2530969/ZARIF/gnn-bandit-thesis")
sys.path.insert(0, str(ROOT))
import json
import torch
import numpy as np

from src.utils.data_loader import load_dataset
from experiments.run_main import (
    DEFAULT_CONFIG, train_lightgcn, build_states,
    train_reward_model, train_cate_model, train_gnn_bandit, evaluate_policy
)

def test():
    config = DEFAULT_CONFIG.copy()
    config["gcn_epochs"] = 1
    config["rm_epochs"] = 1
    config["cate_epochs"] = 1
    config["bcq_epochs_bc"] = 1
    config["bcq_epochs_q"] = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("criteo", root=str(ROOT))
    
    val = 0.25
    config["bcq_cvar_alpha"] = val
    
    gcn = train_lightgcn(dataset, config, device, 0)
    s_train = build_states(dataset.train.contexts, dataset.train.user_ids, gcn, device)
    s_test = build_states(dataset.test.contexts, dataset.test.user_ids, gcn, device)

    rm = train_reward_model(dataset, s_train, config, device)
    cate = train_cate_model(dataset, s_train, config, device)
    
    agent = train_gnn_bandit(dataset, s_train, config, device, 0, gcn_model=gcn, cate_model=cate)
    probs = agent.action_probabilities(s_test)
    rm_preds = rm.predict(s_test)
    
    test = dataset.test
    ope = evaluate_policy(
        probs, test.rewards.astype(np.float32), test.propensities,
        test.actions, dataset.n_items, rm_preds,
        label=f"cvar_alpha={val}",
    )
    
    dr = ope.get("DR")
    print(dr)

test()
