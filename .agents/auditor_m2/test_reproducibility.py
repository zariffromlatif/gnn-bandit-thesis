
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import torch
import numpy as np
from src.agent.bcq import BCQAgent
from src.causal.cate_estimator import CATEEstimator
from src.graph.lightgcn import LightGCN
from scipy.sparse import csr_matrix

def run_pipeline(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Adjacency
    adj = csr_matrix([[0, 1], [1, 0]], dtype=np.float32)
    gcn = LightGCN(n_nodes=2, embed_dim=8, n_layers=2, adj=adj, n_users=1)
    
    states = np.random.RandomState(seed).randn(20, 8).astype(np.float32)
    actions = np.random.RandomState(seed).randint(0, 2, size=20)
    rewards = np.random.RandomState(seed).binomial(1, 0.5, size=20).astype(np.float32)
    
    cate = CATEEstimator(state_dim=8, n_actions=2, hidden=16, n_hidden=1)
    cate.fit_from_outcomes(states, actions, rewards, n_epochs=5, batch_size=10, cfr_lambda=0.05, verbose=False)
    cate_preds = cate.predict(states)
    
    agent = BCQAgent(state_dim=8, n_actions=2, hidden=16, n_hidden=1, num_quantiles=8)
    agent.train(states, actions, rewards, n_epochs_bc=5, n_epochs_q=5, batch_size=10, verbose=False)
    probs = agent.action_probabilities(states)
    
    return cate_preds, probs

print('Testing seed determinism and reproducibility...')
c1, p1 = run_pipeline(seed=123)
c2, p2 = run_pipeline(seed=123)
c_diff = np.max(np.abs(c1 - c2))
p_diff = np.max(np.abs(p1 - p2))

print(f'Max CATE difference between two runs with seed 123: {c_diff:.10f}')
print(f'Max BCQ prob difference between two runs with seed 123: {p_diff:.10f}')

assert c_diff == 0.0, 'CATE predictions are not deterministic!'
assert p_diff == 0.0, 'BCQ action probabilities are not deterministic!'
print('REPRODUCIBILITY & DETERMINISM: PASS (Exact 0.0 diff)')
