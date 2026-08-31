
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))

import unittest
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

from src.ope.estimators import ipw, snipw, direct_method, doubly_robust, _importance_weights
from src.graph.lightgcn import LightGCN, _symmetric_norm
from src.graph.tgn import TGNEncoder, TimeEncoder
from src.causal.cate_estimator import CATEEstimator, GradientReversal
from src.agent.bcq import BCQAgent
from src.baselines.policies import (
    RandomPolicy, BTSPolicy, DQNPolicy, MFBanditPolicy,
    GreedyGNNPolicy, UpliftPolicy, LinUCBPolicy, NeuralUCBPolicy,
    CQLPolicy, IQLPolicy, DecisionTransformerPolicy
)

class TestOPEEstimators(unittest.TestCase):
    def test_importance_weights(self):
        pi_new = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
        pi_old = np.array([0.5, 0.5, 0.5])
        actions = np.array([0, 1, 0])
        w = _importance_weights(pi_new, pi_old, actions, n_actions=2, clip=10.0)
        np.testing.assert_allclose(w, [1.6, 1.8, 1.0])

    def test_ipw_and_snipw(self):
        rewards = np.array([1.0, 0.0, 1.0])
        pi_new = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
        pi_old = np.array([0.5, 0.5, 0.5])
        actions = np.array([0, 1, 0])
        res_ipw = ipw(rewards, pi_new, pi_old, actions, n_actions=2)
        res_snipw = snipw(rewards, pi_new, pi_old, actions, n_actions=2)
        self.assertAlmostEqual(res_ipw.value, 2.6 / 3.0, places=5)
        self.assertAlmostEqual(res_snipw.value, 2.6 / 4.4, places=5)

    def test_dm_and_dr(self):
        rewards = np.array([1.0, 0.0, 1.0])
        pi_new = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
        pi_old = np.array([0.5, 0.5, 0.5])
        actions = np.array([0, 1, 0])
        rm_preds = np.array([[0.6, 0.1], [0.2, 0.7], [0.4, 0.3]])
        res_dm = direct_method(rm_preds, pi_new, n_actions=2)
        self.assertAlmostEqual(res_dm.value, 0.50, places=5)

        res_dr = doubly_robust(rewards, pi_new, pi_old, actions, n_actions=2, reward_model=rm_preds)
        self.assertAlmostEqual(res_dr.value, 0.50 - 0.02 / 3.0, places=5)

    def test_dr_estimator_bounds_and_clipping(self):
        # Extreme propensity case
        pi_new = np.array([[1.0, 0.0]])
        pi_old = np.array([0.0001])
        actions = np.array([0])
        rewards = np.array([1.0])
        rm_preds = np.array([[0.5, 0.5]])
        res_dr = doubly_robust(rewards, pi_new, pi_old, actions, n_actions=2, reward_model=rm_preds, clip=50.0)
        # w_clipped = 50.0. DM = 0.5. Correction = 50.0 * (1 - 0.5) = 25.0 -> DR = 25.5
        self.assertAlmostEqual(res_dr.value, 25.5, places=5)

class TestGraphEncoders(unittest.TestCase):
    def test_lightgcn_norm(self):
        A = csr_matrix([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
        A_norm = _symmetric_norm(A).toarray()
        self.assertAlmostEqual(A_norm[0, 1], 1.0 / np.sqrt(2.0), places=5)
        self.assertAlmostEqual(A_norm[1, 0], 1.0 / np.sqrt(2.0), places=5)
        self.assertAlmostEqual(A_norm[0, 2], 1.0 / np.sqrt(2.0), places=5)
        self.assertAlmostEqual(A_norm[1, 2], 0.0, places=5)

    def test_lightgcn_forward(self):
        A = csr_matrix([[0, 1], [1, 0]], dtype=np.float32)
        model = LightGCN(n_nodes=2, embed_dim=16, n_layers=2, adj=A, n_users=1)
        emb = model.forward()
        self.assertEqual(emb.shape, (2, 16))
        u_emb = model.encode_users(np.array([0]))
        self.assertEqual(u_emb.shape, (1, 16))

    def test_tgn_forward(self):
        tgn = TGNEncoder(n_nodes=4, embed_dim=16, n_users=2, memory_dim=16, time_dim=8)
        u = torch.tensor([0, 1], dtype=torch.long)
        i = torch.tensor([2, 3], dtype=torch.long)
        ts = torch.tensor([1.0, 2.0], dtype=torch.float32)
        tgn.update_events(u, i, ts)
        emb = tgn.forward()
        self.assertEqual(emb.shape, (4, 16))

class TestCATEEstimator(unittest.TestCase):
    def test_gradient_reversal(self):
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = GradientReversal.apply(x, 0.5)
        loss = (y ** 2).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), [-2.0, -3.0])

    def test_cate_fit_predict(self):
        cate = CATEEstimator(state_dim=8, n_actions=3, hidden=32, n_hidden=2)
        states = np.random.randn(20, 8).astype(np.float32)
        actions = np.random.randint(0, 3, size=20)
        rewards = np.random.binomial(1, 0.5, size=20).astype(np.float32)
        cate.fit_from_outcomes(states, actions, rewards, n_epochs=2, batch_size=10, cfr_lambda=0.05, verbose=False)
        preds = cate.predict(states)
        self.assertEqual(preds.shape, (20, 3))

    def test_gp_cate_propagation(self):
        cate = CATEEstimator(state_dim=8, n_actions=2)
        raw_cate = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        uids = np.array([0, 1])
        adj = csr_matrix([[0, 1], [1, 0]], dtype=np.float32)
        smooth = cate.propagate_cate(raw_cate, uids, adj, n_nodes=2, n_hops=1, beta=0.5)
        self.assertEqual(smooth.shape, (2, 2))

class TestBCQAgent(unittest.TestCase):
    def test_cvar_computation(self):
        agent = BCQAgent(state_dim=4, n_actions=2, num_quantiles=4, cvar_alpha=0.5)
        q_quantiles = torch.tensor([[[4.0, 1.0, 3.0, 2.0], [8.0, 7.0, 6.0, 5.0]]])
        cvar = agent._compute_cvar(q_quantiles)
        self.assertAlmostEqual(cvar[0, 0].item(), 1.5, places=5)
        self.assertAlmostEqual(cvar[0, 1].item(), 5.5, places=5)

    def test_safe_mask(self):
        agent = BCQAgent(state_dim=4, n_actions=3, threshold_ratio=0.3, min_actions=2)
        bc_probs = torch.tensor([[0.9, 0.05, 0.05]])
        mask = agent._safe_mask(bc_probs)
        self.assertEqual(mask.sum().item(), 2)
        self.assertTrue(mask[0, 0].item())

    def test_action_probabilities_sum_to_one(self):
        agent = BCQAgent(state_dim=4, n_actions=3)
        states = np.random.randn(10, 4).astype(np.float32)
        actions = np.random.randint(0, 3, size=10)
        rewards = np.random.rand(10).astype(np.float32)
        agent.train(states, actions, rewards, n_epochs_bc=2, n_epochs_q=2, batch_size=5, verbose=False)
        probs = agent.action_probabilities(states)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(10), atol=1e-5)

class TestBaselines(unittest.TestCase):
    def test_all_baselines(self):
        N, D, A = 20, 6, 4
        states = np.random.randn(N, D).astype(np.float32)
        actions = np.random.randint(0, A, size=N)
        rewards = np.random.binomial(1, 0.3, size=N).astype(np.float32)

        # Random
        p_rand = RandomPolicy(A).action_probabilities(states)
        np.testing.assert_allclose(p_rand.sum(axis=1), np.ones(N), atol=1e-5)

        # LinUCB
        lin = LinUCBPolicy(D, A)
        lin.train(states, actions, rewards, verbose=False)
        p_lin = lin.action_probabilities(states)
        np.testing.assert_allclose(p_lin.sum(axis=1), np.ones(N), atol=1e-5)

        # CQL
        cql = CQLPolicy(D, A, hidden=32, n_hidden=1)
        cql.train(states, actions, rewards, n_epochs=2, batch_size=10, verbose=False)
        p_cql = cql.action_probabilities(states)
        np.testing.assert_allclose(p_cql.sum(axis=1), np.ones(N), atol=1e-5)

        # IQL
        iql = IQLPolicy(D, A, hidden=32, n_hidden=1)
        iql.train(states, actions, rewards, n_epochs=2, batch_size=10, verbose=False)
        p_iql = iql.action_probabilities(states)
        np.testing.assert_allclose(p_iql.sum(axis=1), np.ones(N), atol=1e-5)

        # DT
        dt = DecisionTransformerPolicy(D, A, hidden=32, n_hidden=1)
        dt.train(states, actions, rewards, n_epochs=2, batch_size=10, verbose=False)
        p_dt = dt.action_probabilities(states)
        np.testing.assert_allclose(p_dt.sum(axis=1), np.ones(N), atol=1e-5)

if __name__ == '__main__':
    unittest.main()
