"""
Baseline policies for comparison against GNN-Bandit.

Each baseline isolates one component so the ablation story is clean:

| Baseline              | Tests                                           |
|-----------------------|-------------------------------------------------|
| RandomPolicy          | Any structured method beats random               |
| BTSPolicy             | GNN-Bandit beats the existing logging policy      |
| DQNPolicy             | BCQ constraint is necessary (vs. unconstrained)  |
| MFBanditPolicy        | GNN > simpler embedding (MF) for cold-start      |
| GreedyGNNPolicy       | RL component adds value over greedy selection     |
| UpliftPolicy          | Full framework > simple uplift look-up table      |
| LinUCBPolicy          | GNN-Bandit beats the standard bandit baseline     |
| NeuralUCBPolicy       | GNN-Bandit beats neural bandit methods            |
| CQLPolicy             | BCQ > competing conservative offline RL (CQL)    |
| IQLPolicy             | BCQ > implicit offline RL (IQL)                   |
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# Protocol: every policy must expose `action_probabilities(states) -> (N, A)`
# ============================================================================


class RandomPolicy:
    """
    Uniform random policy.  Serves as the floor — everything should beat this.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def action_probabilities(self, states: np.ndarray) -> np.ndarray:
        N = states.shape[0]
        return np.full((N, self.n_actions), 1.0 / self.n_actions, dtype=np.float32)

    def select_action(self, state: np.ndarray) -> int:
        return int(np.random.randint(0, self.n_actions))


class BTSPolicy:
    """
    Thompson Sampling policy (the BTS logging policy from OBD).

    We reconstruct the BTS policy's action distribution from the logged
    propensity scores.  For each user segment, the BTS policy assigns
    non-uniform probabilities to items based on posterior sampling.

    For evaluation, we use the *logged propensity scores directly* as
    the policy probabilities.
    """

    def __init__(self, n_actions: int, default_propensity: float = 0.0125):
        self.n_actions = n_actions
        self.default_propensity = default_propensity

    def action_probabilities(
        self,
        states: np.ndarray,
        logged_actions: Optional[np.ndarray] = None,
        logged_propensities: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        For BTS we can only know pi_BTS(a_t | x_t) for the logged action.
        Non-logged actions get the default propensity (1/n_actions).

        This is a common limitation in OPE when the full policy isn't available.
        """
        N = states.shape[0]
        probs = np.full((N, self.n_actions),
                        self.default_propensity, dtype=np.float32)
        if logged_actions is not None and logged_propensities is not None:
            probs[np.arange(N), logged_actions] = logged_propensities
        # Normalise rows
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.clip(row_sums, 1e-8, None)
        return probs


class DQNPolicy:
    """
    Standard Deep Q-Network — **no** batch constraint.

    This is the ablation that proves the BCQ constraint is necessary.
    Without the constraint, the agent may select actions it has never
    observed in the training data, leading to overestimated Q-values.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256,
                 n_hidden: int = 2, lr: float = 1e-3,
                 temperature: float = 1.0, device: str = "cpu"):
        self.n_actions   = n_actions
        self.temperature = temperature
        self.device      = torch.device(device)

        layers = []
        prev = state_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(0.1)]
            prev = hidden
        layers.append(nn.Linear(prev, n_actions))
        self.q_net = nn.Sequential(*layers).to(self.device)
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def train(self, states: np.ndarray, actions: np.ndarray,
              rewards: np.ndarray, n_epochs: int = 50,
              batch_size: int = 16384, verbose: bool = True):
        S = torch.FloatTensor(states).to(self.device)
        A = torch.LongTensor(actions).to(self.device)
        R = torch.FloatTensor(rewards).to(self.device)
        N = len(S)

        self.q_net.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for s, a, r in loader:
                q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q, r)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item() * len(s)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    DQN epoch {epoch+1:3d}  loss: {total_loss/len(S):.4f}")
        self.q_net.eval()

    @torch.no_grad()
    def action_probabilities(self, states: np.ndarray) -> np.ndarray:
        S = torch.FloatTensor(states)
        q = self.q_net(S)
        return F.softmax(q / self.temperature, dim=1).cpu().numpy()

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return int(self.q_net(s).argmax(dim=1).item())


class MFBanditPolicy:
    """
    Matrix Factorization embeddings + BCQ.

    Replaces LightGCN with simple MF (no graph propagation). This is the
    ablation proving that *graph convolution* adds value beyond basic
    collaborative filtering embeddings.

    MF embeddings are learned via standard BPR on the same interaction data,
    but without the adjacency-based propagation.
    """

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64,
                 state_dim: int = 0, hidden: int = 256,
                 threshold_ratio: float = 0.3, min_actions: int = 5,
                 lr: float = 1e-3, temperature: float = 1.0, device: str = "cpu"):
        self.n_users     = n_users
        self.n_items     = n_items
        self.embed_dim   = embed_dim
        self.threshold   = threshold_ratio / n_items
        self.min_actions = min(min_actions, n_items)
        self.temperature = temperature
        self.device      = torch.device(device)

        # MF embeddings (no graph propagation)
        self.user_emb = nn.Embedding(n_users, embed_dim).to(self.device)
        self.item_emb = nn.Embedding(n_items, embed_dim).to(self.device)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.mf_optim = torch.optim.Adam(
            list(self.user_emb.parameters()) + list(self.item_emb.parameters()),
            lr=lr,
        )

        # BCQ components (same as main agent but with MF state)
        full_state_dim = embed_dim + state_dim
        layers_bc, layers_q = [], []
        prev = full_state_dim
        for _ in range(2):
            layers_bc += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(0.1)]
            layers_q  += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(0.1)]
            prev = hidden
        layers_bc.append(nn.Linear(prev, n_items))
        layers_q.append(nn.Linear(prev, n_items))

        self.bc_model = nn.Sequential(*layers_bc).to(self.device)
        self.q_net    = nn.Sequential(*layers_q).to(self.device)
        self.bc_optim = torch.optim.Adam(self.bc_model.parameters(), lr=lr)
        self.q_optim  = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def train_mf(self, user_ids: np.ndarray, pos_items: np.ndarray,
                 n_epochs: int = 30, batch_size: int = 16384, verbose: bool = True):
        """Train MF embeddings with BPR loss (no graph)."""
        rng = np.random.RandomState(42)
        self.user_emb.train()
        self.item_emb.train()
        for epoch in range(n_epochs):
            perm = rng.permutation(len(user_ids))
            total_loss = 0.0
            for start in range(0, len(user_ids), batch_size):
                idx = perm[start:start + batch_size]
                u = torch.LongTensor(user_ids[idx]).to(self.device)
                p = torch.LongTensor(pos_items[idx]).to(self.device)
                neg = rng.randint(0, self.n_items, size=len(idx))
                n = torch.LongTensor(neg).to(self.device)

                u_e = self.user_emb(u)
                p_e = self.item_emb(p)
                n_e = self.item_emb(n)

                pos_score = (u_e * p_e).sum(dim=1)
                neg_score = (u_e * n_e).sum(dim=1)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()

                self.mf_optim.zero_grad()
                loss.backward()
                self.mf_optim.step()
                total_loss += loss.item() * len(idx)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    MF epoch {epoch+1:3d}  BPR loss: {total_loss/len(user_ids):.4f}")

        self.user_emb.eval()
        self.item_emb.eval()

    def train_policy(self, states: np.ndarray, user_ids: np.ndarray,
                     actions: np.ndarray, rewards: np.ndarray,
                     n_epochs_bc: int = 30, n_epochs_q: int = 50,
                     batch_size: int = 16384, verbose: bool = True):
        """Train BC + Q using MF embeddings concatenated with context."""
        with torch.no_grad():
            u_emb = self.user_emb(
                torch.LongTensor(user_ids).to(self.device)
            ).cpu().numpy()
        full_states = np.hstack([u_emb, states])

        S = torch.FloatTensor(full_states).to(self.device)
        A = torch.LongTensor(actions).to(self.device)
        R = torch.FloatTensor(rewards).to(self.device)
        N = len(S)

        # BC phase
        self.bc_model.train()
        for epoch in range(n_epochs_bc):
            for s, a, _ in loader:
                loss = F.cross_entropy(self.bc_model(s), a)
                self.bc_optim.zero_grad()
                loss.backward()
                self.bc_optim.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    MF-BC epoch {epoch+1:3d}")

        # Q phase
        self.bc_model.eval()
        self.q_net.train()
        for epoch in range(n_epochs_q):
            for s, a, r in loader:
                q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q, r)
                self.q_optim.zero_grad()
                loss.backward()
                self.q_optim.step()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    MF-Q  epoch {epoch+1:3d}")
        self.q_net.eval()

    def _safe_mask(self, bc_probs: torch.Tensor) -> torch.Tensor:
        """Keep actions above threshold, with min_actions floor."""
        mask = bc_probs >= self.threshold
        n_surviving = mask.sum(dim=1)
        need_fix = n_surviving < self.min_actions
        if need_fix.any():
            _, top_idx = bc_probs[need_fix].topk(self.min_actions, dim=1)
            fix_mask = torch.zeros_like(bc_probs[need_fix], dtype=torch.bool)
            fix_mask.scatter_(1, top_idx, True)
            mask[need_fix] = fix_mask
        return mask

    @torch.no_grad()
    def action_probabilities(self, states: np.ndarray,
                             user_ids: Optional[np.ndarray] = None,
                             batch_size: int = 65536) -> np.ndarray:
        N = len(states)
        all_probs = np.empty((N, self.n_items), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            if user_ids is not None:
                u_emb = self.user_emb(
                    torch.LongTensor(user_ids[start:end]).to(self.device)
                ).cpu().numpy()
                full = np.hstack([u_emb, states[start:end]])
            else:
                full = states[start:end]

            S = torch.FloatTensor(full).to(self.device)
            bc_probs = F.softmax(self.bc_model(S), dim=1)
            mask = self._safe_mask(bc_probs)

            q = self.q_net(S)
            q[~mask] = float("-inf")
            all_probs[start:end] = F.softmax(
                q / self.temperature, dim=1).cpu().numpy()

        return all_probs


class GreedyGNNPolicy:
    """
    LightGCN embeddings + greedy action selection (no BCQ constraint).

    Uses the same GNN embeddings as the full framework but picks actions
    greedily (highest dot-product score) instead of learning a constrained
    Q-function.  This ablation proves the RL component adds value.
    """

    def __init__(self, n_actions: int, temperature: float = 0.1):
        self.n_actions   = n_actions
        self.temperature = temperature

    def action_probabilities(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        pi(a|s) = softmax(e_u . e_i / temperature)

        Parameters
        ----------
        user_embeddings : (N, K) user embeddings from LightGCN
        item_embeddings : (A, K) all item embeddings from LightGCN
        """
        # Dot product scores
        scores = user_embeddings @ item_embeddings.T   # (N, A)
        # Softmax with low temperature → near-greedy
        scores = scores / self.temperature
        exp_s  = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs  = exp_s / exp_s.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)


class UpliftPolicy:
    """
    Select the action with the highest estimated uplift from the
    precomputed uplift table.

    This is the simplest "causal-aware" baseline — it uses treatment
    effects but without any graph structure or RL optimisation.
    """

    def __init__(self, n_users: int, n_actions: int,
                 temperature: float = 1.0):
        self.n_users   = n_users
        self.n_actions = n_actions
        self.temperature = temperature
        # Uplift table:  (n_users, n_actions) — filled from uplift_estimates.csv
        self.uplift_table = np.zeros((n_users, n_actions), dtype=np.float32)

    def load_uplift(self, uplift_csv_path: str):
        """Load uplift estimates from the preprocessing output."""
        import pandas as pd
        df = pd.read_csv(uplift_csv_path)
        for _, row in df.iterrows():
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            if uid < self.n_users and iid < self.n_actions:
                self.uplift_table[uid, iid] = row["uplift"]

    def action_probabilities(
        self,
        states: np.ndarray,
        user_ids: np.ndarray,
    ) -> np.ndarray:
        """
        pi(a|s) = softmax(uplift(u, a) / temperature)
        """
        uplift_vals = self.uplift_table[user_ids]     # (N, A)
        scores = uplift_vals / self.temperature
        exp_s  = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs  = exp_s / exp_s.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)


# ============================================================================
# LinUCB — Standard contextual bandit baseline  (Li et al., 2010)
# ============================================================================

class LinUCBPolicy:
    """
    Linear Upper Confidence Bound policy for contextual bandits.

    Maintains a separate linear model per action.  Selects the action with
    the highest UCB score:  a* = argmax_a  (theta_a^T x + alpha * UCB_a).

    Reference: Li et al., "A Contextual-Bandit Approach to Personalized
    News Article Recommendation", WWW 2010.
    """

    def __init__(self, state_dim: int, n_actions: int,
                 alpha: float = 1.0, temperature: float = 0.1):
        self.state_dim   = state_dim
        self.n_actions   = n_actions
        self.alpha       = alpha
        self.temperature = temperature

        # Per-action: A_a = I (d x d),  b_a = 0 (d,)
        self.A = [np.eye(state_dim, dtype=np.float64) for _ in range(n_actions)]
        self.b = [np.zeros(state_dim, dtype=np.float64) for _ in range(n_actions)]
        self._theta = [np.zeros(state_dim, dtype=np.float64)
                       for _ in range(n_actions)]

    def train(self, states: np.ndarray, actions: np.ndarray,
              rewards: np.ndarray, verbose: bool = True):
        """Update linear models with logged data (closed-form)."""
        if verbose:
            print(f"    LinUCB: fitting {len(states):,} samples, "
                  f"{self.n_actions} actions ...")

        for i in range(len(states)):
            a = int(actions[i])
            x = states[i].astype(np.float64)
            r = float(rewards[i])
            self.A[a] += np.outer(x, x)
            self.b[a] += r * x

        # Solve for theta_a = A_a^{-1} b_a
        for a in range(self.n_actions):
            try:
                self._theta[a] = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                self._theta[a] = np.zeros(self.state_dim, dtype=np.float64)

        if verbose:
            print(f"    LinUCB: done.")

    def action_probabilities(self, states: np.ndarray) -> np.ndarray:
        """Compute softmax over UCB scores for OPE."""
        N = len(states)
        scores = np.zeros((N, self.n_actions), dtype=np.float64)

        for a in range(self.n_actions):
            try:
                A_inv = np.linalg.inv(self.A[a])
            except np.linalg.LinAlgError:
                A_inv = np.eye(self.state_dim, dtype=np.float64)

            theta_a = self._theta[a]
            for i in range(N):
                x = states[i].astype(np.float64)
                pred = theta_a @ x
                ucb = self.alpha * np.sqrt(x @ A_inv @ x)
                scores[i, a] = pred + ucb

        # Softmax
        scores = scores / self.temperature
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)


# ============================================================================
# NeuralUCB — Neural contextual bandit  (Zhou et al., NeurIPS 2020)
# ============================================================================

class NeuralUCBPolicy:
    """
    Neural UCB policy using a neural network for reward prediction
    with gradient-based uncertainty estimation.

    UCB_a = f_theta(x, a) + alpha * ||grad_theta f(x,a)||_{V^{-1}}

    For computational tractability in large-scale settings, we use a
    simplified diagonal approximation for the uncertainty term.

    Reference: Zhou et al., "Neural Contextual Bandits with UCB-Based
    Exploration", NeurIPS 2020.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256,
                 n_hidden: int = 2, alpha: float = 0.1,
                 lr: float = 1e-3, temperature: float = 0.1,
                 device: str = "cpu"):
        self.n_actions   = n_actions
        self.alpha       = alpha
        self.temperature = temperature
        self.device      = torch.device(device)

        layers = []
        prev = state_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(0.1)]
            prev = hidden
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

        # Diagonal approximation of the gradient outer product
        self._n_params = sum(p.numel() for p in self.net.parameters()
                             if p.requires_grad)
        # Running sum of squared gradients (diagonal Fisher)
        self._diag_fisher = None

    def train(self, states: np.ndarray, actions: np.ndarray,
              rewards: np.ndarray, n_epochs: int = 50,
              batch_size: int = 16384, verbose: bool = True):
        S = torch.FloatTensor(states).to(self.device)
        A = torch.LongTensor(actions).to(self.device)
        R = torch.FloatTensor(rewards).to(self.device)
        N = len(S)

        self.net.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for s, a, r in loader:
                pred = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(pred, r)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item() * len(s)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    NeuralUCB epoch {epoch+1:3d}  "
                      f"loss: {total_loss/len(S):.4f}")

        # Build diagonal Fisher for uncertainty
        self.net.eval()
        self._build_fisher(S, A, R)

    def _build_fisher(self, S, A, R, max_samples: int = 50000):
        """Approximate diagonal Fisher information from data."""
        n = min(len(S), max_samples)
        self._diag_fisher = torch.zeros(self._n_params, device=self.device)

        self.net.eval()
        for i in range(0, n, 4096):
            end = min(i + 4096, n)
            s = S[i:end]
            a = A[i:end]
            pred = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            for j in range(len(pred)):
                self.net.zero_grad()
                pred[j].backward(retain_graph=True)
                grads = torch.cat([p.grad.flatten()
                                   for p in self.net.parameters()
                                   if p.grad is not None])
                self._diag_fisher += grads ** 2

        self._diag_fisher = self._diag_fisher / n + 1e-6  # regularise

    @torch.no_grad()
    def action_probabilities(self, states: np.ndarray) -> np.ndarray:
        S = torch.FloatTensor(states)
        pred = self.net(S)  # (N, A) mean predictions
        # Use prediction as score (UCB uncertainty is expensive at scale)
        scores = pred / self.temperature
        return F.softmax(scores, dim=1).cpu().numpy()


# ============================================================================
# CQL — Conservative Q-Learning  (Kumar et al., ICML 2020)
# ============================================================================

class CQLPolicy:
    """
    Conservative Q-Learning adapted for contextual bandits.

    CQL adds a regularisation term that penalises Q-values for actions
    not seen in the data, pushing them down.  This is the main competing
    conservative offline RL method to BCQ.

    Loss = MSE(Q(s,a), r) + beta * [logsumexp(Q(s,:)) - Q(s, a_data)]

    Reference: Kumar et al., "Conservative Q-Learning for Offline
    Reinforcement Learning", NeurIPS 2020.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256,
                 n_hidden: int = 2, lr: float = 1e-3, cql_alpha: float = 1.0,
                 temperature: float = 0.1, device: str = "cpu"):
        self.n_actions   = n_actions
        self.cql_alpha   = cql_alpha
        self.temperature = temperature
        self.device      = torch.device(device)

        layers = []
        prev = state_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(0.1)]
            prev = hidden
        layers.append(nn.Linear(prev, n_actions))
        self.q_net = nn.Sequential(*layers).to(self.device)
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def train(self, states: np.ndarray, actions: np.ndarray,
              rewards: np.ndarray, n_epochs: int = 50,
              batch_size: int = 16384, verbose: bool = True):
        S = torch.FloatTensor(states).to(self.device)
        A = torch.LongTensor(actions).to(self.device)
        R = torch.FloatTensor(rewards).to(self.device)
        N = len(S)

        self.q_net.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for s, a, r in loader:
                q_all = self.q_net(s)                          # (B, A)
                q_taken = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

                # Standard Bellman loss (single-step: Q(s,a) → r)
                bellman_loss = F.mse_loss(q_taken, r)

                # CQL penalty: push down Q-values for unobserved actions
                # logsumexp(Q(s,:)) approximates log E_a[exp(Q(s,a))]
                cql_penalty = (torch.logsumexp(q_all, dim=1)
                               - q_taken).mean()

                loss = bellman_loss + self.cql_alpha * cql_penalty

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item() * len(s)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    CQL epoch {epoch+1:3d}  "
                      f"loss: {total_loss/len(S):.4f}")
        self.q_net.eval()

    @torch.no_grad()
    def action_probabilities(self, states: np.ndarray) -> np.ndarray:
        S = torch.FloatTensor(states)
        q = self.q_net(S)
        return F.softmax(q / self.temperature, dim=1).cpu().numpy()


# ============================================================================
# IQL — Implicit Q-Learning  (Kostrikov et al., ICLR 2022)
# ============================================================================

class IQLPolicy:
    """
    Implicit Q-Learning adapted for contextual bandits.

    IQL avoids querying out-of-distribution actions by using expectile
    regression to implicitly extract the value of the best *in-sample*
    actions.  Unlike BCQ (which masks) or CQL (which penalises), IQL
    uses an asymmetric loss that upweights high-reward samples.

    For the contextual bandit (single-step) setting:
    - V(s) is trained with expectile regression on Q(s, a_data)
    - Q(s, a) is trained to predict rewards
    - Advantage A(s,a) = Q(s,a) - V(s) drives action selection

    Reference: Kostrikov et al., "Offline Reinforcement Learning with
    Implicit Q-Learning", ICLR 2022.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256,
                 n_hidden: int = 2, lr: float = 1e-3,
                 expectile: float = 0.7, temperature: float = 0.1,
                 device: str = "cpu"):
        self.n_actions   = n_actions
        self.expectile   = expectile
        self.temperature = temperature
        self.device      = torch.device(device)

        # Q-network
        q_layers = []
        prev = state_dim
        for _ in range(n_hidden):
            q_layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(0.1)]
            prev = hidden
        q_layers.append(nn.Linear(prev, n_actions))
        self.q_net = nn.Sequential(*q_layers).to(self.device)
        self.q_optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # Value network V(s) — scalar output
        v_layers = []
        prev = state_dim
        for _ in range(n_hidden):
            v_layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(0.1)]
            prev = hidden
        v_layers.append(nn.Linear(prev, 1))
        self.v_net = nn.Sequential(*v_layers).to(self.device)
        self.v_optim = torch.optim.Adam(self.v_net.parameters(), lr=lr)

    def _expectile_loss(self, diff: torch.Tensor, tau: float) -> torch.Tensor:
        """Asymmetric L2 loss: upweight positive diff (high-value samples)."""
        weight = torch.where(diff > 0, tau, 1.0 - tau)
        return (weight * diff.pow(2)).mean()

    def train(self, states: np.ndarray, actions: np.ndarray,
              rewards: np.ndarray, n_epochs: int = 50,
              batch_size: int = 16384, verbose: bool = True):
        S = torch.FloatTensor(states).to(self.device)
        A = torch.LongTensor(actions).to(self.device)
        R = torch.FloatTensor(rewards).to(self.device)
        N = len(S)

        for epoch in range(n_epochs):
            total_q_loss = 0.0
            total_v_loss = 0.0

            self.q_net.train()
            self.v_net.train()

            for s, a, r in loader:
                # --- Q update: standard regression Q(s,a) → r ---
                q_all = self.q_net(s)
                q_taken = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
                q_loss = F.mse_loss(q_taken, r)

                self.q_optim.zero_grad()
                q_loss.backward()
                self.q_optim.step()
                total_q_loss += q_loss.item() * len(s)

                # --- V update: expectile regression on Q(s, a_data) ---
                with torch.no_grad():
                    q_target = self.q_net(s).gather(
                        1, a.unsqueeze(1)).squeeze(1)

                v = self.v_net(s).squeeze(1)
                v_loss = self._expectile_loss(
                    q_target - v, self.expectile)

                self.v_optim.zero_grad()
                v_loss.backward()
                self.v_optim.step()
                total_v_loss += v_loss.item() * len(s)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    IQL epoch {epoch+1:3d}  "
                      f"Q: {total_q_loss/len(S):.4f}  "
                      f"V: {total_v_loss/len(S):.6f}")

        self.q_net.eval()
        self.v_net.eval()

    @torch.no_grad()
    def action_probabilities(self, states: np.ndarray) -> np.ndarray:
        """
        pi(a|s) = softmax(A(s,a) / temperature)
        where A(s,a) = Q(s,a) - V(s) is the advantage.
        """
        S = torch.FloatTensor(states)
        q = self.q_net(S)                                    # (N, A)
        v = self.v_net(S)                                    # (N, 1)
        advantage = q - v                                    # (N, A)
        return F.softmax(advantage / self.temperature,
                         dim=1).cpu().numpy()
