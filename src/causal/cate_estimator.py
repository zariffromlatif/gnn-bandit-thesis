"""
Conditional Average Treatment Effect (CATE) estimation module.

Bridges the gap between graph embeddings and offline policy learning
by estimating heterogeneous treatment effects (HTE) per user-action pair.

The CATE estimator answers: "What is the expected *lift* in reward if we
show item a to user u, compared to not showing it?"  This is fundamentally
different from predicting the raw reward — it captures the *causal* effect
of the intervention.

Architecture (T-learner approach)
---------------------------------
1. Train two models on GNN-augmented states:
   - mu_1(x) = E[Y | X=x, T=a]  (response under treatment a)
   - mu_0(x) = E[Y | X=x, T!=a] (response under control / other actions)

2. CATE(x, a) = mu_1(x, a) - mu_0(x)

For the multi-arm OBD setting, we use a **modified S-learner** that directly
predicts the uplift for each (user, item) pair using the precomputed uplift
table from the preprocessing stage, augmented with GNN embeddings.

The estimated CATE scores are used to:
  - Augment the BCQ reward signal (uplift-weighted rewards)
  - Segment users into uplift quadrants (Sleeping Dogs detection)
  - Provide deconfounded states for OPE

References
----------
- Kunzel et al., "Metalearners for estimating heterogeneous treatment
  effects using machine learning", PNAS 2019.
- Nie & Wager, "Quasi-oracle estimation of heterogeneous treatment
  effects", Biometrika 2021.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class _CATENetwork(nn.Module):
    """MLP that predicts CATE(x, a) for all actions simultaneously."""

    def __init__(self, input_dim: int, n_actions: int, hidden: int = 128,
                 n_hidden: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CATEEstimator:
    """
    Estimates Conditional Average Treatment Effects using GNN embeddings.

    Uses a neural S-learner approach: a single network predicts the uplift
    for each (state, action) pair.  The network is trained on precomputed
    uplift estimates from the randomised portion of OBD data, or can be
    trained directly from outcomes using a T-learner decomposition.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the input state (context + GNN embedding).
    n_actions : int
        Number of discrete actions / items.
    hidden : int
        Hidden layer width.  Default 128.
    n_hidden : int
        Number of hidden layers.  Default 2.
    lr : float
        Learning rate.  Default 1e-3.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        n_hidden: int = 2,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = torch.device(device)

        self.model = _CATENetwork(
            state_dim, n_actions, hidden, n_hidden
        ).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit_from_uplift_table(
        self,
        states: np.ndarray,
        user_ids: np.ndarray,
        uplift_table: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 16384,
        verbose: bool = True,
    ):
        """
        Train the CATE network to predict uplift values from states.

        The uplift table contains precomputed treatment effects from the
        randomised data.  We train the neural network to generalise these
        estimates to unseen states using GNN-augmented features.

        Parameters
        ----------
        states : (N, D) augmented state vectors (context + GNN embedding).
        user_ids : (N,) user indices into the uplift table.
        uplift_table : (n_users, n_actions) precomputed uplift values.
        n_epochs : number of training epochs.
        batch_size : mini-batch size.
        verbose : print training progress.
        """
        # Build per-sample uplift targets
        targets = uplift_table[user_ids]  # (N, n_actions)

        S = torch.FloatTensor(states)
        T = torch.FloatTensor(targets)
        loader = DataLoader(TensorDataset(S, T), batch_size=batch_size,
                            shuffle=True, pin_memory=True)

        if verbose:
            print(f"  [CATE] Training on {len(states):,} samples, "
                  f"{n_epochs} epochs")

        self.model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for s_batch, t_batch in loader:
                s_batch = s_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                pred = self.model(s_batch)
                loss = F.mse_loss(pred, t_batch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item() * len(s_batch)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    CATE epoch {epoch+1:3d}  "
                      f"MSE: {total_loss / len(states):.8f}")
        self.model.eval()

    def fit_from_outcomes(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 16384,
        verbose: bool = True,
    ):
        """
        Train CATE directly from (state, action, reward) tuples.

        Uses a pseudo-uplift approach: for each sample, the target is
        reward(a) - mean_reward(other actions).  This is noisier than
        the uplift table approach but works when no precomputed table
        is available (e.g. Criteo dataset).

        Parameters
        ----------
        states : (N, D) state vectors.
        actions : (N,) taken actions.
        rewards : (N,) observed rewards.
        """
        # Compute per-action mean reward as a baseline
        action_mean_reward = np.zeros(self.n_actions, dtype=np.float32)
        action_counts = np.zeros(self.n_actions, dtype=np.float32)
        for a, r in zip(actions, rewards):
            action_mean_reward[a] += r
            action_counts[a] += 1
        nonzero = action_counts > 0
        action_mean_reward[nonzero] /= action_counts[nonzero]
        global_mean = rewards.mean()

        # Build pseudo-uplift targets: (N, n_actions)
        targets = np.full((len(states), self.n_actions),
                          -global_mean, dtype=np.float32)
        for i in range(len(states)):
            a = actions[i]
            r = rewards[i]
            # For the taken action: observed reward - global mean
            targets[i, a] = r - global_mean

        S = torch.FloatTensor(states)
        T = torch.FloatTensor(targets)
        loader = DataLoader(TensorDataset(S, T), batch_size=batch_size,
                            shuffle=True, pin_memory=True)

        if verbose:
            print(f"  [CATE] Training from outcomes on {len(states):,} samples")

        self.model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for s_batch, t_batch in loader:
                s_batch = s_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                pred = self.model(s_batch)
                loss = F.mse_loss(pred, t_batch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item() * len(s_batch)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    CATE epoch {epoch+1:3d}  "
                      f"MSE: {total_loss / len(states):.8f}")
        self.model.eval()

    @torch.no_grad()
    def predict(self, states: np.ndarray,
                batch_size: int = 65536) -> np.ndarray:
        """
        Predict CATE(x, a) for all actions.

        Returns
        -------
        cate_scores : (N, n_actions) estimated treatment effects.
        """
        N = len(states)
        all_cate = np.empty((N, self.n_actions), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            S = torch.FloatTensor(states[start:end]).to(self.device)
            all_cate[start:end] = self.model(S).cpu().numpy()

        return all_cate

    def segment_users(
        self,
        states: np.ndarray,
        user_ids: np.ndarray,
        uplift_threshold: float = 0.0,
    ) -> dict:
        """
        Segment users into uplift quadrants based on predicted CATE.

        Returns
        -------
        segments : dict with keys:
            'user_segments' : (n_unique_users,) array of segment labels
            'segment_counts' : dict mapping segment name to count
            'per_sample_cate' : (N, n_actions) predicted CATE scores
        """
        cate_scores = self.predict(states)
        mean_cate = cate_scores.mean(axis=1)  # (N,)

        # Per-user aggregation
        unique_users = np.unique(user_ids)
        user_mean_cate = np.zeros(unique_users.max() + 1, dtype=np.float32)
        user_counts = np.zeros(unique_users.max() + 1, dtype=np.float32)
        for i, uid in enumerate(user_ids):
            user_mean_cate[uid] += mean_cate[i]
            user_counts[uid] += 1
        nonzero = user_counts > 0
        user_mean_cate[nonzero] /= user_counts[nonzero]

        # Segment: positive uplift = Persuadable, negative = Sleeping Dog
        segment_labels = np.where(
            user_mean_cate > uplift_threshold, 0, 3
        )  # 0=Persuadable, 3=Sleeping Dog (simplified two-class)

        segment_names = {0: "Persuadable", 3: "Sleeping Dog"}
        segment_counts = {
            name: int((segment_labels[unique_users] == sid).sum())
            for sid, name in segment_names.items()
        }

        return {
            "user_segments": segment_labels,
            "segment_counts": segment_counts,
            "per_sample_cate": cate_scores,
        }

    def uplift_weighted_rewards(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        uplift_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Create uplift-weighted rewards for BCQ training.

        Combined reward = (1 - w) * raw_reward + w * CATE(x, a_taken)

        This teaches the Q-network to optimise for treatment effect,
        not just raw response.  Users with negative uplift (Sleeping Dogs)
        get penalised even if they clicked.

        Parameters
        ----------
        states : (N, D)
        actions : (N,)
        rewards : (N,)
        uplift_weight : float, blending weight for CATE component.

        Returns
        -------
        weighted_rewards : (N,) blended rewards.
        """
        cate_scores = self.predict(states)                  # (N, A)
        cate_taken = cate_scores[np.arange(len(actions)), actions]  # (N,)

        # Normalise CATE to [0, 1] range for blending
        cate_min = cate_taken.min()
        cate_max = cate_taken.max()
        if cate_max - cate_min > 1e-8:
            cate_norm = (cate_taken - cate_min) / (cate_max - cate_min)
        else:
            cate_norm = np.full_like(cate_taken, 0.5)

        weighted = (1.0 - uplift_weight) * rewards + uplift_weight * cate_norm
        return weighted.astype(np.float32)
