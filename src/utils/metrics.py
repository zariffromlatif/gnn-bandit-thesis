"""
Evaluation metrics and reward model for the GNN-Bandit framework.

Provides:
  1. A reward model r_hat(x, a) needed by the DM and DR OPE estimators.
  2. A unified ``evaluate_policy`` function that runs all OPE estimators
     and returns a results table.
  3. Uplift segmentation (Persuadable / Sure Thing / Lost Cause / Sleeping Dog)
     for the Sleeping Dogs analysis.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..ope.estimators import evaluate_all, OPEResult


# ============================================================================
# Reward Model  r_hat(x, a)
# ============================================================================

class RewardModel:
    """
    Learns  r_hat(x, a) = E[r | x, a]  from logged data.

    This is a simple regression model needed for the Direct Method and
    Doubly Robust OPE estimators.  It predicts the expected reward for
    every (state, action) pair.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128,
                 device: str = "cpu"):
        self.n_actions = n_actions
        self.device = torch.device(device)

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_actions),
            nn.Sigmoid(),         # rewards are in [0, 1]
        ).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, states: np.ndarray, actions: np.ndarray,
            rewards: np.ndarray, n_epochs: int = 30,
            batch_size: int = 16384, verbose: bool = True):
        """Train the reward model on logged (s, a, r) tuples."""
        S = torch.FloatTensor(states).to(self.device)
        A = torch.LongTensor(actions).to(self.device)
        R = torch.FloatTensor(rewards).to(self.device)

        self.model.train()
        N = len(S)
        for epoch in range(n_epochs):
            total = 0.0
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = indices[start:end]
                s, a, r = S[idx], A[idx], R[idx]
                pred = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = F.binary_cross_entropy(pred, r.float())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total += loss.item() * len(s)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    RewardModel epoch {epoch+1:3d}  "
                      f"loss: {total / len(S):.6f}")
        self.model.eval()

    @torch.no_grad()
    def predict(self, states: np.ndarray) -> np.ndarray:
        """
        Predict r_hat(x, a) for all actions.

        Returns
        -------
        predictions : (N, n_actions) array of predicted rewards.
        """
        S = torch.FloatTensor(states).to(self.device)
        return self.model(S).cpu().numpy()


# ============================================================================
# Unified policy evaluation
# ============================================================================

def evaluate_policy(
    policy_probs: np.ndarray,
    rewards:      np.ndarray,
    propensities: np.ndarray,
    actions:      np.ndarray,
    n_actions:    int,
    reward_model_preds: Optional[np.ndarray] = None,
    clip: float = 100.0,
    label: str = "",
) -> dict[str, OPEResult]:
    """
    Evaluate a policy using all available OPE estimators.

    Parameters
    ----------
    policy_probs       : (N, A) action probabilities under the evaluation policy.
    rewards            : (N,)   observed rewards from the logging policy.
    propensities       : (N,)   logging policy propensity scores.
    actions            : (N,)   actions taken by the logging policy.
    n_actions          : int    number of actions.
    reward_model_preds : (N, A) predicted rewards (optional — needed for DM/DR).
    clip               : float  importance weight clipping.
    label              : str    label prefix for printing.

    Returns
    -------
    results : dict mapping estimator name to OPEResult.
    """
    results = evaluate_all(
        rewards=rewards,
        pi_new=policy_probs,
        pi_old=propensities,
        actions=actions,
        n_actions=n_actions,
        reward_model=reward_model_preds,
        clip=clip,
    )

    if label:
        print(f"\n  {label}:")
        for name, res in results.items():
            print(f"    {res}")

    return results


# ============================================================================
# Uplift segmentation (Sleeping Dogs analysis)
# ============================================================================

def segment_users(
    uplift_values: np.ndarray,
    baseline_response: np.ndarray,
    uplift_threshold: float = 0.0,
    response_threshold: float = 0.5,
) -> np.ndarray:
    """
    Segment users into the four classic uplift quadrants.

    Parameters
    ----------
    uplift_values     : (N,) estimated treatment effect per user.
    baseline_response : (N,) response probability WITHOUT treatment.
    uplift_threshold  : cutoff for positive/negative uplift.
    response_threshold: cutoff for high/low baseline response.

    Returns
    -------
    segments : (N,) array with labels:
        0 = Persuadable    (low baseline, positive uplift)  → INTERVENE
        1 = Sure Thing      (high baseline, positive uplift) → save budget
        2 = Lost Cause      (low baseline, negative uplift)  → don't bother
        3 = Sleeping Dog    (high baseline, negative uplift)  → DO NOT TOUCH
    """
    segments = np.zeros(len(uplift_values), dtype=np.int32)

    pos_uplift  = uplift_values > uplift_threshold
    high_base   = baseline_response > response_threshold

    segments[(~high_base) & pos_uplift]  = 0   # Persuadable
    segments[high_base & pos_uplift]     = 1   # Sure Thing
    segments[(~high_base) & (~pos_uplift)] = 2 # Lost Cause
    segments[high_base & (~pos_uplift)]  = 3   # Sleeping Dog

    return segments


SEGMENT_NAMES = {
    0: "Persuadable",
    1: "Sure Thing",
    2: "Lost Cause",
    3: "Sleeping Dog",
}


def sleeping_dogs_analysis(
    policy_probs: np.ndarray,
    user_ids:     np.ndarray,
    uplift_table: np.ndarray,
    n_actions:    int,
) -> dict:
    """
    Analyse how the policy treats Sleeping Dog users.

    A good policy should assign *low* intervention probability to
    users with negative uplift (Sleeping Dogs).

    Parameters
    ----------
    policy_probs : (N, A) action distribution for each sample.
    user_ids     : (N,) user indices.
    uplift_table : (n_users, n_actions) uplift values per user-action pair.
    n_actions    : int

    Returns
    -------
    analysis : dict with segment-level statistics.
    """
    N = len(user_ids)

    # Per-sample best uplift and mean uplift
    user_uplifts = uplift_table[user_ids]                   # (N, A)
    mean_uplift  = user_uplifts.mean(axis=1)                # (N,)

    # Simple segmentation by mean uplift sign
    is_sleeping_dog = mean_uplift < 0
    is_persuadable  = mean_uplift > 0

    # Max action probability assigned by the policy (how confidently it intervenes)
    max_prob = policy_probs.max(axis=1)

    results = {
        "n_total":       N,
        "n_sleeping_dog": int(is_sleeping_dog.sum()),
        "n_persuadable":  int(is_persuadable.sum()),
        "avg_max_prob_sleeping_dog": float(
            max_prob[is_sleeping_dog].mean()) if is_sleeping_dog.any() else 0.0,
        "avg_max_prob_persuadable": float(
            max_prob[is_persuadable].mean()) if is_persuadable.any() else 0.0,
        "avg_uplift_sleeping_dog": float(
            mean_uplift[is_sleeping_dog].mean()) if is_sleeping_dog.any() else 0.0,
        "avg_uplift_persuadable": float(
            mean_uplift[is_persuadable].mean()) if is_persuadable.any() else 0.0,
    }

    return results
