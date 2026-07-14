"""
Batch-Constrained Contextual Bandit (BCQ) for retention intervention selection.

Adapts Fujimoto et al. (ICML 2019) from full MDP offline RL to the simpler
contextual-bandit setting (single-step decisions, no transitions).

Architecture
------------
1. **Behavioural Cloning (BC) model** --- P_beta(a | s)
   Learns the logging policy's action distribution from historical data.
   Acts as a *safety filter*: the agent only considers actions the logging
   policy took with probability above a threshold tau.

2. **Q-Network** --- Q(s, a)
   Estimates the expected reward for each (state, action) pair.

3. **Policy** --- pi(s) = argmax_{a : P_beta(a|s) > tau} Q(s, a)
   Selects the highest-value action among those the BC model deems plausible.
   For OPE we output a *softmax* distribution over the constrained action set.

Threshold Strategy
------------------
The BCQ threshold must be calibrated to the action space.  With A actions and
a near-uniform logging policy, each action gets probability ~1/A.  An absolute
threshold tau > 1/A would mask out ALL actions, producing NaN.

We use a **relative** approach:  keep the top-K actions per sample where K is
determined by the BC model, plus a hard floor ensuring at least ``min_actions``
survive.  The ``threshold_ratio`` parameter sets tau = threshold_ratio / A,
so it adapts automatically to the action space size.

Hybrid Scoring
--------------
When item embeddings are provided, the agent combines learned Q-values with
explicit GNN dot-product similarity (following Neural Collaborative Filtering,
He et al. WWW 2017).  Both signals are z-score normalised per sample so
neither dominates, then summed with a tuneable ``hybrid_weight``.

    score(s, a) = z_norm(Q(s,a)) + hybrid_weight * z_norm(e_u . e_a)

This is critical because the Q-network maps states to anonymous action
outputs; it cannot easily learn the dot-product structure implicitly.
The explicit dot-product provides collaborative structure while Q-values
contribute reward-based corrections.

References
----------
- Fujimoto et al., "Off-Policy Deep Reinforcement Learning without
  Exploration", ICML 2019.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# Network components
# ============================================================================

class _MLP(nn.Module):
    """Simple feed-forward network used for both BC and Q."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256,
                 n_hidden: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# BCQ Agent
# ============================================================================

class BCQAgent:
    """
    Batch-Constrained Q-Learning agent for contextual bandits.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state vector (context features + GNN embedding).
    n_actions : int
        Number of discrete actions (e.g. 80 items for OBD-all, 2 for Criteo).
    hidden : int
        Hidden layer width.  Default 256.
    n_hidden : int
        Number of hidden layers.  Default 2.
    threshold_ratio : float
        BCQ constraint expressed as a *fraction* of (1 / n_actions).
        Effective threshold = threshold_ratio / n_actions.
        With ratio=0.3 and 80 actions: tau = 0.00375.
        Default 0.3.
    min_actions : int
        Minimum number of actions that must survive the mask per sample.
        If fewer pass the threshold, the top-min_actions by BC probability
        are kept.  Prevents the all-masked NaN catastrophe.  Default 5.
    lr : float
        Learning rate.  Default 1e-3.
    temperature : float
        Softmax temperature for stochastic policy (used in OPE).  Default 0.1.
    item_embeddings : np.ndarray, optional
        (n_actions, K) item embeddings from LightGCN.  When provided, the
        agent combines Q-values with dot-product similarity (hybrid scoring).
    gnn_embed_dim : int
        Dimensionality of GNN embeddings appended to the state vector.
        Used to extract the user embedding from the last *gnn_embed_dim*
        dimensions of the state.  Default 64.
    hybrid_weight : float
        Weight for the dot-product component in hybrid scoring.
        0.0 = pure Q-values, 1.0 = equal weight.  Default 1.0.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 256,
        n_hidden: int = 2,
        threshold_ratio: float = 0.3,
        min_actions: int = 5,
        lr: float = 1e-3,
        temperature: float = 0.1,
        item_embeddings: Optional[np.ndarray] = None,
        gnn_embed_dim: int = 64,
        hybrid_weight: float = 1.0,
        num_quantiles: int = 32,
        cvar_alpha: float = 0.10,
        device: str = "cpu",
    ):
        self.state_dim       = state_dim
        self.n_actions       = n_actions
        self.threshold_ratio = threshold_ratio
        self.min_actions     = min(min_actions, n_actions)
        self.temperature     = temperature
        self.device          = torch.device(device)
        self.gnn_embed_dim   = gnn_embed_dim
        self.hybrid_weight   = hybrid_weight
        self.num_quantiles   = num_quantiles
        self.cvar_alpha      = cvar_alpha

        # Effective threshold adapts to action space size
        self.threshold = threshold_ratio / n_actions

        # Cache item embeddings for hybrid scoring (not a learnable param)
        if item_embeddings is not None:
            self._item_emb = torch.FloatTensor(item_embeddings).to(self.device)
        else:
            self._item_emb = None

        # Behavioural Cloning model:  P_beta(a | s)
        self.bc_model = _MLP(state_dim, n_actions, hidden, n_hidden).to(self.device)
        self.bc_optim = torch.optim.Adam(self.bc_model.parameters(), lr=lr)

        # Q-network:  Q(s, a)  → outputs quantiles for all actions at once
        self.q_net = _MLP(state_dim, n_actions * num_quantiles, hidden, n_hidden).to(self.device)
        self.q_optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        n_epochs_bc: int = 30,
        n_epochs_q: int = 50,
        batch_size: int = 2048,
        verbose: bool = True,
    ) -> dict:
        """
        Two-phase training:
          1. Train the BC model on (states, actions).
          2. Train the Q-network on (states, actions, rewards), with the BC
             model frozen.

        Returns
        -------
        history : dict with keys "bc_loss", "q_loss" — lists of per-epoch losses.
        """
        # --- Prepare tensors ---
        S = torch.FloatTensor(states)
        A = torch.LongTensor(actions)
        R = torch.FloatTensor(rewards)

        dataset = TensorDataset(S, A, R)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             drop_last=False, pin_memory=True)

        history = {"bc_loss": [], "q_loss": []}

        # ---- Phase 1: Behavioural Cloning ----
        if verbose:
            print(f"  [BCQ] Phase 1: Behavioural Cloning ({n_epochs_bc} epochs)")

        self.bc_model.train()
        for epoch in range(n_epochs_bc):
            epoch_loss = 0.0
            for s_batch, a_batch, _ in loader:
                s_batch = s_batch.to(self.device)
                a_batch = a_batch.to(self.device)
                logits = self.bc_model(s_batch)
                loss = F.cross_entropy(logits, a_batch)
                self.bc_optim.zero_grad()
                loss.backward()
                self.bc_optim.step()
                epoch_loss += loss.item() * len(s_batch)
            epoch_loss /= len(dataset)
            history["bc_loss"].append(epoch_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1:3d}  BC loss: {epoch_loss:.4f}")

        # ---- Phase 2: Q-Network ----
        if verbose:
            print(f"  [BCQ] Phase 2: Q-Network ({n_epochs_q} epochs)")

        self.bc_model.eval()     # freeze BC
        self.q_net.train()
        for epoch in range(n_epochs_q):
            epoch_loss = 0.0
            for s_batch, a_batch, r_batch in loader:
                s_batch = s_batch.to(self.device)
                a_batch = a_batch.to(self.device)
                r_batch = r_batch.to(self.device)
                # q_all shape: (B, n_actions * num_quantiles)
                q_all = self.q_net(s_batch)
                B = q_all.size(0)
                # Reshape to (B, n_actions, num_quantiles)
                q_all = q_all.view(B, self.n_actions, self.num_quantiles)
                
                # Gather quantiles for the taken action: shape (B, num_quantiles)
                q_taken = q_all[torch.arange(B), a_batch, :]

                # Quantile Huber Loss
                # r_batch shape: (B,). Expand to (B, num_quantiles)
                r_expanded = r_batch.unsqueeze(1).expand_as(q_taken)
                
                # Compute TD errors
                td_error = r_expanded - q_taken
                
                # Huber loss with delta (kappa) = 1.0
                huber_loss = F.huber_loss(q_taken, r_expanded, reduction='none', delta=1.0)
                
                # Quantile midpoints (tau): shape (num_quantiles,)
                tau = (torch.arange(self.num_quantiles).float().to(self.device) + 0.5) / self.num_quantiles
                # Expand tau to (B, num_quantiles)
                tau = tau.unsqueeze(0).expand_as(q_taken)
                
                # QR-DQN Loss
                quantile_weight = torch.abs(tau - (td_error < 0).float())
                quantile_loss = quantile_weight * huber_loss
                
                # Sum over quantiles, mean over batch
                loss = quantile_loss.sum(dim=1).mean()

                self.q_optim.zero_grad()
                loss.backward()
                self.q_optim.step()
                epoch_loss += loss.item() * len(s_batch)
            epoch_loss /= len(dataset)
            history["q_loss"].append(epoch_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1:3d}  Q  loss: {epoch_loss:.4f}")

        self.q_net.eval()
        return history

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------

    def _safe_mask(self, bc_probs: torch.Tensor) -> torch.Tensor:
        """
        Build the BCQ action mask with a safety floor.

        1. Threshold mask: keep actions where P_beta(a|s) >= tau.
        2. Safety floor:   if fewer than ``min_actions`` survive,
           fall back to keeping the top-``min_actions`` by BC probability.

        Returns a boolean mask of shape (N, n_actions).
        """
        mask = bc_probs >= self.threshold                           # (N, A)

        # Check which rows have too few surviving actions
        n_surviving = mask.sum(dim=1)                               # (N,)
        need_fix = n_surviving < self.min_actions                   # (N,)

        if need_fix.any():
            # For those rows, keep the top-min_actions by BC probability
            _, top_idx = bc_probs[need_fix].topk(self.min_actions, dim=1)
            fix_mask = torch.zeros_like(bc_probs[need_fix], dtype=torch.bool)
            fix_mask.scatter_(1, top_idx, True)
            mask[need_fix] = fix_mask

        return mask

    # ------------------------------------------------------------------
    # Hybrid scoring
    # ------------------------------------------------------------------

    def _hybrid_scores(self, S: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Combine learned Q-values with GNN dot-product similarity.

        Both signals are z-score normalised per sample so neither dominates,
        then combined:  score = z(Q) + hybrid_weight * z(e_u . e_a)

        If no item embeddings were provided, returns Q unchanged.
        """
        if self._item_emb is None or self.hybrid_weight == 0:
            return q

        # Extract user embedding from the last gnn_embed_dim dims of state
        user_emb = S[:, -self.gnn_embed_dim:]                    # (B, K)
        dot_scores = user_emb @ self._item_emb.T                 # (B, A)

        # Z-score normalise across actions (per sample)
        def _znorm(x):
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.std(dim=1, keepdim=True).clamp(min=1e-8)
            return (x - mu) / sigma

        return _znorm(q) + self.hybrid_weight * _znorm(dot_scores)

    def _compute_cvar(self, q_quantiles: torch.Tensor) -> torch.Tensor:
        """
        Compute Conditional Value at Risk (CVaR) from quantiles.
        q_quantiles shape: (B, A, num_quantiles)
        Returns: (B, A) expected CVaR values
        """
        # Sort quantiles to ensure we get the true bottom alpha%
        sorted_q, _ = torch.sort(q_quantiles, dim=2)
        # Calculate how many quantiles represent the worst alpha%
        k = max(1, int(self.num_quantiles * self.cvar_alpha))
        # Take the mean of the bottom k quantiles (worst-case scenarios)
        return sorted_q[:, :, :k].mean(dim=2)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> int:
        """
        Select the BCQ-constrained best action for a single state.
        Now uses CVaR instead of expected value.
        """
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # BC probabilities
        bc_probs = F.softmax(self.bc_model(s), dim=1)           # (1, A)
        mask = self._safe_mask(bc_probs).squeeze(0)              # (A,)

        # Q-values with optional hybrid scoring
        q_all = self.q_net(s).view(1, self.n_actions, self.num_quantiles)
        q_cvar = self._compute_cvar(q_all)
        q = self._hybrid_scores(s, q_cvar).squeeze(0)

        # Mask out unlikely actions with -inf
        q[~mask] = float("-inf")
        return int(q.argmax().item())

    @torch.no_grad()
    def action_probabilities(
        self,
        states: np.ndarray,
        temperature: Optional[float] = None,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """
        Compute pi(a | s) for OPE.
        """
        temp = temperature or self.temperature
        N = len(states)
        all_probs = np.empty((N, self.n_actions), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            S = torch.FloatTensor(states[start:end]).to(self.device)

            bc_probs = F.softmax(self.bc_model(S), dim=1)       # (B, A)
            mask = self._safe_mask(bc_probs)                     # (B, A)

            q_all = self.q_net(S).view(-1, self.n_actions, self.num_quantiles)
            q_cvar = self._compute_cvar(q_all)
            q = self._hybrid_scores(S, q_cvar)                   # (B, A)
            q[~mask] = float("-inf")

            probs = F.softmax(q / temp, dim=1)                   # (B, A)
            all_probs[start:end] = probs.cpu().numpy()

        return all_probs

    @torch.no_grad()
    def q_values(self, states: np.ndarray) -> np.ndarray:
        """Return raw CVaR(s, a) for all actions.  Shape (N, n_actions)."""
        S = torch.FloatTensor(states).to(self.device)
        q_all = self.q_net(S).view(-1, self.n_actions, self.num_quantiles)
        return self._compute_cvar(q_all).cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save both networks to a checkpoint file."""
        torch.save({
            "bc_state_dict": self.bc_model.state_dict(),
            "q_state_dict":  self.q_net.state_dict(),
            "config": {
                "state_dim":       self.state_dim,
                "n_actions":       self.n_actions,
                "threshold":       self.threshold,
                "threshold_ratio": self.threshold_ratio,
                "min_actions":     self.min_actions,
                "temperature":     self.temperature,
                "hybrid_weight":   self.hybrid_weight,
                "gnn_embed_dim":   self.gnn_embed_dim,
                "num_quantiles":   self.num_quantiles,
                "cvar_alpha":      self.cvar_alpha,
            },
        }, path)

    def load(self, path: str):
        """Load networks from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.bc_model.load_state_dict(ckpt["bc_state_dict"])
        self.q_net.load_state_dict(ckpt["q_state_dict"])
        self.bc_model.eval()
        self.q_net.eval()
