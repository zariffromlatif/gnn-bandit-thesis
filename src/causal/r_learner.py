"""
Robinson Residual-on-Residual R-Learner for Multi-Arm Graph Bandits.

Implements Neyman-orthogonal double machine learning for heterogeneous treatment
effects (HTE) in offline bandit and recommendation settings, following:
  - Robinson, P. M. "Root-N-consistent semiparametric regression." Econometrica (1988).
  - Nie, X. & Wager, S. "Quasi-oracle estimation of heterogeneous treatment effects."
    Biometrika 108.2 (2021): 299-319.
  - Chernozhukov et al. "Double/debiased machine learning for treatment and structural
    parameters." The Econometrics Journal 21.1 (2018): C1-C68.

Mathematical Formulation:
-------------------------
For each sample i with context X_i, taken discrete action A_i in {0, ..., K-1},
and observed reward Y_i in {0, 1}:
  1. Nuisance Outcome Model: m(X) = E[Y | X]
  2. Nuisance Propensity Model: e(X, a) = P(A = a | X)
  3. Robinson Residuals:
       Y_tilde_i = Y_i - hat{m}(X_i)
       W_tilde_{i, A_i} = 1 - hat{e}(X_i, A_i)
  4. Multi-Arm R-Objective:
       min_tau (1 / N) sum_{i=1}^N ( Y_tilde_i - tau(X_i, A_i) * W_tilde_{i, A_i} )^2

Cross-fitting (K-fold) ensures out-of-fold residualization, eliminating
in-sample overfitting bias and guaranteeing root-N convergence.
"""

from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _NuisanceOutcomeNet(nn.Module):
    """Predicts baseline expected reward m(X) = E[Y | X]."""

    def __init__(self, state_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _NuisancePropensityNet(nn.Module):
    """Predicts logging policy propensity e(X, a) = P(A = a | X)."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _RLearnerNet(nn.Module):
    """CATE network tau(X, a) predicting treatment effect across all actions."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.tau_head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.trunk(x)
        return self.tau_head(phi)


class RLearner:
    """
    Neyman-Orthogonal Multi-Arm R-Learner with Cross-Fitting.

    Parameters
    ----------
    state_dim : int
        Dimensionality of input state vector.
    n_actions : int
        Number of discrete actions.
    hidden : int
        Hidden layer width for neural components (default 128).
    lr : float
        Learning rate (default 1e-3).
    device : str
        Target device ("cuda" or "cpu").
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.lr = lr
        self.hidden = hidden

        self.tau_model = _RLearnerNet(state_dim, n_actions, hidden).to(self.device)
        self.optim = torch.optim.Adam(self.tau_model.parameters(), lr=lr)

    def _train_nuisance_outcome(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        n_epochs: int = 15,
        batch_size: int = 8192,
    ) -> _NuisanceOutcomeNet:
        """Fit outcome nuisance model m(X) = E[Y | X]."""
        net = _NuisanceOutcomeNet(self.state_dim, self.hidden).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        n = len(states)
        net.train()

        for _ in range(n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                pred = net(states[idx])
                loss = F.mse_loss(pred, rewards[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        return net

    def _train_nuisance_propensity(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        n_epochs: int = 15,
        batch_size: int = 8192,
    ) -> _NuisancePropensityNet:
        """Fit propensity nuisance model e(X, a) = P(A = a | X)."""
        net = _NuisancePropensityNet(self.state_dim, self.n_actions, self.hidden).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        n = len(states)
        net.train()

        for _ in range(n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                logits = net(states[idx])
                loss = F.cross_entropy(logits, actions[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        return net

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        propensities: Optional[np.ndarray] = None,
        n_folds: int = 2,
        nuisance_epochs: int = 15,
        tau_epochs: int = 40,
        batch_size: int = 8192,
        verbose: bool = True,
    ) -> None:
        """
        Fit the R-Learner using K-fold cross-fitting.

        Parameters
        ----------
        states : (N, D) state vectors.
        actions : (N,) discrete actions.
        rewards : (N,) observed rewards.
        propensities : Optional (N,) logging policy propensities if known.
        n_folds : Number of cross-fitting folds (default: 2).
        nuisance_epochs : Epochs for training nuisance estimators.
        tau_epochs : Epochs for minimizing the Neyman-orthogonal R-loss.
        batch_size : Mini-batch size.
        verbose : Whether to log progress.
        """
        N = len(states)
        S = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        A = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        Y = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)

        # Step 1: K-fold cross-fitting for nuisance estimates
        m_hat = np.zeros(N, dtype=np.float32)
        e_hat = np.zeros(N, dtype=np.float32)

        indices = np.arange(N)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)

        if verbose:
            print(f"  [R-Learner] Cross-fitting {n_folds} folds on {N:,} samples...")

        for f_idx in range(n_folds):
            val_idx = folds[f_idx]
            train_idx = np.setdiff1d(indices, val_idx)

            s_train, y_train, a_train = S[train_idx], Y[train_idx], A[train_idx]
            s_val, a_val = S[val_idx], A[val_idx]

            # Fit outcome model m(X) on train_fold
            out_model = self._train_nuisance_outcome(
                s_train, y_train, n_epochs=nuisance_epochs, batch_size=batch_size
            )
            with torch.no_grad():
                m_hat[val_idx] = out_model(s_val).cpu().numpy()

            # Propensity model e(X, A)
            if propensities is not None:
                e_hat[val_idx] = propensities[val_idx]
            else:
                prop_model = self._train_nuisance_propensity(
                    s_train, a_train, n_epochs=nuisance_epochs, batch_size=batch_size
                )
                with torch.no_grad():
                    logits = prop_model(s_val)
                    probs = F.softmax(logits, dim=-1)
                    e_val = probs.gather(1, a_val.unsqueeze(1)).squeeze(1)
                    e_hat[val_idx] = e_val.cpu().numpy()

        # Step 2: Compute Robinson residuals
        # Outcome residual: Y_tilde = Y - m_hat
        # Action residual for taken action: W_tilde = 1 - e_hat
        eps = 1e-4
        e_hat = np.clip(e_hat, eps, 1.0 - eps)
        y_tilde = (rewards - m_hat).astype(np.float32)
        w_tilde = (1.0 - e_hat).astype(np.float32)

        Y_res = torch.as_tensor(y_tilde, dtype=torch.float32, device=self.device)
        W_res = torch.as_tensor(w_tilde, dtype=torch.float32, device=self.device)

        if verbose:
            print(f"  [R-Learner] Minimizing multi-arm R-loss for {tau_epochs} epochs | "
                  f"Residual mean: {y_tilde.mean():.6f}, std: {y_tilde.std():.4f}")

        # Step 3: Train tau network on Robinson R-loss
        self.tau_model.train()
        for epoch in range(tau_epochs):
            perm = torch.randperm(N, device=self.device)
            total_loss = 0.0
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = perm[start:end]
                s_b = S[idx]
                a_b = A[idx]
                y_res_b = Y_res[idx]
                w_res_b = W_res[idx]

                # Forward pass: tau(X, a) for all actions
                tau_pred = self.tau_model(s_b)
                tau_taken = tau_pred.gather(1, a_b.unsqueeze(1)).squeeze(1)

                # Robinson objective: ( Y_tilde - tau * W_tilde )^2
                loss = F.mse_loss(tau_taken * w_res_b, y_res_b)

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.tau_model.parameters(), max_norm=1.0)
                self.optim.step()
                total_loss += loss.item() * len(s_b)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    R-Learner epoch {epoch+1:3d}  Loss: {total_loss / N:.8f}")

        self.tau_model.eval()

    @torch.no_grad()
    def predict(self, states: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        """
        Predict CATE tau(X, a) for all arms across query states.

        Returns
        -------
        cate_scores : (N, n_actions) estimated treatment effects.
        """
        N = len(states)
        all_cate = np.empty((N, self.n_actions), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            S_b = torch.as_tensor(states[start:end], dtype=torch.float32, device=self.device)
            pred = self.tau_model(S_b)
            all_cate[start:end] = pred.cpu().numpy()

        return all_cate

    def uplift_weighted_rewards(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        uplift_weight: float = 0.5,
    ) -> np.ndarray:
        """Blend raw rewards with R-Learner CATE estimates for policy training."""
        cate_scores = self.predict(states)
        cate_taken = cate_scores[np.arange(len(actions)), actions]

        # Normalise CATE to [0, 1] range
        c_min, c_max = cate_taken.min(), cate_taken.max()
        if c_max - c_min > 1e-8:
            cate_norm = (cate_taken - c_min) / (c_max - c_min)
        else:
            cate_norm = np.full_like(cate_taken, 0.5)

        weighted = (1.0 - uplift_weight) * rewards + uplift_weight * cate_norm
        return weighted.astype(np.float32)

    def segment_users(
        self,
        states: np.ndarray,
        user_ids: np.ndarray,
        baseline_responses: Optional[np.ndarray] = None,
        treatment_arm: Optional[int] = None,
        control_arm: Optional[int] = None,
        uplift_threshold: float = 0.0,
        response_threshold: Optional[float] = None,
        aggregation: str = "max",
    ) -> Dict[str, object]:
        """
        Segment users into the 4 classic uplift quadrants.

        Quadrants:
            0 = Persuadable  (low baseline, positive uplift)  → INTERVENE
            1 = Sure Thing    (high baseline, positive uplift) → ORGANIC CONVERT
            2 = Lost Cause    (low baseline, negative uplift)  → INEFFECTIVE
            3 = Sleeping Dog  (high baseline, negative uplift) → DO NOT TOUCH
        """
        cate_scores = self.predict(states)
        if treatment_arm is not None:
            if control_arm is not None:
                sample_uplift = cate_scores[:, treatment_arm] - cate_scores[:, control_arm]
            else:
                sample_uplift = cate_scores[:, treatment_arm]
        else:
            if control_arm is not None:
                rel_cate = cate_scores - cate_scores[:, [control_arm]]
            else:
                rel_cate = cate_scores

            if aggregation == "max":
                sample_uplift = rel_cate.max(axis=1)
            elif aggregation == "mean":
                sample_uplift = rel_cate.mean(axis=1)
            else:
                sample_uplift = rel_cate.max(axis=1)

        unique_users = np.unique(user_ids)
        max_uid = int(unique_users.max())
        user_uplift = np.zeros(max_uid + 1, dtype=np.float32)
        user_counts = np.zeros(max_uid + 1, dtype=np.float32)

        for i, uid in enumerate(user_ids):
            user_uplift[uid] += sample_uplift[i]
            user_counts[uid] += 1.0

        nonzero = user_counts > 0
        user_uplift[nonzero] /= user_counts[nonzero]

        user_baseline = np.zeros(max_uid + 1, dtype=np.float32)
        if baseline_responses is not None:
            for i, uid in enumerate(user_ids):
                user_baseline[uid] += baseline_responses[i]
            user_baseline[nonzero] /= user_counts[nonzero]

            user_bases = user_baseline[unique_users]
            if response_threshold is None:
                response_threshold = float(np.median(user_bases))
                if response_threshold == 0.0:
                    response_threshold = float(np.mean(user_bases))
                if response_threshold == 0.0:
                    response_threshold = 0.5
        else:
            if response_threshold is None:
                response_threshold = 0.5
            user_baseline.fill(0.0)

        u_uplift = user_uplift[unique_users]
        u_base = user_baseline[unique_users]

        pos_uplift = u_uplift > uplift_threshold
        high_base = u_base > response_threshold

        if baseline_responses is None:
            user_segments = np.where(pos_uplift, 0, 2)
        else:
            user_segments = np.zeros(len(unique_users), dtype=np.int32)
            user_segments[(~high_base) & pos_uplift] = 0   # Persuadable
            user_segments[high_base & pos_uplift] = 1      # Sure Thing
            user_segments[(~high_base) & (~pos_uplift)] = 2 # Lost Cause
            user_segments[high_base & (~pos_uplift)] = 3   # Sleeping Dog

        segment_names = {
            0: "Persuadable",
            1: "Sure Thing",
            2: "Lost Cause",
            3: "Sleeping Dog",
        }
        segment_counts = {
            name: int((user_segments == sid).sum())
            for sid, name in segment_names.items()
        }

        return {
            "user_segments": user_segments,
            "segment_counts": segment_counts,
            "per_sample_cate": cate_scores,
            "response_threshold": response_threshold,
        }
