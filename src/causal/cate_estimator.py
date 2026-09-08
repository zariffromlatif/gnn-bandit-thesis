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
from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class _CATENetwork(nn.Module):
    """MLP that predicts CATE(x, a) and treatment assignment.

    The treatment head uses spectral normalization to bound its
    Lipschitz constant, preventing the adversarial discriminator
    from overpowering the encoder during GRL training.
    """

    def __init__(self, input_dim: int, n_actions: int, hidden: int = 128,
                 n_hidden: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        self.encoder = nn.Sequential(*layers)
        self.uplift_head = nn.Linear(prev, n_actions)
        # Spectral norm bounds the discriminator's Lipschitz constant,
        # preventing mode collapse during adversarial training.
        self.treatment_head = nn.utils.spectral_norm(
            nn.Linear(prev, n_actions)
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        phi = self.encoder(x)
        uplift = self.uplift_head(phi)
        
        phi_rev = grad_reverse(phi, alpha)
        treatment = self.treatment_head(phi_rev)
        return uplift, treatment


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
        logged_actions: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 16384,
        cfr_lambda: float = 0.1,
        verbose: bool = True,
        observed_mask: Optional[np.ndarray] = None,
    ):
        """
        Train the CATE network to predict uplift values from states.

        The uplift table contains precomputed treatment effects from the
        randomised data. We train the neural network to generalise these
        estimates to unseen states using GNN-augmented features.

        Parameters
        ----------
        states : (N, D) augmented state vectors (context + GNN embedding).
        user_ids : (N,) user indices into the uplift table.
        uplift_table : (n_users, n_actions) precomputed uplift values.
        logged_actions: (N,) taken actions for CFR-GNN constraint.
        n_epochs : number of training epochs.
        batch_size : mini-batch size.
        cfr_lambda: strength of counterfactual regularisation.
        verbose : print training progress.
        observed_mask: optional (n_users, n_actions) boolean mask of observed entries.
        """
        # Build per-sample uplift targets
        targets = uplift_table[user_ids]  # (N, n_actions)

        # Build mask for sparse uplift tables to avoid zero-collapse
        if observed_mask is None and (uplift_table != 0.0).sum() < 0.3 * uplift_table.size:
            observed_mask = (uplift_table != 0.0)

        if observed_mask is not None:
            mask_per_sample = observed_mask[user_ids]
            M = torch.as_tensor(mask_per_sample, dtype=torch.bool, device=self.device)
        else:
            M = None

        S = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        T = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        A = torch.as_tensor(logged_actions, dtype=torch.long, device=self.device)

        if verbose:
            print(f"  [CATE] Training on {len(states):,} samples, "
                  f"{n_epochs} epochs | CFR lambda: {cfr_lambda}")

        self.model.train()
        N = len(S)
        warmup_epochs = int(n_epochs * 0.6)
        for epoch in range(n_epochs):
            # Curriculum α: ramp from 0 → cfr_lambda over first 60% of epochs.
            # This lets the encoder learn useful uplift representations
            # before adversarial pressure kicks in.
            if epoch < warmup_epochs:
                current_alpha = cfr_lambda * (epoch / warmup_epochs)
            else:
                current_alpha = cfr_lambda

            total_loss = 0.0
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = indices[start:end]
                s_batch, t_batch, a_batch = S[idx], T[idx], A[idx]
                
                pred_uplift, pred_treatment = self.model(s_batch, alpha=current_alpha)
                if M is not None:
                    m_batch = M[idx]
                    if m_batch.any():
                        loss_uplift = F.mse_loss(pred_uplift[m_batch], t_batch[m_batch])
                    else:
                        loss_uplift = F.mse_loss(pred_uplift, t_batch)
                else:
                    loss_uplift = F.mse_loss(pred_uplift, t_batch)
                loss_treatment = F.cross_entropy(pred_treatment, a_batch)
                
                loss = loss_uplift + loss_treatment
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
                total_loss += loss.item() * len(s_batch)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    CATE epoch {epoch+1:3d}  "
                      f"MSE: {total_loss / len(states):.8f}  "
                      f"alpha: {current_alpha:.4f}")
        self.model.eval()

    def fit_from_outcomes(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 16384,
        cfr_lambda: float = 0.1,
        verbose: bool = True,
    ):
        """
        Train CATE directly from (state, action, reward) tuples.

        Uses a residualized outcome formulation: for each sample, the target is
        reward(a) - global_mean. We compute MSE loss ONLY on the taken action,
        preventing unobserved counterfactual actions from being artificially
        forced to negative values.

        Parameters
        ----------
        states : (N, D) state vectors.
        actions : (N,) taken actions.
        rewards : (N,) observed rewards.
        cfr_lambda : strength of counterfactual regularisation.
        """
        global_mean = float(rewards.mean())
        residuals = (rewards - global_mean).astype(np.float32)

        S = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        R = torch.as_tensor(residuals, dtype=torch.float32, device=self.device)
        A = torch.as_tensor(actions, dtype=torch.long, device=self.device)

        if verbose:
            print(f"  [CATE] Training from outcomes on {len(states):,} samples | CFR lambda: {cfr_lambda}")

        self.model.train()
        N = len(S)
        warmup_epochs = int(n_epochs * 0.6)
        for epoch in range(n_epochs):
            # Curriculum α: ramp from 0 → cfr_lambda over first 60% of epochs.
            if epoch < warmup_epochs:
                current_alpha = cfr_lambda * (epoch / warmup_epochs)
            else:
                current_alpha = cfr_lambda

            total_loss = 0.0
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = indices[start:end]
                s_batch, r_batch, a_batch = S[idx], R[idx], A[idx]
                
                pred_uplift, pred_treatment = self.model(s_batch, alpha=current_alpha)
                # Compute uplift loss ONLY on the taken action
                pred_taken = pred_uplift.gather(1, a_batch.unsqueeze(1)).squeeze(1)
                loss_uplift = F.mse_loss(pred_taken, r_batch)
                loss_treatment = F.cross_entropy(pred_treatment, a_batch)
                
                loss = loss_uplift + loss_treatment
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
                total_loss += loss.item() * len(s_batch)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    CATE epoch {epoch+1:3d}  "
                      f"MSE: {total_loss / len(states):.8f}  "
                      f"alpha: {current_alpha:.4f}")
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
            # Only use the uplift head for prediction, ignore treatment predictions
            pred_uplift, _ = self.model(S)
            all_cate[start:end] = pred_uplift.cpu().numpy()

        return all_cate

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
    ) -> dict:
        """
        Segment users into the 4 causal uplift quadrants based on predicted CATE.

        Quadrants:
            0 = Persuadable  (low baseline, positive uplift)  → INTERVENE
            1 = Sure Thing    (high baseline, positive uplift) → ORGANIC CONVERT
            2 = Lost Cause    (low baseline, negative uplift)  → INEFFECTIVE
            3 = Sleeping Dog  (high baseline, negative uplift) → DO NOT TOUCH

        Parameters
        ----------
        states : (N, D) state vectors
        user_ids : (N,) user identifiers
        baseline_responses : Optional (N,) observed baseline response probabilities / rewards
        treatment_arm : Optional specific treatment action index to evaluate
        control_arm : Optional control action index to compute relative treatment effect (e.g. 0)
        uplift_threshold : cutoff for positive/negative uplift (default 0.0)
        response_threshold : cutoff for high/low baseline (if None, adaptive median/mean)
        aggregation : "max" for max potential uplift, or "mean" for average arm uplift

        Returns
        -------
        segments : dict with keys:
            'user_segments' : (n_unique_users,) array of quadrant IDs (0, 1, 2, 3)
            'segment_counts' : dict mapping segment names to count
            'per_sample_cate' : (N, n_actions) predicted CATE scores
            'response_threshold' : float value used
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

    def propagate_cate(
        self,
        raw_cate: np.ndarray,
        user_ids: np.ndarray,
        adj,
        n_nodes: Optional[int] = None,
        n_hops: int = 2,
        beta: float = 0.3,
    ) -> np.ndarray:
        """
        Graph-Propagated CATE (GP-CATE): smooth uplift estimates through the
        interaction graph for denoised, neighborhood-aware predictions.

        Formula:
            T^{(l+1)} = (1 - beta) * T^{(l)} + beta * A_norm * T^{(l)}
            C^{GP}_i = (1 - beta) * C_i + beta * T^{(L)}_{u_i}

        Parameters
        ----------
        raw_cate : (N, n_actions) raw model predictions
        user_ids : (N,) user index corresponding to each sample
        adj : scipy.sparse matrix or torch.sparse.Tensor of shape (n_nodes, n_nodes)
        n_nodes : total graph nodes (users + items). If None, inferred from adj.
        n_hops : number of graph propagation layers (default: 2)
        beta : graph smoothing coefficient in [0, 1] (default: 0.3)

        Returns
        -------
        propagated_cate : (N, n_actions) smoothed uplift estimates
        """
        if adj is None or beta <= 0.0 or n_hops <= 0:
            return raw_cate

        import scipy.sparse as sp

        if n_nodes is None:
            n_nodes = adj.shape[0]

        n_actions = raw_cate.shape[1]
        unique_users = np.unique(user_ids)
        max_user_idx = int(unique_users.max())

        # Step 1: Aggregate sample CATE to user-node level
        node_uplift = np.zeros((n_nodes, n_actions), dtype=np.float32)
        user_counts = np.zeros(max_user_idx + 1, dtype=np.float32)

        for i, uid in enumerate(user_ids):
            node_uplift[uid] += raw_cate[i]
            user_counts[uid] += 1.0

        nonzero = user_counts > 0
        for uid in unique_users:
            if user_counts[uid] > 0:
                node_uplift[uid] /= user_counts[uid]

        # Step 2: Symmetrically normalize adjacency if needed
        if sp.issparse(adj):
            adj_csr = adj.tocsr().astype(np.float32)
            rowsum = np.array(adj_csr.sum(axis=1)).flatten()
            d_inv_sqrt = np.power(np.maximum(rowsum, 1e-12), -0.5)
            d_mat = sp.diags(d_inv_sqrt)
            norm_adj = d_mat.dot(adj_csr).dot(d_mat)

            # Step 3: Multi-hop graph propagation
            T = node_uplift.copy()
            for _ in range(n_hops):
                T = (1.0 - beta) * T + beta * norm_adj.dot(T)
        else:
            # Dense numpy fallback
            rowsum = adj.sum(axis=1, keepdims=True)
            d_inv = np.power(np.maximum(rowsum, 1e-12), -0.5)
            norm_adj = d_inv * adj * d_inv.T
            T = node_uplift.copy()
            for _ in range(n_hops):
                T = (1.0 - beta) * T + beta * (norm_adj @ T)

        # Step 4: Map back smoothed node uplift to individual samples
        smoothed_user_cate = T[user_ids]  # (N, n_actions)
        propagated_cate = (1.0 - beta) * raw_cate + beta * smoothed_user_cate
        return propagated_cate.astype(np.float32)

    def uplift_weighted_rewards(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        uplift_weight: float = 0.5,
        user_ids: Optional[np.ndarray] = None,
        adj=None,
        n_nodes: Optional[int] = None,
        gp_cate_hops: int = 2,
        gp_cate_beta: float = 0.3,
    ) -> np.ndarray:
        """
        Create uplift-weighted rewards for BCQ training, optionally enhanced by GP-CATE.

        Combined reward = (1 - w) * raw_reward + w * CATE(x, a_taken)

        Parameters
        ----------
        states : (N, D)
        actions : (N,)
        rewards : (N,)
        uplift_weight : float, blending weight for CATE component.
        user_ids : Optional (N,) array of user indices for GP-CATE.
        adj : Optional graph adjacency matrix for GP-CATE.
        n_nodes : Optional total graph node count.
        gp_cate_hops : Number of propagation hops for GP-CATE.
        gp_cate_beta : Mixing factor for GP-CATE.

        Returns
        -------
        weighted_rewards : (N,) blended rewards.
        """
        cate_scores = self.predict(states)                  # (N, A)

        # Apply Graph-Propagated CATE if graph structure is provided
        if adj is not None and user_ids is not None:
            cate_scores = self.propagate_cate(
                cate_scores, user_ids, adj, n_nodes,
                n_hops=gp_cate_hops, beta=gp_cate_beta
            )

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
