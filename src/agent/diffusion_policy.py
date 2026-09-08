"""
Diff-GNN-Bandit: Score-Based Generative Diffusion Policy for Offline Graph Bandits.

Replaces rigid discrete action classification with a continuous score-based
diffusion model operating in the latent action embedding space.

Key Architectural Principles:
-----------------------------
1. Continuous Action Latent Space:
   Each discrete catalog item a in {0, ..., K-1} corresponds to a GNN node
   embedding e_a in R^d. If explicit item embeddings are absent, a learnable
   embedding dictionary or state projection is maintained.

2. Forward Diffusion Process:
   Adds scheduled Gaussian perturbation to the catalog item embedding:
     q(a_t | a_0) = N(a_t; sqrt(alpha_bar_t) * a_0, (1 - alpha_bar_t) * I)

3. Score / Denoising Network:
   Time-conditioned neural network epsilon_theta(a_t, s, t) with sinusoidal
   positional embeddings conditioned on the GNN-augmented state vector s.

4. Causal Uplift Guidance (Tweedie's Formula):
   Steers the reverse diffusion trajectory toward actions with maximum treatment
   effect (positive CATE) while avoiding harm to Sleeping Dogs:
     a_hat_0(a_t) = (1 / sqrt(alpha_bar_t)) * (a_t - sqrt(1 - alpha_bar_t) * epsilon_theta(a_t, s, t))
     Guided score: epsilon_guided = epsilon_theta - omega * sqrt(1 - alpha_bar_t) * grad_{a_t} CATE(s, a_hat_0)

5. Fast DDIM Reverse Sampler (Song et al., ICLR 2021):
   Deterministic/sub-sampled reverse trajectory completing in 15-25 steps
   to meet industrial sub-35ms SLA latency budgets.

6. Action Space Projection:
   Maps continuous reverse-diffused latent vectors back to discrete catalog
   items via inner-product scoring and temperature-scaled softmax.

References:
-----------
- Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
- Wang et al., "Diffusion Policies as an Expressive Policy Class for Offline RL", ICLR 2023.
- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023.
"""

import math
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class _DiffusionScoreNet(nn.Module):
    """
    Time-conditioned score network estimating noise epsilon_theta(a_t, s, t).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
        )

        in_dim = state_dim + action_dim + hidden
        layers = []
        prev = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, hidden), nn.Mish(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, action_dim))
        self.mid_mlp = nn.Sequential(*layers)

    def forward(
        self, action_t: torch.Tensor, state: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_mlp(timesteps)
        x = torch.cat([action_t, state, t_emb], dim=-1)
        return self.mid_mlp(x)


class DiffGNNBandit(nn.Module):
    """
    Score-Based Generative Diffusion Policy for Contextual Graph Bandits.

    Parameters
    ----------
    state_dim : int
        Dimensionality of state vector (context + GNN embedding).
    n_actions : int
        Number of discrete actions / catalog items.
    action_dim : int
        Dimensionality of the continuous latent action space (e.g. 64).
    item_embeddings : Optional[np.ndarray]
        Precomputed (n_actions, action_dim) item embeddings from LightGCN.
    num_timesteps : int
        Number of forward diffusion steps (default 100).
    ddim_steps : int
        Number of reverse inference steps (default 20).
    guidance_weight : float
        Strength of CATE guidance omega (default 1.0).
    temperature : float
        Softmax temperature for catalog projection (default 0.1).
    lr : float
        Learning rate (default 1e-3).
    device : str
        Target device ("cuda" or "cpu").
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        action_dim: int = 64,
        item_embeddings: Optional[np.ndarray] = None,
        num_timesteps: int = 100,
        ddim_steps: int = 20,
        guidance_weight: float = 1.0,
        temperature: float = 0.5,
        hidden: int = 256,
        n_hidden: int = 2,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps
        self.ddim_steps = ddim_steps
        self.guidance_weight = guidance_weight
        self.temperature = temperature
        self.device = torch.device(device)

        # Item embedding dictionary in latent action space
        if item_embeddings is not None:
            assert item_embeddings.shape[0] == n_actions
            emb = torch.as_tensor(item_embeddings, dtype=torch.float32)
            if emb.shape[1] != action_dim:
                self.action_dim = emb.shape[1]
                action_dim = emb.shape[1]
            self.item_embeddings = nn.Parameter(emb, requires_grad=False)
        else:
            self.item_embeddings = nn.Parameter(
                torch.randn(n_actions, action_dim) / math.sqrt(action_dim),
                requires_grad=True,
            )

        # Score network
        self.score_net = _DiffusionScoreNet(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden=hidden,
            n_hidden=n_hidden,
        ).to(self.device)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

        # Precompute diffusion schedule (bounded linear beta schedule)
        betas = torch.linspace(1e-4, 0.05, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.to(self.device)

    def q_sample(
        self, a_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion noise perturbation: q(a_t | a_0)."""
        if noise is None:
            noise = torch.randn_like(a_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * a_start + sqrt_one_minus_alpha * noise

    def loss(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes denoising score-matching loss."""
        B = len(states)
        a_0 = self.item_embeddings[actions]  # (B, action_dim)

        # Random timesteps uniformly distributed across schedule
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(a_0)

        a_t = self.q_sample(a_0, t, noise)
        pred_noise = self.score_net(a_t, states, t)

        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def ddim_sample(
        self,
        states: torch.Tensor,
        cate_scores: Optional[torch.Tensor] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Fast reverse diffusion using DDIM (deterministic ODE when eta=0.0).

        Parameters
        ----------
        states : (B, state_dim) batch of query states.
        cate_scores : Optional (B, n_actions) estimated CATE matrix for guidance.
        eta : Stochasticity parameter (0.0 = deterministic DDIM).

        Returns
        -------
        a_0_pred : (B, action_dim) generated continuous action latent vectors.
        """
        B = len(states)
        shape = (B, self.action_dim)
        a_t = torch.randn(shape, device=self.device)

        # Sub-sample time steps for fast sampling
        times = torch.linspace(0, self.num_timesteps - 1, steps=self.ddim_steps).long()
        time_pairs = list(zip(times[:-1], times[1:]))[::-1]

        for t_prev, t_curr in time_pairs:
            t_batch = torch.full((B,), t_curr, device=self.device, dtype=torch.long)
            eps_pred = self.score_net(a_t, states, t_batch)

            alpha_curr = self.alphas_cumprod[t_curr]
            alpha_prev = self.alphas_cumprod[t_prev]

            # Tweedie's formula: estimate a_0
            a_0_est = (a_t - math.sqrt(1.0 - alpha_curr) * eps_pred) / math.sqrt(alpha_curr)

            # Causal CATE guidance via analytical Tweedie score gradient
            if cate_scores is not None and self.guidance_weight > 0.0:
                affinities = torch.matmul(a_0_est, self.item_embeddings.t())  # (B, K)
                p = F.softmax(affinities / self.temperature, dim=-1)
                p_dot_tau = (p * cate_scores).sum(dim=-1, keepdim=True)       # (B, 1)
                # Direction: sum_k p_k * (tau_k - E_p[tau]) * e_k
                uplift_weights = p * (cate_scores - p_dot_tau)                 # (B, K)
                guidance_grad = torch.matmul(uplift_weights, self.item_embeddings) # (B, action_dim)
                norm_guidance = F.normalize(guidance_grad, dim=-1)
                # Scale guidance with diffusion schedule
                guidance_scale = math.sqrt(1.0 - alpha_prev.item()) if isinstance(alpha_prev, torch.Tensor) else math.sqrt(1.0 - alpha_prev)
                a_0_est = a_0_est + self.guidance_weight * guidance_scale * norm_guidance

            # DDIM update step
            dir_xt = math.sqrt(1.0 - alpha_prev - (eta ** 2) * (1.0 - alpha_prev) / (1.0 - alpha_curr)) * eps_pred
            noise = torch.randn_like(a_t) if eta > 0.0 else torch.zeros_like(a_t)
            sigma_t = eta * math.sqrt((1.0 - alpha_prev) / (1.0 - alpha_curr)) * math.sqrt(1.0 - alpha_curr / alpha_prev)

            a_t = math.sqrt(alpha_prev) * a_0_est + dir_xt + sigma_t * noise

        return a_0_est

    def action_probabilities(
        self,
        states: np.ndarray,
        cate_scores: Optional[np.ndarray] = None,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """
        Evaluate full action probability distribution pi(a | s) across query states.

        Returns
        -------
        probs : (N, n_actions) valid probability distribution for OPE evaluation.
        """
        self.eval()
        N = len(states)
        all_probs = np.empty((N, self.n_actions), dtype=np.float32)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            S_b = torch.as_tensor(states[start:end], dtype=torch.float32, device=self.device)
            C_b = (
                torch.as_tensor(cate_scores[start:end], dtype=torch.float32, device=self.device)
                if cate_scores is not None
                else None
            )

            # 1. Reverse diffuse continuous action vectors
            a_latent = self.ddim_sample(S_b, cate_scores=C_b)

            # 2. Project continuous vectors onto item embeddings via normalized cosine affinity
            a_norm = F.normalize(a_latent, dim=-1)
            item_norm = F.normalize(self.item_embeddings, dim=-1)
            logits = torch.matmul(a_norm, item_norm.t())  # (B, n_actions) in [-1.0, 1.0]
            probs = F.softmax(logits / self.temperature, dim=-1)

            all_probs[start:end] = probs.cpu().numpy()

        return all_probs

    def predict_action(
        self, states: np.ndarray, cate_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Greedy action recommendation: argmax_a pi(a | s)."""
        probs = self.action_probabilities(states, cate_scores=cate_scores)
        return probs.argmax(axis=1)

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        n_epochs: int = 30,
        batch_size: int = 2048,
        verbose: bool = True,
    ) -> None:
        """Train Diff-GNN-Bandit policy via score matching."""
        N = len(states)
        S = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        A = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        R = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)

        if verbose:
            print(f"  [Diff-GNN-Bandit] Training diffusion policy on {N:,} samples, {n_epochs} epochs | DDIM steps: {self.ddim_steps}")

        self.train()
        for epoch in range(n_epochs):
            perm = torch.randperm(N, device=self.device)
            total_loss = 0.0
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = perm[start:end]
                s_b, a_b, r_b = S[idx], A[idx], R[idx]

                loss = self.loss(s_b, a_b, r_b)
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optim.step()
                total_loss += loss.item() * len(s_b)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Diff-GNN-Bandit epoch {epoch+1:3d}  Score Loss: {total_loss / N:.8f}")

        self.eval()
