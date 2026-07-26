"""
Dynamic Batch-Constrained Contextual Bandit (Dynamic BCQ)
Extends BCQ to multi-step temporal transitions using a State Dynamics Model.
"""

from typing import Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agent.bcq import _MLP

class DynamicBCQAgent:
    """
    Multi-Step Risk-Averse Distributional BCQ.
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
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.threshold_ratio = threshold_ratio
        self.min_actions = min(min_actions, n_actions)
        self.temperature = temperature
        self.device = torch.device(device)
        self.gnn_embed_dim = gnn_embed_dim
        self.hybrid_weight = hybrid_weight
        self.num_quantiles = num_quantiles
        self.cvar_alpha = cvar_alpha
        self.gamma = gamma

        self.threshold = threshold_ratio / n_actions

        if item_embeddings is not None:
            self._item_emb = torch.FloatTensor(item_embeddings).to(self.device)
        else:
            self._item_emb = None

        self.bc_model = _MLP(state_dim, n_actions, hidden, n_hidden).to(self.device)
        self.bc_optim = torch.optim.Adam(self.bc_model.parameters(), lr=lr)

        self.q_net = _MLP(state_dim, n_actions * num_quantiles, hidden, n_hidden).to(self.device)
        self.q_optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
        # Target network for stable multi-step Q-learning
        self.q_target = _MLP(state_dim, n_actions * num_quantiles, hidden, n_hidden).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())

    def train(
        self,
        s_t: np.ndarray,
        a_t: np.ndarray,
        r_t: np.ndarray,
        s_next: np.ndarray,
        n_epochs_bc: int = 30,
        n_epochs_q: int = 50,
        batch_size: int = 2048,
        verbose: bool = True,
        tau_update: float = 0.005
    ) -> dict:
        """
        Two-phase training over explicit temporal transitions.
        """
        S = torch.as_tensor(s_t, dtype=torch.float32, device=self.device)
        A = torch.as_tensor(a_t, dtype=torch.long, device=self.device)
        R = torch.as_tensor(r_t, dtype=torch.float32, device=self.device)
        S_NEXT = torch.as_tensor(s_next, dtype=torch.float32, device=self.device)

        history = {"bc_loss": [], "q_loss": []}
        N = len(S)

        # ---- Phase 1: Behavioural Cloning ----
        if verbose:
            print(f"  [Dynamic BCQ] Phase 1: Behavioural Cloning ({n_epochs_bc} epochs)")

        self.bc_model.train()
        for epoch in range(n_epochs_bc):
            epoch_loss = 0.0
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = indices[start:end]
                s_batch, a_batch = S[idx], A[idx]
                logits = self.bc_model(s_batch)
                loss = F.cross_entropy(logits, a_batch)
                self.bc_optim.zero_grad()
                loss.backward()
                self.bc_optim.step()
                epoch_loss += loss.item() * len(s_batch)
            epoch_loss /= N
            history["bc_loss"].append(epoch_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1:3d}  BC loss: {epoch_loss:.4f}")

        # ---- Phase 2: Multi-Step Q-Network ----
        if verbose:
            print(f"  [Dynamic BCQ] Phase 2: Q-Network ({n_epochs_q} epochs)")

        self.bc_model.eval()
        self.q_net.train()
        for epoch in range(n_epochs_q):
            epoch_loss = 0.0
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = indices[start:end]
                s_batch, a_batch, r_batch, s_next_batch = S[idx], A[idx], R[idx], S_NEXT[idx]
                
                # 1. Current Quantiles
                q_all = self.q_net(s_batch)
                B = q_all.size(0)
                q_all = q_all.view(B, self.n_actions, self.num_quantiles)
                q_taken = q_all[torch.arange(B), a_batch, :] # (B, num_quantiles)

                # 2. Next State Target
                with torch.no_grad():
                    # Evaluate next actions via target network
                    q_next_all = self.q_target(s_next_batch).view(B, self.n_actions, self.num_quantiles)
                    # Use CVaR to select the best next action (risk-averse target)
                    q_next_cvar = self._compute_cvar(q_next_all)
                    next_actions = q_next_cvar.argmax(dim=1) # (B,)
                    
                    # Get the target quantiles for the chosen next action
                    target_quantiles = q_next_all[torch.arange(B), next_actions, :] # (B, num_quantiles)
                    
                    # Bellman target for quantiles
                    target_Z = r_batch.unsqueeze(1) + self.gamma * target_quantiles

                # 3. Quantile Huber Loss (cross-comparison of all quantiles)
                # target_Z is (B, 1, num_quantiles), q_taken is (B, num_quantiles, 1)
                target_Z = target_Z.unsqueeze(1)
                q_taken = q_taken.unsqueeze(2)
                
                target_Z_exp = target_Z.expand(B, self.num_quantiles, self.num_quantiles)
                q_taken_exp = q_taken.expand(B, self.num_quantiles, self.num_quantiles)
                
                td_error = target_Z_exp - q_taken_exp
                huber_loss = F.huber_loss(q_taken_exp, target_Z_exp, reduction='none', delta=1.0)
                
                tau = (torch.arange(self.num_quantiles).float().to(self.device) + 0.5) / self.num_quantiles
                tau = tau.view(1, self.num_quantiles, 1)
                
                quantile_weight = torch.abs(tau - (td_error < 0).float())
                quantile_loss = quantile_weight * huber_loss
                
                loss = quantile_loss.sum(dim=1).mean()

                self.q_optim.zero_grad()
                loss.backward()
                self.q_optim.step()
                
                # Soft update target network
                for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                    target_param.data.copy_(tau_update * param.data + (1 - tau_update) * target_param.data)
                
                epoch_loss += loss.item() * len(s_batch)
                
            epoch_loss /= N
            history["q_loss"].append(epoch_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1:3d}  Q  loss: {epoch_loss:.4f}")

        self.q_net.eval()
        return history

    def _safe_mask(self, bc_probs: torch.Tensor) -> torch.Tensor:
        mask = bc_probs >= self.threshold
        n_surviving = mask.sum(dim=1)
        need_fix = n_surviving < self.min_actions
        if need_fix.any():
            _, top_idx = bc_probs[need_fix].topk(self.min_actions, dim=1)
            fix_mask = torch.zeros_like(bc_probs[need_fix], dtype=torch.bool)
            fix_mask.scatter_(1, top_idx, True)
            mask[need_fix] = fix_mask
        return mask

    def _hybrid_scores(self, S: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if self._item_emb is None or self.hybrid_weight == 0:
            return q
        user_emb = S[:, -self.gnn_embed_dim:]
        dot_scores = user_emb @ self._item_emb.T
        def _znorm(x):
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.std(dim=1, keepdim=True).clamp(min=1e-8)
            return (x - mu) / sigma
        return _znorm(q) + self.hybrid_weight * _znorm(dot_scores)

    def _compute_cvar(self, q_quantiles: torch.Tensor) -> torch.Tensor:
        sorted_q, _ = torch.sort(q_quantiles, dim=2)
        k = max(1, int(self.num_quantiles * self.cvar_alpha))
        return sorted_q[:, :, :k].mean(dim=2)

    @torch.no_grad()
    def action_probabilities(self, states: np.ndarray, temperature: Optional[float] = None, batch_size: int = 65536) -> np.ndarray:
        temp = temperature or self.temperature
        N = len(states)
        all_probs = np.empty((N, self.n_actions), dtype=np.float32)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            S = torch.FloatTensor(states[start:end]).to(self.device)
            bc_probs = F.softmax(self.bc_model(S), dim=1)
            mask = self._safe_mask(bc_probs)
            q_all = self.q_net(S).view(-1, self.n_actions, self.num_quantiles)
            q_cvar = self._compute_cvar(q_all)
            q = self._hybrid_scores(S, q_cvar)
            q[~mask] = float("-inf")
            probs = F.softmax(q / temp, dim=1)
            all_probs[start:end] = probs.cpu().numpy()
        return all_probs
        
    def save_checkpoint(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'bc_state_dict': self.bc_model.state_dict(),
            'bc_optim_state_dict': self.bc_optim.state_dict(),
            'q_state_dict': self.q_net.state_dict(),
            'q_optim_state_dict': self.q_optim.state_dict(),
            'q_target_state_dict': self.q_target.state_dict(),
        }, filepath)
        
    def load_checkpoint(self, filepath: str) -> bool:
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.bc_model.load_state_dict(checkpoint['bc_state_dict'])
            self.bc_optim.load_state_dict(checkpoint['bc_optim_state_dict'])
            self.q_net.load_state_dict(checkpoint['q_state_dict'])
            self.q_optim.load_state_dict(checkpoint['q_optim_state_dict'])
            self.q_target.load_state_dict(checkpoint['q_target_state_dict'])
            print(f"  [Checkpoint Loaded] Resuming Dynamic BCQ from {filepath}")
            return True
        return False
