import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class StateDynamicsModel(nn.Module):
    """
    World Model for Backward RL.
    Predicts the residual state change: \Delta s = f(s_t, a_t)
    So s_{t+1} = s_t + \Delta s
    """
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        # Action embedding
        self.action_embed = nn.Embedding(n_actions, 32)
        
        # MLP for predicting residual \Delta s
        self.net = nn.Sequential(
            nn.Linear(state_dim + 32, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, state_dim)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, D) current state
            action: (B,) taken action
        Returns:
            next_state_pred: (B, D) predicted next state
        """
        a_emb = self.action_embed(action) # (B, 32)
        x = torch.cat([state, a_emb], dim=-1) # (B, D + 32)
        delta_s = self.net(x)
        return state + delta_s

class DynamicsTrainer:
    def __init__(self, state_dim: int, n_actions: int, lr: float = 1e-3, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = StateDynamicsModel(state_dim, n_actions).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        
    def train_epoch(self, s_t, a_t, s_next, batch_size=16384):
        """Trains for one epoch over the temporal transitions."""
        self.model.train()
        n_samples = len(s_t)
        indices = torch.randperm(n_samples, device=self.device)
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            idx = indices[i : i + batch_size]
            b_s = s_t[idx]
            b_a = a_t[idx]
            b_s_next = s_next[idx]
            
            s_next_pred = self.model(b_s, b_a)
            loss = F.mse_loss(s_next_pred, b_s_next)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / n_batches if n_batches > 0 else 0.0

    def predict(self, state, action):
        """Predicts the next state given a state and action."""
        self.model.eval()
        with torch.no_grad():
            return self.model(state, action)
            
    def save_checkpoint(self, filepath: str):
        """Saves model weights to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        
    def load_checkpoint(self, filepath: str) -> bool:
        """Loads model weights if they exist. Returns True if loaded."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"  [Checkpoint Loaded] Resuming Dynamics Model from {filepath}")
            return True
        return False
