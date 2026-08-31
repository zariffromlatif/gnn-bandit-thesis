"""
Temporal Graph Network (TGN) encoder for sequential recommendation graphs.

Implements a continuous-time temporal graph encoder with:
  1. Learnable Fourier Time Encodings
  2. Node Memory Module (GRU-based state evolution)
  3. Temporal Attention / Aggregation over chronological interaction events
  4. Identical public interface to LightGCN for seamless integration.

References:
  - Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs",
    NeurIPS 2020 Workshop on Mining and Learning with Graphs.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix


class TimeEncoder(nn.Module):
    """Learnable Fourier time feature mapping."""
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Parameter(torch.randn(time_dim) * 0.1)
        self.b = nn.Parameter(torch.randn(time_dim) * 0.1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1) in normalized timestamp units
        t = t.view(-1, 1)
        angles = t * self.w.view(1, -1) + self.b.view(1, -1)
        return torch.cos(angles)


class TGNEncoder(nn.Module):
    """
    Temporal Graph Network encoder compatible with the GNN-Bandit pipeline.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes (n_users + n_items).
    embed_dim : int
        Dimensionality of node embeddings (default 64).
    n_users : int
        Number of user nodes.
    memory_dim : int
        Dimensionality of node dynamic memory state (default 64).
    time_dim : int
        Dimensionality of temporal Fourier encoding (default 16).
    """

    def __init__(
        self,
        n_nodes: int,
        embed_dim: int = 64,
        n_users: int = 0,
        memory_dim: int = 64,
        time_dim: int = 16,
        adj: Optional[csr_matrix] = None,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.embed_dim = embed_dim
        self.n_users = n_users
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        # Static base embeddings
        self.node_embedding = nn.Embedding(n_nodes, embed_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)

        # Dynamic Node Memory
        self.register_buffer("memory", torch.zeros(n_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(n_nodes))

        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)

        # Memory Updater (GRU cell)
        self.msg_dim = embed_dim * 2 + time_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(self.msg_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.gru = nn.GRUCell(memory_dim, memory_dim)

        # Output projection to embed_dim
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim + memory_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.device = torch.device("cpu")

    def reset_memory(self):
        """Reset temporal node states to initial zeros."""
        self.memory.zero_()
        self.last_update.zero_()

    def to(self, device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)

    def update_events(
        self,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor,
        timestamps: torch.Tensor,
    ):
        """
        Process a batch of chronological interaction events to update node memories.
        """
        src = src_nodes.to(self.device)
        dst = dst_nodes.to(self.device)
        ts = timestamps.float().to(self.device)

        dt_src = ts - self.last_update[src]
        dt_dst = ts - self.last_update[dst]

        t_enc_src = self.time_encoder(dt_src)
        t_enc_dst = self.time_encoder(dt_dst)

        e_src = self.node_embedding(src)
        e_dst = self.node_embedding(dst)

        # Raw interaction message: [e_src, e_dst, time_enc]
        msg_src = torch.cat([e_src, e_dst, t_enc_src], dim=-1)
        msg_dst = torch.cat([e_dst, e_src, t_enc_dst], dim=-1)

        h_src = self.msg_mlp(msg_src)
        h_dst = self.msg_mlp(msg_dst)

        # Update GRU states
        new_mem_src = self.gru(h_src, self.memory[src])
        new_mem_dst = self.gru(h_dst, self.memory[dst])

        self.memory[src] = new_mem_src.detach()
        self.memory[dst] = new_mem_dst.detach()
        self.last_update[src] = ts.detach()
        self.last_update[dst] = ts.detach()

    def forward(self) -> torch.Tensor:
        """
        Produce temporal graph embeddings for all nodes.

        Returns:
            all_embeddings: (n_nodes, embed_dim)
        """
        static_emb = self.node_embedding.weight
        combined = torch.cat([static_emb, self.memory], dim=-1)
        return self.out_proj(combined)

    def get_all_embeddings(self) -> np.ndarray:
        """Return all node embeddings as numpy array."""
        self.eval()
        with torch.no_grad():
            emb = self.forward().cpu().numpy()
        return emb

    def get_user_embeddings(self, user_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        all_emb = self.forward()
        user_emb = all_emb[: self.n_users] if self.n_users > 0 else all_emb
        if user_ids is not None:
            return user_emb[user_ids]
        return user_emb

    def get_item_embeddings(self, item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        all_emb = self.forward()
        if self.n_users > 0 and self.n_nodes > self.n_users:
            item_emb = all_emb[self.n_users:]
        else:
            item_emb = all_emb
        if item_ids is not None:
            return item_emb[item_ids]
        return item_emb

    def encode_users(self, user_ids: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            u_t = torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
            emb = self.get_user_embeddings(u_t).cpu().numpy()
        return emb

    def encode_items(self, item_ids: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            i_t = torch.as_tensor(item_ids, dtype=torch.long, device=self.device)
            emb = self.get_item_embeddings(i_t).cpu().numpy()
        return emb

    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        timestamps: np.ndarray,
        epochs: int = 20,
        batch_size: int = 4096,
        lr: float = 0.001,
        reg: float = 1e-4,
    ):
        """
        Train temporal memory and static embeddings on chronological interaction events.
        """
        # Sort chronologically
        sort_idx = np.argsort(timestamps)
        u_sorted = user_ids[sort_idx]
        has_separate_items = (self.n_users > 0 and self.n_nodes > self.n_users)
        i_offset = self.n_users if has_separate_items else 0
        i_sorted = item_ids[sort_idx] + i_offset
        t_sorted = timestamps[sort_idx]

        # Normalize timestamps to [0, 100] scale
        t_min = t_sorted.min()
        t_range = max(t_sorted.max() - t_min, 1.0)
        t_norm = ((t_sorted - t_min) / t_range * 100.0).astype(np.float32)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
        N = len(u_sorted)
        print(f"Training TGN on {N:,} temporal events over {epochs} epochs ...")

        neg_low = self.n_users if has_separate_items else 0

        for epoch in range(epochs):
            self.train()
            self.reset_memory()
            total_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                b_src = torch.as_tensor(u_sorted[start:end], dtype=torch.long, device=self.device)
                b_dst = torch.as_tensor(i_sorted[start:end], dtype=torch.long, device=self.device)
                b_ts = torch.as_tensor(t_norm[start:end], dtype=torch.float32, device=self.device)

                # Random negative items for BPR loss
                neg_dst = torch.randint(
                    neg_low,
                    self.n_nodes,
                    (end - start,),
                    device=self.device,
                )

                # Embeddings before event
                emb = self.forward()
                pos_score = (emb[b_src] * emb[b_dst]).sum(dim=-1)
                neg_score = (emb[b_src] * emb[neg_dst]).sum(dim=-1)

                loss = -F.logsigmoid(pos_score - neg_score).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update memory sequentially
                self.update_events(b_src, b_dst, b_ts)

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1:02d}/{epochs:02d} - Loss: {total_loss / max(n_batches, 1):.4f}")
