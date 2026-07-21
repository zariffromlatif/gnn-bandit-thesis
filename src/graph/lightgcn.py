"""
LightGCN encoder for graph-propagated treatment effect estimation.

Implements the simplified graph convolution from He et al. (SIGIR 2020):
no feature transformation, no activation — just neighbourhood aggregation
across the user-item (or user-user + user-item + item-item) graph.

The key insight for our paper:
    Each convolution layer propagates treatment-effect signals from neighbours.
    After L layers, a user's embedding encodes not just their own history, but
    the aggregated *causal response patterns* of all users within L hops.  For
    cold-start users (42.6 % in OBD) this is the primary information source.

References
----------
- He et al., "LightGCN: Simplifying and Powering Graph Convolution Network
  for Recommendation", SIGIR 2020.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix


# ============================================================================
# Sparse helpers
# ============================================================================

def _scipy_to_torch_sparse(sp: csr_matrix, device: torch.device) -> torch.Tensor:
    """Convert a scipy CSR matrix to a PyTorch sparse COO tensor."""
    coo = sp.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values  = torch.FloatTensor(coo.data)
    shape   = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to(device)


def _symmetric_norm(adj: csr_matrix) -> csr_matrix:
    """
    Compute symmetric normalisation  D^{-1/2} A D^{-1/2}.

    This is the standard normalisation for GCN / LightGCN so that
    high-degree nodes do not dominate the aggregation.
    Isolated nodes (degree = 0) get zero entries, which is correct:
    they receive no messages from neighbours.
    """
    deg = np.asarray(adj.sum(axis=1)).flatten()
    # Safe inverse square root: avoid divide-by-zero for isolated nodes
    deg_inv_sqrt = np.zeros_like(deg, dtype=np.float64)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    from scipy.sparse import diags
    D_inv_sqrt = diags(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


# ============================================================================
# LightGCN model
# ============================================================================

class LightGCN(nn.Module):
    """
    LightGCN graph encoder.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes in the graph (n_users + n_items for OBD,
        or n_clusters for Criteo).
    embed_dim : int
        Embedding dimensionality K.  Default 64.
    n_layers : int
        Number of graph convolution layers L.  Default 3.
    adj : scipy.sparse.csr_matrix
        Raw adjacency matrix (will be symmetrically normalised internally).
    n_users : int
        Number of user nodes (first n_users rows/cols of the adjacency).
        Used to split the final embedding table back into user / item parts.
    dropout : float
        Edge dropout during training (0 = no dropout).  Default 0.0.
    """

    def __init__(
        self,
        n_nodes: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        adj: Optional[csr_matrix] = None,
        n_users: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_nodes   = n_nodes
        self.embed_dim = embed_dim
        self.n_layers  = n_layers
        self.n_users   = n_users
        self.dropout   = dropout

        # Learnable initial embeddings  E^{(0)}
        self.embedding = nn.Embedding(n_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        # Pre-compute normalised adjacency (not a parameter — fixed)
        if adj is not None:
            self.register_adjacency(adj)
        else:
            self._adj_norm = None

    # ------------------------------------------------------------------
    # Adjacency management
    # ------------------------------------------------------------------

    def register_adjacency(self, adj: csr_matrix):
        """
        Normalise *adj* and cache the sparse torch tensor.

        Call this once at init or whenever the graph changes (e.g. when
        switching between OBD campaigns).
        """
        normed = _symmetric_norm(adj)
        device = next(self.parameters()).device
        self._adj_norm = _scipy_to_torch_sparse(normed, device)

    def to(self, device, *args, **kwargs):
        """Override to move the cached adjacency tensor along with parameters."""
        out = super().to(device, *args, **kwargs)
        if self._adj_norm is not None:
            out._adj_norm = out._adj_norm.to(device)
        return out

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self) -> torch.Tensor:
        """
        Run L layers of graph convolution and return the mean-pooled
        embeddings for **all** nodes.

        Returns
        -------
        all_embeddings : Tensor of shape (n_nodes, embed_dim)
        """
        assert self._adj_norm is not None, (
            "Adjacency not set.  Call register_adjacency(adj) first."
        )

        E = self.embedding.weight                # (n_nodes, K)
        layer_outputs = [E]                       # collect E^{(0)} .. E^{(L)}

        A = self._adj_norm
        for _ in range(self.n_layers):
            # LightGCN convolution:  E^{(l+1)} = A_norm @ E^{(l)}
            if self.training and self.dropout > 0:
                E = self._sparse_dropout(E)
            E = torch.sparse.mm(A, E)
            layer_outputs.append(E)

        # Final embedding = mean over all layers (including layer 0)
        all_embeddings = torch.stack(layer_outputs, dim=0).mean(dim=0)
        return all_embeddings

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_user_embeddings(
        self, user_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return embeddings for specific users (or all users)."""
        all_emb = self.forward()
        user_emb = all_emb[: self.n_users]
        if user_ids is not None:
            return user_emb[user_ids]
        return user_emb

    def get_item_embeddings(
        self, item_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return embeddings for specific items (or all items)."""
        all_emb = self.forward()
        if self.n_nodes == self.n_users:
            item_emb = all_emb
        else:
            item_emb = all_emb[self.n_users:]
            
        if item_ids is not None:
            return item_emb[item_ids]
        return item_emb

    def encode_users(self, user_ids: np.ndarray) -> np.ndarray:
        """
        Convenience: encode a batch of user IDs to numpy embeddings.

        Useful for feeding embeddings into the BCQ agent without requiring
        the caller to manage torch tensors.
        """
        self.eval()
        with torch.no_grad():
            ids = torch.LongTensor(user_ids).to(self.embedding.weight.device)
            emb = self.get_user_embeddings(ids)
        return emb.cpu().numpy()

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def bpr_loss(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        reg_weight: float = 1e-4,
    ) -> torch.Tensor:
        """
        Bayesian Personalised Ranking loss for training.

        Parameters
        ----------
        user_ids     : (B,) user indices
        pos_item_ids : (B,) positive (clicked) item indices
        neg_item_ids : (B,) negative (not clicked) item indices
        reg_weight   : L2 regularisation weight on the initial embeddings

        Returns
        -------
        loss : scalar tensor
        """
        all_emb = self.forward()

        u_emb   = all_emb[user_ids]
        pos_emb = all_emb[self.n_users + pos_item_ids]
        neg_emb = all_emb[self.n_users + neg_item_ids]

        pos_score = (u_emb * pos_emb).sum(dim=1)
        neg_score = (u_emb * neg_emb).sum(dim=1)

        bpr = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()

        # L2 regularisation on initial (layer-0) embeddings only
        reg = reg_weight * (
            self.embedding.weight[user_ids].norm(2).pow(2)
            + self.embedding.weight[self.n_users + pos_item_ids].norm(2).pow(2)
            + self.embedding.weight[self.n_users + neg_item_ids].norm(2).pow(2)
        ) / len(user_ids)

        return bpr + reg

    def _sparse_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout by zeroing random elements."""
        mask = torch.bernoulli(
            torch.full_like(x, 1 - self.dropout)
        )
        return x * mask / (1 - self.dropout)

    # ------------------------------------------------------------------
    # Negative sampling utility
    # ------------------------------------------------------------------

    @staticmethod
    def sample_negatives(
        user_ids: np.ndarray,
        pos_items: np.ndarray,
        n_items: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Sample one negative item per user that is not the positive item.

        Parameters
        ----------
        user_ids  : (B,) user indices (unused but kept for signature clarity)
        pos_items : (B,) positive item indices
        n_items   : total number of items
        rng       : optional random state

        Returns
        -------
        neg_items : (B,) sampled negative item indices
        """
        if rng is None:
            rng = np.random.RandomState()
        neg = rng.randint(0, n_items, size=len(pos_items))
        # Re-sample collisions
        collisions = neg == pos_items
        while collisions.any():
            neg[collisions] = rng.randint(0, n_items, size=collisions.sum())
            collisions = neg == pos_items
        return neg
