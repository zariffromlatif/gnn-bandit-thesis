import numpy as np
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """
    Dataset that extracts multi-step transitions (s_t, a_t, r_t, s_{t+1}) 
    from chronological single-step observation logs.
    """
    def __init__(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, user_ids: np.ndarray):
        """
        Args:
            states: (N, D) float array of states
            actions: (N,) int array of actions
            rewards: (N,) float array of rewards
            user_ids: (N,) int array of user segment IDs
        """
        self.s_t = []
        self.a_t = []
        self.r_t = []
        self.s_next = []
        
        # We assume the dataset is already ordered chronologically (as guaranteed by preprocessing).
        # We find the sequence of row indices for each user.
        print("Extracting trajectories from chronological logs...")
        
        # Sort by user_id to easily group. Since python's sort/argsort is stable, 
        # chronological order within each user is preserved.
        # Wait, argsort in numpy might not be stable by default. We must use kind='stable'
        sorted_idx = np.argsort(user_ids, kind='stable')
        
        unique_users, split_indices = np.unique(user_ids[sorted_idx], return_index=True)
        # Split into lists of indices per user
        user_groups = np.split(sorted_idx, split_indices[1:])
        
        extracted_transitions = 0
        for group in user_groups:
            # group contains the chronological row indices for a single user
            n_impressions = len(group)
            if n_impressions < 2:
                continue # Cannot form a transition s_t -> s_{t+1}
            
            # For a sequence of N impressions, we have N-1 transitions
            for t in range(n_impressions - 1):
                idx_curr = group[t]
                idx_next = group[t+1]
                
                self.s_t.append(states[idx_curr])
                self.a_t.append(actions[idx_curr])
                self.r_t.append(rewards[idx_curr])
                self.s_next.append(states[idx_next])
                extracted_transitions += 1
                
        # Convert to numpy arrays for fast PyTorch streaming
        if extracted_transitions > 0:
            self.s_t = np.array(self.s_t, dtype=np.float32)
            self.a_t = np.array(self.a_t, dtype=np.int64)
            self.r_t = np.array(self.r_t, dtype=np.float32)
            self.s_next = np.array(self.s_next, dtype=np.float32)
        else:
            # Handle edge case where no user has >1 impression (e.g., tiny test sets)
            self.s_t = np.zeros((0, states.shape[1]), dtype=np.float32)
            self.a_t = np.zeros((0,), dtype=np.int64)
            self.r_t = np.zeros((0,), dtype=np.float32)
            self.s_next = np.zeros((0, states.shape[1]), dtype=np.float32)
            
        print(f"Extracted {extracted_transitions:,} temporal transitions from {len(states):,} single-step logs.")

    def __len__(self):
        return len(self.s_t)

    def __getitem__(self, idx):
        return self.s_t[idx], self.a_t[idx], self.r_t[idx], self.s_next[idx]

    def get_tensors(self, device):
        """Zero-copy streaming directly to GPU, avoiding RAM duplication."""
        return (
            torch.as_tensor(self.s_t, device=device),
            torch.as_tensor(self.a_t, device=device),
            torch.as_tensor(self.r_t, device=device),
            torch.as_tensor(self.s_next, device=device)
        )
