
import numpy as np
import torch
from src.baselines.policies import DecisionTransformerPolicy

states = np.random.randn(100, 103).astype(np.float32)
actions = np.random.randint(0, 34, size=100)
rewards = np.random.randn(100).astype(np.float32)

dt = DecisionTransformerPolicy(state_dim=103, n_actions=34, device="cpu")
print("Training DT...")
dt.train(states, actions, rewards, n_epochs=2, batch_size=16)

print("Inference DT...")
probs = dt.action_probabilities(states)
print("Probs shape:", probs.shape)
print("Sum of probs:", probs.sum(axis=1)[:5])
print("Success!")

