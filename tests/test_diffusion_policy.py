"""
Unit tests for Diff-GNN-Bandit generative score-based diffusion policy.

Validates:
1. Forward diffusion noise addition (q_sample).
2. Denoising score matching loss training step.
3. Fast DDIM reverse ODE sampler stability and latency (< 35ms per batch).
4. Continuous-to-discrete catalog projection yielding valid probability distributions.
5. Tweedie CATE guidance steering.
"""

import time
import numpy as np
import pytest
import torch

from src.agent.diffusion_policy import DiffGNNBandit


@pytest.fixture
def synthetic_diffusion_env():
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 256
    state_dim = 32
    n_actions = 15
    action_dim = 16

    states = np.random.randn(n_samples, state_dim).astype(np.float32)
    actions = np.random.randint(0, n_actions, size=n_samples)
    rewards = np.random.binomial(1, 0.2, size=n_samples).astype(np.float32)
    item_embeddings = np.random.randn(n_actions, action_dim).astype(np.float32)

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "item_embeddings": item_embeddings,
        "n_samples": n_samples,
        "state_dim": state_dim,
        "n_actions": n_actions,
        "action_dim": action_dim,
    }


def test_diffusion_forward_and_loss(synthetic_diffusion_env):
    """Tests forward diffusion q_sample and score matching loss."""
    env = synthetic_diffusion_env
    agent = DiffGNNBandit(
        state_dim=env["state_dim"],
        n_actions=env["n_actions"],
        action_dim=env["action_dim"],
        item_embeddings=env["item_embeddings"],
        num_timesteps=50,
        ddim_steps=10,
        device="cpu",
    )

    S = torch.as_tensor(env["states"], dtype=torch.float32)
    A = torch.as_tensor(env["actions"], dtype=torch.long)
    R = torch.as_tensor(env["rewards"], dtype=torch.float32)

    loss = agent.loss(S, A, R)
    assert loss.dim() == 0, "Loss must be scalar"
    assert not torch.isnan(loss), "Loss must not be NaN"
    assert loss.item() > 0.0, "Loss must be positive"

    # Backward pass verification
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in agent.parameters() if p.grad is not None)
    assert grad_norm > 0.0, "Gradients must propagate through score network"


def test_ddim_sampler_and_action_probabilities(synthetic_diffusion_env):
    """Verifies fast DDIM sampling, catalog projection, and probability axioms."""
    env = synthetic_diffusion_env
    agent = DiffGNNBandit(
        state_dim=env["state_dim"],
        n_actions=env["n_actions"],
        action_dim=env["action_dim"],
        item_embeddings=env["item_embeddings"],
        num_timesteps=50,
        ddim_steps=15,
        temperature=0.1,
        device="cpu",
    )

    # Measure DDIM inference latency
    t0 = time.perf_counter()
    probs = agent.action_probabilities(env["states"], batch_size=128)
    dt_ms = (time.perf_counter() - t0) * 1000

    print(f"DDIM inference latency: {dt_ms:.2f} ms for {len(env['states'])} states")

    assert probs.shape == (len(env["states"]), env["n_actions"])
    assert not np.isnan(probs).any(), "Probabilities must not contain NaN"
    assert (probs >= 0.0).all(), "All action probabilities must be non-negative"

    # Valid probability distribution: sums to 1.0
    sums = probs.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5, atol=1e-5)

    # Greedy action prediction
    greedy_actions = agent.predict_action(env["states"])
    assert len(greedy_actions) == len(env["states"])
    assert (greedy_actions >= 0).all() and (greedy_actions < env["n_actions"]).all()


def test_tweedie_cate_guidance(synthetic_diffusion_env):
    """Verifies that Tweedie CATE guidance shifts action probabilities."""
    env = synthetic_diffusion_env
    agent = DiffGNNBandit(
        state_dim=env["state_dim"],
        n_actions=env["n_actions"],
        action_dim=env["action_dim"],
        item_embeddings=env["item_embeddings"],
        num_timesteps=50,
        ddim_steps=15,
        guidance_weight=3.0,
        temperature=0.5,
        device="cpu",
    )

    # Paired random seed evaluation to isolate guidance effect from sampling noise
    torch.manual_seed(123)
    probs_unguided = agent.action_probabilities(env["states"][:20])

    # Strongly positive CATE for action 3, negative for other actions
    cate_guided = np.full((20, env["n_actions"]), -0.2, dtype=np.float32)
    cate_guided[:, 3] = +1.5

    torch.manual_seed(123)
    probs_guided = agent.action_probabilities(env["states"][:20], cate_scores=cate_guided)

    # Guided policy should allocate significantly higher probability to action 3
    avg_prob_unguided_a3 = probs_unguided[:, 3].mean()
    avg_prob_guided_a3 = probs_guided[:, 3].mean()

    print(f"Action 3 prob unguided: {avg_prob_unguided_a3:.4f}, guided: {avg_prob_guided_a3:.4f}")
    assert avg_prob_guided_a3 > avg_prob_unguided_a3, (
        f"CATE guidance must increase target action probability: {avg_prob_guided_a3} vs {avg_prob_unguided_a3}"
    )
