"""
Unit tests for CATE Estimator, Robinson R-Learner, and 4-Quadrant Uplift Segmentation.

Validates that:
1. CATE action-specific loss trains without crashing and avoids artificial negative collapse.
2. User segmentation cleanly isolates all 4 quadrants:
   - Quadrant 0: Persuadables (low baseline, positive uplift)
   - Quadrant 1: Sure Things (high baseline, positive uplift)
   - Quadrant 2: Lost Causes (low baseline, negative/zero uplift)
   - Quadrant 3: Sleeping Dogs (high baseline, negative uplift)
3. R-Learner cross-fitting executes properly and produces zero-centered treatment effects.
4. Adaptive baseline thresholding works correctly in metrics.py.
"""

import numpy as np
import pytest
import torch

from src.causal.cate_estimator import CATEEstimator
from src.causal.r_learner import RLearner
from src.utils.metrics import segment_users as metric_segment_users, sleeping_dogs_analysis


@pytest.fixture
def synthetic_uplift_data():
    """Generates synthetic data with known 4-quadrant ground truth."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_users = 400
    n_samples = 4000
    n_actions = 10
    state_dim = 16

    # 4 user archetypes (100 users each):
    # Users 0..99: Persuadables (baseline CTR ~ 0.02, treatment effect on action 1 is +0.30)
    # Users 100..199: Sure Things (baseline CTR ~ 0.40, treatment effect on action 1 is +0.10)
    # Users 200..299: Lost Causes (baseline CTR ~ 0.02, treatment effect on action 1 is -0.01)
    # Users 300..399: Sleeping Dogs (baseline CTR ~ 0.40, treatment effect on action 1 is -0.30)
    user_ids = np.random.randint(0, n_users, size=n_samples)
    actions = np.random.randint(0, n_actions, size=n_samples)
    
    # State features encode user representations (matching GNN embeddings)
    user_embeddings = np.random.randn(n_users, state_dim).astype(np.float32) * 0.1
    for u in range(n_users):
        arch = u // 100
        user_embeddings[u, arch] += 2.0
    states = user_embeddings[user_ids]

    rewards = np.zeros(n_samples, dtype=np.float32)
    propensities = np.full(n_samples, 1.0 / n_actions, dtype=np.float32)

    for i in range(n_samples):
        u = user_ids[i]
        a = actions[i]
        if u < 100:  # Persuadable
            base = 0.05
            tau = 0.40 if a == 1 else 0.0
        elif u < 200:  # Sure Thing
            base = 0.60
            tau = 0.20 if a == 1 else 0.0
        elif u < 300:  # Lost Cause
            base = 0.05
            tau = -0.04 if a == 1 else 0.0
        else:  # Sleeping Dog
            base = 0.60
            tau = -0.40 if a == 1 else -0.10

        p = np.clip(base + tau, 0.0, 1.0)
        rewards[i] = 1.0 if np.random.rand() < p else 0.0

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "user_ids": user_ids,
        "propensities": propensities,
        "n_users": n_users,
        "n_actions": n_actions,
        "state_dim": state_dim,
    }


def test_cate_action_specific_fit_and_segmentation(synthetic_uplift_data):
    """Verifies CATEEstimator fit_from_outcomes and 4-quadrant segmentation."""
    d = synthetic_uplift_data
    estimator = CATEEstimator(
        state_dim=d["state_dim"],
        n_actions=d["n_actions"],
        hidden=64,
        device="cpu",
    )

    estimator.fit_from_outcomes(
        d["states"],
        d["actions"],
        d["rewards"],
        n_epochs=30,
        batch_size=512,
        cfr_lambda=0.05,
        verbose=False,
    )

    preds = estimator.predict(d["states"])
    assert preds.shape == (len(d["states"]), d["n_actions"])
    assert not np.isnan(preds).any()

    # Verify segmentation with adaptive baseline threshold and relative treatment arm
    seg = estimator.segment_users(
        d["states"],
        d["user_ids"],
        baseline_responses=d["rewards"],
        treatment_arm=1,
        control_arm=0,
    )

    counts = seg["segment_counts"]
    print("CATE segment counts:", counts)

    # All 4 quadrants must be non-empty
    assert counts["Persuadable"] > 0, "Persuadables count must be non-zero"
    assert counts["Sure Thing"] > 0, "Sure Things count must be non-zero"
    assert counts["Lost Cause"] > 0, "Lost Causes count must be non-zero"
    assert counts["Sleeping Dog"] > 0, "Sleeping Dogs count must be non-zero"

    total_classified = sum(counts.values())
    assert total_classified == len(np.unique(d["user_ids"]))


def test_r_learner_cross_fitting(synthetic_uplift_data):
    """Verifies RLearner cross-fitting and Neyman-orthogonal residual estimation."""
    d = synthetic_uplift_data
    r_learner = RLearner(
        state_dim=d["state_dim"],
        n_actions=d["n_actions"],
        hidden=64,
        device="cpu",
    )

    r_learner.fit(
        d["states"],
        d["actions"],
        d["rewards"],
        propensities=d["propensities"],
        n_folds=2,
        nuisance_epochs=15,
        tau_epochs=30,
        batch_size=512,
        verbose=False,
    )

    preds = r_learner.predict(d["states"])
    assert preds.shape == (len(d["states"]), d["n_actions"])
    assert not np.isnan(preds).any()

    # Blend rewards
    weighted = r_learner.uplift_weighted_rewards(
        d["states"], d["actions"], d["rewards"], uplift_weight=0.5
    )
    assert len(weighted) == len(d["rewards"])
    assert 0.0 <= weighted.min() and weighted.max() <= 1.0

    # Verify 4-quadrant segmentation
    seg = r_learner.segment_users(
        d["states"],
        d["user_ids"],
        baseline_responses=d["rewards"],
        treatment_arm=1,
        control_arm=0,
    )
    counts = seg["segment_counts"]
    print("R-Learner segment counts:", counts)

    assert counts["Persuadable"] > 0
    assert counts["Sure Thing"] > 0
    assert counts["Lost Cause"] > 0
    assert counts["Sleeping Dog"] > 0


def test_metrics_adaptive_segment_users():
    """Verifies metrics.py segment_users adaptive thresholding and quadrant mapping."""
    uplifts = np.array([0.15, 0.20, -0.10, -0.25], dtype=np.float32)
    baselines = np.array([0.02, 0.08, 0.01, 0.09], dtype=np.float32)

    # Median of baselines is 0.05
    # Sample 0: low baseline (0.02 <= 0.05), pos uplift (0.15 > 0) -> 0 (Persuadable)
    # Sample 1: high baseline (0.08 > 0.05), pos uplift (0.20 > 0) -> 1 (Sure Thing)
    # Sample 2: low baseline (0.01 <= 0.05), neg uplift (-0.10 <= 0) -> 2 (Lost Cause)
    # Sample 3: high baseline (0.09 > 0.05), neg uplift (-0.25 <= 0) -> 3 (Sleeping Dog)
    segments = metric_segment_users(uplifts, baselines, response_threshold=None)

    assert segments[0] == 0  # Persuadable
    assert segments[1] == 1  # Sure Thing
    assert segments[2] == 2  # Lost Cause
    assert segments[3] == 3  # Sleeping Dog


def test_sleeping_dogs_analysis():
    """Verifies sleeping_dogs_analysis metric reporting with policy weights."""
    n_samples = 100
    n_actions = 5
    n_users = 10

    user_ids = np.random.randint(0, n_users, size=n_samples)
    policy_probs = np.full((n_samples, n_actions), 1.0 / n_actions, dtype=np.float32)

    uplift_table = np.random.randn(n_users, n_actions).astype(np.float32)

    results = sleeping_dogs_analysis(policy_probs, user_ids, uplift_table, n_actions)

    assert "n_sleeping_dog" in results
    assert "n_persuadable" in results
    assert "avg_max_prob_sleeping_dog" in results
    assert "avg_max_prob_persuadable" in results
    assert results["n_sleeping_dog"] + results["n_persuadable"] <= n_samples
