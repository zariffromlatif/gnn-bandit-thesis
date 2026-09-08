"""
Unit tests for Multiple Gradient Descent Algorithm (MGDA) Pareto Solver.

Validates:
1. Simplex projection axioms: sum(alpha) == 1.0, alpha >= 0.0.
2. Conflict resolution when task gradients oppose each other (<g1, g2> < 0).
3. Correct parameter .grad population during apply_pareto_backward.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.agent.pareto_solver import MGDAParetoSolver


def test_mgda_simplex_weights():
    """Verifies that solved weights lie strictly on the probability simplex."""
    solver = MGDAParetoSolver(max_iters=30)
    
    # 3 tasks with random gradients
    torch.manual_seed(42)
    grad_matrix = torch.randn(3, 64)

    alpha = solver.solve_weights(grad_matrix)

    assert len(alpha) == 3
    assert (alpha >= -1e-6).all(), "Weights must be non-negative"
    assert torch.isclose(alpha.sum(), torch.tensor(1.0), atol=1e-4), "Weights must sum to 1.0"


def test_mgda_conflict_resolution():
    """
    Verifies that MGDA finds a Pareto-descent direction when two tasks conflict.
    Task 1 gradient: [+1.0, 0.0]
    Task 2 gradient: [-0.5, +0.866] (angle ~ 120 degrees, negative inner product)
    """
    solver = MGDAParetoSolver(max_iters=25)
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([-0.5, 0.866])
    
    # Confirm negative inner product
    inner = torch.dot(g1, g2)
    assert inner < 0.0, "Gradients must have conflicting directions"

    grad_matrix = torch.stack([g1, g2], dim=0)
    alpha = solver.solve_weights(grad_matrix)

    # Combined Pareto gradient
    d = torch.matmul(alpha, grad_matrix)

    # Both task inner products must be non-negative with d (common descent)
    # <g1, d> >= 0 and <g2, d> >= 0
    dp1 = torch.dot(g1, d).item()
    dp2 = torch.dot(g2, d).item()

    print(f"Pareto weights: {alpha.tolist()}, <g1, d>: {dp1:.4f}, <g2, d>: {dp2:.4f}")
    assert dp1 >= -1e-5, "Direction must not conflict with Task 1"
    assert dp2 >= -1e-5, "Direction must not conflict with Task 2"


def test_apply_pareto_backward():
    """Verifies end-to-end backward pass on a shared multi-task network."""
    torch.manual_seed(42)
    shared_trunk = nn.Linear(16, 32)
    head_click = nn.Linear(32, 1)
    head_retention = nn.Linear(32, 1)

    x = torch.randn(8, 16)
    y_click = torch.ones(8, 1)
    y_ret = torch.zeros(8, 1)

    # Forward
    h = torch.relu(shared_trunk(x))
    pred_click = head_click(h)
    pred_ret = head_retention(h)

    # Competing losses
    loss_click = torch.nn.functional.mse_loss(pred_click, y_click)
    loss_ret = torch.nn.functional.mse_loss(pred_ret, y_ret)

    solver = MGDAParetoSolver()
    shared_params = list(shared_trunk.parameters())

    alpha = solver.apply_pareto_backward([loss_click, loss_ret], shared_params)

    assert len(alpha) == 2
    assert torch.isclose(alpha.sum(), torch.tensor(1.0), atol=1e-4)

    # Shared parameters must have populated gradients
    for p in shared_params:
        assert p.grad is not None, "Shared parameter grad must be populated"
        assert not torch.isnan(p.grad).any(), "Grad must not contain NaN"
        assert p.grad.norm() > 0.0, "Grad norm must be positive"
