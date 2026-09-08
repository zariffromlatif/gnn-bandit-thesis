"""
Multi-Objective Pareto Stationarity Solver (MGDA) for Causal Graph Bandits.

Resolves multi-task gradient conflict between short-term CTR engagement
and long-term user retention / anti-churn objectives.

Mathematical Foundation:
------------------------
Let L_1(theta), ..., L_M(theta) be M competing objective losses sharing
representations theta in R^d:
  - L_click: Immediate click / conversion rate
  - L_retention: Multi-day return rate / churn prevention
  - L_safety: Offline batch policy drift / CVaR tail risk

Under standard scalarization (weighted sum L = sum_m w_m L_m), negative
gradient interference (g_i^T g_j < 0) destroys long-term objectives to chase
noisy short-term clicks.

The Multiple Gradient Descent Algorithm (MGDA; Sener & Koltun, NeurIPS 2018;
Desideri, CRAS 2012) finds a Pareto-stationary descent direction by solving
the minimum-norm vector in the convex hull of task gradients:
    min_{alpha in Delta^M} || sum_{m=1}^M alpha_m nabla_theta L_m(theta) ||_2^2
where Delta^M = { alpha in R^M : alpha_m >= 0, sum_m alpha_m = 1 }.

If the minimum norm is 0, the current parameter point theta is Pareto stationary.
Otherwise, the combined vector d = - sum_m alpha_m^* nabla_theta L_m is a common
descent direction satisfying <nabla_theta L_m, d> <= 0 for all m = 1, ..., M.

We solve the quadratic program using the Frank-Wolfe algorithm on the gradient
Gram matrix G in R^{M x M}, which converges in < 25 iterations with negligible
computational overhead.

References:
-----------
- Sener, O. & Koltun, V. "Multi-Task Learning as Multi-Objective Optimization."
  NeurIPS (2018): 527-538.
- Desideri, J.-A. "Multiple-gradient descent algorithm (MGDA) for multiobjective
  optimization." Comptes Rendus Mathematique 350.5-6 (2012): 313-318.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class MGDAParetoSolver:
    """
    Multiple Gradient Descent Algorithm (MGDA) Frank-Wolfe Solver.

    Parameters
    ----------
    max_iters : int
        Maximum iterations for the Frank-Wolfe solver (default: 25).
    tolerance : float
        Early stopping threshold on Frank-Wolfe duality gap (default: 1e-5).
    """

    def __init__(self, max_iters: int = 25, tolerance: float = 1e-5):
        self.max_iters = max_iters
        self.tolerance = tolerance

    def solve_weights(self, grad_matrix: torch.Tensor) -> torch.Tensor:
        """
        Find optimal convex weights alpha* in Delta^M minimizing ||sum_m alpha_m g_m||^2.

        Parameters
        ----------
        grad_matrix : (M, d) tensor containing task gradient vectors.

        Returns
        -------
        alpha : (M,) optimal convex weights on the simplex.
        """
        M = grad_matrix.shape[0]
        if M == 1:
            return torch.ones(1, device=grad_matrix.device, dtype=grad_matrix.dtype)

        # Compute Gram matrix G = grad_matrix @ grad_matrix.T (M x M)
        G = torch.matmul(grad_matrix, grad_matrix.t())  # (M, M)

        # Initialize alpha uniformly on the simplex
        alpha = torch.full((M,), 1.0 / M, device=grad_matrix.device, dtype=grad_matrix.dtype)

        for _ in range(self.max_iters):
            # Gradient of 0.5 * alpha^T G alpha is G alpha
            t = torch.matmul(G, alpha)

            # Linear subproblem: find corner of simplex minimizing <t, e_i>
            i_star = torch.argmin(t)

            # Direction toward corner: d = e_{i^*} - alpha
            d = -alpha.clone()
            d[i_star] += 1.0

            # Duality gap: <t, -d> = <G alpha, alpha - e_{i^*}>
            gap = torch.dot(t, -d)
            if gap <= self.tolerance:
                break

            # Line search: find optimal gamma in [0, 1] minimizing (alpha + gamma d)^T G (alpha + gamma d)
            Gd = torch.matmul(G, d)
            dGd = torch.dot(d, Gd)

            if dGd <= 1e-12:
                gamma = 1.0
            else:
                gamma = torch.clamp(gap / dGd, 0.0, 1.0)

            alpha = alpha + gamma * d

        return alpha

    def compute_pareto_gradient(
        self, losses: List[torch.Tensor], shared_parameters: List[nn.Parameter]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Pareto-optimal gradient for shared parameters across tasks.

        Parameters
        ----------
        losses : List of scalar task losses [L_1, ..., L_M].
        shared_parameters : List of shared network parameters.

        Returns
        -------
        pareto_grad : Flattened Pareto-optimal common descent vector.
        alpha : (M,) task weights on the simplex.
        """
        M = len(losses)
        assert M > 0, "Losses list cannot be empty"

        # Compute per-task gradients with retain_graph=True
        task_grads = []
        for i, loss in enumerate(losses):
            retain = (i < M - 1)
            grads = torch.autograd.grad(
                loss, shared_parameters, retain_graph=retain, allow_unused=True
            )
            # Flatten into a single 1D vector per task
            flat_grad = torch.cat([
                g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1)
                for g, p in zip(grads, shared_parameters)
            ])
            task_grads.append(flat_grad)

        grad_matrix = torch.stack(task_grads, dim=0)  # (M, total_params)

        # Solve for optimal convex weights
        alpha = self.solve_weights(grad_matrix)

        # Combined Pareto-optimal gradient: sum_m alpha_m g_m
        pareto_grad = torch.matmul(alpha, grad_matrix)

        return pareto_grad, alpha

    def apply_pareto_backward(
        self, losses: List[torch.Tensor], shared_parameters: List[nn.Parameter]
    ) -> torch.Tensor:
        """
        Executes Pareto backward pass and populates .grad on shared_parameters.

        Parameters
        ----------
        losses : List of scalar task losses [L_1, ..., L_M].
        shared_parameters : List of shared network parameters.

        Returns
        -------
        alpha : (M,) task weights solved by MGDA.
        """
        pareto_grad, alpha = self.compute_pareto_gradient(losses, shared_parameters)

        # Populate .grad attributes on shared_parameters
        offset = 0
        for p in shared_parameters:
            numel = p.numel()
            g_chunk = pareto_grad[offset : offset + numel].view_as(p)
            if p.grad is None:
                p.grad = g_chunk.clone()
            else:
                p.grad.copy_(g_chunk)
            offset += numel

        return alpha
