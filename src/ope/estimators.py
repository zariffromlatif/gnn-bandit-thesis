"""
Off-Policy Evaluation (OPE) estimators for counterfactual policy evaluation.

All estimators answer the same question:

    "If our learned policy pi had been deployed instead of the logging
     policy pi_0, what would the expected reward have been?"

We implement four estimators of increasing sophistication:

1. **IPW**  (Inverse Propensity Weighting) --- unbiased but high variance.
2. **SNIPW** (Self-Normalised IPW) --- biased but lower variance.
3. **DM**  (Direct Method) --- low variance but biased if the reward model
   is misspecified.
4. **DR**  (Doubly Robust) --- combines IPW + DM for the best of both.
   Consistent if *either* the reward model or the propensity model is
   correct.  This is our primary evaluation metric.

References
----------
- Dudik et al., "Doubly Robust Policy Evaluation and Optimization", 2011.
- Saito et al., "Open Bandit Dataset and Pipeline", NeurIPS 2021.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================================
# Result container
# ============================================================================

@dataclass
class OPEResult:
    """Holds the output of an OPE estimate."""
    name:     str
    value:    float      # point estimate of V(pi)
    std:      float      # standard deviation (for confidence intervals)
    n:        int        # number of data points used
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    def __post_init__(self):
        # 95 % confidence interval
        se = self.std / np.sqrt(max(self.n, 1))
        self.ci_lower = self.value - 1.96 * se
        self.ci_upper = self.value + 1.96 * se

    def __repr__(self):
        return (f"{self.name}: {self.value:.6f} "
                f"[{self.ci_lower:.6f}, {self.ci_upper:.6f}]  "
                f"(n={self.n:,})")


# ============================================================================
# Importance weights
# ============================================================================

def _importance_weights(
    pi_new:    np.ndarray,
    pi_old:    np.ndarray,
    actions:   np.ndarray,
    n_actions: int,
    clip:      float = 100.0,
) -> np.ndarray:
    """
    Compute per-sample importance weights  w_t = pi(a_t | x_t) / pi_0(a_t | x_t).

    Parameters
    ----------
    pi_new   : (N, A) action probabilities under the evaluation policy.
    pi_old   : (N,)   propensity scores from the logging policy.
    actions  : (N,)   actions actually taken.
    n_actions: int    number of actions (unused but kept for clarity).
    clip     : float  max allowed weight to limit variance.

    Returns
    -------
    w : (N,) clipped importance weights.
    """
    N = len(actions)
    # pi_new can be (N, A) — pick the probability of the taken action
    if pi_new.ndim == 2:
        pi_a = pi_new[np.arange(N), actions]
    else:
        pi_a = pi_new  # already (N,)

    # Avoid division by zero
    pi_old_safe = np.clip(pi_old, 1e-8, None)
    w = pi_a / pi_old_safe
    return np.clip(w, 0, clip)


# ============================================================================
# 1. Inverse Propensity Weighting (IPW)
# ============================================================================

def ipw(
    rewards:   np.ndarray,
    pi_new:    np.ndarray,
    pi_old:    np.ndarray,
    actions:   np.ndarray,
    n_actions: int,
    clip:      float = 100.0,
) -> OPEResult:
    """
    Standard IPS / IPW estimator.

        V_IPW(pi) = (1/n) * sum_t  w_t * r_t

    Unbiased under correct propensities but can have extreme variance
    when w_t is large.

    Parameters
    ----------
    rewards   : (N,) observed rewards.
    pi_new    : (N, A) evaluation policy action probabilities.
    pi_old    : (N,) logging policy propensity scores.
    actions   : (N,) actions taken by the logging policy.
    n_actions : int
    clip      : float, importance weight clipping bound.
    """
    w = _importance_weights(pi_new, pi_old, actions, n_actions, clip)
    weighted = w * rewards
    return OPEResult(
        name="IPW",
        value=float(weighted.mean()),
        std=float(weighted.std()),
        n=len(rewards),
    )


# ============================================================================
# 2. Self-Normalised IPW (SNIPW)
# ============================================================================

def snipw(
    rewards:   np.ndarray,
    pi_new:    np.ndarray,
    pi_old:    np.ndarray,
    actions:   np.ndarray,
    n_actions: int,
    clip:      float = 100.0,
) -> OPEResult:
    """
    Self-normalised IPW.

        V_SNIPW(pi) = sum(w_t * r_t) / sum(w_t)

    Trades a small bias for much lower variance than raw IPW.
    """
    w = _importance_weights(pi_new, pi_old, actions, n_actions, clip)
    w_sum = w.sum()
    if w_sum < 1e-12:
        return OPEResult(name="SNIPW", value=0.0, std=0.0, n=len(rewards))

    value = float((w * rewards).sum() / w_sum)
    # Bootstrap-style std estimate
    normalised = (w * rewards) / w_sum
    return OPEResult(
        name="SNIPW",
        value=value,
        std=float(normalised.std()),
        n=len(rewards),
    )


# ============================================================================
# 3. Direct Method (DM)
# ============================================================================

def direct_method(
    reward_model: np.ndarray,
    pi_new:       np.ndarray,
    n_actions:    int,
) -> OPEResult:
    """
    Direct Method (model-based) estimator.

        V_DM(pi) = (1/n) * sum_t sum_a  pi(a | x_t) * r_hat(x_t, a)

    Requires a reward model r_hat(x, a).  Low variance but biased
    if the model is misspecified.

    Parameters
    ----------
    reward_model : (N, A) predicted rewards r_hat(x_t, a) for all actions.
    pi_new       : (N, A) evaluation policy action probabilities.
    n_actions    : int
    """
    # Expected reward under the new policy
    per_sample = (pi_new * reward_model).sum(axis=1)     # (N,)
    return OPEResult(
        name="DM",
        value=float(per_sample.mean()),
        std=float(per_sample.std()),
        n=len(per_sample),
    )


# ============================================================================
# 4. Doubly Robust (DR)  — PRIMARY METRIC
# ============================================================================

def doubly_robust(
    rewards:      np.ndarray,
    pi_new:       np.ndarray,
    pi_old:       np.ndarray,
    actions:      np.ndarray,
    n_actions:    int,
    reward_model: np.ndarray,
    clip:         float = 100.0,
) -> OPEResult:
    """
    Doubly Robust estimator (Dudik et al., 2011).

        V_DR(pi) = V_DM(pi)
                   + (1/n) sum_t  w_t * (r_t  -  r_hat(x_t, a_t))

    Consistent if *either* the propensity model or the reward model is
    correctly specified.  This is our primary evaluation metric.

    Parameters
    ----------
    rewards      : (N,) observed rewards.
    pi_new       : (N, A) evaluation policy action probabilities.
    pi_old       : (N,) logging policy propensity scores.
    actions      : (N,) actions taken.
    n_actions    : int
    reward_model : (N, A) predicted rewards for all actions.
    clip         : float
    """
    N = len(rewards)

    # DM component
    dm_per_sample = (pi_new * reward_model).sum(axis=1)  # (N,)

    # IPW correction term
    w = _importance_weights(pi_new, pi_old, actions, n_actions, clip)
    r_hat_taken = reward_model[np.arange(N), actions]
    correction  = w * (rewards - r_hat_taken)

    dr_per_sample = dm_per_sample + correction

    return OPEResult(
        name="DR",
        value=float(dr_per_sample.mean()),
        std=float(dr_per_sample.std()),
        n=N,
    )


# ============================================================================
# Convenience: run all estimators at once
# ============================================================================

def evaluate_all(
    rewards:      np.ndarray,
    pi_new:       np.ndarray,
    pi_old:       np.ndarray,
    actions:      np.ndarray,
    n_actions:    int,
    reward_model: Optional[np.ndarray] = None,
    clip:         float = 100.0,
) -> dict[str, OPEResult]:
    """
    Run all four OPE estimators and return results keyed by name.

    If *reward_model* is ``None``, DM and DR are skipped (only IPW and SNIPW
    are returned).
    """
    results = {}
    results["IPW"]   = ipw(rewards, pi_new, pi_old, actions, n_actions, clip)
    results["SNIPW"] = snipw(rewards, pi_new, pi_old, actions, n_actions, clip)

    if reward_model is not None:
        results["DM"] = direct_method(reward_model, pi_new, n_actions)
        results["DR"] = doubly_robust(
            rewards, pi_new, pi_old, actions, n_actions, reward_model, clip
        )

    return results
