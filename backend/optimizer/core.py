"""Shared math used by both optimization modes."""

from __future__ import annotations

import numpy as np


def regularize_covariance(Sigma: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Tikhonov-regularize a covariance matrix for numerical stability -
    and, adaptively, for outright validity. A plain sample covariance is
    guaranteed PSD; a *pairwise-complete* one (each entry computed from
    whatever dates that specific pair of assets both have data for, needed
    to keep a recently-listed asset's shorter history instead of dropping
    it - see optimizer.data.estimate_parameters) is not guaranteed PSD, and
    can end up with a real negative eigenvalue, not just floating-point
    noise around zero. A fixed epsilon this small can't fix that: CVXPY's
    quad_form rejects a matrix that isn't (verifiably) PSD outright, so the
    shift here is widened past whatever the actual most-negative eigenvalue
    is whenever epsilon alone wouldn't cover it.
    """
    n = Sigma.shape[0]
    min_eigenvalue = float(np.linalg.eigvalsh(Sigma).min())
    if min_eigenvalue < -epsilon:
        epsilon = -min_eigenvalue + 1e-6
    return Sigma + epsilon * np.eye(n)


def portfolio_metrics(
    weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, risk_free_rate: float = 0.0
) -> dict:
    """Return, volatility and Sharpe ratio for a weight vector.

    Normalized by how much capital `weights` actually deploys
    (weights.sum()), not assumed to already be 1: enable_flows lets buys and
    sells change total portfolio value, so a weight vector summing to, say,
    1.9 (net buys deploying 90% more capital than the original portfolio)
    is a real, valid solution there - but mu @ weights on its own is then a
    raw dollar-scaled figure, not a percentage return, and would come out
    proportionally inflated by the extra capital rather than reflecting
    actual per-dollar performance. That made an enable_flows result look
    like it "beat" the efficient frontier, which is always calibrated to
    exactly 100% deployed - not a real violation, just two differently-
    scaled numbers being compared as if they matched. Dividing both return
    and volatility by the same total keeps them a true fraction of capital
    actually invested either way; for every other caller weights already
    sums to 1 (enforced by cp.sum(w)==1), so this division is a no-op there.

    Sharpe subtracts risk_free_rate from the (already-normalized) return
    before dividing by volatility - plain return/volatility, with no
    risk-free rate netted out, isn't a real Sharpe ratio: it blows up
    without bound as volatility shrinks toward zero regardless of whether
    there's any actual compensation for risk, which is exactly what happens
    at the low-volatility end of a frontier dominated by a near-cash
    instrument (e.g. a T-bill ETF) - its raw return there is essentially
    just the risk-free rate itself, not alpha, so an un-netted "Sharpe" of
    10+ at that corner is a real solve, not a bug, but a misleading number.
    Defaults to 0.0 (old behavior) so an explicit rate is opt-in.
    """
    total = float(weights.sum())
    scale = 1.0 / total if total > 1e-9 else 1.0
    ret = float(mu @ weights) * scale
    variance = float(weights @ Sigma @ weights) * scale**2
    vol = float(np.sqrt(max(variance, 0.0)))
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-12 else 0.0
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


def effective_n(weights: np.ndarray) -> float:
    """1 / Herfindahl-Hirschman Index (sum of squared weights) - a standard
    concentration metric read as "this portfolio spreads risk/capital like
    an equally-weighted book of this many names," even when it actually
    holds more (or fewer, if concentrated). A single 100%-weighted position
    gives exactly 1.0; n equal positions of 1/n each give exactly n. Not
    normalized by weights.sum() the way portfolio_metrics' return/volatility
    are - HHI is about the *shape* of the allocation, not its scale, so an
    enable_flows result summing to e.g. 1.2 is treated the same as one
    summing to 1.0 with the same relative proportions."""
    total = float(weights.sum())
    if total <= 1e-9:
        return 0.0
    normalized = weights / total
    hhi = float(np.sum(normalized**2))
    return (1.0 / hhi) if hhi > 1e-12 else 0.0
