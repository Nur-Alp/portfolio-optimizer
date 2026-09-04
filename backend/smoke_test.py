"""Quick end-to-end check of both optimizer modes against the demo universe."""

import numpy as np

from optimizer import GroupLimit, OptimizerParams, solve_existing_portfolio, solve_new_portfolio
from optimizer.data import load_demo_universe

universe = load_demo_universe()
mu, Sigma, labels = universe["mu"], universe["Sigma"], universe["labels"]
n = len(mu)

print(f"Universe: {labels}")

# --- New portfolio: free allocation from scratch ---
sector_limits = [
    GroupLimit(category="sector", group=sector, indices=indices, max_weight=0.40)
    for sector, indices in universe["group_indices"]["sector"].items()
]
new_params = OptimizerParams(risk_aversion=2.0, w_min=0.0, w_max=0.25, group_limits=sector_limits)
new_result = solve_new_portfolio(mu, Sigma, new_params)
print("\n[New portfolio]", new_result.status)
print("weights:", dict(zip(labels, new_result.weights.round(3))))
print(f"return={new_result.expected_return:.2%} vol={new_result.volatility:.2%} sharpe={new_result.sharpe:.2f}")

# --- Existing portfolio: rebalance an equal-weight book, touching only 2 names ---
w_current = np.ones(n) / n
existing_params = OptimizerParams(risk_aversion=2.0, w_min=0.0, w_max=0.30, max_trades=2)
existing_result = solve_existing_portfolio(mu, Sigma, w_current, existing_params)
print("\n[Existing portfolio, max_trades=2]", existing_result.status)
print("weights:", dict(zip(labels, existing_result.weights.round(3))))
print("trades:", dict(zip(labels, existing_result.trades.round(3))))
print(f"n_trades={existing_result.n_trades} return={existing_result.expected_return:.2%} "
      f"vol={existing_result.volatility:.2%} sharpe={existing_result.sharpe:.2f}")
