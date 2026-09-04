"""The two optimization modes.

New portfolio
-------------
No existing holdings. Free to allocate across the whole universe in one
shot - this is the classic Markowitz problem from the notebook (position
limits + sector caps), solved as a plain convex QP.

Existing portfolio
------------------
Starts from ``w_current`` and rebalances it. Two independent knobs:

- ``turnover_max``: caps total value traded (sum of |w_new - w_old|).
  Still a convex QP - no solver change needed.
- ``max_trades``: caps the *number* of positions allowed to change at
  all (e.g. "only touch 1 or 2 names"). This needs a binary "traded?"
  indicator per asset, which turns the problem into a Mixed-Integer QP.
  CVXPY's default solver (CLARABEL) can't do MIQP, so this path requires
  the SCIP solver, which must be installed (`pip install pyscipopt`) and
  is only invoked when max_trades is actually set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np

from .core import effective_n, portfolio_metrics, regularize_covariance
from .params import OptimizerParams

# Default aggregate bounds on total deployed capital (as a fraction of the
# starting portfolio's own value) under enable_flows - see the comment where
# these are applied in _solve_with_flows for why an aggregate cap is needed
# even when every per-trade cap is left uncapped.
FLOWS_MIN_TOTAL_CAPITAL = 0.5
FLOWS_MAX_TOTAL_CAPITAL = 2.0

# Dinkelbach fractional-programming iteration (see _solve_with_flows) - a
# generous cap on repeated SCIP solves, and a tolerance on the ratio q's own
# convergence (q is a return-scale quantity, so 1e-9 is well below anything
# meaningful while still being loose enough to stop promptly once q settles).
MAX_DINKELBACH_ITERS = 20
DINKELBACH_TOL = 1e-9


@dataclass
class OptimizationResult:
    status: str
    weights: np.ndarray | None
    expected_return: float | None
    volatility: float | None
    sharpe: float | None
    n_positions: int | None
    trades: np.ndarray | None = None
    n_trades: int | None = None
    # Only set when params.enable_flows was used - which of n_trades'
    # nonzero moves were reallocations (net-zero, funded internally) vs.
    # buys (new external capital) vs. sells (outright withdrawals).
    n_reallocations: int | None = None
    n_buys: int | None = None
    n_sells: int | None = None
    # Per-asset classification (same length/order as weights/trades):
    # "reallocation"/"buy"/"sell" for a real nonzero move of that kind,
    # None for an untouched asset. The explicit, auditable version of the
    # three counts above - lets a caller show exactly which asset did what,
    # not just how many of each. All three categories are decided together
    # in one solve, not applied in any sequence - there is no "order" for
    # this to disagree with.
    trade_kinds: list[str | None] | None = None
    # 1/HHI of the returned weights - see optimizer.core.effective_n. Always
    # populated on a successful solve, diversify_tolerance or not, since
    # it's informative either way (a plain solve can land concentrated or
    # spread out on its own).
    effective_n: float | None = None
    # Only populated when diversify_tolerance was set AND stage 2 (the
    # entropy re-solve) actually succeeded - the pre-diversification
    # portfolio's own return/volatility/effective_n, for an explicit
    # "here's what you gave up" comparison against the (now diversified)
    # weights/expected_return/volatility/effective_n above.
    pre_diversify_return: float | None = None
    pre_diversify_volatility: float | None = None
    pre_diversify_effective_n: float | None = None

    @property
    def ok(self) -> bool:
        return self.weights is not None


def _base_constraints(
    w: cp.Variable, mu: np.ndarray, params: OptimizerParams, Sigma_reg: np.ndarray | None = None
) -> list:
    constraints = [
        cp.sum(w) == 1,
        w >= params.w_min,
        w <= params.w_max,
    ]
    for limit in params.group_limits:
        constraints.append(cp.sum(w[limit.indices]) <= limit.max_weight)
    if params.target_return is not None:
        constraints.append(w @ mu == params.target_return)
    if params.target_volatility is not None:
        if Sigma_reg is None:
            raise ValueError("Sigma_reg is required when target_volatility is set")
        # Convex: a quadratic form is convex regardless of sign, so "<=" is a
        # proper convex constraint (an equality here would not be).
        constraints.append(cp.quad_form(w, Sigma_reg) <= params.target_volatility**2)
    return constraints


def compute_return_bounds(mu: np.ndarray, params: OptimizerParams) -> tuple[float, float]:
    """The actual feasible range for portfolio expected return given the
    position/sector constraints (ignoring target_return itself - that's what
    this is computing bounds for). Position and sector caps typically make
    this narrower than [min(mu), max(mu)]: e.g. a 25% position cap makes it
    impossible to hit an individual asset's own extreme return, since that
    would require putting the whole book into one name.
    """
    n = len(mu)
    w = cp.Variable(n)
    constraints = _base_constraints(w, mu, replace(params, target_return=None, target_volatility=None))

    lo_problem = cp.Problem(cp.Minimize(w @ mu), constraints)
    lo_problem.solve(solver=cp.CLARABEL)
    hi_problem = cp.Problem(cp.Maximize(w @ mu), constraints)
    hi_problem.solve(solver=cp.CLARABEL)

    return float(lo_problem.value), float(hi_problem.value)


def compute_volatility_bounds(mu: np.ndarray, Sigma: np.ndarray, params: OptimizerParams) -> tuple[float, float]:
    """Feasible range for the target_volatility slider.

    min_volatility is exact: the real global minimum-variance portfolio's
    own volatility under the position/sector constraints - anything below
    that is infeasible by definition, no matter the return.

    max_volatility is a deliberate overestimate, not an exact bound: the
    true maximum achievable variance is maximizing a *convex* quadratic
    form over a polytope, which is NP-hard in general (unlike minimizing
    it), so this uses the highest individual asset's own volatility as a
    generous ceiling instead. A target above what the constraints can
    actually reach still just comes back "infeasible" from the solve
    itself - the same graceful failure every other constraint already has -
    this bound only shapes the slider's range, not solve correctness.
    """
    n = len(mu)
    Sigma_reg = regularize_covariance(Sigma, params.regularization)
    w = cp.Variable(n)
    constraints = _base_constraints(w, mu, replace(params, target_return=None, target_volatility=None))
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma_reg)), constraints)
    problem.solve(solver=cp.CLARABEL)
    min_volatility = float(np.sqrt(max(problem.value, 0.0)))

    max_volatility = float(np.sqrt(np.diag(Sigma_reg)).max())
    return min_volatility, max(max_volatility, min_volatility * 1.01)


def compute_min_group_weight(n: int, w_min: float, w_max: float, indices: list[int]) -> float:
    """The smallest cap for which *this one group* can still be brought down
    to, given only the per-position bounds (no return/covariance, no other
    group's constraints - this is pure feasibility for that group alone,
    since each group now gets its own independent slider rather than sharing
    one category-wide cap). Not simply len(indices) * w_min: the rest of the
    book still has to absorb 1 - (this group's weight) within its own w_max
    per position, which can push this floor higher than the group's own
    naive per-asset minimum would suggest.
    """
    w = cp.Variable(n)
    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        w <= w_max,
    ]
    problem = cp.Problem(cp.Minimize(cp.sum(w[indices])), constraints)
    problem.solve(solver=cp.CLARABEL)
    return float(problem.value)


def _objective_expr(w: cp.Variable, mu: np.ndarray, Sigma_reg: np.ndarray, params: OptimizerParams):
    """The raw scalar expression each mode actually optimizes, plus whether
    it's a minimization - split out from _objective() so _diversify_weights
    can reuse the exact same expression as a *constraint* (stay within some
    tolerance of it) rather than duplicating the three-way branch."""
    if params.target_return is not None:
        return cp.quad_form(w, Sigma_reg), True
    if params.target_volatility is not None:
        # The volatility cap itself lives in _base_constraints (it's a
        # constraint, not part of the objective); here we just maximize
        # return within it - the mirror image of the target_return case.
        return mu @ w, False
    return mu @ w - params.risk_aversion * cp.quad_form(w, Sigma_reg), False


def _objective(w: cp.Variable, mu: np.ndarray, Sigma_reg: np.ndarray, params: OptimizerParams):
    expr, is_minimize = _objective_expr(w, mu, Sigma_reg, params)
    return cp.Minimize(expr) if is_minimize else cp.Maximize(expr)


def _diversify_weights(
    w: cp.Variable, mu: np.ndarray, Sigma_reg: np.ndarray,
    constraints: list, params: OptimizerParams, optimal_value: float,
) -> np.ndarray | None:
    """Stage 2 of the opt-in diversification pass: re-solve the exact same
    feasible region (same constraints list, unchanged), but maximize entropy
    -sum(w*log(w)) - a standard concentration metric, concave so this stays
    a convex problem via CVXPY's cp.entr atom - instead of the original
    objective. The only new constraint is that stage 1's own objective may
    not get worse than params.diversify_tolerance of its optimal value,
    computed relative to |optimal_value| (with a floor, so a near-zero
    optimum doesn't collapse the tolerance to nothing).

    Returns None - caller falls back to the stage-1 weights - if this second
    solve doesn't land on an optimal status (e.g. the tolerance band is too
    tight to allow any real spread, an edge case rather than the norm) or if
    diversify_tolerance wasn't requested in the first place.
    """
    if params.diversify_tolerance is None:
        return None
    expr, is_minimize = _objective_expr(w, mu, Sigma_reg, params)
    delta = params.diversify_tolerance * max(abs(optimal_value), 1e-9)
    near_optimal = expr <= optimal_value + delta if is_minimize else expr >= optimal_value - delta
    diversify_problem = cp.Problem(cp.Maximize(cp.sum(cp.entr(w))), constraints + [near_optimal])
    diversify_problem.solve(solver=cp.CLARABEL)
    if diversify_problem.status not in ("optimal", "optimal_inaccurate"):
        return None
    return w.value


def _package_result(
    status: str, w_value, mu, Sigma, w_current=None,
    z_realloc=None, z_buy=None, z_sell=None,
    pre_diversify_weights: np.ndarray | None = None,
    risk_free_rate: float = 0.0,
) -> OptimizationResult:
    if status not in ("optimal", "optimal_inaccurate") or w_value is None:
        return OptimizationResult(
            status=status, weights=None, expected_return=None,
            volatility=None, sharpe=None, n_positions=None,
        )

    # enable_flows' weights are allowed to move in either direction relative
    # to the position bounds' own sign convention (w_min may be 0, but a
    # sell can legitimately drive a specific holding toward 0 from above) -
    # clip(min=0) still applies since shorting was never supported, only the
    # *reallocation* path's degenerate case this guarded against originally.
    weights = np.asarray(w_value).clip(min=0)
    metrics = portfolio_metrics(weights, mu, Sigma, risk_free_rate)
    n_positions = int(np.sum(weights > 1e-4))

    # pre_diversify_weights is only ever passed when diversify_tolerance was
    # actually requested and stage 2 succeeded - the caller's own stage-1
    # (pure-optimal, pre-diversification) weights, kept so the UI can show
    # "here's what you gave up" alongside "here's what you got" rather than
    # silently replacing one with the other.
    pre_diversify_return = pre_diversify_volatility = pre_diversify_effective_n = None
    if pre_diversify_weights is not None:
        pre_metrics = portfolio_metrics(pre_diversify_weights, mu, Sigma, risk_free_rate)
        pre_diversify_return = pre_metrics["return"]
        pre_diversify_volatility = pre_metrics["volatility"]
        pre_diversify_effective_n = effective_n(pre_diversify_weights)

    trades = None
    n_trades = None
    if w_current is not None:
        trades = weights - w_current
        n_trades = int(np.sum(np.abs(trades) > 1e-6))

    # An indicator can end up "on" for an asset the solver never actually
    # needed to move (nothing penalizes an idle indicator, so SCIP is free
    # to flip one on arbitrarily at a zero-cost corner) - only count it if
    # the trade it's supposedly gating is also actually nonzero, so the
    # reported counts reflect real moves, not unused slack in the MIQP.
    def _active_mask(z) -> np.ndarray | None:
        if z is None or trades is None:
            return None
        return (np.asarray(z.value).round() > 0.5) & (np.abs(trades) > 1e-6)

    realloc_active = _active_mask(z_realloc)
    buy_active = _active_mask(z_buy)
    sell_active = _active_mask(z_sell)

    n_reallocations = int(np.sum(realloc_active)) if realloc_active is not None else None
    n_buys = int(np.sum(buy_active)) if buy_active is not None else None
    n_sells = int(np.sum(sell_active)) if sell_active is not None else None

    trade_kinds = None
    if realloc_active is not None and buy_active is not None and sell_active is not None:
        trade_kinds = [
            "reallocation" if r else "buy" if b else "sell" if s else None
            for r, b, s in zip(realloc_active, buy_active, sell_active)
        ]

    return OptimizationResult(
        status=status,
        weights=weights,
        expected_return=metrics["return"],
        volatility=metrics["volatility"],
        sharpe=metrics["sharpe"],
        n_positions=n_positions,
        trades=trades,
        n_trades=n_trades,
        n_reallocations=n_reallocations,
        n_buys=n_buys,
        n_sells=n_sells,
        trade_kinds=trade_kinds,
        effective_n=effective_n(weights),
        pre_diversify_return=pre_diversify_return,
        pre_diversify_volatility=pre_diversify_volatility,
        pre_diversify_effective_n=pre_diversify_effective_n,
    )


def solve_new_portfolio(mu: np.ndarray, Sigma: np.ndarray, params: OptimizerParams) -> OptimizationResult:
    """Build a portfolio from scratch - no existing positions to disturb."""
    n = len(mu)
    Sigma_reg = regularize_covariance(Sigma, params.regularization)

    w = cp.Variable(n)
    constraints = _base_constraints(w, mu, params, Sigma_reg)
    problem = cp.Problem(_objective(w, mu, Sigma_reg, params), constraints)
    problem.solve(solver=cp.CLARABEL)

    w_value = w.value
    pre_diversify_weights = None
    if problem.status in ("optimal", "optimal_inaccurate"):
        # Capture stage 1's own weights *before* _diversify_weights - it
        # reuses this same `w` Variable for its own solve, which overwrites
        # w.value as a side effect, so grabbing it after would just return
        # the diversified result under a different name.
        stage1_weights = np.asarray(w_value).clip(min=0)
        diversified = _diversify_weights(w, mu, Sigma_reg, constraints, params, problem.value)
        if diversified is not None:
            pre_diversify_weights = stage1_weights
            w_value = diversified

    return _package_result(
        problem.status, w_value, mu, Sigma,
        pre_diversify_weights=pre_diversify_weights, risk_free_rate=params.risk_free_rate,
    )


def build_frontier(
    mu: np.ndarray, Sigma: np.ndarray, params: OptimizerParams, n_points: int = 30
) -> list[tuple[float, float]]:
    """Sweep target returns across the feasible range and solve min-variance
    at each one - the classic "bullet"-shaped efficient frontier curve
    (both branches), under the current position/sector constraints only
    (turnover/max_trades are rebalancing concerns, not properties of the
    achievable risk/return set itself, so they're deliberately excluded
    here - same as the original notebook). Reuses solve_new_portfolio at
    each point rather than a bespoke solve.

    The exact endpoints returned by compute_return_bounds are themselves
    corner solutions (min/max return achieved by pinning box/sector
    constraints as tightly as possible) - solving exactly at them can land
    on a degenerate, concentrated allocation whose volatility jumps sharply
    relative to its neighbors, which used to look like a spike if swept
    through the middle of the curve at low resolution. A small inset avoids
    sweeping *through* those exact corners at n_points' resolution - but the
    corners are themselves real, achievable portfolios (e.g. target_volatility
    mode set to a loose enough cap collapses to exactly the max-return
    corner), so they're solved for directly and appended at the ends rather
    than left out: otherwise the plotted curve stops short of points the
    optimizer can legitimately return, making a correct result look like it
    fell outside the frontier.
    """
    lo, hi = compute_return_bounds(mu, params)
    margin = (hi - lo) * 0.01
    targets = np.linspace(lo + margin, hi - margin, n_points)

    # The plotted curve is the theoretical efficient frontier itself, not
    # any one portfolio the user is actually choosing - diversify_tolerance
    # is a preference about which pick to land on, so it's stripped here the
    # same way target_return/target_volatility already are, rather than
    # bending the frontier's own shape toward more diversified points.
    frontier_params = replace(params, diversify_tolerance=None)

    points = []
    lo_corner = solve_new_portfolio(
        mu, Sigma, replace(frontier_params, target_return=float(lo), target_volatility=None)
    )
    if lo_corner.ok:
        points.append((lo_corner.expected_return, lo_corner.volatility))
    for target in targets:
        result = solve_new_portfolio(
            mu, Sigma, replace(frontier_params, target_return=float(target), target_volatility=None)
        )
        if result.ok:
            points.append((result.expected_return, result.volatility))
    hi_corner = solve_new_portfolio(
        mu, Sigma, replace(frontier_params, target_return=float(hi), target_volatility=None)
    )
    if hi_corner.ok:
        points.append((hi_corner.expected_return, hi_corner.volatility))
    return points


def solve_existing_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_current: np.ndarray,
    params: OptimizerParams,
) -> OptimizationResult:
    """Rebalance an existing book, optionally capping how many names move."""
    if params.enable_flows:
        return _solve_with_flows(mu, Sigma, w_current, params)

    n = len(mu)
    Sigma_reg = regularize_covariance(Sigma, params.regularization)

    w = cp.Variable(n)
    constraints = _base_constraints(w, mu, params, Sigma_reg)

    if params.turnover_max is not None:
        constraints.append(cp.norm(w - w_current, 1) <= params.turnover_max)

    if params.max_trades is not None:
        # Cardinality constraint: at most `max_trades` positions may change.
        # z_i = 1 means "asset i is allowed to move"; big-M ties the two
        # together. Per-asset M is the largest that position could possibly
        # move given w in [w_min, w_max] and its actual starting point -
        # max(w_max - w_current, w_current - w_min), not the flat
        # w_max - w_min. A current holding already more concentrated than
        # w_max (real for an uploaded portfolio) needs a bigger M on the
        # "sell it down" side than a fresh position starting from w_min
        # would, otherwise the constraint blocks a legal move to w_max and
        # can make the whole problem spuriously infeasible.
        z = cp.Variable(n, boolean=True)
        big_m = np.maximum(params.w_max - w_current, w_current - params.w_min)
        constraints += [
            w - w_current <= cp.multiply(z, big_m),
            w_current - w <= cp.multiply(z, big_m),
            cp.sum(z) <= params.max_trades,
        ]
        problem = cp.Problem(_objective(w, mu, Sigma_reg, params), constraints)
        problem.solve(solver=cp.SCIP)
        return _package_result(
            problem.status, w.value, mu, Sigma, w_current=w_current, risk_free_rate=params.risk_free_rate,
        )

    problem = cp.Problem(_objective(w, mu, Sigma_reg, params), constraints)
    problem.solve(solver=cp.CLARABEL)

    w_value = w.value
    pre_diversify_weights = None
    if problem.status in ("optimal", "optimal_inaccurate"):
        stage1_weights = np.asarray(w_value).clip(min=0)
        diversified = _diversify_weights(w, mu, Sigma_reg, constraints, params, problem.value)
        if diversified is not None:
            pre_diversify_weights = stage1_weights
            w_value = diversified

    return _package_result(
        problem.status, w_value, mu, Sigma, w_current=w_current,
        pre_diversify_weights=pre_diversify_weights, risk_free_rate=params.risk_free_rate,
    )


def _solve_with_flows(
    mu: np.ndarray, Sigma: np.ndarray, w_current: np.ndarray, params: OptimizerParams
) -> OptimizationResult:
    """Rebalance with three independently-capped kinds of movement per
    asset, each mutually exclusive (an asset is reallocated, bought into,
    sold from, or left untouched - never more than one at a time):

    - reallocation: internal trading, funded entirely by other reallocated
      assets - not just "capped the same as before", but *required* to net
      to zero in aggregate (see realloc_amount below). Without that, an
      earlier version of this let a single asset "reallocate" upward with
      nothing offsetting it, which (a) isn't a reallocation at all - there's
      no such thing as reallocating one position by itself, it takes at
      least two - and (b) gave the solver a free way to dodge a tight
      per-asset buy/sell cap: since reallocation had no per-asset limit of
      its own, it would just relabel what should have been a capped buy as
      a "reallocation" instead. Net-zero closes both holes at once.
    - buy: funded by new external capital - trade must be >=0, capped in
      size by buy_amount_max (or a tighter per-asset override).
    - sell: an outright withdrawal - trade must be <=0, capped in
      magnitude by sell_amount_max (or a tighter per-asset override).

    Each asset's total trade is the *sum* of three components (realloc/buy/
    sell), each independently bounded by its own category's cap gated on
    that category's own indicator - not one shared `trade` bounded by
    whichever category happens to be active, which is what let a single
    variable satisfy multiple categories' worth of slack simultaneously.
    Only the realloc component carries the extra sum-to-zero constraint;
    buy/sell components deliberately don't, since a nonzero net buy or sell
    total is the entire point of this mode (see enable_flows' docstring).

    Unlike the plain path, `x` is NOT constrained to sum to 1: total
    portfolio value is free to grow (net buys) or shrink (net sells) by
    however much the solve actually uses. Every weight, group cap, and
    return/volatility figure this produces is therefore relative to the
    *original* portfolio value, not the post-flow one - see
    OptimizerParams.enable_flows' docstring. That's what keeps this a
    convex MIQP: normalizing by the new (decision-dependent) total would
    require dividing by a variable, which isn't convex.

    The three-way bounds below need no big-M trick: since each category's
    own [lower, upper] bracket straddles zero (buys' lower bound is exactly
    0, sells' upper bound is exactly 0, reallocation's is symmetric), a
    component whose indicator is off gets bound to exactly 0 for free, with
    no separate "untouched" constraint needed.
    """
    n = len(mu)
    Sigma_reg = regularize_covariance(Sigma, params.regularization)

    x = cp.Variable(n)
    z_realloc = cp.Variable(n, boolean=True)
    z_buy = cp.Variable(n, boolean=True)
    z_sell = cp.Variable(n, boolean=True)
    realloc_amount = cp.Variable(n)
    buy_amount = cp.Variable(n)
    sell_amount = cp.Variable(n)

    realloc_cap = (
        params.reallocation_amount_max
        if params.reallocation_amount_max is not None
        else params.w_max - params.w_min
    )
    default_buy_cap = params.buy_amount_max if params.buy_amount_max is not None else params.w_max
    default_sell_cap = params.sell_amount_max if params.sell_amount_max is not None else params.w_max
    buy_caps = np.array([params.per_asset_buy_limits.get(i, default_buy_cap) for i in range(n)])
    sell_caps = np.array([params.per_asset_sell_limits.get(i, default_sell_cap) for i in range(n)])

    x_expr = w_current + realloc_amount + buy_amount + sell_amount

    # Deliberately not _base_constraints(): its cp.sum(w)==1 is exactly what
    # a flows solve relaxes (total value is free to grow/shrink), so the
    # position-bound/group/target constraints are rebuilt directly here
    # rather than trying to filter one specific constraint back out of that
    # shared helper's output - target_return's equality constraint would be
    # structurally indistinguishable from sum(w)==1 to any such filter.
    constraints: list = [
        x == x_expr,
        x >= params.w_min,
        x <= params.w_max,
        z_realloc + z_buy + z_sell <= 1,
        realloc_amount <= realloc_cap * z_realloc,
        realloc_amount >= -realloc_cap * z_realloc,
        cp.sum(realloc_amount) == 0,
        buy_amount <= cp.multiply(buy_caps, z_buy),
        buy_amount >= 0,
        sell_amount >= -cp.multiply(sell_caps, z_sell),
        sell_amount <= 0,
    ]
    for limit in params.group_limits:
        constraints.append(cp.sum(x[limit.indices]) <= limit.max_weight)
    # total_capital is linear in the decision variables (realloc nets to zero,
    # so only buy/sell move it), so scaling a target by it - rather than
    # dividing x @ mu / total_capital by it - keeps both of these an exact,
    # convex expression of the *normalized* (per-dollar) figure
    # portfolio_metrics() actually reports, instead of pinning target_return/
    # target_volatility to the raw dollar-scale x @ mu / quad_form(x, ...)
    # the way an un-scaled comparison would: x @ mu == target_return would
    # otherwise demand a raw dollar sum equal a percentage, drifting from the
    # displayed return by whatever factor total_capital ends up away from 1.
    total_capital = cp.sum(x)
    # Per-trade caps (buy_amount_max/sell_amount_max/per_asset_*_limits) are
    # opt-in and, left unset, genuinely uncapped per asset - matching a real
    # buy/sell order, which has no inherent size limit of its own either.
    # But nothing else then bounds the *aggregate* across every asset at
    # once, and the objective below can't be normalized by total_capital
    # (dividing by a decision variable breaks convexity - see return_expr's
    # own comment), so an unbounded aggregate lets the solver keep preferring
    # more total capital in above-average assets over the truly efficient
    # (100%-capital) frontier point, since more dollars at a decent return
    # genuinely is more raw dollar profit even as the displayed *percentage*
    # return it corresponds to gets diluted. FLOWS_MIN/MAX_TOTAL_CAPITAL is a
    # deliberately generous default band (the true efficient-frontier corner
    # only ever needs exactly 100% capital, well inside it) that keeps real,
    # substantial buy/sell activity possible while keeping the solver from
    # treating "raise more capital" as a free lever with no ceiling.
    constraints.append(total_capital >= FLOWS_MIN_TOTAL_CAPITAL)
    constraints.append(total_capital <= FLOWS_MAX_TOTAL_CAPITAL)
    if params.target_return is not None:
        constraints.append(x @ mu == params.target_return * total_capital)
    if params.target_volatility is not None:
        # NOT quad_form(x, Sigma_reg) <= target_volatility**2 * total_capital
        # (only total_capital, not squared) - that was tried and is wrong:
        # portfolio_metrics() computes displayed volatility as
        # sqrt(quad_form(x)) / total_capital, so the constraint that actually
        # matches "displayed volatility <= target" is
        # quad_form(x) <= target_volatility**2 * total_capital**2 (squared).
        # But total_capital**2 is convex, not affine/concave, so
        # "quad_form(x) <= convex expression" isn't a constraint CVXPY can
        # verify as convex - DCP requires convex <= concave. The fix is to
        # take the square root of both sides *before* handing it to CVXPY,
        # which turns it into norm(sqrt_Sigma @ x) <= target_volatility *
        # total_capital: a convex norm on the left, a genuinely affine
        # (linear in x, degree 1) expression on the right - a standard,
        # DCP-valid second-order-cone constraint. sqrt_Sigma (an n x n matrix
        # with sqrt_Sigma.T @ sqrt_Sigma == Sigma_reg) is computed once via
        # eigendecomposition, clipping any tiny negative eigenvalues from
        # floating-point noise to 0 first since Sigma_reg is only guaranteed
        # PSD, not strictly positive-definite (Cholesky would need the
        # latter).
        eigvals, eigvecs = np.linalg.eigh(Sigma_reg)
        sqrt_Sigma = np.diag(np.sqrt(np.clip(eigvals, 0, None))) @ eigvecs.T
        constraints.append(cp.norm(sqrt_Sigma @ x, 2) <= params.target_volatility * total_capital)
    if params.max_reallocations is not None:
        constraints.append(cp.sum(z_realloc) <= params.max_reallocations)
    if params.max_buys is not None:
        constraints.append(cp.sum(z_buy) <= params.max_buys)
    if params.max_sells is not None:
        constraints.append(cp.sum(z_sell) <= params.max_sells)
    # Same meaning as the plain path's turnover_max (total |w_new - w_old|
    # traded), just expressed in these three components instead of a single
    # w: each asset falls into at most one of {reallocated, bought, sold}, so
    # its trade's absolute size is |realloc_amount| (signed, needs the norm)
    # or exactly buy_amount (already >=0) or exactly -sell_amount (already
    # <=0) - never more than one of the three at once, so summing all three
    # is the real total turnover, not a double count.
    turnover_expr = cp.norm(realloc_amount, 1) + cp.sum(buy_amount) - cp.sum(sell_amount)
    if params.turnover_max is not None:
        constraints.append(turnover_expr <= params.turnover_max)

    risk_expr = cp.quad_form(x, Sigma_reg)
    # Same tie-break rationale as before: whenever the objective below is
    # flat across every feasible trade, the solver is free to return *any*
    # tied feasible point - confirmed live, identical settings besides a
    # slightly different sell cap flipped the result between "sold every
    # position to zero" and "no trade at all". A turnover penalty far
    # smaller than any real mu difference in this universe (~1e-3, not 1e-6)
    # breaks that tie toward the untraded portfolio without ever overriding
    # an actual return/risk tradeoff.
    tie_break = 1e-6 * turnover_expr

    if params.target_return is not None:
        # No return term needed here at all: target_return's constraint above
        # already pins the normalized return exactly, so risk minimization is
        # the whole objective - nothing to normalize, no ratio to chase.
        objective = cp.Minimize(risk_expr + tie_break)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCIP)
        return _package_result(
            problem.status, x.value, mu, Sigma, w_current=w_current,
            z_realloc=z_realloc, z_buy=z_buy, z_sell=z_sell, risk_free_rate=params.risk_free_rate,
        )

    # Both remaining branches are genuinely maximizing a ratio - (mu @ x -
    # risk_aversion * risk) / total_capital for plain risk-aversion mode, or
    # just (mu @ x) / total_capital for target_volatility mode (its risk
    # term lives in the constraint above, not the objective) - which is
    # exactly what portfolio_metrics() reports and what the frontier's own
    # points are computed at. x isn't constrained to sum to 1 here (buys/
    # sells move total deployed capital), so that ratio can't be handed to
    # CVXPY directly: dividing by a decision variable isn't convex.
    #
    # Dinkelbach's algorithm (Dinkelbach 1967) solves exactly this shape of
    # problem - maximize f(x)/g(x) for f concave, g positive affine, over any
    # convex feasible region - without ever forming the ratio inside the
    # solver: repeatedly maximize f(x) - q * g(x) for a scalar q, updating
    # q <- f(x*)/g(x*) after each solve, until q stops moving. It's provably
    # monotonically convergent for this concave/affine shape and typically
    # takes only a handful of iterations. Each iteration is a QP/MIQP of the
    # exact same structure already built above - just a different linear
    # coefficient on total_capital - so this reuses the whole constraint set
    # as-is rather than needing a Charnes-Cooper substitution (which would
    # additionally have to re-derive every binary-gated constraint above -
    # the mutual-exclusivity constraint, the three per-category caps, the
    # cardinality/turnover limits - in a transformed variable space). The
    # tradeoff is repeated SCIP solves instead of one; MAX_DINKELBACH_ITERS
    # bounds the worst case, and convergence is checked well before that in
    # practice.
    #
    # Verified against the original "why doesn't lambda=0 land on the
    # frontier corner" complaint: with reallocation available, this now
    # reaches the same point compute_return_bounds' hi_problem finds
    # independently, to solver tolerance - not just close, exact.
    lam = 0.0 if params.target_volatility is not None else params.risk_aversion

    def _f_value(x_val: np.ndarray) -> float:
        return float(mu @ x_val) - lam * float(x_val @ Sigma_reg @ x_val)

    q = float(mu @ w_current)  # w_current's own ratio (total_capital == 1 there) - a reasonable start
    # best_status/best_x_value track the last *successful* iterate
    # specifically, not whichever iteration happened to run last: the
    # feasible region never changes across iterations (only q, an objective
    # coefficient, does), so a later solve failing after an earlier one
    # succeeded would only ever be solver flakiness, not a real
    # infeasibility - and should still report the good solution already in
    # hand rather than discarding it for a spurious later failure.
    best_status = "infeasible"
    best_x_value = None
    for _ in range(MAX_DINKELBACH_ITERS):
        objective = cp.Maximize(mu @ x - lam * risk_expr - q * total_capital - tie_break)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCIP)
        if problem.status not in ("optimal", "optimal_inaccurate") or x.value is None:
            break
        x_value = np.asarray(x.value)
        best_status = problem.status
        best_x_value = x_value
        g_value = float(np.sum(x_value))
        if g_value <= 1e-9:
            break
        new_q = _f_value(x_value) / g_value
        if abs(new_q - q) < DINKELBACH_TOL:
            q = new_q
            break
        q = new_q

    return _package_result(
        best_status, best_x_value, mu, Sigma, w_current=w_current,
        z_realloc=z_realloc, z_buy=z_buy, z_sell=z_sell, risk_free_rate=params.risk_free_rate,
    )
