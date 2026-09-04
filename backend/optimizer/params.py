"""Optimizer input parameters.

One dataclass covers both modes so a UI (CLI flags, Streamlit widgets,
a future web form) can bind directly to its fields with either typed
numbers or sliders - the fields are the single source of truth for
what's tunable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GroupLimit:
    """One specific group's own cap - e.g. category="sector", group="Growth",
    indices=[0, 3], max_weight=0.4 means Growth-sector holdings (assets 0
    and 3) may not exceed 40%. One entry per capped group, not per category:
    two sectors (or two currencies, two issuers, ...) can each have their
    own independent limit rather than sharing a single category-wide cap."""

    category: str
    group: str
    indices: list[int]
    max_weight: float


@dataclass
class OptimizerParams:
    # --- Objective ---
    # Exactly one of these should be set at a time (the API layer enforces
    # this; the solver just takes whichever one it finds, target_return
    # first):
    #   target_return set      -> minimize variance subject to hitting it.
    #   target_volatility set  -> maximize return subject to a volatility
    #                              cap (w'Sigma w <= target_volatility^2) -
    #                              the mirror image of target_return.
    #   neither set             -> risk_aversion trade-off:
    #                              max(mu'w - risk_aversion * w'Sigma w).
    target_return: float | None = None
    target_volatility: float | None = None
    risk_aversion: float = 1.0
    # Netted out of return before dividing by volatility for Sharpe - see
    # optimizer.core.portfolio_metrics' docstring for why an un-netted
    # return/volatility ratio is misleading near a near-cash corner of the
    # frontier. Defaults to 0.0 (old behavior); the API layer's own default
    # is a live rate (see app.main's /api/risk-free-rate), not this one.
    risk_free_rate: float = 0.0

    # --- Per-asset bounds ---
    w_min: float = 0.0
    w_max: float = 1.0

    # --- Group limits: any number of independent per-group caps at once,
    # across any category the current universe has group data for (sector
    # for the demo universe; country/currency/issuer/instrument_type for an
    # uploaded portfolio). Each group gets its own slider/cap in the UI - a
    # 30% cap on "Growth" sector and a 20% cap on "Energy" sector can coexist
    # independently, rather than one shared cap applied to every group in a
    # category. ---
    group_limits: list[GroupLimit] = field(default_factory=list)

    # --- Numerical stability ---
    regularization: float = 1e-4

    # --- "Existing portfolio" mode only ---
    # Cap on total value traded (sum of |w_new - w_old|). Independent of
    # max_trades - use one, the other, or both together. Also honored under
    # enable_flows (see below), where it caps reallocation + buys + sells
    # combined rather than a single w.
    turnover_max: float | None = None
    # Cap on the *number* of positions allowed to change at all. This is
    # what makes "rebalance by touching only 1-2 names" solvable.
    max_trades: int | None = None

    # --- "Existing portfolio" mode only, opt-in via enable_flows ---
    # The above model pure reallocation: total portfolio value is fixed
    # (sum(w)==1), so every buy is funded by a sell elsewhere in the same
    # portfolio. enable_flows additionally allows *buys funded by new
    # external capital* and *sells that are outright withdrawals* - each
    # with its own independent count and per-trade amount cap, on top of
    # ordinary reallocation (also re-capped separately here as
    # max_reallocations/reallocation_amount_max rather than reusing
    # max_trades, so enabling flows doesn't change what that already means
    # elsewhere). turnover_max, however, IS reused as-is: it caps the total
    # traded value across all three categories combined (reallocation +
    # buys + sells), independent of and on top of each category's own count/
    # amount caps. See modes._solve_with_flows for the mechanics: an asset
    # falls into at most one of {reallocated, bought, sold, untouched}.
    enable_flows: bool = False
    max_reallocations: int | None = None
    reallocation_amount_max: float | None = None
    max_buys: int | None = None
    buy_amount_max: float | None = None
    max_sells: int | None = None
    sell_amount_max: float | None = None
    # Per-asset index -> cap, overriding buy_amount_max/sell_amount_max for
    # that one asset specifically (additive to the aggregate count caps
    # above, not a separate quota - a per-asset-limited asset still counts
    # against max_buys/max_sells).
    per_asset_buy_limits: dict[int, float] = field(default_factory=dict)
    per_asset_sell_limits: dict[int, float] = field(default_factory=dict)

    # --- Diversification (opt-in) ---
    # None (the default) is a plain single-stage solve, unchanged. When set,
    # a second stage re-solves for maximum entropy (spreads weight across
    # more names, pulling small allocations up from an arbitrary corner-
    # solution zero rather than concentrating in however few names the first
    # stage's optimum happened to pick) subject to staying within this
    # fraction of the first stage's own optimal objective value - e.g. 0.02
    # accepts up to 2% worse return/utility in exchange for a less
    # concentrated portfolio. Multiple portfolios often tie (or nearly tie)
    # on the actual objective - this lets the solver break that tie toward
    # the more diversified one instead of an arbitrary vertex of the
    # feasible region, rather than silently picking whichever corner
    # solution the solver reached first.
    #
    # Only wired into the plain continuous paths (solve_new_portfolio, and
    # solve_existing_portfolio without max_trades) - cp.entr introduces an
    # exponential-cone term that SCIP (used for the cardinality/max_trades
    # and enable_flows paths, both mixed-integer) can't combine with integer
    # variables. diversify_tolerance is silently ignored on those paths
    # rather than erroring: asking for both cardinality-limited (touch only
    # K names) and diversified (spread across more names) is a contradiction
    # in terms anyway.
    diversify_tolerance: float | None = None
