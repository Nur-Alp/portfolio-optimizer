"""Request/response models for the optimizer API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UniverseResponse(BaseModel):
    labels: list[str]
    tickers: list[str]
    mu: list[float]
    volatilities: list[float]
    # category -> {group label -> indices into tickers/labels above}. For
    # the demo universe: "sector" (curated, can overlap) plus
    # issuer/currency/instrument_type from real yfinance metadata. For an
    # uploaded portfolio: country/currency/issuer/instrument_type derived
    # from the workbook's own holdings data.
    group_indices: dict[str, dict[str, list[int]]] = Field(default_factory=dict)
    # ticker -> ISO date of its first real price observation within the
    # fetched window - one entry per surviving ticker, regardless of
    # whether it's later than the requested start. Lets the client both
    # show which holding was listed most recently and warn "your lookback
    # reaches further back than this holding's own history" for whichever
    # specific tickers that's actually true of.
    inception_dates: dict[str, str] = Field(default_factory=dict)


class GroupLimitRequest(BaseModel):
    """One specific group's own cap as sent by the client - which category
    and group, whether it's turned on, and the cap to apply if so. One entry
    per capped group (not per category): "Growth" sector and "Energy" sector
    can each carry their own independent limit."""

    category: str
    group: str
    apply: bool = False
    max_weight: float | None = None


class DateWindow(BaseModel):
    """Which universe to use, and over what price history - shared by every
    endpoint that touches the universe, so a solve/frontier/bounds call
    always uses the same data the user actually selected.

    `portfolio_id` selects an uploaded portfolio's universe (see
    /api/upload/portfolio) instead of the curated demo one; None means the
    demo universe. `start=None` means "3 years back from `end`"; `end=None`
    means "latest available close" - both resolve inside load_demo_universe/
    load_universe_from_tickers, and apply either way (an uploaded portfolio
    still needs a price history window for its own tickers).
    """

    portfolio_id: str | None = None
    start: date | None = None
    end: date | None = None
    # Demo universe only (ignored once portfolio_id is set): "sectors" is
    # the curated S&P sector-ETF set; "sp500_random" is n randomly sampled
    # individual S&P 500 names. demo_seed makes that sample reproducible -
    # required (and load_random_sp500_universe otherwise cheerfully draws a
    # *different* 20 names) whenever demo_variant is "sp500_random", so a
    # slider tweak's re-fetch doesn't silently swap out the universe being
    # optimized underneath the user.
    demo_variant: Literal["sectors", "sp500_random"] = "sectors"
    demo_seed: int | None = None


class OptimizeRequest(DateWindow):
    mode: Literal["new", "existing"]

    # Objective - see OptimizerParams for how target_return/target_volatility/
    # risk_aversion interact (exactly one of the first two, or neither).
    target_return: float | None = None
    target_volatility: float | None = None
    risk_aversion: float = 1.0
    # Netted out of return for Sharpe - see OptimizerParams.risk_free_rate
    # and optimizer.core.portfolio_metrics for why an un-netted return/
    # volatility ratio is misleading near a near-cash portfolio. The client
    # is expected to seed this from GET /api/risk-free-rate (a live ^IRX
    # quote) rather than rely on this 0.0 default.
    risk_free_rate: float = 0.0

    # Position limits
    w_min: float = 0.0
    w_max: float = 1.0

    # Per-group caps - any subset of the current universe's group_indices,
    # each with its own independent max_weight.
    group_limits: list[GroupLimitRequest] = Field(default_factory=list)

    # Existing-portfolio mode only
    current_weights: list[float] | None = None
    max_trades: int | None = None
    turnover_max: float | None = None

    # Existing-portfolio mode only, opt-in: buys funded by new capital and
    # sells that are outright withdrawals, each independently capped, on
    # top of ordinary reallocation (re-capped here separately from
    # max_trades/turnover_max above - see OptimizerParams.enable_flows).
    enable_flows: bool = False
    max_reallocations: int | None = None
    reallocation_amount_max: float | None = None
    max_buys: int | None = None
    buy_amount_max: float | None = None
    max_sells: int | None = None
    sell_amount_max: float | None = None
    # Ticker -> cap, overriding buy_amount_max/sell_amount_max for that one
    # asset - keyed by ticker (not index) since that's what the client
    # actually knows; main.py resolves to the universe's own index order.
    per_asset_buy_limits: dict[str, float] = Field(default_factory=dict)
    per_asset_sell_limits: dict[str, float] = Field(default_factory=dict)

    # Opt-in diversification, e.g. 0.02 = "accept up to 2% worse
    # return/utility for a less concentrated portfolio" - see
    # OptimizerParams.diversify_tolerance. None (the default) is a plain,
    # unchanged single-stage solve.
    diversify_tolerance: float | None = None


class ReturnBoundsRequest(DateWindow):
    w_min: float = 0.0
    w_max: float = 1.0
    group_limits: list[GroupLimitRequest] = Field(default_factory=list)


class ReturnBoundsResponse(BaseModel):
    min_return: float
    max_return: float
    min_volatility: float
    max_volatility: float
    # category -> group -> smallest cap for which that group alone can still
    # be brought down to, given the position limits - one entry for every
    # group in every category the current universe has group data for.
    min_group_caps: dict[str, dict[str, float]] = Field(default_factory=dict)


class FrontierRequest(DateWindow):
    w_min: float = 0.0
    w_max: float = 1.0
    group_limits: list[GroupLimitRequest] = Field(default_factory=list)
    n_points: int = 30


class FrontierPoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # "return" is a Python keyword, but the JSON key should still be "return".
    return_: float = Field(alias="return")
    volatility: float


class FrontierResponse(BaseModel):
    points: list[FrontierPoint]


class UploadPortfolioResponse(BaseModel):
    portfolio_id: str
    report_date: date | None
    labels: list[str]
    tickers: list[str]
    mu: list[float]
    volatilities: list[float]
    current_weights: list[float]
    # Holdings that didn't make it into the universe: an ISIN with no
    # resolvable Yahoo ticker, or a resolved ticker with too few real price
    # observations to say anything meaningful at all (a typo'd symbol, a
    # delisting) - not merely a shorter history than the window, which is
    # kept and used rather than dropped (see optimizer.data.fetch_prices).
    # current_weights is renormalized over the ones that remain.
    skipped: list[str]
    # Human-readable data-quality notes the parser surfaced (e.g. missing
    # identifiers, incomplete ratings) - informational, nothing here blocks
    # using the upload.
    issues: list[str]
    # Group membership: category -> {group label -> indices into
    # tickers/labels above}. Only categories with at least one group are
    # included (e.g. a workbook with a single issuer everywhere would still
    # report it, but a wholly-missing field would be omitted).
    group_indices: dict[str, dict[str, list[int]]] = Field(default_factory=dict)
    # ticker -> ISO date of its first real price observation within the
    # fetched window - see UniverseResponse.inception_dates.
    inception_dates: dict[str, str] = Field(default_factory=dict)


class MatchedRiskLimit(BaseModel):
    category: str
    group: str
    max_weight: float


class UnmatchedRiskLimit(BaseModel):
    """A parsed regulatory limit whose group label doesn't correspond to any
    group this portfolio's own holdings data has - shown for transparency,
    never applied (see ingestion/risk_limits.py's module docstring for why
    these mismatches happen)."""

    category: str
    group: str
    max_weight: float


class RiskLimitsResponse(BaseModel):
    portfolio_id: str
    matched: list[MatchedRiskLimit]
    unmatched: list[UnmatchedRiskLimit]
    # Whole categories in the workbook with no corresponding group data on
    # any uploaded portfolio at all (e.g. GICS sector, open currency
    # position) - informational only.
    unmapped_categories: list[str]


class OptimizeResponse(BaseModel):
    status: str
    ok: bool
    labels: list[str]
    weights: list[float] | None = None
    trades: list[float] | None = None
    n_trades: int | None = None
    # Only populated when enable_flows was set - how many of n_trades' moves
    # were reallocations vs. new-capital buys vs. outright-withdrawal sells.
    n_reallocations: int | None = None
    n_buys: int | None = None
    n_sells: int | None = None
    # Per-asset classification, same order as labels/weights/trades:
    # "reallocation"/"buy"/"sell" for a real nonzero move of that kind, null
    # for an untouched asset. The explicit, per-position version of the
    # three counts above.
    trade_kinds: list[str | None] | None = None
    expected_return: float | None = None
    volatility: float | None = None
    sharpe: float | None = None
    n_positions: int | None = None
    # The starting portfolio's own return/volatility (existing-portfolio mode
    # only) - a fixed reference point independent of the solve outcome, so
    # the frontier chart can show "where you are" alongside "where the
    # optimizer would move you."
    initial_return: float | None = None
    initial_volatility: float | None = None
    # existing-portfolio mode only: initial_total_value is always exactly
    # 1.0 (100%) - the starting book, renormalized. new_total_value is where
    # that ends up: still 1.0 for ordinary reallocation, but enable_flows
    # lets it land above 1.0 (net buys added capital) or below (net sells
    # withdrew it) - see optimizer.core.portfolio_metrics' docstring for why
    # return/volatility are normalized by this same total.
    initial_total_value: float | None = None
    new_total_value: float | None = None
    # 1/HHI of the returned weights - see optimizer.core.effective_n. Always
    # populated on a successful solve.
    effective_n: float | None = None
    # Only populated when diversify_tolerance was set AND actually applied
    # (see OptimizerParams.diversify_tolerance's solver-support note) - the
    # pre-diversification portfolio's own return/volatility/effective_n, so
    # the UI can show "here's what you gave up" next to the returned,
    # already-diversified weights/expected_return/volatility/effective_n.
    pre_diversify_return: float | None = None
    pre_diversify_volatility: float | None = None
    pre_diversify_effective_n: float | None = None
