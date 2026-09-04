"""FastAPI backend for the portfolio optimizer.

GET /api/universe, and the three solve/analyze POST endpoints, all take the
same optional (portfolio_id, start, end) selector (see schemas.DateWindow)
and must resolve to the *same* cached universe for a given selector -
otherwise, say, the frontier could get computed against a different price
history (or even a different asset universe) than the portfolio it's meant
to contextualize.

Two independent caches back that selector: the curated demo universe, keyed
by date window (fetching from yfinance is slow; the window rarely changes),
and uploaded-portfolio universes, keyed by the portfolio_id handed back from
/api/upload/portfolio at upload time.
"""

from __future__ import annotations

import tempfile
import threading
import time
import uuid
from datetime import date
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ingestion.adapter import aggregate_holdings, build_group_indices
from ingestion.issue_text import translate_issue
from ingestion.osip_workbook import OsipWorkbookError, parse_osip_workbook
from ingestion.risk_limits import RiskLimitsError, parse_risk_limits_workbook
from optimizer import (
    GroupLimit,
    OptimizerParams,
    build_frontier,
    compute_min_group_weight,
    compute_return_bounds,
    compute_volatility_bounds,
    portfolio_metrics,
    solve_existing_portfolio,
    solve_new_portfolio,
)
from optimizer.data import (
    fetch_risk_free_rate,
    load_demo_universe,
    load_random_sp500_universe,
    load_universe_from_tickers,
)

from .schemas import (
    DateWindow,
    FrontierPoint,
    FrontierRequest,
    FrontierResponse,
    GroupLimitRequest,
    MatchedRiskLimit,
    OptimizeRequest,
    OptimizeResponse,
    ReturnBoundsRequest,
    ReturnBoundsResponse,
    RiskLimitsResponse,
    UnmatchedRiskLimit,
    UniverseResponse,
    UploadPortfolioResponse,
)
from .storage import load_all_portfolios, save_portfolio, save_risk_limits

app = FastAPI(title="Portfolio Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DemoCacheKey = tuple[date | None, date | None, str, int | None]
_UNIVERSE_CACHE: dict[_DemoCacheKey, dict] = {}
_UNIVERSE_CACHED_AT: dict[_DemoCacheKey, float] = {}
_UNIVERSE_TTL_SECONDS = 3600.0

# Single-flight lock per cache key (demo or uploaded, namespaced in the key
# itself). yfinance.download() is a slow, blocking call - without this,
# quickly switching the date window back and forth fires a genuinely new
# request each time (the frontend's debounce is only 250ms), and since an
# aborted client-side fetch doesn't cancel the server-side work already in
# progress, several of these blocking calls for the *same* window can pile
# up and serialize behind each other. Holding this lock while populating a
# given key makes every concurrent request for that exact key wait for and
# reuse the one real fetch, instead of each starting its own.
_cache_locks: dict[tuple, threading.Lock] = {}
_cache_locks_guard = threading.Lock()


def _lock_for(key: tuple) -> threading.Lock:
    with _cache_locks_guard:
        lock = _cache_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _cache_locks[key] = lock
        return lock

# Uploaded portfolios, keyed by the uuid returned to the client at upload
# time. Each entry holds the raw holdings (ticker/weight/classification -
# window-independent) plus upload-display fields (report_date/issues) and,
# once uploaded, matched/unmatched risk limits. Mirrored to disk (see
# .storage) so a previously uploaded portfolio survives a backend restart or
# a browser refresh, not just in-process state.
_UPLOADED_PORTFOLIOS: dict[str, dict] = load_all_portfolios()

# The price-derived side (tickers that survive the coverage filter for a
# given window, mu/Sigma, group_indices, renormalized current_weights) DOES
# depend on the date window, unlike the holdings above - cached the same way
# as the demo universe, keyed additionally by portfolio_id, so a date-window
# change on the uploaded tab actually refetches instead of silently
# replaying whatever window was active at upload time.
_UploadedCacheKey = tuple[str, date | None, date | None]
_UPLOADED_UNIVERSE_CACHE: dict[_UploadedCacheKey, dict] = {}
_UPLOADED_UNIVERSE_CACHED_AT: dict[_UploadedCacheKey, float] = {}


def _is_cache_fresh(cached_at: float) -> bool:
    return time.time() - cached_at <= _UNIVERSE_TTL_SECONDS


def get_demo_universe(
    start: date | None, end: date | None, variant: str, seed: int | None
) -> dict:
    key: _DemoCacheKey = (start, end, variant, seed)
    if key in _UNIVERSE_CACHE and _is_cache_fresh(_UNIVERSE_CACHED_AT.get(key, 0.0)):
        return _UNIVERSE_CACHE[key]

    with _lock_for(("demo", *key)):
        # Re-check: another thread may have just finished fetching this
        # exact key while we were waiting for the lock.
        if key not in _UNIVERSE_CACHE or not _is_cache_fresh(_UNIVERSE_CACHED_AT.get(key, 0.0)):
            if variant == "sp500_random":
                universe = load_random_sp500_universe(seed=seed or 0, start=start, end=end)
            else:
                universe = load_demo_universe(start=start, end=end)
            _UNIVERSE_CACHE[key] = universe
            _UNIVERSE_CACHED_AT[key] = time.time()
        return _UNIVERSE_CACHE[key]


def _build_uploaded_universe(holdings: list, start: date | None, end: date | None) -> dict:
    """The price-derived side of an uploaded portfolio for one specific
    date window: which of its holdings survive fetch_prices' coverage
    filter, their mu/Sigma, group membership, and current_weights
    renormalized over just the survivors."""
    tickers = [h.yahoo_ticker for h in holdings]
    labels = [f"{h.security_code} ({h.issuer})" for h in holdings]
    universe = load_universe_from_tickers(tickers, labels, start=start, end=end)

    kept_tickers = set(universe["tickers"])
    kept_holdings = [h for h in holdings if h.yahoo_ticker in kept_tickers]
    skipped = [h.security_code for h in holdings if h.yahoo_ticker not in kept_tickers]

    weight_by_ticker = {h.yahoo_ticker: h.weight for h in kept_holdings}
    total_weight = sum(weight_by_ticker.values())
    current_weights = [
        (weight_by_ticker.get(t, 0.0) / total_weight) if total_weight > 0 else 0.0
        for t in universe["tickers"]
    ]

    universe["group_indices"] = build_group_indices(kept_holdings, universe["tickers"])
    return {"universe": universe, "current_weights": current_weights, "skipped": skipped}


def resolve_uploaded(portfolio_id: str, start: date | None, end: date | None) -> dict:
    entry = _UPLOADED_PORTFOLIOS.get(portfolio_id)
    if entry is None:
        raise HTTPException(404, f"Unknown portfolio_id: {portfolio_id!r}")

    key: _UploadedCacheKey = (portfolio_id, start, end)
    if key in _UPLOADED_UNIVERSE_CACHE and _is_cache_fresh(_UPLOADED_UNIVERSE_CACHED_AT.get(key, 0.0)):
        return _UPLOADED_UNIVERSE_CACHE[key]

    with _lock_for(("uploaded", *key)):
        if key not in _UPLOADED_UNIVERSE_CACHE or not _is_cache_fresh(_UPLOADED_UNIVERSE_CACHED_AT.get(key, 0.0)):
            _UPLOADED_UNIVERSE_CACHE[key] = _build_uploaded_universe(entry["holdings"], start, end)
            _UPLOADED_UNIVERSE_CACHED_AT[key] = time.time()
        return _UPLOADED_UNIVERSE_CACHE[key]


def resolve_universe(selector: DateWindow) -> dict:
    if selector.portfolio_id is not None:
        return resolve_uploaded(selector.portfolio_id, selector.start, selector.end)["universe"]
    return get_demo_universe(selector.start, selector.end, selector.demo_variant, selector.demo_seed)


def resolve_group_indices(selector: DateWindow) -> dict[str, dict[str, list[int]]]:
    """Group membership for the selected universe - "sector" (plus real
    issuer/currency/instrument_type metadata) for the demo universe,
    country/currency/issuer/instrument_type for an uploaded portfolio."""
    return resolve_universe(selector).get("group_indices", {})


def build_group_limits(
    requested: list[GroupLimitRequest], group_indices: dict[str, dict[str, list[int]]]
) -> list[GroupLimit]:
    limits = []
    for item in requested:
        if not item.apply or item.max_weight is None:
            continue
        indices = group_indices.get(item.category, {}).get(item.group)
        if not indices:
            continue
        limits.append(
            GroupLimit(category=item.category, group=item.group, indices=indices, max_weight=item.max_weight)
        )
    return limits


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Cached the same way as the universe caches above (module-level + TTL) -
# ^IRX doesn't move enough intraday to justify a live fetch on every page
# load, and this is called once per session (to seed the risk-free-rate
# field's default), not per-solve.
_RISK_FREE_RATE_CACHE: dict[str, float] = {}
_RISK_FREE_RATE_CACHED_AT = 0.0


@app.get("/api/risk-free-rate")
def get_risk_free_rate() -> dict:
    global _RISK_FREE_RATE_CACHED_AT
    if "value" in _RISK_FREE_RATE_CACHE and _is_cache_fresh(_RISK_FREE_RATE_CACHED_AT):
        return {"risk_free_rate": _RISK_FREE_RATE_CACHE["value"]}
    try:
        rate = fetch_risk_free_rate()
    except Exception as exc:
        raise HTTPException(502, f"Could not fetch risk-free rate: {exc}") from exc
    _RISK_FREE_RATE_CACHE["value"] = rate
    _RISK_FREE_RATE_CACHED_AT = time.time()
    return {"risk_free_rate": rate}


@app.get("/api/universe", response_model=UniverseResponse)
def get_universe_endpoint(
    portfolio_id: str | None = None,
    start: date | None = None,
    end: date | None = None,
    demo_variant: Literal["sectors", "sp500_random"] = "sectors",
    demo_seed: int | None = None,
) -> UniverseResponse:
    universe = resolve_universe(
        DateWindow(portfolio_id=portfolio_id, start=start, end=end, demo_variant=demo_variant, demo_seed=demo_seed)
    )
    volatilities = np.sqrt(np.diag(universe["Sigma"]))
    return UniverseResponse(
        labels=universe["labels"],
        tickers=universe["tickers"],
        mu=universe["mu"].tolist(),
        volatilities=volatilities.tolist(),
        group_indices=universe["group_indices"],
        inception_dates=universe.get("inception_dates", {}),
    )


@app.post("/api/upload/portfolio", response_model=UploadPortfolioResponse)
async def upload_portfolio(
    file: UploadFile = File(...),
    start: date | None = Form(None),
    end: date | None = Form(None),
) -> UploadPortfolioResponse:
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".xls", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        try:
            snapshot = parse_osip_workbook(tmp.name, portfolio_code="uploaded", source_name=file.filename)
        except OsipWorkbookError as exc:
            raise HTTPException(400, str(exc)) from exc

    holdings = aggregate_holdings(snapshot)
    if not holdings:
        raise HTTPException(400, "No positions with a positive carrying value were found in this workbook.")

    portfolio_id = uuid.uuid4().hex
    issues = [f"{issue.severity.value}: {translate_issue(issue)}" for issue in snapshot.issues]
    entry = {
        "holdings": holdings,
        "report_date": snapshot.report_date.isoformat() if snapshot.report_date else None,
        "issues": issues,
    }
    _UPLOADED_PORTFOLIOS[portfolio_id] = entry
    save_portfolio(portfolio_id, entry)

    built = _build_uploaded_universe(holdings, start, end)
    _UPLOADED_UNIVERSE_CACHE[(portfolio_id, start, end)] = built
    _UPLOADED_UNIVERSE_CACHED_AT[(portfolio_id, start, end)] = time.time()
    universe = built["universe"]

    volatilities = np.sqrt(np.diag(universe["Sigma"]))
    return UploadPortfolioResponse(
        portfolio_id=portfolio_id,
        report_date=snapshot.report_date,
        labels=universe["labels"],
        tickers=universe["tickers"],
        mu=universe["mu"].tolist(),
        volatilities=volatilities.tolist(),
        current_weights=built["current_weights"],
        skipped=built["skipped"],
        issues=issues,
        group_indices=universe["group_indices"],
        inception_dates=universe.get("inception_dates", {}),
    )


@app.get("/api/upload/portfolio/{portfolio_id}", response_model=UploadPortfolioResponse)
def get_uploaded_portfolio(
    portfolio_id: str, start: date | None = None, end: date | None = None
) -> UploadPortfolioResponse:
    """Rehydrates a previously uploaded portfolio - lets the frontend
    restore an active upload after a page refresh or a backend restart
    without asking the user to re-upload the same workbook, as long as its
    entry still exists under .data/optimizer/uploaded_portfolios/. Accepts
    the same optional date window as everything else, defaulting (like
    load_universe_from_tickers) to the standard lookback ending today."""
    entry = _UPLOADED_PORTFOLIOS.get(portfolio_id)
    if entry is None:
        raise HTTPException(404, f"Unknown portfolio_id: {portfolio_id!r}")
    built = resolve_uploaded(portfolio_id, start, end)
    universe = built["universe"]
    volatilities = np.sqrt(np.diag(universe["Sigma"]))
    return UploadPortfolioResponse(
        portfolio_id=portfolio_id,
        report_date=entry.get("report_date"),
        labels=universe["labels"],
        tickers=universe["tickers"],
        mu=universe["mu"].tolist(),
        volatilities=volatilities.tolist(),
        current_weights=built["current_weights"],
        skipped=built["skipped"],
        issues=entry.get("issues", []),
        group_indices=universe["group_indices"],
        inception_dates=universe.get("inception_dates", {}),
    )


@app.post("/api/upload/risk-limits", response_model=RiskLimitsResponse)
async def upload_risk_limits(
    portfolio_id: str = Form(...),
    file: UploadFile = File(...),
) -> RiskLimitsResponse:
    """Parses a "Отчет о соблюдении лимитов инвестирования" workbook's real
    regulatory caps and matches them against `portfolio_id`'s own group data
    - see ingestion/risk_limits.py for why several groups don't match by
    label even for the same real holding. Matched limits are handed back for
    the client to apply (as ordinary group_limits) if it chooses to; nothing
    here is stored or applied server-side automatically.
    """
    if portfolio_id not in _UPLOADED_PORTFOLIOS:
        raise HTTPException(404, f"Unknown portfolio_id: {portfolio_id!r}")
    # Matched against the default-window universe - group membership
    # (country/currency/issuer/instrument_type) is an intrinsic property of
    # each holding, not something that changes with the price window, aside
    # from the rare case where a ticker only survives the coverage filter in
    # some other window.
    group_indices = resolve_uploaded(portfolio_id, None, None)["universe"]["group_indices"]

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".xls", delete=True) as tmp:
        tmp.write(content)
        tmp.flush()
        try:
            parsed, unmapped_categories = parse_risk_limits_workbook(tmp.name)
        except RiskLimitsError as exc:
            raise HTTPException(400, str(exc)) from exc

    matched = []
    unmatched = []
    for limit in parsed:
        if limit.group in group_indices.get(limit.category, {}):
            matched.append(MatchedRiskLimit(category=limit.category, group=limit.group, max_weight=limit.max_weight))
        else:
            unmatched.append(
                UnmatchedRiskLimit(category=limit.category, group=limit.group, max_weight=limit.max_weight)
            )

    risk_limits_payload = {
        "matched": [m.model_dump() for m in matched],
        "unmatched": [u.model_dump() for u in unmatched],
        "unmapped_categories": unmapped_categories,
    }
    _UPLOADED_PORTFOLIOS[portfolio_id]["risk_limits"] = risk_limits_payload
    save_risk_limits(portfolio_id, risk_limits_payload)

    return RiskLimitsResponse(
        portfolio_id=portfolio_id,
        matched=matched,
        unmatched=unmatched,
        unmapped_categories=unmapped_categories,
    )


@app.get("/api/upload/risk-limits/{portfolio_id}", response_model=RiskLimitsResponse)
def get_uploaded_risk_limits(portfolio_id: str) -> RiskLimitsResponse:
    """Rehydrates previously matched risk limits for a portfolio, if any
    were ever uploaded for it - 404 (not an error state to the frontend)
    means simply "nothing uploaded yet", same as a fresh portfolio."""
    entry = _UPLOADED_PORTFOLIOS.get(portfolio_id)
    if entry is None:
        raise HTTPException(404, f"Unknown portfolio_id: {portfolio_id!r}")
    risk_limits = entry.get("risk_limits")
    if risk_limits is None:
        raise HTTPException(404, "No risk-limits workbook has been uploaded for this portfolio yet.")
    return RiskLimitsResponse(portfolio_id=portfolio_id, **risk_limits)


@app.post("/api/return-bounds", response_model=ReturnBoundsResponse)
def get_return_bounds(request: ReturnBoundsRequest) -> ReturnBoundsResponse:
    universe = resolve_universe(request)
    mu, Sigma = universe["mu"], universe["Sigma"]

    group_indices = resolve_group_indices(request)
    group_limits = build_group_limits(request.group_limits, group_indices)

    params = OptimizerParams(
        w_min=request.w_min,
        w_max=request.w_max,
        group_limits=group_limits,
    )
    min_return, max_return = compute_return_bounds(mu, params)
    min_volatility, max_volatility = compute_volatility_bounds(mu, Sigma, params)
    min_group_caps = {
        category: {
            group: compute_min_group_weight(len(mu), request.w_min, request.w_max, indices)
            for group, indices in groups.items()
        }
        for category, groups in group_indices.items()
    }
    return ReturnBoundsResponse(
        min_return=min_return,
        max_return=max_return,
        min_volatility=min_volatility,
        max_volatility=max_volatility,
        min_group_caps=min_group_caps,
    )


@app.post("/api/frontier", response_model=FrontierResponse)
def get_frontier(request: FrontierRequest) -> FrontierResponse:
    universe = resolve_universe(request)
    mu, Sigma = universe["mu"], universe["Sigma"]

    group_limits = build_group_limits(request.group_limits, resolve_group_indices(request))

    params = OptimizerParams(
        w_min=request.w_min,
        w_max=request.w_max,
        group_limits=group_limits,
    )
    points = build_frontier(mu, Sigma, params, n_points=request.n_points)
    return FrontierResponse(
        points=[FrontierPoint(return_=r, volatility=v) for r, v in points]
    )


@app.post("/api/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    if request.target_return is not None and request.target_volatility is not None:
        raise HTTPException(400, "Set at most one of target_return, target_volatility")

    universe = resolve_universe(request)
    mu, Sigma = universe["mu"], universe["Sigma"]
    n = len(mu)

    group_limits = build_group_limits(request.group_limits, resolve_group_indices(request))

    ticker_index = {t: i for i, t in enumerate(universe["tickers"])}
    per_asset_buy_limits = {
        ticker_index[t]: cap for t, cap in request.per_asset_buy_limits.items() if t in ticker_index
    }
    per_asset_sell_limits = {
        ticker_index[t]: cap for t, cap in request.per_asset_sell_limits.items() if t in ticker_index
    }

    params = OptimizerParams(
        target_return=request.target_return,
        target_volatility=request.target_volatility,
        risk_aversion=request.risk_aversion,
        risk_free_rate=request.risk_free_rate,
        w_min=request.w_min,
        w_max=request.w_max,
        group_limits=group_limits,
        turnover_max=request.turnover_max,
        max_trades=request.max_trades,
        enable_flows=request.enable_flows,
        max_reallocations=request.max_reallocations,
        reallocation_amount_max=request.reallocation_amount_max,
        max_buys=request.max_buys,
        buy_amount_max=request.buy_amount_max,
        max_sells=request.max_sells,
        sell_amount_max=request.sell_amount_max,
        per_asset_buy_limits=per_asset_buy_limits,
        per_asset_sell_limits=per_asset_sell_limits,
        diversify_tolerance=request.diversify_tolerance,
    )

    initial_metrics = None
    if request.mode == "new":
        result = solve_new_portfolio(mu, Sigma, params)
    else:
        if request.current_weights is None:
            raise HTTPException(400, "current_weights is required for 'existing' mode")
        if len(request.current_weights) != n:
            raise HTTPException(400, f"current_weights must have exactly {n} entries")
        w_current = np.array(request.current_weights, dtype=float)
        total = w_current.sum()
        if total > 0:
            w_current = w_current / total
        # The starting portfolio's own return/volatility - a fixed reference
        # point, independent of whether the solve below finds anything -
        # so the frontier chart can plot "where you are" alongside "where the
        # optimizer would put you."
        initial_metrics = portfolio_metrics(w_current, mu, Sigma)
        result = solve_existing_portfolio(mu, Sigma, w_current, params)

    return OptimizeResponse(
        status=result.status,
        ok=result.ok,
        labels=universe["labels"],
        weights=result.weights.tolist() if result.weights is not None else None,
        trades=result.trades.tolist() if result.trades is not None else None,
        n_trades=result.n_trades,
        n_reallocations=result.n_reallocations,
        n_buys=result.n_buys,
        n_sells=result.n_sells,
        trade_kinds=result.trade_kinds,
        expected_return=result.expected_return,
        volatility=result.volatility,
        sharpe=result.sharpe,
        n_positions=result.n_positions,
        initial_return=initial_metrics["return"] if initial_metrics else None,
        initial_volatility=initial_metrics["volatility"] if initial_metrics else None,
        # existing-portfolio mode only: the starting book is always exactly
        # 1.0 (100%) by construction - w_current is renormalized above
        # before it ever reaches the solver. new_total_value is where that
        # actually goes: 1.0 for ordinary reallocation (sum(w)==1 is
        # enforced), but enable_flows lets it end up above 1.0 (net buys -
        # new capital came in) or below 1.0 (net sells - capital went out).
        initial_total_value=1.0 if initial_metrics else None,
        new_total_value=float(result.weights.sum()) if result.weights is not None and initial_metrics else None,
        effective_n=result.effective_n,
        pre_diversify_return=result.pre_diversify_return,
        pre_diversify_volatility=result.pre_diversify_volatility,
        pre_diversify_effective_n=result.pre_diversify_effective_n,
    )
