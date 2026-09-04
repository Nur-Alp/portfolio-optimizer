"""Demo asset universe and parameter estimation.

This is the same S&P sector-ETF universe used in portfolio-optimizer.ipynb,
factored out so both the notebook and the app can share it. It stands in
for a real holdings feed - swapping in actual portfolio data (e.g. TABYS
or the own book from the operations dashboard) means writing a loader
that produces the same (tickers, mu, Sigma, sector_indices) shape and,
for "existing portfolio" mode, a w_current vector. That translation is
a separate step - not built here, since it touches another app's
sensitive source data.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_LOOKBACK_DAYS = 3 * 365

ASSETS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Consumer Disc",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
}

SECTOR_GROUPS = {
    "Growth": ["XLK", "XLY"],
    "Defensive": ["XLP", "XLU", "XLV"],
    "Cyclical": ["XLF", "XLI", "XLB", "XLE"],
    "Real Assets": ["XLRE", "XLE", "XLB"],
}


def fetch_risk_free_rate() -> float:
    """Current annualized risk-free rate proxy: ^IRX, the CBOE 13-Week
    Treasury Bill yield index - a standard, widely-used risk-free proxy that
    yfinance already quotes directly as an annualized yield (unlike a price
    series, there's no return conversion needed - the ticker's own value
    IS the rate). Quoted in percent (e.g. 5.25 for 5.25%), so divided by
    100 to match this app's fraction convention everywhere else (mu,
    target_return, risk_free_rate itself once set, etc.).
    """
    history = yf.Ticker("^IRX").history(period="5d")
    if history.empty:
        raise RuntimeError("^IRX returned no data")
    return float(history["Close"].iloc[-1]) / 100


def fetch_prices(
    tickers: list[str], start: date, end: date | None = None, min_observations: int = 20
) -> pd.DataFrame:
    """Close prices over [start, end], inclusive of end. `end=None` means
    "up to the latest available close" (yfinance simply has nothing newer,
    so this just works rather than needing to know what today's date is).

    Only genuinely bad tickers get dropped here - a typo'd symbol, a
    delisting, or a real listing with too few observations to say anything
    meaningful (min_observations, an absolute trading-day count, not a
    fraction of the requested window). A real, currently-held position that
    simply IPO'd partway through a long lookback window is *not* one of
    those: dropping it there would silently throw the holding out of the
    return/risk estimate and the renormalized current weights, which is a
    much worse outcome than just using however much real history it has.
    NaNs from that short history are left in place rather than forced into
    a shared dropna() - estimate_parameters handles them per-column/pair
    (each asset's own mean, each pair's own overlap), not as a reason to cut
    every other ticker's history down to the newest asset's inception.
    """
    end_exclusive = (end or datetime.now().date()) + timedelta(days=1)
    data = yf.download(tickers, start=start, end=end_exclusive, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        elif "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])

    observations = data.notna().sum()
    good_columns = observations[observations >= min_observations].index
    return data[good_columns]


def estimate_parameters(prices: pd.DataFrame, annualize: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """mu uses each column's own mean (pandas skips NaN per-column by
    default - a recently-listed asset's mu is just its own, shorter, real
    history, not diluted or excluded because older assets have more data).

    Sigma uses pairwise-complete covariance (pandas' own .cov() default):
    each entry is computed from whatever dates *that specific pair* of
    assets both have prices for, rather than requiring a single window where
    every asset in the whole portfolio has data simultaneously. min_periods
    guards against a wildly noisy estimate from too little overlap between
    a specific pair - which shows up as NaN for that pair rather than a
    number, and those assets are excluded (returned alongside the kept
    tickers) rather than let a NaN reach the solver's quadratic form. This
    is rare in practice: two real holdings' histories only fail to overlap
    at all near the shared window's own end, which would mean one of them
    has no recent price data at all - a data problem, not a short-history one.
    """
    # A zero/negative print (bad tick, vendor data glitch) would otherwise
    # feed -inf/NaN into np.log and propagate into mu/Sigma; treat it as a
    # missing observation instead, same as any other data gap.
    prices = prices.where(prices > 0)
    returns = np.log(prices / prices.shift(1))
    factor = 252 if annualize else 1
    mu = returns.mean() * factor
    Sigma = returns.cov(min_periods=min(20, len(returns))) * factor

    good = ~Sigma.isna().any(axis=0)
    mu = mu[good]
    Sigma = Sigma.loc[good, good]
    return mu.values, Sigma.values, list(mu.index)


def sector_indices_for(tickers: list[str]) -> dict[str, list[int]]:
    indices = {}
    for sector, members in SECTOR_GROUPS.items():
        idx = [tickers.index(t) for t in members if t in tickers]
        if idx:
            indices[sector] = idx
    return indices


# yfinance's per-ticker .info call is slow (one HTTP round trip each) and
# this metadata never changes for a given ticker, so it's cached in-process
# indefinitely - unlike prices, there's no "latest close" to go stale.
_ASSET_METADATA_CACHE: dict[str, dict[str, str]] = {}


def fetch_asset_metadata(tickers: list[str]) -> dict[str, dict[str, str]]:
    """Real per-ticker classification from yfinance - issuer (fundFamily),
    currency, and instrument type (legalType, falling back to quoteType).
    yfinance doesn't expose a fund's domicile/country for ETFs (info.country
    comes back None for all of them), so that category is deliberately left
    out rather than reported as if it were real data. A field missing for a
    specific ticker becomes "(unspecified)" rather than dropping the ticker.
    """
    for ticker in tickers:
        if ticker in _ASSET_METADATA_CACHE:
            continue
        info = yf.Ticker(ticker).info
        _ASSET_METADATA_CACHE[ticker] = {
            "issuer": info.get("fundFamily") or "(unspecified)",
            "currency": info.get("currency") or "(unspecified)",
            "instrument_type": info.get("legalType") or info.get("quoteType") or "(unspecified)",
        }
    return {t: _ASSET_METADATA_CACHE[t] for t in tickers}


def demo_group_indices(tickers: list[str]) -> dict[str, dict[str, list[int]]]:
    """Group membership for every advanced-limit category available on the
    demo universe: "sector" from the curated SECTOR_GROUPS (can overlap -
    see sector_indices_for), plus issuer/currency/instrument_type from real
    yfinance metadata (no faked/hardcoded values - see fetch_asset_metadata).
    Country is omitted entirely since yfinance has no real data for it here.
    """
    group_indices: dict[str, dict[str, list[int]]] = {"sector": sector_indices_for(tickers)}

    metadata = fetch_asset_metadata(tickers)
    for category in ("issuer", "currency", "instrument_type"):
        groups: dict[str, list[int]] = {}
        for i, ticker in enumerate(tickers):
            label = metadata[ticker][category]
            groups.setdefault(label, []).append(i)
        group_indices[category] = groups

    return group_indices


# A curated sample of real S&P 500 constituents, spanning all 11 GICS
# sectors - NOT the complete, current 500-member list (constituents change
# over time and there's no live index-membership feed wired in here), but
# every ticker on it is a real S&P 500 company. Large enough to draw a
# genuinely varied random 20-name demo portfolio from.
SP500_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "ORCL", "CRM",
    "ADBE", "AMD", "CSCO", "ACN", "IBM", "INTU", "TXN", "QCOM", "NOW", "AMAT",
    "PANW", "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "MU", "APH", "ANET", "FTNT",
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SPGI", "BLK",
    "SCHW", "C", "PGR", "CB", "MMC", "ICE", "PYPL", "AON", "USB", "PNC",
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "AMGN",
    "ISRG", "VRTX", "SYK", "BSX", "GILD", "MDT", "CI", "ELV", "REGN", "ZTS",
    "BRK-B", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG", "ORLY",
    "MAR", "GM", "F", "TGT", "ROST", "YUM", "LULU", "DHI", "LEN", "AZO",
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "HSY", "STZ", "KR", "SYY", "ADM", "KDP", "CAG", "MKC", "CLX",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "WMB", "KMI",
    "CAT", "GE", "RTX", "HON", "UNP", "BA", "DE", "LMT", "UPS", "ADP",
    "GD", "NOC", "ETN", "ITW", "EMR", "CSX", "NSC", "FDX", "WM", "PH",
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "DD", "NUE", "DOW", "PPG",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "WEC",
    "PLD", "AMT", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB", "EQR",
    "CMCSA", "DIS", "NFLX", "TMUS", "VZ", "T", "CHTR", "EA", "TTWO", "WBD",
]


def _pick_random_tickers(n: int, seed: int) -> list[str]:
    pool = SP500_SAMPLE
    return random.Random(seed).sample(pool, min(n, len(pool)))


# Same in-process, indefinite cache pattern as _ASSET_METADATA_CACHE, but for
# individual equities: sector/industry (real GICS-style classification,
# available on yfinance's equity .info the way it isn't for ETFs) plus
# currency. "issuer" doesn't apply to a company's own stock the way it does
# to a fund, so it's not a category here.
_EQUITY_METADATA_CACHE: dict[str, dict[str, str]] = {}


def fetch_equity_metadata(tickers: list[str]) -> dict[str, dict[str, str]]:
    for ticker in tickers:
        if ticker in _EQUITY_METADATA_CACHE:
            continue
        info = yf.Ticker(ticker).info
        _EQUITY_METADATA_CACHE[ticker] = {
            "sector": info.get("sector") or "(unspecified)",
            "industry": info.get("industry") or "(unspecified)",
            "currency": info.get("currency") or "(unspecified)",
            "name": info.get("shortName") or ticker,
        }
    return {t: _EQUITY_METADATA_CACHE[t] for t in tickers}


def random_sp500_group_indices(tickers: list[str]) -> dict[str, dict[str, list[int]]]:
    metadata = fetch_equity_metadata(tickers)
    group_indices: dict[str, dict[str, list[int]]] = {}
    for category in ("sector", "industry", "currency"):
        groups: dict[str, list[int]] = {}
        for i, ticker in enumerate(tickers):
            label = metadata[ticker][category]
            groups.setdefault(label, []).append(i)
        group_indices[category] = groups
    return group_indices


def load_universe_from_tickers(
    tickers: list[str], labels: list[str], start: date | None = None, end: date | None = None
) -> dict:
    """Build a universe from an arbitrary ticker list (e.g. real holdings
    from an uploaded portfolio), not just the curated demo set. Tickers that
    fail to fetch entirely (see fetch_prices) or that never overlap enough
    with the others to compute a reliable covariance (see
    estimate_parameters) are silently excluded from mu/Sigma/tickers/labels
    - a shorter-but-real history is kept and used, not dropped just for
    being shorter than the requested window. Callers that need to know what
    got dropped should diff their input list against the returned "tickers".
    """
    if start is None:
        start = (end or datetime.now().date()) - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    prices = fetch_prices(tickers, start=start, end=end)
    mu, Sigma, kept = estimate_parameters(prices)
    label_by_ticker = dict(zip(tickers, labels))

    # The first date each surviving ticker actually has a real price for -
    # only reported when that's *meaningfully later* than `start` (more than
    # a week, comfortably past any run of market holidays/weekends around
    # the requested start date itself). fetch_prices was asked for data
    # going back to `start`, but `start` need not be a trading day - if it
    # falls on, say, a Labor Day Monday, literally every ticker's first row
    # lands a day or two later, which would make a plain "> start" check
    # misreport every single holding as newly listed simultaneously (this
    # happened - a request for "3y 0d" ago landed on 2023-09-04, and SPY,
    # QQQ, XLF etc. all "inception"-ed on 2023-09-05 the next trading day).
    # A gap of more than a week is not explainable by market closures and
    # means the ticker's real listing date fell inside the window - a
    # genuine inception boundary actually observed, not a fabricated one.
    # Lets a caller warn when the chosen lookback reaches further back than
    # a real holding's own history, and say exactly which holdings that
    # affects, without having to drop them from the analysis (see
    # estimate_parameters).
    inception_dates = {
        t: first_valid.date().isoformat()
        for t in kept
        if (first_valid := prices[t].first_valid_index()) is not None
        and (first_valid.date() - start).days > 7
    }

    return {
        "tickers": kept,
        "labels": [label_by_ticker.get(t, t) for t in kept],
        "mu": mu,
        "Sigma": Sigma,
        "group_indices": {},
        "inception_dates": inception_dates,
    }


def load_demo_universe(start: date | None = None, end: date | None = None) -> dict:
    tickers = list(ASSETS.keys())
    labels = [ASSETS[t] for t in tickers]
    universe = load_universe_from_tickers(tickers, labels, start=start, end=end)
    universe["group_indices"] = demo_group_indices(universe["tickers"])
    return universe


def load_random_sp500_universe(
    seed: int, n: int = 20, start: date | None = None, end: date | None = None
) -> dict:
    """The same demo-universe shape as load_demo_universe, but n randomly
    sampled individual S&P 500 names instead of the curated sector ETFs -
    `seed` makes the sample reproducible (same seed -> same 20 tickers), so
    repeated calls across a UI session (each slider tweak re-fetches the
    universe) don't silently swap out which names are even being optimized.
    """
    tickers = _pick_random_tickers(n, seed)
    metadata = fetch_equity_metadata(tickers)
    labels = [metadata[t]["name"] for t in tickers]
    universe = load_universe_from_tickers(tickers, labels, start=start, end=end)
    universe["group_indices"] = random_sp500_group_indices(universe["tickers"])
    return universe
