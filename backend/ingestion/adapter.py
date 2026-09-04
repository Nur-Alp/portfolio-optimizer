"""Turns a parsed OSIP PortfolioSnapshot into an optimizer-ready universe.

The workbook's own weight/value columns are blank cached formulas (see
osip_workbook.py's BROKEN_CALCULATED_FIELDS) - real weights come from
aggregating PositionLotSnapshot.derived_carrying_value_kzt by ISIN instead,
since a single instrument is often split across several purchase lots.
"""

from __future__ import annotations

from dataclasses import dataclass

from .domain import PortfolioSnapshot

# Bloomberg exchange-code suffix -> Yahoo Finance suffix, for the handful of
# exchanges this fund actually holds on. Best-effort: an unrecognized suffix
# is dropped (tried as the plain ticker) rather than guessed at, since a
# wrong guess would silently fetch the wrong instrument's price history.
# The workbook's own instrument-type column ("Тип ценной бумаги") is
# free-text Russian - translated here (by exact match, case-insensitive) so
# it never surfaces untranslated in the UI. An unrecognized value falls back
# to itself rather than guessing at a translation.
_INSTRUMENT_TYPE_LABELS = {
    "акции": "Equities",
    "гцб": "Government bonds",
    "корпоративные облигации": "Corporate bonds",
    "депозит": "Deposit",
    "авторепо": "Repo",
}


def _translate_instrument_type(raw: str) -> str:
    return _INSTRUMENT_TYPE_LABELS.get(raw.strip().casefold(), raw)


_BLOOMBERG_TO_YAHOO_SUFFIX = {
    "US": "",
    "GY": ".DE",
    "GR": ".DE",
    "LN": ".L",
    "FP": ".PA",
    "IM": ".MI",
    "SW": ".SW",
    "NA": ".AS",
}


def bloomberg_to_yahoo(security_code: str) -> str:
    """"AVUV US" -> "AVUV", "ZPRX GY" -> "ZPRX.DE", "SPTL US Equity" -> "SPTL"."""
    parts = security_code.split()
    ticker = parts[0]
    suffix = parts[1].upper() if len(parts) > 1 else "US"
    return ticker + _BLOOMBERG_TO_YAHOO_SUFFIX.get(suffix, "")


@dataclass(frozen=True)
class UploadedHolding:
    isin: str
    yahoo_ticker: str
    security_code: str
    issuer: str
    currency: str
    country: str
    instrument_type: str
    weight: float


def aggregate_holdings(snapshot: PortfolioSnapshot) -> list[UploadedHolding]:
    """One row per distinct ISIN, weight = its share of total carrying value."""
    by_isin: dict[str, dict] = {}
    for position in snapshot.positions:
        value = float(position.derived_carrying_value_kzt or 0)
        entry = by_isin.setdefault(
            position.isin,
            {
                "value": 0.0,
                "security_code": position.security_code,
                "issuer": position.issuer,
                "currency": position.instrument_currency,
                "instrument_type": _translate_instrument_type(position.raw_security_type or ""),
            },
        )
        entry["value"] += value

    total = sum(entry["value"] for entry in by_isin.values())
    if total <= 0:
        return []

    return [
        UploadedHolding(
            isin=isin,
            yahoo_ticker=bloomberg_to_yahoo(entry["security_code"]),
            security_code=entry["security_code"],
            issuer=entry["issuer"],
            currency=entry["currency"],
            country=isin[:2] if len(isin) >= 2 else "",
            instrument_type=entry["instrument_type"],
            weight=entry["value"] / total,
        )
        for isin, entry in by_isin.items()
    ]


# Advanced-limit categories derivable straight from the upload, no external
# reference data needed - country from the ISIN prefix (ISO 3166 alpha-2,
# already encoded in the identifier itself), the rest are columns on the
# position rows themselves.
_GROUP_CATEGORIES: dict[str, str] = {
    "country": "country",
    "currency": "currency",
    "issuer": "issuer",
    "instrument_type": "instrument_type",
}


def build_group_indices(
    holdings: list[UploadedHolding], tickers: list[str]
) -> dict[str, dict[str, list[int]]]:
    """{category: {group_label: [indices into `tickers`]}} for every
    advanced-limit category, aligned to the final universe's own ticker
    order (which may have dropped some holdings - see fetch_prices'
    coverage filter - so this can't just use the holdings list's own order).
    """
    ticker_index = {ticker: i for i, ticker in enumerate(tickers)}
    result: dict[str, dict[str, list[int]]] = {category: {} for category in _GROUP_CATEGORIES}

    for holding in holdings:
        index = ticker_index.get(holding.yahoo_ticker)
        if index is None:
            continue
        for category, attr in _GROUP_CATEGORIES.items():
            label = getattr(holding, attr) or "(unspecified)"
            result[category].setdefault(label, []).append(index)

    return result
