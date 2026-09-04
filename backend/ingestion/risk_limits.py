"""Parses the "Отчет о соблюдении лимитов инвестирования" risk-limits
workbook (e.g. "Риски_Tabys_Лимиты на 01.07.26.xls") into real regulatory
caps, keyed the same way as optimizer.GroupLimit's category/group.

The sheet is one flat table with no real headers: a section starts with a
label-only row ("По стране", "По валюте", ...) and is followed by data rows
(group label, limit fraction, actual fraction, limit in KZT, actual in KZT,
OK/violation signal) until the next section or a blank-label totals row.
Only the limit fraction (column 2) is used here - the rest is the fund's
current actual usage, not a constraint to carry into the optimizer.

Critically, this report's own labels do NOT line up 1:1 with the group
labels optimizer/adapter.py derives from an uploaded OSIP holdings workbook:
- country is a full Russian name here ("ИРЛАНДИЯ") vs. an ISO alpha-2 code
  there ("IE") - translated for the handful of countries this fund actually
  holds, not guessed at for others.
- issuer names come from a different classification field in this report
  than the one adapter.py reads from OSIP positions, so several genuinely
  don't match even for the same real holding (e.g. this report attributes
  the TIPS ETF to "BlackRock Fund Advisors" while OSIP's own issuer field
  says "iShares Trust" for the same position).
- instrument type here is a finer, differently-worded classification than
  OSIP's simple "Тип ценной бумаги" column.
- GICS sector and "open currency position" have no corresponding group data
  on an uploaded portfolio at all (nothing currently classifies holdings by
  GICS sector).

Rather than silently apply a limit under a label that doesn't actually
match anything in the portfolio's own group_indices (or worse, matching the
wrong group), parsing separates "matched" caps (label lines up with a real
group for this portfolio) from "unmatched" ones (parsed correctly, but this
portfolio has no group by that exact name) - both are reported so nothing
is silently dropped, but only "matched" ones are ever wired into the
optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Section header (as it literally appears in column 1) -> our canonical
# category key. Sections not listed here (GICS sector, open currency
# position, per-security "Акции <issuer>") have no corresponding group data
# on an uploaded portfolio and are reported as unmapped categories instead.
_SECTION_TO_CATEGORY = {
    "По стране": "country",
    "По валюте": "currency",
    "По эмитенту": "issuer",
    "По виду финансового инструмента": "instrument_type",
}

# The sheet's own document banner ("По лимитам инвестирования" = "On
# investment limits") - starts with "По " like a real section but never has
# data rows of its own, so it's not worth reporting as an unmapped category.
_IGNORED_SECTION_HEADERS = {"По лимитам инвестирования"}

# Full Russian country names -> the ISO alpha-2 code adapter.py derives from
# an ISIN prefix. Only the countries actually observed in TABYS reports are
# listed - an unrecognized country name is reported as unmatched rather
# than guessed at.
_COUNTRY_NAME_TO_ISO2 = {
    "ИРЛАНДИЯ": "IE",
    "КАЗАХСТАН": "KZ",
    "СОЕДИНЕННЫЕ ШТАТЫ": "US",
    "США": "US",
    "ВЕЛИКОБРИТАНИЯ": "GB",
    "ГЕРМАНИЯ": "DE",
    "ФРАНЦИЯ": "FR",
    "ЛЮКСЕМБУРГ": "LU",
    "НИДЕРЛАНДЫ": "NL",
}


@dataclass(frozen=True)
class ParsedRiskLimit:
    category: str
    group: str
    max_weight: float


class RiskLimitsError(Exception):
    pass


def _normalize_group_label(category: str, raw_label: str) -> str:
    if category == "country":
        return _COUNTRY_NAME_TO_ISO2.get(raw_label.strip().upper(), raw_label)
    return raw_label


def parse_risk_limits_workbook(path: str) -> tuple[list[ParsedRiskLimit], list[str]]:
    """Returns (parsed limits, unmapped section headers encountered) - the
    unmapped list is purely informational (e.g. "По GICS отраслям"),
    surfaced so the caller can tell the user those exist but aren't applied.
    """
    try:
        df = pd.read_excel(path, sheet_name=0, header=None, engine="calamine")
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean HTTP error upstream
        raise RiskLimitsError(f"Could not read this workbook: {exc}") from exc

    limits: list[ParsedRiskLimit] = []
    unmapped_sections: list[str] = []
    current_category: str | None = None
    current_section_mapped = False

    for _, row in df.iterrows():
        label = row[1] if len(row) > 1 else None
        raw_limit = row[2] if len(row) > 2 else None
        if pd.isna(label):
            continue
        label = str(label).strip()

        if pd.isna(raw_limit):
            if label.startswith("По ") and label not in _IGNORED_SECTION_HEADERS:
                current_category = _SECTION_TO_CATEGORY.get(label)
                current_section_mapped = current_category is not None
                if not current_section_mapped:
                    unmapped_sections.append(label)
            continue

        if not current_section_mapped or current_category is None:
            continue
        try:
            limit = float(raw_limit)
        except (TypeError, ValueError):
            continue
        if not (0 <= limit <= 1):
            continue

        group = _normalize_group_label(current_category, label)
        limits.append(ParsedRiskLimit(category=current_category, group=group, max_weight=limit))

    return limits, unmapped_sections
