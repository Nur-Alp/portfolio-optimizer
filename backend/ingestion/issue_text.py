"""English text for the vendored parser's data-quality issues.

osip_workbook.py's messages are Russian (the source app's audience), and
some embed dynamic values (a lot code, a count) rather than being fully
static. Translating by the stable `code` field - never by matching the
Russian text itself - and pulling any embedded numbers/identifiers back out
with a small regex keeps this correct even if the vendored parser's exact
wording changes, as long as the code and general shape of the message don't.
Any code this doesn't recognize still gets a plain English fallback rather
than ever surfacing the original Russian.
"""

from __future__ import annotations

import re

from .domain import DataQualityIssue

_STATIC_MESSAGES: dict[str, str] = {
    "DQ-03": "A settlement is still listed as upcoming even though its due date has already passed.",
    "DQ-04": "The workbook has no stable identifiers for portfolio, account, lot, transaction, or settlement.",
    "DQ-05": "The workbook has no price source or valuation timestamp to support the stated market value.",
    "DQ-07": "The sector text in the source mixes GICS sectors, asset classes, and multi-sector lists.",
    "DQ-16": "The sheet's stated dimensions include a large number of formatted, empty trailing rows.",
}


def translate_issue(issue: DataQualityIssue) -> str:
    if issue.code == "DQ-01":
        lot_match = re.search(r"лота\s+([^\(]+?)\s*\(", issue.message)
        lot = lot_match.group(1).strip() if lot_match else "a lot"
        if "carrying_amount_native" in issue.affected_fields:
            return (
                f"Carrying value is unavailable for {lot} - this row will be excluded from computed "
                "carrying value and operational totals. Check whether the value is listed under a "
                "different header for this instrument type before treating the import as complete."
            )
        return f"Required calculated figures are unavailable for {lot}."

    if issue.code == "DQ-02":
        count_match = re.search(r"(\d+)", issue.message)
        count = count_match.group(1) if count_match else "multiple"
        return f"A settlement appears {count} times with an identical business signature."

    if issue.code == "DQ-12":
        numbers = re.findall(r"\d+", issue.message)
        listing_missing, ratings_missing = (numbers + ["some", "some"])[:2]
        return (
            f"Ratings/listing coverage is incomplete: {listing_missing} lot(s) have no listing "
            f"classification, and {ratings_missing} have no ratings from any agency."
        )

    return _STATIC_MESSAGES.get(issue.code, f"Data-quality note ({issue.code}).")
