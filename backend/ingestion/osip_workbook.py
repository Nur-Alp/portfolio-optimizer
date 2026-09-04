"""Tolerant parser and deterministic quality rules for OSIP legacy workbooks.

Vendored from portfolio-operations-dashboard's
backend/osip_dashboard/ingestion/osip_workbook.py - this app doesn't need
that app's full persistence/reporting stack, but real OSIP workbooks have
enough real-world quirks (blank cached-formula columns, column-order
revisions between generator versions, repo/deposit edge cases) that it's
worth reusing the tested parser rather than re-deriving it. Keep this and
domain.py in sync with the source if that parser changes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path
import re
import struct
from typing import Any, Iterable

from olefile import DEFECT_FATAL, OleFileIO
from python_calamine import CalamineWorkbook

from .domain import (
    CashBalanceSnapshot,
    DataQualityIssue,
    PortfolioSnapshot,
    PositionLotSnapshot,
    SettlementEvent,
    Severity,
    SourceRef,
)


SHEET_NAME = "ОСИП_ПОРТФЕЛЬ"
EXPECTED_COLUMN_COUNT = 83
CASH_PREFIX = "ОСТАТОК ДЕНЕЖНЫХ СРЕДСТВ"
IGNORED_SECTIONS = frozenset({"предстоящие расчеты"})

# Column *labels*, not positions - a generator revision (confirmed 2026-08:
# five new rating/classification columns inserted, one field relocated
# entirely out of sequence, one field split into two, three fields reworded)
# moves almost every column, but the label text for a given business field is
# stable far more often than its position. Each field lists every label text
# it has actually appeared under, in known real/test workbooks - not a fuzzy
# match, an exact match against a small curated set, so a genuinely renamed
# or removed column (test_parser_rejects_changed_column_contract) still
# fails to resolve rather than guessing. "№" (row/section anchor) and column
# 0 generally are NOT in here: both known layouts keep it fixed at column A,
# and it doubles as the section-label/cash-row-prefix column for every row,
# not just the header - it stays positional.
_FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "isin": ("НИН",),
    "security_type": ("Тип ценной бумаги",),
    # The original check was a substring match ("Количество" in text), which
    # also accepted a bare "Количество" without "(шт.)" - kept as a second
    # exact alternate rather than reintroducing substring matching, so this
    # field can't accidentally match some unrelated column that merely
    # contains the word "Количество" elsewhere in a longer label.
    "quantity": ("Количество (шт.)", "Количество"),
    "security_code": ("Код ценной бумаги",),
    "issuer": ("Эмитент",),
    "valuation_method": ("Метод определения балансовой стоимости ",),
    "instrument_currency": ("Валюта инструмента", "Валюта инструмета"),
    "raw_sector": ("Отрасль GICS",),
    "rating_sp": ("Рейтинг S&P",),
    "rating_moodys": ("Рейтинг Moody's",),
    "rating_fitch": ("Рейтинг Fitch",),
    "coupon_or_repo_rate": ("Ставка купона/ репо",),
    "nominal_value": ("Номинальная стоимость",),
    "open_date": ("Дата открытия", "Дата выпуска/ Дата открытия репо"),
    "close_date": ("Дата закрытия", "Дата выпуска/ Дата закрытия репо"),
    "purchase_date": ("Дата покупки",),
    "purchase_price": ("Цена покупки",),
    "purchase_yield": ("Доходность при покупке (%)",),
    "current_ytm": ("YTM на дату отчета",),
    "purchase_amount_native": ("Объем покупки в валюте сделки",),
    "purchase_amount_kzt": ("Объем покупки в тенге",),
    "carrying_amount_native": ("Балансовая стоимость: для РЕПО - в валюте сделки, для покупок в валюте интрумента",),
    # Dual-purpose column: a repo's *price* (not an amount - never usable as
    # a carrying value) or a deposit's closing *amount* (see the deposit
    # carrying-value fallback below, where this is the only one of the two
    # readings ever used).
    "deposit_closing_amount_native": ("Цена закрытия (для репо) / Объем закрытия (для депозита)",),
    "carrying_price_native": (
        "Балансовая цена: для облигация чистая (%) , для акций/РЕПО в валюте учета",
        "Чистая цена для облигаций (%), рыночная цена для акций/РЕПО в валюте учета",
    ),
    "official_carrying_value_kzt": ("Балансовая стоимость, в тенге",),
    "official_market_value_kzt": ("Рыночная стоимость в ТЕНГЕ на отчетную дату",),
    "reserve_kzt": ("Сумма резерва, в тенге",),
    "organizer_fee_kzt": ("Комиссия организатора торгов (KZT)",),
    "broker_fee_kzt": ("Брокерское вознаграждение (KZT)",),
    "days_held": ("дней в портфеле",),
    "portfolio_weight": ("Доля в портфеле",),
    "income": ("Доход",),
    "holding_period_yield": ("HPY",),
    "previous_coupon_date": ("Дата последней купонной выплаты",),
    "next_coupon_date": ("Дата следующей купонной выплаты",),
    "expected_coupon": ("Сумма ожидаемого купона",),
    "coupon_period_days": ("Купоннный период",),
    "accrued_income_kzt": ("Накопленный купон в ТЕНГЕ / Начисленное вознаграждение по депозиту",),
    "principal_indexation": ("индексация основной суммы",),
    "coupon_indexation": ("индексация для купона",),
    "report_fx_rate": ("курс для ЦБ",),
    "listing_rating": ("листинг/\nрейтинг",),
}
# Legacy calculated fields the OSIP generator often leaves blank (a cached
# formula result Excel would recalculate on open) - see BROKEN_CALCULATED_FIELDS
# usage below and the reference in services/holdings_export/coupons.py.
# Field names only now (not columns): each one's actual column is resolved
# per-workbook via _FIELD_LABELS like everything else.
BROKEN_CALCULATED_FIELDS: tuple[str, ...] = (
    "carrying_price_native",
    "official_carrying_value_kzt",
    "official_market_value_kzt",
    "days_held",
    "portfolio_weight",
    "income",
    "holding_period_yield",
    "expected_coupon",
)
# Header-row requirements for recognizing the sheet as OSIP at all - unlike
# the fields above, these must resolve or the workbook is rejected
# (test_parser_rejects_changed_column_contract). Column A's "№" is checked
# separately since it also anchors every non-header row, not just this one.
_REQUIRED_HEADER_FIELDS: tuple[str, ...] = ("isin", "security_type", "quantity")
# flag_be/flag_bf (Excel columns BE/BF) back the one Excel-recalculated
# formula this parser reproduces (_formula_carrying_price) but carry no
# header text in either known layout - there is nothing to search for.
# Reused only when the rest of the sheet resolves to the original column
# positions (see _uses_legacy_layout); on a workbook using the newer layout,
# carrying_price_native simply falls through to "unavailable" rather than
# risking a value read from whatever now occupies BE/BF.
_LEGACY_FLAG_BE_COLUMN = 56
_LEGACY_FLAG_BF_COLUMN = 57
BIFF_BOUNDSHEET = 0x0085
BIFF_EOF = 0x000A
BIFF_FORMULA = 0x0006
CASH_PATTERN = re.compile(r"ДЕНЕЖНЫХ СРЕДСТВ В ([A-Z]{3})(?: в(?: (.*))?)?$")


class OsipWorkbookError(ValueError):
    """Raised when a workbook cannot be interpreted as an OSIP snapshot."""


def parse_osip_workbook(
    path: str | Path, *, portfolio_code: str, source_name: str | None = None
) -> PortfolioSnapshot:
    source_path = Path(path)
    content = source_path.read_bytes()
    workbook = CalamineWorkbook.from_path(source_path)
    if SHEET_NAME not in workbook.sheet_names:
        raise OsipWorkbookError(f"Отсутствует обязательный лист: {SHEET_NAME}")

    rows = workbook.get_sheet_by_name(SHEET_NAME).to_python(skip_empty_area=False)
    display_path = Path(source_name) if source_name is not None else source_path
    return _parse_rows(
        rows,
        display_path,
        hashlib.sha256(content).hexdigest(),
        portfolio_code=portfolio_code,
        formula_cells=_formula_cells(source_path),
    )


def _parse_rows(
    rows: list[list[Any]],
    source_path: Path,
    source_sha256: str,
    *,
    portfolio_code: str = "TEST",
    formula_cells: set[tuple[int, int]] | None = None,
) -> PortfolioSnapshot:
    """Build a snapshot from cells using stable columns and content-based rows.

    OSIP generators may insert, remove, or reorder business rows. The parser
    therefore discovers the header and classifies every later row by the
    stable column contract instead of relying on worksheet row numbers.
    """
    if not rows or max((len(row) for row in rows), default=0) < EXPECTED_COLUMN_COUNT:
        raise OsipWorkbookError("Лист OSIP содержит меньше ожидаемых 83 столбцов")

    header_index = _find_header_row(rows)
    columns = _resolve_columns(rows[header_index])
    report_date = _find_report_date(rows[: header_index + 1])

    if report_date is None:
        raise OsipWorkbookError("Отчётная дата отсутствует над бизнес-заголовком в столбце J")

    positions: list[PositionLotSnapshot] = []
    raw_settlements: list[SettlementEvent] = []
    cash_balances: list[CashBalanceSnapshot] = []
    issues: list[DataQualityIssue] = []
    section = ""

    for row_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        padded = tuple(row) + ("",) * max(0, EXPECTED_COLUMN_COUNT - len(row))
        label = _text(_cell(padded, 0))
        security_type = _text(_fc(padded, columns, "security_type"))
        quantity = _decimal(_fc(padded, columns, "quantity"))

        if label.startswith(CASH_PREFIX):
            if label.endswith(" в"):
                # Empty cash templates are layout placeholders, not balances.
                continue
            cash = _parse_cash(
                padded, portfolio_code, report_date, source_path.name, row_number, columns=columns
            )
            cash_balances.append(cash)
            continue

        if not security_type or quantity is None or quantity == 0:
            if label:
                section = label
            continue

        source = SourceRef(source_path.name, SHEET_NAME, row_number)
        if _is_ignored_section(section):
            continue
        if quantity > 0:
            position = _parse_position(
                padded,
                portfolio_code,
                report_date,
                source,
                section,
                columns=columns,
                formula_cells=formula_cells or set(),
            )
            positions.append(position)
            if position.unavailable_fields:
                issues.append(
                    DataQualityIssue(
                        code="DQ-01",
                        severity=Severity.BLOCKER,
                        message=_dq01_message(position),
                        source_refs=(source,),
                        affected_fields=position.unavailable_fields,
                    )
                )
        else:
            raw_settlements.append(
                _parse_settlement(padded, portfolio_code, report_date, source, columns=columns)
            )

    settlements = _deduplicate_settlements(raw_settlements, issues)
    for settlement in settlements:
        if settlement.settlement_date and settlement.settlement_date < report_date:
            issues.append(
                DataQualityIssue(
                    code="DQ-03",
                    severity=Severity.HIGH,
                    message="Расчёт остаётся в разделе предстоящих событий после наступления срока.",
                    source_refs=settlement.source_refs,
                    affected_fields=("settlement_date", "status"),
                )
            )

    issues.extend(
        [
            DataQualityIssue(
                code="DQ-04",
                severity=Severity.HIGH,
                message="В рабочей книге отсутствуют устойчивые идентификаторы портфеля, счёта, лота, операции или расчёта.",
                affected_fields=(
                    "portfolio_id",
                    "account_id",
                    "lot_id",
                    "transaction_id",
                    "settlement_id",
                ),
            ),
            DataQualityIssue(
                code="DQ-05",
                severity=Severity.HIGH,
                message="В рабочей книге нет источника цены и времени оценки, необходимых для подтверждения рыночной стоимости.",
                affected_fields=("price_source", "valuation_timestamp"),
            ),
        ]
    )
    issues.extend(
        _additional_quality_issues(
            positions=positions,
            rows=rows,
            header_index=header_index,
        )
    )

    return PortfolioSnapshot(
        portfolio_code=portfolio_code,
        report_date=report_date,
        source_path=source_path,
        source_sha256=source_sha256,
        positions=tuple(positions),
        raw_settlements=tuple(raw_settlements),
        settlements=tuple(settlements),
        cash_balances=tuple(cash_balances),
        issues=tuple(issues),
        resolved_columns=dict(columns),
    )


def _find_header_row(rows: list[list[Any]]) -> int:
    for index, row in enumerate(rows):
        if _text(_cell(row, 0)) != "№":
            continue
        columns = _resolve_columns(row)
        if all(field in columns for field in _REQUIRED_HEADER_FIELDS):
            return index
    raise OsipWorkbookError("Не удалось найти бизнес-заголовок OSIP по стабильному контракту столбцов")


def _resolve_columns(header_row: list[Any]) -> dict[str, int]:
    """Map each known field name to the column its label was actually found
    in, searching the whole row rather than assuming a fixed position - see
    _FIELD_LABELS. A field whose label isn't found at all is simply absent
    from the returned dict; callers read it as unavailable, the same as a
    blank cell, rather than raising (only the header-level fields in
    _REQUIRED_HEADER_FIELDS being absent is treated as a hard failure).
    """
    texts = [_text(value) for value in header_row]
    resolved: dict[str, int] = {}
    for field_name, labels in _FIELD_LABELS.items():
        for index, text in enumerate(texts):
            if text in labels:
                resolved[field_name] = index
                break
    return resolved


_LEGACY_COLUMNS: dict[str, int] = {
    "isin": 6,
    "quantity": 16,
    "nominal_value": 11,
    "carrying_amount_native": 26,
    "carrying_price_native": 24,
}


def _uses_legacy_layout(columns: dict[str, int]) -> bool:
    """True only when the resolved columns match the original fixed
    positions this parser was written against - the sole condition under
    which the unlabeled BE/BF formula-flag columns can safely be assumed to
    still be at BE/BF (see _formula_carrying_price)."""
    return columns.get("isin") == 6 and columns.get("carrying_amount_native") == 26


def _fc(row: tuple[Any, ...] | list[Any], columns: dict[str, int], field: str) -> Any:
    """Read a field by its resolved column, or "" if that field's label
    wasn't found anywhere in this workbook's header. Degrades exactly like
    an empty/blank cell (the parser's existing behavior for those) instead
    of needing separate handling at every call site."""
    column = columns.get(field)
    return _cell(row, column) if column is not None else ""


def _find_report_date(rows: list[list[Any]]) -> date | None:
    for row in rows:
        for value in row:
            candidate = _as_date(value)
            if candidate is not None:
                return candidate
    return None


def _is_ignored_section(section: str) -> bool:
    return section.casefold().replace("ё", "е").strip() in IGNORED_SECTIONS


def _dq01_message(position: PositionLotSnapshot) -> str:
    """DQ-01 fires as a BLOCKER on every incomplete lot - it already caught
    the D_14 deposit whose "Балансовая стоимость" was blank (the source
    states a deposit's value under a different, dual-purpose column
    instead - see the carrying-value fallback above), but the generic
    message it used to have didn't say *which* lot, what the consequence
    was, or that a hidden column mapping might be the real cause. Someone
    still acknowledged and published through it. This is the same
    detection, made worth reading before doing that again.
    """
    fields = position.unavailable_fields
    security_type = position.raw_security_type or "без указанного типа"
    # carrying_amount_native is the one that actually blanks a headline
    # total - worth the specific tip regardless of what else on the row is
    # also blank (commonly one of BROKEN_CALCULATED_FIELDS, a much lower-
    # stakes cached-formula gap Excel would recalculate on open).
    if "carrying_amount_native" in fields:
        return (
            f"Балансовая сумма недоступна для лота {position.security_code} ({security_type}) - "
            "строка будет исключена из расчётной балансовой стоимости и операционного итога. "
            "Проверьте, не указано ли значение под другим заголовком для этого типа инструмента "
            "(например, «Объём покупки» для депозитов, «Цена закрытия» для РЕПО), прежде чем "
            "подтверждать импорт."
        )
    return (
        f"Обязательные расчётные показатели недоступны для лота {position.security_code} "
        f"({security_type}): {', '.join(fields)}. Строка будет исключена из расчётной балансовой "
        "стоимости и операционного итога."
    )


def _normalize_issuer(value: str) -> str:
    """Correct a verified typo for display while retaining the raw source row."""
    return value.replace("MSCIope", "MSCI Europe")


def _parse_position(
    row: tuple[Any, ...],
    portfolio_code: str,
    report_date: date,
    source: SourceRef,
    section: str,
    *,
    columns: dict[str, int],
    formula_cells: set[tuple[int, int]],
) -> PositionLotSnapshot:
    unavailable = tuple(
        field_name
        for field_name in BROKEN_CALCULATED_FIELDS
        if field_name not in columns
        or (
            _is_blank(_fc(row, columns, field_name))
            and (source.row_number - 1, columns[field_name]) not in formula_cells
        )
    )
    # These are inputs to the one operational calculation the dashboard is
    # allowed to make.  Unlike the legacy calculated fields above, a blank
    # cached value cannot be treated as harmless formula evidence: without
    # either input the carrying value is genuinely unknowable.  Flag it at
    # parsing time so it cannot silently disappear from an aggregate as zero.
    carrying_amount_native = _decimal(_fc(row, columns, "carrying_amount_native"))
    row_close_date = _as_date(_fc(row, columns, "close_date"))
    if carrying_amount_native is None:
        # The source workbook never fills "Балансовая стоимость" for a
        # deposit at all - confirmed by inspecting its own "Рыночная
        # стоимость" formula directly (via LibreOffice, since the saved
        # file only carries a stale cached result): for a deposit
        # (BI=5), that formula reads
        # =IF(BI=9,"",IF(BI=5,AX40+AA40,...)) - accrued deposit interest
        # (AX, "Накопленный купон.../Начисленное вознаграждение по
        # депозиту") plus the purchase amount (AA, "Объем покупки в
        # тенге"), never the carrying-value column at all. Mirror that
        # exactly: substitute the native purchase amount for a deposit
        # specifically (a repo's blank carrying value is a different,
        # unrelated gap - this substitution only makes sense for the
        # workbook's own deposit branch). accrued_income_kzt is already
        # added separately in derived_carrying_value_kzt below, matching
        # the formula's own "+AX" term, so it isn't added twice here.
        raw_type = _text(_fc(row, columns, "security_type")).casefold()
        if "депозит" in raw_type:
            carrying_amount_native = _decimal(_fc(row, columns, "purchase_amount_native"))
    report_fx_rate = _decimal(_fc(row, columns, "report_fx_rate"))
    unavailable = unavailable + tuple(
        field_name
        for field_name, value in (
            ("carrying_amount_native", carrying_amount_native),
            ("report_fx_rate", report_fx_rate),
        )
        if value is None
    )
    carrying_price_native = _decimal(_fc(row, columns, "carrying_price_native"))
    carrying_price_column = columns.get("carrying_price_native")
    if (
        carrying_price_native is None
        and carrying_price_column is not None
        and (source.row_number - 1, carrying_price_column) in formula_cells
    ):
        # The OSIP .xls generator often stores this formula with an empty
        # cached result. Excel recalculates it on open, while Calamine exposes
        # the cache as blank. Reproduce the published formula only when BIFF
        # proves that the source cell is formula-backed; a genuinely blank
        # non-formula source remains unavailable.
        carrying_price_native = _formula_carrying_price(row, columns)

    expected_coupon_cached = _decimal(_fc(row, columns, "expected_coupon"))

    return PositionLotSnapshot(
        portfolio_code=portfolio_code,
        report_date=report_date,
        source=source,
        source_section=section,
        security_code=_text(_fc(row, columns, "security_code")),
        isin=_text(_fc(row, columns, "isin")),
        raw_security_type=_text(_fc(row, columns, "security_type")),
        issuer=_normalize_issuer(_text(_fc(row, columns, "issuer"))),
        valuation_method=_text(_fc(row, columns, "valuation_method")),
        instrument_currency=_text(_fc(row, columns, "instrument_currency")),
        raw_sector=_text(_fc(row, columns, "raw_sector")),
        rating_sp=_text(_fc(row, columns, "rating_sp")),
        rating_moodys=_text(_fc(row, columns, "rating_moodys")),
        rating_fitch=_text(_fc(row, columns, "rating_fitch")),
        coupon_or_repo_rate=_decimal(_fc(row, columns, "coupon_or_repo_rate")),
        nominal_value=_decimal(_fc(row, columns, "nominal_value")),
        open_date=_as_date(_fc(row, columns, "open_date")),
        close_date=row_close_date,
        quantity=_decimal(_fc(row, columns, "quantity")) or Decimal("0"),
        purchase_date=_as_date(_fc(row, columns, "purchase_date")),
        purchase_price=_decimal(_fc(row, columns, "purchase_price")),
        purchase_yield=_decimal(_fc(row, columns, "purchase_yield")),
        current_ytm=_decimal(_fc(row, columns, "current_ytm")),
        purchase_amount_native=_decimal(_fc(row, columns, "purchase_amount_native")),
        purchase_amount_kzt=_decimal(_fc(row, columns, "purchase_amount_kzt")),
        carrying_amount_native=carrying_amount_native,
        carrying_price_native=carrying_price_native,
        reserve_kzt=_decimal(_fc(row, columns, "reserve_kzt")),
        organizer_fee_kzt=_decimal(_fc(row, columns, "organizer_fee_kzt")),
        broker_fee_kzt=_decimal(_fc(row, columns, "broker_fee_kzt")),
        accrued_income_kzt=_decimal(_fc(row, columns, "accrued_income_kzt")),
        principal_indexation=_decimal(_fc(row, columns, "principal_indexation")),
        report_fx_rate=report_fx_rate,
        previous_coupon_date=_as_date(_fc(row, columns, "previous_coupon_date")),
        next_coupon_date=_as_date(_fc(row, columns, "next_coupon_date")),
        listing_rating=_text(_fc(row, columns, "listing_rating")),
        expected_coupon_cached=expected_coupon_cached,
        coupon_period_days=_decimal(_fc(row, columns, "coupon_period_days")),
        coupon_indexation=_decimal(_fc(row, columns, "coupon_indexation")),
        unavailable_fields=unavailable,
        raw_row=row,
    )


def _formula_carrying_price(
    row: tuple[Any, ...] | list[Any], columns: dict[str, int] | None = None
) -> Decimal | None:
    """Evaluate OSIP's published formula for the balance-price column.

    Excel formula (using its one-based column letters)::

        IF(Q=0,0,IF(OR(AA=0,ISBLANK(AA)),"",
           IF(BF=4,AA/Q,IF(BE=3,AA/Q,AA/Q/L*100))))

    Q/L/AA are resolved per-workbook like every other field (see
    _FIELD_LABELS); BE/BF have no header text in any known layout (see
    _uses_legacy_layout) and stay at their original fixed positions,
    used only when the rest of the sheet also resolves to the original
    layout - never assumed on a workbook using the newer one.
    """
    columns = columns if columns is not None else _LEGACY_COLUMNS
    quantity = _decimal(_fc(row, columns, "quantity"))
    if quantity is None:
        return None
    if quantity == 0:
        return Decimal("0")
    carrying_amount = _decimal(_fc(row, columns, "carrying_amount_native"))
    if carrying_amount is None or carrying_amount == 0:
        return None
    if not _uses_legacy_layout(columns):
        # BE/BF carry no header label to resolve in any layout - only safe
        # to read positionally when the rest of the sheet confirms this is
        # the original layout they were always at. On a workbook using the
        # newer layout, whatever now occupies those two columns is unknown,
        # so the formula fallback simply doesn't apply rather than risking
        # a value read from the wrong cells.
        return None
    flag_be = _decimal(_cell(row, _LEGACY_FLAG_BE_COLUMN))
    flag_bf = _decimal(_cell(row, _LEGACY_FLAG_BF_COLUMN))
    if flag_bf == 4 or flag_be == 3:
        return carrying_amount / quantity
    nominal = _decimal(_fc(row, columns, "nominal_value"))
    if nominal is None or nominal == 0:
        return None
    return carrying_amount / quantity / nominal * Decimal("100")


def _formula_cells(path: Path) -> set[tuple[int, int]]:
    """Return formula coordinates in the OSIP sheet of a legacy ``.xls`` file.

    The supplied OSIP generator writes formula records with invalid cached
    results. Excel recalculates them when the workbook opens, whereas Calamine
    correctly exposes the invalid cache as an empty cell. Formula presence is
    therefore evidence that a calculated field exists and must not become a
    false DQ-01 finding. This helper does not evaluate formulas or invent their
    values; it only reads their BIFF coordinates.
    """
    try:
        with OleFileIO(path, raise_defects=DEFECT_FATAL) as container:
            stream_name = "Workbook" if container.exists("Workbook") else "Book"
            if not container.exists(stream_name):
                return set()
            workbook_stream = container.openstream(stream_name).read()
        sheet_offsets = _biff_sheet_offsets(workbook_stream)
        start = sheet_offsets.get(SHEET_NAME)
        if start is None:
            return set()
        end = min(
            (offset for offset in sheet_offsets.values() if offset > start),
            default=len(workbook_stream),
        )
    except (OSError, ValueError, struct.error):
        return set()

    formulas: set[tuple[int, int]] = set()
    position = start
    while position + 4 <= end:
        record_type, length = struct.unpack_from("<HH", workbook_stream, position)
        position += 4
        if position + length > end:
            break
        payload = workbook_stream[position : position + length]
        position += length
        if record_type == BIFF_EOF:
            break
        if record_type == BIFF_FORMULA and length >= 6:
            row, column = struct.unpack_from("<HH", payload)
            formulas.add((row, column))
    return formulas


def _biff_sheet_offsets(workbook_stream: bytes) -> dict[str, int]:
    offsets: dict[str, int] = {}
    position = 0
    while position + 4 <= len(workbook_stream):
        record_type, length = struct.unpack_from("<HH", workbook_stream, position)
        position += 4
        if position + length > len(workbook_stream):
            break
        payload = workbook_stream[position : position + length]
        position += length
        if record_type != BIFF_BOUNDSHEET or length < 8:
            continue
        offset = struct.unpack_from("<I", payload)[0]
        character_count = payload[6]
        flags = payload[7]
        encoded_name = payload[8:]
        if flags & 0x01:
            name = encoded_name[: character_count * 2].decode("utf-16le", errors="replace")
        else:
            name = encoded_name[:character_count].decode("latin-1", errors="replace")
        offsets[name] = offset
    return offsets


def _additional_quality_issues(
    *,
    positions: list[PositionLotSnapshot],
    rows: list[list[Any]],
    header_index: int,
) -> list[DataQualityIssue]:
    """Deterministic rules from the approved workbook quality register."""
    findings: list[DataQualityIssue] = []

    sector_refs = tuple(
        position.source
        for position in positions
        if "," in position.raw_sector
        or position.raw_sector
        in {
            "Government bonds",
            "Exchange Traded Fund",
            "Development Institutions",
        }
    )
    if sector_refs:
        findings.append(
            DataQualityIssue(
                code="DQ-07",
                severity=Severity.MEDIUM,
                message="Текст сектора в источнике смешивает сектора GICS, классы активов и списки нескольких секторов.",
                source_refs=sector_refs,
                affected_fields=("raw_sector", "normalized_sector"),
            )
        )

    coverage_refs = tuple(
        position.source
        for position in positions
        if not position.listing_rating
        or not any((position.rating_sp, position.rating_moodys, position.rating_fitch))
    )
    if coverage_refs:
        listing_missing = sum(not position.listing_rating for position in positions)
        ratings_missing = sum(
            not any((position.rating_sp, position.rating_moodys, position.rating_fitch))
            for position in positions
        )
        findings.append(
            DataQualityIssue(
                code="DQ-12",
                severity=Severity.MEDIUM,
                message=(
                    f"Покрытие рейтингами/листингом неполное: у {listing_missing} лотов нет "
                    f"классификации листинга, а у {ratings_missing} отсутствуют рейтинги всех агентств."
                ),
                source_refs=coverage_refs,
                affected_fields=("listing_rating", "ratings"),
            )
        )

    last_content_index = max(
        (
            index
            for index, row in enumerate(rows)
            if any(not _is_blank(value) for value in row[:EXPECTED_COLUMN_COUNT])
        ),
        default=header_index,
    )
    if len(rows) - last_content_index - 1 >= 50:
        findings.append(
            DataQualityIssue(
                code="DQ-16",
                severity=Severity.LOW,
                message="Заявленные размеры листа содержат большое число отформатированных пустых строк в конце.",
                affected_fields=("physical_sheet_dimensions",),
            )
        )

    return findings


def _parse_settlement(
    row: tuple[Any, ...],
    portfolio_code: str,
    report_date: date,
    source: SourceRef,
    *,
    columns: dict[str, int],
) -> SettlementEvent:
    return SettlementEvent(
        portfolio_code=portfolio_code,
        report_date=report_date,
        security_code=_text(_fc(row, columns, "security_code")),
        isin=_text(_fc(row, columns, "isin")),
        raw_security_type=_text(_fc(row, columns, "security_type")),
        issuer=_text(_fc(row, columns, "issuer")),
        currency=_text(_fc(row, columns, "instrument_currency")),
        quantity=_decimal(_fc(row, columns, "quantity")) or Decimal("0"),
        settlement_date=_as_date(_fc(row, columns, "purchase_date")),
        purchase_price=_decimal(_fc(row, columns, "purchase_price")),
        amount_native=_decimal(_fc(row, columns, "purchase_amount_native")),
        amount_kzt=_decimal(_fc(row, columns, "purchase_amount_kzt")),
        source_refs=(source,),
        raw_rows=(row,),
    )


def _parse_cash(
    row: tuple[Any, ...],
    portfolio_code: str,
    report_date: date,
    workbook_name: str,
    row_number: int,
    *,
    columns: dict[str, int],
) -> CashBalanceSnapshot:
    label = _text(_cell(row, 0))
    match = CASH_PATTERN.search(label)
    if not match:
        raise OsipWorkbookError(f"Не удалось разобрать наименование денежных средств в строке {row_number}: {label}")
    currency, custodian = match.groups()
    return CashBalanceSnapshot(
        portfolio_code=portfolio_code,
        report_date=report_date,
        source=SourceRef(workbook_name, SHEET_NAME, row_number),
        raw_label=label,
        currency=currency,
        custodian=custodian.strip() if custodian and custodian.strip() else None,
        native_amount=_decimal(_fc(row, columns, "carrying_amount_native")) or Decimal("0"),
        kzt_amount=_decimal(_fc(row, columns, "official_carrying_value_kzt")) or Decimal("0"),
        raw_row=row,
    )


def _deduplicate_settlements(
    raw_settlements: Iterable[SettlementEvent],
    issues: list[DataQualityIssue],
) -> list[SettlementEvent]:
    groups: dict[tuple[Any, ...], list[SettlementEvent]] = defaultdict(list)
    for settlement in raw_settlements:
        groups[settlement.signature].append(settlement)

    unique: list[SettlementEvent] = []
    for group in groups.values():
        first = group[0]
        refs = tuple(ref for event in group for ref in event.source_refs)
        raw_rows = tuple(row for event in group for row in event.raw_rows)
        unique.append(replace(first, source_refs=refs, raw_rows=raw_rows))
        if len(group) > 1:
            issues.append(
                DataQualityIssue(
                    code="DQ-02",
                    severity=Severity.BLOCKER,
                    message=f"Расчёт встречается {len(group)} раз с одинаковой бизнес-сигнатурой.",
                    source_refs=refs,
                    affected_fields=("settlement",),
                )
            )
    return unique


def _cell(row: tuple[Any, ...] | list[Any], index: int) -> Any:
    return row[index] if index < len(row) else ""


def _text(value: Any) -> str:
    return str(value).strip() if not _is_blank(value) else ""


def _decimal(value: Any) -> Decimal | None:
    if _is_blank(value):
        return None
    if isinstance(value, bool):
        return Decimal(int(value))
    try:
        return Decimal(str(value).strip().replace(" ", "").replace(",", "."))
    except (InvalidOperation, ValueError):
        return None


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if _is_blank(value):
        return None
    text = str(value).strip().split()[0]
    for separator, order in (("-", (0, 1, 2)), (".", (2, 1, 0))):
        parts = text.split(separator)
        if len(parts) == 3:
            try:
                year, month, day = (int(parts[index]) for index in order)
                return date(year, month, day)
            except ValueError:
                pass
    return None


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())
