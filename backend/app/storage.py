"""SQLite persistence for uploaded portfolios and their matched risk limits.

Everything in main.py's _UPLOADED_PORTFOLIOS is otherwise in-process only -
gone on a backend restart, and never available again after a browser
refresh even within the same run (the frontend only ever kept the
portfolio_id in React state). This mirrors each portfolio's entry to
.data/optimizer/optimizer.db - the same gitignored state directory the
launcher scripts already use for pid/log files, confirmed via
`git check-ignore` to never have been tracked - and reloads everything back
into memory on startup, so a previously uploaded TABYS portfolio (and, if
uploaded, its risk limits) is still there after either kind of reset.

What's persisted is the *raw holdings* (ticker/weight/classification) as a
JSON blob per row, not a fixed universe - the price-derived side (mu/Sigma/
which tickers survive the coverage filter) depends on the date window,
which main.py re-resolves and caches per window on top of these holdings,
rather than baking in whatever window happened to be active at upload time.
A real relational schema would be overkill for a single-user local dev
tool's two small tables; SQLite here is about atomic, single-file writes
and a conventional "it's a database" story, not normalization.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from ingestion.adapter import UploadedHolding

# backend/app/storage.py -> parents[2] is the portfolio-optimizer project
# root, alongside start-optimizer.command's own STATE_DIR=".data/optimizer".
DB_PATH = Path(__file__).resolve().parents[2] / ".data" / "optimizer" / "optimizer.db"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolios (
                portfolio_id TEXT PRIMARY KEY,
                holdings_json TEXT NOT NULL,
                report_date TEXT,
                issues_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_limits (
                portfolio_id TEXT PRIMARY KEY REFERENCES portfolios(portfolio_id),
                risk_limits_json TEXT NOT NULL
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_portfolio(portfolio_id: str, entry: dict) -> None:
    """`entry` matches the shape stored in _UPLOADED_PORTFOLIOS: a
    "holdings" list of UploadedHolding, plus report_date and issues."""
    holdings_json = json.dumps([asdict(h) for h in entry["holdings"]])
    issues_json = json.dumps(entry.get("issues", []))
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO portfolios (portfolio_id, holdings_json, report_date, issues_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(portfolio_id) DO UPDATE SET
                holdings_json = excluded.holdings_json,
                report_date = excluded.report_date,
                issues_json = excluded.issues_json
            """,
            (portfolio_id, holdings_json, entry.get("report_date"), issues_json),
        )


def save_risk_limits(portfolio_id: str, risk_limits: dict) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO risk_limits (portfolio_id, risk_limits_json)
            VALUES (?, ?)
            ON CONFLICT(portfolio_id) DO UPDATE SET risk_limits_json = excluded.risk_limits_json
            """,
            (portfolio_id, json.dumps(risk_limits)),
        )


def load_all_portfolios() -> dict[str, dict]:
    """Reloaded once at startup - a row that fails to deserialize is
    skipped rather than failing the whole backend, since this is recoverable
    (the client just re-uploads that one portfolio)."""
    if not DB_PATH.exists():
        return {}

    portfolios: dict[str, dict] = {}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT portfolio_id, holdings_json, report_date, issues_json FROM portfolios"
        ).fetchall()
        for portfolio_id, holdings_json, report_date, issues_json in rows:
            try:
                holdings = [UploadedHolding(**h) for h in json.loads(holdings_json)]
                portfolios[portfolio_id] = {
                    "holdings": holdings,
                    "report_date": report_date,
                    "issues": json.loads(issues_json),
                }
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        risk_rows = conn.execute("SELECT portfolio_id, risk_limits_json FROM risk_limits").fetchall()
        for portfolio_id, risk_limits_json in risk_rows:
            if portfolio_id not in portfolios:
                continue
            try:
                portfolios[portfolio_id]["risk_limits"] = json.loads(risk_limits_json)
            except json.JSONDecodeError:
                continue

    return portfolios
