# Portfolio Optimizer

A full-stack portfolio optimization tool: a FastAPI + CVXPY backend and a
React/TypeScript frontend. Build a portfolio from scratch, or rebalance an
existing one under real trading constraints - position limits, sector/
country/currency/issuer caps, turnover budgets, a cap on how many positions
may trade at all, and an optional buy/sell/reallocation mode for modeling new
capital coming in or capital being withdrawn.

## What it does

**Two universes:**

- **Demo** - a curated set of S&P sector ETFs (or a random sample of S&P 500
  names), with real price history pulled from Yahoo Finance.
- **Uploaded** - upload a holdings workbook (`.xls`) and the app parses it
  into real positions, resolves each ticker's own price history, and
  estimates return/risk from there. The parser targets one specific
  spreadsheet architecture (an "OSIP" workbook - a particular custom report
  layout, column contract, and formula set) it was built and tested against;
  it isn't a general-purpose "any portfolio spreadsheet" importer, and won't
  recognize a workbook laid out differently. A second, optional upload
  ("Отчет о соблюдении лимитов инвестирования") pulls in real regulatory
  position caps against that same specific report format, and matches them
  against the parsed holdings' own sector/country/currency/issuer groupings.

**Two modes:**

- **New portfolio** - allocate from scratch. A plain convex QP (CLARABEL).
- **Existing portfolio** - rebalance a current book. Either a turnover cap
  and/or a cap on the *number* of positions allowed to trade at all (both
  make it a Mixed-Integer QP, solved with SCIP), or opt into **flows**: buys
  funded by new capital, sells as outright withdrawals, and internal
  reallocation, each independently capped by count and by size. Buy/sell
  contributions are scored against your book's own current blended return
  (not raw dollars) so the solver can't inflate its objective just by
  injecting more capital into a merely-average asset, and a total-capital
  band (50%-200% of the starting portfolio by default) keeps that from
  running away even when nothing else caps it.

**Objective**, any mode: risk-aversion tradeoff (`max(return - λ·risk)`), a
target return (minimize risk to hit it exactly), or a target volatility
(maximize return under a risk cap) - all three exact under flows too, via a
proper normalized-return formulation (Dinkelbach's algorithm for the
risk-aversion/target-volatility cases, since dividing by a decision variable
directly isn't convex).

**Also:** live risk-free-rate-adjusted Sharpe (pulled from `^IRX`), opt-in
diversification (a two-stage entropy re-solve that trades a bounded amount of
the optimum for a less concentrated result), and an efficient frontier chart
plotted alongside wherever your actual portfolio (before and after) lands.

## Layout

- `backend/optimizer/` - the solver core: `params.py` (`OptimizerParams`,
  `GroupLimit`), `core.py` (covariance regularization, return/vol/Sharpe,
  effective-N), `modes.py` (`solve_new_portfolio`, `solve_existing_portfolio`,
  the flows solve, frontier/return-bounds helpers), `data.py` (the demo
  universe).
- `backend/ingestion/` - parses that one specific OSIP workbook layout (and
  its matching regulatory risk-limits report) into the solver's own inputs;
  see the caveat above - adapting it to a different holdings export means
  writing a new parser here, not reconfiguring this one.
- `backend/app/` - FastAPI wrapper (`main.py`, `schemas.py`, `storage.py` -
  a small SQLite store for uploaded portfolios/risk limits).
- `backend/smoke_test.py` - scripted end-to-end check of both modes, run
  directly against the solver core (no HTTP).
- `frontend/` - React + Vite + TypeScript + Recharts. `src/App.tsx` has the
  mode/objective toggles, every constraint as a slider or typed number, and
  the results chart/table; `src/api.ts` is the typed fetch client.

## Running it

**macOS:** double-click `start-optimizer.command`. **Windows:** double-click
`start-optimizer.bat`. Either one creates the backend's virtual environment,
installs both sets of dependencies on first run, starts the backend (port
8511) and frontend dev server (port 5173), and opens your browser.
`stop-optimizer.command` / `stop-optimizer.bat` stop both. Needs
[Node.js](https://nodejs.org) and Python 3.11+ on your `PATH`.

Manually:

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8511

# frontend, in a second terminal
cd frontend
npm install
npm run dev
```

## Solvers

CLARABEL handles every plain convex QP (new-portfolio mode, existing-
portfolio without a trade-count cap). SCIP is required whenever a cardinality
constraint turns the problem into a Mixed-Integer QP - `max_trades`, or any
of the flows mode's per-count caps (`pip install pyscipopt`, already in
`requirements.txt`).
