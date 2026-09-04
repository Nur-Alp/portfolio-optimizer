import { memo, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Customized,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchFrontier,
  fetchReturnBounds,
  fetchRiskFreeRate,
  fetchUniverse,
  fetchUploadedPortfolio,
  fetchUploadedRiskLimits,
  groupLimitKey,
  runOptimize,
  uploadPortfolio,
  uploadRiskLimits,
  type FrontierPoint,
  type GroupLimitRequest,
  type OptimizeResponse,
  type RiskLimitsResponse,
  type Universe,
  type UploadPortfolioResponse,
} from "./api";
import "./App.css";

type Mode = "new" | "existing";
type ObjectiveMode = "riskAversion" | "targetReturn" | "targetVolatility";
type DemoVariant = "sectors" | "sp500_random";

const UPLOADED_PORTFOLIO_ID_KEY = "optimizer_uploadedPortfolioId";
const ACTIVE_TAB_KEY = "optimizer_activeTab";

/** "instrument_type" -> "Instrument Type" - categories are dynamic strings
 * from the backend (whatever group data the current universe has), not a
 * fixed set, so labels are derived rather than hardcoded. */
function formatCategoryLabel(category: string): string {
  return category.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * A native <input type="range"> can only reach min + n*step for whole n -
 * if (max - min) isn't an exact multiple of step (e.g. min is a fetched
 * float like 28.333...%), dragging to the far end lands just short of max,
 * never at it. Nudges min up to the smallest value that still lands exactly
 * on max after whole steps, so the slider's true ceiling is reachable by
 * drag. The typed field keeps using the real, unrounded min - only the
 * native range element's own min attribute needs this.
 */
function alignRangeMin(min: number, max: number, step: number): number {
  if (step <= 0 || !Number.isFinite(min) || !Number.isFinite(max)) return min;
  const stepsFromMax = Math.floor((max - min) / step);
  return max - stepsFromMax * step;
}

// Matplotlib's viridis colormap, sampled at 6 stops - same palette the
// notebook's frontier plot uses (plt.cm.viridis), since Recharts has no
// built-in colormap.
const VIRIDIS = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"];

function viridisColor(t: number): string {
  const index = Math.round(clamp(t, 0, 1) * (VIRIDIS.length - 1));
  return VIRIDIS[index];
}

// How many consecutive frontier points share one Scatter+line segment (see
// FrontierChart) - trades a slightly coarser gradient for far fewer live
// Recharts component instances per render.
const FRONTIER_SEGMENT_BATCH = 5;

/** Decimals to show/accept when none is given - derived from the step. */
function precisionFor(step: number): number {
  if (step >= 1) return 0;
  const text = step.toString();
  const dot = text.indexOf(".");
  return dot === -1 ? 2 : text.length - dot - 1;
}

function formatValue(value: number, decimals: number): string {
  return value.toFixed(decimals);
}

// Risk-aversion presets, in this app's own scale (objective is
// mu'w - lambda*w'Sigma*w, with no 1/2). Textbooks and the CFA Institute
// curriculum (Bodie/Kane/Marcus "Investments") quote a risk aversion
// coefficient A for U = E[r] - (A/2)*Var(r) - the standard convention *does*
// carry that 1/2, so our lambda = A/2. Bands below follow that literature's
// categorization (aggressive A~1-2, moderate ~2-4, conservative ~4-8, very
// conservative/institutional ~8-20+), converted to our scale; treat "pension
// fund-like" as a practitioner rule of thumb for that top band, not a cited
// figure for a specific fund.
const RISK_AVERSION_PRESETS = [
  { short: "None", label: "No risk adjustment (max return)", value: 0 },
  { short: "Aggressive", label: "Aggressive", value: 0.5 },
  { short: "Moderate", label: "Moderate", value: 1.5 },
  { short: "Conservative", label: "Conservative", value: 3 },
  { short: "V. conservative", label: "Very conservative (pension-like)", value: 6 },
];

const PALETTE = [
  "#2f6fed", "#20b26c", "#f2a93b", "#e0554f", "#7d5fd6",
  "#26a1a8", "#c2588f", "#7b8794", "#b08d3e", "#4f8ef7",
];

type DateMode = "range" | "lookback";
type Tab = "demo" | "uploaded";

function isoDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

export default function App() {
  const [universe, setUniverse] = useState<Universe | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Which data source is active. "uploaded" needs an actual upload before
  // there's a universe to show - see the early-return panel for that tab
  // further down. Upload state persists across tab switches, so flipping
  // back to "demo" and then back to "uploaded" doesn't lose it. Both this
  // and uploadedPortfolioId are seeded from localStorage and restored via
  // fetchUploadedPortfolio() below, since the backend now persists an
  // uploaded portfolio to disk across a refresh or backend restart - only
  // the browser's own memory of *which* portfolio_id to ask for was ever
  // missing.
  const [activeTab, setActiveTab] = useState<Tab>(
    () => (localStorage.getItem(ACTIVE_TAB_KEY) as Tab | null) ?? "demo",
  );
  const [uploadedPortfolioId, setUploadedPortfolioId] = useState<string | null>(() =>
    localStorage.getItem(UPLOADED_PORTFOLIO_ID_KEY),
  );
  const [uploadedInfo, setUploadedInfo] = useState<UploadPortfolioResponse | null>(null);
  const [demoVariant, setDemoVariant] = useState<DemoVariant>("sectors");
  const [demoSeed, setDemoSeed] = useState(() => Math.floor(Math.random() * 1e9));
  // Ticker -> weight from the upload, kept separately from currentWeights so
  // that changing the price-history window (which can change which tickers
  // have enough coverage to survive) re-derives weights by ticker instead of
  // by array position - a real holding's weight shouldn't silently shift to
  // a different instrument just because the universe array got reordered.
  const [uploadedWeightsByTicker, setUploadedWeightsByTicker] = useState<Record<string, number> | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // The TABYS regulatory risk-limits workbook - a separate upload, optional,
  // and only meaningful once a portfolio is already uploaded (it's matched
  // against that portfolio's own group data). Cleared whenever a different
  // portfolio file is uploaded, since matches were computed against the old
  // one's group_indices.
  const [riskLimitsInfo, setRiskLimitsInfo] = useState<RiskLimitsResponse | null>(null);
  const [riskLimitsUploading, setRiskLimitsUploading] = useState(false);
  const [riskLimitsError, setRiskLimitsError] = useState<string | null>(null);

  // The frontier chart does its own nearest-point hit-testing on real pixel
  // coordinates, rather than relying on Recharts' Tooltip/cursor resolution -
  // that's what a working reference implementation for this exact "curve
  // bends back on itself" case does, and it sidesteps every issue we hit
  // trying to make Recharts' own per-series hover logic behave (same-name
  // series ambiguity, small hit targets, wrong-branch swaps): true 2D
  // distance across every rendered point naturally picks whichever one is
  // visually closest to the cursor, regardless of curve shape or which
  // series a point happens to belong to. Each Scatter's shape function
  // pushes its real cx/cy here as it renders; the array is rebuilt fresh
  // every render since positions change with the data/chart size.
  const frontierPointsRegistry = useRef<
    Array<{ x: number; y: number; data: { return: number; volatility: number; label?: string } }>
  >([]);
  const [frontierHover, setFrontierHover] = useState<{
    pointX: number; pointY: number; data: { return: number; volatility: number; label?: string };
  } | null>(null);

  // Data window: either an explicit [start, end] range (end defaults to
  // today - "the latest close", since yfinance simply returns whatever's
  // most recently available up to that date), or a lookback duration
  // (years + days) anchored to the latest close.
  const [dateMode, setDateMode] = useState<DateMode>("lookback");
  const [rangeStart, setRangeStart] = useState(() => {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 3);
    return isoDate(d);
  });
  const [rangeEnd, setRangeEnd] = useState(() => isoDate(new Date()));
  const [lookbackYears, setLookbackYears] = useState(3);
  const [lookbackDays, setLookbackDays] = useState(0);

  const dateWindow = useMemo(() => {
    const portfolio_id = activeTab === "uploaded" ? uploadedPortfolioId : null;
    if (dateMode === "range") {
      return { portfolio_id, start: rangeStart, end: rangeEnd || null, demo_variant: demoVariant, demo_seed: demoSeed };
    }
    const end = new Date();
    const start = new Date(end);
    start.setFullYear(start.getFullYear() - lookbackYears);
    start.setDate(start.getDate() - lookbackDays);
    return { portfolio_id, start: isoDate(start), end: null, demo_variant: demoVariant, demo_seed: demoSeed };
  }, [
    dateMode, rangeStart, rangeEnd, lookbackYears, lookbackDays, activeTab, uploadedPortfolioId,
    demoVariant, demoSeed,
  ]);

  // universe.inception_dates only ever contains a ticker whose real listing
  // date fell inside the *previously fetched* window - so as long as the
  // window hasn't grown since that fetch, every entry in it is, by
  // construction, later than dateWindow.start (an ISO "YYYY-MM-DD" string,
  // so a plain string comparison is a valid date comparison). Sorted so the
  // single most-recently-listed holding is always first.
  const inceptionEntries = useMemo(
    () =>
      universe
        ? Object.entries(universe.inception_dates).sort(([, a], [, b]) => (a < b ? 1 : a > b ? -1 : 0))
        : [],
    [universe],
  );

  const [mode, setMode] = useState<Mode>("new");

  // Default to "existing" whenever the uploaded tab has a real portfolio to
  // rebalance - covers both a fresh upload and a rehydrated one from a
  // previous session (localStorage), not just handleUpload's own success
  // path. Only fires on those two triggers, so manually switching back to
  // "new" while staying on this tab doesn't get silently reverted.
  useEffect(() => {
    if (activeTab === "uploaded" && uploadedPortfolioId) setMode("existing");
  }, [activeTab, uploadedPortfolioId]);

  const [objectiveMode, setObjectiveMode] = useState<ObjectiveMode>("riskAversion");
  const [targetReturn, setTargetReturn] = useState(0.1);
  const [targetVolatility, setTargetVolatility] = useState(0.15);
  const [riskAversion, setRiskAversion] = useState(2.0);

  // Netted out of return before dividing by volatility for Sharpe - a plain
  // return/volatility ratio blows up near a near-cash portfolio (e.g. a
  // T-bill ETF at the low-vol end of the frontier) regardless of whether
  // there's any real compensation for risk. Seeded once on mount from a
  // live ^IRX quote (see fetchRiskFreeRate) rather than left at a stale
  // hardcoded guess - the user can still override it by typing a new value.
  const [riskFreeRate, setRiskFreeRate] = useState(0.045);
  function resetRiskFreeRate() {
    fetchRiskFreeRate()
      .then(({ risk_free_rate }) => setRiskFreeRate(risk_free_rate))
      .catch(() => {
        /* keep whatever's currently there - not worth surfacing an error
           for a metric-display convenience */
      });
  }
  useEffect(() => {
    let cancelled = false;
    fetchRiskFreeRate()
      .then(({ risk_free_rate }) => {
        if (!cancelled) setRiskFreeRate(risk_free_rate);
      })
      .catch(() => {
        /* keep the fallback default - not worth surfacing an error for a
           metric-display convenience */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const [wMin, setWMin] = useState(0.0);
  const [wMax, setWMax] = useState(1.0);
  // Per-group caps: one independent checkbox+slider per (category, group)
  // pair the current universe actually has group data for - "sector" for
  // the demo universe, country/currency/issuer/instrument_type for an
  // uploaded portfolio. Keyed by groupLimitKey(category, group) rather than
  // nested objects, since categories/groups are dynamic (not a fixed set).
  const [groupLimits, setGroupLimits] = useState<Record<string, { apply: boolean; maxWeight: number }>>({});

  const [currentWeights, setCurrentWeights] = useState<number[]>([]);
  const [capTradeCount, setCapTradeCount] = useState(false);
  const [maxTrades, setMaxTrades] = useState(2);
  const [capTurnover, setCapTurnover] = useState(false);
  const [turnoverMax, setTurnoverMax] = useState(0.3);

  // Opt-in diversification: re-solves for maximum entropy (spreads weight
  // across more names) subject to staying within this fraction of the
  // plain solve's own optimal objective value. Off by default - only takes
  // effect on the plain continuous paths (new-portfolio, or existing
  // without "Limit number of positions traded"/enable_flows), since it's
  // mathematically incompatible with those mixed-integer paths (see
  // OptimizeRequest.diversify_tolerance).
  const [enableDiversify, setEnableDiversify] = useState(false);
  const [diversifyTolerance, setDiversifyTolerance] = useState(0.02);

  // Buy/sell flows: opt-in alternative to the plain reallocation model
  // above - lets the portfolio's total value change via buys funded by new
  // capital or sells that are outright withdrawals, each independently
  // capped in count and per-trade amount, on top of ordinary reallocation
  // (re-capped separately here so enabling this doesn't change what
  // maxTrades/turnoverMax above already mean for the plain path).
  const [enableFlows, setEnableFlows] = useState(false);
  const [capReallocations, setCapReallocations] = useState(true);
  // 1 is never a real cap: a reallocation must net to zero across whichever
  // assets it touches, so a single active one is always forced to trade
  // exactly 0 - the default and the slider below both skip straight to 2.
  const [maxReallocations, setMaxReallocations] = useState(2);
  const [capReallocationAmount, setCapReallocationAmount] = useState(false);
  const [reallocationAmountMax, setReallocationAmountMax] = useState(0.1);
  const [capBuys, setCapBuys] = useState(true);
  const [maxBuys, setMaxBuys] = useState(1);
  const [capBuyAmount, setCapBuyAmount] = useState(false);
  const [buyAmountMax, setBuyAmountMax] = useState(0.1);
  const [capSells, setCapSells] = useState(true);
  const [maxSells, setMaxSells] = useState(1);
  const [capSellAmount, setCapSellAmount] = useState(false);
  const [sellAmountMax, setSellAmountMax] = useState(0.1);

  // Per-asset buy/sell caps, additive to the aggregate ones above - only
  // meaningful (and only shown) once flows are enabled. Keyed by ticker;
  // an entry with both fields undefined behaves as "no override" and is
  // pruned on toggle-off so a stale limit doesn't linger unseen.
  const [perAssetLimitsEnabled, setPerAssetLimitsEnabled] = useState(false);
  const [perAssetLimits, setPerAssetLimits] = useState<Record<string, { maxBuy?: number; maxSell?: number }>>({});

  const [result, setResult] = useState<OptimizeResponse | null>(null);
  const [solveError, setSolveError] = useState<string | null>(null);

  // The feasible return range under the *current* position/sector caps -
  // narrower than [min(mu), max(mu)] whenever those caps are active, since
  // e.g. a 25% position cap makes an individual asset's own extreme return
  // unreachable (that would need 100% in one name). Recomputed server-side
  // whenever those caps change, and the target-return slider is bounded by
  // this instead of the raw per-asset extremes.
  const [returnBounds, setReturnBounds] = useState<{
    min: number; max: number; minVolatility: number; maxVolatility: number;
    minGroupCaps: Record<string, Record<string, number>>;
  } | null>(null);
  const [frontierPoints, setFrontierPoints] = useState<FrontierPoint[]>([]);
  const [frontierLoading, setFrontierLoading] = useState(false);

  // Keep groupLimits in sync with whatever (category, group) pairs the
  // current universe actually has - adding new ones with sensible defaults,
  // dropping ones that no longer exist (e.g. after a date-window change
  // that reshapes the uploaded portfolio's surviving tickers).
  useEffect(() => {
    if (!universe) return;
    setGroupLimits((current) => {
      const next: typeof current = {};
      for (const [category, groups] of Object.entries(universe.group_indices)) {
        for (const group of Object.keys(groups)) {
          const key = groupLimitKey(category, group);
          next[key] = current[key] ?? { apply: false, maxWeight: 0.5 };
        }
      }
      return next;
    });
  }, [universe]);

  const groupLimitsPayload: GroupLimitRequest[] = useMemo(() => {
    if (!universe) return [];
    const payload: GroupLimitRequest[] = [];
    for (const [category, groups] of Object.entries(universe.group_indices)) {
      for (const group of Object.keys(groups)) {
        const entry = groupLimits[groupLimitKey(category, group)];
        if (entry) payload.push({ category, group, apply: entry.apply, max_weight: entry.maxWeight });
      }
    }
    return payload;
  }, [universe, groupLimits]);

  const perAssetBuyLimitsPayload = useMemo(() => {
    if (!perAssetLimitsEnabled) return {};
    const payload: Record<string, number> = {};
    for (const [ticker, limit] of Object.entries(perAssetLimits)) {
      if (limit.maxBuy != null) payload[ticker] = limit.maxBuy;
    }
    return payload;
  }, [perAssetLimitsEnabled, perAssetLimits]);

  const perAssetSellLimitsPayload = useMemo(() => {
    if (!perAssetLimitsEnabled) return {};
    const payload: Record<string, number> = {};
    for (const [ticker, limit] of Object.entries(perAssetLimits)) {
      if (limit.maxSell != null) payload[ticker] = limit.maxSell;
    }
    return payload;
  }, [perAssetLimitsEnabled, perAssetLimits]);

  // Memoized so their array identity stays stable across renders that don't
  // actually change the underlying data (e.g. a hover-only re-render from
  // moving the mouse over the frontier chart) - FrontierChart is wrapped in
  // React.memo, and relies on that stability to skip redoing its label
  // collision-avoidance layout and redrawing ~80 gradient segments on every
  // mousemove, which is what caused the labels to visibly lag while hovering.
  const frontierData = useMemo(
    () =>
      frontierPoints.map((p, i, arr) => ({
        volatility: p.volatility * 100,
        return: p.return * 100,
        color: viridisColor(arr.length > 1 ? i / (arr.length - 1) : 0),
      })),
    [frontierPoints],
  );
  const assetPoints = useMemo(
    () =>
      universe
        ? universe.tickers.map((ticker, i) => ({
            volatility: universe.volatilities[i] * 100,
            return: universe.mu[i] * 100,
            label: ticker,
          }))
        : [],
    [universe],
  );
  const currentPoint = useMemo(
    () =>
      result?.ok
        ? { volatility: result.volatility! * 100, return: result.expected_return! * 100, label: "Optimized portfolio" }
        : null,
    [result],
  );
  // The starting portfolio's own return/volatility (existing-portfolio mode
  // only) - plotted as a fixed reference point distinct from the optimizer's
  // result above, so the frontier chart shows "where you are" alongside
  // "where the optimizer would move you." The backend computes this from
  // current_weights independently of whether the solve itself finds a
  // feasible portfolio, so it stays populated even when result.ok is false.
  const initialPoint = useMemo(
    () =>
      result?.initial_return != null && result?.initial_volatility != null
        ? { volatility: result.initial_volatility * 100, return: result.initial_return * 100, label: "Initial portfolio" }
        : null,
    [result],
  );

  // Memoized so a slider drag that hasn't yet produced a new solve result
  // (the network call is debounced separately) doesn't hand the bar chart a
  // brand-new array identity on every keystroke - Recharts treats a new
  // array as new data and replays its enter animation, which is what made
  // dragging any slider feel laggy even before the actual re-solve returned.
  // result.weights is relative to the *original* capital base (so
  // new_total_value - e.g. 187.8% after a net buy - is meaningful); dividing
  // back through by it here converts to each asset's actual share of the
  // resulting book, which is what "Weight" should show. Without this, a buy
  // makes the bought asset's raw share jump while every other position's raw
  // share sits frozen at its old value even though it's now a smaller slice
  // of a bigger pie - and the column stops summing to 100%. Trades stay in
  // the original, undivided basis: "how much of your starting capital moved"
  // is the meaningful quantity there, not a share of the new total.
  const totalDeployed = result?.new_total_value || 1;
  const chartData = useMemo(
    () =>
      universe
        ? universe.labels.map((label, i) => ({
            label,
            weight: ((result?.weights?.[i] ?? 0) / totalDeployed) * 100,
            trade: result?.trades ? result.trades[i] * 100 : undefined,
          }))
        : [],
    [universe, result, totalDeployed],
  );

  async function handleUpload(file: File) {
    setUploading(true);
    setUploadError(null);
    try {
      const response = await uploadPortfolio(file, { start: dateWindow.start, end: dateWindow.end });
      setUploadedWeightsByTicker(
        Object.fromEntries(response.tickers.map((t, i) => [t, response.current_weights[i]])),
      );
      setUploadedInfo(response);
      setUploadedPortfolioId(response.portfolio_id);
      setMode("existing");
      // Matches were computed against the *previous* portfolio's own group
      // data - stale once a different file replaces it.
      setRiskLimitsInfo(null);
      setRiskLimitsError(null);
    } catch (err) {
      setUploadError(String(err));
    } finally {
      setUploading(false);
    }
  }

  async function handleUploadRiskLimits(file: File) {
    if (!uploadedPortfolioId) return;
    setRiskLimitsUploading(true);
    setRiskLimitsError(null);
    try {
      const response = await uploadRiskLimits(uploadedPortfolioId, file);
      setRiskLimitsInfo(response);
    } catch (err) {
      setRiskLimitsError(String(err));
    } finally {
      setRiskLimitsUploading(false);
    }
  }

  function applyMatchedRiskLimits() {
    if (!riskLimitsInfo) return;
    setGroupLimits((current) => {
      const next = { ...current };
      for (const limit of riskLimitsInfo.matched) {
        const key = groupLimitKey(limit.category, limit.group);
        next[key] = { apply: true, maxWeight: limit.max_weight };
      }
      return next;
    });
    // The workbook has no blanket per-position cap of its own (e.g. SGOV
    // carries no issuer limit at all - it's genuinely uncapped there) - the
    // real constraints are entirely the per-group caps just applied above.
    // The flat "max weight per position" slider is a separate UI convenience
    // with an arbitrary default (25%); left in place, it would silently
    // impose a limit the actual regulatory workbook never specified. Relax
    // it to 100% so the group caps are the only real limits in effect.
    setWMax(1.0);
  }

  // Keep localStorage in sync so a page refresh knows which portfolio/tab
  // to restore (the rehydration effect below does the actual restoring).
  useEffect(() => {
    if (uploadedPortfolioId) localStorage.setItem(UPLOADED_PORTFOLIO_ID_KEY, uploadedPortfolioId);
    else localStorage.removeItem(UPLOADED_PORTFOLIO_ID_KEY);
  }, [uploadedPortfolioId]);

  useEffect(() => {
    localStorage.setItem(ACTIVE_TAB_KEY, activeTab);
  }, [activeTab]);

  // On mount, if a previous session left an uploaded portfolio's id behind,
  // try to restore it - the backend persists uploaded portfolios (and any
  // matched risk limits) to disk now, so this works across both a page
  // refresh and a backend restart, not just a re-render. A 404 means the
  // backend genuinely has nothing for that id (e.g. its .data file was
  // removed) - fails quietly back to the normal "please upload" prompt
  // rather than surfacing an error for state the user never asked about.
  useEffect(() => {
    if (!uploadedPortfolioId) return;
    let cancelled = false;
    fetchUploadedPortfolio(uploadedPortfolioId)
      .then((response) => {
        if (cancelled) return;
        setUploadedWeightsByTicker(
          Object.fromEntries(response.tickers.map((t, i) => [t, response.current_weights[i]])),
        );
        setUploadedInfo(response);
      })
      .catch(() => {
        if (cancelled) return;
        setUploadedPortfolioId(null);
      });
    fetchUploadedRiskLimits(uploadedPortfolioId)
      .then((response) => {
        if (!cancelled) setRiskLimitsInfo(response);
      })
      .catch(() => {
        /* no risk-limits workbook uploaded for this portfolio yet - fine */
      });
    return () => {
      cancelled = true;
    };
    // Deliberately only on mount, using whatever uploadedPortfolioId
    // localStorage seeded the very first render with - a *new* upload
    // already sets uploadedInfo/riskLimitsInfo directly, it doesn't need
    // this refetch-by-id path.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const universeAbortRef = useRef<AbortController | null>(null);

  // Reload the universe whenever the data window changes. Debounced since
  // typing a lookback year/day is otherwise a request per keystroke - though
  // the NumberEditor's commit-on-blur already limits that in practice.
  useEffect(() => {
    const timer = setTimeout(() => {
      universeAbortRef.current?.abort();
      const controller = new AbortController();
      universeAbortRef.current = controller;

      fetchUniverse(dateWindow, controller.signal)
        .then((u) => {
          setUniverse(u);
          setLoadError(null);
          if (uploadedWeightsByTicker) {
            const raw = u.tickers.map((t) => uploadedWeightsByTicker[t] ?? 0);
            const total = raw.reduce((a, b) => a + b, 0);
            setCurrentWeights(total > 0 ? raw.map((w) => w / total) : new Array(u.labels.length).fill(1 / u.labels.length));
          } else {
            setCurrentWeights(new Array(u.labels.length).fill(1 / u.labels.length));
          }
        })
        .catch((err) => {
          if (err instanceof DOMException && err.name === "AbortError") return;
          setLoadError(String(err));
        });
    }, 250);

    return () => clearTimeout(timer);
  }, [dateWindow, uploadedWeightsByTicker]);

  const n = universe?.labels.length ?? 0;
  const equalShare = n > 0 ? 1 / n : 1;

  // Keep w_min/w_max valid if the universe size ever changes (n determines
  // 1/n, the real feasibility boundary for both).
  useEffect(() => {
    if (n === 0) return;
    setWMin((current) => clamp(current, 0, equalShare));
    setWMax((current) => clamp(current, equalShare, 1));
  }, [n, equalShare]);

  const holdingsTotal = currentWeights.reduce((a, b) => a + b, 0);
  const holdingsAreValid = Math.abs(holdingsTotal - 1) < 0.001;

  const canSolve = useMemo(() => {
    if (!universe) return false;
    if (mode === "existing") return holdingsTotal > 0;
    return true;
  }, [universe, mode, holdingsTotal]);

  const abortRef = useRef<AbortController | null>(null);

  async function solve(signal: AbortSignal) {
    if (!universe) return;
    setSolveError(null);
    try {
      const response = await runOptimize(
        {
          ...dateWindow,
          mode,
          target_return: objectiveMode === "targetReturn" ? targetReturn : null,
          target_volatility: objectiveMode === "targetVolatility" ? targetVolatility : null,
          risk_aversion: riskAversion,
          risk_free_rate: riskFreeRate,
          w_min: wMin,
          w_max: wMax,
          group_limits: groupLimitsPayload,
          current_weights: mode === "existing" ? currentWeights : null,
          max_trades: mode === "existing" && !enableFlows && capTradeCount ? maxTrades : null,
          turnover_max: mode === "existing" && capTurnover ? turnoverMax : null,
          enable_flows: mode === "existing" && enableFlows,
          max_reallocations: enableFlows && capReallocations ? maxReallocations : null,
          reallocation_amount_max: enableFlows && capReallocationAmount ? reallocationAmountMax : null,
          max_buys: enableFlows && capBuys ? maxBuys : null,
          buy_amount_max: enableFlows && capBuyAmount ? buyAmountMax : null,
          max_sells: enableFlows && capSells ? maxSells : null,
          sell_amount_max: enableFlows && capSellAmount ? sellAmountMax : null,
          per_asset_buy_limits: enableFlows ? perAssetBuyLimitsPayload : {},
          per_asset_sell_limits: enableFlows ? perAssetSellLimitsPayload : {},
          // Only meaningful on the plain continuous paths - the backend
          // silently ignores it under enable_flows/max_trades anyway, but
          // there's no point sending it (or showing its effect as if it
          // applied) on a path it mathematically can't take effect on.
          diversify_tolerance:
            enableDiversify && (mode === "new" || (!enableFlows && !capTradeCount))
              ? diversifyTolerance
              : null,
        },
        signal,
      );
      setResult(response);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setSolveError(String(err));
      setResult(null);
    }
  }

  // Re-solve whenever a control changes, once the universe is loaded.
  // Debounced so a dragged slider doesn't fire a request per pixel - only the
  // last value after a short pause triggers a solve, and any request already
  // in flight is aborted rather than left to race the new one.
  useEffect(() => {
    if (!universe || !canSolve) return;

    const timer = setTimeout(() => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      solve(controller.signal);
    }, 250);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    universe, dateWindow, mode, objectiveMode, targetReturn, targetVolatility, riskAversion, riskFreeRate, wMin, wMax,
    groupLimitsPayload, currentWeights, capTradeCount, maxTrades,
    capTurnover, turnoverMax,
    enableFlows, capReallocations, maxReallocations, capReallocationAmount, reallocationAmountMax,
    capBuys, maxBuys, capBuyAmount, buyAmountMax, capSells, maxSells, capSellAmount, sellAmountMax,
    perAssetBuyLimitsPayload, perAssetSellLimitsPayload,
    enableDiversify, diversifyTolerance,
  ]);

  const boundsAbortRef = useRef<AbortController | null>(null);

  // Recompute the feasible return range whenever the constraints it depends
  // on change, and snap the current target return into range if it falls
  // outside the new bounds (this is exactly what fixes "the slider's own
  // minimum is infeasible").
  useEffect(() => {
    if (!universe) return;

    const timer = setTimeout(() => {
      boundsAbortRef.current?.abort();
      const controller = new AbortController();
      boundsAbortRef.current = controller;

      const boundsParams = {
        ...dateWindow,
        w_min: wMin,
        w_max: wMax,
        group_limits: groupLimitsPayload,
      };

      fetchReturnBounds(boundsParams, controller.signal)
        .then(({ min_return, max_return, min_volatility, max_volatility, min_group_caps }) => {
          setReturnBounds({
            min: min_return, max: max_return,
            minVolatility: min_volatility, maxVolatility: max_volatility,
            minGroupCaps: min_group_caps,
          });
          setTargetReturn((current) => Math.min(Math.max(current, min_return), max_return));
          setTargetVolatility((current) => Math.min(Math.max(current, min_volatility), max_volatility));
          setGroupLimits((current) => {
            // Bail out with the *same* reference when nothing actually
            // changes - this result feeds groupLimitsPayload (useMemo),
            // which is this effect's own dependency. Always returning a
            // freshly-spread object here (even one with identical values)
            // makes React see a "new" groupLimits every time, which
            // recomputes groupLimitsPayload, which refires this very
            // effect - an infinite refetch loop that never lets
            // frontierLoading settle to false.
            let changed = false;
            const next = { ...current };
            for (const [category, groups] of Object.entries(min_group_caps)) {
              for (const [group, floor] of Object.entries(groups)) {
                const key = groupLimitKey(category, group);
                if (next[key] && floor > next[key].maxWeight) {
                  next[key] = { ...next[key], maxWeight: floor };
                  changed = true;
                }
              }
            }
            return changed ? next : current;
          });
        })
        .catch((err) => {
          if (err instanceof DOMException && err.name === "AbortError") return;
          setReturnBounds(null);
        });

      setFrontierLoading(true);
      fetchFrontier({ ...boundsParams, n_points: 80 }, controller.signal)
        .then(({ points }) => {
          setFrontierPoints(points);
          setFrontierLoading(false);
        })
        .catch((err) => {
          if (err instanceof DOMException && err.name === "AbortError") return;
          setFrontierPoints([]);
          setFrontierLoading(false);
        });
    }, 250);

    return () => clearTimeout(timer);
  }, [universe, dateWindow, wMin, wMax, groupLimitsPayload]);

  const tabBar = (
    <nav className="tab-bar">
      <button
        type="button"
        className={activeTab === "demo" ? "tab active" : "tab"}
        onClick={() => setActiveTab("demo")}
      >
        Demo Universe
      </button>
      <button
        type="button"
        className={activeTab === "uploaded" ? "tab active" : "tab"}
        onClick={() => setActiveTab("uploaded")}
      >
        {uploadedInfo ? `Uploaded Portfolio${uploadedInfo.report_date ? ` — ${uploadedInfo.report_date}` : ""}` : "Upload Portfolio"}
      </button>
    </nav>
  );

  if (activeTab === "uploaded" && !uploadedPortfolioId) {
    return (
      <div className="page">
        {tabBar}
        <div className="upload-page">
          <h2>Upload a portfolio</h2>
          <p className="hint">
            Upload an OSIP-format portfolio workbook (.xls) - one specific report layout this parser was built
            for, not a general spreadsheet importer - to optimize against real holdings instead of the demo
            universe. Position weights are derived from carrying value per ISIN; each instrument's own price
            history is then fetched to estimate return/risk.
          </p>
          <input
            type="file"
            accept=".xls"
            disabled={uploading}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleUpload(file);
            }}
          />
          {uploading && <p className="hint">Parsing the workbook and fetching price history...</p>}
          {uploadError && <div className="error-banner">{uploadError}</div>}
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="page">
        {tabBar}
        <div className="error-page">Could not load the asset universe: {loadError}</div>
      </div>
    );
  }
  if (!universe) {
    return (
      <div className="page">
        {tabBar}
        <div className="loading-page">Loading asset universe...</div>
      </div>
    );
  }

  const allVolatilities = [
    ...frontierData.map((p) => p.volatility),
    ...assetPoints.map((p) => p.volatility),
    ...(currentPoint ? [currentPoint.volatility] : []),
    ...(initialPoint ? [initialPoint.volatility] : []),
  ];
  // Both ends must be concrete numbers (not "auto"/"dataMax") and paired with
  // allowDataOverflow on the axis - otherwise Recharts "nice"-ifies the
  // domain to round tick values by default, which can silently override this
  // computed minimum back down to 0 depending on what step size it picks for
  // the other end of the range.
  const frontierXMin = allVolatilities.length > 0 ? Math.max(0, Math.floor(Math.min(...allVolatilities) - 2)) : 0;
  const frontierXMax = allVolatilities.length > 0 ? Math.ceil(Math.max(...allVolatilities) + 2) : 30;

  function handleFrontierMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best: (typeof frontierPointsRegistry.current)[number] | null = null;
    let bestDist = 20; // px - beyond this, show nothing rather than a far-off guess
    for (const p of frontierPointsRegistry.current) {
      const dist = Math.hypot(p.x - mx, p.y - my);
      if (dist < bestDist) {
        bestDist = dist;
        best = p;
      }
    }
    setFrontierHover(best ? { pointX: best.x, pointY: best.y, data: best.data } : null);
  }

  return (
    <div className="page">
      {tabBar}
      <div className="app">
      <aside className="sidebar">
        <h1>Portfolio Optimizer</h1>

        {activeTab === "uploaded" && uploadedInfo && (
          <section>
            <h2>Uploaded Portfolio</h2>
            <div className="info-card">
              <div className="stat-row">
                <span className="stat-value">{uploadedInfo.tickers.length}</span>
                <span className="stat-label">
                  instrument{uploadedInfo.tickers.length === 1 ? "" : "s"}
                  {uploadedInfo.report_date ? ` · as of ${uploadedInfo.report_date}` : ""}
                </span>
              </div>
              {uploadedInfo.skipped.length > 0 && (
                <p className="upload-status-warn">
                  ⚠ Skipped (no usable price history): {uploadedInfo.skipped.join(", ")}
                </p>
              )}
              {uploadedInfo.issues.length > 0 && (
                <details className="notes-disclosure">
                  <summary>
                    {uploadedInfo.issues.length} data-quality note{uploadedInfo.issues.length === 1 ? "" : "s"} from
                    the workbook
                  </summary>
                  <ul>
                    {uploadedInfo.issues.map((issue, i) => (
                      <li key={i}>{issue}</li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
            <label className={uploading ? "file-picker disabled" : "file-picker"}>
              <span aria-hidden="true">⤴</span> Upload a different file
              <input
                type="file"
                accept=".xls"
                disabled={uploading}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleUpload(file);
                }}
              />
            </label>
            {uploadError && <div className="error-banner">{uploadError}</div>}
          </section>
        )}

        {activeTab === "uploaded" && uploadedInfo && (
          <section>
            <h2>Regulatory Limits (TABYS)</h2>
            <p className="hint">
              Upload the "Отчет о соблюдении лимитов инвестирования" workbook to pull in the fund's real
              regulatory caps - matched against this portfolio's own holdings data, since the two reports
              don't always use the same labels for the same country/issuer.
            </p>
            <label className={riskLimitsUploading ? "file-picker disabled" : "file-picker"}>
              <span aria-hidden="true">⤴</span> {riskLimitsInfo ? "Upload a different file" : "Choose file"}
              <input
                type="file"
                accept=".xls"
                disabled={riskLimitsUploading}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleUploadRiskLimits(file);
                }}
              />
            </label>
            {riskLimitsUploading && <p className="hint">Parsing the risk-limits workbook...</p>}
            {riskLimitsError && <div className="error-banner">{riskLimitsError}</div>}
            {riskLimitsInfo && (
              <div className="info-card">
                <div className="limit-badges">
                  <span className="limit-badge positive">{riskLimitsInfo.matched.length} matched</span>
                  {riskLimitsInfo.unmatched.length > 0 && (
                    <span className="limit-badge muted">{riskLimitsInfo.unmatched.length} unmatched</span>
                  )}
                </div>
                {riskLimitsInfo.matched.length > 0 && (
                  <button type="button" className="btn-accent" onClick={applyMatchedRiskLimits}>
                    Apply matched limits
                  </button>
                )}
                {riskLimitsInfo.unmatched.length > 0 && (
                  <details className="notes-disclosure">
                    <summary>{riskLimitsInfo.unmatched.length} unmatched (not applied)</summary>
                    <ul>
                      {riskLimitsInfo.unmatched.map((limit, i) => (
                        <li key={i}>
                          {formatCategoryLabel(limit.category)} - {limit.group}: {(limit.max_weight * 100).toFixed(0)}%
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
                {riskLimitsInfo.unmapped_categories.length > 0 && (
                  <p className="hint">
                    Not tracked on uploaded portfolios at all: {riskLimitsInfo.unmapped_categories.join(", ")}.
                  </p>
                )}
              </div>
            )}
          </section>
        )}

        <section>
          <h2>Data Window</h2>
          <div className="mode-toggle">
            <button className={dateMode === "range" ? "active" : ""} onClick={() => setDateMode("range")}>
              Date range
            </button>
            <button className={dateMode === "lookback" ? "active" : ""} onClick={() => setDateMode("lookback")}>
              Lookback period
            </button>
          </div>

          {dateMode === "range" ? (
            <div className="date-range-fields">
              <label className="date-field">
                <span>Start</span>
                <input type="date" value={rangeStart} max={rangeEnd || undefined}
                  onChange={(e) => setRangeStart(e.target.value)} />
              </label>
              <label className="date-field">
                <span>End</span>
                <input type="date" value={rangeEnd} placeholder="Latest close"
                  onChange={(e) => setRangeEnd(e.target.value)} />
              </label>
            </div>
          ) : (
            <div className="date-range-fields">
              <div className="field">
                <div className="field-header">
                  <span className="field-label">Years</span>
                  <NumberEditor value={lookbackYears} min={0} max={30} decimals={0}
                    onCommit={(v) => setLookbackYears(Math.round(v))} />
                </div>
              </div>
              <div className="field">
                <div className="field-header">
                  <span className="field-label">Days</span>
                  <NumberEditor value={lookbackDays} min={0} max={364} decimals={0}
                    onCommit={(v) => setLookbackDays(Math.round(v))} />
                </div>
              </div>
            </div>
          )}
          <p className="hint">
            {dateMode === "range"
              ? "End defaults to today - yfinance simply returns data up to the latest available close."
              : `Using the ${lookbackYears}y ${lookbackDays}d of price history ending at the latest close.`}
          </p>
          {inceptionEntries.length > 0 && (
            <>
              <p className="hint">
                Most recently listed holding: <strong>{inceptionEntries[0][0]}</strong> (since{" "}
                {inceptionEntries[0][1]}).
              </p>
              <p className="upload-status-warn">
                Lookback reaches further back than {inceptionEntries.length === 1 ? "this holding's" : "these holdings'"} own
                history - {inceptionEntries.length === 1 ? "its" : "their"} return/risk is estimated from just the
                real data available: {inceptionEntries.map(([t, d]) => `${t} (since ${d})`).join(", ")}.
              </p>
            </>
          )}
        </section>

        {activeTab === "demo" && (
          <section>
            <h2>Demo Universe</h2>
            <div className="mode-toggle">
              <button
                className={demoVariant === "sectors" ? "active" : ""}
                onClick={() => setDemoVariant("sectors")}
              >
                Sector ETFs
              </button>
              <button
                className={demoVariant === "sp500_random" ? "active" : ""}
                onClick={() => setDemoVariant("sp500_random")}
              >
                Random 20 (S&amp;P 500)
              </button>
            </div>
            {demoVariant === "sectors" ? (
              <p className="hint">The 10 S&amp;P sector ETFs (XLK, XLF, XLV, ...).</p>
            ) : (
              <>
                <p className="hint">20 individual S&amp;P 500 names, randomly sampled from a curated pool.</p>
                <button
                  type="button"
                  className="holdings-normalize"
                  onClick={() => setDemoSeed(Math.floor(Math.random() * 1e9))}
                >
                  Reshuffle
                </button>
              </>
            )}
          </section>
        )}

        <section>
          <h2>Mode</h2>
          <div className="mode-toggle">
            <button className={mode === "new" ? "active" : ""} onClick={() => setMode("new")}>
              New portfolio
            </button>
            <button className={mode === "existing" ? "active" : ""} onClick={() => setMode("existing")}>
              Existing portfolio
            </button>
          </div>
          <p className="hint">
            {mode === "new"
              ? "Allocate from scratch - every position is free to move."
              : "Rebalance a current book - optionally cap how many names may trade."}
          </p>
        </section>

        <section>
          <h2>Objective</h2>
          <div className="mode-toggle">
            <button
              className={objectiveMode === "riskAversion" ? "active" : ""}
              onClick={() => setObjectiveMode("riskAversion")}
            >
              Risk aversion
            </button>
            <button
              className={objectiveMode === "targetReturn" ? "active" : ""}
              onClick={() => setObjectiveMode("targetReturn")}
            >
              Target return
            </button>
            <button
              className={objectiveMode === "targetVolatility" ? "active" : ""}
              onClick={() => setObjectiveMode("targetVolatility")}
            >
              Target volatility
            </button>
          </div>

          <div className="field">
            <div className="field-header">
              <span className="field-label">Risk-free rate (for Sharpe)</span>
              <NumberEditor
                value={riskFreeRate * 100}
                min={0}
                max={20}
                decimals={2}
                unit="%"
                onCommit={(v) => setRiskFreeRate(v / 100)}
              />
            </div>
          </div>
          <p className="hint">
            Seeded from the live 13-week T-bill yield (^IRX); edit freely, or{" "}
            <button type="button" className="link-button" onClick={resetRiskFreeRate}>
              reset to the live rate
            </button>
            .
          </p>

          {objectiveMode === "targetReturn" && (
            <>
              <SliderField
                label="Target annual return"
                value={targetReturn * 100}
                min={(returnBounds?.min ?? Math.min(...universe.mu)) * 100}
                max={(returnBounds?.max ?? Math.max(...universe.mu)) * 100}
                step={0.1}
                unit="%"
                onChange={(v) => setTargetReturn(v / 100)}
              />
              {returnBounds && (
                <p className="hint">
                  Feasible range under the current position/sector limits:{" "}
                  {(returnBounds.min * 100).toFixed(1)}% to {(returnBounds.max * 100).toFixed(1)}%.
                </p>
              )}
            </>
          )}

          {objectiveMode === "targetVolatility" && (
            <>
              <SliderField
                label="Target annual volatility"
                value={targetVolatility * 100}
                min={(returnBounds?.minVolatility ?? 0) * 100}
                max={(returnBounds?.maxVolatility ?? Math.max(...universe.volatilities)) * 100}
                step={0.1}
                unit="%"
                onChange={(v) => setTargetVolatility(v / 100)}
              />
              <p className="hint">Maximizes return under this volatility cap; a very high target may still come back infeasible.</p>
            </>
          )}

          {objectiveMode === "riskAversion" && (
            <div className="field">
              <div className="field-header">
                <span className="field-label">Risk aversion (lambda)</span>
                <NumberEditor value={riskAversion} min={0} max={20} decimals={1} onCommit={setRiskAversion} />
              </div>
              <div className="preset-pills">
                {RISK_AVERSION_PRESETS.map((p) => (
                  <button
                    key={p.label}
                    type="button"
                    className={riskAversion === p.value ? "preset-pill active" : "preset-pill"}
                    title={p.label}
                    onClick={() => setRiskAversion(p.value)}
                  >
                    {p.short} ({p.value})
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>

        <section>
          <h2>Diversification</h2>
          <ToggleRow
            label="Prefer diversified solutions"
            checked={enableDiversify}
            onChange={setEnableDiversify}
          />
          {enableDiversify && (
            <>
              <SliderField
                label="Tolerance below optimal"
                value={diversifyTolerance * 100}
                min={0}
                max={20}
                step={0.5}
                unit="%"
                onChange={(v) => setDiversifyTolerance(v / 100)}
              />
              {mode === "existing" && (enableFlows || capTradeCount) ? (
                <p className="hint hint-warning">
                  Not applied right now: {enableFlows ? "buy/sell flows are enabled" : "\"Limit number of positions traded\" is on"} -
                  {" "}both solve as a mixed-integer problem this technique can't combine with (see below). Turn
                  {enableFlows ? " that off" : " off the trade-count cap"} to actually see it take effect.
                </p>
              ) : (
                <p className="hint">
                  Two-stage solve: after finding the optimum, re-solves to maximize entropy (spread weight
                  across more names) subject to staying within this tolerance of it.
                </p>
              )}
            </>
          )}
        </section>

        <section>
          <h2>Position limits</h2>
          {/* A min above 1/n makes n * w_min > 1 (can't fund every position);
              a max below 1/n makes n * w_max < 1 (can't reach full
              allocation) - so those are the real bounds, not 0/100%. */}
          <SliderField label="Min weight per position" value={wMin * 100} min={0} max={equalShare * 100} step={1}
            decimals={2} unit="%" onChange={(v) => setWMin(v / 100)} />
          <SliderField label="Max weight per position" value={wMax * 100}
            min={Math.max(wMin, equalShare) * 100} max={100} step={1}
            decimals={2} unit="%" onChange={(v) => setWMax(v / 100)} />
        </section>

        {Object.entries(universe.group_indices).map(([category, groups]) => {
          // A category with only one group spans the entire universe by
          // definition - its floor is always exactly 100%, so a native
          // range input ends up with min===max, which browsers render at a
          // meaningless, undraggable position. Skip the interactive control
          // rather than show a slider that can never actually do anything.
          const isCappable = Object.keys(groups).length > 1;
          return (
            <section key={category}>
              <h2>{formatCategoryLabel(category)} limits</h2>
              {!isCappable && (
                <p className="hint">
                  Every holding falls into a single {formatCategoryLabel(category).toLowerCase()} group here
                  ({Object.keys(groups)[0]}), so capping it below 100% isn't meaningful.
                </p>
              )}
              {isCappable &&
                Object.entries(groups).map(([group, indices]) => {
                  const key = groupLimitKey(category, group);
                  const entry = groupLimits[key] ?? { apply: false, maxWeight: 0.5 };
                  return (
                    <ConstraintCard key={group} title={group} accent="var(--accent)">
                      <p className="hint group-limit-tickers">{indices.map((i) => universe.tickers[i]).join(", ")}</p>
                      <ToggleRow
                        label="Cap exposure"
                        checked={entry.apply}
                        onChange={(checked) =>
                          setGroupLimits((current) => ({
                            ...current,
                            [key]: { ...entry, apply: checked },
                          }))
                        }
                      />
                      {entry.apply && (
                        <SliderField
                          label={`Max weight - ${group}`}
                          value={entry.maxWeight * 100}
                          min={(returnBounds?.minGroupCaps[category]?.[group] ?? wMax) * 100}
                          max={100}
                          step={1}
                          decimals={2}
                          unit="%"
                          onChange={(v) =>
                            setGroupLimits((current) => ({
                              ...current,
                              [key]: { ...entry, maxWeight: v / 100 },
                            }))
                          }
                        />
                      )}
                    </ConstraintCard>
                  );
                })}
            </section>
          );
        })}

        {mode === "existing" && (
          <>
            <section>
              <h2>Current holdings</h2>
              <HoldingsEditor labels={universe.labels} weights={currentWeights} onChange={setCurrentWeights} />
              <div className={holdingsAreValid ? "holdings-total" : "holdings-total warn"}>
                <span>Total: {(holdingsTotal * 100).toFixed(1)}%</span>
                {!holdingsAreValid && (
                  <button
                    type="button"
                    className="holdings-normalize"
                    onClick={() => {
                      if (holdingsTotal <= 0) return;
                      setCurrentWeights(currentWeights.map((w) => w / holdingsTotal));
                    }}
                  >
                    Normalize to 100%
                  </button>
                )}
              </div>
              {!holdingsAreValid && (
                <p className="hint">
                  Weights don't need to sum to exactly 100% - they're normalized before solving - but the
                  ratios between them are what actually matters, so check they reflect your real holdings.
                </p>
              )}
            </section>

            <section>
              <h2>Rebalancing constraints</h2>
              <ToggleRow label="Enable buy/sell trading" checked={enableFlows} onChange={setEnableFlows} />
              <p className="hint">
                {enableFlows
                  ? "Buys (new capital) and sells (withdrawals) change the portfolio's total value, alongside ordinary reallocation - each capped independently below."
                  : "Off: pure reallocation only - total portfolio value stays fixed, every buy is funded by a sell elsewhere."}
              </p>

              {!enableFlows && (
                <ConstraintCard title="Reallocation" accent="var(--accent)">
                  <ToggleRow label="Limit number of positions traded" checked={capTradeCount} onChange={setCapTradeCount} />
                  {capTradeCount && (
                    <SliderField label="Max positions allowed to trade" value={maxTrades} min={1} max={n} step={1}
                      decimals={0} onChange={(v) => setMaxTrades(Math.round(v))} />
                  )}
                  <ToggleRow label="Also cap total turnover" checked={capTurnover} onChange={setCapTurnover} />
                  {capTurnover && (
                    <SliderField label="Max total turnover (sum |trades|)" value={turnoverMax} min={0} max={2} step={0.05}
                      decimals={2} onChange={setTurnoverMax} />
                  )}
                </ConstraintCard>
              )}

              {enableFlows && (
                <>
                  <ConstraintCard title="Overall turnover" accent="var(--text-muted)">
                    <ToggleRow label="Cap total turnover (reallocation + buys + sells)" checked={capTurnover} onChange={setCapTurnover} />
                    {capTurnover && (
                      <SliderField label="Max total turnover (sum |trades|)" value={turnoverMax} min={0} max={2} step={0.05}
                        decimals={2} onChange={setTurnoverMax} />
                    )}
                  </ConstraintCard>

                  <ConstraintCard title="Reallocation" accent="var(--accent)">
                    <ToggleRow label="Limit number of reallocations" checked={capReallocations} onChange={setCapReallocations} />
                    {capReallocations && (
                      <SliderField label="Max positions reallocated" value={maxReallocations} min={0} max={n} step={1}
                        decimals={0} onChange={(v) => {
                          const rounded = Math.round(v);
                          // 1 is unreachable: it can never produce a real trade (see the
                          // state declaration above), so dragging through it snaps to
                          // whichever real value (0 or 2) the drag was heading toward.
                          setMaxReallocations(rounded === 1 ? (v < 1 ? 0 : 2) : rounded);
                        }} />
                    )}
                    <ToggleRow label="Cap amount per reallocation" checked={capReallocationAmount} onChange={setCapReallocationAmount} />
                    {capReallocationAmount && (
                      <SliderField label="Max reallocation amount" value={reallocationAmountMax * 100} min={0} max={100}
                        step={1} decimals={2} unit="%" onChange={(v) => setReallocationAmountMax(v / 100)} />
                    )}
                  </ConstraintCard>

                  <ConstraintCard title="Buys (new capital)" accent="var(--positive)">
                    <ToggleRow label="Limit number of buys" checked={capBuys} onChange={setCapBuys} />
                    {capBuys && (
                      <SliderField label="Max positions bought" value={maxBuys} min={0} max={n} step={1}
                        decimals={0} onChange={(v) => setMaxBuys(Math.round(v))} />
                    )}
                    <ToggleRow label="Cap amount per buy" checked={capBuyAmount} onChange={setCapBuyAmount} />
                    {capBuyAmount && (
                      <SliderField label="Max buy amount" value={buyAmountMax * 100} min={0} max={100}
                        step={1} decimals={2} unit="%" onChange={(v) => setBuyAmountMax(v / 100)} />
                    )}
                  </ConstraintCard>

                  <ConstraintCard title="Sells (withdrawals)" accent="var(--negative)">
                    <ToggleRow label="Limit number of sells" checked={capSells} onChange={setCapSells} />
                    {capSells && (
                      <SliderField label="Max positions sold" value={maxSells} min={0} max={n} step={1}
                        decimals={0} onChange={(v) => setMaxSells(Math.round(v))} />
                    )}
                    <ToggleRow label="Cap amount per sell" checked={capSellAmount} onChange={setCapSellAmount} />
                    {capSellAmount && (
                      <SliderField label="Max sell amount" value={sellAmountMax * 100} min={0} max={100}
                        step={1} decimals={2} unit="%" onChange={(v) => setSellAmountMax(v / 100)} />
                    )}
                  </ConstraintCard>

                  <ConstraintCard title="Per-asset limits" accent="var(--text-muted)">
                    <ToggleRow
                      label="Set individual buy/sell limits per asset"
                      checked={perAssetLimitsEnabled}
                      onChange={setPerAssetLimitsEnabled}
                    />
                  </ConstraintCard>
                  {perAssetLimitsEnabled && universe && (
                    <table className="holdings-editor">
                      <thead>
                        <tr>
                          <th>Asset</th>
                          <th>Max buy %</th>
                          <th>Max sell %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {universe.tickers.map((ticker) => {
                          const limit = perAssetLimits[ticker] ?? {};
                          return (
                            <tr key={ticker}>
                              <td>{ticker}</td>
                              <td>
                                <input
                                  type="number" min={0} max={100} step={1}
                                  placeholder="agg."
                                  value={limit.maxBuy != null ? limit.maxBuy * 100 : ""}
                                  onChange={(e) => {
                                    const raw = e.target.value;
                                    setPerAssetLimits((current) => ({
                                      ...current,
                                      [ticker]: { ...limit, maxBuy: raw === "" ? undefined : Number(raw) / 100 },
                                    }));
                                  }}
                                />
                              </td>
                              <td>
                                <input
                                  type="number" min={0} max={100} step={1}
                                  placeholder="agg."
                                  value={limit.maxSell != null ? limit.maxSell * 100 : ""}
                                  onChange={(e) => {
                                    const raw = e.target.value;
                                    setPerAssetLimits((current) => ({
                                      ...current,
                                      [ticker]: { ...limit, maxSell: raw === "" ? undefined : Number(raw) / 100 },
                                    }));
                                  }}
                                />
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  )}
                </>
              )}
            </section>
          </>
        )}
      </aside>

      <main className="content">
        {solveError && <div className="error-banner">{solveError}</div>}

        {result && !result.ok && (
          <div className="error-banner">
            No feasible portfolio under these constraints - try loosening the weight/sector
            limits, the trade-count cap, or the target return.
          </div>
        )}

        {result?.ok && (
          <>
            <div className="metrics-row">
              <Metric label="Expected return" value={`${(result.expected_return! * 100).toFixed(2)}%`} />
              <Metric label="Volatility" value={`${(result.volatility! * 100).toFixed(2)}%`} />
              <Metric label="Sharpe" value={result.sharpe!.toFixed(2)} />
              <Metric label="Positions held" value={`${result.n_positions}`} />
              {result.effective_n != null && (
                <Metric label="Effective N" value={result.effective_n.toFixed(1)} />
              )}
              {result.initial_total_value != null && (
                <Metric label="Initial portfolio value" value={`${(result.initial_total_value * 100).toFixed(1)}%`} />
              )}
              {result.new_total_value != null && (
                <Metric label="New portfolio value" value={`${(result.new_total_value * 100).toFixed(1)}%`} />
              )}
              {result.n_reallocations != null || result.n_buys != null || result.n_sells != null ? (
                <>
                  {result.n_reallocations != null && (
                    <Metric label="Reallocated" value={`${result.n_reallocations}`} />
                  )}
                  {result.n_buys != null && <Metric label="Bought" value={`${result.n_buys}`} />}
                  {result.n_sells != null && <Metric label="Sold" value={`${result.n_sells}`} />}
                </>
              ) : (
                result.n_trades !== null && <Metric label="Positions traded" value={`${result.n_trades}`} />
              )}
            </div>

            {result.pre_diversify_return != null && (
              <div className="chart-card">
                <h3 className="chart-title">Diversification tradeoff</h3>
                <p className="hint">
                  What you gave up for a less concentrated portfolio - the pure optimum (before spreading
                  weight out) vs. what you actually got.
                </p>
                <table className="weights-table">
                  <thead>
                    <tr>
                      <th></th>
                      <th>Pure optimum</th>
                      <th>Diversified (returned)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Expected return</td>
                      <td>{(result.pre_diversify_return * 100).toFixed(2)}%</td>
                      <td>{(result.expected_return! * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td>Volatility</td>
                      <td>{(result.pre_diversify_volatility! * 100).toFixed(2)}%</td>
                      <td>{(result.volatility! * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td>Effective N</td>
                      <td>{result.pre_diversify_effective_n!.toFixed(1)}</td>
                      <td>{result.effective_n!.toFixed(1)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}

            <div className="chart-card">
              <div className="chart-title-row">
                <h3 className="chart-title">Efficient Frontier: Risk-Return Tradeoff</h3>
                {frontierLoading && (
                  <span className="chart-loading-label">
                    <span className="spinner" />
                    Recalculating…
                  </span>
                )}
              </div>
              <div
                className={frontierLoading ? "frontier-chart-wrap is-loading" : "frontier-chart-wrap"}
                onMouseMove={handleFrontierMouseMove}
                onMouseLeave={() => setFrontierHover(null)}
              >
                <FrontierChart
                  frontierData={frontierData}
                  assetPoints={assetPoints}
                  currentPoint={currentPoint}
                  initialPoint={initialPoint}
                  frontierXMin={frontierXMin}
                  frontierXMax={frontierXMax}
                  pointsRegistry={frontierPointsRegistry}
                />

                {frontierHover && (
                  <>
                    {/* Ring on the exact matched point - otherwise, on a
                        dense curve, there's no way to tell which point the
                        tooltip's numbers actually belong to. */}
                    <div
                      className="frontier-hover-ring"
                      style={{ left: frontierHover.pointX, top: frontierHover.pointY }}
                    />
                    <div
                      className="frontier-tooltip"
                      style={{ position: "absolute", left: frontierHover.pointX + 22, top: frontierHover.pointY + 22, pointerEvents: "none" }}
                    >
                      <div className="frontier-tooltip-title">{frontierHover.data.label ?? "Efficient frontier"}</div>
                      <div>Return: {frontierHover.data.return.toFixed(2)}%</div>
                      <div>Volatility: {frontierHover.data.volatility.toFixed(2)}%</div>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="chart-card">
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis dataKey="label" tick={{ fill: "var(--text-muted)", fontSize: 12 }} />
                  <YAxis tick={{ fill: "var(--text-muted)", fontSize: 12 }} unit="%" />
                  <Tooltip
                    formatter={(value) => `${Number(value ?? 0).toFixed(2)}%`}
                    contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }}
                    labelStyle={{ color: "var(--text)" }}
                    itemStyle={{ color: "var(--text)" }}
                  />
                  <Legend />
                  <Bar dataKey="weight" name="Weight" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                    {chartData.map((_, i) => (
                      <Cell key={i} fill={PALETTE[i % PALETTE.length]} />
                    ))}
                  </Bar>
                  {wMin > 0 && (
                    <ReferenceLine
                      y={(wMin / totalDeployed) * 100}
                      stroke="#5b8def"
                      strokeDasharray="6 4"
                      label={{ value: `Min ${(wMin * 100).toFixed(0)}%`, fill: "#5b8def", fontSize: 11, position: "insideBottomRight" }}
                    />
                  )}
                  {wMax < 1 && (
                    <ReferenceLine
                      y={(wMax / totalDeployed) * 100}
                      stroke="#e0554f"
                      strokeDasharray="6 4"
                      label={{ value: `Max ${(wMax * 100).toFixed(0)}%`, fill: "#e0554f", fontSize: 11, position: "insideTopRight" }}
                    />
                  )}
                </BarChart>
              </ResponsiveContainer>
            </div>

            <table className="weights-table">
              <thead>
                <tr>
                  <th>Asset</th>
                  <th>Weight</th>
                  {result.trades && <th>Trade</th>}
                  {result.trade_kinds && <th>Type</th>}
                </tr>
              </thead>
              <tbody>
                {universe.labels.map((label, i) => {
                  const rawAfter = result.weights?.[i] ?? 0;
                  // result.trades is raw capital moved (relative to the original
                  // book - see the memo above chartData/totalDeployed), so it's 0
                  // for any asset the solver didn't touch even when its actual
                  // share of the resulting portfolio shrank because *other*
                  // assets grew the total. Reconstructing the pre-trade raw value
                  // (weights minus trades - both still on the original-capital
                  // basis) and diffing it against the normalized post-trade share
                  // folds that passive dilution into the same number the Weight
                  // column already reflects, so the two columns reconcile: every
                  // row's starting % (whatever this asset's weight was before)
                  // plus this Trade equals the Weight shown, not just for assets
                  // the solver actively moved.
                  const rawBefore = rawAfter - (result.trades?.[i] ?? 0);
                  const displayTrade = result.trades ? rawAfter / totalDeployed - rawBefore : 0;
                  return (
                  <tr key={label}>
                    <td>{label}</td>
                    <td>{((rawAfter / totalDeployed) * 100).toFixed(2)}%</td>
                    {result.trades && <TradeCell trade={displayTrade} />}
                    {result.trade_kinds && (
                      <td className={`trade-kind trade-kind-${result.trade_kinds[i] ?? "none"}`}>
                        {result.trade_kinds[i] ?? "–"}
                      </td>
                    )}
                  </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        )}
      </main>
      </div>
    </div>
  );
}

/**
 * Greedily nudges each label straight down past any already-placed label it
 * would overlap, in data order. Takes real rendered pixel positions rather
 * than an estimated layout - critically, these must be the *exact same*
 * cx/cy Recharts used to draw each dot (see the "asset" registrations in
 * FrontierChart), not a position independently recomputed by Recharts'
 * own LabelList machinery: those two pixel-position computations can drift
 * apart under a dense/animated layout, which is what made labels visibly
 * detach from their dots. Sourcing both from one registry guarantees they
 * can't disagree.
 */
function layoutLabels(
  points: { x: number; y: number; text: string }[],
): { x: number; y: number; text: string }[] {
  const placed: { xStart: number; xEnd: number; y: number }[] = [];
  const lineHeight = 13;
  const dx = 8;

  return points.map(({ x, y, text }) => {
    const xStart = x + dx;
    const xEnd = xStart + text.length * 5.5 + 4;

    let placedY = y;
    for (let attempts = 0; attempts < 60; attempts += 1) {
      const collision = placed.find(
        (p) => xStart < p.xEnd && xEnd > p.xStart && Math.abs(placedY - p.y) < lineHeight,
      );
      if (!collision) break;
      placedY = collision.y + lineHeight;
    }

    placed.push({ xStart, xEnd, y: placedY });
    return { x: xStart, y: placedY, text };
  });
}

/** Renders asset ticker labels as a Recharts `Customized` overlay, so they
 * share the exact same SVG coordinate space as the dots drawn by the
 * Scatter shape callbacks - not a separately-computed LabelList layout. */
function AssetLabelsLayer({ positions }: { positions: { x: number; y: number; text: string }[] }) {
  return (
    <g>
      {positions.map((p, i) => (
        <text key={i} x={p.x} y={p.y} fontSize={10} fill="var(--text-muted)" dominantBaseline="middle">
          {p.text}
        </text>
      ))}
    </g>
  );
}

/**
 * Colors a trade by what it *displays* as, not its raw sign - a solver can
 * leave a position at -0.00001 (floating-point noise around "unchanged"),
 * which rounds to "0.00%" but would still read as red/green if colored off
 * the raw value. Neutral once the rounded 2-decimal percent is exactly zero.
 */
function TradeCell({ trade }: { trade: number }) {
  const pct = trade * 100;
  const rounded = pct.toFixed(2);
  const isZero = Number(rounded) === 0;
  const cls = isZero ? "" : pct > 0 ? "positive" : "negative";
  const sign = isZero || pct < 0 ? "" : "+";
  return (
    <td className={cls}>
      {sign}
      {rounded}%
    </td>
  );
}

type PointDatum = { return: number; volatility: number; label?: string };
type RegisteredPoint = { x: number; y: number; data: PointDatum; kind?: "asset" };

/**
 * The frontier chart's actual rendering, isolated behind React.memo so a
 * hover-only re-render of the parent (moving the mouse) doesn't force this
 * to redo its label collision-avoidance layout or redraw the ~80 gradient
 * segments - that recomputation on every mousemove was exactly what made
 * the labels visibly lag while hovering. Memo only pays off because the
 * parent passes these arrays in via useMemo, so their identity is stable
 * across those hover-only renders; without that this would re-render every
 * time regardless.
 *
 * pointsRegistry is a ref, not state - resetting and repopulating it here
 * (on every render *this component* actually does) is what backs the
 * parent's manual hit-testing. Since a hover-only re-render is fully
 * absorbed by memo (this component doesn't re-render at all), the registry
 * simply isn't touched then - which is correct, because the pixel positions
 * it holds haven't changed either.
 */
const FrontierChart = memo(function FrontierChart({
  frontierData, assetPoints, currentPoint, initialPoint, frontierXMin, frontierXMax, pointsRegistry,
}: {
  frontierData: Array<PointDatum & { color: string }>;
  assetPoints: PointDatum[];
  currentPoint: PointDatum | null;
  initialPoint: PointDatum | null;
  frontierXMin: number;
  frontierXMax: number;
  pointsRegistry: React.MutableRefObject<RegisteredPoint[]>;
}) {
  pointsRegistry.current = [];
  function register(data: PointDatum, cx?: number, cy?: number, kind?: "asset") {
    if (cx != null && cy != null) pointsRegistry.current.push({ x: cx, y: cy, data, kind });
  }

  const [assetLabelPositions, setAssetLabelPositions] = useState<
    { x: number; y: number; text: string }[]
  >([]);

  // Runs after every actual render of this component (memo means that's
  // only when frontierData/assetPoints/currentPoint/initialPoint really
  // changed) - the Scatter shape callbacks below have, by then, populated
  // pointsRegistry with each asset dot's real rendered cx/cy, so the label
  // layout is computed from the exact same coordinates the dots use.
  useEffect(() => {
    // Deduped by ticker: React.StrictMode double-invokes render in dev,
    // which can register the same dot into pointsRegistry twice before this
    // effect reads it - left undeduped, the collision-avoidance layout below
    // sees two identical points, decides they "collide" with each other, and
    // shoves the second one down a lineHeight, printing every ticker twice.
    const assetPixelPoints = Array.from(
      new Map(
        pointsRegistry.current
          .filter((p) => p.kind === "asset" && p.data.label)
          .map((p) => [p.data.label, { x: p.x, y: p.y, text: p.data.label! }]),
      ).values(),
    );
    setAssetLabelPositions(layoutLabels(assetPixelPoints));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frontierData, assetPoints, currentPoint, initialPoint, frontierXMin, frontierXMax]);

  return (
    <ResponsiveContainer width="100%" height={380}>
      <ComposedChart margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
        <XAxis
          type="number"
          dataKey="volatility"
          name="Volatility"
          unit="%"
          domain={[frontierXMin, frontierXMax]}
          allowDataOverflow
          tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          label={{ value: "Volatility (%)", position: "insideBottom", offset: -10, fill: "var(--text-muted)", fontSize: 12, fontWeight: 600 }}
        />
        <YAxis
          type="number"
          dataKey="return"
          name="Return"
          unit="%"
          tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          label={{ value: "Expected Return (%)", angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12, fontWeight: 600 }}
        />

        {/* Frontier visual: a viridis-gradient line, drawn as short
            multi-point Scatter+line segments so each still gets its own
            stroke color (a single Scatter can only have one color). Batched
            FRONTIER_SEGMENT_BATCH points per segment (with a 1-point overlap
            to keep the line connected) rather than one Scatter per adjacent
            pair - at n_points=80 that's ~80 live Recharts component
            instances recreated on every debounced slider/universe change vs.
            ~1 batch, which was measurably driving up render/GC pressure
            during an active tuning session without any visible loss of
            gradient smoothness. No Recharts Tooltip/hover anywhere in this
            chart at all - hover is handled by the parent's wrapping div via
            onMouseMove, doing real 2D nearest-point search over every
            point's actual rendered pixel position (registered below as each
            shape renders). That's what correctly handles the curve bending
            back on itself; Recharts' own per-series hover resolution kept
            picking the wrong branch. */}
        {Array.from(
          { length: Math.ceil((frontierData.length - 1) / FRONTIER_SEGMENT_BATCH) },
          (_, batchIndex) => {
            const start = batchIndex * FRONTIER_SEGMENT_BATCH;
            const end = Math.min(start + FRONTIER_SEGMENT_BATCH, frontierData.length - 1);
            const segment = frontierData.slice(start, end + 1);
            const color = segment[Math.floor(segment.length / 2)].color;
            return (
              <Scatter
                key={batchIndex}
                data={segment}
                line={{ stroke: color, strokeWidth: 4 }}
                shape={() => null}
                isAnimationActive={false}
                legendType="none"
              />
            );
          },
        )}

        <Scatter
          data={frontierData}
          shape={(props: { cx?: number; cy?: number; payload?: PointDatum }) => {
            if (props.payload) register(props.payload, props.cx, props.cy);
            return null;
          }}
          legendType="none"
          isAnimationActive={false}
        />

        {/* Individual assets: small grey dots labeled inline by name, the
            way the notebook annotates each point directly rather than
            relying on a legend. */}
        <Scatter
          data={assetPoints}
          shape={(props: { cx?: number; cy?: number; payload?: PointDatum }) => {
            if (props.payload) register(props.payload, props.cx, props.cy, "asset");
            return <circle cx={props.cx} cy={props.cy} r={5} fill="#8b93a1" stroke="var(--surface)" strokeWidth={1.5} />;
          }}
          legendType="none"
          isAnimationActive={false}
        />

        {/* The starting portfolio, before rebalancing - a hollow ring so it
            reads as "where you are" rather than a result, distinct from the
            solid dot below marking what the optimizer would move you to. */}
        {initialPoint && (
          <Scatter
            data={[initialPoint]}
            shape={(props: { cx?: number; cy?: number; payload?: PointDatum }) => {
              if (props.payload) register(props.payload, props.cx, props.cy);
              return <circle cx={props.cx} cy={props.cy} r={7} fill="none" stroke="#f2a93b" strokeWidth={3} />;
            }}
            legendType="none"
            isAnimationActive={false}
          />
        )}

        {currentPoint && (
          <Scatter
            data={[currentPoint]}
            shape={(props: { cx?: number; cy?: number; payload?: PointDatum }) => {
              if (props.payload) register(props.payload, props.cx, props.cy);
              return <circle cx={props.cx} cy={props.cy} r={8} fill="#e0554f" stroke="#fff" strokeWidth={2} />;
            }}
            legendType="none"
            isAnimationActive={false}
          />
        )}

        <Customized component={<AssetLabelsLayer positions={assetLabelPositions} />} />
      </ComposedChart>
    </ResponsiveContainer>
  );
});

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

/**
 * A pill-style switch, not a plain checkbox - the whole row (label + switch)
 * is one click target, following the "settings list" pattern (bordered
 * rows, label/description on the left, a switch on the right) common to
 * shadcn/ui and most native settings panels, adapted here to replace a
 * stack of identical `<label><input type="checkbox">` rows that read as
 * cluttered next to each other.
 */
function ToggleRow({
  label, checked, onChange,
}: {
  label: string; checked: boolean; onChange: (checked: boolean) => void;
}) {
  return (
    <label className="toggle-row">
      <span className="toggle-row-label">{label}</span>
      <span className={checked ? "toggle-switch on" : "toggle-switch"}>
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="toggle-switch-track" />
      </span>
    </label>
  );
}

/**
 * Groups one rebalancing-constraint category (reallocation/buys/sells) into
 * its own bordered card with a colored accent border, instead of a flat
 * ALL-CAPS label followed directly by more checkbox rows - gives each
 * category a clear visual boundary so three stacked categories' worth of
 * toggles+sliders don't read as one undifferentiated wall of controls.
 */
function ConstraintCard({
  title, accent, children,
}: {
  title: string; accent: string; children: React.ReactNode;
}) {
  return (
    <div className="constraint-card" style={{ borderLeftColor: accent }}>
      <div className="constraint-card-title">{title}</div>
      {children}
    </div>
  );
}

function SliderField({
  label, value, min, max, step, unit, decimals, onChange,
}: {
  label: string; value: number; min: number; max: number; step: number;
  unit?: string; decimals?: number; onChange: (value: number) => void;
}) {
  // A dragged native range input can fire far more 'input' events than the
  // screen can even paint (especially trackpad/high-poll-rate mice) - each
  // one used to flow straight into parent state, forcing the whole sidebar
  // + charts to re-render that many times per second. rafValue tracks the
  // slider's own visual position locally so dragging always feels
  // immediate; the parent's onChange (which drives the debounced solve/
  // frontier network calls) only fires once per animation frame at most.
  const [rafValue, setRafValue] = useState<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const pendingRef = useRef<number | null>(null);

  useEffect(() => () => {
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
  }, []);

  function handleChange(next: number) {
    setRafValue(next);
    pendingRef.current = next;
    if (rafRef.current != null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      if (pendingRef.current != null) onChange(pendingRef.current);
    });
  }

  const displayValue = rafValue ?? value;

  return (
    <div className="field">
      <div className="field-header">
        <span className="field-label">{label}</span>
        <NumberEditor
          value={displayValue}
          min={min}
          max={max}
          decimals={decimals ?? precisionFor(step)}
          unit={unit}
          onCommit={(v) => {
            setRafValue(null);
            onChange(v);
          }}
        />
      </div>
      <input
        type="range"
        className="field-slider"
        min={alignRangeMin(min, max, step)}
        max={max}
        step={step}
        value={Math.max(displayValue, alignRangeMin(min, max, step))}
        // A native range input with min>=max (the floor already equals the
        // ceiling - e.g. a group that covers the entire universe) has
        // nothing to drag between and renders at a meaningless, seemingly
        // "stuck" position; disable it rather than show a broken control.
        disabled={max <= min}
        onChange={(e) => handleChange(Number(e.target.value))}
        onPointerUp={() => setRafValue(null)}
      />
    </div>
  );
}

/**
 * A compact numeric field for precise typed entry. Holds the in-progress text
 * in local draft state so the displayed value only snaps to the committed
 * number on blur/Enter - typing "0.1" doesn't fight a re-render on every
 * keystroke, and Escape reverts an in-progress edit.
 */
function NumberEditor({
  value, min, max, decimals, unit, onCommit,
}: {
  value: number; min: number; max: number; decimals: number; unit?: string;
  onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const display = draft ?? formatValue(value, decimals);

  function commit() {
    if (draft === null) return;
    const parsed = Number.parseFloat(draft.replace(",", "."));
    setDraft(null);
    if (!Number.isFinite(parsed)) return;
    onCommit(clamp(parsed, min, max));
  }

  return (
    <div className="number-editor">
      <input
        type="text"
        inputMode="decimal"
        value={display}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
          else if (e.key === "Escape") {
            setDraft(null);
            e.currentTarget.blur();
          }
        }}
      />
      {unit && <span className="number-editor-unit">{unit}</span>}
    </div>
  );
}

function HoldingsEditor({
  labels, weights, onChange,
}: {
  labels: string[]; weights: number[]; onChange: (weights: number[]) => void;
}) {
  function setOne(index: number, value: number) {
    const next = [...weights];
    next[index] = value;
    onChange(next);
  }

  return (
    <table className="holdings-editor">
      <tbody>
        {labels.map((label, i) => (
          <tr key={label}>
            <td>{label}</td>
            <td>
              <input
                type="number"
                min={0}
                step={0.01}
                value={weights[i] ?? 0}
                onChange={(e) => setOne(i, Number(e.target.value))}
              />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
