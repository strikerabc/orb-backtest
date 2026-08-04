"""
config.py — single source of truth for all parameters.
Edit this file to change any knob without touching algorithm code.
"""
from __future__ import annotations

# ── Instruments ────────────────────────────────────────────────────────────
#
# Per-instrument fields
# ---------------------
# tick_size          : minimum price increment in native price units
# tick_value_usd     : dollar value of one tick per contract
# continuous_symbol  : Databento stype=continuous symbol for GLBX.MDP3
# sessions           : sessions where this instrument shows meaningful open volatility
# has_local_data     : True = merge Databento 2019-2022 with local Dukascopy 2023-2026
#                      False = download full range from Databento (no local backup)
# data_start         : earliest available / desired start date (ISO string)
# description        : human-readable label
# asset_class        : coarse grouping tag
#
# ZN price note: Databento normalises Treasury note prices to decimal points
# (e.g. 110.515625) via display_factor. tick_size = 1/64 = 0.015625 pts,
# tick_value = 0.015625 × $1,000 par-point = $15.625.
#
INSTRUMENTS: dict[str, dict] = {
    # ── Equity index futures ─────────────────────────────────────────────
    "ES": {
        "tick_size":         0.25,
        "tick_value_usd":    12.50,
        "continuous_symbol": "ES.c.0",
        "sessions":          ["NY", "LDN", "TOK"],
        "has_local_data":    True,
        "data_start":        "2019-01-01",
        "description":       "S&P 500 E-mini",
        "asset_class":       "equity_index",
    },
    "NQ": {
        "tick_size":         0.25,
        "tick_value_usd":    5.00,
        "continuous_symbol": "NQ.c.0",
        "sessions":          ["NY", "LDN", "TOK"],
        "has_local_data":    True,
        "data_start":        "2019-01-01",
        "description":       "Nasdaq-100 E-mini",
        "asset_class":       "equity_index",
    },
    "RTY": {
        "tick_size":         0.10,
        "tick_value_usd":    5.00,
        "continuous_symbol": "RTY.c.0",
        "sessions":          ["NY"],           # small-cap: US open only
        "has_local_data":    False,
        "data_start":        "2019-01-01",
        "description":       "Russell 2000 E-mini",
        "asset_class":       "equity_index",
    },
    # ── Metals ───────────────────────────────────────────────────────────
    "GC": {
        "tick_size":         0.10,
        "tick_value_usd":    10.00,            # 100 troy oz × $0.10
        "continuous_symbol": "GC.c.0",
        "sessions":          ["NY", "LDN", "TOK"],   # London gold fix proximity + Asian demand
        "has_local_data":    False,
        "data_start":        "2019-01-01",
        "description":       "Gold futures (100 troy oz)",
        "asset_class":       "metal",
    },
    # ── Energy ───────────────────────────────────────────────────────────
    "CL": {
        "tick_size":         0.01,
        "tick_value_usd":    10.00,            # 1,000 bbl × $0.01
        "continuous_symbol": "CL.c.0",
        "sessions":          ["NY", "LDN"],    # thin at Tokyo open
        "has_local_data":    False,
        "data_start":        "2019-01-01",
        "description":       "WTI Crude Oil (1,000 bbl)",
        "asset_class":       "energy",
    },
    # ── Fixed income ─────────────────────────────────────────────────────
    "ZN": {
        "tick_size":         0.015625,         # 1/64 of a point
        "tick_value_usd":    15.625,           # $1,000 × 1/64
        "continuous_symbol": "ZN.c.0",
        "sessions":          ["NY", "LDN"],    # negligible volatility at Tokyo open
        "has_local_data":    False,
        "data_start":        "2019-01-01",
        "description":       "10-Year Treasury Note",
        "asset_class":       "fixed_income",
    },
    # ── Crypto ───────────────────────────────────────────────────────────
    "BTC": {
        "tick_size":         5.00,
        "tick_value_usd":    25.00,            # 5 BTC × $5.00/BTC
        "continuous_symbol": "BTC.c.0",
        "sessions":          ["NY", "LDN", "TOK"],   # CME futures trade ~23h
        "has_local_data":    False,
        "data_start":        "2019-01-01",     # CME BTC launched Dec 2017; 2019 for data hygiene
        "description":       "Bitcoin futures (5 BTC)",
        "asset_class":       "crypto",
    },
    "ETH": {
        "tick_size":         0.05,
        "tick_value_usd":    2.50,             # 50 ETH × $0.05/ETH
        "continuous_symbol": "ETH.c.0",
        "sessions":          ["NY", "LDN", "TOK"],
        "has_local_data":    False,
        "data_start":        "2021-02-08",     # CME Ether futures launch date
        "description":       "Ether futures (50 ETH)",
        "asset_class":       "crypto",
    },
    # ── Forex ────────────────────────────────────────────────────────────
    # CME FX futures (GLBX.MDP3). Both quoted as USD per foreign unit.
    # Inverting to "USD/JPY" display requires 1/(6J price) — irrelevant
    # for the strategy; ORB trades the breakout direction symmetrically.
    #
    # Session rationale:
    #   6E: London open (08:00 BST) is the dominant EUR/USD event;
    #       NY open adds a second meaningful session.
    #       TOK excluded — 09:00 JST ≈ 00:00 UTC, dead zone for EUR.
    #   6J: Tokyo open (09:00 JST) is the primary JPY event (BoJ,
    #       Japanese data releases). All three sessions are relevant.
    "6E": {
        "tick_size":         0.00005,
        "tick_value_usd":    6.25,             # 125,000 EUR × $0.00005
        "continuous_symbol": "6E.c.0",
        "sessions":          ["LDN", "NY"],
        "has_local_data":    False,
        "data_start":        "2019-01-01",
        "description":       "Euro FX futures (125,000 EUR) — EUR/USD",
        "asset_class":       "forex",
    },
    "6J": {
        "tick_size":         0.0000005,
        "tick_value_usd":    6.25,             # 12,500,000 JPY × $0.0000005
        "continuous_symbol": "6J.c.0",
        "sessions":          ["TOK", "LDN", "NY"],
        "has_local_data":    False,
        "data_start":        "2019-01-01",
        "description":       "Japanese Yen futures (12,500,000 JPY) — USD/JPY inverse",
        "asset_class":       "forex",
    },
}

# ── Sessions ───────────────────────────────────────────────────────────────
# open/exit are (hour, minute) in LOCAL timezone.
# DST is handled automatically via tz-aware pandas operations.
#
# DST schedules that affect these sessions:
#   NY  (America/New_York) : 2nd Sun Mar → 1st Sun Nov  (+1h offset, ET stays the label)
#   LDN (Europe/London)    : last Sun Mar → last Sun Oct (+1h BST, session stays 08:00 local)
#   TOK (Asia/Tokyo)       : no DST — fixed JST UTC+9 year-round
#
# Consequence: NY-LDN overlap varies seasonally. All inter-session relationships
# should be computed in UTC, then expressed in local time for display only.
SESSIONS: dict[str, dict] = {
    "NY":  {"tz": "America/New_York", "open": (9, 30),  "exit": (12, 0)},
    "LDN": {"tz": "Europe/London",    "open": (8, 0),   "exit": (12, 0)},
    "TOK": {"tz": "Asia/Tokyo",       "open": (9, 0),   "exit": (12, 0)},
}
# Exit bar = the bar labeled "11:59" in local time (close = 12:00:00 local exactly).
EXIT_BAR_OFFSET_MINUTES: int = -1   # minutes before exit_hour:exit_min

# ── Opening Range Sizes ────────────────────────────────────────────────────
RANGE_MINUTES: list[int] = [5, 15, 30]

# Candle-close confirmation timeframes per range size (CC / R-CC variants).
# II, TI, R-II always operate on 1-minute bars regardless of range size.
CLOSURE_TFS: dict[int, list[int]] = {
    5:  [1, 5],
    15: [1, 5, 15],
    30: [1, 5, 15, 30],
}

# ── Risk-Reward Levels ─────────────────────────────────────────────────────
RR_LEVELS: list[float] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

# ── ATR Settings ───────────────────────────────────────────────────────────
ATR_PERIOD: int    = 14
ATR_BAR_MINUTES: int = 240            # 4-hour bars
ATR_ANCHOR_ET: tuple[int,int] = (18, 0)   # 18:00 ET = futures session open
ATR_CAP_MULTIPLE: float = 2.5

# ── Swing Detector ─────────────────────────────────────────────────────────
SWING_MAX_LOOKBACK_BARS: int  = 50    # per variant timeframe, may cross session boundary
SWING_MIN_SL_TICKS: int       = 4     # minimum stop distance in ticks (floor)

# ── Cost Model (controllable) ──────────────────────────────────────────────
# Set COMMISSION_PER_SIDE_USD = 0 and SLIPPAGE_TICKS = 0 for frictionless gross.
COMMISSION_PER_SIDE_USD: float = 2.50    # per contract per side (one-way)
SLIPPAGE_TICKS_ROUND_TRIP: int = 1       # fallback when spread_ticks absent
# net_r = gross_r - (round_trip_cost_in_R)
# where round_trip_cost_ticks = slippage_ticks + 2*(commission / tick_value)

# Per-instrument round-trip slippage in TICKS.
#
# Why this is not one global constant: a flat "1 tick" is honest only where the
# real bid-ask genuinely is one tick. Tick granularity varies enormously here.
# CME ETH has tick_size=0.05 on a ~$3,000 asset -- one tick is 0.0017% of price,
# so a flat 1 tick undercharges ETH by roughly an order of magnitude and made it
# the only instrument to "survive" costs in the first 10-instrument sweep.
#
# Values below are ESTIMATES of typical round-trip spread cost, not measured.
# ohlcv-1m carries no bid/ask, so measuring them requires the mbp-1 or tbbo
# schema (additional Databento spend). Treat as an assumption to be calibrated,
# and note that results for ETH/BTC are sensitive to it.
SLIPPAGE_TICKS_BY_SYMBOL: dict[str, float] = {
    "ES":  1.0,    # 1-tick market, extremely liquid
    "NQ":  1.0,
    "RTY": 1.0,
    "GC":  1.0,
    "CL":  1.0,
    "ZN":  1.0,    # 1/64 market, tight
    "6E":  1.0,
    "6J":  1.0,
    "BTC": 2.0,    # tick $5; spread often $5-25
    "ETH": 10.0,   # tick $0.05; spread commonly $0.50+ -> ~10 ticks
}

# Minimum fillable take-profit distance, in ticks.
#
# A TP closer than one tick to entry sits inside the spread and cannot fill,
# yet the exit walk records it as a win. This was inflating ZN at rr=0.25,
# where 44.4% of trades had sub-tick targets and win rate read 0.86.
# Trades below this threshold are flagged via `tp_unfillable` (kept, not
# dropped -- same convention as atr_exceeds_cap) so they can be filtered.
MIN_TP_TICKS: float = 1.0

# ── Regime Sampling ────────────────────────────────────────────────────────
REGIME_SEED: int            = 42
N_REGIMES: int              = 10
REGIME_WINDOW_MONTHS: int   = 6
# Strategy: divide full available history into N_REGIMES equal segments,
# randomly place one 6-month window within each segment. Guarantees
# temporal spread while remaining random within each segment.
HOLDOUT_MONTHS: int         = 3   # most-recent N months excluded from all windows

# ── Null Calibrator ────────────────────────────────────────────────────────
BOOTSTRAP_N: int            = 1000
BOOTSTRAP_BLOCK_SIZE_DAYS: int = 5    # block-bootstrap block length

# Random-entry null benchmark.
#
# The p-value must compare LIKE WITH LIKE: an observed MEAN against a
# distribution of null MEANS. The original implementation compared the
# observed mean against a pool of INDIVIDUAL random-trade outcomes, which
# made the statistic track the random-entry TP hit rate instead of tail
# probability (corr with win_rate = +0.99, corr with expectancy = -0.10),
# putting a hard floor near 1/(1+rr) so p < 0.05 was unreachable.
#
# Corrected design: draw N_NULL_DRAWS_PER_DAY random entries per session-day
# to build a null trade pool, then block-bootstrap that pool at the observed
# variant's sample size to obtain a distribution of null means.
N_NULL_DRAWS_PER_DAY: int   = 3    # random entries generated per session-day
NULL_SAMPLE_DAYS: int       = 500  # session-days sampled per null pool
NULL_BOOTSTRAP_N: int       = 1000 # resamples when building null-mean dist

# ── Reporting ──────────────────────────────────────────────────────────────
# Minimum trades before a variant may appear in a RANKED table.
#
# Without this, variants that fired once and won sort to the top with
# win_rate=1.0, profit_factor=9999 (stats.py: no losses -> sentinel) and a
# degenerate CI where ci_lo == ci_hi (stats.py: len < block -> point estimate
# twice). Those are arithmetic artefacts of tiny samples, not edges.
#
# 100 keeps ~76% of variants in the current sweep and gives a standard error
# near +/-0.1 R for typical per-trade R dispersion. All rows are still written
# to summary.parquet/csv in full -- the filter applies only to ranked output.
MIN_TRADES_FOR_RANKING: int = 100

# ── Data / Paths ───────────────────────────────────────────────────────────
DATASET   = "GLBX.MDP3"
SCHEMA_1M = "ohlcv-1m"
SCHEMA_1D = "ohlcv-1d"
STYPE     = "continuous"

DOWNLOAD_START = "2019-01-01"   # new download window
DOWNLOAD_END   = "2022-12-31"

# Days to back off from "now" when a download/cost window ends at the present.
#
# GLBX.MDP3 historical access has a licence boundary near real time. Passing
# end = today returns:
#   422 dataset_unavailable_range -- "requires a subscription and/or license
#   to access. Try again with an end time before <today>T<HH:MM>Z"
# Databento's `end` is exclusive, so backing off 2 days keeps requests inside
# the historical window at the cost of the two most recent sessions.
DOWNLOAD_END_BUFFER_DAYS: int = 2

# Existing cached data (from prior Databento pull, already on disk)
EXISTING_DATA_ROOT = r"C:\Users\strik\Downloads\Backtesting Kit\DukascopyStudio\data\futures"

# Output paths (relative to project root)
DATA_DIR    = "data"
OUTPUTS_DIR = "outputs"
