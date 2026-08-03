"""
config.py — single source of truth for all parameters.
Edit this file to change any knob without touching algorithm code.
"""
from __future__ import annotations

# ── Instruments ────────────────────────────────────────────────────────────
INSTRUMENTS: dict[str, dict] = {
    "ES": {
        "tick_size": 0.25,
        "tick_value_usd": 12.50,     # $ per tick per contract
        "continuous_symbol": "ES.c.0",
    },
    "NQ": {
        "tick_size": 0.25,
        "tick_value_usd": 5.00,
        "continuous_symbol": "NQ.c.0",
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
SLIPPAGE_TICKS_ROUND_TRIP: int = 1       # total ticks lost on round-trip execution
# net_r = gross_r - (round_trip_cost_in_R)
# where round_trip_cost_ticks = slippage_ticks + 2*(commission / tick_value)

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

# ── Data / Paths ───────────────────────────────────────────────────────────
DATASET   = "GLBX.MDP3"
SCHEMA_1M = "ohlcv-1m"
SCHEMA_1D = "ohlcv-1d"
STYPE     = "continuous"

DOWNLOAD_START = "2019-01-01"   # new download window
DOWNLOAD_END   = "2022-12-31"

# Existing cached data (from prior Databento pull, already on disk)
EXISTING_DATA_ROOT = r"C:\Users\strik\Downloads\Backtesting Kit\DukascopyStudio\data\futures"

# Output paths (relative to project root)
DATA_DIR    = "data"
OUTPUTS_DIR = "outputs"
