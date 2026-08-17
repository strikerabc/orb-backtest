# WORK ORDER — Cross-Asset Standardisation of the Volatility-Regime ORB Strategy

**Repository:** `strikerabc/orb-backtest`
**Status of this document:** Implementation specification. Tasks are ordered; do not
skip ahead. Task 0 is a gate — if it fails, stop and report rather than proceeding.
**Prime directive:** The cross-asset extension is only informative if the rule set is
frozen before touching new-instrument data. Any per-instrument recalibration converts
this from an out-of-sample test into another selection exercise, which the repo's own
holdout verdict (`NO EDGE ESTABLISHED`, 38.3% net-positive OOS vs. 50% by chance)
already tells us how to interpret.

---

## 0. Context the agent must internalise before writing code

The repo's statistical machinery exists because exhaustive sweeps manufacture ~5%
spurious winners by construction. The prescribed reading order — **holdout verdict
first, breadth second, ranked tables last** — applies to every result produced under
this work order. The NQ configuration in `STANDARDISATION_METHOD.md` (CC/5/5,
three-barrier exits, 90-pt regime switch, Sharpe 1.188, +0.0918 R/trade) has, as far
as the committed evidence shows, **not** been passed through the repo's gates: matched-stop
null calibration, Westfall–Young maxT, breadth, holdout. It carries at least six
fitted parameters selected on the full 2019–2026 sample. Treat it as a hypothesis,
not a result.

Two bookkeeping notes. First, the entry-mode taxonomy in `STANDARDISATION_METHOD.md`
(CC/OC/CB/BC/BB) does not match the repo's (`II/CC/TI/R-II/R-CC` in
`entry_detector.py`). Second, the committed artifacts were produced with
`HOLDOUT_PIN_START = "2026-02-01"` while `main` defaults to `None`; any run intended
to be compared against committed evidence must set the pin.

---

## Task 0 — Reconciliation and gate check (blocking)

1. Map the CC/OC/CB/BC/BB taxonomy onto `entry_detector.py`'s state machines.
   Determine whether the tested configuration is expressible in the existing repo
   (likely: repo-`CC` with 5m range / 5m closure TF) or whether the agents forked
   the entry logic. If forked, port the configuration back into the repo so it flows
   through `journal.py`, `filters.py`, and `stats.py` — the single sources of truth.
2. Run the NQ CC/5/5 three-barrier configuration through the **existing** gate stack
   with `HOLDOUT_PIN_START = "2026-02-01"`: matched-stop null (`null_calibrator.py`),
   maxT + permutation FDR (`multiplicity.py`), breadth, and `tools/holdout_test.py`.
3. **Gate:** if the configuration fails holdout on NQ, report that and stop. There is
   nothing to standardise across asset classes if the source instrument's result is a
   selection artefact.

---

## Task 1 — Replace the 90-point threshold with a dimensionless regime variable

### 1.1 Why

"90 points" has units. It bundles NQ's price level (~7,000 → ~21,000 over the
sample), its 2019–2026 volatility regime, and its point value. It cannot mean the
same thing on CL (ATR ~ $1.50), ZN (ATR in 32nds), or 6E (ATR in pips) — and it did
not even mean the same thing on NQ-2019 as on NQ-2026. The trailing distances
(0.75× / 1.0× ATR) are already dimensionless and transfer as-is. Only the regime
**classifier** needs replacing.

### 1.2 The change

Express current volatility relative to the instrument's own trailing baseline:

```python
# src/range_builder.py (or data_layer.py enrichment step — wherever atr_4h
# is currently attached to SessionDay)

def attach_regime_ratio(session_days: list, lookback: int, stat: str = "median") -> None:
    """
    Attach a dimensionless volatility-regime ratio to each SessionDay.

    Baseline is the rolling `stat` of the 4h ATR over the prior `lookback`
    sessions, shifted by one session so the current day's own ATR never
    contributes to its own baseline. No lookahead by construction.
    """
    ordered = sorted(session_days, key=lambda sd: sd.date)
    atr = pd.Series(
        [sd.atr_4h for sd in ordered],
        index=[sd.date for sd in ordered],
    )
    baseline = (
        atr.rolling(lookback, min_periods=lookback)
           .median()          # median: spike-resistant; do not use mean
           .shift(1)          # strictly-prior data only
    )
    for sd in ordered:
        b = baseline.get(sd.date, float("nan"))
        sd.atr_baseline = None if pd.isna(b) else float(b)
        sd.atr_ratio = None if pd.isna(b) else sd.atr_4h / float(b)
        sd.regime_warmup = pd.isna(b)
```

```python
# src/config.py — new tunables. Provisional values; frozen after Task 2, then
# never modified. All are committed to prereg/ before any non-NQ data is run.

REGIME_BASELINE_LOOKBACK = 252       # sessions (~1 trading year)
REGIME_BASELINE_STAT     = "median"
REGIME_THRESHOLD         = 1.0       # dimensionless; see Task 2 freezing procedure
TRAIL_MULT_LOW           = 0.75      # atr_ratio <  REGIME_THRESHOLD
TRAIL_MULT_HIGH          = 1.00      # atr_ratio >= REGIME_THRESHOLD

BREAKEVEN_TRIGGER_R      = 0.2       # stop -> entry when trade reaches +0.2R
TRAIL_ACTIVATE_R         = 0.5       # trail arms at +0.5R
INITIAL_STOP_R           = 1.0
FIXED_TARGET_R           = 1.0       # cancelled once breakeven arms

FRICTION_FLOOR_MULT      = 8.0       # min 1R value in round-trip-friction units
```

```python
# src/trade_sim.py — regime lookup used inside the exit walk

def get_trail_multiplier(session_day) -> float | None:
    """Returns None during baseline warm-up; caller flags and handles."""
    if session_day.regime_warmup:
        return None
    return (config.TRAIL_MULT_LOW
            if session_day.atr_ratio < config.REGIME_THRESHOLD
            else config.TRAIL_MULT_HIGH)
```

A ratio of 1.0 means "at this instrument's own trailing median volatility" —
identically for NQ, ES, CL, GC, 6E, and ZN. No lookup table, no per-family
pre-calibration, no threshold that silently re-embeds an NQ fit.

### 1.3 Warm-up handling

The first `REGIME_BASELINE_LOOKBACK` sessions of every instrument have no baseline.
**Exclude these sessions from all analysis arms** (see Task 4) so that every arm is
evaluated on an identical trade population — do not let the adaptive arm quietly run
on fewer trades than the static arms. With 7.6 years per instrument, one year of
warm-up is cheap. Journal them with `regime_warmup = True` rather than deleting them,
consistent with the repo's flag-and-filter-at-analysis philosophy (`atr_exceeds_cap`
precedent).

---

## Task 2 — Freeze the threshold constant (one-time, NQ-only, then locked)

1. Compute `atr_ratio` for every historical NQ CC/5/5 entry.
2. Find the ratio value `r*` that maximises classification agreement with the legacy
   90-point rule over those entries (equivalently: the ratio at the same entry-count
   percentile that 90 points occupied).
3. **Coarsen it.** Round to the nearest 0.05. If `r*` lands anywhere in
   [0.90, 1.10], set `REGIME_THRESHOLD = 1.0` and be done. A round,
   theory-motivated constant ("above or below its own median") is more defensible
   than 0.937, and every decimal of NQ-fitted precision carried forward is a decimal
   of overfitting exported to five other instruments.
4. Record `r*`, the coarsened value, and the agreement rate in
   `prereg/PREREG_CROSS_ASSET_REGIME.md` (Task 6). After this commit the constant is
   immutable for the entire study.

---

## Task 3 — Instrument specifications and friction floor

### 3.1 Instrument specs

```python
# src/config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    asset_class: str
    tick_size: float
    tick_value: float           # $ per tick per contract
    typical_spread_ticks: float # MEASURED via tools/costs/, never assumed
    commission_rt: float        # $ round trip
    session_key: str            # key into SESSION_ANCHORS

SESSION_ANCHORS = {
    # Anchors are hypotheses about where each asset class's daily liquidity
    # event sits. They are PRE-REGISTERED PER ASSET CLASS before any results
    # are viewed. They are not tunables. Do not sweep them.
    "equity_ny": ("09:30", "America/New_York"),
    "energy_ny": ("09:00", "America/New_York"),   # CL pit-open convention
    "metals_ny": ("08:20", "America/New_York"),   # COMEX pit-open convention
    "rates_ny":  ("08:30", "America/New_York"),   # US data-release window
    "fx_london": ("08:00", "Europe/London"),      # existing London session
}

INSTRUMENTS = [
    #              symbol class     tick      tick$    spread  comm   session
    InstrumentSpec("NQ", "equity", 0.25,     5.00,    None,   None,  "equity_ny"),
    InstrumentSpec("ES", "equity", 0.25,     12.50,   None,   None,  "equity_ny"),
    InstrumentSpec("CL", "energy", 0.01,     10.00,   None,   None,  "energy_ny"),
    InstrumentSpec("GC", "metals", 0.10,     10.00,   None,   None,  "metals_ny"),
    InstrumentSpec("6E", "fx",     0.00005,  6.25,    None,   None,  "fx_london"),
    InstrumentSpec("ZN", "rates",  0.015625, 15.625,  None,   None,  "rates_ny"),
]
# `None` fields are filled by tools/costs/ measurement runs BEFORE backtesting,
# and the measured values are committed to prereg/.
```

Exit time remains the existing convention: the 11:59 **local** bar of the anchored
session, closing at exactly 12:00:00 local.

### 3.2 Friction floor

On NQ a 1R stop dwarfs the spread. On ZN or 6E the opening range can be a handful of
ticks, and a strategy with an ~11% win rate whose expectancy lives in rare large
winners is exquisitely sensitive to friction. Without this floor, rates and FX will
produce fiction in one direction or the other.

```python
# src/filters.py — add alongside existing eligibility logic (single source of truth)

def friction_floor_fail(trade, spec: InstrumentSpec) -> bool:
    """
    Flag trades whose 1R dollar value is too small relative to round-trip
    friction for the R-arithmetic to be meaningful. Flagged trades are
    SIMULATED ANYWAY and filtered at analysis time (atr_exceeds_cap pattern).
    """
    stop_ticks = trade.stop_distance_points / spec.tick_size
    stop_value = stop_ticks * spec.tick_value
    round_trip = spec.typical_spread_ticks * spec.tick_value + spec.commission_rt
    return stop_value < config.FRICTION_FLOOR_MULT * round_trip
```

Journal field: `friction_floor_fail: bool`, plus `friction_ratio: float`
(= stop_value / round_trip) so the floor's sensitivity can be examined post hoc
without re-running.

---

## Task 4 — Three-barrier exit walk with regime-adaptive trail

Extend (do not fork) the bar-by-bar walk in `trade_sim.py`. Conventions below either
follow the repo's existing rules or must be documented as new decisions in `docs/`.

```python
# src/trade_sim.py — sketch of the modified exit walk. Long side shown;
# short is the mirror. `r` is the initial stop distance in points.

def walk_exits(trade, bars_1m, spec, trail_mult: float):
    entry, r, sign = trade.entry_price, trade.r_dist, trade.sign
    stop   = entry - sign * r * config.INITIAL_STOP_R
    target = entry + sign * r * config.FIXED_TARGET_R
    be_armed = trail_armed = False

    for bar in bars_1m:                      # through the 11:59 local bar
        # --- exits first, conservative ordering (existing repo convention) ---
        stop_hit   = bar_touches(bar, stop, sign, direction="against")
        target_hit = target is not None and bar_touches(bar, target, sign, "with")
        if stop_hit:
            fill = worst_of(stop, bar.open, sign)   # gap-through fills at open
            return make_exit(trade, bar, fill,
                             reason="stop" if not be_armed else "breakeven_or_trail",
                             same_bar_ambiguous=target_hit,
                             exit_slippage_r=(fill - stop) * sign / r)
        if target_hit:
            return make_exit(trade, bar, target, reason="target")

        # --- state transitions on bar CLOSE only (pessimistic: no intra-bar
        #     path assumptions; document in docs/) ---
        excursion_r = sign * (bar.close - entry) / r
        if not be_armed and excursion_r >= config.BREAKEVEN_TRIGGER_R:
            be_armed = True
            stop = favorable(stop, entry, sign)     # never loosens
            target = None                           # fixed TP cancelled
        if be_armed and not trail_armed and excursion_r >= config.TRAIL_ACTIVATE_R:
            trail_armed = True
        if trail_armed:
            candidate = bar.close - sign * trail_mult * trade.atr_4h
            stop = favorable(stop, candidate, sign) # monotone ratchet

    return make_exit(trade, bars_1m[-1], bars_1m[-1].close, reason="session_close")
```

**Correction to the source document that must be journaled, not hidden:** the claim
"once breakeven hit, worst exit = 0R (mathematically guaranteed)" is false. A 1m bar
can gap through the breakeven stop (fill at `bar.open`, worse than entry), and
friction makes even a perfect breakeven fill net-negative. The sim must model both;
`exit_slippage_r` and cost-inclusive net R go in the journal. If the NQ Sharpe of
1.188 was computed under the guaranteed-0R assumption, expect it to degrade on
re-simulation — report the delta explicitly.

New journal columns (one row per (signal, trade, day), as now): `atr_ratio`,
`atr_baseline`, `regime` (`low`/`high`/`warmup`), `trail_mult`, `be_armed`,
`trail_armed`, `exit_reason`, `exit_slippage_r`, `friction_floor_fail`,
`friction_ratio`.

---

## Task 5 — Three-arm evaluation design

Every instrument runs three arms on the **identical** trade population (same entries,
same warm-up exclusion, same friction flags):

| Arm | Trail distance |
|---|---|
| A — static tight | 0.75 × ATR always |
| B — static wide | 1.00 × ATR always |
| C — regime-adaptive | 0.75×/1.00× via frozen `REGIME_THRESHOLD` |

The regime switch survives only if arm C beats **both** static arms in pooled
cross-asset trade-weighted R with a bootstrap CI excluding zero. If it doesn't, the
switch was fit to NQ noise; delete it and carry the better static arm forward — one
fewer parameter is a feature, not a loss.

Multiplicity is deliberately tiny here: 3 arms × 6 instruments = 18 families through
`multiplicity.py`, not another thousand-variant sweep. Do not widen it. Route
everything through the existing gate stack: matched-stop null per instrument
(`null_calibrator.py` — the matched-stop design already isolates entry timing from
stop width, which is exactly what's needed since the arms differ only in exits),
breadth, maxT/FDR across the 18 families, and holdout via `tools/holdout_test.py`
with the pin set. The cross-asset ensemble is itself the strongest holdout: a real
edge should be net-positive on a clear majority of five unrelated instruments, not
just the one it was born on.

---

## Task 6 — Pre-registration (commit BEFORE running any non-NQ backtest)

Create `prereg/PREREG_CROSS_ASSET_REGIME.md`. The repo's own principle applies: a
pre-registration absent from history has no evidential value. Draft:

```markdown
# Pre-registration: Cross-Asset Transfer of the Regime-Adaptive ORB Strategy
Committed: <date>   Commit: <hash>   HOLDOUT_PIN_START: 2026-02-01

## Hypotheses
H1 (transfer): The frozen NQ rule set, with the dimensionless regime variable,
    has positive net expectancy out-of-sample on instruments it was never
    fitted to.
H2 (regime switch): Regime-adaptive trailing (arm C) outperforms both static
    arms (A, B) pooled across instruments.

## Frozen parameters (immutable after this commit)
| Parameter | Value | Provenance |
|---|---|---|
| Entry family              | CC, 5m range, 5m closure | NQ config under test |
| Initial stop              | 1.0R                     | NQ config |
| Breakeven trigger         | +0.2R                    | NQ config |
| Trail activation          | +0.5R                    | NQ config |
| Trail multipliers         | 0.75× / 1.00× ATR(4h)    | NQ config |
| REGIME_THRESHOLD          | <value from Task 2>      | NQ equivalence, coarsened |
| REGIME_BASELINE_LOOKBACK  | 252 sessions, median     | theory, not fitted |
| FRICTION_FLOOR_MULT       | 8.0                      | theory, not fitted |
| Session anchors           | table in config.py       | asset-class convention |
| Measured frictions        | <filled from tools/costs/ before backtests> | measured |

## Instruments
ES (equity, near-replication check), CL (energy), GC (metals), 6E (FX),
ZN (rates). NQ is the source instrument and does not count toward H1.

## Primary endpoints & decision rules
H1 SUPPORTED iff: >= 4 of 5 instruments show net-positive holdout expectancy
  (friction-flagged and warm-up trades excluded) AND the pooled matched-null
  comparison clears maxT at alpha = 0.05.
H2 SUPPORTED iff: arm C pooled trade-weighted R exceeds max(arm A, arm B) with
  95% bootstrap CI on the difference excluding zero.
If H2 fails: the regime switch is removed from the strategy regardless of H1.
If H1 fails: verdict is NO CROSS-ASSET EDGE, reported with the same prominence
  as any positive result, consistent with the repo's existing holdout verdict.

## Deviations policy
No per-instrument recalibration of any frozen parameter, no session-anchor
shopping, no post hoc instrument exclusion. Any deviation demotes the entire
study to exploratory and must be recorded here with rationale.
```

---

## Task 7 — Tests to add (extends the existing 76)

`tests/test_regime_ratio.py`:

- **No lookahead:** perturb all ATR values after session *t*; `atr_ratio` at *t* is
  unchanged. Perturb session *t*'s own ATR; its baseline is unchanged (the `shift(1)`).
- **Dimensionless invariance:** scale an instrument's entire price series by
  *k* ∈ {0.01, 100}; every `atr_ratio`, every regime classification, and every
  trade outcome in R are bit-identical.
- **Warm-up:** first `REGIME_BASELINE_LOOKBACK` sessions carry
  `regime_warmup = True` and are excluded identically from all three arms.

`tests/test_three_barrier.py`:

- **Ratchet monotonicity:** the trailing stop never moves against the position.
- **Breakeven bound:** on synthetic bars with no gaps, post-arm gross exit ≥ 0R;
  with a synthetic gap through the stop, the fill is `bar.open` and
  `exit_slippage_r` is negative and recorded.
- **Same-bar ambiguity:** stop-first convention preserved; flag set when both
  barriers fall inside one bar.
- **Arm population identity:** arms A/B/C over a fixture dataset produce journals
  with identical (signal, day) key sets.

`tests/test_friction_floor.py`: flagged trades are simulated and present in the
journal; analysis-layer filtering removes them; `friction_ratio` is correct against
hand-computed values for one contract of each `InstrumentSpec`.

---

## Execution order and reporting

Task 0 (gate) → Tasks 1–3 (code) → Task 7 (tests green) → Task 2 freeze → measure
frictions via `tools/costs/` → Task 6 prereg commit → **only then** run Task 5 on
non-NQ data, once per instrument, `HOLDOUT_PIN_START = "2026-02-01"`. Final report
follows the repo's reading order — holdout verdict, then breadth, then per-instrument
tables — and includes every instrument, including the failures. Given the base rate
established by the repo's committed holdout verdict, the prior is skeptical; the
design above is meant to make it impossible to quietly rescue a disappointing
instrument and call it a success.