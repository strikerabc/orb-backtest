# orb-backtest

Opening-range breakout research across futures instruments, with the statistical
machinery needed to tell an edge from a selection artefact.

**Current verdict: NO EDGE ESTABLISHED.** Out-of-sample, the in-sample survivors are
indistinguishable from a coin flip — 40.4% net-positive against 50% by chance,
trade-weighted −0.0573 R, CI [−0.1555, −0.0510]. The ranked in-sample tables are best
read as selection artefacts. Full detail in `outputs/holdout_verdict.json` after a run.

---

## Running it

```bash
python run.py
```

Asks two questions, then states what it decided and why:

1. **Accelerated or single-core.** Accelerated uses several CPU worker processes and,
   when a CUDA device is present, the GPU for bootstrap calibration. Single-core uses
   one worker and leaves the machine fully usable.
2. **Resume or fresh**, if checkpoints exist.

Results are identical either way — the sweep self-tests CPU/CUDA parity before it runs.

Non-interactive (CI, background, scheduled):

```bash
python run.py --accelerated          # maximum safe capacity, no prompts
python run.py --single-core          # one core, no prompts
python run.py --accelerated --fresh  # discard checkpoints and recompute
python run.py --self-test            # CPU/CUDA parity checks only
python run.py --workers 8            # explicit override (warns if unsafe)
python run.py --events all           # event-regime analysis, see below
```

If stdin is not a TTY the safe default is taken and the choice is logged, so a piped
or scheduled run never hangs waiting for input that is not coming.

Windows:

```powershell
.\run.ps1                            # interactive
.\run.ps1 -Accelerated -Background   # detached, logs to runs/
```

### What "maximum capacity before degradation" means

Not "every core" — that *is* the degradation. Three effects, and only the first is
obvious:

| effect | why it matters |
|---|---|
| **Core starvation** | `workers == cores` leaves nothing for the compositor or shell. The machine keeps its throughput and loses its responsiveness. Reserve is `max(2, 25%)`. |
| **BLAS oversubscription** | numpy links a threaded BLAS defaulting to one thread per core. N workers × cores threads competing for N cores collapses throughput *and* the UI, with no error to explain it. Every worker is pinned to one BLAS thread. |
| **Memory pressure** | Each worker holds a full instrument frame. A swapping machine is unusable in a way no core reservation prevents, so workers are capped by RAM (~1.5 GB each) as well as by cores — whichever binds first wins. |

`src/engine/capacity.py`, stdlib only. Monitor a run with:

```powershell
Get-Content .\runs\gpu_cpu_sweep.stdout.log -Wait
Get-Content .\runs\gpu_cpu_sweep.status.json
```

---

## Layout

```
run.py / run.ps1        interactive entry point
main.py                 single-threaded pipeline (reference implementation)

src/
  config.py             all tunables: instruments, sessions, thresholds
  data_layer.py         Databento fetch, caching, enrichment
  range_builder.py      SessionDay construction, opening ranges
  entry_detector.py     entry state machines (II, CC, TI, R-II, R-CC)
  swing_detector.py     pre-breakout cluster stop placement
  trade_sim.py          bar-by-bar exit walk, MAE/MFE, costs
  sizing.py             contracts, friction limits
  filters.py            trade eligibility — single source of truth
  journal.py            one row per (signal, trade, day)
  stats.py              per-variant and per-regime statistics
  regime_sampler.py     regime window selection + holdout boundary
  null_calibrator.py    matched-stop random-entry null
  multiplicity.py       Westfall-Young maxT, permutation FDR
  report.py             report.md generation
  engine/
    capacity.py         compute-capacity decisions
    sweep.py            parallel CPU + CUDA sweep runner
  events/               event-regime conditioning (optional, see below)

tools/
  holdout_test.py       out-of-sample verdict
  diagnostics/          null calibration and coverage investigations
  costs/                spread measurement and cost modelling
  equity/               equity curve renderers
  data/                 symbology probes, cache extension
  analysis/             recalibration, repricing, forward validation

tests/                  76 tests
docs/                   hypothesis specifications and reviews
prereg/                 pre-registrations, gate results, pinned baselines
data/events/            versioned event calendar (tracked)
```

### Sessions

| Session | Local open | Local exit | DST |
|---|---|---|---|
| NY | 09:30 America/New_York | 12:00 (11:59 bar) | 2nd Sun Mar → 1st Sun Nov |
| London | 08:00 Europe/London | 12:00 (11:59 bar) | Last Sun Mar → last Sun Oct |
| Tokyo | 09:00 Asia/Tokyo | 12:00 (11:59 bar) | None — fixed JST UTC+9 |

### Entry modes

| Mode | Description |
|---|---|
| II | Immediate Identification — first bar touching boundary + 1 tick |
| CC | Candle Close — first close beyond boundary (1/5/15/30m closure TF) |
| TI | Tap-in — re-touch of boundary after initial breakout |
| R-II | Retest-II — II trigger after a prior tap-in |
| R-CC | Retest-CC — candle close beyond boundary after a prior tap-in |

### What is and is not in version control

Tracked: source, tests, docs, the event calendar, and `prereg/`. Pre-registrations and
gate results are committed deliberately — **a pre-registration absent from history has
no evidential value**, because it could have been edited after seeing results.

Ignored: `data/` caches (large, regenerable), `outputs/` (generated), `runs/` logs and
checkpoints, and credentials.

---

## Event-regime conditioning (optional)

Tests whether the ORB edge degrades when a session's path is governed by gradually
revealed macro narrative — pressers, Q&A, testimony — rather than order-flow
continuation from the opening range. Specification in `docs/EVENT_REGIME_PLAN.md`;
findings and corrections in `docs/EVENT_REGIME_REVIEW.md`.

```bash
python run.py --events probe        # FOMC anticipation gate
python run.py --events validate    # permutation-machinery validation
python run.py --events calendar    # rebuild the FOMC calendar
python run.py --events case-study  # 2026-08-07 path-shape study
python run.py --events all
```

**Analysis-only.** Nothing in the pipeline imports `src/events`; it reads
`outputs/trade_log.parquet` after the fact and writes JSON into `prereg/`. Running it
changes no sweep number.

`config.EVENTS_ENABLED` (default `False`) gates the *unbuilt* Phase C work — an
`EVENT_FILTER_MODE` in `filters.py` and an event-aware null pool. That default is
load-bearing: enabling event filtering moves expectancies, breadth metrics, the null
pool, the maxT hurdle and the holdout verdict simultaneously, so it must never switch
on as a side effect of merging the feature.

### Result

−0.0565 R on FOMC anticipation days. Unstratified p = 0.024, but weekday-stratified
p = 0.087 — roughly **half the raw effect is Wednesday, not FOMC** (FOMC decisions are
44 of 47 Wednesdays in the sample, and Wednesdays are simply worse days). MDE 0.067 R
exceeds both the observed effect and the pre-registered 0.05, so this is
**underpowered rather than null**.

The level matters more than the delta: clean-day expectancy −0.0885 R, contaminated
−0.1471 R. Both firmly negative. The event effect is real in direction and a rounding
error against the level — **no event filter rescues this strategy.**

---

## Statistical gates

A variant must clear all of these. The design assumes any single gate can be fooled.

| gate | what it rules out |
|---|---|
| **Null calibration** | Compares each variant against random entry at the *same* stop distance, isolating entry timing from stop width. Median `null_p_matched` is 0.978 — the median variant is *worse* than its own comparator. |
| **Multiplicity** | Westfall-Young step-down maxT plus permutation FDR across ~1,250 families. Sweeping thousands of variants produces ~5% spurious winners by construction. |
| **Breadth** | Rejects variants whose P&L concentrates in a few days or a single window. |
| **Holdout** | Data never used for selection. The only gate that can refute the other three. |

Reading order for any result: holdout verdict first, then breadth, then the ranked
tables — never the reverse.

### Key design decisions

- **Swing stop:** searches down-closing clusters only *before* the initial breakout
  bar, so slow retrace clusters above entry are excluded by construction.
- **Exit bar:** the 11:59 local bar, closing at exactly 12:00:00 local.
- **ATR cap:** trades where TP > 2.5 × 4h ATR are flagged `atr_exceeds_cap` and
  simulated anyway — apply the filter at analysis time.
- **Same-bar ambiguity:** if stop and target are both hit within one 1m bar, the stop
  is assumed first (conservative) and `same_bar_ambiguous` is set.
- **Holdout boundary:** `regime_sampler` guarantees no fitted window crosses it. That
  guard exists because the boundary month was previously miscounted, leaking fitted
  trades into the out-of-sample region — see `tests/test_regime_windows.py`.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Databento credentials go in `databento_key.txt` (gitignored) or the
`DATABENTO_API_KEY` environment variable. A key is only needed to extend the cache; a
run against existing caches needs none.

CuPy is optional. Without a usable CUDA device the bootstrap runs on CPU — identical
results, materially slower.

```bash
python -m pytest tests/ -q      # 76 tests
python run.py --self-test       # CPU/CUDA parity
```
