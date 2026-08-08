# Event-Regime Conditioning — Implementation Specification

**Intended path:** `EVENT_REGIME_PLAN.md` (repo root, alongside `IMPROVEMENTS.md`)
**Audience:** the implementing agent
**Depends on:** the correctness and null-calibration work already landed (`filters.py`, `multiplicity.py`, the reworked `null_calibrator.py`, `SessionDay` flag fields). This plan hooks into those; it does not duplicate them.

**Hypothesis under test:** the ORB continuation edge degrades when a session's price path is governed by *gradually revealed* macro narrative — pressers, Q&A, testimony, panel remarks — rather than by order-flow continuation from the opening range.

---

## §0 — Scope change from the review draft

**Volume testing is out of scope.** The review draft proposed measuring relative volume as an alternative explanatory variable. That is dropped at the user's direction, on the reasoning that elevated volume at the session open is well-established across asset classes including continuously-traded FX, and the assumption was baked into the strategy from inception.

That reasoning is sound as far as the *level* goes. One consequence should be recorded rather than argued:

> The event hypothesis is not about the level of open volume, it is about *variation* on particular mornings — whether FOMC morning is a lower-participation session than a typical Tuesday for the same instrument. The literature settles the former, not the latter. Omitting volume therefore means that if the pooled test finds degradation, the plan cannot distinguish "narrative-driven tape" from "thin pre-event tape". Both are real edge deteriorators and both argue for the same practical response, so the omission does not change what you would *do* — only how you would *describe* the cause.

The mechanism metrics in §6.3 (path efficiency, reversal rate, boundary crossings, vol-profile shape) discriminate continuation from absorption directly, without volume. They carry the mechanism claim on their own. **The omission is affordable. Proceed without it.**

If it ever becomes cheap to revisit, the minimal version is a single journal column — `rel_vol_range`, the range-window volume divided by a trailing 20-session weekday-matched median, excluding the current day — added as a covariate and nothing else. Roughly ten lines. Not required by anything below.

Everything else from the review draft is in scope. §8 of this document carries the continuation-score work, renumbered.

---

## §1 — Framing, provenance, pre-registration

### 1.1 This is a conditioning variable, not a filter

The instinct is to drop contaminated days and see whether the remainder improves. It will. Dropping any subset of days chosen after inspecting the data improves the remainder. The test must be a **comparison of two partitions under a null that respects how the partition was constructed**, never a before/after on a filtered set.

This is why `EVENT_FILTER_MODE` defaults to `"flag"` (§5) and why the null pool must inherit any exclusion (§5.2).

### 1.2 Provenance of the hypothesis — better than post-hoc, with one hazard

The observation came from a **paper-trading account running forward**, not from mining this backtest. That matters: the hypothesis was generated on data outside the backtest's fitted history, which puts it in a stronger evidential position than a post-hoc in-sample split.

**But this creates a specific hazard that must be resolved before anything else.**

> **Action, before writing any code:** determine the calendar span of the paper-trading period and compare it against `HOLDOUT_MONTHS` (the most recent 3 months, excluded from all regime windows). If the paper period overlaps the holdout, **the holdout is contaminated for this hypothesis** — it is the very data the hypothesis was generated on, and it cannot serve as independent confirmation.
>
> Record the answer in `outputs/event_prereg.json` as `paper_period_start`, `paper_period_end`, `overlaps_holdout: bool`. If it overlaps, either (a) extend the holdout backwards so a clean out-of-sample slice exists, or (b) state plainly in the report that no clean out-of-sample test of the event hypothesis is available, and treat the whole exercise as in-sample.

Silently reusing a contaminated holdout is the single most likely way this work produces a confident wrong answer.

### 1.3 Pre-registration

Write `outputs/event_prereg.json` **before the first analysis run**, and commit it:

```json
{
  "written_utc": "<timestamp>",
  "git_sha": "<sha at time of writing>",
  "predicted_sign": "contaminated_expectancy < clean_expectancy",
  "predicted_magnitude_r": 0.05,
  "predicted_mechanism": [
    "higher post-breakout reversal rate (tap_in_bar_idx not None)",
    "lower path efficiency",
    "degradation steepens with rr"
  ],
  "primary_test": "day-label permutation, pooled across rankable families, T1+T2",
  "originating_families": ["<family key A>", "<family key B>"],
  "paper_period_start": "<date>",
  "paper_period_end": "<date>",
  "overlaps_holdout": null
}
```

`originating_families` must be filled in with the two models' family keys `(instrument, session, range_minutes, entry_mode, closure_tf, direction)`. Those two are contaminated by the observation and are **analysed separately and excluded from the pooled headline** (§6.2). If the effect appears only in them, it is noise. If it appears across families that had nothing to do with the observation, that is the finding.

### 1.4 The session clocks invert the naive design

This is the most important structural point in the document, and getting it wrong produces a false null.

| Session | Window (local) | Scheduled events **inside** the window | Major events **outside**, same day |
|---|---|---|---|
| **NY** | 09:30–12:00 ET | 10:00 ET releases (ISM, JOLTS, consumer confidence, UMich final); Powell semiannual testimony (~10:00 ET, runs 2–3 h) | **FOMC statement 14:00 ET, presser 14:30 ET — entirely outside**; 08:30 ET data — before the open |
| **LDN** | 08:00–12:00 London | Eurozone morning data (~09:00–10:00 London); occasional BoE speeches | **BoE decision 12:00 London — lands on the exit bar**; ECB decision + presser — early afternoon, outside |
| **TOK** | 09:00–12:00 JST | **BoJ policy announcement — no fixed time, frequently 11:30–12:30 JST, often inside or straddling the exit**; Japanese data 08:50 JST is pre-open | BoJ Governor presser ~15:30 JST |

**Read that again.** For NY and LDN — your two most-traded sessions — the events you are most worried about **do not overlap the trading window at all**. A filter written as "drop days where an event overlaps `[range_end, exit]`" would catch almost nothing there, find no effect, and lead you to discard a live hypothesis.

The channel that actually operates on NY and LDN is **anticipation**: on FOMC morning, participation thins, market-makers widen, directional flow stands aside, and 09:30–12:00 becomes a low-conviction drift-and-fade regime *precisely because* everyone is waiting. That is an in-window effect caused by an out-of-window event, and it matches the described observation — gradual, narrative-driven — better than a discrete shock does.

TOK/BoJ is the one genuine in-window case, and it is the awkward one: the announcement time is not fixed, so the anticipation window covers the whole morning.

**Design consequence:** exposure is modelled as **five distinct channels** (§2.2), tested separately. Never collapsed into one binary flag.

---

## §2 — Event taxonomy

Two orthogonal axes. Both are required; collapsing either loses the measurement.

### 2.1 Axis 1 — information geometry

| Class | Definition | Expected path signature |
|---|---|---|
| `IMPULSE` | Single-timestamp release of a scalar. NFP, CPI, PPI, retail sales, GDP, PCE, claims. | One-bar vol spike, then decay. Directional continuation *is* the normal response. **Should not harm the thesis** — plausibly helps it. |
| `DIFFUSE` | Information arrives over 30–120 min as prose. FOMC presser Q&A, semiannual testimony, Jackson Hole, ECB/BoE/BoJ pressers, panel remarks, minutes releases. | Sustained elevated vol, repeated direction changes, low path efficiency. **The hypothesised edge-killer.** |
| `SCHEDULED_DECISION` | Rate decision — an impulse whose *interpretation* is diffuse (statement wording, dot plot, vote split). | Impulse spike followed by a diffuse tail. |
| `ANTICIPATION` | No event in-window, but a `DIFFUSE` or `SCHEDULED_DECISION` lands later the same day. | Suppressed participation, compressed range, weak follow-through. |

`IMPULSE` is a **control arm, included specifically so the mechanism claim can be falsified.** If degradation shows equally on `IMPULSE` and `DIFFUSE` days, the narrative mechanism is wrong and something simpler — raw volatility, or a calendar artefact — explains it.

### 2.2 Axis 2 — exposure channel

Computed per `(instrument, session, local_date)`:

| Channel | Interval tested for overlap | Why separate |
|---|---|---|
| `RANGE` | `[open, open + max(RANGE_MINUTES)]` | Corrupts the reference level itself. `range_width_ticks` inflates and the breakout threshold becomes meaningless. |
| `PATH` | `[open + range_minutes, exit]` | Directly contaminates the tradeable path. |
| `PRE_OPEN` | `[open − EVENT_PRE_OPEN_HOURS, open]` | Range forms on a decaying shock; different mechanism. |
| `POST_EXIT_SAME_DAY` | `(exit, open + 24 h]` | **The anticipation channel. Dominant for NY and LDN.** |
| `PRIOR_DAY_DIFFUSE` | Event in the preceding 24 h | Narrative digestion / repricing hangover. Cheap to add. |

`RANGE` is evaluated against the largest `RANGE_MINUTES` so one flag covers all three sizes; a per-`rm` refinement is available if the result warrants it.

### 2.3 Severity tiering

Fed speeches alone exceed 200/year. Tag everything, tier it, and **run tiers as a dose–response, never as one binary split.**

| Tier | Contents | Approx. days/yr |
|---|---|---|
| `T1` | FOMC decisions, Powell pressers, semiannual testimony, Jackson Hole, ECB/BoE/BoJ decisions and pressers | ~40 |
| `T2` | Regional Fed president speeches with prepared remarks + Q&A, FOMC minutes, other central-bank governor remarks | ~80 |
| `T3` | Minor speeches, panel appearances, non-policy remarks | ~150 |

A monotone gradient T1 > T2 > T3 is coherent evidence. A flat response across tiers means the flag is picking up something other than event severity, and the placebo test (§6.4) will identify it.

---

## §3 — Data layer: `src/event_calendar.py`

No calendar exists in the repo. Build it as a **versioned CSV under `data/events/`, not a live API call.** Reproducibility is non-negotiable; a p-value computed against a calendar that shifts between runs is not a p-value.

### 3.1 Schema — `data/events/events.csv`

```
event_id                str      stable unique key
start_utc               ISO8601 with offset  e.g. 2024-03-20T18:30:00Z
end_utc                 ISO8601  REQUIRED for DIFFUSE — duration is the point
geometry                IMPULSE | DIFFUSE | SCHEDULED_DECISION
tier                    T1 | T2 | T3
region                  US | EU | UK | JP | GLOBAL
speaker                 str or ""            ("Powell", "Lagarde", "Ueda")
title                   str
affects_symbols         comma list or "*"    ("6J,ZN" for BoJ; "*" for FOMC)
scheduled_time_certain  bool                 False for BoJ, unscheduled remarks
source                  str                  provenance URL or archive ref
verified                bool                 has a human confirmed the timestamp
```

Three fields carry disproportionate weight:

- **`end_utc`** — a presser with no duration is just an impulse, and the entire `IMPULSE`/`DIFFUSE` contrast collapses. Where the end time cannot be sourced, apply a documented per-type default from `EVENT_DEFAULT_DURATION_MIN` and set `verified=False`. Sensitivity-test the defaults (§6.6).
- **`affects_symbols`** — a BoJ decision is first-order for 6J and second-order for ES. Tagging every instrument on every event swamps the signal. FOMC = `*`; ECB = `6E` plus a weaker global tag; BoJ = `6J`.
- **`scheduled_time_certain`** — `False` extends the exposure interval to the whole session, which is the correct treatment for BoJ.

### 3.2 Timestamps to source, not assume

Publication conventions moved during the 2019–2026 sample.

| Event | Time | Confidence |
|---|---|---|
| FOMC statement | 14:00 ET | High (unchanged since 2013) |
| FOMC presser | 14:30 ET | High |
| US NFP / CPI / PPI | 08:30 ET | High |
| ISM, JOLTS, UMich final | 10:00 ET | High |
| Powell semiannual testimony | ~10:00 ET, 2–3 h | Medium — confirm per year |
| BoE decision | 12:00 London | Medium — confirm convention within sample |
| ECB decision + presser | Early afternoon CET; **decision time moved in 2022** | **Low — must be sourced per era** |
| BoJ decision | No fixed time, typically late morning JST | Low by nature — `scheduled_time_certain=False` |

Federal Reserve calendar pages, ECB/BoE/BoJ press-release archives, and BLS/BEA release schedules are free and authoritative. **Budget one day of manual work for T1 across seven years — roughly 300 rows. This is the highest-value day of work in the entire plan**, because every downstream number depends on the timestamps.

### 3.3 Loader contract

```python
def load_events(path: str = EVENT_CALENDAR_PATH) -> pd.DataFrame:
    """Return events with tz-aware UTC start/end, validated.

    Assertions — fail loudly, never coerce:
      - start_utc < end_utc
      - no DIFFUSE row where end_utc == start_utc
      - tier in {T1, T2, T3}; geometry in the enum
      - no duplicate event_id
      - every row carries a non-empty source

    Returns a frame sorted by start_utc, plus an IntervalIndex for O(log n)
    overlap queries.
    """
```

On load, log counts by `(tier, geometry, year)`. **A year with zero T1 events is a sourcing bug, not a quiet market** — assert on it.

---

## §4 — Tagging layer: `src/event_tagging.py`

### 4.1 Where it runs

As a **post-pass over the output of `build_session_days`**, not inside it. This keeps the data layer separable and the tagging independently testable.

```python
def tag_session_days(days: list[SessionDay],
                     events: pd.DataFrame) -> list[SessionDay]:
    """Mutate/return SessionDays with event exposure fields populated."""
```

### 4.2 New `SessionDay` fields

Append to the dataclass, following the existing default-valued flag-field pattern (`contract_changed_in_session`, `session_bar_completeness`):

```python
event_range_tier:      str | None = None   # highest tier overlapping RANGE
event_path_tier:       str | None = None
event_pre_open_tier:   str | None = None
event_post_exit_tier:  str | None = None   # anticipation
event_prior_day_tier:  str | None = None
event_geometry_path:   str | None = None   # IMPULSE | DIFFUSE | SCHEDULED_DECISION
event_geometry_post:   str | None = None
event_ids:             tuple[str, ...] = ()
event_minutes_in_path: float = 0.0         # dose: overlap minutes
event_regime:          str = "CLEAN"       # derived label
```

### 4.3 Derived label

One column the analysis groups on, by precedence:

```python
def derive_event_regime(sd: SessionDay) -> str:
    if sd.event_path_tier == "T1" and sd.event_geometry_path == "DIFFUSE":
        return "DIFFUSE_IN_PATH"
    if sd.event_path_tier == "T1":
        return "IMPULSE_IN_PATH"
    if sd.event_range_tier == "T1":
        return "EVENT_IN_RANGE"
    if sd.event_post_exit_tier == "T1" and sd.event_geometry_post in (
            "DIFFUSE", "SCHEDULED_DECISION"):
        return "ANTICIPATION"              # dominant NY / LDN channel
    if sd.event_pre_open_tier == "T1":
        return "PRE_OPEN_SHOCK"
    if sd.event_prior_day_tier == "T1":
        return "HANGOVER"
    if any tier in ("T2", "T3") on any channel:
        return "MINOR"
    return "CLEAN"
```

Retain the raw channel columns. The derived label is for convenience; the channels are what the tests actually operate on.

### 4.4 Timezone correctness — the one place this breaks silently

Overlap must be computed in **UTC**, against window bounds derived *for that specific date*. Never in local wall-clock time: NY and London DST transitions diverge for roughly three weeks each spring and one week each autumn, so a 14:00 ET event maps to different London times depending on the date.

**Do not re-derive the window.** Every `SessionDay` already carries `bar_timestamps` as UTC int64 ns and `session_open_idx`. Take the bounds directly:

```python
open_utc_ns = int(sd.bar_timestamps[sd.session_open_idx])
exit_utc_ns = int(sd.bar_timestamps[-1]) + 60 * 1_000_000_000   # close of last bar
range_end_ns = open_utc_ns + max(RANGE_MINUTES) * 60 * 1_000_000_000
```

No re-derivation, no drift, no DST logic to get wrong.

### 4.5 Journal columns

Add to `journal.build_row`:

```
event_regime, event_path_tier, event_post_exit_tier,
event_geometry_path, event_minutes_in_path, event_ids (joined str)
```

These flow to `trade_log.parquet` and become groupby keys for the analysis.

---

## §5 — Config and the filter switch

### 5.1 Config block

```python
# config.py — Event regime
EVENT_CALENDAR_PATH: str    = "data/events/events.csv"
EVENT_PRE_OPEN_HOURS: float = 2.0
EVENT_TIERS_ACTIVE: tuple   = ("T1", "T2")   # T3 tagged, excluded from headline
EVENT_DEFAULT_DURATION_MIN: dict[str, int] = {
    "FOMC_PRESSER": 60, "TESTIMONY": 150, "PRESSER": 60, "SPEECH": 45,
}
# off     : tag only, no column reaches filters
# flag    : write `excluded_event_regime` but force it all-False (no effect)
# exclude : contaminated days removed from `eligible`
EVENT_FILTER_MODE: str = "flag"              # DEFAULT — do not change silently
EVENT_EXCLUDE_REGIMES: tuple = (
    "DIFFUSE_IN_PATH", "ANTICIPATION", "EVENT_IN_RANGE")
```

### 5.2 Two rules that must not be broken

**Rule 1 — `EVENT_FILTER_MODE` defaults to `"flag"`.**

Setting `"exclude"` silently changes every number in the repo: expectancies, breadth metrics, null pools, the maxT hurdle, the holdout. The mode must therefore appear in three places so it can never be invisible:

- the `manifest.json` written by `null_calibrator.write_null_artifacts`,
- the report header,
- `summary.parquet` as a constant column `event_filter_mode`.

**Wiring into `filters.py` — the trap.** `filters.trade_eligibility` computes `eligible = ~out[list(RULE_COLUMNS)].any(axis=1)`, and `stats.compute_summary` iterates `RULE_COLUMNS` to emit `n_{rule}` counts. Adding a naive rule would change `eligible` under every mode. The correct construction:

```python
RULE_COLUMNS = (..., "excluded_event_regime")   # added, so counts are reported

regime = out.get("event_regime", pd.Series("CLEAN", index=index))
if EVENT_FILTER_MODE == "exclude":
    out["excluded_event_regime"] = regime.isin(EVENT_EXCLUDE_REGIMES)
else:
    out["excluded_event_regime"] = pd.Series(False, index=index)
```

The column always exists (so `n_excluded_event_regime` is always reported), but is all-`False` unless the mode is `exclude`. This preserves the regression gate in §9 Phase C.

`filters.trade_eligibility` must also **tolerate the column being absent** — `null_calibrator._eligible_results` constructs a small frame per draw with a fixed column list. Use `out.get("event_regime", default)` as above, and add `sd.event_regime` to the dict built in `_eligible_results` so the null path sees the same rule.

**Rule 2 — if you exclude from observed, you must exclude from the null pool.**

This is the subtle one and it is decisive. `sample_null_days` currently draws from all available `SessionDay`s. If observed trades come only from clean days while the null pool spans all days, **the p-value measures the day-mix difference, not the timing edge** — and it does so in the flattering direction.

Thread a predicate through:

```python
def sample_null_days(session_days, n_target, rng, *,
                     stratify_by_window=True, window_of=None,
                     window_weights=None,
                     day_predicate=None):          # NEW
    days = [sd for sd in session_days
            if day_predicate is None or day_predicate(sd)]
    n_pre_filter = len(session_days)
    ...
```

and from `enrich_summary_with_null`, pass:

```python
day_predicate=(
    (lambda sd: sd.event_regime not in EVENT_EXCLUDE_REGIMES)
    if EVENT_FILTER_MODE == "exclude" else None)
```

Record `null_days_event_filtered = n_pre_filter - len(days)` in the provenance frame written by `write_null_artifacts`. Note the filter runs **before** the exhausted check, so `null_days_used` and `null_days_exhausted` semantics shift — document that in the provenance columns.

The matched-day pool (`null_p_matched`) already restricts to fired dates and inherits the filter automatically. Only the broad pool needs the predicate.

---

## §6 — The tests

### 6.1 Primary: day-label permutation

The right null, and cheap. For a family and channel:

```
observed_delta = mean(net_r | contaminated days) − mean(net_r | clean days)

for k in 1..K            (K = 20,000, matching NULL_BOOTSTRAP_N)
    shuffle contaminated/clean labels ACROSS DAYS,
        holding the contaminated day COUNT fixed
    delta_k = recomputed delta under the shuffled labels

p = (1 + #{delta_k <= observed_delta} + 0.5·#{delta_k == observed_delta}) / (1 + K)
```

One-sided in the pre-registered direction (degradation). Mid-p tie handling, consistent with `null_calibrator.null_p_value`.

**Why this null and not "score each subset against the random-entry null separately":** label permutation holds the variant, instrument, cost model, stop geometry and total day count fixed. Only *which* days wear the label varies. Everything else cancels exactly.

**Shuffle days, never trades.** A day contributes several correlated trades; shuffling trade labels breaks the clustering and yields a p-value that is far too small. Permute at the day level and carry each day's whole trade block with its label — the same discipline as `stats.day_clustered_t` and `stats._block_bootstrap_ci`.

**Stratified variant — run both.** Contaminated days may simply be higher-volatility days, and volatility alone moves R-multiples. Block the permutation within volatility terciles computed from `parkinson_vol_14d` (now correctly `.shift(1)`-lagged, so no lookahead). Survival under stratification means the effect is not merely a volatility story.

**Degenerate cases:** if a family has zero contaminated days or zero clean days, return `NaN` and exclude from pooling. Record the count.

### 6.2 Pooling — where the power lives

Per-family tests are hopeless (§7). The headline must pool.

| Level | Unit | Permutation construction |
|---|---|---|
| **Level 1 — headline** | All rankable families, **excluding the two originating models**. Statistic: trade-weighted mean of per-family deltas. | Shuffle day labels within each family independently, then re-pool. Preserves family structure. |
| Level 2 | Per `(instrument, session)` | Same, within the group. |
| Level 3 | Per family | Reported, **explicitly marked underpowered**. |
| **Level 0 — separate** | The two originating models | Reported apart, **never pooled in**. The observation source cannot be its own confirmation. |

**Level 1 with a pre-registered sign is the number that decides this.** Everything else is texture.

**Joint draws for multiplicity.** The channel × tier × pooling grid is a new family of hypotheses and needs the same treatment as the main sweep. To make `multiplicity.step_down_max_t` valid here, draws must be **joint**: for draw *k*, generate **one** random permutation π of the day index per `(instrument, session)`, then apply that same π to *every* channel's label vector and *every* family and rr level. This preserves the cross-hypothesis correlation — which is large, since channels overlap and the six rr levels share entries — and makes `max` across hypotheses meaningful. Independent per-hypothesis permutation would destroy that structure and make the correction wildly conservative.

Feed the resulting `(K × H)` matrix straight into `multiplicity.step_down_max_t` and `multiplicity.permutation_fdr`. Both already accept exactly this shape.

### 6.3 Mechanism tests — the part that makes it convincing

P&L degradation alone is weak evidence; a hundred things degrade P&L. The hypothesis makes a *specific* claim about path shape, and it is directly testable from data already present.

| Metric | Definition | Prediction under `DIFFUSE` |
|---|---|---|
| **Reversal rate** | Fraction of breakouts with `tap_in_bar_idx is not None` | **Higher.** Free — TI detection *is* a reversal detector. |
| **Path efficiency** | `abs(c[exit] − c[entry]) / Σ abs(1m returns)` over the hold | **Lower.** The cleanest continuation-vs-chop discriminator. |
| **Boundary crossings** | Count of `rh` (or `rl`) crossings in `[range_end, exit]` | **Higher** — whipsaw. |
| **MFE/MAE ratio** | `mfe_r / abs(mae_r)`, already in the trade log | **Lower.** |
| **Vol-profile shape** | Mean 1m absolute return in the final third of the window ÷ the first third | `IMPULSE` → sharply **falling**. `DIFFUSE` → **flat or rising**. |
| **TP hit rate by rr** | `tp_hit_rate` (already computed in `stats`) across the rr grid | Degradation **concentrated at high rr**, since continuation is what carries a 2R target. |

Two of these do real work beyond confirmation:

- **The vol-profile row validates the taxonomy.** It checks empirically whether the `IMPULSE`/`DIFFUSE` hand-labelling corresponds to genuinely different path shapes. If tagged-`DIFFUSE` days do not show a flatter profile than tagged-`IMPULSE` days, the calendar is mislabelled and every downstream number measures nothing. **Run this before interpreting any P&L result.**
- **The rr row is a sharp falsifier.** Uniform degradation across rr means the mechanism is not "continuation fails" — it is something dumber, like wider spreads. Steepening with rr supports the mechanism claim.

### 6.4 Placebo — run before believing anything

Re-run the primary test against fake event dates:

- **Shift placebo:** every event date moved **+7 and −7 days**. Preserves day-of-week, month-position and seasonality; destroys event alignment.
- **Weekday-matched placebo:** for each real event day, a random non-event day with the same weekday and the same `regime_window`.

If the "effect" appears on placebo dates at comparable magnitude, you have a calendar artefact.

**This is a live risk, not a formality.** `config.py` already documents that 6E and 6J degrade in post-expiry months and ZN thins in delivery months — both calendar-locked. FOMC dates are calendar-locked. Month-end, quarter-end and option-expiry effects are further calendar-locked confounds. Without the placebo you cannot separate any of them from the event signal.

### 6.5 Holdout

Cheap, because there are few hypotheses — one directional prediction per channel. Add `event_regime_delta` and `event_perm_p` to `holdout_verdict.json`; `report._verdict_banner` already reads that file, so extend the banner to display them.

**Gated on §1.2.** If the paper-trading period overlaps the holdout, this test is not independent and must be reported as such rather than run and quoted.

### 6.6 Calendar sensitivity

Two knobs materially affect the labelling, so vary them and confirm the sign is stable:

- **Duration defaults** — rerun with `EVENT_DEFAULT_DURATION_MIN` at 0.5× and 2×. Affects only rows with `verified=False`.
- **Tier boundary** — rerun the headline at `T1` only and at `T1 ∪ T2 ∪ T3`. A monotone gradient is the expected shape.

Report both as a small sensitivity table. A result that flips sign under a 2× duration assumption is not a result.

---

## §7 — Power reality check

Do this arithmetic before writing code; it determines whether the design is feasible at all.

Assume 2019–2026, ~1,750 session-days per instrument-session.

| Regime | Days/yr | Days in sample | % of days |
|---|---|---|---|
| `DIFFUSE_IN_PATH` (NY — testimony, Jackson Hole, in-window speeches) | ~10 | ~70 | **4%** |
| `ANTICIPATION` (NY — FOMC mornings) | ~8 | ~56 | **3%** |
| `IMPULSE_IN_PATH` (NY — 10:00 releases) | ~50 | ~350 | 20% |
| `DIFFUSE_IN_PATH` (TOK — BoJ) | ~8 | ~56 | 3% |
| T1, any channel | ~40 | ~280 | 16% |

At 3–4% of days, a family with 500 trades has roughly **15–20 contaminated trades**. At per-trade sd ≈ 1.0 R, the standard error on that subset mean is ≈ **0.24 R**. A 0.05 R degradation is undetectable there — not with a better test, not with more permutations. The information is not present.

Three consequences, each of which shapes the plan:

1. **Per-family event tests are decorative.** Report them, label them underpowered, draw no conclusions. Across ~1,000 underpowered tests, ~50 will show `p < 0.05` by construction. Apply `multiplicity.step_down_max_t` here as well, and resist pointing at the winner.
2. **Pooling is mandatory.** Pooled across ~1,000 families × ~280 contaminated days per instrument-session, the contaminated sample reaches tens of thousands of trades and the test has genuine power. This is why §6.2 is built the way it is.
3. **Widen to buy power, then narrow to confirm.** Headline on `T1 ∪ T2` across all channels (~16–25% of days); confirmation is that the effect *concentrates* in `T1` and in `DIFFUSE`. The wide net makes the test possible; the concentration is what makes it credible.

**Report the minimum detectable effect alongside every null result:**

```
MDE ≈ 2.8 × sd_per_trade × sqrt(1/n_contaminated + 1/n_clean)      # 80% power, α=0.05, one-sided
```

A null result must be bounded, not over-read as absence.

---

## §8 — Continuation regime score

Both the event flag and any participation proxy stand in for one latent thing: **is today's tape in a continuation regime or an absorption regime?** That is directly measurable, and measuring it turns "events are bad" into "events are one cause of a state you can observe".

Compute per session-day:

- **Signed 1-minute return autocorrelation** at lags 1–5, or
- **Variance ratio** — `Var(k-minute returns) / (k · Var(1-minute returns))`, with `k ∈ {5, 15}`. Above 1 indicates trending/continuation; below 1 indicates mean-reversion/absorption.

Then check whether `continuation_score` predicts variant P&L better than the event label does. If it does, the honest conclusion is that events are one cause among several of a regime that is measurable in its own right.

### The hard constraint

A score computed from the full session is **explanatory only** — it uses the future. Split cleanly and tag both:

| Feature | Window | Use |
|---|---|---|
| `continuation_score_full` | Whole session | Explanation and mechanism validation. **Must never enter a filter.** |
| `continuation_score_range` | Opening-range bars only | Known by range-end. **Tradable.** |

Tag every derived feature in the journal with a `tradable: bool` companion, or maintain an explicit `TRADABLE_FEATURES` frozenset in `config.py` that `filters.py` asserts against. The distinction between "explains the past" and "usable in advance" is precisely what gets lost between analysis and deployment, and it is the whole difference between a finding and a fantasy.

`continuation_score_range` is computed from at most 30 bars, so it will be noisy. Report its correlation with `continuation_score_full` — if that correlation is near zero, the tradable version carries no information about the regime and should not be promoted to a filter regardless of how well the full-session version explains P&L.

---

## §9 — Implementation order

### Phase A — Calendar (everything depends on it)
1. Build `data/events/events.csv` for **T1, 2019 → present**. ~300 rows, manual, sourced from central-bank and statistical-agency archives. `verified=True` only for timestamps actually confirmed.
2. `src/event_calendar.py` — loader, validators, `IntervalIndex`.
3. **Tests:** every FOMC decision date in the sample appears exactly once; no `DIFFUSE` row has zero duration; counts by `(tier, geometry, year)` printed and asserted non-zero per year.
4. Extend to T2 once T1 is validated. T3 optional.

### Phase B — Tagging
5. `src/event_tagging.py` — UTC overlap against bounds derived from `bar_timestamps`; five channels; `derive_event_regime`.
6. `SessionDay` fields; `journal.build_row` columns.
7. **Test — the one that catches the timezone bug:** hand-verify ~20 known days across all three sessions. Specifically assert that a known FOMC date tags NY as `ANTICIPATION` and **not** `DIFFUSE_IN_PATH`. If it comes back in-path, the UTC handling is wrong — and you have caught it here rather than in the results.
8. Report the regime distribution per instrument-session. If `CLEAN` is below 60% or above 95%, the tiering is miscalibrated — fix before proceeding.

### Phase C — Plumbing that must not silently change results
9. `filters.py` — `excluded_event_regime` added to `RULE_COLUMNS`, forced all-`False` unless mode is `exclude`; `.get()` defaults so the null path tolerates missing columns.
10. `null_calibrator._eligible_results` — pass `sd.event_regime` into the per-draw frame.
11. `sample_null_days` — `day_predicate` parameter; `null_days_event_filtered` in provenance.
12. Mode recorded in `manifest.json`, the report header, and `summary.parquet`.
13. **Regression gate:** with `EVENT_FILTER_MODE="flag"`, assert every **pre-existing** column of `summary.parquet` is bit-identical to the pre-change run. (New columns are expected; existing values must not move.) If this fails, the tagging layer is leaking into the base pipeline.

### Phase D — Analysis
14. `src/event_analysis.py` — day-label permutation (plain and vol-stratified), the four pooling levels, dose–response across tiers, mechanism metrics, placebo, calendar sensitivity.
15. Joint-draw construction (§6.2) feeding `multiplicity.step_down_max_t` and `permutation_fdr`.
16. Report section (§10); holdout extension gated on §1.2.

### Phase E — Continuation score
17. `continuation_score_full` and `continuation_score_range`; `tradable` tagging.
18. Correlation between the two; comparison against the event label as a P&L predictor.

### Validation suite — run before interpreting anything

| Test | Catches |
|---|---|
| Identity: `mode="flag"` reproduces pre-change summary exactly | Tagging leaking into the base pipeline |
| Known-FOMC-date channel assignment across all three sessions | UTC/DST handling |
| Permutation on **shuffled event labels** yields p ≈ Uniform(0,1) | A test that finds significance in noise |
| Injected synthetic effect (force −0.3 R on tagged days) is detected | Insufficient power / broken pooling |
| Joint-draw integrity: within draw *k*, the same day permutation applies to every hypothesis | Invalid maxT correction with no other symptom |
| Vol-profile separates tagged `IMPULSE` from tagged `DIFFUSE` | A mislabelled calendar |
| Placebo (±7 d) returns null | Calendar artefact masquerading as an event effect |

---

## §10 — Report section

```markdown
## Event-Regime Conditioning

Pre-registered prediction (outputs/event_prereg.json, written <date>, sha <sha>):
contaminated-day expectancy below clean-day expectancy by ~0.05 R, driven by
higher post-breakout reversal and lower path efficiency, steepening with rr.

Filter mode this run: FLAG — contaminated days retained; no other table affected.
Null pool event-filtered: NO.
Paper-observation period <start>–<end>; overlaps holdout: <yes/no>.

### Day-regime distribution
| instrument | session | CLEAN | ANTICIPATION | DIFFUSE_IN_PATH | IMPULSE_IN_PATH | EVENT_IN_RANGE | MINOR |

### Pooled delta — all rankable families, excl. originating models
| channel | tier | n_contam_days | n_trades | delta_net_r | perm_p | perm_p_adj | MDE |

### Dose-response
| tier | delta_net_r | 95% CI |        <- monotone if the mechanism is real

### Mechanism
| metric | clean | contaminated | delta | perm_p |
| reversal_rate          | ... |
| path_efficiency        | ... |
| boundary_crossings     | ... |
| vol_profile_ratio      | ... |   <- validates the IMPULSE/DIFFUSE taxonomy
| tp_hit_rate @ rr=0.25  | ... |
| tp_hit_rate @ rr=2.00  | ... |   <- degradation should steepen with rr

### Placebo (+/-7 day shift)
| channel | delta_net_r | perm_p |   <- must be null, or this is a calendar artefact

### Calendar sensitivity
| assumption | delta_net_r | perm_p |
| duration defaults x0.5 | ... |
| duration defaults x2.0 | ... |
| T1 only                | ... |
| T1+T2+T3               | ... |

### Originating models — reported separately, NOT pooled
| family | delta_net_r | perm_p |   <- the observation source cannot confirm itself
```

Notes to carry:
- Filter mode verbatim, and whether the null pool was event-filtered.
- Minimum detectable effect beside every null result.
- Which timestamps are `verified=False` and what durations were assumed.
- **That FOMC statements fall outside the NY window**, so the NY result is an *anticipation* finding, not a shock finding. Without this sentence the table is easy to misread.
- BoJ/TOK reported separately, flagged for timestamp uncertainty.

---

## §11 — Limitations

**On the hypothesis.** Generated from two models on a paper account. The pre-registration in §1.3 and the forward-looking provenance in §1.2 place it above a post-hoc split, but §1.2's holdout-overlap check must be resolved first — otherwise the out-of-sample confirmation is circular.

**On the calendar.** ECB and BoJ timestamps are the weakest links. BoJ's floating announcement time cannot be reconstructed retrospectively with confidence, which means the TOK in-path result — the *only* channel with genuine in-window exposure — rests on the shakiest data in the file. Report it separately and flag it.

**On power.** At 3–4% of days for the sharpest channels, only the pooled test has meaningful power. Per-family results are decorative. A null bounds the effect; it does not establish absence.

**On confounding.** Event days are calendar-locked and so are your roll artefacts (6E/6J post-expiry, ZN delivery months, all documented in `config.py`), plus month-end, quarter-end and expiry effects. The ±7-day placebo absorbs these partially, not fully. **Run the placebo before believing the finding.**

**On the omitted volume dimension (§0).** If degradation is found, this plan cannot distinguish "narrative-driven tape" from "thin pre-event tape". Both are genuine deteriorators arguing for the same response, so the practical conclusion is unaffected — but the causal description in the report must stay agnostic between them rather than asserting the narrative mechanism.

**On the exclusion path.** Setting `EVENT_FILTER_MODE="exclude"` changes every number in the repo and requires the null pool to be filtered identically (§5.2). Phase C step 13 exists to catch a partial application. Do not skip it.

**On what a positive result would mean.** Even a clean, pooled, placebo-surviving, dose-responsive result establishes that the edge is *weaker* on event days — not that a clean-day-only strategy is profitable. Given that the holdout verdict already reports the survivor set as indistinguishable from a coin flip, the most likely outcome is that clean-day expectancy is also indistinguishable from zero, merely less negative. **This work sharpens the question; it does not by itself rescue the thesis.** Set that expectation before running, so the result is read honestly in either direction.

---

## Appendix A — Config diff

```python
# ── NEW ────────────────────────────────────────────────────────────────
EVENT_CALENDAR_PATH: str          = "data/events/events.csv"
EVENT_PRE_OPEN_HOURS: float       = 2.0
EVENT_TIERS_ACTIVE: tuple         = ("T1", "T2")
EVENT_DEFAULT_DURATION_MIN: dict  = {"FOMC_PRESSER": 60, "TESTIMONY": 150,
                                     "PRESSER": 60, "SPEECH": 45}
EVENT_FILTER_MODE: str            = "flag"        # off | flag | exclude
EVENT_EXCLUDE_REGIMES: tuple      = ("DIFFUSE_IN_PATH", "ANTICIPATION",
                                     "EVENT_IN_RANGE")
EVENT_PERM_N: int                 = 20_000        # matches NULL_BOOTSTRAP_N
TRADABLE_FEATURES: frozenset      = frozenset({   # §8 guardrail
    "continuation_score_range", "range_width_ticks", "gap_ticks",
    "atr_4h", "parkinson_vol_14d", "realized_vol_14d",
})

# ── CHANGED ────────────────────────────────────────────────────────────
# filters.RULE_COLUMNS  += ("excluded_event_regime",)
#   -> always present so n_excluded_event_regime is reported,
#      but forced all-False unless EVENT_FILTER_MODE == "exclude"
```

## Appendix B — File manifest

| File | Change |
|---|---|
| `src/event_calendar.py` | **new** — loader, validators, interval index |
| `src/event_tagging.py` | **new** — `tag_session_days`, `derive_event_regime` |
| `src/event_analysis.py` | **new** — permutation, pooling, mechanism, placebo, sensitivity |
| `data/events/events.csv` | **new** — versioned calendar, ~300 T1 rows |
| `outputs/event_prereg.json` | **new** — pre-registration, committed before first run |
| `src/config.py` | Appendix A block |
| `src/range_builder.py` | 10 `SessionDay` fields (§4.2) |
| `src/journal.py` | 6 journal columns (§4.5) |
| `src/filters.py` | `excluded_event_regime` rule, mode-gated (§5.2) |
| `src/null_calibrator.py` | `day_predicate` on `sample_null_days`; `event_regime` into `_eligible_results`; `null_days_event_filtered` provenance; mode into manifest |
| `src/report.py` | `_event_section()`, mode in header, holdout banner fields |
| `src/multiplicity.py` | unchanged — reused as-is for the channel × tier grid |