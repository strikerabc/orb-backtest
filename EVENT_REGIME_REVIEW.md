# Review of `EVENT_REGIME_PLAN.md`

**Reviewer:** implementing agent
**Date:** 2026-08-08
**Branch:** `hypothesis/event-regime-conditioning`
**Repo state reviewed at:** `330a191` (main)

---

## Verdict

The methodology is sound and the central structural insight (§1.4) is genuinely
good — it is the thing that stops this being a wasted month. Every claim the plan
makes about the existing codebase checks out against the source; whoever wrote it
actually read `filters.py`, `null_calibrator.py` and `multiplicity.py` rather than
guessing.

Three defects change conclusions rather than cosmetics. One of them is a straight
bug that would manufacture significance out of noise. Separately, the **sequencing
is wrong**: the plan spends its entire budget building infrastructure before it
learns anything, and §7 and §11 between them already tell you the most likely
outcome. Fix the three defects, then run the cheap gate first.

Sections below are ordered by whether they change a number.

---

## A. Blocking — these change results

### A1. §6.2 Level-1 pooling permutation is wrong as written

> "Shuffle day labels **within each family independently**, then re-pool."

This destroys the cross-family correlation that is the whole reason pooling is
delicate. All ~1,190 rankable families draw from only ~24 `(instrument, session)`
pairs, and within a pair every family trades **the same calendar days on the same
price path**. A choppy FOMC morning hurts nearly all of them at once, so the
per-family deltas are strongly positively correlated.

Under the true null, the pooled statistic therefore has a sampling sd close to
that of a *single* family's delta — not that sd divided by `sqrt(1190)`.
Independent per-family shuffling produces the second thing. The null comes out
far too tight, and the p-value far too small.

This is exactly the error §6.1 warns about ("shuffling trade labels breaks the
clustering and yields a p-value that is far too small") committed one level up,
at the family instead of the trade.

It also contradicts the plan's own next paragraph, which gets it right for the
maxT draws: *"for draw k, generate **one** random permutation π of the day index
per `(instrument, session)`, then apply that same π to every channel's label
vector and every family and rr level."* That rule is correct. It must govern the
headline statistic too, not just the multiplicity grid.

**Fix:** one permutation per draw, applied to every family, channel and rr level.
And go further than the plan: because all 24 instrument-sessions share one
calendar, permute the label over the **union date axis once per draw** and apply
it everywhere. That preserves within-family clustering *and* cross-instrument
correlation. Independent permutation per instrument-session has the same flaw at
a smaller scale.

**Structural mitigation:** collapse each day to a mean before permuting. If the
unit of analysis is `(instrument, session, date) → mean net_r`, the joint-draw
discipline is enforced by the data layout and cannot be got wrong by accident.
The probe on this branch does it that way.

### A2. §7's MDE formula overstates power by roughly 3–5×

```
MDE ≈ 2.8 × sd_per_trade × sqrt(1/n_contaminated + 1/n_clean)
```

`n_contaminated` here is a **trade** count. Trades are not independent draws.
They cluster twice over: many trades per day, and many near-duplicate families
sharing that same day and price path (`range_minutes` × `entry_mode` ×
`closure_tf` × `direction` × 6 rr levels off one underlying session). The plan
knows this in §6.1 and forgets it in §7.

The effective sample size is the number of **distinct contaminated days**, not
trades. Plugging trade counts into that formula inflates power by the square root
of the design effect — plausibly 3–5× here. Since the plan instructs reporting
the MDE beside every null result, a wrong MDE means confidently concluding "we
had the power to see 0.05 R and it isn't there" when you never did.

**Fix:** don't use a closed form at all. The permutation null already encodes
every correlation in the design. Take

```
MDE ≈ 2.486 × sd(delta_permuted)      # 80% power, one-sided α = 0.05
```

directly from the draws you are already generating. It costs nothing extra and
cannot be wrong about the clustering.

### A3. The pre-registration file is written to a gitignored directory

§1.3 says write `outputs/event_prereg.json` "**before the first analysis run**,
and commit it." `.gitignore` line 3 is `outputs/`. The commit is a silent no-op —
`git add` refuses it without `-f`, nothing lands in history, and a
pre-registration that isn't in history has no evidential value whatsoever. It
becomes a local file you could edit after seeing results, which is precisely the
thing pre-registration exists to rule out.

The same applies to `EVENT_REGIME_PLAN.md` itself: `.gitignore` ends with
`IMPROVEMENTS.md` and (uncommitted, in the working tree) `EVENT_REGIME_PLAN.md`.
The document defining the hypothesis was excluded from the repository.

**Fixed on this branch:** pre-registration lives at tracked
`prereg/event_prereg.json`; the plan and this review are tracked; `data/events/`
is un-ignored by negation so the calendar is versioned as §3 requires.

---

## B. Facts from the repo that move the plan's numbers

I checked these against the actual artifacts rather than accepting the plan's
assumptions.

### B1. The sample is ~28% smaller than §7 assumes

§7 assumes "~1,750 session-days per instrument-session." The real counts from
`outputs/trade_log.parquet`:

| | value |
|---|---|
| date range | **2019-04-01 → 2026-04-30** (not 2019-01-01) |
| session-days, mature symbols | **1,233–1,285** per instrument-session |
| BTC/ETH | 355–1,004 (already disqualified) |
| instrument-sessions | 24 |
| variants in `summary.parquet` | 7,529 |
| ranking threshold | `MIN_TRADES_FOR_RANKING = 100` |

Trading days are ~72% of calendar days, and the sample starts in April 2019, so
every row of §7's table needs scaling by ~0.72. `ANTICIPATION` (NY) goes from
~56 days to **~40**. `DIFFUSE_IN_PATH` (NY) from ~70 to **~50**. The channels the
plan calls sharpest get thinner, and the MDE correction in A2 lands on top of
that.

This does not kill the pooled design — ~40 event days observed through 24
correlated lenses still carries information. It does mean the honest description
of where power comes from is "≈40 independent event days per instrument-session,"
not "tens of thousands of trades."

### B2. Only 57 FOMC decisions exist in the sample window

Sourced from the Fed's own archives, not memory
([2019 historical](https://www.federalreserve.gov/monetarypolicy/fomchistorical2019.htm),
[2020 historical](https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm),
[current calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)):
**57 decision dates** fall in 2019-04-01 → 2026-04-30. Of those, **55 are
pre-holdout** (holdout starts 2026-02-01), and after weekends/data gaps you land
near ~40 usable session-days per instrument-session. That is the entire budget
for the cleanest channel in the design. Plan accordingly.

*(Measured after the gate ran: 47 of those 55 survive into the traded date axis,
median 36 contaminated days per instrument-session, min 11. See §H.)*

### B3. Two artifacts disagree with each other

`outputs/summary.parquet` reports **2** survivors of 7,529 variants;
`outputs/holdout_verdict.json` reports **91** survivor variants across **48**
families out of 1,190 rankable. These are from different runs. Before the Phase C
step-13 regression gate can mean anything, the baseline must be **pinned and
hashed into version control** — otherwise "bit-identical to the pre-change run"
compares against a moving target. Right now there is no single agreed baseline to
be identical to.

---

## C. Design corrections — worth making, not blocking

### C1. Stratify the permutation on weekday, not just volatility tercile

FOMC decisions land on **Wednesday** in essentially every case (Tuesday–Wednesday
meetings, statement on day two). ECB/BoE are Thursday. A free permutation across
all days assigns contaminated labels to Mondays and Fridays — states that cannot
occur under the real calendar. That leaves the door open to a weekday effect
loading onto the event flag.

The ±7-day placebo does preserve weekday, which is a genuine strength of the plan.
But it is cleaner to block on weekday in the **primary** test and let the placebo
confirm, than to run a primary test with a known confound and lean on the placebo
to rescue it. `day_of_week` is already a `trade_log` column. Stratify on
`weekday × vol_tercile`.

### C2. `POST_EXIT_SAME_DAY` double-counts, and is wrong for TOK

Defined as `(exit, open + 24h]`. Two problems.

**Double-counting.** For NY that interval runs 12:00 ET to 09:30 the next day. An
08:00 ET Tuesday event is `PRE_OPEN` for Tuesday *and* `POST_EXIT_SAME_DAY` for
Monday. `derive_event_regime` hides this by emitting one label per day, but the raw
channel columns — which §2.2 says are "what the tests actually operate on" — carry
the event twice.

**TOK.** FOMC at 14:00 ET is 03:00–04:00 JST the **next** JST date. So the Tokyo
session on the same JST date sits **15–18 hours** ahead of the announcement, versus
NY's 2–4.5 hours. `POST_EXIT_SAME_DAY` will technically fire, and the plan will
call it "anticipation," but an 18-hour-ahead anticipation is not the same
phenomenon as a 2-hour-ahead one. The session that actually precedes an FOMC in
Tokyo is the *next* JST date.

**Fix:** define the channel as `(exit, local_midnight]`, and add an explicit
`hours_to_event` dose column. Dose is more informative than the binary anyway, and
it makes the NY/TOK asymmetry visible in the output instead of buried in a label.
Report TOK separately — which §11 already commits to for a different reason.

### C3. §4.4's exit bound drifts with data availability

```python
exit_utc_ns = int(sd.bar_timestamps[-1]) + 60 * 1_000_000_000
```

`MIN_SESSION_BAR_COMPLETENESS = 0.90`, so an eligible day can be missing 10% of
its bars. If the tail is missing, `bar_timestamps[-1]` is the last *available*
bar, not the scheduled exit — and the channel boundary moves because of a data
gap. For `PATH` that's arguably right (you can't trade an absent bar). For
`POST_EXIT_SAME_DAY` it silently shifts the window.

Derive the channel bound from the `SESSIONS` table (all three sessions exit 12:00
local) so the channel definition is a property of the session, not of coverage.
Keep `bar_timestamps` for the open, where §4.4's reasoning is correct and the DST
argument holds.

### C4. `EVENT_TIERS_ACTIVE` is dead config

Declared in §5.1 as `("T1", "T2")`, never read by `derive_event_regime` (§4.3),
which hardcodes `T1` and collapses T2/T3 to `MINOR`. Either wire it in or delete
it. Dead config that looks live is how a run gets misdescribed in a report.

Related: §4.3's precedence means a day with a T1 event in range **and** a T1
presser after the exit is labelled `EVENT_IN_RANGE`, and vanishes from the
anticipation count. Fine for the derived label, but the dose–response table in
§10 must be built from the raw channels or it will misattribute days.

### C5. 2020-03-03 needs its own flag

The Fed's 2020 archive shows an **unscheduled** meeting on March 2 with the
statement released **March 3** — the emergency 50bp cut, announced ~10:00 ET,
which is **inside** the NY window. It is the only genuinely violent in-path FOMC
event in the sample, and it sits in the most extreme month in the sample.

One day cannot be allowed to carry a pooled result. Flag it, pre-register its
treatment, and report the headline both with and without it. (2020-03-15 was a
Sunday — no session — so it drops out naturally.)

---

## D. What the plan gets right — keep these untouched

Worth naming explicitly, because the corrections above shouldn't obscure that the
core design is good.

- **§1.4, the session-clock inversion.** This is the insight that makes the work
  non-trivial. FOMC at 14:00 ET is *outside* 09:30–12:00, so the obvious
  "drop days where an event overlaps the trade window" filter catches almost
  nothing on your two most-traded sessions, finds nothing, and buries a live
  hypothesis. Recognising anticipation as the operative channel is the difference
  between a real test and a null result caused by a design error.
- **`IMPULSE` as a deliberate control arm (§2.1).** Falsifiable by construction.
  If degradation appears equally on impulse days, the narrative mechanism is dead
  and you learn that instead of confirming yourself.
- **Vol-profile validation before P&L interpretation (§6.3).** Checking that the
  hand-labelled taxonomy corresponds to genuinely different path shapes, *before*
  reading any P&L number off it, is the right order and an easy thing to skip.
- **Rule 2, null-pool inheritance (§5.2).** Correct and decisive. Filtering
  observed days without filtering the null pool measures day-mix, in the
  flattering direction.
- **Placebo before belief (§6.4).** With roll artefacts, month-end, quarter-end and
  expiry all calendar-locked — and `config.py` already documenting 6E/6J
  post-expiry and ZN delivery-month degradation — this is not a formality.
- **§11's expectation-setting.** "This work sharpens the question; it does not by
  itself rescue the thesis." That is the correct frame and it is stated before the
  run rather than after.

---

## E. Sequencing — the main disagreement

Phases A–E are three new modules, 10 `SessionDay` fields, 6 journal columns,
surgery on `filters.py` / `null_calibrator.py` / `report.py`, a 7-test validation
suite, and a day of manual calendar sourcing — **before a single number arrives.**

Meanwhile the plan already tells you what to expect. §7: the sharpest channels are
3–4% of days. §11: "the most likely outcome is that clean-day expectancy is also
indistinguishable from zero, merely less negative." And `holdout_verdict.json`
already reads `NO EDGE ESTABLISHED`, trade-weighted net **−0.0074 R**, CI
[−0.0269, +0.0454], 50.0% of families net-positive out-of-sample — a coin flip.

So build the cheapest decisive slice first and let it gate the rest.

### The gate: FOMC-only, ANTICIPATION-only, day-level

| | |
|---|---|
| **Calendar cost** | 57 dates. Date only — no end times, durations, geometry labels, `affects_symbols`, tiering. Fed archives, ~20 minutes, versioned CSV + builder script. |
| **Pipeline cost** | Zero. `groupby(date)` on the existing `trade_log.parquet`. No `SessionDay` fields, no filter changes, no null-calibrator surgery. |
| **Channel** | `POST_EXIT_SAME_DAY` — the channel §1.4 identifies as dominant for NY/LDN, and the plan's central structural claim. |
| **Why this slice** | FOMC + Powell presser is the purest `DIFFUSE` anticipation case in the calendar. Every decision in the sample has a presser (Powell moved to all-meeting pressers in Jan 2019). If narrative-driven macro degrades ORB continuation, this is where the effect is largest and cleanest. |

Same day-permutation null as §6.1, same MDE discipline, corrected per A1/A2 and
stratified per C1. If the **strongest available version of the dominant channel**
shows nothing at day level with an honest MDE, the remaining ~250 calendar rows
will not change that — and you have spent an hour instead of a week.

**This is a feasibility gate, not a finding, and it is pre-registered as such.**
~40 usable days per instrument-session, one channel, in-sample. Its p-value does
not decide the hypothesis; the §6.2 Level-1 pooled test does. Its only job is to
decide whether Phase A is worth a day. Recording that distinction *before* running
is what stops a lucky gate result being promoted to a headline afterwards.

---

## F. Two inputs only you can supply

Both are §1.2/§1.3 requirements and both currently block the corresponding steps.
Neither blocks the gate, which is in-sample by construction.

1. **Paper-trading period start and end.** §1.2 calls resolving this "before
   writing any code," and it is the single most likely route to a confident wrong
   answer. Holdout starts **2026-02-01**; data ends **2026-04-30**; today is
   **2026-08-08**. A paper account running forward almost certainly overlaps —
   in which case the holdout is contaminated *for this hypothesis*, §6.5's test is
   not independent, and the honest move is §1.2 option (b): state it plainly and
   treat the exercise as in-sample. Currently `UNRESOLVED` in the pre-registration.
2. **The two originating family keys** —
   `(instrument, session, range_minutes, entry_mode, closure_tf, direction)`.
   Needed to hold them out of the pooled headline per §1.3/§6.2. Currently
   `UNRESOLVED`; the probe accepts an exclusion list and reports Level-0 separately
   once they're supplied.

---

## G. What landed on this branch

| Path | What |
|---|---|
| `EVENT_REGIME_PLAN.md` | The spec, now tracked (was gitignored) |
| `EVENT_REGIME_REVIEW.md` | This review |
| `prereg/event_prereg.json` | Pre-registration, tracked, committed **before** the probe ran |
| `data/events/fomc_decisions.csv` | 57 sourced FOMC dates, UTC-resolved, provenance per row |
| `scripts/build_fomc_calendar.py` | Regenerates the CSV; DST via `zoneinfo`, no hardcoded offsets |
| `scripts/probe_fomc_anticipation.py` | The gate (§E) |
| `prereg/probe_results.json` | Gate output |

Recommended order from here: read the gate result → resolve F1/F2 → decide Phase A.
Fix A1 and A2 before any Phase D code is written, regardless of what the gate says.

---

## H. Gate results

Run at `cc8df9b`, after the pre-registration was committed. Full output in
`prereg/probe_results.json`. Universe: 2,809,918 eligible pre-holdout trades in
1,129 rankable families across 21 instrument-sessions, 1,542 union dates, 47 FOMC
dates in the traded axis (median 36 contaminated days per instrument-session).

| test | stratify | delta net R | null mean | p | MDE |
|---|---|---|---|---|---|
| FOMC anticipation | none | **−0.0558** | +0.0000 | **0.024** | 0.071 |
| FOMC anticipation | weekday | **−0.0558** | −0.0261 | **0.136** | 0.067 |
| FOMC anticipation | weekday × vol | **−0.0558** | −0.0247 | **0.121** | 0.067 |
| ex-2020-03-03 | weekday | −0.0563 | −0.0272 | 0.144 | 0.067 |
| placebo +7d | weekday | +0.0003 | −0.0255 | 0.820 | 0.070 |
| placebo −7d | weekday | −0.0106 | −0.0258 | 0.712 | 0.067 |

### H1. The direction and magnitude match the pre-registration

Observed **−0.0558 R** against a predicted −0.05 R. Right sign, near-exact
magnitude, on a channel that was pre-registered before the run. Placebos are
clean (+0.0003 and −0.0106 against a −0.0255 null centre, p = 0.82 and 0.71), so
this is not the generic calendar artefact §6.4 and §11 warn about. Dropping the
2020-03-03 emergency cut moves the delta by 0.0005 — one day is not carrying it.

### H2. Half the apparent effect is Wednesday, not FOMC

This is the substantive result, and it only appears because of correction C1.

Unstratified, p = **0.024**. Blocking the permutation on weekday, p = **0.136**.
The delta does not move — the *null centre* does, from +0.0000 to **−0.0261**.
FOMC decisions are 44 of 47 Wednesdays in this sample, and Wednesdays are simply
worse days: about **47% of the −0.0558** is a weekday effect that the unstratified
test credits to FOMC.

Run as specified in §6.1 — stratifying on volatility tercile only — this reports
p ≈ 0.024 and reads as a confirmed pre-registered prediction. It isn't one. The
±7d placebo would not have caught it either, because shifting by exactly 7 days
*preserves* weekday by design; that property makes the placebo a good control for
month-position and a blind spot for weekday.

Worth stating plainly: had the plan been implemented as written, the most likely
outcome was a false positive on the headline channel, presented with a
pre-registered direction behind it.

### H3. The gate was underpowered, so this is not a null result

MDE ≈ **0.067 R** at 80% power. That is *above* both the observed 0.0558 and the
pre-registered 0.05. The gate could not have reliably detected the effect it was
predicting.

Per the pre-registered decision rule — "an underpowered null is uninformative, not
evidence of absence" — **Phase A proceeds.** The gate did its job: it says the
signal is plausibly there at roughly the predicted size, and that resolving it
needs the wider `T1 ∪ T2` net §7 prescribes. Note this is the honest MDE from the
permutation draws; §7's per-trade formula would have reported roughly 0.015–0.02
here and declared the gate comfortably powered.

### H4. One mechanism prediction is falsified

| metric | clean | contaminated | delta | prediction |
|---|---|---|---|---|
| reversal rate | 0.5323 | 0.5313 | **−0.0011** | higher — **fails** |
| MFE/MAE median | 0.9780 | 0.8400 | −0.1380 | lower — holds |
| TP hit rate | 0.4672 | 0.4404 | −0.0267 | lower — holds |

The reversal rate is flat. §1.3 pre-registered "higher post-breakout reversal rate"
as the *first* mechanism claim, and it is the one metric the plan called free
because TI detection already is a reversal detector. It does not show up.

TP-hit degradation depends on how you read §6.3's "concentrated at high rr":

| rr | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| absolute Δ | −0.021 | −0.018 | −0.024 | −0.031 | −0.031 | −0.026 |
| relative Δ | −2.8% | −2.8% | −4.7% | −7.2% | −9.8% | **−11.1%** |

Absolute deltas are flat-to-humped; relative degradation steepens monotonically.
The plan did not pre-specify which, so both are reported rather than the flattering
one. **Fix the metric in the pre-registration before Phase D**, because choosing
after the fact is exactly the freedom pre-registration is meant to remove.

Net: continuation does fail harder at high rr in relative terms, but it does not
fail by tapping back into the range more often. That is a real strike against the
stated mechanism, and it raises the §0/§11 alternative — thin pre-event tape rather
than narrative-driven reversal. The vol-profile validation (§6.3), which needs 1m
bars, is now the load-bearing test and should run early in Phase B rather than
late.

### H5. The thing that matters more than any of the above

Clean-day mean is **−0.0920 R per trade**. Contaminated is −0.1542 R.

Both are firmly negative. The event effect is real in direction and roughly the
predicted size, and it is a rounding error against the level. Removing every FOMC
day from the sample leaves a strategy losing ~0.09 R per trade before any event
conditioning is applied.

§11 predicted this: *"the most likely outcome is that clean-day expectancy is also
indistinguishable from zero, merely less negative."* The measurement is worse than
that — clean-day expectancy is not indistinguishable from zero, it is reliably
negative. **No event filter rescues this strategy.** Phase A is worth funding to
answer the mechanism question honestly, and it should not be funded on the
expectation that it recovers the edge.

### H6. Loose end

8 of the 55 pre-holdout FOMC dates are absent from the traded date axis (47
present). Probably holidays and coverage gaps, but I did not verify it, and 15%
attrition on the scarcest input is worth ten minutes before Phase A.
