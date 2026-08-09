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

### H4. Mechanism metrics — one result retracted, see §I3

| metric | clean | contaminated | delta | prediction |
|---|---|---|---|---|
| reversal rate | 0.5323 | 0.5313 | −0.0011 | **metric invalid — see §I3** |
| MFE/MAE median | 0.9780 | 0.8400 | −0.1380 | lower — holds |
| TP hit rate | 0.4672 | 0.4404 | −0.0267 | lower — holds |

**The reversal-rate row is withdrawn.** I originally read the flat −0.0011 as
falsifying §1.3's first mechanism claim. That was wrong twice over: the metric
does not measure reversal (§I3), and the channel tested is not the one the
prediction is about (§I2). The prediction is **untested**, not failed.

TP-hit degradation depends on how you read §6.3's "concentrated at high rr":

| rr | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| absolute Δ | −0.021 | −0.018 | −0.024 | −0.031 | −0.031 | −0.026 |
| relative Δ | −2.8% | −2.8% | −4.7% | −7.2% | −9.8% | **−11.1%** |

Absolute deltas are flat-to-humped; relative degradation steepens monotonically.
The plan did not pre-specify which, so both are reported rather than the flattering
one. **Fix the metric in the pre-registration before Phase D**, because choosing
after the fact is exactly the freedom pre-registration is meant to remove.

Net: continuation fails harder at high rr in relative terms. Whether it fails *by
reversing* is unmeasured, because no valid reversal metric exists in the trade log
yet (§I3). The vol-profile and path-efficiency work (§6.3), which needs 1m bars, is
therefore load-bearing rather than confirmatory, and should run early in Phase B.

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

---

## I. Update: §1.2 and §1.3 inputs resolved (2026-08-08)

Paper period **2026-07-25 → 2026-08-07**. Originating setup **EURUSD / NY / 5m /
CC / 5m / long+short → `6E/NY/5/CC/5/{long,short}`**. Pre-registration amended
(separate block, own commit); gate re-run with Level 0 split out.

### I1. The holdout is clean — better than §1.2 anticipated

The paper period falls entirely **after** `data_end` 2026-04-30, so it never
touches the 2026-02-01 holdout. I predicted overlap was near-certain and that was
wrong.

This is stronger than the plan's best case. §1.2 hoped the hypothesis was generated
outside the *fitted windows*; it was generated outside the **sample entirely**.
§6.5's holdout test is genuinely independent for this hypothesis and is now live.

The converse cuts the other way and should be stated: because the paper period sits
outside the sample, **the backtest cannot test the observation on the data that
produced it.** Confirming it on its own evidence means extending the dataset past
2026-04-30. If you do that, flag 2026-07-25 → 2026-08-07 as observation-origin and
exclude it from confirmatory tests exactly as `originating_families` are — otherwise
the extension re-imports the circularity §1.2 exists to prevent.

### I2. The gate tested the wrong channel for your mechanism

You describe a speaker talking **while the position is open**, direction switching
on each macro hint. That is `DIFFUSE_IN_PATH` — the PATH channel. The gate tested
`ANTICIPATION` (`POST_EXIT_SAME_DAY`), because FOMC 14:00/14:30 ET is outside the
09:30–12:00 ET window.

So the gate did not test your mechanism, and §2.1 predicts *different* signatures
for the two channels — suppressed participation and weak follow-through for
anticipation, sustained vol and repeated direction changes for diffuse. A flat
reversal result on the anticipation channel says nothing about the diffuse channel.

**Sharper, for the paper window specifically:** FOMC 2026-07-29 falls inside it, but
the presser runs 14:30–15:30 ET — **2.5 hours after the 12:00 ET NY close.**
Whatever you watched intraday was not the FOMC presser. In-window candidates are
10:00 ET releases (ISM, JOLTS, consumer confidence, UMich final — `IMPULSE`, the
control arm) and regional Fed or other central-bank speakers with mid-morning
remarks (`T2`/`T3`, `DIFFUSE`). Which of those it was is now the highest-value
sourcing question in Phase A.

### I3. `tap_in_bar_idx` is not a reversal detector — §6.3 defect

§6.3 lists reversal rate as **free**, on the reasoning that "TI detection *is* a
reversal detector." Measured across all 5.29M trades:

| entry_mode | n | reversal rate |
|---|---|---|
| CC | 1,743,276 | **0.0** |
| II | 686,874 | **0.0** |
| R-CC | 1,588,656 | **1.0** |
| R-II | 617,988 | **1.0** |
| TI | 657,462 | **1.0** |

Exactly 0 or exactly 1, per mode. `tap_in_bar_idx` is a **deterministic function of
entry mode**, constant within every family. Pooled across modes it measures
entry-mode *mix*, not path shape. My reported 0.5339 → 0.5327 was the mode mix
holding steady across FOMC and non-FOMC days — a mild sanity check that day labels
aren't correlated with mode, and nothing about reversals.

Consequence for the plan: **the one mechanism metric §6.3 calls free does not
exist.** Testing reversal needs a genuine path-shape measure from 1m bars —
boundary crossings or path efficiency, both already specified in §6.3 but both
requiring the bar re-walk. Budget for it rather than expecting it for nothing.

### I4. Level 0 — your setup shows the effect *reversed*, and cannot resolve it

`6E/NY/5/CC/5/{long,short}`, 8,280 eligible pre-holdout trades, 32 FOMC dates,
reported apart and never pooled per §6.2.

| stratify | delta net R | null mean | p | MDE |
|---|---|---|---|---|
| none | **+0.1160** | −0.0016 | 0.800 | 0.345 |
| weekday | **+0.1160** | −0.0225 | 0.851 | 0.328 |

Clean −0.0952 R, FOMC-day **+0.0698 R** on 229 trades. The sign is **opposite** the
pre-registered direction, p = 0.80–0.85.

Do not read that as refutation. MDE is **0.33 R, roughly 6× the effect sought** —
229 trades in one instrument-session is the most underpowered cut in the file, and
the sign is almost certainly noise. It is uninformative in both directions.

What it does establish: across seven years and 32 FOMC decision days, **your
specific setup shows no FOMC-day degradation — nominally the reverse.** That does
not contradict what you saw over ten days on a different channel. It does mean the
observation gets no corroboration from its own family's history, which is precisely
why §6.2 keeps Level 0 out of the headline. The pooled result barely moved when
these two families were removed (−0.0558 → −0.0559), so the headline was never
resting on them.

*Also structurally degenerate at Level 0: MFE/MAE median is exactly 1.0000 on clean
days. Not chased. Suspect the same class of problem as I3 and worth a look before
these metrics are trusted anywhere.*

### I5. Phase A is rescoped — and it is a power upgrade

§9 builds T1 first, T2 optional. Your mechanism lives in **T2/T3 in-window
speakers**, so invert it: source in-window speaker events for **6E/NY** first.

This makes the design *stronger*, not weaker. Fed speeches alone exceed **200/year**
against 8 FOMC decisions, so in-path exposure covers a far larger share of days. The
gate's MDE of 0.067 R was an artefact of testing the **rarest event class available**
(47 dates), not a ceiling on the design. `EVENT_TIERS_ACTIVE` already defaults to
`("T1", "T2")` — that config, dead per C4, becomes load-bearing.

Revised order: T2 in-window speakers for 6E/NY → path-efficiency and vol-profile
metrics from 1m bars (I3 makes these mandatory, not optional) → `DIFFUSE_IN_PATH`
test on 6E/NY → widen to `T1 ∪ T2` across all families for the §6.2 headline.

---

## J. Data extension to 2026-08-07, and the 2026-08-07 case study

### J1. The April cutoff was a stale run, not a data limit

The caches already held everything through **2026-07-31/08-02**. `trade_log.parquet`
stopped at 2026-04-30 because the pipeline run was old. So May → July was already on
disk and free; only 2026-08-01 → 08-08 had to be bought.

| item | cost |
|---|---|
| Aug 1–8 gap, all 10 instruments (1m + 1d) | $0.2355 |
| ES/NQ May 1 → Aug 8 (see J3) | $0.7035 |
| **total spent** | **$0.939** of ~$32.14 |
| avoided by not re-downloading from scratch | ~$17.30 |

All 10 instruments now reach **2026-08-07**. Caches backed up to
`data/_backup_pre_august/` first; pre-extension outputs pinned and hashed to
`outputs/_baseline_pre_extension/BASELINE_HASHES.json`, which also closes the B3 gap
(there was previously no single artifact set for the Phase C gate to be identical to).

### J2. Extending the data would have destroyed the clean holdout — silently

`regime_sampler.select_windows` computed `holdout_cutoff = data_end - HOLDOUT_MONTHS`.
The holdout was therefore a *function of how much data happened to be loaded*.

Moving `data_end` to 2026-08-07 slides the holdout to **2026-05-07 → 2026-08-07**,
which **swallows the paper period** (2026-07-25 → 2026-08-07). The holdout would have
contained the very observation that generated the hypothesis — re-importing the
circularity §1.2 exists to prevent, as an invisible side effect of adding data. One
turn after establishing the holdout was clean, adding data would have contaminated it.

Fixed via `HOLDOUT_PIN_START = "2026-02-01"` in `config.py`, honoured in
`regime_sampler.py`. Verified:

| | fitted windows | last window ends |
|---|---|---|
| pinned, `data_end` 2026-04-30 | 10 | 2025-12-31 |
| pinned, `data_end` 2026-08-07 | 10 | 2025-12-31 — **bit-identical** |
| **unpinned**, `data_end` 2026-08-07 | 10 | **2026-03-31** — old holdout absorbed |

Net effect: the 2026-02-01 boundary is fixed, so the out-of-sample region is now
**2026-02-01 → 2026-08-07** — roughly six months rather than three, since the pin stops
the fitted windows advancing into the newly added data. Jul 25 → Aug 7 remains
observation-origin and is barred from confirmatory use, leaving **2026-02-01 → 2026-07-24**
usable for confirmation.

**Correction, and it was two errors in one sentence.** I previously wrote that
"May 1 → Jul 24 becomes a second independent OOS window (~59 trading days)" and then
checked for it in `trade_log.parquet`, where it is empty. Both halves were wrong:

1. **Wrong artifact.** `test_holdout.py` is a separate script — `HOLDOUT_START` is
   hardcoded at line 74, it filters `timestamp >= HOLDOUT_START` at line 141, and calls
   `build_session_days` directly at line 148. Holdout trades are never written to the
   trade log, which by construction contains **only fitted-window trades**. No sweep
   configuration would ever have put an OOS window there.
2. **Not a *second* window.** With the pin, 2026-02-01 → 08-07 is one continuous OOS
   region, not the old holdout plus a new slice. Calling it "second" implied the
   original 3-month holdout was still separately bounded, which the pin makes untrue.

### J2b. The pin had a bug — mine — and no guard caught it

The first pinned run put **6,168 ETH trades inside the holdout region** (trade log
running to 2026-02-26, past the 2026-02-01 pin). Cause, in `select_windows`:

- `eligible_months` was `len(period_range(first_month, eligible_end, freq="M"))`, which
  counts the **boundary month as fully usable**. With `eligible_end = 2026-02-01` that
  credited all of February.
- Window ends are `start + REGIME_WINDOW_MONTHS − 1 day` and were **never clamped**
  against `eligible_end`.
- The only post-hoc assertion checked windows for **mutual overlap**, not for crossing
  the holdout.

ETH is where it surfaced because `data_start = 2021-02-08` gives it exactly 10 windows
in 61 counted months with **zero slack**, so W9 landed at 2025-09-01 → **2026-02-28**.
Every other instrument had spare months absorbing the miscount. A bug that only fires
on the one instrument with no slack, with no assertion covering it, is the kind that
survives indefinitely.

Fixed by counting only months ending strictly before `eligible_end`, and adding the
missing guard that raises if any fitted window satisfies `w.end >= eligible_end`.
Verified zero breaches across all ten instruments. Two honest consequences:

- **ETH now yields 9 windows, not 10**, with `History supports 9 non-overlapping
  windows, requested 10`. That is the correct answer — ETH genuinely lacks 60 clean
  eligible months before the pin, and previously reached 10 only by borrowing 27 days
  of holdout.
- **W9 moves 2025-12-31 → 2025-10-31 for the other nine**, because dropping the
  miscounted month reduces `spare` and redistributes the gaps.

Windows changed again, so checkpoints were invalidated and the sweep was re-run cold.

**Correction to an earlier claim in this document.** I previously wrote that the pin
reproduces "what the baseline actually ran." That holds for **ES/NQ only**. It is wrong
for the other eight instruments, and the reason matters.

`select_windows` is called **per symbol**, with that symbol's own `data_end`
(`main.py:116`, `runs/gpu_cpu_sweep.py:111`). At baseline time the caches were ragged:

| instruments | cache ended | unpinned cutoff | window 9 ended |
|---|---|---|---|
| ES, NQ | 2026-05-03 | 2026-02-03 | 2025-12-31 |
| RTY, CL, BTC, ETH, GC, ZN, 6E, 6J | 2026-08-02 | **2026-05-02** | **2026-03-31** |

So for eight of ten instruments the baseline's last fitted window ran three months
later than I claimed — which is precisely why the baseline trade log carries 143,478
trades in 2026 despite my asserting the fitted history stopped at 2025-12-31.

This also explains the small row delta. The pin does not *remove* fitted data, it
*relocates* the windows: 10 × 6 months is 60 months either way, so the total barely
moves (4,788 rows, 0.09%). Different dates, not less data. The earlier "enrichment
recomputation" hypothesis was unnecessary.

**What the baseline was actually doing is worse than the pin.** With per-symbol
cutoffs, "the holdout" was not one slice of history — it was up to ten different
boundaries, each set by how fresh that symbol's cache happened to be. A holdout whose
start date depends on download order is not a holdout. The pin makes it uniform, and
unpinned at today's data 6E's holdout would be 2026-05-07 → 08-07, containing the paper
period. The pin stays.

**Consequence for Phase C:** `summary.parquet` will **not** be bit-identical to the
pre-extension baseline for those eight instruments, and should not be expected to be.
The gate must be **re-baselined** against the first pinned run rather than pointed at
`prereg/baseline_hashes_pre_extension.json`. Those hashes remain useful as a record of
the ragged-window era, not as a regression target.

**Row delta — resolved.** The rebuild logs 5,289,468 trade rows against a baseline of
5,294,256, i.e. 4,788 fewer (0.09%) despite strictly more data, and the pinned run
produces 7,508 variants against 7,529. Both follow from window *relocation*, per the
correction above: the eight ragged-cache instruments had window 9 pulled back from
2026-03-31 to 2025-12-31, while total fitted span stayed at 10 × 6 months. Different
dates, near-identical volume. No enrichment-recomputation effect needs to be invoked,
and the ES/NQ `_mixed` vs `_db` caches were separately confirmed bit-identical
(2,456,056 and 2,453,542 rows, matching timestamps and closes).

### J3. A cache-path bug nearly cost $18

ES/NQ carry `has_local_data=True`, so `_cache_path` appends a `_db` vendor tag — but
the legacy fallback is gated on `not vendor`, leaving the existing
`ES_1m.parquet`/`NQ_1m.parquet` unreachable. The repo's own preflight (free) reported
`DOWNLOAD ES, NQ`, i.e. full 7-year pulls at roughly $18 rather than 7-day gaps.

Resolved by aliasing to the resolvable `*_c_db_1m.parquet` names after verifying the
files are pure Databento (they carry `_contract` and no `_source` — exactly what
`_download_databento` emits), then gap-filling from May. Cost $0.70 instead of ~$18.

Worth noting the near-miss: the 1d copies I made initially *shadowed* the already-
extended legacy 1d files — the same "config change appears to work and does nothing"
failure the `_cache_path` docstring warns about. Removed.

### J4. 2026-08-07 — your account holds up, and my first read was wrong

6E/NY, 5m range. Aggregate metrics across all 1,963 6E/NY sessions:

| metric | Aug 7 | median | pctile |
|---|---|---|---|
| path efficiency (full post-range) | 0.0441 | 0.0719 | 31.4 |
| boundary crossings | 9 | 9 | 48.4 |
| vol-profile ratio | 0.8364 | 0.7289 | 68.6 |
| **range width (ticks)** | **19** | **12** | **81.6** |
| session range (ticks) | 49 | 64 | 29.4 |

Read only that, and the day looks **IMPULSE-like** — ratio below 1.0, crossings dead
average. I initially read it that way. The segment breakdown says otherwise:

| window | efficiency | travel | ticks/min |
|---|---|---|---|
| 09:35–10:00 | **0.273** | 55t | 2.20 |
| 10:00–11:00 | **0.059** | 119t | 1.98 |
| 11:00–12:00 | **0.060** | 117t | 1.95 |

Efficiency collapses **4.6×** while the travel rate falls only **11%**. That is not
impulse decay — decay would collapse *travel*. It is sustained activity going
nowhere, which is precisely §2.1's `DIFFUSE` signature.

Ranked against all 1,962 comparable sessions on the **10:00–12:00 window alone**:

| metric | Aug 7 | median | pctile |
|---|---|---|---|
| path efficiency | 0.0126 | 0.0755 | **9.1** |
| net displacement | 3 t | 22 t | **6.1** |
| travel | 239 t | 299 t | 25.0 |

**236 ticks travelled, 3 ticks net, over two hours** — 9th percentile efficiency, and
travel at a fairly normal 25th percentile. Your description ("initial move expected,
everything after untradeable, including a short re-entry") matches the measurement:
the only two directional flips were 09:40 long and 09:56 short, both inside 21 minutes
of the range closing, after which the tape produced motion without displacement for
two hours. The onset at the 10:00 boundary is consistent with a ~10:00 ET speech and
Q&A.

**Partially resolved (sourced).** Reuters copy carrying Barkin quotes is stamped
**Fri 2026-08-07 09:29 CDT = 10:29 ET**, from a National Association for Business
Economics video presentation. That is an **upper bound**: he was already speaking by
10:29 ET. The measured efficiency collapse begins in the 10:00–11:00 block, so the
two are consistent — and the bound rules out a late-morning start.

The same page's calendar confirms the pre-open shock quantitatively: **NFP released
08:30 ET, payrolls 57 → −23, unemployment 4.2 → 4.1.** A negative payrolls print is a
large surprise, which corroborates the `PRE_OPEN_SHOCK` reading independently of the
range percentile.

**Still unresolved:** the wire timestamp bounds when he *had* spoken by, not when he
started, and no source distinguishes prepared remarks from Q&A. Your claim is
specifically about Q&A *answers*, which is the finer-grained event the sources do not
carry. Your own session notes remain the only record at that resolution.

And n=1 regardless: 2026-08-07 is observation-origin data and can never confirm the
hypothesis. Its legitimate use is metric validation and effect sizing.

### J4b. The real pipeline is gitignored

Every artifact in `outputs/` is produced by `runs/gpu_cpu_sweep.py`, launched via
`runs/start_gpu_cpu_sweep.ps1` — a process pool for the CPU simulation and statistics
stages, CuPy/CUDA for bootstrap calibration, checkpointed per stage. `main.py` is the
single-threaded path and is **not** what generated the baseline.

`.gitignore` line 4 is `runs/`, so none of it is in version control. I spent 1.5 hours
running the wrong entry point because the right one was invisible.

That is a reproducibility hole larger than anything else in this review. The
pre-registration, the calendar and the probe are all tracked, but **the program that
computes the numbers they are compared against is not.** A branch that cannot rebuild
its own artifacts cannot support the Phase C regression gate, and `manifest.json`'s
`git_sha` is misleading while the executing code sits outside the repo.

Recommend tracking the two sources — `runs/gpu_cpu_sweep.py` and
`runs/start_gpu_cpu_sweep.ps1` — while keeping logs, checkpoints, `*.pid` and
`gpu_cpu_sweep/` ignored. Small diff, and it closes the hole.

Related, and worth knowing before quoting timings: the `163.4s` in
`runs/gpu_cpu_sweep.stdout.log` is dated 2026-08-06 and sits alongside
`gpu_cpu_sweep.pre_resume.stdout.log` — it is the tail of a **resumed** run, not a cold
sweep. A `--fresh` run recomputes trade shards, stats and null pools from zero and
takes materially longer. Useful as a resume benchmark, not as full-sweep cost.

### J5. Two more plan defects this exposed

**`vol_profile_ratio` cannot discriminate on a mixed-channel day.** §6.3 uses mean
|1m return| in the final third over the first third. On Aug 7 the first third contains
the NFP-driven open, so the ratio reads 0.84 — *impulse-like* — on a session whose
post-10:00 behaviour is textbook diffuse. §6.3 makes this metric the taxonomy
validator that must run *before* any P&L interpretation, so a metric that inverts on
mixed days is load-bearing and wrong. **Windowed path efficiency is the real
discriminator** (9.1 vs 31.4 percentile — the windowed version separates, the
aggregate does not). Same for crossings: 9 total looks average, but they are
back-loaded, 2 in the first hour against 4 in the last. Recommend replacing the
absolute-return ratio with **efficiency by third**, and reporting crossings **per
window** rather than as a session total.

**`derive_event_regime` cannot represent a multi-channel day.** Aug 7 is
simultaneously `PRE_OPEN_SHOCK` (NFP 08:30 ET, T1 — and the range came in at the 81.6th
percentile, exactly the reference-level corruption §2.2 predicts) *and*
`DIFFUSE_IN_PATH` (Barkin, T2). §4.3's precedence emits one label, so the day is
classified by whichever fires first and the interaction is lost. That interaction is
plausibly the whole story here: a range inflated 19 ticks wide by a pre-open shock
makes the breakout threshold meaningless, and *then* the diffuse tape punishes the
re-entry. Neither channel alone explains it. Keep the raw channel columns as the test
inputs (§2.2 already says so) and add an explicit `n_active_channels` / channel-set
column, or multi-channel days will be silently misattributed in the dose–response
table.
