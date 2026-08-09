# HYP-03 Findings — The Opening Range Is Not a Special Level

**Status:** CLOSED — placebo designs were under-powered; level specialness is UNTESTED  
**Strategy verdict unchanged:** gross expectancy negative before costs; arithmetic ceiling below friction floor  
**Pre-registration:** `prereg/hyp03_level_specialness_prereg.json`  
**Results:** `outputs/hyp03_results.json`  
**Diagnostic:** `tools/diagnostics/hyp03_width_check.py`  
**Committed:** `hypothesis/level-specialness` → merged to `main`  
**Depends on:** a5bdc2b (holdout-pin fix), 8f54e7e (comm_ticks fix), HYP-04 complete

---

## §0 — What the placebos measured, and what they didn't

Two placebo designs were run. Both were defective in ways that make the trigger-rate
comparison uninformative. The strategy conclusion (negative expectancy regardless) is
unchanged. The specific claim "the level is not special" is NOT supported.

| Placebo | Defect | Consequence |
|---------|--------|-------------|
| WIDTH | Trigger rate saturates — price travels more than one range-width in almost every post-range window | Zero discriminating power; fires on identical days as real |
| SHIFT | Detection window is 60 minutes shorter (84 vs 144 bars for NY rm=5, a 42% reduction) | Confounds level location with time-at-risk; negative deltas are a window-length artefact |

The correct statement is:

> **The trigger-rate test was saturated and therefore uninformative. Level specialness
> is untested. The strategy's expected value is negative before costs regardless, so the
> question is moot for this programme.**

---

## §1 — The WIDTH diagnostic

The WIDTH result showed 0.00pp difference for all 25 instrument-sessions, including
BTC/LDN (60.3% trigger rate), BTC/TOK (58.3%), and ETH/TOK (58.3%). Before accepting
this as an empirical finding, two candidate explanations were distinguished:

**(a) Ceiling effect** — both arms near-saturate; the delta is structurally bounded near zero.  
**(b) Band identity** — the WIDTH band is not relocating, so both arms are the same test.

The diagnostic (`tools/diagnostics/hyp03_width_check.py`) ran a contingency table for
each sub-saturated session:

| Session | Trigger rate | Days | Both trigger | Real only | WIDTH only | Neither |
|---------|-------------|------|-------------|-----------|------------|---------|
| BTC/TOK | 58.3% | 49 | 49 (100%) | 0 | 0 | 0 |
| ETH/TOK | 58.3% | 60 | 60 (100%) | 0 | 0 | 0 |
| BTC/LDN | 60.3% | 47 | 47 (100%) | 0 | 0 | 0 |

The WIDTH bands are visibly relocating on every sample day (Δrh/rl from −250 to +102
ticks on BTC/TOK, −6.25 to +2.25 ticks on ETH/TOK). Explanation (b) is eliminated.

**Explanation (a) is confirmed.** On days that trigger under real, price has moved far
enough to cross the WIDTH band as well — because price travels more than one range-width
from the range-end close in almost every post-range session. On days that don't trigger,
price moves so little it doesn't cross either band. The location of the boundary is
irrelevant once volatility determines whether the session is a trigger day or not.

**For BTC/LDN specifically:** 47 days, zero discordant days, despite a non-trivial
40% non-trigger rate. This is the most informative single data point: a session with
meaningful non-trigger frequency, clearly relocated WIDTH bands, and still perfect
correlation. It confirms that "did this day's price break some band of this width?" is
entirely a function of that day's volatility, not of where the band is located.

### The reframe this implies

ORB is not a rare-setup strategy. It fires near-100% of the time for most
instrument-sessions. The opening range is not selecting *when* to trade — it is
determining *direction* and *stop placement* only. What was tested across six
hypotheses was not "does this setup identify good times to take a trade" but "given a
trade taken every day, does the opening-range-specific direction and stop predict
forward returns?"

---

## §2 — The SHIFT result, corrected

SHIFT showed mixed deltas (−26pp to +6pp). The pre-registered interpretation was "real
fires more than shifted → level is special where positive." This is wrong.

SHIFT's detection window is `exit_min − (open_min + 60 + rm)` vs real's
`exit_min − (open_min + rm)`. For NY session, rm=5:
- Real: 144 post-range bars
- SHIFT: 84 post-range bars — **42% fewer**

With 42% less time, the SHIFT arm fires less often on purely mechanical grounds. The
negative deltas for BTC/TOK (−19pp) and ETH/TOK (−26pp) where SHIFT fires MORE are
interesting: the hour after the Tokyo open is more active for crypto than the open
itself. But this is also confounded by window length — it would be a finding on its
own only if both arms had equal time-at-risk.

A non-confounded SHIFT test would hold time-at-risk constant: compare the same
window length at two different starting offsets. This was not pre-registered and
is not run here.

---

## §3 — What a valid test would require

Both placebos were specified at the wrong level of the analysis. Trigger rate is
determined by daily volatility, not by level location, at the window lengths used.

A non-saturating metric exists in the trade log already:
- **Time-to-first-break** — conditional on triggering, is the real level broken sooner
  than the placebo? (Regime-robustness: uses only triggered days)
- **Tap-in rate** — does the real range attract retests before continuation?
- **Post-break path efficiency** — conditional on a break, does price travel farther in
  the breakout direction from the real level than from a same-width placebo?

All three are computable from the existing trade log without re-running the engine.
The pre-registration did not include these metrics; running them now would be exploratory.
If run, they would need a new pre-registration before being used as evidence.

---

## §4 — Implications for prior hypotheses

The design failure changes the *characterisation* of what was tested, not the
*outcome* of the programme:

- **HYP-01:** Selection premium falsified. This result stands — it measured whether
  selection discrimination persists out of sample, independent of whether the level is
  special.
- **HYP-02:** Closed on arithmetic (gross expectancy negative before inversion). Stands.
- **HYP-04:** Range quality / horizon tested. Stands — these were conditioning a given
  signal, not testing level specialness.
- **HYP-05:** Closes on arithmetic ceiling (§6.3), not on this gate. Documented in
  `prereg/hyp05_closure.json`.

---

## §5 — Limitations and what remains open

**Level specialness is genuinely untested.** The most direct way to test it —
conditional metrics (path shape, retrace rate, time-to-break) on triggered days — was
not pre-registered and has not been run.

**ROTATE is still deferred.** ROTATE (OHLC-preserving circular rotation, IMPROVEMENTS
§6.8) tests path shape rather than trigger rate and would not suffer from the saturation
problem. It remains the cleanest available placebo. Whether running it is worthwhile
depends on whether the conditional metrics above show anything; if path shape is also
null, ROTATE adds nothing.

**The resolution floor is the binding constraint.** With ~6 effective clusters
(asset-class level), the floor on detecting any cross-instrument pattern is roughly
±0.03–0.05 R. Every effect explored in this project sat at or below that floor. This
is a property of the universe size, not of the analysis, and limits what any future
ORB study on these ten instruments can conclude.

---

## §6 — Conclusion

The HYP-03 placebo tests were mis-specified. Neither arm measured what was intended:
WIDTH saturated to zero discriminating power; SHIFT confounded level location with
time-at-risk.

The correct record is: **level specialness is untested**. The trigger-rate test was the
wrong metric for this question.

The strategy conclusion is unchanged and rests on independent arithmetic:

- Pooled gross expectancy: −0.014 R
- Population holdout net R: −0.0957 R (CI entirely below zero)
- Trade-weighted survivor net R: −0.0451 R (CI [−0.0794, −0.0115], excludes zero)
- Best-case arithmetic (HYP-05 §6.3): −0.046 R ceiling vs 0.062 R required for ES

The project closes on those numbers, not on a level-specialness test that lacked power.
