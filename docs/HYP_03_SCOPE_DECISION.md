# HYP-03 scope decision

## The tension

HYP-01's pre-registered kill rule says: *"premium CI includes zero → selection is
noise, close HYP-01/02/03."* That criterion fired.

HYP-03's own rationale says: *"the only test in the set that can definitively close
the project… a negative result is definitive in a way a positive expectancy result
never is."*

Both are true, and they point in different directions. Resolving the tension requires
being precise about what question HYP-03 answers.

## What HYP-03 actually tests

Whether the opening-range boundary carries information relative to a mechanically
identical placebo band. It does **not** test the same claim as HYP-01. HYP-01 asks
whether selection discrimination persists out of sample — a question about the
quality of a sweep on top of a signal. HYP-03 asks whether the signal itself is
special. You can have real discrimination (HYP-01 TRUE) built on a worthless level
(HYP-03 FALSE), or a meaningful level (HYP-03 TRUE) that selection failed to exploit
(HYP-01 FALSE). The two hypotheses are logically independent.

HYP-01 failing its kill criterion therefore does not answer HYP-03's question. The
kill rule was designed to prevent spending a week on a question whose premise had
just been disproved. But HYP-03's premise is the opening-range level itself, which
HYP-01 says nothing about.

## What running HYP-03 costs

- **SHIFT / PRIOR / WIDTH placebos**: these are `build_session_days` with different
  range bounds, then the existing pipeline. Each is roughly one sweep (~7 minutes
  with the GPU/CPU launcher). Three sweeps × 7 minutes ≈ **21 minutes of compute**,
  zero dollars.
- **ROTATE**: needs the OHLC-preserving tuple decomposition (spec IMPROVEMENTS §6.8)
  which is not yet implemented. Budget **one day of development time** plus ~7 minutes
  of compute. The spec calls ROTATE "the strictest and most informative" of the four;
  it is also the only one that definitively separates level specialness from
  volatility-regime effects. Without it, a SHIFT/PRIOR/WIDTH comparison can be
  confounded by the range's first-bar volatility content.

## Decision: run SHIFT + WIDTH; defer ROTATE

Run the two cheapest placebos that are unconfounded:

**SHIFT** destroys the privileged status of the open — if breakout rate above a
band starting 60 minutes after the range matches the real level's breakout rate, the
open is not special. SHIFT is the most direct test of whether timing matters.

**WIDTH** destroys the range's location while preserving its width — if a
same-width band centred on the range-end close performs identically, range *width*
(stop and target size) explains the whole result and the *boundary location* carries
nothing.

**PRIOR** inherits yesterday's volatility, which correlates with today's — it is the
weakest of the four placebos and explicitly described as a lower bound. Skipping it
loses nothing.

**ROTATE** is the decisive one. Defer it rather than skipping it: if SHIFT or WIDTH
beats the real level, or if either is statistically indistinguishable from it, the
decision to run ROTATE has a clearer answer. If SHIFT and WIDTH are both worse than
the real level (level is special), ROTATE is needed to close the argument because
SHIFT and WIDTH leave the volatility-regime confound open.

## What to measure

Per the spec, **trigger frequency** is the most informative statistic and the least
confounded by friction — if the real range is broken no more often than a random band
of the same width, the premise fails before expectancy is consulted. Run:

1. Breakout rate: fraction of days with at least one trigger
2. Path efficiency after trigger (median |exit − entry| / session range)
3. E[gross_r] per (instrument, session) — the same table HYP-01 §3.2 built for the
   real level

All compared paired on (instrument, session, date) against the placebo. Permute the
real/placebo label at the day level; joint draws across instruments so the maxT
correction is valid.

This is the smallest scope that answers the question the spec asks for SHIFT and
WIDTH, and it reuses the permutation machinery from `src/events`.

## Implementation

Create branch `hypothesis/level-specialness`. Build placebo levels in
`src/range_builder.py` as new `mode` parameters, defaulting to `"real"`. Add
`tools/analysis/hyp03_level_specialness.py`. Pre-register before running the sweeps.
ROTATE is left as a TODO with the spec reference (IMPROVEMENTS §6.8).

## Timeline

SHIFT and WIDTH re-simulation: ~14 minutes of compute. Analysis: ~2 hours of
development. ROTATE if warranted: 1 development day.
