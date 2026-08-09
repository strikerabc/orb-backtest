# HYP-03 Findings — The Opening Range Is Not a Special Level

**Status:** CLOSED — gate-level test falsifies the premise  
**Pre-registration:** `prereg/hyp03_level_specialness_prereg.json`  
**Results:** `outputs/hyp03_results.json`  
**Committed:** `hypothesis/level-specialness` branch  
**Depends on:** a5bdc2b (holdout-pin fix), 8f54e7e (comm_ticks fix), HYP-04 complete

---

## §0 — Result summary

The SHIFT and WIDTH trigger-rate tests both fail to detect any specialness in the
opening-range level. Neither placebo is significantly different from the real range
on the pre-registered primary statistic (trigger rate). The HYP-05 gate fails, and
the cross-sectional programme closes.

| Test | Significant instrument-sessions (Holm p < 0.05) |
|------|-------------------------------|
| SHIFT (60-min)  | 0 of 25 |
| WIDTH (same-width, different location) | 0 of 25 |

---

## §1 — The WIDTH = 0 result

The most informative finding is the WIDTH column, which shows **exactly 0.00pp
difference** for every single instrument-session.

The WIDTH placebo replaces the real range's location with a band of the same width,
centred on the range-end close price. If the ORB level's location carried
information — if price is attracted to the opening boundary in a way that makes
breakouts more likely there than at an alternative same-width location — the WIDTH
placebo should fire less often.

It does not. WIDTH fires on exactly the same days as the real range, for all 25
instrument-sessions.

**What this means:** whether a given session-day's price crosses a level of a given
width is determined by that day's volatility, not by where the boundary is located.
The ORB level is indistinguishable from a same-width band at a different location.
The boundary location adds no information beyond what the range width already
encodes (and range width is a proxy for intraday volatility).

**Ceiling effect (partial caveat):** eleven instrument-sessions have trigger rates
≥ 99%. At those rates the WIDTH test has essentially no power — both real and
placebo fire almost every day, and a genuine location premium could not be
detected. The informative sessions are the ones with sub-80% trigger rates:

| Instrument-session | Real | WIDTH | Δ (pp) |
|---|---|---|---|
| BTC/TOK | 58.3% | 58.3% | 0.0 |
| ETH/TOK | 58.3% | 58.3% | 0.0 |
| BTC/LDN | 60.3% | 60.3% | 0.0 |
| ETH/LDN | 76.9% | 76.9% | 0.0 |
| ZN/LDN  | 92.3% | 92.3% | 0.0 |

For BTC/TOK and ETH/TOK with 58.3% real trigger rates, if the level were special
we would expect WIDTH to fire substantially less often (it would have to keep its
centre far enough from the real boundary that the different location matters). The
fact that WIDTH fires on identical days here, even at these moderate trigger rates,
confirms that location is irrelevant at the session-day level.

---

## §2 — The SHIFT result

SHIFT moves the range window 60 minutes forward. Results are mixed — no pattern
of "real is better than shifted":

| Instrument-session | Real | SHIFT | Δ (pp) | Direction |
|---|---|---|---|---|
| BTC/NY  | 94.5% | 88.3% | +6.2 | real > shift |
| BTC/LDN | 60.3% | 55.1% | +5.1 | real > shift |
| 6J/LDN  | 96.2% | 93.2% | +3.0 | real > shift |
| ETH/NY  | 99.2% | 96.8% | +2.4 | real > shift |
| ZN/LDN  | 92.3% | 96.2% | −3.8 | shift > real |
| BTC/TOK | 58.3% | 77.4% | −19.0 | shift > real |
| ETH/TOK | 58.3% | 84.5% | −26.2 | shift > real |

None of the differences survive the joint Holm-corrected permutation test.

**The negative deltas (shift > real) for BTC/TOK and ETH/TOK are structurally
informative:** the range window shifted 60 minutes into the Tokyo session fires
substantially more often than the real open range. This means the Tokyo open is
*not* the highest-volatility window for BTC and ETH — the hour after the open
is more active. This is consistent with crypto markets being near-continuous;
the 09:00 JST CME open is an artifact of when the contract begins trading on
that exchange, not a volatility event in the underlying.

---

## §3 — Implications for the prior hypotheses

**HYP-01:** The selection premium was tested on families built on top of this
signal. The signal itself does not break out more often than a placebo. This is
consistent with the HYP-01 finding that the pooled gross expectancy (−0.014 R)
is indistinguishable from zero: a placebo signal of the same width would have
the same trigger rate and, plausibly, a similar gross expectancy.

**HYP-04:** The range-quality conditioner (range_width / atr_4h) was explored as
a predictor of forward expectancy. The WIDTH result here says the level location
is not special. Range width encodes intraday volatility, so the conditioner may
pick up a volatility effect rather than a genuine breakout quality signal.

**HYP-02/HYP-05:** Both close unrun. HYP-02 depended on a signal that this
test cannot distinguish from a random location. HYP-05 was gated on HYP-03
explicitly — the gate fails.

---

## §4 — Limitations

**Power at near-100% trigger rates.** For equity-index and forex sessions, the
trigger rate is 99–100%. At those rates this test cannot detect level specialness
even if it exists. The conclusion "level is not special" is well-supported for
BTC and ETH; it is underpowered for ES, NQ, 6E, and similar.

**Trigger rate is the right primary statistic but is not the whole story.** A level
could be special in expectancy (path shape after crossing) without being special in
trigger rate. This test does not address that. The scope decision judged that trigger
rate is the most informative gate because: if the level does not attract breakouts,
the path-shape question is moot.

**ROTATE is not run.** The ROTATE placebo (OHLC-preserving circular rotation,
IMPROVEMENTS §6.8) is the strictest test and addresses the volatility-regime
confound that SHIFT/WIDTH leave open. Given the WIDTH = 0 result, running ROTATE
would test path shape, not level attraction. Deferred unless a specific path-shape
hypothesis is separately pre-registered.

---

## §5 — Conclusion

The ORB level is not special at the level of "does it fire?" A same-width band at
a different location fires on the same days. The opening-range boundary location
adds no detectable information beyond what the range width encodes.

The project's cross-sectional programme (HYP-03/05) closes here.

The ORB strategy has now been tested from multiple angles across this project:

| Test | Outcome |
|---|---|
| In-sample selection (HYP-01) | FALSIFIED on primary; selection is noise |
| Signal inversion (HYP-02) | Closed unrun (gross expectancy negative before costs) |
| Level specialness (HYP-03) | CLOSED — level fires same as placebo |
| Range quality conditioner (HYP-04a) | FALSIFIED |
| Horizon shortening (HYP-04b) | Survives in-sample only; holdout CI includes zero |
| Instrument heterogeneity (HYP-05) | Closed unrun (gate failed) |

**The honest summary:** This ORB strategy as configured does not demonstrate edge.
The negative result is clean: the signal fires no more often than a placebo, in-
sample selection does not persist out of sample, and costs are not the binding
constraint (gross expectancy is already negative). A well-characterised negative
result is a genuine research output.

**One open item:** The HYP-05 pre-registration noted that the trade_count ≥ 400
conditioner (exploratory, monotone, survives clustered bootstrap) needs its own
out-of-sample test. This is outstanding but requires new data not yet collected.
It cannot be run on the current holdout period because the scan touched it (see
`ex_ante_conditioner_status` in `outputs/hyp01_results.json`).
