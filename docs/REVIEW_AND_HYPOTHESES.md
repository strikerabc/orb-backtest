Part 1 — Review
R1. The holdout CI looks wrong, and it matters
trade-weighted −0.0451 R, CI [−0.151, −0.045]

The point estimate sits exactly on the upper bound to three significant figures. That is not a coincidence a bootstrap produces. Candidates:

The CI is computed on a different statistic than the point estimate (family-level mean vs trade-weighted mean), and the two are being printed as a matched pair.
Percentile indexing is off by one end, or the interval is effectively one-sided.
One family dominates the trade weighting, pinning the upper tail.
Check: print the full bootstrap distribution's [2.5, 50, 97.5] percentiles alongside the point estimate and confirm the point estimate lands near the median, not the edge. Until then the interval should not be quoted.

R2. "Indistinguishable from a coin flip" understates the result
36 of 94 families net-positive. Under H₀ = 0.5: mean 47, sd 4.85, z ≈ −2.27, two-sided p ≈ 0.023. That is not indistinguishable from chance — it is significantly worse than chance. A significantly negative result carries exploitable information; a null result does not. The README currently discards the more informative reading.

R3. …but zero is the wrong baseline, and the right one may reverse the conclusion
Selection was performed on in-sample expectancy. The correct out-of-sample benchmark for the survivor set is not zero — it is the holdout-period unconditional mean across all rankable families. In-sample that unconditional figure is around −0.0885 R (your clean-day number).

If the holdout-period unconditional mean is also ≈ −0.09 and survivors came in at −0.045, then selection delivered roughly +0.04 R of persistent, out-of-sample discrimination. It simply started from too deep a hole to cross zero.

That is a completely different finding from "no edge," and it is one line of code away:

python
# tools/holdout_test.py
baseline = holdout_all_rankable["net_r"].mean()      # the correct benchmark
premium  = survivors["net_r"].mean() - baseline      # persistent selection value

Run

Run this before anything else in this document. It determines whether the project's headline should be "no edge" or "real discrimination, buried under friction," and those two conclusions point at entirely different next steps. HYP-01 is built around it.

R4. null_p_matched median = 0.978 is the strongest number in the project and it is sitting in a table cell
The median variant performs worse than random entry timing at its own matched stop distance. Combined with your null_calibrator docstring recording that the matched null sits at ≈ +0.0084 gross, the inference chain is:

matched null ≈ 0 gross
median variant < matched null  (p = 0.978)
⟹ the median breakout entry is gross-negative
⟹ range-boundary breakouts are anti-predictive, not merely unprofitable

Anti-predictive is a usable property. HYP-02 tests it directly.

R5. main cannot reproduce the committed evidence
HOLDOUT_PIN_START defaults to None on main while prereg/ was produced at "2026-02-01". Documenting this is right, but a reader running main gets different numbers than the committed artefacts. Either make the pin the default with the sliding boundary as an opt-in, or add a CI job that runs with the pin and diffs against prereg/baseline_hashes_pinned.json. Also: retire baseline_hashes_pre_extension.json explicitly — two baseline files is the exact ambiguity the pinning exercise was meant to end.

Part 2 — Hypothesis Documents
Four specs follow, ordered by expected value per unit of work. Each is written to the EVENT_REGIME_PLAN.md house style and each is designed to be able to fail.

Doc	Question	Cost	Kills the project if…
HYP-01	Is selection worth more than friction?	Hours	Selection premium ≈ 0
HYP-02	Is the signal anti-predictive?	Days	Gross ≈ 0 (inversion is arithmetically dead)
HYP-03	Is the opening range a special level at all?	Days	Random levels perform identically
HYP-04	Is the trade wrapper mis-specified?	Hours	No conditioner shows a gradient
Run HYP-01 and HYP-04 first. Both are analysis-only against trade_log.parquet, and both can invalidate the premises of HYP-02 and HYP-03 before you spend a week on them.

docs/HYP_01_FRICTION_PREMIUM.md
Hypothesis 01 — The Selection Premium vs the Friction Floor
Claim: variant selection carries genuine, persistent out-of-sample discriminative power, but the friction floor sits above it. The project's failure is an execution-cost failure, not a signal failure.

Falsifiable form: survivor_holdout_mean − unconditional_holdout_mean ≤ 0.

§1 Why now
R3 above. The survivor set landed at −0.0451 R against a population mean plausibly near −0.0885 R. If that gap is real, selection is worth ≈ +0.04 R persistently — which is not nothing, it is roughly the size of the entire FOMC effect you spent a week measuring.

The gap is currently invisible because the report benchmarks against zero.

§2 Pre-registration
Write prereg/hyp01_prereg.json before running:

json
{
  "predicted_sign": "selection_premium > 0",
  "predicted_magnitude_r": 0.03,
  "primary_statistic": "survivor_holdout_net_r - unconditional_holdout_net_r",
  "null": "family-level bootstrap over holdout families, 20000 draws",
  "secondary": "same in GROSS R, to separate signal from friction",
  "kill_criterion": "premium CI includes zero -> selection is noise, close HYP-01/02/03"
}

§3 Tests
3.1 The premium (one groupby).

Quantity	Definition
baseline_net	Mean net R over all rankable families in the holdout
survivor_net	Mean net R over the 94 selected families in the holdout
premium_net	Difference; bootstrap CI by resampling families, not trades
premium_gross	Same in gross R
If premium_gross > premium_net, friction is eating selection value — quantify by how much.

3.2 Gross/net decomposition per instrument-session. This is the operative table and it does not exist anywhere in the repo:

| instrument | session | E[gross_r] | mean cost_r | E[net_r] | cost as % of |gross| | median r_ticks |
|---|---|---|---|---|---|---|

Sort by cost_r. Your own config records the spread: ES/ZN/CL/6E/6J at ~1 tick, ETH TOK at 53.14. A strategy whose median stop is 20 ticks pays 5% friction on ES and 266% on ETH TOK. These are not the same strategy and pooling them produces a number describing neither.

3.3 The viability frontier. For each instrument-session, compute the minimum gross expectancy required to clear friction:

required_gross_r = cost_r = (slippage_ticks + comm_ticks) / r_ticks

Then ask: for how many instrument-sessions does the observed selection premium exceed the friction floor? If the answer is "ES/NY and ZN/NY only", the project's universe should shrink to two instrument-sessions and every conclusion drawn from the ten-instrument pool is a friction artefact.

3.4 Friction sensitivity. Rerun 3.3 at 0.5×, 1×, 2× measured slippage. config.py already flags ETH/BTC as sensitive; this bounds it.

§4 Power
Trivial for 3.2–3.4 (deterministic). For 3.1: 94 families, family-level sd of net R plausibly ≈ 0.15 R ⟹ SE ≈ 0.015 R. A 0.04 R premium is detectable at ~2.6σ. Adequate, but only just — report the CI, not a p-value.

§5 Limitations
The holdout-period unconditional mean is computed on the same holdout used for the survivor test. The comparison is internally valid (both from the same period) but the survivor set was chosen on pre-holdout data, so only the difference is out-of-sample, not the level.
Measured spread is a floor on execution cost; it excludes market impact. A positive premium at measured cost may still be negative at realised cost.
94 families is thin, and they are correlated (shared instruments, shared days). Bootstrap over families, and expect the CI to be wide.
§6 What a positive result licenses
Not "trade this." It licenses narrowing the universe to instrument-sessions where the premium clears friction, and re-running the entire sweep on that reduced universe with the multiplicity budget recomputed for the smaller hypothesis space. That is a materially different — and much more powerful — study.

docs/HYP_02_SIGNAL_INVERSION.md
Hypothesis 02 — Range-Boundary Breakouts Are Anti-Predictive
Claim: price crossing an opening-range boundary predicts reversion, not continuation. The strategy has the right level and the wrong sign.

Falsifiable form: a fade construction at the same boundaries, with matched risk geometry, has gross expectancy ≤ the breakout construction's.

§1 Why now — and the gate that must pass first
Evidence for: null_p_matched median 0.978 (R4), sub-chance holdout persistence (R2), and your own case study — 6E/NY 2026-08-07 broke above at 09:40 and below at 09:56, sixteen minutes later. Two direction flips in a 150-bar session.

GATE — do not proceed past this
Inversion only works if GROSS expectancy is meaningfully negative. Costs do not invert.

If E[gross_r] = −0.03 and cost_r = 0.06, then net is −0.09, and the inverted trade is +0.03 − 0.06 = −0.03. Still negative. You have burned a week to move from one losing strategy to a slightly less losing one.

Compute E[gross_r] per instrument-session (HYP-01 §3.2) first. Proceed only where |E[gross_r]| > 2 × cost_r. If gross is within noise of zero everywhere, HYP-02 is arithmetically dead and should be closed unrun.

§2 What "inversion" actually means
Not a sign flip on the existing rows. The engine generates longs only at rh + tick and shorts only at rl − tick; a fade is a short at the high boundary and a long at the low boundary, which the sweep has never produced. It needs its own construction:

Element	Breakout (existing)	Fade (new)
Trigger	h ≥ rh + tick	same trigger, opposite side
Direction	long	short
Entry	max(rh + tick, o[i])	rh + FADE_ENTRY_TICKS (limit, further out)
Stop	swing cluster below	above the breakout extreme, or k × ATR
Target	rr × R above	back toward rl, or rr × R below
Exit	11:59 bar	same
Three design choices need pre-registering, because each is a researcher degree of freedom:

Entry offset. A fade filled at rh + 1 tick is filled on every breakout including the ones that run. Filling further out (rh + 0.25 × range_width) selects for extension, which is the classic fade setup. Pre-register one value; test alternatives only as a declared sensitivity.
Stop. The swing-cluster detector is built for continuation and has no natural mirror for a fade. Use stop = breakout_extreme + SWING_MIN_SL_TICKS — simple, defensible, and independent of the existing detector's quirks.
Target. rr × R keeps it comparable to the existing grid. A "return to rl" target is more natural to the thesis but is not comparable. Run both; pre-register rr × R as primary.
§3 Test design
Reuse the whole pipeline. detect_entries gains a FADE-II / FADE-CC mode pair; everything downstream — trade_sim, filters, stats, null_calibrator, multiplicity — is unchanged.

Critical: the fade variants enter the same multiplicity budget. Adding 2 modes × 3 ranges × 2 directions × 6 rr ≈ 72 variants per instrument-session is a ~10% expansion of the hypothesis space. Recompute the maxT hurdle over the combined family set; do not test the fades against the old hurdle.

Primary comparison: paired, per family. For each (instrument, session, range_minutes, direction), compute E[gross_r]_fade − E[gross_r]_breakout on the same days. Permute the fade/breakout label at the day level, joint draws across families (the §6.2 discipline you already validated). Paired-on-day removes the day-effect entirely, which is where most of the variance lives.

§4 The trap that kills most inversion studies
If the breakout strategy is losing because of friction, the fade will lose for the same reason. The paired test above controls for this only if the fade's r_ticks distribution matches the breakout's — otherwise cost_r differs and you are comparing cost structures, not signals.

Report median(r_ticks) for both arms. If they differ by more than ~30%, add a matched-r_ticks variant where the fade's stop is forced to the breakout's stop distance, and treat that as primary.

§5 Power
Fades trigger on every breakout, so n matches the breakout population — roughly 2.8M eligible trades pre-holdout. Paired on day with ~1,400 union dates, the day-level SE is ≈ sd_day / √1400. At sd_day ≈ 0.4 R, SE ≈ 0.011 R. A 0.03 R difference is detectable at ~2.7σ. This is the best-powered hypothesis in the set.

§6 Limitations
Fade entries are limit orders and never fill on the runs that gap through. The backtest will fill them. That is a systematic optimism that this codebase cannot correct without quote data, and it biases toward the hypothesis. State it prominently.
Same-bar ambiguity resolves SL-first in both arms, which penalises the tighter-stopped arm more. Report same_bar_ambiguous rates for both and, if they diverge, report the optimistic bracket (gross_r_optimistic already exists).
Anti-predictive on 2019–2026 futures does not mean anti-predictive going forward. If breakout-fading were free money it would be arbitraged; the more likely truth is that it is a friction-scale effect, which loops straight back to HYP-01.
docs/HYP_03_LEVEL_SPECIALNESS.md
Hypothesis 03 — The Opening Range Is Not a Special Level
Claim: the opening-range boundary carries no more information than an arbitrary price level of comparable construction. The strategy is a re-description of intraday volatility, not a level-based edge.

Falsifiable form: placebo levels produce statistically indistinguishable trigger rates, path shapes, and expectancies.

§1 Why this is worth running even though it cannot make money
It is the only test in the set that can close the project. Every other hypothesis assumes the level means something and asks how to trade it better. If it does not, HYP-01/02/04 are all refinements of a premise that failed, and the honest move is to stop.

It is also the cheapest way to bound the entire research programme, because a negative result is definitive in a way a positive expectancy result never is.

§2 Four placebo levels
All constructed to be mechanically identical to the real range — same width, same construction, same session — differing only in what they are anchored to.

Placebo	Construction	Destroys
SHIFT	Range built from [open + 60, open + 60 + rm]	The privileged status of the open
PRIOR	Yesterday's range at the same clock offset, re-anchored to today's range-end close	Same-day information entirely
WIDTH	A band of identical width centred on the range-end close	The range's location; keeps its width
ROTATE	Real range, but the post-range path circularly rotated (IMPROVEMENTS §6.8)	The phase relationship only; keeps the full realised path
ROTATE is the strictest and the most informative. Under rotation the post-range path is a real, tradable path with real momentum and real volatility clustering; the only thing destroyed is the alignment between the range and what follows. If the real range beats ROTATE, the level is special. If not, it is not.

§3 Statistics to compare — not just expectancy
Expectancy is the least sensitive of these, because it is dominated by friction. The trigger-frequency and path-shape comparisons are where the signal will show if it exists.

Statistic	Prediction if the level is special
Breakout rate (fraction of days with a trigger)	Real > placebo
Tap-in rate given breakout	Real ≠ placebo
Path efficiency after trigger	Real > placebo
MFE/MAE distribution	Real shifted right
E[gross_r]	Real > placebo
Trigger frequency alone is a publishable result. If the real range is broken no more often than a random band of the same width, the premise fails before expectancy is consulted. Conversely, a significant excess of trigger frequency with zero directional edge is a clean finding: the level attracts price without predicting direction — which is itself an argument for the fade in HYP-02.

§4 Test
For each placebo, run the full pipeline and compare paired on (instrument, session, date). Permute the real/placebo label at the day level. Joint draws across all four placebos so the maxT correction is valid across the comparison set.

Reuse src/events/'s permutation machinery — gate_validation.json shows it passing uniformity (KS 0.054 vs critical 0.096), injected-effect recovery (−0.300 exact), and joint-draw integrity. That validation transfers; do not rebuild it.

§5 Cost
SHIFT, PRIOR and WIDTH are cheap: they are build_session_days with different range bounds, then the existing pipeline. Roughly one sweep each.

ROTATE needs the OHLC-preserving tuple decomposition from IMPROVEMENTS §6.8 — (g, m, u, d) = (o−c_prev, c−o, h−o, o−l), permute, re-integrate from the real range-end close, snap to the tick grid. Budget a day, and write the identity test first (forced identity permutation must reproduce the observed trade log exactly).

§6 Limitations
ROTATE introduces one wrap-seam discontinuity per day (~0.5–0.8% of bars at n ≈ 150). Record the seam index; report the fraction of placebo triggers firing on or adjacent to it; exclude if it exceeds a few percent.
PRIOR inherits yesterday's volatility, which correlates with today's. It is the weakest placebo and should be read as a lower bound on the effect.
A positive result — real beats all four placebos — establishes that the level carries information. It does not establish that the information is worth more than friction. That is still HYP-01's question.
docs/HYP_04_TRADE_CONSTRUCTION.md
Hypothesis 04 — The Signal Is Fine; the Trade Wrapper Is Mis-Specified
Claim: conditional expectancy varies systematically with three variables all known before the trade is taken. The pooled negative expectancy averages over a real gradient.

Three sub-hypotheses, all testable in one afternoon against trade_log.parquet.

§1 H4a — Range quality
Claim: expectancy is monotone decreasing in range_width_ticks / atr_4h.

Mechanism: a wide range gives a wide breakout threshold and (via the swing detector) a wide stop, while consuming the session's available travel. Your own case study is the archetype: 6E/NY 2026-08-07 had range_width at the 81.6th percentile inside a session whose total range was at the 29.4th. Nineteen of 49 ticks — 39% of the day's entire range — occurred in the first five minutes. There was nothing left to travel.

Test:

python
df["range_atr"] = df.range_width_ticks * df.tick_size / df.atr_4h
df.groupby([pd.qcut(df.range_atr, 10), "instrument", "session"])["gross_r"].agg(["mean","count"])

Run

Prediction: monotone decreasing. Kill criterion: flat, or non-monotone, across deciles.

Why this is the highest-value cheap test: range_width / atr_4h is known at range-end. It is a live filter, unlike anything the event work produced. If the gradient is steeper than the −0.056 R FOMC effect — and the case-study percentiles suggest it might be — then the calendar was a lossy proxy for something directly observable, and the priority ordering across this whole document changes.

§2 H4b — Hold horizon
Claim: the 11:59 exit is arbitrary and materially wrong. Edge, if present, decays inside the first N minutes.

Test — requires no re-run. mae_r, mfe_r, bars_held and exit_reason are already logged.

Analysis	Reads
E[gross_r] by bars_held decile	Whether long holds destroy short-hold gains
MFE realisation curve: mfe_r vs bars_held at exit	When favourable excursion peaks
exit_reason == "TIME" subset expectancy	Whether the time exit is a systematic loser
Cumulative E[gross_r] under a synthetic exit at t = 15, 30, 45, 60, 90, 120 min	The optimal horizon, directly
Prediction: MFE peaks well before the 11:59 bar; time exits are the worst-performing exit reason; a 30–60 minute cap improves gross expectancy.

Warning: the optimal horizon read off this curve is in-sample and selected. Pre-register a single alternative horizon before looking, or treat the curve as descriptive only and validate the chosen horizon on the holdout.

§3 H4c — Stop specification
Claim: the swing-cluster stop couples entry timing to stop width in a way that corrupts both R and cost.

Why this is subtler than it looks. null_p_matched uses the observed stop distance with random timing, so the 0.978 median says timing is the problem, not width. That argues H4c is lower priority than H4a/H4b — the stop width is not obviously the failure.

But width is not placement. Test three alternatives against the incumbent, all at matched distance so cost is held constant:

Rule	Stop
Incumbent	Swing cluster before breakout
ATR	entry − k × atr_4h, k pre-registered
RANGE	entry − f × range_width, f pre-registered
FIXED	entry − c ticks, c per instrument
Report median(r_ticks) and median(cost_r) for each. If the alternatives differ materially in r_ticks, you are comparing cost structures, not stop rules, and the comparison is void.

§4 Combined pre-registration
json
{
  "h4a_predicted_sign": "expectancy decreasing in range_width/atr",
  "h4a_kill": "flat or non-monotone across deciles",
  "h4b_predicted_optimal_horizon_min": 45,
  "h4b_kill": "expectancy flat in bars_held",
  "h4c_predicted": "ATR stop >= swing stop at matched r_ticks",
  "h4c_priority": "LOW - null_p_matched implicates timing, not width",
  "multiplicity": "3 sub-hypotheses x ~24 instrument-sessions; maxT over the combined grid"
}

§5 Limitations
All three condition on variables correlated with volatility, which is correlated with everything. Stratify by parkinson_vol_14d tercile — the same discipline the FOMC gate needed, and for the same reason.
Decile analysis on 2.8M trades will produce significant gradients on almost any variable. The relevant question is whether the gradient is monotone, stable across instrument-sessions, and survives on the holdout — not whether it is significant.
H4a's practical value depends entirely on the top-decile exclusion being large enough to matter and small enough to preserve sample. Report the trade count retained at each candidate cutoff.
Recommended order
HYP-01 §3.1 — the selection premium. One line. Determines whether the project's headline is wrong.
HYP-01 §3.2 — gross/net by instrument-session. The table that should already exist.
HYP-04a — range-quality deciles. One groupby, and it is a live filter if it works.
HYP-04b — hold horizon. No re-run required.
HYP-02 gate — is gross meaningfully negative anywhere? Proceed or close.
HYP-03 — level specialness. The expensive one; run it when the cheap ones have narrowed the question.