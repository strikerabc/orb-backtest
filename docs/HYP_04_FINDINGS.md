# HYP-04 findings — is the trade wrapper mis-specified?

Pre-registration: `prereg/hyp04_prereg.json` (committed at `ca1c7fc`).
Tool: `tools/analysis/hyp04_trade_construction.py`. Artifacts: `outputs/hyp04_*`.
No new data, no spend. Two confirmatory tests, Bonferroni alpha 0.025 each.

## Verdict

| sub-hypothesis | result | notes |
|---|---|---|
| **H4a** range quality (rho < 0) | **FALSIFIED** | rho = +0.0021, CI [−0.0021, +0.0080]; wrong sign, includes zero |
| **H4b** 45-min horizon (delta > 0) | **SURVIVES in-sample; fails holdout** | +0.0032, CI [+0.0008, +0.0054] in-sample; +0.0037 CI [−0.0010, +0.0085] holdout |

H4a fails both kill criteria (CI includes zero, not strictly monotone). H4b survives the confirmatory test but fails the holdout validation — the in-sample result does not replicate. H4c was deferred per the pre-registration.

## H4a — range quality

**rho = +0.0021**, CI [−0.0021, +0.0080]. Zero is inside the interval and the sign is **positive** — the opposite of the prediction. The prediction was that wider ranges relative to ATR would produce worse gross expectancy because they consume available session travel. The data says the relationship is indistinguishable from flat.

The decile table confirms it: the pattern is noisy, not monotone. 5 of 9 steps move in the predicted direction, 4 move against it. The vol-tercile stratification shows inconsistent signs (low: +0.0074 *significantly positive*; mid: null; high: null) — the requirement that the gradient hold in all three terciles fails decisively.

The per-cell stability table has 12 cells with negative rho and 11 with positive — the effect has no clear directional structure across the universe.

The case-study percentiles from the review (6E/NY 2026-08-07: range at the 81.6th percentile, session total at the 29.4th) were compelling as anecdote and not representative as a sample. The population shows no gradient.

**Why the mechanism argument fails**: a wider range does widen the swing stop, which widens the TP by the same multiple (both scale with R). So "consuming available travel" and "setting a larger target" happen simultaneously. In a random walk with a proportional TP and SL, a wider range does not change the gross win probability — it changes the bet size, not the bet's expected value. The case needs the additional claim that intraday momentum decays faster than a proportionally wider TP can be reached, and that is not a general property of the instruments in this universe.

## H4b — hold horizon

**In-sample: +0.0032 R, CI [+0.0008, +0.0054]**. Positive, excludes zero at Bonferroni 0.025, both bounds on the same side. The pre-registered 45-minute horizon does capture a real improvement over the 11:59 exit in the in-sample period.

**Holdout: +0.0037 R, CI [−0.0010, +0.0085]**. Same sign, point estimate slightly *larger*, but the CI includes zero. The holdout test is underpowered — 577k trades across 24 cells at 20k bootstrap draws over 10 clusters — so the failure to replicate is as consistent with "real but small and noisy" as with "selected in-sample." It is not confirmation.

The mechanism is real and symmetric:

| effect | magnitude |
|---|---|
| TIME exits under the cap | 9.8% → 19.6% |
| TPs lost to the cap | 4.51% of trades |
| SLs avoided by the cap | 5.26% of trades |
| Mean bars held | 25.4 → 17.6 |

More SLs are avoided than TPs are lost (+0.75% net). That is the positive direction. The improvement is small because only ~10% of trades sit in the window between 45 and the current 11:59 exit where either target fires. The full grid shows monotone decay from +0.0091 at 15 minutes to +0.0009 at 120 minutes — consistent with the predicted mechanism but also consistent with pure noise at scale.

**The practical test, regardless of significance**: a 45-minute cap improves gross by +0.0032 R. Cost invariance is confirmed (sum_cost_diff = 0.000 throughout), so the net improvement is the same. Against a current net expectancy of −0.090 R, this moves the number from −0.090 to **−0.087**. The strategy still loses per trade. The 45-minute result is not a path to profitability — it is a direction, and a small one.

**Why the holdout underpowered matters**: the in-sample CI is [+0.0008, +0.0054] — a width of 0.0046 R. The holdout CI is [−0.0010, +0.0085] — a width of 0.0095, more than twice as wide. The signal is too small to distinguish from zero over a 6-month holdout at the available trade count per instrument.

## The friction question

The H4b result interacts with HYP-01's winner's-curse finding. If the in-sample selection premium is concentrated in high-firing families (≥400 in-sample trades, exploratory), and a 45-minute cap reduces the hold from 25 bars to 17 bars, the natural question is whether the cap and the conditioner interact: do high-firing families show a *larger* delta at 45 minutes? That is exploratory, not pre-registered, and is recorded for a future pre-registered test.

## H4c — stop specification (deferred)

The pre-registration marked this LOW priority on the grounds that `null_p_matched` uses the observed stop distance with random timing, so the 0.978 median implicates timing rather than width. H4b's result is consistent with this interpretation: capping the hold time (addressing timing) produces a small positive improvement; there is no evidence that the stop width is the failure mode.

## What this changes for the project

H4a's failure removes the live filter the spec called "the highest-value cheap test." Range quality relative to ATR is not a useful pre-trade signal at population scale — not because the mechanism is wrong but because it averages out across an instrument universe where the opposite instruments are as common as the predicted ones.

H4b's marginal in-sample survival with holdout failure is the fifth finding this session to survive i.i.d. inference and either fail or weaken under clustering or out-of-sample testing. The pattern is consistent: the signals in this universe are real at very small scale and are swamped by between-instrument heterogeneity. The clustering correction is doing real work.

The most actionable finding from these four hypotheses is the HYP-01 winner's-curse conditioner: in-sample trade count ≥ 400 gives a premium that survives both i.i.d. and instrument-clustered intervals. It needs its own pre-registered out-of-sample test.
