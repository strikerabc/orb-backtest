# HYP-01 findings — the selection premium vs the friction floor

Pre-registration: `prereg/hyp01_prereg.json` (committed at `da018bf`, before any result).
Tool: `tools/analysis/hyp01_friction_premium.py`. Artifacts: `outputs/hyp01_*`.
No new data, no spend.

## Verdict

**FALSIFIED on the pre-registered primary.** Family-level premium
**−0.0008 R, 95% CI [−0.0532, +0.0488]** — includes zero, which is the declared
kill criterion.

The pre-registered *secondary* disagrees, and both were declared in advance, so
neither can be dropped after the fact. The full picture:

| statistic | point | 95% CI | reading |
|---|---|---|---|
| **premium, family level (PRIMARY)** | **−0.0008** | [−0.0532, +0.0488] | null |
| premium, trade-weighted | +0.0413 | [+0.0080, +0.0740] | significant |
| premium, trade-weighted, **instrument-clustered** | +0.0413 | [−0.0015, +0.0870] | null |
| premium, gross, trade-weighted | +0.0370 | [+0.0045, +0.0699] | significant |
| premium, gross, clustered | +0.0370 | [−0.0009, +0.0826] | null |

Whatever the weighting, the premium **does not survive resampling whole
instruments**. Four findings this session have died on that same correction (R1's
marginal significance, R2's p=0.023→0.113, the ≥10 threshold, and this) — it is the
load-bearing limitation of a 94-family, 9-instrument sample, and the last two miss
by 0.0015 R and 0.0009 R.

## R3's premise is refuted at the root

R3 reasons: survivors came in at −0.045 against a population near −0.0885, so
selection delivered ≈ +0.04 R of persistent discrimination.

Measured with **one estimator across both arms**:

| | survivor | population | premium |
|---|---|---|---|
| family-level mean | **−0.0965** | **−0.0957** | −0.0008 |
| trade-weighted mean | −0.0469 | −0.0882 | +0.0413 |

At the family level the two arms are **the same number**. R3's +0.04 was not
survivor-vs-population — it was **trade-weighted-vs-unweighted**, the same defect
as review finding R1 one section earlier. The pre-registration anticipated this and
fixed one estimator per comparison, which is why the tool reports both rows.

R3's *conclusion* is nevertheless partly vindicated under precision weighting: a
+0.0413 R premium does exist there, and trade-weighting is defensible as
inverse-variance weighting because per-family precision scales as `sqrt(trades)`.
The reasoning was wrong; the intuition was not worthless.

## What actually separates the two estimators: the winner's curse, quantified

| | survivors firing 1–9 times | survivors firing ≥10 times |
|---|---|---|
| count | 17 | 77 |
| median in-sample `trade_count` | 196 | 440 |
| mean in-sample expectancy | **+0.0564** | +0.0389 |
| mean holdout give-back | **−0.3939** | −0.0821 |

The low-firing group looked **better in sample** and gave back **4.8× more**.
`corr(log in-sample count, log holdout count) = +0.69`, so holdout firing rate
proxies for how much in-sample noise fed the selection decision. Thirteen of the 17
are `rr=2.0` — the level that fires least often, hence structurally the smallest
samples. This is not noise; it is the mechanism, and it is the one part of HYP-01
that generalises.

Equal weighting lets those 17 dominate. Precision weighting does not. Neither is
wrong — they answer different questions — and the pre-registered primary governs.

## Exploratory: an ex-ante conditioner

In-sample `trade_count` is knowable **at selection time**, unlike holdout firing
rate, so conditioning on it could become a live rule.

| min in-sample trades | n survivors | premium | iid CI | clustered CI |
|---|---|---|---|---|
| 100 | 94 | −0.0008 | [−0.0532, +0.0488] | [−0.0843, +0.0836] |
| 200 | 78 | +0.0090 | [−0.0365, +0.0527] | [−0.0777, +0.0905] |
| 300 | 68 | +0.0246 | [−0.0224, +0.0691] | [−0.0663, +0.0956] |
| **400** | 45 | **+0.0496** | **[+0.0032, +0.0907]** | **[+0.0112, +0.0946]** |
| 600 | 14 | +0.0614 | [−0.0057, +0.1316] | [+0.0174, +0.1565] |

At ≥400 the premium survives **both** intervals — the only thing this session to do
so. But **thresholds were scanned**, so this is a forking path and is recorded as
exploratory. It is monotone in the threshold, which is what the winner's curse
predicts rather than what one lucky cut looks like, but monotone-and-exploratory is
still exploratory. It needs its own pre-registered out-of-sample test.

## §3.2 gross/net decomposition — and an arithmetic correction to the review

The table did not exist in the repo; it is now
`outputs/hyp01_gross_net_by_instrument_session.csv`. Pooled `E[gross_r] = −0.0141`,
pooled `cost_r = 0.0760`.

**The review's cost-ratio claim is wrong.** It states "a strategy whose median stop
is 20 ticks pays 5% friction on ES and 266% on ETH TOK", inferring a ~100×
dispersion. Measured, `cost_r` spans **0.0335 (NQ/NY) to 0.1351 (BTC/TOK) — a
factor of 4.0×**, not 100×.

The error is assuming one 20-tick stop across all instruments. Stops scale with
tick granularity: **ETH's median `r_ticks` is 290, not 20**. ETH's 53.14-tick
slippage divided by a 290-tick stop is 0.18 R, not 2.66 R. The directional point
(cost varies materially by venue, so pooling misleads) survives; the magnitude does
not, and 4× is a much weaker argument for shrinking the universe than 100×.

`cost_pct_of_abs_gross` reaches 16,302% for NQ/NY, which is not a meaningful
quantity — it is a ratio to a denominator near zero (`E_gross_r = −0.0002`). Read
`headroom_r` instead.

## §3.3 mis-specified; §3.6 replaces it

The spec asks whether the **premium** clears the friction floor. That can pass while
survivors cannot pay costs, because the baseline it subtracts is itself
gross-negative. Worked case: **NQ/NY "clears"** on premium +0.0370 > required
0.0263, while survivor gross there is **−0.0327**. You cannot pay a 0.0263 toll out
of negative edge.

Correct test: survivor `E[gross_r] > required_gross_r`, i.e. survivor
`E[net_r] > 0`, per cell, with multiplicity control across the 15 cells.

| | count |
|---|---|
| gross clears floor on the **point estimate** | 5 of 15 |
| net positive, nominal 95% CI | 1 of 15 |
| **net positive, Bonferroni-corrected** | **0 of 15** |
| net negative, Bonferroni-corrected | 1 of 15 (6J/TOK) |

The only nominally positive cell is BTC/NY, CI [+0.0063, +0.6153] on 79 trades — a
0.61 R interval width. Fifteen positivity tests at the 2.5% tail produce ~0.38
winners by chance; one was observed. ETH/NY's apparent +0.1316 rests on **12
trades**.

Survivor trade-weighted **gross** expectancy is **+0.0036 R** against a
cheapest-venue floor of **0.0263 R** — short by 7×. Even granting the premium at
face value, there is no venue where it pays for trading.

## §3.4 friction sensitivity

| slippage | mean `cost_r` | `E[net_r]` |
|---|---|---|
| 0.5× | 0.0465 | −0.0606 |
| 1× | 0.0760 | −0.0901 |
| 2× | 0.1351 | −0.1492 |

Halving all measured slippage does not reach break-even. The conclusion is not
sensitive to the cost calibration, which also means the `provisional` slippage
figures for GC/ZN/6E/6J are not load-bearing here.

## Review finding R2: the baseline was wrong

R2 argues 36 of 94 net-positive is significantly *worse* than chance (z = −2.27,
p = 0.023 against a 0.5 null).

Measured population rate: **26.6%**. Survivors: **37.2%**. Difference **+0.1064, CI
[+0.0155, +0.2002]** — survivors are net-positive **significantly more often** than
the population they were drawn from, not less.

"Chance = 50%" is the same wrong-baseline error R3 identifies one line above: with a
gross-negative population, a random family is net-positive well under half the time.
R2's own clustering correction also matters — over 9 instruments the rate sd inflates
1.43× and p moves 0.023 → 0.113, with 6J alone contributing 1 of 18 net-positive.

`src/report.py`'s banner still prints "chance is 50%". That is now known to be
wrong and is queued as a base-code fix.

## HYP-02 gate: the answer is no

HYP-02 §1 says proceed only where `|E[gross_r]| > 2 × cost_r`. Measured per
instrument-session, the most negative gross values are ZN/LDN (−0.0466 vs required
0.2514), ES/LDN (−0.0465 vs 0.1530) and 6E/LDN (−0.0413 vs 0.1858) — **failing by
3–5×**. Pooled, |−0.0141| against 2 × 0.0760 = 0.1520 fails by **10.8×**.

By the spec's own instruction, **HYP-02 is arithmetically dead and closes unrun.**
Inverting a −0.014 R gross edge yields +0.014 R against 0.076 R of cost.

## What this licenses

Not "narrow the universe and re-run" — §3.6 shows no venue qualifies. The one
result worth pursuing is the **winner's curse conditioner**, because it is ex-ante
observable and has a mechanism. That is a HYP-04-shaped question (is the *wrapper*
mis-specified?) rather than a friction question, and it needs its own
pre-registration.
