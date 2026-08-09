Hypothesis 05 — Per-Instrument Edge vs Pooled Edge
Status: specification, pre-registration pending
Depends on: HYP-01 findings (family-level premium falsified, friction frontier computed), the a5bdc2b holdout-pin fix, the 8f54e7e comm_ticks fix
Supersedes: the standalone trade_count ≥ 400 conditioner test (§6 subsumes it)

§0 — The reframe
Five hypotheses have now returned effects at or below the pooled study's resolution floor. The natural reading is that the effects are absent. There is a second reading that fits the same evidence and has not been tested:

The pooled test is imprecise because instruments differ. Its standard error is driven by between-instrument variance. If that variance is real rather than sampling noise, then the thing limiting the pooled test is itself the finding — and it is being treated as a nuisance parameter.

This is not a rescue narrative. It is a specific, falsifiable claim with a clean null: τ² = 0, all instruments share one true expectancy, and every apparent difference is sampling error. If τ² ≈ 0 the pooled results stand unchallenged and this document closes the project's cross-sectional line of enquiry cleanly.

The document also folds in three outstanding items that belong to the same statistical family: the winner's-curse conditioner (§6), the 38.3%/26.6% discrepancy (§7), and the HYP-03 scope decision (§8).

§1 — The unit-of-independence argument, stated precisely
The prior session concluded that the honest unit of independence is the instrument, of which there are 10. That is correct for cross-instrument claims and incorrect for within-instrument claims, and the distinction determines what remains answerable.

Question	Unit of independence	Effective n	Detectable effect
"The ORB edge is non-zero"	Instrument	10 (really ~6, see §9)	≈ 0.036 – 0.052 R
"ES/NY expectancy is non-zero"	Session-day	~1,400	≈ 0.032 R after Bonferroni over 21 instrument-sessions
"Instruments differ from each other"	Instrument	10	See §2 — this is a variance question, not a mean question
The middle row is the one that has never been run as a primary test. It has ~140× more effective observations than the top row, and it is Bonferroni-correctable to a comparable threshold. It is not more powerful in the loose sense — it is answering a narrower question that the data can actually address.

Consequence: "no ORB edge exists" is currently unsupported at the required precision, and will remain so at this universe size. "No ORB edge exists in ES/NY" is testable now. The project's headline should be restated at the resolution the design supports.

§2 — H5a: Is between-instrument heterogeneity real?
Claim: true expectancy varies across instruments by more than sampling error explains.
Null: τ² = 0.
Falsified if: the τ² confidence interval includes zero, or the shrinkage factor exceeds ~0.85 for the median instrument (i.e. optimal estimation ignores the instrument label).

2.1 Model
Two-level random effects on gross R, fitted at the instrument-session level:

y_i = μ + u_i + e_i        u_i ~ N(0, τ²)      e_i ~ N(0, σ_i²)

where y_i is instrument-session i's day-clustered mean gross R and σ_i² its day-clustered sampling variance. Fit by REML. Report:

τ̂² with a profile-likelihood CI (not Wald — Wald intervals on variance components at n=10 are badly behaved and will produce negative lower bounds)
I² = τ̂² / (τ̂² + median σ_i²) — the fraction of observed dispersion attributable to real differences
Per-instrument shrinkage factor B_i = τ̂² / (τ̂² + σ_i²)
2.2 Why gross, not net
Friction differs by ~50× across the universe (§4). Fitting on net conflates "this instrument's signal is better" with "this instrument is cheaper to trade." Fit on gross; bring friction back explicitly in §4. Report the net fit as a secondary so the two can be compared.

2.3 Interpretation
I²	Reading
< 0.25	Heterogeneity is noise. Pooling is correct. Close the cross-sectional line.
0.25 – 0.60	Real but modest. Shrunk per-instrument estimates are the right summary; raw ones overstate spread.
> 0.60	Instruments are substantially different objects. Pooled statements are not meaningful and should be withdrawn from the README.
§3 — H5b: Is the instrument ranking persistent?
Heterogeneity in §2 is a variance statement. It does not establish that any particular instrument is better — that requires the ordering to hold out of sample.

Test: Spearman ρ between in-sample and holdout instrument-session gross expectancy rankings.

Power, stated honestly: with 21 items, SE(ρ) ≈ 1/√20 ≈ 0.22. A one-sided test at 5% needs ρ > 0.37. That is detectable if the ranking is genuinely stable, and undetectable for weak persistence. Report the point estimate and CI regardless; a ρ of 0.25 with CI [−0.20, 0.62] is uninformative and must be labelled so rather than read as "some persistence."

Secondary: top-quartile persistence. Of the top 5 instrument-sessions in-sample, how many are top-10 in holdout? Under H₀ the expected count is 2.4. This is a coarser statistic but more robust to a single instrument's holdout noise.

Confound to control: holdout trade counts vary enormously across instrument-sessions. An instrument-session with 40 holdout trades has a mean with SE ≈ 0.16 R and will rank near-randomly regardless of its true position. Weight the rank correlation by holdout precision, or restrict to instrument-sessions with ≥ 200 holdout trades and report how many that leaves. If it leaves fewer than 12, the test is not worth running and should be deferred until the cache extends.

§4 — H5c: Per-instrument friction viability
HYP-01 found no instrument-session clears its friction floor after Bonferroni. That test used a pooled required-gross figure. The correct construction is instrument-specific throughout, and the spread is extreme:

Instrument	Slippage (ticks)	comm_ticks	Total friction (ticks)	Required gross @ r=20 ticks
ES	1.00	0.248	1.25	0.062 R
ZN	1.00	0.198	1.20	0.060 R
6E	1.00	0.496	1.50	0.075 R
NQ TOK	3.16	0.620	3.78	0.189 R
BTC TOK	4.20	0.124	4.32	0.216 R
ETH TOK	53.14	1.240	54.38	2.719 R
(comm_ticks from contracts.py where present, DEFAULT_RT_COMMISSION_USD / tick_value_usd otherwise — the path 8f54e7e fixed. Verify these against the corrected function before use.)

ES needs 0.062 R of gross edge. ETH TOK needs 2.719 R. These are not the same strategy under different conditions; they are different strategies. A pooled "required 0.152" describes neither, and the pooled E[gross_r] = −0.014 is an average over hurdle rates spanning 44×.

4.1 Test
For each of the 21 instrument-sessions:

viable_i  ⟺  E[gross_r]_i  >  cost_r_i,  with a day-clustered lower confidence bound above cost_r_i

Bonferroni over 21, or Holm for slightly more power. Report the gap E[gross_r]_i − cost_r_i with its CI for every instrument-session, not just the significant ones — the ordered gap table is the useful artefact even when nothing clears.

4.2 The pre-registered expectation
Given pooled gross of −0.014, most instrument-sessions will have negative gross and the question will not arise. The interesting cell is any instrument-session with positive gross and low friction (ES, ZN, 6E, CL at ~1 tick). If none of the four cheap instruments shows positive gross, friction is not the binding constraint and the strategy fails on signal — which contradicts the framing in HYP-01 and should be recorded as such.

4.3 Friction sensitivity
Rerun at 0.5× and 2× measured slippage. config.py flags ETH/BTC as sensitive and four instruments' weights as provisional. Any viability conclusion that flips under 2× is not a conclusion.

§5 — H5d: Is heterogeneity structural or idiosyncratic?
If §2 finds real τ², the follow-up is whether it is predictable from instrument characteristics known a priori. This is the only part of the study that generates a testable forward prediction.

Candidate regressors (all fixed, none fitted from outcomes):

Regressor	Rationale
tick_size / median_price	Relative tick granularity. Coarse ticks quantise the range boundary and the stop; fine ticks let noise cross levels.
cost_r at median r_ticks	Direct friction burden
median r_ticks / atr_4h	Stop width relative to instrument volatility
asset_class	Categorical — captures anything the above miss
session	Cross-cutting: does the effect live in NY across instruments?
Fit: meta-regression of instrument-session gross expectancy on these, weighted by inverse σ_i². Report residual τ² — how much heterogeneity survives explanation.

The forward prediction, and why it matters: if tick_size / price explains a material share of τ², that predicts expectancy for instruments not in the universe. That is the only genuinely out-of-sample test available without waiting for calendar time, and it is how the universe should be expanded (§9).

Power caveat, stated up front: 21 observations, 5 regressors. This is at the edge of estimability and will overfit if run as a search. Pre-register exactly two regressors — tick_size/price and cost_r — as primary; everything else is exploratory and labelled so.

§6 — Shrinkage subsumes the trade_count ≥ 400 conditioner
The prior session found that conditioning on in-sample trade_count ≥ 400 produced a premium of +0.0496 surviving both i.i.d. and clustered intervals, monotone across thresholds, but scanned and therefore exploratory.

The monotonicity is the important part, and it is more diagnostic than it appears. Under a pure-noise null — all families sharing true expectancy μ, in-sample estimates μ + ε — selection picks high ε and out-of-sample everyone reverts to μ. Survivor OOS mean equals population OOS mean at every threshold. A premium that increases with n is therefore the signature of genuine heterogeneity in true family expectancy: as n rises, ε shrinks, and selection increasingly tracks real differences rather than noise.

That is the same mechanism §2 tests. The threshold conditioner is a crude, single-cut approximation to what empirical Bayes does continuously.

6.1 The parameter-free replacement
μ̂_shrunk,i = B_i · ŷ_i + (1 − B_i) · ȳ,     B_i = τ̂² / (τ̂² + σ_i²)

Families with many trades have small σ_i², shrink less, and retain their in-sample estimate — automatically, with no threshold to scan. Families with few trades collapse toward the pooled mean, which is exactly the winner's-curse correction the threshold was approximating.

Test: does μ̂_shrunk predict holdout net R better than raw ŷ? Compare by out-of-sample MSE and by rank correlation with holdout outcomes. Pre-registrable with no free parameter, since τ̂² is estimated from the data rather than chosen.

6.2 Two conditions before the threshold version proceeds separately
If the threshold test is retained as a secondary, both must hold:

(a) Stratum-matched baseline. The premium must be

E[OOS | selected ∧ n ≥ 400]  −  E[OOS | all families with n ≥ 400]

not survivors-at-high-n against the full population. High-n families skew toward 5-minute ranges, permissive entry modes, and dense-coverage instruments — any of which may carry its own expectancy offset. Computed against the unstratified population, an unknown share of +0.0496 is "what makes a family fire often," not "what makes selection work." This is the same estimator-mismatch class as R1 and R3. Check it before anything else in this section.

(b) Establish where the threshold scan ran. If it ran on the holdout, the holdout is spent for this conditioner and a fresh out-of-sample slice is required — extend the cache, or scan on windows 1–7 and test on 8–10. If it ran only in-sample, the holdout is clean. Record which in the pre-registration. This is structurally identical to the paper-trading/holdout overlap the event work had to resolve.

6.3 The arithmetic ceiling
Population −0.0957 plus a fully-replicating +0.0496 lands at −0.046 R, against a required gross of 0.152 pooled (or 0.062 for ES specifically). Pre-register the framing explicitly:

This estimates how much of the winner's curse is removable by a selection-time conditioner. A positive result does not produce a tradable strategy and will not be reported as one.

Written down now, it cannot be quietly promoted later.

§7 — The 38.3% / 26.6% discrepancy
Survivors were 38.3% net-positive out-of-sample; the population baseline is 26.6%. Selection lifted the hit rate by 11.7 points while leaving the mean unmoved (−0.0965 vs −0.0957).

Those two facts are jointly informative: the survivor distribution has more mass above zero at the same centre, so either its positives are smaller or its left tail is heavier.

Before drawing anything from it, two checks:

Same run? 38.3% predates the a5bdc2b holdout-pin validation fix. If that fix changed holdout membership, 38.3% is stale and must be recomputed before being set against 26.6%.
Is it an n effect? P(positive) for a family with 5 holdout trades is close to a coin flip regardless of true expectancy. If survivors have systematically more holdout trades than the population, the hit-rate gap is partly mechanical. Compare P(positive) within matched holdout-trade-count strata.
If it survives both: overlay the survivor and population holdout net-R densities. Selection compressing variance without shifting the mean is a distinct property from expectancy — a genuine characterisation of what selection does. It is not monetizable (P(positive) without magnitude does not pay), so this is a characterisation question, not a strategy one, and should be reported in those terms.

Also fix the banner. src/report.py prints "chance is 50%" where the correct population baseline is 26.6%. That sentence is the most-read line in the project and it currently states a wrong benchmark.

§8 — HYP-03 scope: run SHIFT and WIDTH now
HYP-03 (level specialness) is deferred pending ~14 minutes of compute for SHIFT and WIDTH, with ROTATE blocked on IMPROVEMENTS §6.8.

It should run before this document's tests, not after. Reasoning:

It is the only remaining test that can return a definitive answer rather than another sub-floor effect. Trigger-frequency comparison does not depend on expectancy resolution — if the real range is broken no more often than a width-matched band at the range-end close, the premise fails outright, at a precision the data supports.
HYP-01's kill rule does not close it. HYP-01 asked "is selection worth more than friction"; HYP-03 asks "does the level carry information at all." Independent claims — a level can carry information that friction then consumes.
It costs 14 minutes and no spend.
If SHIFT and WIDTH match the real range, §2–§5 are studying heterogeneity in a signal that does not exist, and the correct move is to stop rather than to characterise noise across instruments.
Deprioritise ROTATE until SHIFT and WIDTH report. If a plain width-matched band performs identically, the specialness question is answered and IMPROVEMENTS §6.8 is a day spent confirming it.

§9 — Power, and what cannot be resolved at this universe size
9.1 Effective cluster count is worse than 10
The 10 instruments are not 10 independent draws. Grouped by asset_class:

Class	Members	Independent?
equity_index	ES, NQ, RTY	Highly correlated — one factor
forex	6E, 6J	Correlated via USD
crypto	BTC, ETH	Highly correlated
metal / energy / fixed_income	GC / CL / ZN	Roughly independent
Effective clusters ≈ 6, not 10. Recomputing the resolution floor with t₀.₉₇₅,₅ = 2.571:

Assumed τ	SE (6 clusters)	Half-width
0.03	0.012	±0.031 R
0.05	0.020	±0.052 R
0.08	0.033	±0.084 R
The floor quoted in the prior session (±0.036) was optimistic. Cluster at asset_class as primary and at instrument as a secondary; report both, and treat the wider one as the honest bound.

9.2 τ² itself is barely estimable at n = 10
A variance component estimated from 10 groups has a profile-likelihood CI typically spanning close to an order of magnitude. §2 may well return "τ² is somewhere between 0 and 0.01 and we cannot tell." That is a legitimate outcome and must be reported as such rather than as evidence for homogeneity — an uninformative interval is not a null result.

This is why §3 (rank persistence) matters independently: ranking is a weaker claim than variance magnitude and is somewhat more tractable at n = 21.

9.3 The only real remedy is more clusters
More trades do not help. More permutations do not help. Better estimators do not help. The floor is set by cluster count.

20 instruments across 10 asset classes would roughly halve the floor.
The natural expansion candidates are cheap-friction, liquid CME contracts absent from the universe: YM (equity index, but adds a fourth to an existing class — low value), 6B / 6A / 6C (forex, ~1 tick), ZF / ZB (fixed income, extends an under-represented class), NG / RB / HO (energy, currently n=1), SI / HG (metals, currently n=1).
Prioritise under-represented classes — energy, metals, fixed income — over adding a fourth equity index. Cluster count, not instrument count, is what binds.
If §5 finds tick_size / price predicts expectancy, that determines which additions are informative rather than merely numerous.
Estimate the Databento cost of extending to 6 additional instruments before committing. This is the highest-leverage spend available to the project, and it is the only intervention that raises the ceiling on every future test.

§10 — Pre-registration
prereg/hyp05_prereg.json, committed before any run:

json
{
  "written_utc": null,
  "git_sha": null,
  "depends_on": ["HYP-03 SHIFT+WIDTH result", "a5bdc2b", "8f54e7e"],

  "gate": {
    "condition": "HYP-03 SHIFT and WIDTH must not match the real range on trigger rate or gross expectancy",
    "action_if_gate_fails": "close HYP-05 unrun; characterising heterogeneity in a non-signal is not informative"
  },

  "h5a_variance_components": {
    "model": "REML two-level random effects on day-clustered gross R, instrument-session level",
    "primary": "tau_squared with profile-likelihood CI, and I_squared",
    "kill": "I_squared < 0.25 or tau_squared CI includes zero -> pooling is correct, close cross-sectional line",
    "note": "an uninformative CI is NOT evidence for homogeneity and will be reported as inconclusive"
  },

  "h5b_rank_persistence": {
    "primary": "Spearman rho, in-sample vs holdout instrument-session gross expectancy",
    "restriction": "instrument-sessions with >= 200 holdout trades",
    "threshold": "rho > 0.37 for one-sided 5% at n=21",
    "abort_if": "fewer than 12 instrument-sessions clear the trade-count restriction"
  },

  "h5c_friction_viability": {
    "primary": "per-instrument-session gap = E[gross_r] - cost_r, day-clustered CI, Holm over 21",
    "report": "full ordered gap table regardless of significance",
    "sensitivity": "0.5x and 2x measured slippage"
  },

  "h5d_structural": {
    "primary_regressors": ["tick_size_over_price", "cost_r_at_median_r_ticks"],
    "exploratory": ["median_r_ticks_over_atr", "asset_class", "session"],
    "note": "21 observations; primary is two regressors only, everything else labelled exploratory"
  },

  "h5e_shrinkage": {
    "primary": "does shrunk in-sample family expectancy predict holdout net R better than raw, by OOS MSE",
    "parameter_free": true,
    "supersedes": "trade_count >= 400 threshold scan",
    "threshold_secondary_conditions": [
      "stratum-matched baseline (E[OOS | selected AND n>=400] - E[OOS | all n>=400])",
      "establish whether the threshold scan touched the holdout; if so, a fresh OOS slice is required"
    ]
  },

  "clustering": "asset_class (~6) as primary, instrument (10) as secondary; report both",
  "framing_constraint": "No result in this document produces a tradable strategy. Ceiling is -0.046 R against a required 0.062 (ES) to 2.719 (ETH TOK)."
}

§11 — Implementation order
#	Step	Cost	Gate
1	HYP-03 SHIFT + WIDTH	14 min	If placebos match, close HYP-05 unrun
2	Banner fix: 26.6% baseline, post-a5bdc2b run	Minutes	—
3	§7 checks: same-run confirmation, n-stratified P(positive)	One groupby	—
4	§6.2(a) stratum-matched baseline for the threshold conditioner	One groupby	If the premium vanishes, drop the threshold arm entirely
5	§6.2(b) establish where the threshold scan ran	Inspection	Determines whether an OOS slice exists
6	§4 per-instrument-session gap table	Hours	The most useful artefact in the document
7	§2 variance components	Hours	Gates §3 and §5
8	§3 rank persistence	Hours	Abort if fewer than 12 qualify
9	§6.1 shrinkage vs raw prediction	Hours	—
10	§5 structural meta-regression	Hours	Only if §2 finds material τ²
11	Cost-estimate universe expansion (§9.3)	Minutes	Informs the next project phase
12	Retire baseline_hashes_pre_extension.json	Minutes	Two baseline files is the ambiguity the pin was meant to end
Steps 1–5 are under an hour combined and three of them can invalidate the rest.

§12 — Limitations
On the reframe. "The pooled test is imprecise because instruments differ" is a hypothesis, not a defence. §2 exists to falsify it, and at n = 10 the falsification may be inconclusive rather than clean — an outcome that must be reported as inconclusive, not as support.

On cluster count. Effective independence is ~6, not 10, and possibly fewer if a common volatility factor drives all of them. Every cross-instrument interval in this document should be read as a lower bound on the true width.

On per-instrument holdout thinness. Three months of holdout for a single instrument-session is roughly 60 session-days. Per-instrument out-of-sample validation is severely underpowered, and §3's trade-count restriction may leave too few instrument-sessions to test. This is a design constraint, not a fixable analysis choice — it argues for extending the cache before extending the universe.

On the ceiling. Nothing here produces a tradable result. The best case is a well-characterised negative with a per-instrument breakdown and a structural predictor for universe expansion. That is a genuine research output and it should be stated as the goal rather than discovered as a disappointment.

On multiplicity across documents. HYP-01 through HYP-05 constitute a search. The maxT machinery corrects within a hypothesis, not across them. By the time §5's exploratory regressors are examined, the project has tested well over a hundred distinct claims. Any finding surviving to this point should be treated as requiring independent out-of-sample confirmation on data not yet collected — which loops back to §9.3 as the binding constraint on the entire programme.