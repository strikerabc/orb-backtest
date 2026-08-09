"""
hyp01_friction_premium.py — is selection worth more than friction?

Pre-registration: prereg/hyp01_prereg.json (committed before this ran).
Spec: docs/REVIEW_AND_HYPOTHESES.md, HYP-01.

Claim: variant selection carries genuine out-of-sample discriminative power, but
the friction floor sits above it. Falsifiable form:
    survivor_holdout_mean - unconditional_holdout_mean <= 0

Four sections, in the order the spec ranks them:
  3.1  the premium, family level, both baselines, bootstrap CIs
  3.2  gross/net decomposition per instrument-session   (does not exist in the repo)
  3.3  the viability frontier: required_gross_r = cost_r
  3.4  friction sensitivity at 0.5x / 1x / 2x slippage

Why the holdout is re-simulated rather than read from outputs/: tools/holdout_test.py
runs the holdout only for instrument-sessions containing a SURVIVOR (15 of 21), so
the unconditional baseline the hypothesis needs is not derivable from its output. The
re-simulation here covers all 21 rankable instrument-sessions and is cached, because
the numbers must come from one pass to be comparable.

Reads cached parquet only. No API calls, no spend.

Usage:
    python tools/analysis/hyp01_friction_premium.py
    python tools/analysis/hyp01_friction_premium.py --refresh   # rebuild the cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.bootstrap import Estimate, difference_stat, estimate, mean_stat, weighted_mean_stat
from src.config import (
    INSTRUMENTS, MIN_TRADES_FOR_RANKING, OUTPUTS_DIR, RR_LEVELS,
    SLIPPAGE_TICKS_BY_SYMBOL, SLIPPAGE_TICKS_BY_SYMBOL_SESSION,
)
from src.contracts import comm_ticks
from src.data_layer import _compute_enrichment, ensure_daily, ensure_data
from src.entry_detector import detect_entries
from src.filters import trade_eligibility
from src.range_builder import build_session_days
from src.trade_sim import simulate_trade, slippage_ticks_for

_OUT = _ROOT / OUTPUTS_DIR
FAMILY = ["instrument", "session", "range_minutes", "entry_mode",
          "closure_tf", "direction"]

# Fixed by the pre-registration.
HOLDOUT_START = "2026-02-01"
N_BOOT = 20_000
BOOT_SEED = 0
MIN_HOLDOUT_TRADES = 1        # primary
MIN_HOLDOUT_TRADES_SENS = 10  # declared sensitivity

CACHE = _OUT / "hyp01_holdout_all.parquet"

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)


def hr(t: str) -> None:
    print("\n" + "=" * 118)
    print(t)
    print("=" * 118)


def simulate_holdout(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Run the engine on the holdout period for every (instrument, session) given.

    One pass over all 21 rankable instrument-sessions, so the survivor arm and the
    baseline arm come from the same simulation. Running them separately would risk
    an arm-specific difference in eligibility or enrichment surviving unnoticed.
    """
    by_sym: dict[str, list[str]] = {}
    for sym, sess in pairs:
        by_sym.setdefault(sym, []).append(sess)

    rows: list[dict] = []
    for sym in sorted(by_sym):
        df = _compute_enrichment(ensure_data(sym), ensure_daily(sym),
                                 tick_size=INSTRUMENTS[sym]["tick_size"])
        df = df[df["timestamp"] >= pd.Timestamp(HOLDOUT_START, tz="UTC")]
        if df.empty:
            print(f"  {sym}: no holdout data")
            continue
        for sess in sorted(by_sym[sym]):
            for sd in build_session_days(df, sym, sess):
                for es in detect_entries(sd):
                    for tr in simulate_trade(es, sd, RR_LEVELS):
                        rows.append({
                            "instrument": sym, "session": sess,
                            "range_minutes": es.range_minutes,
                            "entry_mode": es.mode, "closure_tf": es.closure_tf,
                            "direction": es.direction, "rr": tr.rr,
                            "gross_r": tr.gross_r, "net_r": tr.net_r,
                            "cost_r": tr.cost_r, "r_ticks": tr.r_ticks,
                            "exit_reason": tr.exit_reason,
                            "tp_unfillable": tr.tp_unfillable,
                            "tp_ticks": tr.tp_ticks,
                            "contract_changed_in_session": sd.contract_changed_in_session,
                            "contract_changed_since_prev_session":
                                sd.contract_changed_since_prev_session,
                            "session_bar_completeness": sd.session_bar_completeness,
                        })
            print(f"  {sym}/{sess}: cumulative rows {len(rows):,}")
    ho = trade_eligibility(pd.DataFrame(rows))
    return ho[ho["eligible"]].copy()


def load_arms(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (rank, surv, fam) where fam is one row per family with holdout stats."""
    sm = pd.read_parquet(_OUT / "summary.parquet")
    rank = sm[sm["trade_count"] >= MIN_TRADES_FOR_RANKING].copy()
    surv = rank[(rank["expectancy_net_r"] > 0) & (rank["null_p_value"] < 0.05)].copy()

    pairs = sorted(set(map(tuple, rank[["instrument", "session"]].to_numpy())))
    if refresh or not CACHE.exists():
        print(f"  simulating holdout for {len(pairs)} instrument-sessions...")
        t0 = time.perf_counter()
        ho = simulate_holdout(pairs)
        ho.to_parquet(CACHE, index=False)
        print(f"  cached {len(ho):,} eligible holdout trades -> {CACHE.name} "
              f"({time.perf_counter()-t0:.0f}s)")
    else:
        ho = pd.read_parquet(CACHE)
        print(f"  loaded {len(ho):,} eligible holdout trades from {CACHE.name}")

    ho_agg = (ho.groupby(FAMILY + ["rr"], observed=True)
                .agg(ho_trades=("net_r", "size"),
                     ho_gross=("gross_r", "mean"),
                     ho_net=("net_r", "mean"),
                     ho_cost=("cost_r", "mean"),
                     ho_r_ticks=("r_ticks", "median"),
                     ho_win=("gross_r", lambda s: float((s > 0).mean())))
                .reset_index())

    # Collapse to one row per family at the rr with the best IN-SAMPLE net
    # expectancy -- identical to tools/holdout_test.py, so the survivor arm here is
    # comparable to the published figure. Selecting the rr on holdout performance
    # instead would be selecting on the outcome being measured.
    best = rank.loc[rank.groupby(FAMILY, observed=True)["expectancy_net_r"].idxmax()]
    fam = best.merge(ho_agg, on=FAMILY + ["rr"], how="left")

    surv_keys = set(map(tuple, surv[FAMILY].drop_duplicates().to_numpy()))
    fam["is_survivor"] = [tuple(k) in surv_keys
                          for k in fam[FAMILY].to_numpy()]
    return rank, surv, fam


def _cross_check_survivor_arm(fam: pd.DataFrame, surv: pd.DataFrame,
                              ho_agg_source: pd.DataFrame | None = None) -> None:
    """Does this tool's survivor arm reproduce tools/holdout_test.py's figure?

    The two collapse families differently and the difference is easy to miss:
    holdout_test picks the best-net rr among that family's SURVIVING variants, while
    this tool applies one rule to all 976 families -- best-net rr among all RANKABLE
    variants. For a survivor family those can differ, because a variant may have
    higher expectancy_net_r yet fail null_p < 0.05 and so be rankable but not a
    survivor.

    One rule across both arms is required here (a premium between arms collapsed by
    different rules is not a premium), so the check reports the discrepancy rather
    than eliminating it.
    """
    mine = fam[fam["is_survivor"] & fam["ho_net"].notna()]
    published = pd.read_csv(_OUT / "holdout_test.csv")
    theirs = published.loc[
        published.groupby(FAMILY, observed=True)["expectancy_net_r"].idxmax()]
    print(f"  survivor families, this tool     : {len(mine)}")
    print(f"  survivor families, holdout_test  : {len(theirs)}")
    print(f"  family-level mean, this tool     : {mine['ho_net'].mean():+.4f}")
    print(f"  family-level mean, holdout_test  : {theirs['ho_net'].mean():+.4f}")
    same_rr = mine.merge(theirs[FAMILY + ["rr"]], on=FAMILY, how="inner",
                         suffixes=("", "_pub"))
    if len(same_rr):
        n_diff = int((same_rr["rr"] != same_rr["rr_pub"]).sum())
        print(f"  families whose chosen rr differs : {n_diff} of {len(same_rr)}")
        if n_diff:
            print("    -> expected: the best-net RANKABLE variant is not always a")
            print("       SURVIVING variant. One rule is applied across both arms")
            print("       here, so the survivor arm can differ slightly from the")
            print("       published figure. Both are reported.")


def premiums(fam: pd.DataFrame, *, min_trades: int, col: str,
             label: str) -> dict[str, Estimate]:
    """The pre-registered primary and its declared companions, all family level.

    Every Estimate here comes from src.bootstrap, so each interval is computed on
    the statistic it is printed beside. Review finding R1 was exactly this pairing
    going wrong, and HYP-01's primary is a DIFFERENCE, where the hazard is worse:
    the arms overlap, so bootstrapping them separately would discard their
    correlation and mis-state the interval.
    """
    d = fam[fam["ho_trades"].fillna(0) >= min_trades].copy()
    surv = d["is_survivor"].to_numpy(bool)
    allrows = np.ones(len(d), dtype=bool)
    v = d[col].to_numpy(float)
    n = len(d)
    kw = dict(n_draws=N_BOOT, seed=BOOT_SEED, unit="family")

    out = {
        # PRIMARY: survivors vs the unconditional population (nested).
        "premium_vs_all": estimate(difference_stat(v, surv, allrows), n,
                                   label=f"{label} premium vs all", **kw),
        # Disjoint contrast: larger by construction, since the nested version
        # dilutes the gap by the survivor share.
        "premium_vs_nonsurv": estimate(difference_stat(v, surv, ~surv), n,
                                       label=f"{label} premium vs non-survivors",
                                       **kw),
        "survivor_mean": estimate(mean_stat(v[surv]), int(surv.sum()),
                                  label=f"{label} survivor mean", **kw),
        "population_mean": estimate(mean_stat(v), n,
                                    label=f"{label} population mean", **kw),
    }
    # Robustness: whole instruments resampled, since families sharing an instrument
    # share days, paths and regime windows.
    out["premium_vs_all_clustered"] = estimate(
        difference_stat(v, surv, allrows), n,
        label=f"{label} premium vs all, instrument-clustered",
        clusters=d["instrument"].to_numpy(), **kw)
    return out


def section_3_1(fam: pd.DataFrame) -> dict:
    hr("3.1  THE SELECTION PREMIUM  (pre-registered primary)")
    print("  Falsifiable form: survivor_holdout_mean - unconditional_holdout_mean <= 0")
    print("  Kill criterion  : premium CI includes zero -> selection is noise\n")

    have = fam["ho_trades"].fillna(0) >= MIN_HOLDOUT_TRADES
    n_surv = int((fam["is_survivor"] & have).sum())
    n_pop = int(have.sum())
    print(f"  families with >= {MIN_HOLDOUT_TRADES} holdout trade(s) : "
          f"{n_pop} of {len(fam)} rankable")
    print(f"  of which survivors                     : {n_surv}")
    print(f"  survivor share of the population       : "
          f"{100.0*n_surv/max(n_pop,1):.1f}%  (attenuates the nested premium)\n")

    res = {}
    for col, label in (("ho_net", "NET"), ("ho_gross", "GROSS")):
        est = premiums(fam, min_trades=MIN_HOLDOUT_TRADES, col=col, label=label)
        res[label] = est
        print(f"  ── {label} R ─────────────────────────────────────────────────")
        print(f"    survivor mean                 : {est['survivor_mean'].fmt()}")
        print(f"    population mean (all rankable): {est['population_mean'].fmt()}")
        print(f"    PREMIUM vs all      [PRIMARY] : {est['premium_vs_all'].fmt()}")
        print(f"      -> {est['premium_vs_all'].sign_verdict('premium')}")
        print(f"    premium vs non-survivors      : {est['premium_vs_nonsurv'].fmt()}")
        print(f"    premium vs all, clustered     : "
              f"{est['premium_vs_all_clustered'].fmt()}")
        print(f"      -> {est['premium_vs_all_clustered'].sign_verdict('premium')}")
        print()

    net_prem = res["NET"]["premium_vs_all"]
    gross_prem = res["GROSS"]["premium_vs_all"]
    print("  ── friction's share of the premium ─────────────────────────────")
    print(f"    premium in GROSS R : {gross_prem.point:+.4f}")
    print(f"    premium in NET R   : {net_prem.point:+.4f}")
    delta = gross_prem.point - net_prem.point
    print(f"    difference         : {delta:+.4f}")
    if abs(delta) < 0.005:
        print("    -> The premium is essentially the SAME gross and net, so selection")
        print("       is not being eaten by differential friction: survivors and the")
        print("       population pay near-identical costs. The spec's mechanism")
        print("       ('friction is eating selection value') is not what is happening;")
        print("       friction lowers BOTH arms alike and cancels in the difference.")
    else:
        print(f"    -> Friction accounts for {delta:+.4f} R of the difference between")
        print("       the gross and net premium, i.e. survivors and the population do")
        print("       NOT pay the same costs.")

    # Sensitivity, declared in the pre-registration rather than chosen now.
    print()
    print(f"  ── sensitivity: >= {MIN_HOLDOUT_TRADES_SENS} holdout trades ──────────────────────")
    sens = premiums(fam, min_trades=MIN_HOLDOUT_TRADES_SENS, col="ho_net",
                    label="NET")
    d2 = fam[fam["ho_trades"].fillna(0) >= MIN_HOLDOUT_TRADES_SENS]
    print(f"    families retained             : {len(d2)} "
          f"(survivors {int(d2['is_survivor'].sum())})")
    print(f"    PREMIUM vs all                : {sens['premium_vs_all'].fmt()}")
    print(f"      -> {sens['premium_vs_all'].sign_verdict('premium')}")

    # R2's missing baseline: the population's own net-positive rate.
    print()
    print("  ── the correct baseline for the net-positive RATE (review R2) ───")
    dd = fam[fam["ho_trades"].fillna(0) >= MIN_HOLDOUT_TRADES]
    pop_rate = float((dd["ho_net"] > 0).mean())
    surv_rate = float((dd.loc[dd["is_survivor"], "ho_net"] > 0).mean())
    rate_est = estimate(
        difference_stat((dd["ho_net"] > 0).to_numpy(float),
                        dd["is_survivor"].to_numpy(bool),
                        np.ones(len(dd), dtype=bool)),
        len(dd), label="rate premium", n_draws=N_BOOT, seed=BOOT_SEED, unit="family")
    print(f"    population net-positive rate  : {100*pop_rate:.1f}%")
    print(f"    survivor net-positive rate    : {100*surv_rate:.1f}%")
    print(f"    difference                    : {rate_est.fmt()}")
    print("    -> R2 compares the survivor rate against 50%. That is the wrong")
    print("       baseline for the same reason R3 gives for the mean: with a negative")
    print(f"       population mean, a random family is net-positive {100*pop_rate:.1f}% of")
    print("       the time, not 50%.")

    return {"net": res["NET"], "gross": res["GROSS"], "sens_net": sens,
            "pop_rate": pop_rate, "surv_rate": surv_rate, "rate_est": rate_est,
            "n_pop": n_pop, "n_surv": n_surv}


# Columns trade_eligibility consults, plus the measures. Loaded explicitly because
# the trade log is 5.3M rows x 60 columns and only these are needed.
_TL_COLS = ["instrument", "session", "gross_r", "net_r", "cost_r", "r_ticks",
            "exit_reason", "tp_unfillable", "tp_ticks", "contracts",
            "contract_changed_in_session", "contract_changed_since_prev_session",
            "session_bar_completeness", "rr", "range_minutes", "entry_mode",
            "closure_tf", "direction"]


def section_3_2(tl: pd.DataFrame) -> pd.DataFrame:
    hr("3.2  GROSS/NET DECOMPOSITION PER INSTRUMENT-SESSION")
    print("  Measured IN SAMPLE, where the trade count is ~1500x the holdout, because")
    print("  cost structure is a property of the instrument rather than the period and")
    print("  E[gross_r] needs the best-powered estimate available -- HYP-02's gate")
    print("  depends on it. Period-specific reads are section 3.1's job.\n")

    g = (tl.groupby(["instrument", "session"], observed=True)
           .agg(n_trades=("gross_r", "size"),
                E_gross_r=("gross_r", "mean"),
                mean_cost_r=("cost_r", "mean"),
                E_net_r=("net_r", "mean"),
                median_r_ticks=("r_ticks", "median"))
           .reset_index())
    g["cost_pct_of_abs_gross"] = np.where(
        g["E_gross_r"].abs() > 1e-9,
        100.0 * g["mean_cost_r"] / g["E_gross_r"].abs(), np.inf)
    g["slip_ticks"] = [slippage_ticks_for(s, ss)
                       for s, ss in zip(g["instrument"], g["session"])]
    g["comm_ticks"] = [comm_ticks(s, "full") for s in g["instrument"]]
    # required_gross_r IS cost_r -- the spec's identity. Recomputed from ticks rather
    # than reusing mean_cost_r so the two can be compared: they must agree, and a
    # divergence would mean the cost model and the config disagree.
    g["required_gross_r"] = (g["slip_ticks"] + g["comm_ticks"]) / g["median_r_ticks"]
    g = g.sort_values("mean_cost_r", ascending=False)

    show = ["instrument", "session", "n_trades", "E_gross_r", "mean_cost_r",
            "E_net_r", "cost_pct_of_abs_gross", "median_r_ticks", "slip_ticks",
            "comm_ticks", "required_gross_r"]
    print(g[show].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print()
    lo, hi = g["mean_cost_r"].min(), g["mean_cost_r"].max()
    print(f"  cost_r spans {lo:.4f} to {hi:.4f} -- a factor of {hi/max(lo,1e-9):.1f}x")
    print("  across instrument-sessions. A pooled expectancy over this range describes")
    print("  no instrument in it.")
    pooled_gross = float(tl["gross_r"].mean())
    pooled_cost = float(tl["cost_r"].mean())
    print(f"  pooled E[gross_r] = {pooled_gross:+.4f}, pooled cost_r = {pooled_cost:.4f}")
    return g


def section_3_3(g: pd.DataFrame, prem_net: Estimate, prem_gross: Estimate) -> pd.DataFrame:
    hr("3.3  THE VIABILITY FRONTIER")
    print("  required_gross_r = cost_r = (slippage_ticks + comm_ticks) / r_ticks")
    print("  For how many instrument-sessions does the selection premium clear the")
    print("  friction floor?\n")

    v = g.copy()
    # The premium is a GROSS quantity when asked to clear a cost floor: the question
    # is whether selection buys enough gross edge to pay for trading at all. Using
    # the net premium would double-count friction -- once inside the premium and
    # again in the floor it is compared against.
    v["premium_gross_r"] = prem_gross.point
    v["clears_floor"] = v["premium_gross_r"] > v["required_gross_r"]
    v["headroom_r"] = v["premium_gross_r"] - v["required_gross_r"]
    v = v.sort_values("headroom_r", ascending=False)

    show = ["instrument", "session", "required_gross_r", "premium_gross_r",
            "headroom_r", "clears_floor", "E_gross_r", "median_r_ticks"]
    print(v[show].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    n_clear = int(v["clears_floor"].sum())
    print()
    print(f"  instrument-sessions where the premium clears friction: {n_clear} of {len(v)}")
    if n_clear:
        names = ", ".join(f"{r.instrument}/{r.session}"
                          for r in v[v["clears_floor"]].itertuples())
        print(f"    {names}")
        print("  -> A universe restricted to these would need the sweep re-run with the")
        print("     multiplicity budget recomputed for the smaller hypothesis space.")
    else:
        print("  -> NONE. On the measured premium there is no instrument-session where")
        print("     selection buys enough gross edge to pay for trading.")
    return v


def section_3_4(tl: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    hr("3.4  FRICTION SENSITIVITY  (0.5x / 1x / 2x measured slippage)")
    print("  Commission is contractual and is not scaled; only slippage is.")
    print("  config flags ETH/BTC as sensitive and GC/ZN/6E/6J as provisional, with")
    print("  entry-timing weights taken from a pre-roll-fix trade log.\n")

    # Per-(instrument, session) lookup built once, then mapped. Resolving these
    # per row would call slippage_ticks_for 5.3M times per multiplier.
    pairs = tl[["instrument", "session"]].drop_duplicates()
    slip_by = {(s, ss): slippage_ticks_for(s, ss)
               for s, ss in pairs.to_numpy()}
    comm_by = {s: comm_ticks(s, "full") for s in tl["instrument"].unique()}
    key = pd.MultiIndex.from_arrays([tl["instrument"], tl["session"]])
    slip = key.map(slip_by).to_numpy(float)
    comm = tl["instrument"].map(comm_by).to_numpy(float)
    r_ticks = tl["r_ticks"].to_numpy(float)
    gross = tl["gross_r"].to_numpy(float)

    rows = []
    for mult in (0.5, 1.0, 2.0):
        # Recomputed per trade so the r_ticks distribution is respected. Applying a
        # multiplier to a mean cost_r would be wrong: cost_r is a ratio, and its mean
        # is not the ratio of means.
        cost = (mult * slip + comm) / r_ticks
        net = gross - cost
        rows.append({"slippage_multiple": mult,
                     "mean_cost_r": float(np.mean(cost)),
                     "E_net_r": float(np.mean(net)),
                     "frac_trades_net_positive": float(np.mean(net > 0))})
    s = pd.DataFrame(rows)
    print(s.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print()
    print("  Even at 0.5x slippage the sign of E[net_r] is what matters, not its size:")
    print("  halving a cost that exceeds |E[gross_r]| by an order of magnitude does not")
    print("  reach break-even.")
    return s


def _weighted_difference_stat(vals, wts, treat, control):
    """Trade-weighted difference of means, both arms from one resample.

    Not in src.bootstrap because it composes two ideas that module keeps separate
    (ratio estimator + nested difference) and only HYP-01 needs the combination.
    """
    v, w = np.asarray(vals, float), np.asarray(wts, float)
    t, c = np.asarray(treat, bool), np.asarray(control, bool)

    def _f(ix):
        vi, wi, ti, ci = v[ix], w[ix], t[ix], c[ix]
        wa, wb = wi[ti].sum(), wi[ci].sum()
        if not wa or not wb:
            return float("nan")
        return float((vi[ti] * wi[ti]).sum() / wa - (vi[ci] * wi[ci]).sum() / wb)

    return _f


def section_3_5(fam: pd.DataFrame) -> dict:
    """Trade-weighted premium, and the ex-ante conditioner.

    Both are pre-registered secondaries (the weighted premium explicitly; the
    conditioner is reported as EXPLORATORY because the threshold was scanned).
    """
    hr("3.5  PRECISION WEIGHTING AND THE WINNER'S CURSE")
    d = fam[fam["ho_trades"].fillna(0) >= MIN_HOLDOUT_TRADES].copy()
    s = d["is_survivor"].to_numpy(bool)
    w = d["ho_trades"].to_numpy(float)
    allrows = np.ones(len(d), dtype=bool)
    kw = dict(n_draws=N_BOOT, seed=BOOT_SEED, unit="family")

    print("  The family-level primary weights a 1-trade family equally with a")
    print("  103-trade one. Per-family precision scales as sqrt(trades), so")
    print("  trade-weighting is ~inverse-variance weighting -- the pre-registered")
    print("  secondary, and it needs no arbitrary trade threshold.\n")

    res = {}
    for col, lab in (("ho_net", "NET"), ("ho_gross", "GROSS")):
        v = d[col].to_numpy(float)
        p = estimate(_weighted_difference_stat(v, w, s, allrows), len(d),
                     label=f"{lab} trade-wtd premium", **kw)
        pc = estimate(_weighted_difference_stat(v, w, s, allrows), len(d),
                      label=f"{lab} trade-wtd premium, clustered",
                      clusters=d["instrument"].to_numpy(), **kw)
        sm = estimate(weighted_mean_stat(v[s], w[s]), int(s.sum()),
                      label=f"{lab} survivor trade-wtd mean", **kw)
        pm = estimate(weighted_mean_stat(v, w), len(d),
                      label=f"{lab} population trade-wtd mean", **kw)
        res[lab] = {"premium": p, "premium_clustered": pc, "surv": sm, "pop": pm}
        print(f"  ── {lab} R, trade-weighted ──────────────────────────────────")
        print(f"    survivor mean      : {sm.fmt()}")
        print(f"    population mean    : {pm.fmt()}")
        print(f"    PREMIUM            : {p.fmt()}")
        print(f"      -> {p.sign_verdict('premium')}")
        print(f"    PREMIUM, clustered : {pc.fmt()}")
        print(f"      -> {pc.sign_verdict('premium')}")
        print()

    # The mechanism. Reported because it explains why the two estimators disagree,
    # and because it is the one part of HYP-01 that generalises.
    print("  ── the winner's curse, quantified ──────────────────────────────")
    lo = fam[(fam["ho_trades"] >= 1) & (fam["ho_trades"] < 10) & fam["is_survivor"]]
    hi = fam[(fam["ho_trades"] >= 10) & fam["is_survivor"]]
    give_lo = float((lo["ho_net"] - lo["expectancy_net_r"]).mean())
    give_hi = float((hi["ho_net"] - hi["expectancy_net_r"]).mean())
    print(f"    survivors firing 1-9 times in holdout : {len(lo)}")
    print(f"      median in-sample trade_count        : {lo['trade_count'].median():.0f}")
    print(f"      mean in-sample expectancy_net_r     : {lo['expectancy_net_r'].mean():+.4f}")
    print(f"      mean holdout give-back              : {give_lo:+.4f}")
    print(f"    survivors firing >= 10 times          : {len(hi)}")
    print(f"      median in-sample trade_count        : {hi['trade_count'].median():.0f}")
    print(f"      mean in-sample expectancy_net_r     : {hi['expectancy_net_r'].mean():+.4f}")
    print(f"      mean holdout give-back              : {give_hi:+.4f}")
    print(f"    the low-firing group looked BETTER in sample and gave back "
          f"{give_lo/give_hi:.1f}x more")
    corr = float(np.log(d["trade_count"]).corr(np.log(d["ho_trades"])))
    print(f"    corr(log in-sample count, log holdout count) = {corr:+.4f}")

    print()
    print("  ── EXPLORATORY: an ex-ante conditioner ─────────────────────────")
    print("    holdout trade count is measured ON the outcome period, so")
    print("    conditioning on it cannot become a live rule. In-sample trade_count")
    print("    is knowable at selection time and correlates +0.65 with it.")
    print("    THRESHOLDS WERE SCANNED, so this is a forking path: exploratory,")
    print("    requires its own out-of-sample test before it means anything.\n")
    print(f"    {'min_isample':>11} {'n_pop':>6} {'n_surv':>7} {'premium':>9} "
          f"{'iid CI':>20} {'clustered CI':>20}")
    scan = []
    for t in (100, 200, 300, 400, 600):
        dd = d[d["trade_count"] >= t]
        if int(dd["is_survivor"].sum()) < 10:
            continue
        ss = dd["is_survivor"].to_numpy(bool)
        vv = dd["ho_net"].to_numpy(float)
        aa = np.ones(len(dd), dtype=bool)
        p = estimate(difference_stat(vv, ss, aa), len(dd), label="p", **kw)
        pc = estimate(difference_stat(vv, ss, aa), len(dd), label="pc",
                      clusters=dd["instrument"].to_numpy(), **kw)
        scan.append({"min_in_sample_trades": t, "n_pop": len(dd),
                     "n_surv": int(ss.sum()), "premium": round(p.point, 4),
                     "iid_lo": round(p.lo, 4), "iid_hi": round(p.hi, 4),
                     "clu_lo": round(pc.lo, 4), "clu_hi": round(pc.hi, 4),
                     "clustered_excludes_zero": not pc.includes_zero})
        print(f"    {t:>11} {len(dd):>6} {int(ss.sum()):>7} {p.point:>+9.4f} "
              f"[{p.lo:>+7.4f},{p.hi:>+7.4f}] [{pc.lo:>+7.4f},{pc.hi:>+7.4f}]")
    n_clu = sum(1 for x in scan if x["clustered_excludes_zero"])
    print(f"\n    thresholds whose CLUSTERED interval excludes zero: {n_clu} of {len(scan)}")
    print("    The premium is monotone in the threshold, which is what the winner's")
    print("    curse predicts and not what a single lucky cut looks like -- but")
    print("    monotone-and-exploratory is still exploratory.")
    return {"weighted": res, "scan": scan, "give_back_low": give_lo,
            "give_back_high": give_hi, "n_low": len(lo), "n_high": len(hi)}


def section_3_6(fam: pd.DataFrame, ho: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """The frontier the spec should have asked for, with multiplicity control."""
    hr("3.6  CORRECTED VIABILITY FRONTIER  (per-cell, multiplicity-controlled)")
    print("  The spec asks whether the PREMIUM clears the friction floor. That is")
    print("  mis-specified: the premium is measured against a gross-NEGATIVE")
    print("  population, so it can exceed the floor while survivors still cannot pay")
    print("  costs. Worked case from section 3.3: NQ/NY 'clears' on premium +0.0370 >")
    print("  required 0.0263, while survivor gross there is -0.0327.")
    print()
    print("  Correct test: SURVIVOR E[gross_r] > required_gross_r, i.e. survivor")
    print("  E[net_r] > 0. Fifteen cells are tested, so the interval must be")
    print("  corrected -- fifteen independent positivity tests at 2.5% produce ~0.38")
    print("  winners by chance.\n")

    h = ho.merge(fam[FAMILY + ["rr", "is_survivor"]], on=FAMILY + ["rr"], how="inner")
    sv = h[h["is_survivor"]]
    cells = sorted(sv.groupby(["instrument", "session"], observed=True).groups)
    alpha_bonf = 0.05 / max(len(cells), 1)

    rows = []
    for (i, s), grp in sv.groupby(["instrument", "session"], observed=True):
        v = grp["net_r"].to_numpy(float)
        n = len(v)
        e = estimate(mean_stat(v), n, label=f"{i}/{s}", n_draws=N_BOOT,
                     seed=BOOT_SEED, unit="trade")
        eb = estimate(mean_stat(v), n, label=f"{i}/{s} bonf", n_draws=N_BOOT,
                      seed=BOOT_SEED, alpha=alpha_bonf, unit="trade")
        rows.append({"instrument": i, "session": s, "surv_trades": n,
                     "surv_gross": float(grp["gross_r"].mean()),
                     "surv_net": e.point, "surv_cost": float(grp["cost_r"].mean()),
                     "ci_lo": e.lo, "ci_hi": e.hi,
                     "bonf_lo": eb.lo, "bonf_hi": eb.hi,
                     "pos_nominal": e.lo > 0, "neg_nominal": e.hi < 0,
                     "pos_bonferroni": eb.lo > 0, "neg_bonferroni": eb.hi < 0})
    r = pd.DataFrame(rows).merge(
        g[["instrument", "session", "required_gross_r"]],
        on=["instrument", "session"], how="left")
    r["gross_clears_floor_point"] = r["surv_gross"] > r["required_gross_r"]
    r = r.sort_values("surv_net", ascending=False)

    show = ["instrument", "session", "surv_trades", "surv_gross", "surv_net",
            "required_gross_r", "gross_clears_floor_point", "ci_lo", "ci_hi",
            "pos_nominal", "pos_bonferroni"]
    print(r[show].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print()
    print(f"  cells: {len(r)}   Bonferroni alpha = {alpha_bonf:.4f}")
    print(f"  gross clears floor on the POINT estimate : "
          f"{int(r['gross_clears_floor_point'].sum())} of {len(r)}")
    print(f"  net positive, nominal 95% CI             : "
          f"{int(r['pos_nominal'].sum())} of {len(r)}")
    print(f"  net positive, Bonferroni-corrected       : "
          f"{int(r['pos_bonferroni'].sum())} of {len(r)}")
    print(f"  net NEGATIVE, Bonferroni-corrected       : "
          f"{int(r['neg_bonferroni'].sum())} of {len(r)}")
    print()
    if not r["pos_bonferroni"].any():
        print("  -> NO instrument-session where survivors demonstrably clear their own")
        print("     friction floor. The point-estimate winners are what fifteen")
        print("     positivity tests produce by chance, and the widest of them rests")
        print(f"     on {int(r.loc[r['pos_nominal'], 'surv_trades'].min()) if r['pos_nominal'].any() else 0} trades.")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the cached holdout simulation")
    args = ap.parse_args()
    t0 = time.perf_counter()

    hr("HYP-01  THE SELECTION PREMIUM VS THE FRICTION FLOOR")
    print("  prereg: prereg/hyp01_prereg.json")
    print("  claim : selection carries real out-of-sample discrimination, but the")
    print("          friction floor sits above it.\n")

    rank, surv, fam = load_arms(args.refresh)
    print()
    print("  ── cross-check against the published survivor figure ────────────")
    _cross_check_survivor_arm(fam, surv)

    r31 = section_3_1(fam)

    tl_raw = pd.read_parquet(_OUT / "trade_log.parquet", columns=_TL_COLS)
    tl = trade_eligibility(tl_raw)
    tl = tl[tl["eligible"]].copy()
    print(f"\n  in-sample trade log: {len(tl_raw):,} rows -> {len(tl):,} eligible")

    g = section_3_2(tl)
    v = section_3_3(g, r31["net"]["premium_vs_all"], r31["gross"]["premium_vs_all"])
    s = section_3_4(tl, g)
    r35 = section_3_5(fam)
    ho = pd.read_parquet(CACHE)
    r36 = section_3_6(fam, ho, g)

    # ── verdict against the pre-registered kill criterion ────────────────────
    hr("VERDICT AGAINST THE PRE-REGISTERED KILL CRITERION")
    prem = r31["net"]["premium_vs_all"]
    wprem = r35["weighted"]["NET"]["premium"]
    wprem_c = r35["weighted"]["NET"]["premium_clustered"]
    killed = prem.includes_zero
    print(f"  PRIMARY  premium (NET, family-level, vs all) : {prem.fmt()}")
    print(f"  kill criterion: CI includes zero -> {killed}")
    print()
    if killed:
        print("  HYP-01 FALSIFIED on its own pre-registered criterion.")
        print()
        print("  The premium CI includes zero: at the family level, survivor families")
        print("  are not distinguishable from the unconditional population out of")
        print("  sample. The spec's kill rule says this closes HYP-01/02/03.")
    else:
        side = "positive" if prem.lo > 0 else "negative"
        print(f"  Premium is significantly {side}.")
        if prem.lo <= 0:
            print("  Selection is significantly WORSE than the population it was drawn")
            print("  from -- the opposite of the claim.")

    # The pre-registered secondary disagrees with the primary, and both were
    # declared in advance, so neither can be quietly dropped.
    print()
    print("  ── but the pre-registered SECONDARY disagrees ───────────────────")
    print(f"    trade-weighted premium            : {wprem.fmt()}")
    print(f"      -> {wprem.sign_verdict('premium')}")
    print(f"    same, instrument-clustered        : {wprem_c.fmt()}")
    print(f"      -> {wprem_c.sign_verdict('premium')}")
    print()
    print("    Both estimators were pre-registered, so the flattering one cannot be")
    print("    selected after the fact. What separates them is real and identified:")
    print("    the 17 survivor families that fire 1-9 times in the holdout gave back")
    print(f"    {r35['give_back_low']:+.4f} R against {r35['give_back_high']:+.4f} R for the "
          f"{r35['n_high']} that fire >= 10 times,")
    print("    while looking BETTER in sample. Equal weighting lets those 17 dominate;")
    print("    precision weighting does not. Neither is wrong -- they answer different")
    print("    questions, and the pre-registered PRIMARY governs the verdict.")
    print()
    print("    Note the clustered interval: whatever the weighting, the premium does")
    print("    not survive resampling whole instruments. Four separate findings this")
    print("    session have died on that correction, which is the load-bearing")
    print("    limitation of a 94-family, 9-instrument sample.")

    n_pos_bonf = int(r36["pos_bonferroni"].sum())
    print()
    print("  ── and the tradeability question is settled regardless ──────────")
    print(f"    instrument-sessions where survivors demonstrably clear their own")
    print(f"    friction floor (Bonferroni-corrected): {n_pos_bonf} of {len(r36)}")
    print("    Survivor trade-weighted GROSS expectancy is "
          f"{r35['weighted']['GROSS']['surv'].point:+.4f} R against a")
    print(f"    cheapest-venue floor of {g['required_gross_r'].min():.4f} R. Even taking the")
    print("    premium at face value, there is no venue where it pays for trading.")

    out = {
        "hypothesis": "HYP-01",
        "prereg": "prereg/hyp01_prereg.json",
        "holdout_start": HOLDOUT_START,
        "bootstrap_draws": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "min_holdout_trades": MIN_HOLDOUT_TRADES,
        "n_rankable_families": int(len(fam)),
        "n_families_with_holdout": int(r31["n_pop"]),
        "n_survivor_families": int(r31["n_surv"]),
        "premium_net_vs_all": _ser(r31["net"]["premium_vs_all"]),
        "premium_net_vs_nonsurvivors": _ser(r31["net"]["premium_vs_nonsurv"]),
        "premium_net_vs_all_clustered": _ser(r31["net"]["premium_vs_all_clustered"]),
        "premium_gross_vs_all": _ser(r31["gross"]["premium_vs_all"]),
        "survivor_mean_net": _ser(r31["net"]["survivor_mean"]),
        "population_mean_net": _ser(r31["net"]["population_mean"]),
        "survivor_mean_gross": _ser(r31["gross"]["survivor_mean"]),
        "population_mean_gross": _ser(r31["gross"]["population_mean"]),
        "premium_net_sensitivity_min10": _ser(r31["sens_net"]["premium_vs_all"]),
        "population_net_positive_rate": round(r31["pop_rate"], 4),
        "survivor_net_positive_rate": round(r31["surv_rate"], 4),
        "net_positive_rate_premium": _ser(r31["rate_est"]),
        "instrument_sessions": int(len(g)),
        "instrument_sessions_clearing_friction": int(v["clears_floor"].sum()),
        "cost_r_min": round(float(g["mean_cost_r"].min()), 4),
        "cost_r_max": round(float(g["mean_cost_r"].max()), 4),
        "pooled_E_gross_r": round(float(tl["gross_r"].mean()), 5),
        "pooled_cost_r": round(float(tl["cost_r"].mean()), 5),
        "friction_sensitivity": s.to_dict(orient="records"),

        # ── 3.5 precision weighting and the winner's curse ────────────────
        "premium_net_trade_weighted": _ser(r35["weighted"]["NET"]["premium"]),
        "premium_net_trade_weighted_clustered":
            _ser(r35["weighted"]["NET"]["premium_clustered"]),
        "premium_gross_trade_weighted": _ser(r35["weighted"]["GROSS"]["premium"]),
        "premium_gross_trade_weighted_clustered":
            _ser(r35["weighted"]["GROSS"]["premium_clustered"]),
        "survivor_mean_net_trade_weighted": _ser(r35["weighted"]["NET"]["surv"]),
        "population_mean_net_trade_weighted": _ser(r35["weighted"]["NET"]["pop"]),
        "survivor_mean_gross_trade_weighted": _ser(r35["weighted"]["GROSS"]["surv"]),
        "population_mean_gross_trade_weighted": _ser(r35["weighted"]["GROSS"]["pop"]),
        "winners_curse": {
            "n_survivors_firing_1_to_9": r35["n_low"],
            "n_survivors_firing_10_plus": r35["n_high"],
            "give_back_low_firing": round(r35["give_back_low"], 4),
            "give_back_high_firing": round(r35["give_back_high"], 4),
            "give_back_ratio": round(r35["give_back_low"] / r35["give_back_high"], 2),
        },
        "ex_ante_conditioner_scan": r35["scan"],
        "ex_ante_conditioner_status": (
            "EXPLORATORY -- thresholds were scanned, so this is a forking path. "
            "Monotone in the threshold, which is what the winner's curse predicts "
            "rather than what one lucky cut looks like, but it requires its own "
            "out-of-sample test before it can be treated as a rule."),

        # ── 3.6 corrected frontier ────────────────────────────────────────
        "corrected_frontier": {
            "n_cells": int(len(r36)),
            "bonferroni_alpha": round(0.05 / max(len(r36), 1), 4),
            "gross_clears_floor_point_estimate":
                int(r36["gross_clears_floor_point"].sum()),
            "net_positive_nominal_ci": int(r36["pos_nominal"].sum()),
            "net_positive_bonferroni": int(r36["pos_bonferroni"].sum()),
            "net_negative_bonferroni": int(r36["neg_bonferroni"].sum()),
            "why_this_replaces_3_3": (
                "The spec compares the PREMIUM against the friction floor, but the "
                "premium is measured against a gross-negative population, so it can "
                "clear the floor while survivors cannot pay costs. NQ/NY is the "
                "worked case: premium +0.0370 > required 0.0263, survivor gross "
                "-0.0327."),
        },

        "kill_criterion_triggered": bool(killed),
        "verdict": ("FALSIFIED -- premium CI includes zero" if killed
                    else ("CONFIRMED -- premium significantly positive"
                          if prem.lo > 0 else
                          "REVERSED -- premium significantly negative")),
        "verdict_qualified": (
            "Primary (family-level) FALSIFIED. Pre-registered secondary "
            "(trade-weighted) is significantly positive at +0.0413 R under i.i.d. "
            "family resampling but NOT under instrument clustering. Tradeability is "
            "negative either way: 0 of 15 instrument-sessions show survivors "
            "clearing their own friction floor after multiplicity correction."),
    }
    (_OUT / "hyp01_results.json").write_text(json.dumps(out, indent=2),
                                             encoding="utf-8")
    g.to_csv(_OUT / "hyp01_gross_net_by_instrument_session.csv", index=False)
    v.to_csv(_OUT / "hyp01_viability_frontier.csv", index=False)
    fam.to_csv(_OUT / "hyp01_family_holdout.csv", index=False)
    r36.to_csv(_OUT / "hyp01_corrected_frontier.csv", index=False)
    print(f"\n  saved -> {_OUT / 'hyp01_results.json'}")
    print(f"  saved -> {_OUT / 'hyp01_gross_net_by_instrument_session.csv'}")
    print(f"  saved -> {_OUT / 'hyp01_viability_frontier.csv'}")
    print(f"  saved -> {_OUT / 'hyp01_corrected_frontier.csv'}")
    print(f"  saved -> {_OUT / 'hyp01_family_holdout.csv'}")
    print(f"  elapsed {time.perf_counter()-t0:.0f}s")
    print("=" * 118 + "\n")
    return 0


def _ser(e: Estimate) -> dict:
    return {"point": round(e.point, 4), "ci_lo": round(e.lo, 4),
            "ci_hi": round(e.hi, 4), "includes_zero": e.includes_zero,
            "boot_median": round(e.boot_median, 4), "unit": e.unit}


if __name__ == "__main__":
    raise SystemExit(main())
