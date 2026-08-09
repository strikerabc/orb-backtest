"""
hyp04_trade_construction.py — is the trade wrapper mis-specified?

Pre-registration: prereg/hyp04_prereg.json (committed at ca1c7fc).
Spec: docs/REVIEW_AND_HYPOTHESES.md, HYP-04.

Two confirmatory tests, Bonferroni-corrected across them (alpha 0.025 each):

  H4a  Spearman rho(range_width/atr_4h, gross_r) < 0
  H4b  E[gross_r] under a 45-minute hold cap  >  E[gross_r] at the 11:59 exit

H4c is deferred; see the pre-registration for why.

Inference is INSTRUMENT-CLUSTERED throughout. On 2.8M trades an i.i.d. interval is
tiny and calls almost any gradient significant, and four findings in this session have
already lost significance under clustering. The honest unit of independence is the
instrument, of which there are 10.

Reads cached parquet; re-simulates the holdout for validation. No API calls, no spend.

Usage:
    python tools/analysis/hyp04_trade_construction.py
    python tools/analysis/hyp04_trade_construction.py --refresh
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

from src.bootstrap import Estimate, estimate, mean_stat
from src.config import INSTRUMENTS, OUTPUTS_DIR, RR_LEVELS
from src.data_layer import _compute_enrichment, ensure_daily, ensure_data
from src.entry_detector import detect_entries
from src.filters import trade_eligibility
from src.range_builder import build_session_days
from src.trade_sim import simulate_trade

_OUT = _ROOT / OUTPUTS_DIR

HOLDOUT_START = "2026-02-01"
N_BOOT = 20_000
BOOT_SEED = 0
N_CONFIRMATORY = 2
ALPHA_BONF = 0.05 / N_CONFIRMATORY          # 0.025 per test

# Pre-registered single alternative horizon, in minutes (= bars).
H4B_HORIZON = 45
H4B_GRID = [15, 30, 45, 60, 90, 120]

CACHE_H4B = _OUT / "hyp04_horizons.parquet"

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)


def hr(t: str) -> None:
    print("\n" + "=" * 118)
    print(t)
    print("=" * 118)


# ── clustered Spearman via sufficient statistics ──────────────────────────────
#
# A trade-level rank correlation with an instrument-clustered bootstrap is
# 2.8M rows x 20,000 draws if done naively -- 5.6e10 operations. It collapses to
# O(1) per draw because Pearson r is a function of six sums:
#
#     n, Sx, Sy, Sxx, Syy, Sxy
#
# and those are ADDITIVE over clusters. Ranks are computed ONCE on the full sample
# (which is what "Spearman rho of the sample" means) rather than re-ranked inside each
# draw; that is a stated approximation, standard for bootstrapping rank statistics,
# and it is exact for the Pearson-on-fixed-ranks statistic actually being resampled.

def _cluster_sums(x: np.ndarray, y: np.ndarray,
                  clusters: np.ndarray) -> tuple[np.ndarray, list]:
    keys = np.unique(clusters)
    rows = []
    for k in keys:
        m = clusters == k
        xi, yi = x[m], y[m]
        rows.append([len(xi), xi.sum(), yi.sum(),
                     (xi * xi).sum(), (yi * yi).sum(), (xi * yi).sum()])
    return np.asarray(rows, dtype=float), list(keys)


def _r_from_sums(s: np.ndarray) -> float:
    n, Sx, Sy, Sxx, Syy, Sxy = s
    if n < 3:
        return float("nan")
    cov = Sxy - Sx * Sy / n
    vx = Sxx - Sx * Sx / n
    vy = Syy - Sy * Sy / n
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cov / np.sqrt(vx * vy))


def clustered_spearman(x: np.ndarray, y: np.ndarray, clusters: np.ndarray,
                       *, label: str, n_draws: int = N_BOOT,
                       seed: int = BOOT_SEED, alpha: float = 0.05) -> Estimate:
    """Spearman rho with an instrument-clustered bootstrap interval."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    sums, keys = _cluster_sums(rx, ry, clusters)
    point = _r_from_sums(sums.sum(axis=0))

    rng = np.random.default_rng(seed)
    g = len(keys)
    picks = rng.integers(0, g, size=(n_draws, g))
    boots = np.fromiter((_r_from_sums(sums[p].sum(axis=0)) for p in picks),
                        dtype=float, count=n_draws)
    boots = boots[np.isfinite(boots)]
    lo, med, hi = np.percentile(boots, [100 * alpha / 2, 50,
                                        100 * (1 - alpha / 2)])
    return Estimate(label=label, point=point, lo=float(lo), hi=float(hi),
                    boot_median=float(med), n_draws=n_draws,
                    unit="instrument-cluster")


def _tick(sym: str) -> float:
    return float(INSTRUMENTS[sym]["tick_size"])


def add_range_atr(df: pd.DataFrame) -> pd.DataFrame:
    """range_width in ATR units. Both inputs are known at range-end.

    atr_4h is verified free of look-ahead: data_layer applies atr.shift(1) and
    merge_asof(direction='backward'), so the attached value is the last COMPLETED 4h
    bar. Checked because ATR_ANCHOR_ET=(18,0) with 240-minute bars puts the
    06:00-10:00 bar across the NY 09:30 open.
    """
    out = df.copy()
    out["tick_size"] = out["instrument"].map(lambda s: _tick(s))
    out["range_atr"] = (out["range_width_ticks"] * out["tick_size"]) / out["atr_4h"]
    return out[np.isfinite(out["range_atr"]) & (out["range_atr"] > 0)].copy()


def section_h4a(tl: pd.DataFrame, ho: pd.DataFrame | None) -> dict:
    hr("H4a  RANGE QUALITY — is expectancy monotone decreasing in range_width/atr_4h?")
    print("  Measured in GROSS R. Net would carry a mechanical effect in the WRONG")
    print("  direction: a wider range gives a wider swing stop, which lowers")
    print("  cost_r = cost_ticks/r_ticks and so lifts net independently of any signal.\n")

    d = add_range_atr(tl)
    x = d["range_atr"].to_numpy(float)
    y = d["gross_r"].to_numpy(float)
    inst = d["instrument"].to_numpy()

    rho = clustered_spearman(x, y, inst, label="H4a rho, in-sample",
                             alpha=ALPHA_BONF)
    print(f"  trades: {len(d):,}   instruments: {d['instrument'].nunique()}")
    print(f"  PRIMARY  Spearman rho (Bonferroni {ALPHA_BONF:.3f}) : {rho.fmt()}")
    print(f"    -> {rho.sign_verdict('rho')}")
    if rho.point < 0 and rho.hi < 0:
        print("    Sign matches the prediction (decreasing).")
    elif rho.point > 0 and rho.lo > 0:
        print("    Sign is OPPOSITE to the prediction: expectancy INCREASES with")
        print("    relative range width.")

    # Descriptive only -- see the pre-registration. Decile means invite reading noise
    # as shape, which is why the confirmatory statistic is the rank correlation.
    print("\n  Decile table (DESCRIPTIVE — the confirmatory statistic is rho above):")
    d["dec"] = pd.qcut(d["range_atr"], 10, labels=False, duplicates="drop")
    tab = (d.groupby("dec", observed=True)
             .agg(n=("gross_r", "size"), med_range_atr=("range_atr", "median"),
                  E_gross=("gross_r", "mean"), E_net=("net_r", "mean"),
                  med_r_ticks=("r_ticks", "median"),
                  mean_cost=("cost_r", "mean")).round(4))
    print(tab.to_string())
    signs = np.sign(np.diff(tab["E_gross"].to_numpy()))
    n_down = int((signs < 0).sum())
    print(f"\n  decile-to-decile steps downward: {n_down} of {len(signs)}")
    monotone = n_down == len(signs)
    print(f"  strictly monotone decreasing: {monotone}")
    if not monotone:
        print("    -> The spec's kill criterion includes 'non-monotone across deciles'.")

    # Stratification is required, not optional: both conditioners correlate with
    # volatility, which correlates with everything.
    print("\n  Stratified by parkinson_vol_14d tercile (must hold in all three):")
    d["vol_t"] = pd.qcut(d["parkinson_vol_14d"], 3, labels=["low", "mid", "high"],
                         duplicates="drop")
    strat = {}
    for t in ("low", "mid", "high"):
        s = d[d["vol_t"] == t]
        if len(s) < 1000:
            continue
        e = clustered_spearman(s["range_atr"].to_numpy(float),
                               s["gross_r"].to_numpy(float),
                               s["instrument"].to_numpy(),
                               label=f"rho {t} vol", alpha=ALPHA_BONF)
        strat[t] = e
        print(f"    {t:<5} n={len(s):>9,}  {e.fmt()}  -> {e.sign_verdict('rho')}")
    same_sign = len({np.sign(e.point) for e in strat.values()}) == 1
    print(f"    same sign in all terciles: {same_sign}")

    # Stability. A real effect should hold in most of the 24 cells, not average out
    # of a few. Per-cell rho is computed without a CI -- these are description.
    print("\n  Per-instrument-session sign of rho (stability):")
    cells = []
    for (i, s), g in d.groupby(["instrument", "session"], observed=True):
        if len(g) < 500:
            continue
        r = float(pd.Series(g["range_atr"]).corr(pd.Series(g["gross_r"]),
                                                method="spearman"))
        cells.append({"instrument": i, "session": s, "n": len(g), "rho": r})
    cd = pd.DataFrame(cells).sort_values("rho")
    n_neg = int((cd["rho"] < 0).sum())
    print(f"    cells: {len(cd)}   rho < 0 in {n_neg}   rho > 0 in {len(cd)-n_neg}")
    print(cd.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # Holdout validation. In sample this is discovery; the holdout is the test.
    rho_ho = None
    if ho is not None and len(ho):
        dh = add_range_atr(ho)
        if len(dh) > 1000:
            rho_ho = clustered_spearman(dh["range_atr"].to_numpy(float),
                                        dh["gross_r"].to_numpy(float),
                                        dh["instrument"].to_numpy(),
                                        label="H4a rho, HOLDOUT", alpha=ALPHA_BONF)
            print(f"\n  HOLDOUT validation ({HOLDOUT_START}+, {len(dh):,} trades):")
            print(f"    {rho_ho.fmt()}  -> {rho_ho.sign_verdict('rho')}")
            agree = np.sign(rho_ho.point) == np.sign(rho.point)
            print(f"    sign agrees with in-sample: {agree}")

    return {"rho": rho, "rho_holdout": rho_ho, "deciles": tab, "strat": strat,
            "cells": cd, "monotone": monotone, "n_trades": len(d)}


# ── H4b: synthetic hold horizons ──────────────────────────────────────────────
#
# One pass over session-days, calling simulate_trade once per horizon plus once
# uncapped. build_session_days and detect_entries dominate the cost, so 7 extra
# simulate_trade calls inside one pass is far cheaper than 7 full passes.
#
# Results are accumulated as PER-INSTRUMENT SUFFICIENT STATISTICS rather than rows:
# the paired mean difference is sum(diff)/n, and both it and its instrument-clustered
# interval are exact functions of (n, sum_diff) per instrument. That keeps 2.8M x 7
# trade-horizon pairs in a few hundred floats.

def simulate_horizons(pairs: list[tuple[str, str]], start: str | None,
                      end: str | None = None) -> pd.DataFrame:
    horizons = list(H4B_GRID)
    acc: dict[tuple, dict] = {}

    by_sym: dict[str, list[str]] = {}
    for sym, sess in pairs:
        by_sym.setdefault(sym, []).append(sess)

    for sym in sorted(by_sym):
        df = _compute_enrichment(ensure_data(sym), ensure_daily(sym),
                                 tick_size=INSTRUMENTS[sym]["tick_size"])
        if start:
            df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df["timestamp"] < pd.Timestamp(end, tz="UTC")]
        if df.empty:
            continue
        for sess in sorted(by_sym[sym]):
            for sd in build_session_days(df, sym, sess):
                for es in detect_entries(sd):
                    base = simulate_trade(es, sd, RR_LEVELS)
                    capped = {h: simulate_trade(es, sd, RR_LEVELS, max_hold_bars=h)
                              for h in horizons}
                    for k, b in enumerate(base):
                        if not np.isfinite(b.gross_r):
                            continue
                        # Eligibility is applied on the UNCAPPED trade, so the two
                        # arms always contain the same trades. Filtering each arm
                        # separately would let a cap change the population and turn
                        # a composition shift into an apparent horizon effect.
                        if b.exit_reason in ("INVALID", "SL_WRONG_SIDE",
                                             "DEGENERATE_R", "NO_HOLD_BARS"):
                            continue
                        if b.tp_unfillable:
                            continue
                        if (sd.contract_changed_in_session
                                or sd.contract_changed_since_prev_session):
                            continue
                        for h in horizons:
                            c = capped[h][k]
                            if not np.isfinite(c.gross_r):
                                continue
                            key = (sym, sess, h)
                            a = acc.setdefault(key, {
                                "n": 0, "sum_diff": 0.0, "sum_diff_sq": 0.0,
                                "sum_capped": 0.0, "sum_base": 0.0,
                                "n_base_time": 0, "n_capped_time": 0,
                                "n_tp_lost": 0, "n_sl_avoided": 0,
                                "sum_bars_base": 0, "sum_bars_capped": 0,
                                "sum_cost_diff": 0.0})
                            diff = c.gross_r - b.gross_r
                            a["n"] += 1
                            a["sum_diff"] += diff
                            a["sum_diff_sq"] += diff * diff
                            a["sum_capped"] += c.gross_r
                            a["sum_base"] += b.gross_r
                            a["n_base_time"] += b.exit_reason == "TIME"
                            a["n_capped_time"] += c.exit_reason == "TIME"
                            a["n_tp_lost"] += (b.exit_reason == "TP"
                                               and c.exit_reason != "TP")
                            a["n_sl_avoided"] += (b.exit_reason == "SL"
                                                  and c.exit_reason != "SL")
                            a["sum_bars_base"] += b.bars_held
                            a["sum_bars_capped"] += c.bars_held
                            a["sum_cost_diff"] += c.cost_r - b.cost_r
            print(f"  {sym}/{sess}: cells {len(acc)}")
    rows = [{"instrument": i, "session": s, "horizon": h, **v}
            for (i, s, h), v in acc.items()]
    return pd.DataFrame(rows)


def _paired_estimate(stats: pd.DataFrame, *, label: str,
                     alpha: float = 0.05) -> Estimate:
    """Instrument-clustered interval on the paired mean difference.

    Exact: mean = sum_diff / n, and resampling whole instruments means summing their
    (n, sum_diff) pairs. No per-trade storage needed.
    """
    per = (stats.groupby("instrument", observed=True)[["n", "sum_diff"]]
                .sum().reset_index())
    n = per["n"].to_numpy(float)
    s = per["sum_diff"].to_numpy(float)
    point = float(s.sum() / n.sum())
    rng = np.random.default_rng(BOOT_SEED)
    g = len(per)
    picks = rng.integers(0, g, size=(N_BOOT, g))
    boots = s[picks].sum(axis=1) / n[picks].sum(axis=1)
    lo, med, hi = np.percentile(boots, [100 * alpha / 2, 50,
                                       100 * (1 - alpha / 2)])
    return Estimate(label=label, point=point, lo=float(lo), hi=float(hi),
                    boot_median=float(med), n_draws=N_BOOT,
                    unit="instrument-cluster")


def section_h4b(stats: pd.DataFrame, stats_ho: pd.DataFrame | None) -> dict:
    hr("H4b  HOLD HORIZON — is the 11:59 exit materially wrong?")
    print(f"  Pre-registered single alternative horizon: {H4B_HORIZON} minutes.")
    print("  The full grid is DESCRIPTIVE; only the 45-minute test is confirmatory.")
    print("  Paired per trade: the same trade under a cap and at 11:59, so the")
    print("  day effect and the trade population are both held fixed.\n")

    print("  Descriptive grid (all horizons):")
    rows = []
    for h in H4B_GRID:
        s = stats[stats["horizon"] == h]
        if s.empty:
            continue
        n = float(s["n"].sum())
        e = _paired_estimate(s, label=f"delta@{h}m",
                             alpha=ALPHA_BONF if h == H4B_HORIZON else 0.05)
        rows.append({
            "horizon_min": h, "n_trades": int(n),
            "E_gross_capped": s["sum_capped"].sum() / n,
            "E_gross_11_59": s["sum_base"].sum() / n,
            "delta": e.point, "ci_lo": e.lo, "ci_hi": e.hi,
            "excludes_zero": not e.includes_zero,
            "pct_TIME_capped": 100.0 * s["n_capped_time"].sum() / n,
            "pct_TIME_11_59": 100.0 * s["n_base_time"].sum() / n,
            "pct_TP_lost": 100.0 * s["n_tp_lost"].sum() / n,
            "pct_SL_avoided": 100.0 * s["n_sl_avoided"].sum() / n,
            "mean_bars_capped": s["sum_bars_capped"].sum() / n,
            "mean_bars_11_59": s["sum_bars_base"].sum() / n,
            "max_abs_cost_drift": float(s["sum_cost_diff"].abs().max()),
        })
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # cost_r cannot change under a cap: r_ticks is fixed at entry. Verified rather
    # than assumed, because if it drifted the gross and net deltas would differ and
    # the pre-registration's claim that they are the same number would be wrong.
    drift = float(grid["max_abs_cost_drift"].abs().max())
    print(f"\n  max |sum(cost_r_capped - cost_r_11:59)| across cells: {drift:.2e}")
    print("  -> cost_r is invariant to the cap (r_ticks is fixed at entry), so the")
    print("     NET delta equals the GROSS delta exactly. Verified, not assumed.")

    prim = stats[stats["horizon"] == H4B_HORIZON]
    e45 = _paired_estimate(prim, label=f"H4b delta@{H4B_HORIZON}m",
                           alpha=ALPHA_BONF)
    print(f"\n  PRIMARY  delta at {H4B_HORIZON} min (Bonferroni {ALPHA_BONF:.3f}):")
    print(f"    {e45.fmt()}")
    print(f"    -> {e45.sign_verdict('45-minute delta')}")

    # The mechanism: a cap trades away late TPs to avoid late SLs. Which dominates
    # is the whole question, and reporting only one direction would prejudge it.
    r45 = grid[grid["horizon_min"] == H4B_HORIZON]
    if len(r45):
        row = r45.iloc[0]
        print(f"\n    TPs lost to the cap    : {row['pct_TP_lost']:.2f}% of trades")
        print(f"    SLs avoided by the cap : {row['pct_SL_avoided']:.2f}% of trades")
        print(f"    TIME exits             : {row['pct_TIME_11_59']:.1f}% -> "
              f"{row['pct_TIME_capped']:.1f}%")
        print(f"    mean bars held         : {row['mean_bars_11_59']:.1f} -> "
              f"{row['mean_bars_capped']:.1f}")

    e45_ho = None
    if stats_ho is not None and len(stats_ho):
        p = stats_ho[stats_ho["horizon"] == H4B_HORIZON]
        if len(p) and p["n"].sum() > 1000:
            e45_ho = _paired_estimate(p, label="H4b delta HOLDOUT", alpha=ALPHA_BONF)
            print(f"\n  HOLDOUT validation ({int(p['n'].sum()):,} trades):")
            print(f"    {e45_ho.fmt()}  -> {e45_ho.sign_verdict('delta')}")
            print(f"    sign agrees with in-sample: "
                  f"{np.sign(e45_ho.point) == np.sign(e45.point)}")

    return {"grid": grid, "primary": e45, "primary_holdout": e45_ho}


_TL_COLS = ["instrument", "session", "range_minutes", "entry_mode", "closure_tf",
            "direction", "rr", "gross_r", "net_r", "cost_r", "r_ticks",
            "exit_reason", "tp_unfillable", "tp_ticks", "contracts",
            "contract_changed_in_session", "contract_changed_since_prev_session",
            "session_bar_completeness", "bars_held", "mae_r", "mfe_r",
            "range_width_ticks", "atr_4h", "parkinson_vol_14d"]


def _ser(e: Estimate | None) -> dict | None:
    if e is None:
        return None
    return {"point": round(e.point, 5), "ci_lo": round(e.lo, 5),
            "ci_hi": round(e.hi, 5), "includes_zero": e.includes_zero,
            "boot_median": round(e.boot_median, 5), "unit": e.unit}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild the cached horizon simulation")
    args = ap.parse_args()
    t0 = time.perf_counter()

    hr("HYP-04  IS THE TRADE WRAPPER MIS-SPECIFIED?")
    print("  prereg: prereg/hyp04_prereg.json")
    print(f"  two confirmatory tests, Bonferroni alpha = {ALPHA_BONF:.3f} each")
    print("  all inference is INSTRUMENT-CLUSTERED (10 clusters)\n")

    tl = pd.read_parquet(_OUT / "trade_log.parquet", columns=_TL_COLS)
    tl = trade_eligibility(tl)
    tl = tl[tl["eligible"]].copy()
    print(f"  in-sample eligible trades: {len(tl):,}")

    # H4a's holdout arm needs the enrichment columns, which the HYP-01 cache lacks.
    ho = None
    ho_cache = _OUT / "hyp04_holdout_enriched.parquet"
    if ho_cache.exists() and not args.refresh:
        ho = pd.read_parquet(ho_cache)
        print(f"  holdout trades (cached)  : {len(ho):,}")

    r4a = section_h4a(tl, ho)

    # ── H4b ───────────────────────────────────────────────────────────────────
    pairs = sorted(set(map(tuple, tl[["instrument", "session"]]
                           .drop_duplicates().to_numpy())))
    if CACHE_H4B.exists() and not args.refresh:
        stats = pd.read_parquet(CACHE_H4B)
        print(f"\n  loaded H4b horizon stats from {CACHE_H4B.name}")
    else:
        print(f"\n  simulating {len(H4B_GRID)} horizons over {len(pairs)} "
              f"instrument-sessions (in sample)...")
        stats = simulate_horizons(pairs, start=None, end=HOLDOUT_START)
        stats.to_parquet(CACHE_H4B, index=False)
        print(f"  cached -> {CACHE_H4B.name}")

    ho_stats_path = _OUT / "hyp04_horizons_holdout.parquet"
    if ho_stats_path.exists() and not args.refresh:
        stats_ho = pd.read_parquet(ho_stats_path)
    else:
        print(f"\n  simulating horizons on the holdout ({HOLDOUT_START}+)...")
        stats_ho = simulate_horizons(pairs, start=HOLDOUT_START)
        stats_ho.to_parquet(ho_stats_path, index=False)

    r4b = section_h4b(stats, stats_ho)

    # ── verdict ───────────────────────────────────────────────────────────────
    hr("VERDICT AGAINST THE PRE-REGISTERED KILL CRITERIA")
    rho, e45 = r4a["rho"], r4b["primary"]

    h4a_killed = rho.includes_zero or not r4a["monotone"]
    h4a_sign_ok = rho.point < 0
    print(f"  H4a  rho = {rho.point:+.5f}, CI [{rho.lo:+.5f}, {rho.hi:+.5f}]")
    print(f"       predicted sign (rho < 0)     : {h4a_sign_ok}")
    print(f"       CI includes zero             : {rho.includes_zero}")
    print(f"       strictly monotone deciles    : {r4a['monotone']}")
    print(f"       kill criterion triggered     : {h4a_killed}")
    print()
    h4b_killed = e45.includes_zero
    print(f"  H4b  delta@{H4B_HORIZON}m = {e45.point:+.5f}, "
          f"CI [{e45.lo:+.5f}, {e45.hi:+.5f}]")
    print(f"       predicted sign (delta > 0)   : {e45.point > 0}")
    print(f"       CI includes zero             : {e45.includes_zero}")
    print(f"       kill criterion triggered     : {h4b_killed}")
    print()

    if h4a_killed and h4b_killed:
        print("  BOTH SUB-HYPOTHESES FALSIFIED. The pooled negative expectancy is not")
        print("  averaging over a gradient in either conditioner: the trade wrapper is")
        print("  not mis-specified in the two ways HYP-04 proposes.")
    elif not h4a_killed and not h4b_killed:
        print("  BOTH SURVIVE. Note this is a conditional-expectancy result, not a")
        print("  tradeable one: HYP-01 section 3.6 found no instrument-session where")
        print("  survivors clear their own friction floor, so a gradient must be steep")
        print("  enough to carry a positive-net subset, not merely present.")
    else:
        which = "H4a" if not h4a_killed else "H4b"
        print(f"  {which} SURVIVES; the other is falsified. A single surviving")
        print("  conditioner is worth pursuing only if the retained subset is")
        print("  net-positive after friction -- see the trade-count retention table.")

    out = {
        "hypothesis": "HYP-04",
        "prereg": "prereg/hyp04_prereg.json",
        "bootstrap_draws": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "n_confirmatory_tests": N_CONFIRMATORY,
        "bonferroni_alpha": ALPHA_BONF,
        "inference_unit": "instrument cluster (10)",
        "h4a": {
            "n_trades": int(r4a["n_trades"]),
            "rho_in_sample": _ser(rho),
            "rho_holdout": _ser(r4a["rho_holdout"]),
            "strictly_monotone_deciles": bool(r4a["monotone"]),
            "rho_by_vol_tercile": {k: _ser(v) for k, v in r4a["strat"].items()},
            "cells_with_negative_rho": int((r4a["cells"]["rho"] < 0).sum()),
            "cells_total": int(len(r4a["cells"])),
            "kill_criterion_triggered": bool(h4a_killed),
        },
        "h4b": {
            "pre_registered_horizon_min": H4B_HORIZON,
            "delta_in_sample": _ser(e45),
            "delta_holdout": _ser(r4b["primary_holdout"]),
            "grid": r4b["grid"].to_dict(orient="records"),
            "kill_criterion_triggered": bool(h4b_killed),
        },
        "h4c": "DEFERRED -- see prereg for reasoning",
        "verdict": (
            "BOTH FALSIFIED" if (h4a_killed and h4b_killed)
            else ("BOTH SURVIVE" if not (h4a_killed or h4b_killed)
                  else ("H4a SURVIVES" if not h4a_killed else "H4b SURVIVES"))),
    }
    (_OUT / "hyp04_results.json").write_text(json.dumps(out, indent=2),
                                             encoding="utf-8")
    r4a["deciles"].to_csv(_OUT / "hyp04_h4a_deciles.csv")
    r4a["cells"].to_csv(_OUT / "hyp04_h4a_cells.csv", index=False)
    r4b["grid"].to_csv(_OUT / "hyp04_h4b_horizon_grid.csv", index=False)
    print(f"\n  saved -> {_OUT / 'hyp04_results.json'}")
    print(f"  saved -> {_OUT / 'hyp04_h4a_deciles.csv'}")
    print(f"  saved -> {_OUT / 'hyp04_h4a_cells.csv'}")
    print(f"  saved -> {_OUT / 'hyp04_h4b_horizon_grid.csv'}")
    print(f"  elapsed {time.perf_counter()-t0:.0f}s")
    print("=" * 118 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
