"""
preview_cost_fix.py — what do the two fixes do, without re-running the sweep?

Both corrections are recomputable from the existing trade log:

  1. Per-symbol slippage. net_r = gross_r - cost_r, where
     cost_r = (2*COMMISSION/tick_value + slip_ticks) / r_ticks
     The log carries gross_r, r_ticks and instrument, so net_r can be
     recomputed under any slippage assumption.

  2. Unfillable-TP exclusion. tp_ticks = rr * r_ticks, so trades whose target
     sits inside MIN_TP_TICKS can be dropped and per-variant expectancy
     recomputed on what remains.

IMPORTANT: null_p_value is computed against random-entry GROSS expectancy and
is therefore INDEPENDENT of the cost model. Fixing slippage cannot rescue a
variant that already fails the null test. That result stands regardless.

Usage:
    python preview_cost_fix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import (
    COMMISSION_PER_SIDE_USD, INSTRUMENTS, MIN_TP_TICKS,
    MIN_TRADES_FOR_RANKING, OUTPUTS_DIR, SLIPPAGE_PROVENANCE,
    SLIPPAGE_TICKS_BY_SYMBOL, SLIPPAGE_TICKS_ROUND_TRIP,
)
# The engine's own resolver, imported rather than reimplemented.
from src.trade_sim import slippage_ticks_for

_OUT = _ROOT / OUTPUTS_DIR
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

VARIANT_KEYS = ["instrument", "session", "range_minutes",
                "entry_mode", "closure_tf", "direction", "rr"]


def hr(t: str) -> None:
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def main() -> None:
    tl_p = _OUT / "trade_log.parquet"
    if not tl_p.exists():
        print(f"missing {tl_p}")
        sys.exit(1)

    cols = VARIANT_KEYS + ["r_ticks", "gross_r", "net_r", "exit_reason"]
    tl = pd.read_parquet(tl_p, columns=cols)
    tl = tl[tl["exit_reason"] != "INVALID"].copy()
    print(f"loaded {len(tl):,} valid trades")

    # ── recompute cost under MEASURED per-(symbol, session) slippage ───────
    # Uses the same resolver the engine uses, so this preview cannot drift
    # from production behaviour -- the mistake made three times already this
    # session was a second copy of lookup logic diverging from the real one.
    tv = tl["instrument"].map(lambda s: INSTRUMENTS[s]["tick_value_usd"])
    slip_old = float(SLIPPAGE_TICKS_ROUND_TRIP)
    slip_new = [slippage_ticks_for(s, sess) for s, sess
                in zip(tl["instrument"], tl["session"])]
    slip_new = pd.Series(slip_new, index=tl.index, dtype=float)

    comm_ticks = 2.0 * COMMISSION_PER_SIDE_USD / tv
    rt = tl["r_ticks"].replace(0, np.nan)

    tl["cost_r_old"] = (comm_ticks + slip_old) / rt
    tl["cost_r_new"] = (comm_ticks + slip_new) / rt
    tl["net_r_new"]  = tl["gross_r"] - tl["cost_r_new"]

    # sanity: reproduce the stored net_r under the OLD model
    chk = (tl["gross_r"] - tl["cost_r_old"] - tl["net_r"]).abs()
    print(f"reproduction check vs stored net_r: max abs diff = {chk.max():.6f}")
    if chk.max() > 0.01:
        print("  WARNING: cannot reproduce stored net_r; treat results as indicative")

    # ── unfillable TPs ─────────────────────────────────────────────────────
    tl["tp_ticks"] = tl["rr"] * tl["r_ticks"]
    tl["unfillable"] = tl["tp_ticks"] < MIN_TP_TICKS
    hr("1. TRADES REMOVED BY THE UNFILLABLE-TP RULE")
    u = (tl.groupby("instrument", observed=True)["unfillable"]
           .agg(trades="size", n_unfillable="sum")
           .reset_index())
    u["pct"] = (100.0 * u["n_unfillable"] / u["trades"]).round(1)
    print(u.sort_values("pct", ascending=False).to_string(index=False))
    print(f"\ntotal dropped: {int(tl['unfillable'].sum()):,} of {len(tl):,} "
          f"({100.0*tl['unfillable'].mean():.2f}%)")

    # ── per-variant recompute on fillable trades only ──────────────────────
    keep = tl[~tl["unfillable"]]
    g = (keep.groupby(VARIANT_KEYS, observed=True)
             .agg(trade_count=("gross_r", "size"),
                  exp_gross=("gross_r", "mean"),
                  exp_net_new=("net_r_new", "mean"),
                  win_rate=("gross_r", lambda s: float((s > 0).mean())))
             .reset_index())
    g_rank = g[g["trade_count"] >= MIN_TRADES_FOR_RANKING]

    hr("2. NET-POSITIVE VARIANTS: OLD MODEL vs CORRECTED MODEL")
    old = pd.read_parquet(_OUT / "summary.parquet")
    old_rank = old[old["trade_count"] >= MIN_TRADES_FOR_RANKING]
    old_pos = old_rank[old_rank["expectancy_net_r"] > 0]
    new_pos = g_rank[g_rank["exp_net_new"] > 0]

    print(f"  OLD (flat 1-tick slippage, unfillable TPs counted):")
    print(f"    rankable {len(old_rank):,}  net-positive {len(old_pos):,} "
          f"({100.0*len(old_pos)/max(len(old_rank),1):.1f}%)")
    print(f"  NEW (per-symbol slippage, unfillable TPs excluded):")
    print(f"    rankable {len(g_rank):,}  net-positive {len(new_pos):,} "
          f"({100.0*len(new_pos)/max(len(g_rank),1):.1f}%)")

    print("\n  net-positive count by instrument:")
    a = old_pos.groupby("instrument", observed=True).size().rename("old")
    b = new_pos.groupby("instrument", observed=True).size().rename("new")
    comp = pd.concat([a, b], axis=1).fillna(0).astype(int)
    comp["delta"] = comp["new"] - comp["old"]
    print(comp.sort_values("old", ascending=False).to_string())

    hr("3. TOP 15 UNDER THE CORRECTED MODEL")
    if new_pos.empty:
        print("  NO variant is net-positive under the corrected cost model.")
    else:
        top = new_pos.sort_values("exp_net_new", ascending=False).head(15)
        print(top.round(4).to_string(index=False))

    # ── null test: the STORED values are void, not merely stale ────────────
    hr("4. NULL TEST — STORED VALUES ARE VOID, NOT USABLE")
    print("  The null_p_value column in summary.parquet came from the BROKEN")
    print("  statistic and must not be read at all -- not even as a baseline.")
    print()
    print("  It compared an observed MEAN against a pool of INDIVIDUAL random")
    print("  trades, so it tracked the random-entry TP hit rate ~1/(1+rr)")
    print("  rather than any tail probability:")
    print("      corr(null_p, expectancy_gross_r) = -0.0977   <- ~zero")
    print("      corr(null_p, win_rate)           = +0.6594")
    print("      min null_p observed              =  0.1661   <- a floor")
    print()
    print("  p < 0.05 was UNREACHABLE BY CONSTRUCTION. An earlier version of")
    print("  this script concluded '0 of 6,168 significant, ~308 expected by")
    print("  chance, therefore no edge'. That reasoning is RETRACTED: the")
    print("  ~308 figure assumes a uniform null p-distribution, which that")
    print("  statistic never produced. The test was evidence of nothing.")
    print()
    print(f"  Net-positive under measured costs : {len(new_pos)}")
    print("  Surviving a CALIBRATED null test  : UNKNOWN until main.py re-runs")
    print()
    print("  The rewritten test (null_calibrator.py) bootstraps null MEANS at")
    print("  the observed sample size. Validated synthetically: mean p 0.4884")
    print("  (target 0.50), 100% power on a +0.30 R edge at rr=1.0, but only")
    print("  47.5% power on a +0.15 R edge at rr=2.0 -- so a null result at")
    print("  high rr is weak evidence, not proof of absence.")

    # ── provenance of the cost model itself ────────────────────────────────
    hr("5. COST-MODEL PROVENANCE")
    print("  Slippage is now MEASURED (bbo-1m, entry-weighted), not estimated.")
    print("  Where entry TIMING came from the pre-roll-fix trade log, values")
    print("  are marked provisional and use max(entry-weighted, session median).")
    print()
    for sym in sorted(SLIPPAGE_PROVENANCE):
        tag = SLIPPAGE_PROVENANCE[sym]
        mark = "" if tag == "measured" else "   <- timing weights not yet final"
        print(f"    {sym:<5} {tag}{mark}")
    print()
    print("  Still excluded from every figure above: MARKET IMPACT. Measured")
    print("  spread is the quoted spread at the entry minute; ORB entries cross")
    print("  a book in motion, so these remain LOWER BOUNDS on true cost.")

    print("\n" + "=" * 104 + "\n")


if __name__ == "__main__":
    main()
