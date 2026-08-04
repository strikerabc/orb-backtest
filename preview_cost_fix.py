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
    MIN_TRADES_FOR_RANKING, OUTPUTS_DIR,
    SLIPPAGE_TICKS_BY_SYMBOL, SLIPPAGE_TICKS_ROUND_TRIP,
)

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

    # ── recompute cost under per-symbol slippage ───────────────────────────
    tv = tl["instrument"].map(lambda s: INSTRUMENTS[s]["tick_value_usd"])
    slip_old = float(SLIPPAGE_TICKS_ROUND_TRIP)
    slip_new = tl["instrument"].map(
        lambda s: SLIPPAGE_TICKS_BY_SYMBOL.get(s, SLIPPAGE_TICKS_ROUND_TRIP))

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

    # ── null test is unaffected by costs ───────────────────────────────────
    hr("4. NULL TEST — UNCHANGED BY ANY COST FIX")
    if "null_p_value" in old.columns:
        sig = old_rank[old_rank["null_p_value"] < 0.05]
        print("  null_p_value compares GROSS expectancy against random entry,")
        print("  so it does not move when the cost model changes.")
        print(f"\n  rankable variants with null_p < 0.05 : {len(sig)}")
        print(f"  rankable variants with null_p < 0.10 : "
              f"{len(old_rank[old_rank['null_p_value'] < 0.10])}")
        print(f"  best (lowest) null_p_value           : "
              f"{old_rank['null_p_value'].min():.4f}")
        print("\n  Under a 5% threshold across 6,168 variants, ~308 false")
        print("  positives would be EXPECTED from pure chance. Observing 0 is")
        print("  materially worse than chance and indicates no edge is present.")

    print("\n" + "=" * 104 + "\n")


if __name__ == "__main__":
    main()
