"""
diagnose_r_ticks.py — are the top variants riding unexecutable stop distances?

Hypothesis
----------
ZN dominates the ranked table on GROSS expectancy while every row is deeply
negative on NET. _round_cost_r() in trade_sim.py charges:

    comm_ticks  = round_trip_commission_usd / tick_value_usd
    total_ticks = comm_ticks + SLIPPAGE_TICKS_ROUND_TRIP
    cost_r      = total_ticks / r_ticks

cost_r explodes as r_ticks -> 0. If ZN's median r_ticks is ~2, a fixed 1-tick
round-trip slippage eats ~50% of R before commission, and the "edge" is an
artefact of stops too tight to fill in practice.

Worse: at rr=0.25 with r_ticks=2, the TP sits 0.5 ticks from entry -- inside
one tick, i.e. inside the spread. Such a fill cannot occur.

swing_detector.py no longer enforces a floor (SWING_MIN_SL_TICKS is unused
after the A5 fix); trade_sim only rejects r_ticks < 1. So 1-2 tick stops are
reachable.

This script measures r_ticks per instrument and counts sub-tick TPs.

Usage:
    python diagnose_r_ticks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import (
    INSTRUMENTS, MIN_TRADES_FOR_RANKING,
    OUTPUTS_DIR, SLIPPAGE_TICKS_ROUND_TRIP, INVALID_REASONS,
)
from src.contracts import DEFAULT_RT_COMMISSION_USD

_OUT = _ROOT / OUTPUTS_DIR
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 120)


def hr(t: str) -> None:
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def main() -> None:
    tl_p = _OUT / "trade_log.parquet"
    sm_p = _OUT / "summary.parquet"
    if not tl_p.exists():
        print(f"missing {tl_p}")
        sys.exit(1)

    tl = pd.read_parquet(tl_p, columns=[
        "instrument", "session", "rr", "entry_mode", "r_ticks",
        "gross_r", "net_r", "exit_reason",
    ])
    tl = tl[~tl["exit_reason"].isin(INVALID_REASONS)]

    # ── 1. r_ticks distribution per instrument ─────────────────────────────
    hr("1. STOP DISTANCE (r_ticks) PER INSTRUMENT")
    print("cost_r = (2*comm/tick_value + slippage) / r_ticks")
    print(f"cost model: ${DEFAULT_RT_COMMISSION_USD/2:.2f}/side, "
          f"{SLIPPAGE_TICKS_ROUND_TRIP} tick round-trip slippage\n")

    rows = []
    for sym, grp in tl.groupby("instrument", observed=True):
        rt = grp["r_ticks"].to_numpy()
        rt = rt[np.isfinite(rt)]
        if len(rt) == 0:
            continue
        tv = INSTRUMENTS[sym]["tick_value_usd"]
        comm_ticks = DEFAULT_RT_COMMISSION_USD / tv
        total_ticks = comm_ticks + SLIPPAGE_TICKS_ROUND_TRIP
        med = float(np.median(rt))
        rows.append({
            "sym": sym,
            "tick_value": tv,
            "cost_ticks": round(total_ticks, 3),
            "r_ticks_p05": round(float(np.percentile(rt, 5)), 2),
            "r_ticks_med": round(med, 2),
            "r_ticks_p95": round(float(np.percentile(rt, 95)), 2),
            "cost_R_at_med": round(total_ticks / med, 4) if med > 0 else np.nan,
            "pct_r_under_5t": round(100.0 * (rt < 5).mean(), 1),
            "pct_r_under_2t": round(100.0 * (rt < 2).mean(), 1),
        })
    dist = pd.DataFrame(rows).sort_values("cost_R_at_med", ascending=False)
    print(dist.to_string(index=False))
    print("\ncost_R_at_med = R lost to costs on a median-width stop.")
    print("Values near or above the gross expectancy mean the edge is not executable.")

    # ── 2. sub-tick take-profits ───────────────────────────────────────────
    hr("2. UNFILLABLE TAKE-PROFITS  (tp distance < 1 tick from entry)")
    print("tp_ticks = rr * r_ticks. Below 1 tick the TP is inside the spread.\n")
    tl["tp_ticks"] = tl["rr"] * tl["r_ticks"]
    sub = (tl.assign(unfillable=tl["tp_ticks"] < 1.0)
             .groupby("instrument", observed=True)
             .agg(rows=("tp_ticks", "size"),
                  median_tp_ticks=("tp_ticks", "median"),
                  pct_unfillable=("unfillable", lambda s: round(100.0 * s.mean(), 1)))
             .reset_index()
             .sort_values("pct_unfillable", ascending=False))
    print(sub.to_string(index=False))

    print("\nby instrument x rr (% of trades with TP under 1 tick):")
    piv = (tl.assign(u=tl["tp_ticks"] < 1.0)
             .pivot_table(index="instrument", columns="rr", values="u",
                          aggfunc=lambda s: round(100.0 * np.mean(s), 1),
                          observed=True))
    print(piv.to_string())

    # ── 3. gross vs net on the ranked set ──────────────────────────────────
    if sm_p.exists():
        hr("3. GROSS vs NET ON RANKABLE VARIANTS")
        sm = pd.read_parquet(sm_p)
        r = sm[sm["trade_count"] >= MIN_TRADES_FOR_RANKING].copy()
        r["cost_drag_r"] = r["expectancy_gross_r"] - r["expectancy_net_r"]
        g = (r.groupby("instrument", observed=True)
               .agg(variants=("cost_drag_r", "size"),
                    med_gross=("expectancy_gross_r", "median"),
                    med_net=("expectancy_net_r", "median"),
                    med_drag=("cost_drag_r", "median"),
                    pct_net_pos=("expectancy_net_r",
                                 lambda s: round(100.0 * (s > 0).mean(), 1)))
               .reset_index()
               .sort_values("med_drag", ascending=False))
        print(g.round(4).to_string(index=False))

        print("\nVariants profitable NET, ranked (the only ones that matter):")
        net_pos = r[r["expectancy_net_r"] > 0].sort_values(
            "expectancy_net_r", ascending=False)
        if net_pos.empty:
            print("  NONE.")
        else:
            cols = [c for c in ["instrument", "session", "range_minutes",
                                "entry_mode", "closure_tf", "direction", "rr",
                                "trade_count", "win_rate", "expectancy_gross_r",
                                "expectancy_net_r", "null_p_value"]
                    if c in net_pos.columns]
            print(net_pos.head(25)[cols].round(4).to_string(index=False))
            print(f"\n  total net-positive: {len(net_pos)} / {len(r)} rankable "
                  f"({100.0*len(net_pos)/len(r):.1f}%)")
            if "null_p_value" in net_pos.columns:
                sig = net_pos[net_pos["null_p_value"] < 0.05]
                print(f"  net-positive AND null_p < 0.05: {len(sig)}")

    print("\n" + "=" * 104 + "\n")


if __name__ == "__main__":
    main()
