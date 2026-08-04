"""
diagnose_zn_dominance.py — why does ZN top the table at null_p = 0.001?

The suspicion
-------------
Every ZN row in the top 20 shows gross +0.21..+0.28 and net -0.32..-0.52, with
null_p pinned at 0.001 -- the FLOOR of a 1000-resample test, i.e. zero null
means beat it. Something systematic is driving ZN's gross expectancy.

Prime suspect: stats.py filters only `exit_reason != "INVALID"` and never
references `tp_unfillable`. The flag exists in the trade log but was never
applied, so take-profits sitting inside one tick of entry -- physically
unfillable -- still count as wins in the gross expectancy the report ranks on.

Why this would hit ZN hardest, and why it would NOT cancel in the null test
--------------------------------------------------------------------------
TI (tap-in) enters AT the range boundary. The swing low used for the stop is
then very close, so r_ticks is tiny (ZN median was 4). At rr=0.25 a 2-tick
stop puts the TP 0.5 ticks away: inside the spread, unfillable, yet recorded
as a win.

The null pool uses the same swing-SL machinery, so it also contains some
unfillable TPs -- but its entries are placed at RANDOM points in the session,
not at the boundary, so its r_ticks distribution is systematically wider. The
observed set is enriched in tiny-r trades relative to the null. The artefact
therefore inflates observed gross more than null gross and does not cancel.

What this script establishes
----------------------------
1. Is tp_unfillable present, and what is the rate per instrument?
2. For the top-20-by-gross variants, what fraction of trades are unfillable?
3. Recomputed expectancy EXCLUDING unfillable trades -- does the edge survive?
4. Headline: variants that are net-positive AND null_p < 0.05, before and
   after excluding unfillable trades.
5. ZN r_ticks distribution on the corrected .v.0 data.

Reads cached parquet only. No API calls, no spend.

Usage:
    python diagnose_zn_dominance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import (
    COMMISSION_PER_SIDE_USD, INSTRUMENTS, MIN_TRADES_FOR_RANKING, OUTPUTS_DIR,
)
from src.trade_sim import slippage_ticks_for

_OUT = _ROOT / OUTPUTS_DIR
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

VARIANT_KEYS = ["instrument", "session", "range_minutes",
                "entry_mode", "closure_tf", "direction", "rr"]


def hr(t: str) -> None:
    print("\n" + "=" * 110)
    print(t)
    print("=" * 110)


def main() -> None:
    tl_p, sm_p = _OUT / "trade_log.parquet", _OUT / "summary.parquet"
    if not (tl_p.exists() and sm_p.exists()):
        print("missing outputs; run main.py first")
        sys.exit(1)

    sm = pd.read_parquet(sm_p)
    rank = sm[sm["trade_count"] >= MIN_TRADES_FOR_RANKING].copy()
    print(f"summary: {len(sm):,} variants, {len(rank):,} rankable")

    cols = VARIANT_KEYS + ["gross_r", "net_r", "r_ticks", "exit_reason"]
    tl = pd.read_parquet(tl_p, columns=cols + ["tp_ticks", "tp_unfillable"])
    tl = tl[tl["exit_reason"].notna() & (tl["exit_reason"] != "INVALID")].copy()
    print(f"trade log: {len(tl):,} valid rows")

    # ── 1. unfillable rate per instrument ──────────────────────────────────
    hr("1. UNFILLABLE-TP RATE PER INSTRUMENT (flag exists but was never applied)")
    u = (tl.groupby("instrument", observed=True)
           .agg(trades=("tp_unfillable", "size"),
                unfillable=("tp_unfillable", "sum"),
                median_r_ticks=("r_ticks", "median"))
           .reset_index())
    u["pct"] = (100.0 * u["unfillable"] / u["trades"]).round(2)
    print(u.sort_values("pct", ascending=False).to_string(index=False))
    print(f"\ntotal unfillable: {int(tl['tp_unfillable'].sum()):,} of {len(tl):,} "
          f"({100.0*tl['tp_unfillable'].mean():.2f}%)")

    # ── 2. top-20-by-gross: how much is unfillable? ────────────────────────
    hr("2. TOP 20 BY GROSS — unfillable share of each variant's trades")
    top20 = rank.sort_values("expectancy_gross_r", ascending=False).head(20)
    key_cols = VARIANT_KEYS
    per_var = (tl.groupby(key_cols, observed=True)
                 .agg(n=("tp_unfillable", "size"),
                      n_unfill=("tp_unfillable", "sum"),
                      med_r=("r_ticks", "median"),
                      med_tp=("tp_ticks", "median"))
                 .reset_index())
    m = top20.merge(per_var, on=key_cols, how="left")
    m["unfill_pct"] = (100.0 * m["n_unfill"] / m["n"]).round(1)
    show = ["instrument", "session", "range_minutes", "entry_mode", "direction",
            "rr", "trade_count", "win_rate", "expectancy_gross_r",
            "expectancy_net_r", "null_p_value", "med_r", "med_tp", "unfill_pct"]
    print(m[[c for c in show if c in m.columns]].round(4).to_string(index=False))

    # ── 3. recompute excluding unfillable ──────────────────────────────────
    hr("3. EXPECTANCY RECOMPUTED EXCLUDING UNFILLABLE TRADES")
    keep = tl[~tl["tp_unfillable"]]
    print(f"dropped {len(tl)-len(keep):,} unfillable rows "
          f"({100.0*(1-len(keep)/len(tl)):.2f}%)")

    re_g = (keep.groupby(key_cols, observed=True)
                .agg(n_fillable=("gross_r", "size"),
                     gross_excl=("gross_r", "mean"),
                     net_excl=("net_r", "mean"),
                     win_excl=("gross_r", lambda s: float((s > 0).mean())))
                .reset_index())
    comp = (rank.merge(re_g, on=key_cols, how="left")
                .dropna(subset=["gross_excl"]))
    comp["gross_delta"] = comp["gross_excl"] - comp["expectancy_gross_r"]

    print("\nmedian change in gross expectancy, per instrument:")
    g = (comp.groupby("instrument", observed=True)
             .agg(variants=("gross_delta", "size"),
                  med_gross_before=("expectancy_gross_r", "median"),
                  med_gross_after=("gross_excl", "median"),
                  med_delta=("gross_delta", "median"))
             .reset_index()
             .sort_values("med_delta"))
    print(g.round(4).to_string(index=False))

    print("\nsame 20 variants, gross before vs after exclusion:")
    t20 = comp.sort_values("expectancy_gross_r", ascending=False).head(20)
    print(t20[["instrument", "session", "entry_mode", "direction", "rr",
               "trade_count", "n_fillable", "expectancy_gross_r", "gross_excl",
               "gross_delta", "expectancy_net_r", "net_excl", "null_p_value"]]
          .round(4).to_string(index=False))

    # ── 4. the headline number ─────────────────────────────────────────────
    hr("4. HEADLINE — net-positive AND statistically significant")
    has_p = "null_p_value" in rank.columns
    if has_p:
        for label, df_, netcol in (
            ("as reported (unfillable INCLUDED)", rank, "expectancy_net_r"),
            ("unfillable EXCLUDED", comp, "net_excl"),
        ):
            net_pos = df_[df_[netcol] > 0]
            sig = df_[df_["null_p_value"] < 0.05]
            both = df_[(df_[netcol] > 0) & (df_["null_p_value"] < 0.05)]
            print(f"\n  {label}")
            print(f"    rankable                    : {len(df_):,}")
            print(f"    net-positive                : {len(net_pos):,}")
            print(f"    null_p < 0.05               : {len(sig):,}")
            print(f"    BOTH net-positive AND sig   : {len(both):,}")
            if not both.empty:
                c = [x for x in ["instrument", "session", "range_minutes",
                                 "entry_mode", "closure_tf", "direction", "rr",
                                 "trade_count", "expectancy_gross_r", netcol,
                                 "null_p_value"] if x in both.columns]
                print(both.sort_values(netcol, ascending=False)
                          .head(20)[c].round(4).to_string(index=False))

        print("\n  null_p distribution over rankable variants:")
        q = rank["null_p_value"].describe(
            percentiles=[.01, .05, .25, .5, .75, .95])
        print("    " + q.round(4).to_string().replace("\n", "\n    "))
        print(f"\n    at floor (0.001): "
              f"{int((rank['null_p_value'] <= 0.001).sum()):,} variants")

    # ── 5. ZN stop distances on corrected data ─────────────────────────────
    hr("5. ZN STOP DISTANCES ON CORRECTED .v.0 DATA")
    zn = tl[tl["instrument"] == "ZN"]
    if not zn.empty:
        tv = INSTRUMENTS["ZN"]["tick_value_usd"]
        comm = 2.0 * COMMISSION_PER_SIDE_USD / tv
        print(f"  commission {comm:.3f} ticks + measured slippage "
              f"{slippage_ticks_for('ZN','LDN'):.2f} ticks")
        for mode, gg in zn.groupby("entry_mode", observed=True):
            r = gg["r_ticks"].to_numpy()
            cost = (comm + slippage_ticks_for("ZN", "LDN")) / np.median(r)
            print(f"    {mode:<5} n={len(gg):>7,}  r_ticks p05/med/p95 = "
                  f"{np.percentile(r,5):>5.1f}/{np.median(r):>5.1f}/"
                  f"{np.percentile(r,95):>6.1f}   "
                  f"cost at median r = {cost:.3f} R  "
                  f"unfillable {100.0*gg['tp_unfillable'].mean():>5.1f}%")

    print("\n" + "=" * 110 + "\n")


if __name__ == "__main__":
    main()
