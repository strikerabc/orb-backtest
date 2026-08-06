"""
test_holdout.py — do the survivors hold up on data never used for selection?

Two problems this addresses
---------------------------
1. THE 7,140 "TESTS" ARE NOT INDEPENDENT.
   For one entry signal (instrument, session, range_minutes, entry_mode,
   closure_tf, direction) all six rr levels share the SAME entries -- they
   differ only in where the exit sits. In the top-25 survivors,
   NQ/NY/15m/R-CC/15c/short appears at rr 2.0, 1.5 and 0.75, and
   NQ/NY/15m/CC/15c/short at rr 2.0, 1.5, 1.0 and 0.75. Those are one signal
   counted repeatedly, not independent discoveries.

   The engine itself already knows the real count: build_r_ticks_map reported
   exactly 1,350 variant FAMILIES. So the effective number of independent tests
   is ~1,350, not 7,140. This script collapses survivors to families.

2. EVERYTHING SO FAR IS IN-SAMPLE.
   Regime windows stop at 2025-08-01 -> 2026-01-31 (W09), while the data runs to
   2026-07-31. That leaves ~6 months (2026-02-01 onward) that NO variant
   selection has ever touched -- stronger than the configured 3-month holdout.

   Selection used only pre-holdout data, so holdout performance is an honest
   out-of-sample read. In-sample significance cannot fake it.

Method
------
  - load summary, take variants that are net-positive AND null_p < 0.05
  - collapse to distinct signal families
  - re-run the engine on the holdout period only
  - compare in-sample vs holdout expectancy per family

A real edge should carry SOME of its in-sample expectancy into the holdout.
Selection artefacts collapse toward zero or flip sign.

Reads cached parquet only. No API calls, no spend.

Usage:
    python test_holdout.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import (
    INSTRUMENTS, MIN_TRADES_FOR_RANKING, OUTPUTS_DIR, RR_LEVELS, SESSIONS,
)
from src.data_layer import _compute_enrichment, ensure_daily, ensure_data
from src.entry_detector import detect_entries
from src.range_builder import build_session_days
from src.trade_sim import simulate_trade
from src.filters import trade_eligibility

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("orb.holdout")
_OUT = _ROOT / OUTPUTS_DIR
pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 200)

FAMILY = ["instrument", "session", "range_minutes", "entry_mode",
          "closure_tf", "direction"]

# Last regime window ends 2026-01-31; data runs to 2026-07-31.
HOLDOUT_START = "2026-02-01"


def hr(t: str) -> None:
    print("\n" + "=" * 116)
    print(t)
    print("=" * 116)


def main() -> None:
    t0 = time.perf_counter()

    sm = pd.read_parquet(_OUT / "summary.parquet")
    rank = sm[sm["trade_count"] >= MIN_TRADES_FOR_RANKING].copy()
    surv = rank[(rank["expectancy_net_r"] > 0) & (rank["null_p_value"] < 0.05)]

    hr("1. COLLAPSING SURVIVORS TO INDEPENDENT SIGNAL FAMILIES")
    print(f"  rankable variants                 : {len(rank):,}")
    print(f"  net-positive AND null_p < 0.05     : {len(surv):,}")

    fams = surv.groupby(FAMILY, observed=True).agg(
        n_rr=("rr", "size"),
        rr_list=("rr", lambda s: ",".join(f"{x:g}" for x in sorted(s))),
        best_net=("expectancy_net_r", "max"),
        best_rr=("rr", lambda s: np.nan),
        min_p=("null_p_value", "min"),
        trades=("trade_count", "max"),
    ).reset_index()
    # best_rr properly: rr of the max-net row per family
    idx = surv.groupby(FAMILY, observed=True)["expectancy_net_r"].idxmax()
    best = surv.loc[idx, FAMILY + ["rr", "expectancy_gross_r",
                                   "expectancy_net_r", "null_p_value"]]
    fams = fams.drop(columns=["best_rr"]).merge(
        best.rename(columns={"rr": "best_rr"}), on=FAMILY, how="left")

    print(f"  DISTINCT SIGNAL FAMILIES          : {len(fams):,}")
    print(f"  average rr levels per family      : {fams['n_rr'].mean():.1f}")
    print()
    print("  Multiple-comparisons context:")
    n_fam_total = 1350
    print(f"    independent families in sweep    : ~{n_fam_total:,}")
    print(f"    expected false positives at 5%   : ~{int(0.05*n_fam_total)}")
    print(f"    families observed                : {len(fams)}")
    if len(fams) < 0.05 * n_fam_total:
        print("    -> AT OR BELOW the chance rate. Cannot be distinguished")
        print("       from multiple-comparisons noise on in-sample data alone.")
    else:
        print("    -> above chance, but that alone does not identify WHICH")
        print("       families are real. Holdout does.")

    print("\n  survivor families by instrument:")
    print(fams.groupby("instrument", observed=True)
              .agg(families=("n_rr", "size"),
                   best_net=("expectancy_net_r", "max"))
              .sort_values("families", ascending=False)
              .round(4).to_string())

    # ── 2. run the engine on the holdout period ────────────────────────────
    hr(f"2. OUT-OF-SAMPLE TEST — holdout from {HOLDOUT_START}")
    print("  Regime windows end 2026-01-31; this data was never used for")
    print("  selection, so it is an honest out-of-sample read.\n")

    syms = sorted(fams["instrument"].unique())
    rows = []
    for sym in syms:
        df = _compute_enrichment(ensure_data(sym), ensure_daily(sym),
                                 tick_size=INSTRUMENTS[sym]["tick_size"])
        df = df[df["timestamp"] >= pd.Timestamp(HOLDOUT_START, tz="UTC")]
        if df.empty:
            print(f"  {sym}: no holdout data")
            continue

        sub = fams[fams["instrument"] == sym]
        for sess in sorted(sub["session"].unique()):
            sdays = build_session_days(df, sym, sess)
            if not sdays:
                continue
            for sd in sdays:
                for es in detect_entries(sd):
                    for tr in simulate_trade(es, sd, RR_LEVELS):
                        rows.append({
                            "instrument": sym, "session": sess,
                            "range_minutes": es.range_minutes,
                            "entry_mode": es.mode, "closure_tf": es.closure_tf,
                            "direction": es.direction, "rr": tr.rr,
                            "gross_r": tr.gross_r, "net_r": tr.net_r,
                            "exit_reason": tr.exit_reason,
                            "tp_unfillable": tr.tp_unfillable,
                            "tp_ticks": tr.tp_ticks, "r_ticks": tr.r_ticks,
                            "cost_r": tr.cost_r,
                            "contract_changed_in_session": sd.contract_changed_in_session,
                            "contract_changed_since_prev_session": sd.contract_changed_since_prev_session,
                            "session_bar_completeness": sd.session_bar_completeness,
                        })
        print(f"  {sym}: holdout bars {len(df):,}, "
              f"{df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")

    if not rows:
        print("\n  no holdout trades generated")
        return
    ho = trade_eligibility(pd.DataFrame(rows))
    ho = ho[ho["eligible"]].copy()
    print(f"\n  holdout trades simulated: {len(ho):,}")

    ho_agg = (ho.groupby(FAMILY + ["rr"], observed=True)
                .agg(ho_trades=("net_r", "size"),
                     ho_gross=("gross_r", "mean"),
                     ho_net=("net_r", "mean"),
                     ho_win=("gross_r", lambda s: float((s > 0).mean())))
                .reset_index())

    # ── 3. in-sample vs holdout, per surviving variant ─────────────────────
    hr("3. IN-SAMPLE vs HOLDOUT (surviving variants)")
    comp = surv.merge(ho_agg, on=FAMILY + ["rr"], how="left")
    comp = comp.dropna(subset=["ho_net"])
    if comp.empty:
        print("  no overlap between survivors and holdout trades")
        return

    comp["net_delta"] = comp["ho_net"] - comp["expectancy_net_r"]
    comp["held_up"] = comp["ho_net"] > 0

    show = ["instrument", "session", "range_minutes", "entry_mode",
            "closure_tf", "direction", "rr", "trade_count",
            "expectancy_net_r", "ho_trades", "ho_net", "net_delta"]
    print(comp.sort_values("expectancy_net_r", ascending=False)
              .head(30)[show].round(4).to_string(index=False))

    # ── 4. family-level pooled test (the decisive one) ─────────────────────
    hr("4. FAMILY-LEVEL POOLED TEST")
    print("  Variant-level counting is misleading: a family whose rr=2.0 and")
    print("  rr=1.5 both survived contributes TWICE, though they share entries")
    print("  and differ only in exit placement. Collapsing to one row per")
    print("  family (its best-net rr) removes that double count.\n")

    best_idx = comp.groupby(FAMILY, observed=True)["expectancy_net_r"].idxmax()
    fam_comp = comp.loc[best_idx]

    w = fam_comp["ho_trades"].to_numpy(float)
    x = fam_comp["ho_net"].to_numpy(float)
    pooled = float(np.sum(w * x) / np.sum(w)) if w.sum() else float("nan")
    simple = float(np.mean(x))

    rng = np.random.default_rng(0)
    boots = [float(np.mean(rng.choice(x, size=len(x), replace=True)))
             for _ in range(10000)]
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    held_fam = int((x > 0).sum())

    print(f"  families with holdout data         : {len(fam_comp)}")
    print(f"  net-positive out-of-sample         : {held_fam} of {len(fam_comp)} "
          f"({100.0*held_fam/len(fam_comp):.1f}%)")
    print(f"  trade-weighted mean holdout net_r  : {pooled:+.4f}")
    print(f"  simple mean across families        : {simple:+.4f}")
    print(f"  bootstrap 95% CI on the mean       : [{lo:+.4f}, {hi:+.4f}]")
    print(f"  total holdout trades pooled        : {int(w.sum()):,}")
    print()
    print("  Trade-weighting is the honest statistic: families with more")
    print("  holdout trades carry more information. Where the two disagree, the")
    print("  weighted figure governs.")

    # ── 5. multiple-comparisons context at family level ────────────────────
    hr("5. MULTIPLE COMPARISONS — is 48 even above chance?")
    n_fam_rank = rank.groupby(FAMILY, observed=True).ngroups
    exp_fp = int(0.05 * n_fam_rank)
    print(f"  rankable families in sweep         : {n_fam_rank:,}")
    print(f"  expected false positives at 5%     : ~{exp_fp}")
    print(f"  families net-positive AND sig      : {len(fams)}")
    print(f"  -> {100.0*len(fams)/n_fam_rank:.1f}% of families, vs 5.0% by chance")
    print()
    if len(fams) <= exp_fp:
        print("  AT OR BELOW the chance rate. The survivor set cannot be")
        print("  distinguished from multiple-comparisons noise on in-sample")
        print("  evidence alone, independent of anything the holdout shows.")

    hr("6. VERDICT")
    ci_includes_zero = lo <= 0 <= hi
    print(f"  in-sample survivors below chance rate : {len(fams) <= exp_fp}")
    print(f"  holdout net-positive rate             : "
          f"{100.0*held_fam/len(fam_comp):.1f}%  (chance = 50%)")
    print(f"  trade-weighted holdout net_r          : {pooled:+.4f}")
    print(f"  bootstrap CI includes zero            : {ci_includes_zero}")
    print()
    if pooled <= 0 or ci_includes_zero:
        print("  NO EDGE ESTABLISHED.")
        print("  The in-sample survivor count sits at or below what multiple")
        print("  comparisons produce by chance; out-of-sample the families are")
        print("  indistinguishable from a coin flip, and the trade-weighted mean")
        print("  is not distinguishable from zero. The ranked in-sample tables")
        print("  are best read as selection artefacts.")
    else:
        print("  Survives both checks. Treat as a small candidate edge requiring")
        print("  forward testing before any capital is committed.")

    print("\n  Power caveat: at a median of "
          f"{fam_comp['ho_trades'].median():.0f} holdout trades and per-trade")
    print("  sd ~1.0 R, the SE on a single family's holdout mean is ~"
          f"{1.0/np.sqrt(fam_comp['ho_trades'].median()):.2f} R. Individual rows")
    print("  are therefore uninformative; only the pooled figure carries weight.")
    print("  A null result here does not prove no edge exists -- it establishes")
    print("  that this sweep did not find one.")

    out = _OUT / "holdout_test.csv"
    comp.to_csv(out, index=False)
    fams.to_csv(_OUT / "survivor_families.csv", index=False)

    # ── machine-readable verdict for report.py ─────────────────────────────
    # Written as JSON rather than hardcoded into report.py so the report always
    # reflects the LATEST holdout run instead of carrying stale claims.
    verdict = {
        "holdout_start": HOLDOUT_START,
        "families_rankable": int(n_fam_rank),
        "expected_fp_at_5pct": int(exp_fp),
        "survivor_families": int(len(fams)),
        "survivor_variants": int(len(surv)),
        "survivor_pct_of_families": round(100.0 * len(fams) / n_fam_rank, 2),
        "holdout_families_tested": int(len(fam_comp)),
        "holdout_net_positive": int(held_fam),
        "holdout_net_positive_pct": round(100.0 * held_fam / len(fam_comp), 1),
        "holdout_trade_weighted_net_r": round(pooled, 4),
        "holdout_simple_mean_net_r": round(simple, 4),
        "holdout_ci_lo": round(lo, 4),
        "holdout_ci_hi": round(hi, 4),
        "holdout_trades_pooled": int(w.sum()),
        "median_holdout_trades": int(fam_comp["ho_trades"].median()),
        "below_chance_rate": bool(len(fams) <= exp_fp),
        "ci_includes_zero": bool(ci_includes_zero),
        "verdict": ("NO EDGE ESTABLISHED"
                    if (pooled <= 0 or ci_includes_zero)
                    else "CANDIDATE EDGE -- requires forward testing"),
    }
    vpath = _OUT / "holdout_verdict.json"
    vpath.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(f"\n  saved -> {out}")
    print(f"  saved -> {_OUT / 'survivor_families.csv'}")
    print(f"  saved -> {vpath}")
    print(f"  elapsed {time.perf_counter()-t0:.0f}s")
    print("\n" + "=" * 116 + "\n")


if __name__ == "__main__":
    main()
