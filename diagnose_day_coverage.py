"""
diagnose_day_coverage.py — does the matched null compare the SAME DAYS?

The concern
-----------
The matched null draws random entries on EVERY session day in the pool, but an
observed variant only trades on days its entry condition actually fired. If the
two sets differ materially, null_p bundles two effects:

    day selection  -- did the strategy pick better DAYS?
    entry timing   -- did it pick a better MOMENT within a day?

The null was intended to isolate timing. Day selection is arguably part of the
strategy ("wait for a breakout"), but conflating them means null_p cannot be
read as a timing result.

What decides whether this matters
---------------------------------
Coverage. If a variant trades on ~95% of available session days, the two day
sets are nearly identical and the confound is negligible. If it trades on 40%,
the null is drawing from a materially different population and the comparison
is contaminated.

Second question: are traded days DIFFERENT from untraded ones? Compared here on
realised volatility and opening-range width. If they look alike, low coverage
matters less.

Reads cached parquet only. No API calls, no spend.

Usage:
    python diagnose_day_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import INSTRUMENTS, MIN_TRADES_FOR_RANKING, OUTPUTS_DIR

_OUT = _ROOT / OUTPUTS_DIR
pd.set_option("display.width", 220)

FAMILY = ["instrument", "session", "range_minutes", "entry_mode",
          "closure_tf", "direction"]


def hr(t: str) -> None:
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def main() -> None:
    tl_p = _OUT / "trade_log.parquet"
    if not tl_p.exists():
        print("missing outputs/trade_log.parquet")
        sys.exit(1)

    tl = pd.read_parquet(tl_p, columns=FAMILY + [
        "rr", "date", "exit_reason", "realized_vol_14d",
        "range_width_ticks", "gross_r",
    ])
    tl = tl[tl["exit_reason"] != "INVALID"]
    print(f"loaded {len(tl):,} valid trades")

    # Session-days available per (instrument, session): every distinct date the
    # instrument/session appears anywhere in the log. This is a LOWER BOUND on
    # true availability, since a date absent from every variant never appears.
    avail = (tl.groupby(["instrument", "session"], observed=True)["date"]
               .nunique().rename("days_available"))

    # Days each variant family actually traded (rr does not change entry timing)
    fam = (tl[tl["rr"] == 1.0]
           .groupby(FAMILY, observed=True)["date"]
           .nunique().rename("days_traded").reset_index())
    fam = fam.merge(avail, on=["instrument", "session"], how="left")
    fam["coverage_pct"] = (100.0 * fam["days_traded"] / fam["days_available"]).round(1)

    hr("1. DAY COVERAGE — fraction of available session-days a variant trades")
    print("High coverage => matched null draws from nearly the same day set.")
    print("Low coverage  => null_p bundles day selection with timing.\n")
    g = (fam.groupby(["instrument", "entry_mode"], observed=True)["coverage_pct"]
            .agg(families="size", min="min", median="median", max="max")
            .reset_index()
            .sort_values("median"))
    print(g.to_string(index=False))

    print("\n  coverage distribution across all families:")
    for lo, hi in [(0, 40), (40, 60), (60, 80), (80, 95), (95, 101)]:
        n = int(((fam["coverage_pct"] >= lo) & (fam["coverage_pct"] < hi)).sum())
        print(f"    {lo:>3}-{hi:>3}% : {n:>4} families "
              f"({100.0*n/len(fam):>5.1f}%)")

    # ── 2. do traded days differ from untraded days? ───────────────────────
    hr("2. ARE TRADED DAYS DIFFERENT? (realised vol, range width)")
    print("Per instrument/session, comparing days a TI variant traded against")
    print("days it did not. TI fires most often, so it is the strictest test\n")

    rows = []
    for (sym, sess), grp in tl.groupby(["instrument", "session"], observed=True):
        all_days = set(grp["date"].unique())
        ti = grp[(grp["entry_mode"] == "TI") & (grp["rr"] == 1.0)
                 & (grp["range_minutes"] == 15)]
        if ti.empty:
            continue
        traded = set(ti["date"].unique())
        untraded = all_days - traded
        if not untraded or not traded:
            continue

        # one row per date for the vol/width comparison
        per_day = (grp.groupby("date", observed=True)
                      .agg(vol=("realized_vol_14d", "first"),
                           width=("range_width_ticks", "median")))
        t_vol = per_day.loc[per_day.index.isin(traded), "vol"].dropna()
        u_vol = per_day.loc[per_day.index.isin(untraded), "vol"].dropna()
        t_w = per_day.loc[per_day.index.isin(traded), "width"].dropna()
        u_w = per_day.loc[per_day.index.isin(untraded), "width"].dropna()
        if len(t_vol) < 20 or len(u_vol) < 20:
            continue

        rows.append({
            "sym": sym, "session": sess,
            "n_traded": len(traded), "n_untraded": len(untraded),
            "cov_pct": round(100.0 * len(traded) / len(all_days), 1),
            "vol_traded": round(float(t_vol.median()), 4),
            "vol_untraded": round(float(u_vol.median()), 4),
            "vol_ratio": round(float(t_vol.median() / u_vol.median()), 3)
                         if u_vol.median() else np.nan,
            "width_traded": round(float(t_w.median()), 1),
            "width_untraded": round(float(u_w.median()), 1),
            "width_ratio": round(float(t_w.median() / u_w.median()), 3)
                           if u_w.median() else np.nan,
        })

    if rows:
        df = pd.DataFrame(rows).sort_values("width_ratio", ascending=False)
        print(df.to_string(index=False))
        wr = df["width_ratio"].median()
        vr = df["vol_ratio"].median()
        print(f"\n  median width ratio (traded/untraded) : {wr:.3f}")
        print(f"  median vol   ratio (traded/untraded) : {vr:.3f}")
    else:
        print("  insufficient overlap to compare")
        wr = vr = float("nan")

    hr("VERDICT — PER INSTRUMENT (a median hides the split)")
    print("An earlier version of this script judged on the median width ratio")
    print("and printed 'MINOR'. That was wrong: 8 of 10 instruments are clean")
    print("and washed out a severe confound in the other 2. Judged per")
    print("instrument instead.\n")

    if rows:
        per = pd.DataFrame(rows)
        agg = (per.groupby("sym", observed=True)
                  .agg(sessions=("session", "size"),
                       min_cov=("cov_pct", "min"),
                       med_cov=("cov_pct", "median"),
                       max_width_ratio=("width_ratio", "max"))
                  .reset_index())

        def _verdict(r) -> str:
            """
            The confound needs BOTH conditions, not either:
              (a) the null draws on days the variant skipped -> low coverage
              (b) those days are materially different        -> high width ratio

            An earlier version used OR, which flagged ZN as confounded on a
            width ratio of exactly 2.00 despite 92.4% coverage. ZN's absolute
            widths are 4 vs 2 TICKS -- the ratio is inflated by a minimum-quantum
            denominator, not a real population split, and at 92.4% coverage
            there are barely any skipped days for it to act on. Contrast BTC NY:
            69 vs 18 ticks at 27% coverage.
            """
            if r["min_cov"] >= 85:
                return "clean -- same day population"
            if r["min_cov"] < 60 and r["max_width_ratio"] >= 2.0:
                return "CONFOUNDED -- not a timing test"
            return "partial -- interpret with care"

        agg["verdict"] = agg.apply(_verdict, axis=1)
        print(agg.round(2).to_string(index=False))

        bad = agg[agg["verdict"].str.startswith("CONFOUNDED")]["sym"].tolist()
        print()
        if bad:
            print(f"  CONFOUNDED: {', '.join(bad)}")
            print("    These variants trade only on unusually wide-range days,")
            print("    while the matched null also draws on narrow-range days")
            print("    they never touched. null_p for these mixes DAY SELECTION")
            print("    with timing and cannot be read as a timing result.")
            print()
            print("    Likely direction of the bias: on a narrow-range day a")
            print("    stop sampled from the wide-day distribution is large")
            print("    relative to that day's movement, so neither SL nor TP is")
            print("    reached and the trade times out near 0 R. That pulls null")
            print("    gross TOWARD zero rather than negative, making the null")
            print("    HARDER to beat -- i.e. conservative for these symbols.")
            print("    Stated as a hypothesis, not measured.")
        else:
            print("  No instrument is confounded on this test.")
        print()
        clean = agg[agg["verdict"].str.startswith("clean")]["sym"].tolist()
        if clean:
            print(f"  clean: {', '.join(clean)}")
            print("    Coverage >= 85% with comparable range widths, so the")
            print("    matched null draws from effectively the same day set and")
            print("    null_p is a timing result for these.")

    med_cov = float(fam["coverage_pct"].median())
    print()
    print(f"  median coverage across all {len(fam)} families : {med_cov:.1f}%")
    print(f"  families under 60% coverage                : "
          f"{int((fam['coverage_pct'] < 60).sum())}")
    print()
    print("  A clean timing test for the confounded symbols would restrict the")
    print("  null pool to the dates the observed variant actually traded.")
    print()
    print("  Note: days_available is a lower bound -- a date absent from every")
    print("  variant never appears in the trade log at all, so true coverage")
    print("  is somewhat LOWER than reported here.")

    out = _OUT / "day_coverage_check.csv"
    fam.to_csv(out, index=False)
    print(f"\n  saved -> {out}")
    print("\n" + "=" * 104 + "\n")


if __name__ == "__main__":
    main()
