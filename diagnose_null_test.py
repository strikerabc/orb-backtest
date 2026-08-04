"""
diagnose_null_test.py — is null_p_value a valid significance test?

The claim under test
-------------------
null_calibrator.null_p_value() computes:

    mean(null_distribution >= observed_expectancy)

where null_distribution holds ONE gross_r per session-day, i.e. a pool of
INDIVIDUAL trade outcomes. observed_expectancy is a MEAN. Comparing a mean
against a pool of single draws is a category error, and it imposes a floor:

    at rr=2.0 an individual random trade returns ~+2.0 (TP) or ~-1.0 (SL).
    If a fraction h of random entries hit TP, then for any observed
    expectancy below 2.0 the statistic returns ~h -- the random-entry hit
    rate, not a tail probability.

Prediction if the claim is true
-------------------------------
  (a) null_p_value should track TP hit rate, so it should DECREASE as rr
      rises (higher rr -> lower hit rate), independent of any real edge.
  (b) The same broken formula applied to a variant's OWN trades (comparing
      its mean against its own individual trades) should yield values in the
      same 0.15-0.50 band -- even though a variant is, by construction,
      exactly as good as itself. p should be ~0.5 for a valid test of "no
      difference"; a floor well below that with rr-dependence proves the
      statistic is measuring hit rate.
  (c) That self-test p should correlate strongly with win_rate.

This script needs no Databento access and rebuilds no null pools.

Usage:
    python diagnose_null_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import MIN_TRADES_FOR_RANKING, OUTPUTS_DIR

_OUT = _ROOT / OUTPUTS_DIR
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

VARIANT_KEYS = ["instrument", "session", "range_minutes",
                "entry_mode", "closure_tf", "direction", "rr"]


def hr(t: str) -> None:
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def main() -> None:
    sm_p = _OUT / "summary.parquet"
    tl_p = _OUT / "trade_log.parquet"
    if not (sm_p.exists() and tl_p.exists()):
        print("missing outputs/summary.parquet or trade_log.parquet")
        sys.exit(1)

    sm = pd.read_parquet(sm_p)
    rank = sm[sm["trade_count"] >= MIN_TRADES_FOR_RANKING].copy()

    # ── (a) does null_p track rr rather than performance? ──────────────────
    hr("(a) null_p_value BY RR  — a hit-rate statistic falls as rr rises")
    if "null_p_value" in rank.columns:
        g = (rank.groupby("rr", observed=True)["null_p_value"]
                 .agg(variants="size", mean="mean", median="median",
                      min="min", max="max")
                 .reset_index())
        print(g.round(4).to_string(index=False))
        print("\nexpected TP hit rate for a symmetric random walk ~ 1/(1+rr):")
        for rr in sorted(rank["rr"].unique()):
            sub = rank[rank["rr"] == rr]["null_p_value"]
            print(f"  rr={rr:<5} 1/(1+rr)={1.0/(1.0+rr):.4f}   "
                  f"observed mean null_p={sub.mean():.4f}")

    # ── (b)+(c) apply the same formula to each variant's OWN trades ────────
    hr("(b) SELF-CONSISTENCY TEST — same formula, variant vs its own trades")
    print("A valid test of 'is X different from X' returns p ~ 0.5.")
    print("Loading trade log (gross_r only)...\n")

    tl = pd.read_parquet(tl_p, columns=VARIANT_KEYS + ["gross_r", "exit_reason"])
    tl = tl[tl["exit_reason"] != "INVALID"]

    def self_p(s: pd.Series) -> float:
        # identical formula to null_p_value(), but the "null pool" is the
        # variant's own individual trades and the observed value is its mean
        arr = s.to_numpy()
        return float(np.mean(arr >= arr.mean()))

    sp = (tl.groupby(VARIANT_KEYS, observed=True)["gross_r"]
            .agg(trade_count="size", exp_gross="mean", self_p=self_p)
            .reset_index())
    sp = sp[sp["trade_count"] >= MIN_TRADES_FOR_RANKING]

    print(f"variants tested: {len(sp):,}")
    print(f"  self_p mean   : {sp['self_p'].mean():.4f}   (valid test -> ~0.5)")
    print(f"  self_p median : {sp['self_p'].median():.4f}")
    print(f"  self_p min    : {sp['self_p'].min():.4f}")
    print(f"  self_p max    : {sp['self_p'].max():.4f}")
    print(f"  fraction < 0.05: {100.0*(sp['self_p'] < 0.05).mean():.2f}%  "
          f"(a variant can never beat itself, yet a valid test would give ~5%)")

    print("\nself_p by rr — tracks hit rate, not merit:")
    g2 = (sp.groupby("rr", observed=True)["self_p"]
            .agg(variants="size", mean="mean", median="median")
            .reset_index())
    print(g2.round(4).to_string(index=False))

    # ── (c) correlation with win_rate ──────────────────────────────────────
    hr("(c) CORRELATION — does the statistic just measure win rate?")
    merged = sp.merge(
        rank[VARIANT_KEYS + ["win_rate", "null_p_value", "expectancy_gross_r"]],
        on=VARIANT_KEYS, how="inner")
    if not merged.empty:
        print(f"merged variants: {len(merged):,}\n")
        pairs = [
            ("self_p", "win_rate"),
            ("null_p_value", "win_rate"),
            ("null_p_value", "rr"),
            ("null_p_value", "expectancy_gross_r"),
        ]
        for a, b in pairs:
            if a in merged.columns and b in merged.columns:
                sub = merged[[a, b]].dropna()
                if len(sub) > 2:
                    r = float(np.corrcoef(sub[a], sub[b])[0, 1])
                    print(f"  corr({a:<18}, {b:<20}) = {r:+.4f}")

        print("\nIf null_p correlates strongly with win_rate/rr but weakly with")
        print("expectancy, it is measuring hit rate and cannot detect edge.")

    hr("VERDICT")
    if "null_p_value" in rank.columns:
        floor = rank["null_p_value"].min()
        print(f"  observed minimum null_p_value across {len(rank):,} variants: {floor:.4f}")
        print(f"  variants below 0.05: {int((rank['null_p_value'] < 0.05).sum())}")
        print()
        print("  The earlier reading -- '0 significant out of 6,168, worse than")
        print("  the ~308 expected by chance, therefore no edge' -- is NOT")
        print("  supported. The statistic has a floor near the random-entry hit")
        print("  rate, so p < 0.05 is unreachable by construction. The test")
        print("  provides no evidence either way and must be rebuilt as a")
        print("  distribution of MEANS before any edge claim is made.")

    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
