"""
check_null_rate.py — is the matched null CALIBRATED? (random sample, not top-N)

Why this is separate from recalibrate.py --limit
-----------------------------------------------
--limit N selects the top N variants by gross expectancy. Those are
pre-selected FOR high gross, so they would show low p-values under any
comparator. That sample answers "do the leaders survive?" but says nothing
about whether the null is fair.

Calibration is a statement about a RANDOM sample: under a fair comparator,
roughly 5% of arbitrarily chosen variants should land below p = 0.05. The
swing-derived null put 78% there (5,599 of 7,140), which is the symptom that
started this investigation.

So this script draws a STRATIFIED RANDOM sample across instruments and rr
levels and measures the significance rate under both comparators on identical
variants.

Runs on a reduced window count for speed -- absolute p-values will be noisier
than a full run, but the RATE comparison between comparators is the point, and
both designs see exactly the same session days.

Reads cached parquet only. No API calls, no spend.

Usage:
    python check_null_rate.py --n 120 --windows 3
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import (
    BOOTSTRAP_BLOCK_SIZE_DAYS, INSTRUMENTS, MIN_TRADES_FOR_RANKING,
    OUTPUTS_DIR, SESSIONS,
)
from src.data_layer import _compute_enrichment, ensure_daily, ensure_data
from src.null_calibrator import (
    NullPool, _bootstrap_null_means, _matched_entry_signal,
    _random_entry_signal, null_p_value,
)
from src.range_builder import build_session_days
from src.regime_sampler import filter_to_window, select_windows
from src.trade_sim import simulate_trade

logging.basicConfig(level=logging.WARNING, format="%(message)s")
_OUT = _ROOT / OUTPUTS_DIR
pd.set_option("display.width", 220)

FAMILY = ["instrument", "session", "range_minutes", "entry_mode",
          "closure_tf", "direction"]
DRAWS_PER_DAY = 3
N_BOOT = 400


def hr(t: str) -> None:
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="variants to sample")
    ap.add_argument("--windows", type=int, default=3, help="regime windows")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    t0 = time.perf_counter()

    sm = pd.read_parquet(_OUT / "summary.parquet")
    rank = sm[sm["trade_count"] >= MIN_TRADES_FOR_RANKING].copy()
    print(f"rankable variants: {len(rank):,}")

    # ── stratified random sample: spread across instrument x rr ────────────
    rng = np.random.default_rng(args.seed)
    per_cell = max(1, args.n // (rank["instrument"].nunique()
                                 * rank["rr"].nunique()))
    picks = []
    for _, g in rank.groupby(["instrument", "rr"], observed=True):
        k = min(per_cell, len(g))
        picks.append(g.sample(n=k, random_state=int(rng.integers(1 << 31))))
    sample = pd.concat(picks, ignore_index=True)
    if len(sample) > args.n:
        sample = sample.sample(n=args.n, random_state=args.seed)
    print(f"sampled {len(sample)} variants "
          f"({sample['instrument'].nunique()} instruments, "
          f"{sample['rr'].nunique()} rr levels)")
    print(f"NOTE: random sample, NOT top-by-gross. Under a fair null ~5% of")
    print(f"      these should fall below p=0.05.")

    # ── observed r_ticks per family, from the trade log ────────────────────
    tl = pd.read_parquet(_OUT / "trade_log.parquet",
                         columns=FAMILY + ["rr", "r_ticks", "exit_reason"])
    tl = tl[tl["exit_reason"] != "INVALID"]
    r_map = {}
    for keys, grp in tl.groupby(FAMILY, observed=True):
        r = grp["r_ticks"].to_numpy(dtype=float)
        r = r[np.isfinite(r) & (r >= 1.0)]
        if len(r):
            r_map[tuple(keys)] = r
    del tl
    print(f"r_ticks map: {len(r_map):,} families")

    # ── session days for the needed (sym, session) pairs only ─────────────
    need = sorted({(r["instrument"], r["session"]) for _, r in sample.iterrows()})
    syms = sorted({s for s, _ in need})
    data = {}
    for s in syms:
        data[s] = _compute_enrichment(ensure_data(s), ensure_daily(s),
                                      tick_size=INSTRUMENTS[s]["tick_size"])
    all_ts = pd.concat([d["timestamp"] for d in data.values()])
    windows = select_windows(all_ts.min().date(), all_ts.max().date())[:args.windows]

    sd_map = {}
    for sym, sess in need:
        acc = []
        for w in windows:
            dfw = filter_to_window(data[sym], w)
            if len(dfw):
                acc.extend(build_session_days(dfw, sym, sess))
        sd_map[(sym, sess)] = acc
    print(f"session-day pools built for {len(sd_map)} (instrument, session) pairs "
          f"over {len(windows)} windows")

    # ── evaluate both comparators on identical variants ────────────────────
    hr("EVALUATING — matched vs swing comparator, same variants")
    m_cache, s_cache = {}, {}
    rows = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        sym, sess = row["instrument"], row["session"]
        rm, rr = int(row["range_minutes"]), float(row["rr"])
        direction = row["direction"]
        fam = (sym, sess, rm, row["entry_mode"], int(row["closure_tf"]), direction)
        sds = sd_map.get((sym, sess), [])
        if not sds:
            continue
        r_pool = r_map.get(fam)
        n_obs = int(row["trade_count"])
        obs = float(row["expectancy_gross_r"])

        def _pool(kind):
            cache = m_cache if kind == "matched" else s_cache
            ck = fam + (rr, kind)
            if ck in cache:
                return cache[ck]
            g = np.random.default_rng(99)
            per_day = []
            for sd in sds:
                vals = []
                for _ in range(DRAWS_PER_DAY):
                    es = (_matched_entry_signal(sd, rm, direction, r_pool, g)
                          if kind == "matched" and r_pool is not None
                          else _random_entry_signal(sd, rm, direction, g))
                    if es is None:
                        continue
                    tr = simulate_trade(es, sd, rr_levels=[rr])
                    if tr and tr[0].exit_reason not in ("INVALID", None):
                        v = tr[0].gross_r
                        if v is not None and np.isfinite(v):
                            vals.append(float(v))
                if vals:
                    per_day.append(np.asarray(vals, dtype=float))
            p = NullPool(per_day=per_day, n_trades=sum(len(a) for a in per_day))
            cache[ck] = p
            return p

        out = {"instrument": sym, "session": sess, "entry_mode": row["entry_mode"],
               "rr": rr, "trade_count": n_obs, "gross": obs,
               "net": float(row["expectancy_net_r"]),
               "p_reported": float(row.get("null_p_value", np.nan))}
        for kind, col in (("matched", "p_matched"), ("swing", "p_swing")):
            if kind == "matched" and r_pool is None:
                out[col] = np.nan
                continue
            pool = _pool(kind)
            means = _bootstrap_null_means(pool, n_obs, N_BOOT,
                                          BOOTSTRAP_BLOCK_SIZE_DAYS)
            out[col] = null_p_value(obs, means) if len(means) else np.nan
            out[f"nullgross_{kind}"] = (float(np.mean(pool.flat()))
                                        if pool.n_trades else np.nan)
        rows.append(out)
        if i % 20 == 0:
            print(f"  {i}/{len(sample)} ...")

    if not rows:
        print("no variants evaluated")
        return
    df = pd.DataFrame(rows)

    hr("SIGNIFICANCE RATE — a fair null puts ~5% below 0.05")
    for col, label in (("p_reported", "as reported (swing, full 10 windows)"),
                       ("p_swing", f"swing recomputed ({len(windows)} windows)"),
                       ("p_matched", f"MATCHED ({len(windows)} windows)")):
        v = df[col].dropna()
        if v.empty:
            continue
        print(f"\n  {label}")
        print(f"    n={len(v):<4} p<0.05: {100.0*(v<0.05).mean():>5.1f}%   "
              f"p<0.10: {100.0*(v<0.10).mean():>5.1f}%   "
              f"median p: {v.median():.4f}")

    hr("NULL GROSS EXPECTANCY — a fair null sits near 0")
    for k in ("swing", "matched"):
        c = f"nullgross_{k}"
        if c in df.columns:
            v = df[c].dropna()
            if not v.empty:
                print(f"  {k:<8} median {v.median():+.4f}   "
                      f"mean {v.mean():+.4f}   n={len(v)}")

    hr("NET-POSITIVE AND SIGNIFICANT (matched comparator)")
    if "p_matched" in df.columns:
        both = df[(df["net"] > 0) & (df["p_matched"] < 0.05)]
        print(f"  net-positive in sample      : {int((df['net']>0).sum())}")
        print(f"  matched-significant         : "
              f"{int((df['p_matched']<0.05).sum())}")
        print(f"  BOTH                        : {len(both)}")
        if not both.empty:
            print()
            print(both.sort_values("net", ascending=False)
                      .head(15).round(4).to_string(index=False))

    out = _OUT / "null_rate_check.csv"
    df.to_csv(out, index=False)
    print(f"\n  saved -> {out}")
    print(f"  elapsed {time.perf_counter()-t0:.0f}s")
    print("\n" + "=" * 104 + "\n")


if __name__ == "__main__":
    main()
