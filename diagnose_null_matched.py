"""
diagnose_null_matched.py — verify WHY the null loses, and prototype a fix.

Where this came from
--------------------
null_p < 0.05 for 5,599 of 7,140 rankable variants (78%), which is not
credible as 78% genuine edges.

First hypothesis (MINE, WRONG): null entries sit mid-session, so their swing
stops are FURTHER away, making targets harder to reach.
Measured: null stops are TIGHTER, not wider -- median ratio 0.27x.
    NQ/NY/15m/CC/short   obs r=173  null r=12   (0.07x)
    ES/NY/15m/CC/short   obs r=34   null r=4    (0.12x)

Second hypothesis (this script): the tightness itself handicaps the null via
an interaction with the exit rules, not via timing.

    A 4-tick stop on ES is INSIDE a typical 1-minute bar's range. So:
      - the null is stopped out almost immediately (TIME-exit falls from
        26.0% observed to 4.8% null -- nearly everything resolves), and
      - SL and TP frequently land in the SAME BAR, where trade_sim resolves
        SL FIRST by design (the deliberate conservative tiebreak).

    If that is what drives null gross negative, then null_p is
    ANTI-conservative -- too easy to beat -- and the 78% significance rate is
    an artefact of the tiebreak rule, not evidence of edge.

Decisive measurements
---------------------
  1. same_bar_ambiguous rate: observed vs swing-null. If the null rate is far
     higher, the tiebreak explanation holds.
  2. A MATCHED null: random entry timing, but stop distance sampled from the
     OBSERVED variant's own r_ticks distribution. Risk geometry held fixed, so
     only TIMING differs -- which is what the null was always meant to test.
  3. null_p under swing-null vs matched-null, same variant.

Reads cached parquet and rebuilds session days locally. No API calls, no spend.

Usage:
    python diagnose_null_matched.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import INSTRUMENTS, OUTPUTS_DIR, SESSIONS, INVALID_REASONS
from src.data_layer import ensure_data, ensure_daily, _compute_enrichment
from src.entry_detector import EntrySignal
from src.null_calibrator import (
    NullPool, _bootstrap_null_means, _random_entry_signal, null_p_value,
)
from src.range_builder import build_session_days
from src.regime_sampler import filter_to_window, select_windows
from src.trade_sim import simulate_trade

_OUT = _ROOT / OUTPUTS_DIR
pd.set_option("display.width", 240)

CASES = [
    ("ZN",  "LDN", 15, "TI",   "short", 1.0),
    ("NQ",  "NY",  15, "R-CC", "short", 2.0),
    ("NQ",  "NY",  15, "CC",   "short", 2.0),
    ("ETH", "NY",  30, "R-II", "short", 2.0),
    ("ES",  "NY",  15, "CC",   "short", 2.0),
    ("GC",  "LDN", 15, "TI",   "long",  1.0),
]

N_WINDOWS = 4
DRAWS_PER_DAY = 3
N_BOOT = 500


def hr(t: str) -> None:
    print("\n" + "=" * 116)
    print(t)
    print("=" * 116)


def _matched_entry_signal(sd, rm, direction, rr, r_ticks_pool, rng):
    """
    Random entry TIMING, but stop distance drawn from the observed variant's
    own r_ticks distribution instead of swing-derived.

    This holds risk geometry fixed so the comparison isolates timing, which is
    the question the null was meant to answer.
    """
    if rm not in sd.range_highs:
        return None
    n = len(sd.bars_o)
    if n == 0 or len(r_ticks_pool) == 0:
        return None

    sess = SESSIONS[sd.session]
    open_min = sess["open"][0] * 60 + sess["open"][1]
    range_end = int(np.searchsorted(sd.bar_wall_mins, open_min + rm))
    if range_end >= n:
        return None

    idx = int(rng.integers(range_end, n))
    fill = float(sd.bars_o[idx])
    r_t = float(rng.choice(r_ticks_pool))
    if r_t < 1:
        return None
    dist = r_t * sd.tick_size
    is_long = direction == "long"
    sl = fill - dist if is_long else fill + dist
    boundary = sd.range_highs[rm] if is_long else sd.range_lows[rm]

    return EntrySignal(
        mode="NULL-MATCHED", closure_tf=1, range_minutes=rm,
        direction=direction, entry_bar_idx=idx, fill_price=fill,
        breakout_bar_idx=idx, tap_in_bar_idx=None, boundary=boundary,
        sl_price=sl, sl_bars_back=0, sl_source="matched", gap_fill=False,
    )


def _run_null(sds, rm, direction, rr, kind, r_pool, seed=99):
    """Return (per_day list, flat stats dict) for one null variety."""
    rng = np.random.default_rng(seed)
    per_day, amb, times, rts, grs = [], [], [], [], []
    for sd in sds:
        vals = []
        for _ in range(DRAWS_PER_DAY):
            if kind == "swing":
                es = _random_entry_signal(sd, rm, direction, rng)
            else:
                es = _matched_entry_signal(sd, rm, direction, rr, r_pool, rng)
            if es is None:
                continue
            tr = simulate_trade(es, sd, rr_levels=[rr])
            if tr and tr[0].exit_reason not in INVALID_REASONS:
                t = tr[0]
                vals.append(t.gross_r)
                amb.append(t.same_bar_ambiguous)
                times.append(t.exit_reason == "TIME")
                rts.append(t.r_ticks)
                grs.append(t.gross_r)
        if vals:
            per_day.append(np.asarray(vals, dtype=float))
    stats = {
        "n": len(grs),
        "gross": float(np.mean(grs)) if grs else float("nan"),
        "amb_pct": 100.0 * float(np.mean(amb)) if amb else float("nan"),
        "time_pct": 100.0 * float(np.mean(times)) if times else float("nan"),
        "med_r": float(np.median(rts)) if rts else float("nan"),
    }
    return per_day, stats


def main() -> None:
    tl_p = _OUT / "trade_log.parquet"
    if not tl_p.exists():
        print("missing outputs/trade_log.parquet")
        sys.exit(1)

    tl = pd.read_parquet(tl_p, columns=[
        "instrument", "session", "range_minutes", "entry_mode", "direction",
        "rr", "r_ticks", "gross_r", "exit_reason", "same_bar_ambiguous",
    ])
    tl = tl[~tl["exit_reason"].isin(INVALID_REASONS)]

    syms = sorted({c[0] for c in CASES})
    data = {}
    for s in syms:
        data[s] = _compute_enrichment(ensure_data(s), ensure_daily(s),
                                      tick_size=INSTRUMENTS[s]["tick_size"])
    all_ts = pd.concat([d["timestamp"] for d in data.values()])
    windows = select_windows(all_ts.min().date(), all_ts.max().date())

    hr("1. SAME-BAR AMBIGUITY — is the SL-first tiebreak handicapping the null?")
    print("trade_sim resolves SL first when SL and TP fall in one bar. Tight")
    print("stops make that far more common.\n")
    print(f"  {'variant':<32} {'obs amb%':>9} {'null amb%':>10} {'ratio':>7} "
          f"{'obs r':>7} {'null r':>7}")
    print("  " + "-" * 78)

    results = []
    for sym, sess, rm, mode, direction, rr in CASES:
        obs = tl[(tl["instrument"] == sym) & (tl["session"] == sess)
                 & (tl["range_minutes"] == rm) & (tl["entry_mode"] == mode)
                 & (tl["direction"] == direction) & (tl["rr"] == rr)]
        if obs.empty:
            continue
        obs_stats = {
            "gross": float(obs["gross_r"].mean()),
            "amb_pct": 100.0 * float(obs["same_bar_ambiguous"].mean()),
            "time_pct": 100.0 * float((obs["exit_reason"] == "TIME").mean()),
            "med_r": float(np.median(obs["r_ticks"])),
            "n": len(obs),
        }
        r_pool = obs["r_ticks"].to_numpy(dtype=float)
        r_pool = r_pool[np.isfinite(r_pool) & (r_pool >= 1)]

        sds = []
        for w in windows[:N_WINDOWS]:
            dfw = filter_to_window(data[sym], w)
            if len(dfw):
                sds.extend(build_session_days(dfw, sym, sess))

        sw_days, sw = _run_null(sds, rm, direction, rr, "swing", r_pool)
        mt_days, mt = _run_null(sds, rm, direction, rr, "matched", r_pool)

        label = f"{sym}/{sess}/{rm}m/{mode}/{direction}/rr{rr}"
        ratio = sw["amb_pct"] / obs_stats["amb_pct"] if obs_stats["amb_pct"] > 0 else float("inf")
        print(f"  {label:<32} {obs_stats['amb_pct']:>8.2f}% {sw['amb_pct']:>9.2f}% "
              f"{ratio:>6.1f}x {obs_stats['med_r']:>7.1f} {sw['med_r']:>7.1f}")

        # null_p under each design
        p_sw = p_mt = float("nan")
        if sw_days:
            pool = NullPool(per_day=sw_days, n_trades=sum(len(a) for a in sw_days))
            means = _bootstrap_null_means(pool, obs_stats["n"], N_BOOT, 5)
            p_sw = null_p_value(obs_stats["gross"], means)
        if mt_days:
            pool = NullPool(per_day=mt_days, n_trades=sum(len(a) for a in mt_days))
            means = _bootstrap_null_means(pool, obs_stats["n"], N_BOOT, 5)
            p_mt = null_p_value(obs_stats["gross"], means)

        results.append({"variant": label, "obs_gross": obs_stats["gross"],
                        "obs_amb": obs_stats["amb_pct"], "obs_r": obs_stats["med_r"],
                        "obs_time": obs_stats["time_pct"],
                        "swing_gross": sw["gross"], "swing_amb": sw["amb_pct"],
                        "swing_r": sw["med_r"], "swing_time": sw["time_pct"],
                        "matched_gross": mt["gross"], "matched_amb": mt["amb_pct"],
                        "matched_r": mt["med_r"], "matched_time": mt["time_pct"],
                        "p_swing": p_sw, "p_matched": p_mt})

    if not results:
        print("  no cases resolved")
        return
    df = pd.DataFrame(results)

    hr("2. MATCHED NULL — random timing, observed stop distance")
    print("Risk geometry held fixed, so only entry TIMING differs.\n")
    print(f"  {'variant':<32} {'obs gr':>8} {'swing gr':>9} {'match gr':>9} "
          f"{'p_swing':>8} {'p_match':>8}")
    print("  " + "-" * 78)
    for _, r in df.iterrows():
        print(f"  {r['variant']:<32} {r['obs_gross']:>8.4f} "
              f"{r['swing_gross']:>9.4f} {r['matched_gross']:>9.4f} "
              f"{r['p_swing']:>8.4f} {r['p_matched']:>8.4f}")

    hr("3. VERDICT")
    amb_infl = (df["swing_amb"] / df["obs_amb"].replace(0, np.nan)).median()
    n_flip = int((df["p_matched"] >= 0.05).sum())
    print(f"  median same-bar ambiguity inflation (swing null / obs) : {amb_infl:.1f}x")
    print(f"  median null gross, swing-derived stops                 : "
          f"{df['swing_gross'].median():+.4f}")
    print(f"  median null gross, matched stops                       : "
          f"{df['matched_gross'].median():+.4f}")
    print(f"  cases losing significance under matched null           : "
          f"{n_flip} of {len(df)}")
    print()
    if amb_infl > 2.0 and df["matched_gross"].median() > df["swing_gross"].median():
        print("  MECHANISM CONFIRMED. The swing-derived null draws stops far")
        print("  tighter than the observed strategy, which inflates same-bar")
        print("  SL/TP collisions. trade_sim resolves those as SL by design, so")
        print("  the null is penalised by the tiebreak rule rather than by worse")
        print("  timing. null_p from that comparator is ANTI-conservative and")
        print("  the 78% significance rate is largely an artefact.")
        print()
        print("  The matched null holds stop distance fixed and is the")
        print("  defensible comparator. Significance should be re-derived from")
        print("  it before any variant is called an edge.")
    else:
        print("  Mechanism NOT confirmed on these cases. The tightness gap does")
        print("  not translate into an ambiguity-driven penalty, so the 78%")
        print("  significance rate needs a different explanation.")

    out = _OUT / "null_matched_check.csv"
    df.to_csv(out, index=False)
    print(f"\n  saved -> {out}")
    print("\n" + "=" * 116 + "\n")


if __name__ == "__main__":
    main()
