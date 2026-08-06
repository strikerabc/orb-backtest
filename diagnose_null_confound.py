"""
diagnose_null_confound.py — is null_p confounded by STOP WIDTH?

The problem
-----------
The rebuilt null test reports null_p < 0.05 for 5,599 of 7,140 rankable
variants (78%). That is not credible as 78% genuine edges, so the comparator
is suspect.

Hypothesis
----------
_random_entry_signal places the entry at a UNIFORMLY RANDOM bar in the session,
then applies the same swing-SL machinery. A mid-session random bar sits far
from the opening-range boundary, so the most recent down-close cluster is
FURTHER AWAY -> wider r_ticks.

gross_r is normalised by r, and TP sits at rr * r. A wider r puts the TP
further away IN ABSOLUTE PRICE, so it is harder to reach before the 11:59
exit. Random entries would then lose systematically because their targets are
more distant -- not because their timing is worse.

If true, null_p measures "does this variant have a tighter stop than a random
entry?" rather than "does this variant's timing carry an edge". Tap-in (TI)
entries sit exactly AT the boundary, so they have the tightest stops of all --
which is precisely the family that dominates the ranked table.

Test
----
For several real variants, rebuild the null pool and compare:
    observed r_ticks   vs   null r_ticks
    observed TIME-exit rate vs null TIME-exit rate

Predictions if the confound is real:
    null r_ticks MEDIAN >> observed r_ticks median
    null TIME-exit rate >> observed (targets not reached before exit)
    the gap is LARGEST for TI variants and smallest for wide-stop instruments

Reads cached parquet + rebuilds session days locally. No API calls, no spend.

Usage:
    python diagnose_null_confound.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import INSTRUMENTS, OUTPUTS_DIR, RR_LEVELS, INVALID_REASONS
from src.data_layer import ensure_data, ensure_daily, _compute_enrichment
from src.range_builder import build_session_days
from src.regime_sampler import select_windows, filter_to_window
from src.null_calibrator import _random_entry_signal
from src.trade_sim import simulate_trade

_OUT = _ROOT / OUTPUTS_DIR
pd.set_option("display.width", 240)

# (instrument, session, range_minutes, entry_mode, direction, rr)
# TI variants that dominate the table, plus wide-stop controls.
CASES = [
    ("ZN",  "LDN", 15, "TI",   "short", 1.0),
    ("ZN",  "LDN",  5, "TI",   "long",  1.0),
    ("NQ",  "NY",  15, "R-CC", "short", 2.0),
    ("NQ",  "NY",  15, "CC",   "short", 2.0),
    ("ETH", "NY",  30, "R-II", "short", 2.0),
    ("ES",  "NY",  15, "CC",   "short", 2.0),
]

N_WINDOWS_SAMPLE = 4      # windows to pool session-days from
DRAWS_PER_DAY = 3


def hr(t: str) -> None:
    print("\n" + "=" * 112)
    print(t)
    print("=" * 112)


def main() -> None:
    tl_p = _OUT / "trade_log.parquet"
    if not tl_p.exists():
        print("missing outputs/trade_log.parquet")
        sys.exit(1)

    tl = pd.read_parquet(tl_p, columns=[
        "instrument", "session", "range_minutes", "entry_mode", "direction",
        "rr", "r_ticks", "gross_r", "exit_reason",
    ])
    tl = tl[~tl["exit_reason"].isin(INVALID_REASONS)]

    syms = sorted({c[0] for c in CASES})
    data: dict[str, pd.DataFrame] = {}
    for s in syms:
        df1 = ensure_data(s)
        df1d = ensure_daily(s)
        data[s] = _compute_enrichment(df1, df1d,
                                      tick_size=INSTRUMENTS[s]["tick_size"])

    all_ts = pd.concat([d["timestamp"] for d in data.values()])
    windows = select_windows(all_ts.min().date(), all_ts.max().date())

    hr("OBSERVED vs NULL — stop width and exit mix")
    print("Prediction if confounded: null r_ticks >> observed r_ticks,")
    print("and null TIME-exit rate >> observed.\n")
    print(f"  {'variant':<34} {'obs r':>7} {'null r':>7} {'r ratio':>8} "
          f"{'obs TIME%':>10} {'null TIME%':>11} {'obs gr':>8} {'null gr':>8}")
    print("  " + "-" * 104)

    rows = []
    for sym, sess, rm, mode, direction, rr in CASES:
        obs = tl[(tl["instrument"] == sym) & (tl["session"] == sess)
                 & (tl["range_minutes"] == rm) & (tl["entry_mode"] == mode)
                 & (tl["direction"] == direction) & (tl["rr"] == rr)]
        if obs.empty:
            print(f"  {sym}/{sess}/{rm}/{mode}/{direction}/{rr}  -- not found")
            continue

        obs_r = float(np.median(obs["r_ticks"]))
        obs_time = 100.0 * float((obs["exit_reason"] == "TIME").mean())
        obs_gr = float(obs["gross_r"].mean())

        # rebuild session days and draw the null the same way the calibrator does
        sds = []
        for w in windows[:N_WINDOWS_SAMPLE]:
            dfw = filter_to_window(data[sym], w)
            if len(dfw):
                sds.extend(build_session_days(dfw, sym, sess))

        rng = np.random.default_rng(99)
        n_r, n_gr, n_time = [], [], []
        for sd in sds:
            for _ in range(DRAWS_PER_DAY):
                es = _random_entry_signal(sd, rm, direction, rng)
                if es is None:
                    continue
                tr = simulate_trade(es, sd, rr_levels=[rr])
                if tr and tr[0].exit_reason not in INVALID_REASONS:
                    n_r.append(tr[0].r_ticks)
                    n_gr.append(tr[0].gross_r)
                    n_time.append(tr[0].exit_reason == "TIME")
        if not n_r:
            continue

        null_r = float(np.median(n_r))
        null_time = 100.0 * float(np.mean(n_time))
        null_gr = float(np.mean(n_gr))
        ratio = null_r / obs_r if obs_r > 0 else float("nan")

        label = f"{sym}/{sess}/{rm}m/{mode}/{direction}/rr{rr}"
        print(f"  {label:<34} {obs_r:>7.1f} {null_r:>7.1f} {ratio:>7.2f}x "
              f"{obs_time:>9.1f}% {null_time:>10.1f}% {obs_gr:>8.4f} "
              f"{null_gr:>8.4f}")
        rows.append({"variant": label, "obs_r": obs_r, "null_r": null_r,
                     "r_ratio": ratio, "obs_time_pct": obs_time,
                     "null_time_pct": null_time, "obs_gross": obs_gr,
                     "null_gross": null_gr, "n_null": len(n_r)})

    if not rows:
        print("\n  no cases resolved")
        return

    df = pd.DataFrame(rows)

    hr("VERDICT")
    med_ratio = float(df["r_ratio"].median())
    print(f"  median null/observed r_ticks ratio : {med_ratio:.2f}x")
    print(f"  cases where null r > observed r    : "
          f"{int((df['r_ratio'] > 1.1).sum())} of {len(df)}")
    print(f"  cases where null TIME% > obs TIME% : "
          f"{int((df['null_time_pct'] > df['obs_time_pct']).sum())} of {len(df)}")
    print()
    if med_ratio > 1.25:
        print("  CONFOUND CONFIRMED. Null entries carry systematically wider")
        print("  stops, so their targets sit further away in absolute price and")
        print("  are less likely to be reached before the 11:59 exit. null_p")
        print("  then partly measures STOP WIDTH, not entry timing, and cannot")
        print("  be read as evidence of edge on its own.")
        print()
        print("  A sound null must control for r_ticks -- e.g. resample null")
        print("  trades to match the observed r_ticks distribution, or compare")
        print("  only within r_ticks buckets.")
    elif med_ratio < 0.8:
        print("  INVERTED: null stops are TIGHTER than observed. The bias runs")
        print("  the other way and null_p is CONSERVATIVE for these variants.")
    else:
        print("  NO MATERIAL CONFOUND: null and observed stop widths are")
        print("  comparable, so null_p is not explained by stop width.")

    out = _OUT / "null_confound_check.csv"
    df.to_csv(out, index=False)
    print(f"\n  saved -> {out}")
    print("\n" + "=" * 112 + "\n")


if __name__ == "__main__":
    main()
