"""
analyse_entry_spreads.py — is the session-median spread the right cost, given
                           WHEN this strategy actually enters?

The problem with a session-wide median
--------------------------------------
measure_spreads.py reports the median spread across every minute of each
session window. But ORB entries are not uniformly distributed across that
window -- they cluster after the opening range completes, during price
discovery, when spreads are typically widest. ETH shows median 30t but p75 50t
and p95 70t, so the choice of statistic is worth real R.

What this does
--------------
1. Reads the trade log to get the EMPIRICAL distribution of entry times
   (bars_from_open_to_entry) per instrument x session.
2. Reads the cached bbo-1m quotes to compute median spread per
   minute-of-session.
3. Computes an ENTRY-WEIGHTED spread: each minute's spread weighted by the
   fraction of real entries that occurred in it.

Entry-weighted spread is the honest cost: it is what the strategy pays given
its own timing, rather than what an imaginary uniformly-timed trader pays.

Costs nothing -- both inputs are already on disk.

Caveat that survives this analysis: even entry-weighted spread excludes
MARKET IMPACT. A breakout entry crosses a book that is actively moving; the
quoted spread at that instant is a floor on what a real fill gives up.

Usage:
    python analyse_entry_spreads.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import DATA_DIR, INSTRUMENTS, OUTPUTS_DIR, SESSIONS, INVALID_REASONS
from src.data_layer import _roll_tag

_SPREADS = _ROOT / DATA_DIR / "spreads"
_OUT = _ROOT / OUTPUTS_DIR

pd.set_option("display.width", 200)

MAX_PLAUSIBLE_SPREAD_TICKS = 500.0


def hr(t: str) -> None:
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def load_quotes(sym: str) -> pd.DataFrame | None:
    """All cached normalised bbo-1m months for sym."""
    roll = _roll_tag(sym)
    files = sorted(_SPREADS.glob(f"{sym}_{roll}_*_bbo1m.parquet"))
    files = [f for f in files if "_RAW" not in f.name]
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tick = INSTRUMENTS[sym]["tick_size"]
    df["spread_ticks"] = (df["ask"] - df["bid"]) / tick
    ok = (df["spread_ticks"] > 0) & (df["spread_ticks"] <= MAX_PLAUSIBLE_SPREAD_TICKS)
    return df.loc[ok].copy()


def spread_by_minute(df: pd.DataFrame, sess_name: str) -> pd.Series:
    """Median spread indexed by minutes-since-session-open."""
    sess = SESSIONS[sess_name]
    loc = df["timestamp"].dt.tz_convert(sess["tz"])
    wall = loc.dt.hour * 60 + loc.dt.minute
    o = sess["open"][0] * 60 + sess["open"][1]
    x = sess["exit"][0] * 60 + sess["exit"][1]
    m = (wall >= o) & (wall < x)
    if not m.any():
        return pd.Series(dtype=float)
    sub = df.loc[m].copy()
    sub["min_since_open"] = (wall[m] - o).to_numpy()
    return sub.groupby("min_since_open")["spread_ticks"].median()


def main() -> None:
    tl_p = _OUT / "trade_log.parquet"
    if not tl_p.exists():
        print(f"missing {tl_p} -- run main.py first")
        sys.exit(1)

    tl = pd.read_parquet(tl_p, columns=[
        "instrument", "session", "bars_from_open_to_entry", "exit_reason",
    ])
    tl = tl[~tl["exit_reason"].isin(INVALID_REASONS)]
    print(f"loaded {len(tl):,} valid trades")
    print("NOTE: entry timing comes from the EXISTING trade log, which for")
    print("GC/ZN/6E/6J was built on the broken .c.0 data. Their timing")
    print("distribution may shift after the re-download; ES/NQ/RTY/CL/BTC/ETH")
    print("are unaffected by the roll fix and their weights are final.")

    hr("ENTRY TIMING — where in the session do entries actually land?")
    print(f"  {'sym':<5} {'sess':<5} {'n':>8} {'p10':>6} {'med':>6} {'p90':>6}"
          f"  {'first 30m':>10}")
    print("  " + "-" * 56)
    for (sym, sess), g in tl.groupby(["instrument", "session"], observed=True):
        b = g["bars_from_open_to_entry"].to_numpy()
        b = b[np.isfinite(b)]
        if len(b) == 0:
            continue
        pct30 = 100.0 * (b <= 30).mean()
        print(f"  {sym:<5} {sess:<5} {len(b):>8,} {np.percentile(b,10):>6.0f} "
              f"{np.median(b):>6.0f} {np.percentile(b,90):>6.0f}  {pct30:>9.1f}%")

    hr("ENTRY-WEIGHTED vs SESSION-MEDIAN SPREAD (ticks)")
    print("entry-weighted = sum over minutes of  spread(minute) x P(entry in minute)")
    print()
    print(f"  {'sym':<5} {'sess':<5} {'sess med':>9} {'entry-wtd':>10} "
          f"{'ratio':>7} {'p75':>6} {'p95':>6}  verdict")
    print("  " + "-" * 78)

    rows = []
    for sym in INSTRUMENTS:
        q = load_quotes(sym)
        if q is None or q.empty:
            continue
        sessions = INSTRUMENTS[sym].get("sessions", list(SESSIONS.keys()))
        for sess_name in sessions:
            by_min = spread_by_minute(q, sess_name)
            if by_min.empty:
                continue

            sub = tl[(tl["instrument"] == sym) & (tl["session"] == sess_name)]
            if sub.empty:
                continue
            b = sub["bars_from_open_to_entry"].to_numpy()
            b = b[np.isfinite(b)].astype(int)
            if len(b) == 0:
                continue

            # empirical P(entry in minute), aligned to available quote minutes
            counts = pd.Series(b).value_counts()
            common = by_min.index.intersection(counts.index)
            if len(common) == 0:
                continue
            w = counts.reindex(common).astype(float)
            w = w / w.sum()
            entry_wtd = float((by_min.reindex(common) * w).sum())

            sess_med = float(by_min.median())
            # full-window percentiles for context
            sess = SESSIONS[sess_name]
            loc = q["timestamp"].dt.tz_convert(sess["tz"])
            wall = loc.dt.hour * 60 + loc.dt.minute
            o = sess["open"][0] * 60 + sess["open"][1]
            x = sess["exit"][0] * 60 + sess["exit"][1]
            m = ((wall >= o) & (wall < x)).to_numpy()
            s_all = q["spread_ticks"].to_numpy()[m]
            p75 = float(np.percentile(s_all, 75))
            p95 = float(np.percentile(s_all, 95))

            ratio = entry_wtd / sess_med if sess_med > 0 else float("nan")
            if ratio >= 1.15:
                verdict = "entries pay MORE than session median"
            elif ratio <= 0.87:
                verdict = "entries pay LESS"
            else:
                verdict = "median is representative"

            print(f"  {sym:<5} {sess_name:<5} {sess_med:>9.2f} {entry_wtd:>10.2f} "
                  f"{ratio:>6.2f}x {p75:>6.2f} {p95:>6.2f}  {verdict}")
            rows.append({
                "sym": sym, "session": sess_name,
                "session_median": round(sess_med, 3),
                "entry_weighted": round(entry_wtd, 3),
                "ratio": round(ratio, 3),
                "p75": round(p75, 3), "p95": round(p95, 3),
                "n_trades": len(b),
            })

    if not rows:
        print("\n  no overlap between trade log and cached quotes")
        return

    df = pd.DataFrame(rows)

    hr("RECOMMENDED CONFIG — entry-weighted, FRACTIONAL (no rounding)")
    print("Fractional on purpose. An earlier version of this script used")
    print("ceil(), justified as 'you cannot cross a fraction of a tick'. That")
    print("is true for ONE trade and wrong for an EXPECTATION: if a spread is")
    print("2 ticks 80% of the time and 3 ticks 20%, the expected round-trip")
    print("cost over thousands of trades is exactly 2.2 ticks. ceil() turned")
    print("6E LDN's measured 1.001t into 2.0t -- a 100% overcharge -- and")
    print("BTC TOK's 4.20t into 5.0t.")
    print()
    print("The cost model already handles fractional ticks: comm_ticks is")
    print("2*commission/tick_value, which is 0.32 for ZN.")
    print()
    print("SLIPPAGE_TICKS_BY_SYMBOL_SESSION: dict[tuple[str, str], float] = {")
    for _, r in df.sort_values(["sym", "session"]).iterrows():
        v = max(1.0, float(r["entry_weighted"]))
        note = f"sess med {r['session_median']:.2f}"
        if abs(r["ratio"] - 1.0) >= 0.15:
            note += f", entries pay {r['ratio']:.2f}x that"
        print(f'    ("{r["sym"]}", "{r["session"]}"): {v:>6.2f},   # {note}')
    print("}")
    print()
    print("# Fallback: worst session per symbol (conservative).")
    print("SLIPPAGE_TICKS_BY_SYMBOL: dict[str, float] = {")
    for sym, g in df.groupby("sym", observed=True):
        v = max(1.0, float(g["entry_weighted"].max()))
        worst = g.loc[g["entry_weighted"].idxmax(), "session"]
        print(f'    "{sym}": {v:>6.2f},   # worst session: {worst}')
    print("}")

    hr("SUMMARY")
    worse = df[df["ratio"] >= 1.15]
    print(f"  symbol-sessions measured                  : {len(df)}")
    print(f"  where entries pay MORE than session median : {len(worse)}")
    if not worse.empty:
        print()
        for _, r in worse.sort_values("ratio", ascending=False).iterrows():
            print(f"    {r['sym']:<5} {r['session']:<4} "
                  f"{r['session_median']:.2f} -> {r['entry_weighted']:.2f} "
                  f"({r['ratio']:.2f}x)")
        print()
        print("  For these, a session-median slippage constant is optimistic.")
    else:
        print("\n  Session medians are representative of entry timing.")

    out = _SPREADS / "entry_weighted_spreads.csv"
    df.to_csv(out, index=False)
    print(f"\n  saved -> {out}")
    print("\n  Still excluded: MARKET IMPACT. These are quoted spreads at the")
    print("  entry minute, not realised fills on a book in motion.")
    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
