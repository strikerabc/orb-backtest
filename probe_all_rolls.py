"""
probe_all_rolls.py — which instruments are silently degraded by .c.0?

Method
------
Databento bills historical data by VOLUME DELIVERED. For an identical
(dataset, schema, symbol-root, window), get_cost() is therefore a proxy for
record count. Comparing roll suffixes on the same root exposes which one
actually carries the liquid contract:

    .c.0  calendar roll        (front month by expiry date)
    .v.0  volume roll          (most actively traded)
    .n.0  open-interest roll   (highest open interest)

Confirmed for gold (2024-06, ohlcv-1m):
    GC.c.0  $0.0023
    GC.v.0  $0.1003   <- 43.6x more data
    GC.n.0  $0.1003
which matches the ~27x sparsity measured against CL in the cached bars.

Open question this script answers
--------------------------------
6E has 934,954 bars and 6J has 874,243 -- roughly HALF what a ~23h future
should produce (~1,380/day x ~1,300 sessions ~ 1.8M). That was previously
attributed to FX futures having untraded minutes. If .c.0 is degrading them
too, the sweep must not be re-run until they are re-downloaded.

get_cost() and symbology are METADATA calls. This script downloads nothing
and spends no credits.

Usage:
    $env:DATABENTO_API_KEY = "..."
    python probe_all_rolls.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import DATASET, DATA_DIR, INSTRUMENTS, SCHEMA_1M, STYPE

_DATA = _ROOT / DATA_DIR

# A quiet, fully-historical month that every instrument existed in
# (ETH launched 2021-02, so 2024 is safe for all ten).
PROBE_START = "2024-06-01"
PROBE_END   = "2024-07-01"

SUFFIXES = ["c", "v", "n"]

# Ratio above which .c.0 is considered materially degraded.
SUSPECT_RATIO = 1.5


def hr(t: str) -> None:
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def cached_bars(sym: str) -> int:
    p = _DATA / f"{sym}_1m.parquet"
    if not p.exists():
        return -1
    try:
        return len(pd.read_parquet(p, columns=["timestamp"]))
    except Exception:
        return -1


def main() -> None:
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY not set")
        print('  $env:DATABENTO_API_KEY = "db-..."')
        sys.exit(1)
    try:
        import databento as db
    except ImportError:
        print("ERROR: pip install databento")
        sys.exit(1)

    client = db.Historical(key=api_key)

    hr(f"ROLL COST COMPARISON — {PROBE_START} to {PROBE_END}, {SCHEMA_1M}")
    print("get_cost() is a volume proxy. Higher cost = more records delivered.")
    print("A large v/c ratio means .c.0 is tracking a thin contract.\n")
    print(f"  {'sym':<6} {'root':<7} {'.c.0':>10} {'.v.0':>10} {'.n.0':>10} "
          f"{'v/c':>8}  {'cached bars':>12}  verdict")
    print("  " + "-" * 92)

    rows = []
    for sym, instr in INSTRUMENTS.items():
        cont = instr["continuous_symbol"]        # e.g. "GC.c.0"
        root = cont.split(".")[0]

        costs: dict[str, float | None] = {}
        for suf in SUFFIXES:
            candidate = f"{root}.{suf}.0"
            try:
                costs[suf] = float(client.metadata.get_cost(
                    dataset=DATASET, symbols=[candidate], schema=SCHEMA_1M,
                    start=PROBE_START, end=PROBE_END, stype_in=STYPE,
                ))
            except Exception:
                costs[suf] = None

        c, v, n = costs.get("c"), costs.get("v"), costs.get("n")
        ratio = (v / c) if (c and v and c > 0) else None
        bars = cached_bars(sym)

        if ratio is None:
            verdict = "n/a"
        elif ratio >= SUSPECT_RATIO:
            verdict = f"DEGRADED -> use .{('v' if (v or 0) >= (n or 0) else 'n')}.0"
        else:
            verdict = "ok"

        def f(x: float | None) -> str:
            return "err" if x is None else f"${x:.4f}"

        print(f"  {sym:<6} {root:<7} {f(c):>10} {f(v):>10} {f(n):>10} "
              f"{('-' if ratio is None else f'{ratio:.1f}x'):>8}  "
              f"{bars:>12,}  {verdict}")

        rows.append({"sym": sym, "root": root, "cost_c": c, "cost_v": v,
                     "cost_n": n, "ratio": ratio, "cached_bars": bars,
                     "verdict": verdict})

    df = pd.DataFrame(rows)

    hr("SUMMARY")
    deg = df[df["ratio"].notna() & (df["ratio"] >= SUSPECT_RATIO)]
    ok = df[df["ratio"].notna() & (df["ratio"] < SUSPECT_RATIO)]

    print(f"  instruments probed        : {len(df)}")
    print(f"  degraded on .c.0          : {len(deg)}")
    print(f"  fine on .c.0              : {len(ok)}")

    if not deg.empty:
        print("\n  NEEDS RE-DOWNLOAD:")
        for _, r in deg.iterrows():
            better = "v" if (r["cost_v"] or 0) >= (r["cost_n"] or 0) else "n"
            print(f"    {r['sym']:<6} {r['ratio']:>6.1f}x more data on "
                  f".{better}.0   (cached {int(r['cached_bars']):,} bars)")

        # scale the probe month up to a full re-download estimate
        hr("RE-DOWNLOAD COST ESTIMATE (scaled from one probe month)")
        print("  Approximate: probe-month cost x months in each window.")
        print("  Verify exactly with cost_check.py after switching symbols.\n")
        total = 0.0
        for _, r in deg.iterrows():
            sym = r["sym"]
            start = pd.Timestamp(INSTRUMENTS[sym].get("data_start", "2019-01-01"))
            months = max(1, int((pd.Timestamp("2026-08-01") - start).days / 30.44))
            better_cost = max(r["cost_v"] or 0.0, r["cost_n"] or 0.0)
            est = better_cost * months
            total += est
            print(f"    {sym:<6} {months:>3} months x ${better_cost:.4f} "
                  f"= ${est:>7.2f}")
        print(f"\n    {'TOTAL':<6} {'':>12} ${total:>10.2f}")

    if not ok.empty:
        print("\n  Confirmed fine on .c.0: " +
              ", ".join(ok["sym"].tolist()))

    print("\n  Note: ES/NQ merge local Dukascopy data for 2023-2026, so their")
    print("  cached bar counts are not comparable to a pure .c.0 download.")
    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
