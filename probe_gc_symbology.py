"""
probe_gc_symbology.py — what raw contract does GC.c.0 actually track?

Read-only probe. Resolves continuous symbols to raw contracts via the
symbology endpoint (metadata, no timeseries credits), then prints cost
estimates for candidate re-downloads. Downloads NOTHING.

Databento continuous symbology suffixes:
    .c.N  calendar roll        (front month by expiry date)
    .v.N  volume roll          (most actively traded contract)
    .n.N  open-interest roll

COMEX gold liquidity sits in Feb/Apr/Jun/Aug/Oct/Dec (G/J/M/Q/V/Z).
Serial months (F/H/K/N/U/X) trade thinly. If GC.c.0 resolves to serial
months, that explains ~3% minute coverage and .v.0 is the correct symbol.

Usage:
    $env:DATABENTO_API_KEY = "..."
    python probe_gc_symbology.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import DATASET, SCHEMA_1M, STYPE, INSTRUMENTS

MONTH_CODE = {
    "F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
    "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec",
}
GOLD_LIQUID = set("GJMQVZ")

# Probe dates spread across the year to catch roll behaviour in several months.
PROBE_DATES = ["2024-01-16", "2024-03-14", "2024-05-15", "2024-07-16",
               "2024-09-16", "2024-11-14"]

CANDIDATES = ["GC.c.0", "GC.v.0", "GC.n.0"]


def hr(t: str) -> None:
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


def _month_tag(raw: str) -> str:
    """Extract month code from a raw CME symbol like GCZ4 / GCG25."""
    core = raw.rstrip("0123456789")
    if not core:
        return "?"
    code = core[-1]
    month = MONTH_CODE.get(code, "?")
    liquid = "LIQUID" if code in GOLD_LIQUID else "serial/thin"
    return f"{month:<4} {liquid}"


def resolve(client, symbol: str, date: str) -> list[str]:
    """Resolve one continuous symbol on one date to raw contract(s)."""
    try:
        res = client.symbology.resolve(
            dataset=DATASET,
            symbols=[symbol],
            stype_in=STYPE,
            stype_out="raw_symbol",
            start_date=date,
            end_date=date,
        )
    except Exception as exc:
        return [f"ERROR: {exc}"]

    out: list[str] = []
    mappings = res.get("result", res) if isinstance(res, dict) else res
    if isinstance(mappings, dict):
        for _sym, entries in mappings.items():
            if isinstance(entries, list):
                for e in entries:
                    val = e.get("s") if isinstance(e, dict) else str(e)
                    if val:
                        out.append(str(val))
            else:
                out.append(str(entries))
    return out or ["(no mapping)"]


def main() -> None:
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY not set  ($env:DATABENTO_API_KEY = '...')")
        sys.exit(1)
    try:
        import databento as db
    except ImportError:
        print("ERROR: pip install databento")
        sys.exit(1)

    client = db.Historical(key=api_key)

    hr("A. SYMBOLOGY RESOLUTION — which raw contract does each suffix track?")
    print(f"  {'symbol':<10} {'date':<12} {'raw':<12} month / liquidity")
    print("  " + "-" * 62)
    for cand in CANDIDATES:
        for d in PROBE_DATES:
            for raw in resolve(client, cand, d):
                tag = _month_tag(raw) if not raw.startswith(("ERROR", "(")) else raw
                print(f"  {cand:<10} {d:<12} {raw:<12} {tag}")
        print()

    hr("B. COST ESTIMATE — one probe month of 1m data per candidate")
    for cand in CANDIDATES:
        try:
            c = client.metadata.get_cost(
                dataset=DATASET, symbols=[cand], schema=SCHEMA_1M,
                start="2024-06-01", end="2024-07-01", stype_in=STYPE,
            )
            print(f"  {cand:<10} 2024-06  ${c:.4f}")
        except Exception as exc:
            print(f"  {cand:<10} ERROR: {exc}")

    hr("C. COST ESTIMATE — full GC re-download window")
    start = INSTRUMENTS["GC"].get("data_start", "2019-01-01")
    end = pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")
    for cand in CANDIDATES:
        try:
            c = client.metadata.get_cost(
                dataset=DATASET, symbols=[cand], schema=SCHEMA_1M,
                start=start, end=end, stype_in=STYPE,
            )
            print(f"  {cand:<10} {start} -> {end}  ${c:.2f}")
        except Exception as exc:
            print(f"  {cand:<10} ERROR: {exc}")

    print("\n" + "=" * 92)
    print("Nothing downloaded. If .v.0 resolves to LIQUID months while .c.0")
    print("resolves to serial/thin, switch GC to GC.v.0 and rebuild its cache.")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()
