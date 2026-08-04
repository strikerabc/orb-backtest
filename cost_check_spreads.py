"""
cost_check_spreads.py — consolidated budget: ohlcv-1m re-download PLUS the
                        spread-measurement spend needed to replace the
                        estimated slippage constants with measured ones.

Why this exists
---------------
SLIPPAGE_TICKS_BY_SYMBOL in config.py is currently ESTIMATED (ETH 10 ticks,
BTC 2, rest 1). ohlcv-1m carries no bid/ask, so those numbers are assumptions.
They matter: ETH dominated the old net-positive variant list largely because a
flat 1-tick assumption undercharged it by roughly an order of magnitude.

Schema options for measuring real spreads, cheapest first
---------------------------------------------------------
  bbo-1m   1-minute BBO snapshot. ~1,380 records/day. Gives a median spread
           per minute-of-day -- ample for a per-instrument constant, and for
           per-session constants at the three opens.
  tbbo     BBO stamped at every TRADE. One record per trade, so volume scales
           with activity, not with clock time. Better for "spread actually
           paid when a fill occurs", which is closer to what we model.
  mbp-1    Every top-of-book CHANGE. Largest by far. Needed only for queue
           dynamics / depth, which this strategy does not model.

Sampling, not full history
--------------------------
A slippage constant does not need 7.5 years. One month of bbo-1m is ~29,000
snapshots per instrument; a median from that is stable well past the precision
we need. Spread does vary with regime (2020 and 2022 were wider), so this
prices 1-month, 3-month and full-history windows and samples three separate
calendar years to expose regime sensitivity.

Everything here is metadata only. Downloads nothing, spends nothing.

Usage:
    $env:DATABENTO_API_KEY = "db-..."
    python cost_check_spreads.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import (
    DATASET, DOWNLOAD_END, DOWNLOAD_START, INSTRUMENTS,
    SCHEMA_1M, SLIPPAGE_TICKS_BY_SYMBOL, STYPE,
)
from src.data_layer import _cache_path, _roll_tag, safe_end_date

# Schemas to price, cheapest expected first
SPREAD_SCHEMAS = ["bbo-1m", "tbbo", "mbp-1"]

# The two instruments whose slippage estimate is doing the most work
FOCUS = ["ETH", "BTC"]

# Windows: one quiet month, one quarter, and three separate years for regime
WINDOWS = {
    "1 month (2024-06)":  ("2024-06-01", "2024-07-01"),
    "3 months (2024 Q2)": ("2024-04-01", "2024-07-01"),
    "1 month (2022-06)":  ("2022-06-01", "2022-07-01"),
    "1 month (2020-06)":  ("2020-06-01", "2020-07-01"),
}

MAX_RETRIES = 4
BASE_SLEEP = 0.6


def hr(t: str) -> None:
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def cost(client, symbol: str, schema: str, start: str, end: str):
    """get_cost with retry/backoff. Returns (value, error_first_line)."""
    last = ""
    for attempt in range(MAX_RETRIES):
        try:
            c = client.metadata.get_cost(
                dataset=DATASET, symbols=[symbol], schema=schema,
                start=start, end=end, stype_in=STYPE,
            )
            return float(c), ""
        except Exception as exc:
            last = str(exc).split("\n")[0][:80]
            time.sleep(BASE_SLEEP * (2 ** attempt))
    return None, last


def fmt(v) -> str:
    return "err" if v is None else f"${v:,.2f}"


def main() -> None:
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY not set")
        print('  PowerShell:  $env:DATABENTO_API_KEY = "db-..."')
        sys.exit(1)
    try:
        import databento as db
    except ImportError:
        print("ERROR: pip install databento")
        sys.exit(1)

    client = db.Historical(key=api_key)
    end_safe = safe_end_date()
    errors: list[str] = []

    # ── PART 1: the ohlcv-1m re-download already planned ───────────────────
    hr("PART 1 — ohlcv-1m RE-DOWNLOAD (GC / ZN / 6E / 6J switched to .v.0)")
    print(f"  {'sym':<5} {'symbol':<9} {'window':<26} {'cost':>10}")
    print("  " + "-" * 54)

    ohlcv_total = 0.0
    for sym, instr in INSTRUMENTS.items():
        if _cache_path(sym, SCHEMA_1M).exists():
            continue
        start = (DOWNLOAD_START if instr.get("has_local_data")
                 else instr.get("data_start", DOWNLOAD_START))
        end = DOWNLOAD_END if instr.get("has_local_data") else end_safe
        c, e = cost(client, instr["continuous_symbol"], SCHEMA_1M, start, end)
        if e:
            errors.append(f"{instr['continuous_symbol']} {SCHEMA_1M}: {e}")
        if c:
            ohlcv_total += c
        print(f"  {sym:<5} {instr['continuous_symbol']:<9} "
              f"{start[:10] + ' -> ' + end[:10]:<26} {fmt(c):>10}")
    print("  " + "-" * 54)
    print(f"  {'ohlcv-1m subtotal':<42} {fmt(ohlcv_total):>10}")

    # ── PART 2: spread schemas for ETH / BTC ───────────────────────────────
    hr("PART 2 — SPREAD SCHEMAS for ETH / BTC (replaces the estimate)")
    print("Current ESTIMATED slippage: " + ", ".join(
        f"{s}={SLIPPAGE_TICKS_BY_SYMBOL.get(s)}t" for s in FOCUS))
    print()
    print(f"  {'sym':<5} {'schema':<8} {'window':<20} {'cost':>12}")
    print("  " + "-" * 50)

    matrix: dict[tuple, float | None] = {}
    for sym in FOCUS:
        root = INSTRUMENTS[sym]["continuous_symbol"].split(".")[0]
        symbol = f"{root}.{_roll_tag(sym)}.0"
        for schema in SPREAD_SCHEMAS:
            for label, (s, e) in WINDOWS.items():
                c, err = cost(client, symbol, schema, s, e)
                matrix[(sym, schema, label)] = c
                if err:
                    errors.append(f"{symbol} {schema} {label}: {err}")
                print(f"  {sym:<5} {schema:<8} {label:<20} {fmt(c):>12}")
        print()

    # ── PART 3: full history, to show why sampling is the right call ───────
    hr("PART 3 — FULL HISTORY on spread schemas (for contrast, not purchase)")
    print("If these are large, sampling is not a compromise -- it is the only")
    print("sane option, and it is statistically sufficient anyway.\n")
    print(f"  {'sym':<5} {'schema':<8} {'window':<26} {'cost':>12}")
    print("  " + "-" * 56)
    for sym in FOCUS:
        root = INSTRUMENTS[sym]["continuous_symbol"].split(".")[0]
        symbol = f"{root}.{_roll_tag(sym)}.0"
        start = INSTRUMENTS[sym].get("data_start", DOWNLOAD_START)
        for schema in SPREAD_SCHEMAS:
            c, err = cost(client, symbol, schema, start, end_safe)
            if err:
                errors.append(f"{symbol} {schema} full: {err}")
            print(f"  {sym:<5} {schema:<8} "
                  f"{start[:10] + ' -> ' + end_safe[:10]:<26} {fmt(c):>12}")
        print()

    # ── PART 4: bbo-1m for ALL instruments, 1 month ────────────────────────
    hr("PART 4 — bbo-1m ONE MONTH, ALL 10 INSTRUMENTS")
    print("Removes the slippage assumption everywhere, not just ETH/BTC.")
    print("Every constant in SLIPPAGE_TICKS_BY_SYMBOL is currently a guess.\n")
    print(f"  {'sym':<5} {'symbol':<9} {'est. slip':>10} {'cost':>10}")
    print("  " + "-" * 40)
    all_bbo = 0.0
    s, e = WINDOWS["1 month (2024-06)"]
    for sym, instr in INSTRUMENTS.items():
        root = instr["continuous_symbol"].split(".")[0]
        symbol = f"{root}.{_roll_tag(sym)}.0"
        c, err = cost(client, symbol, "bbo-1m", s, e)
        if err:
            errors.append(f"{symbol} bbo-1m: {err}")
        if c:
            all_bbo += c
        print(f"  {sym:<5} {symbol:<9} "
              f"{str(SLIPPAGE_TICKS_BY_SYMBOL.get(sym, '?')) + 't':>10} {fmt(c):>10}")
    print("  " + "-" * 40)
    print(f"  {'bbo-1m all-10 one month':<26} {fmt(all_bbo):>12}")

    # ── BUDGET SUMMARY ─────────────────────────────────────────────────────
    hr("CONSOLIDATED BUDGET")

    eth_bbo_1m = matrix.get(("ETH", "bbo-1m", "1 month (2024-06)"))
    btc_bbo_1m = matrix.get(("BTC", "bbo-1m", "1 month (2024-06)"))
    eth_tbbo_1m = matrix.get(("ETH", "tbbo", "1 month (2024-06)"))
    btc_tbbo_1m = matrix.get(("BTC", "tbbo", "1 month (2024-06)"))

    def _s(*vals) -> float:
        return sum(v for v in vals if v)

    opt_a = ohlcv_total
    opt_b = ohlcv_total + _s(eth_bbo_1m, btc_bbo_1m)
    opt_c = ohlcv_total + _s(eth_tbbo_1m, btc_tbbo_1m)
    opt_d = ohlcv_total + all_bbo

    print(f"  A  ohlcv-1m re-download only (keep estimated slippage)")
    print(f"     {fmt(opt_a):>12}")
    print()
    print(f"  B  A + bbo-1m 1 month for ETH/BTC")
    print(f"     {fmt(opt_b):>12}   <- measures the two that matter most")
    print()
    print(f"  C  A + tbbo 1 month for ETH/BTC")
    print(f"     {fmt(opt_c):>12}   <- spread at actual trade times")
    print()
    print(f"  D  A + bbo-1m 1 month for ALL 10")
    print(f"     {fmt(opt_d):>12}   <- removes every slippage guess")

    print("\n  Note: 3 sampled months (2020/2022/2024) instead of 1 multiplies")
    print("  only the spread portion, and is worth it if the monthly figures")
    print("  above differ materially across those years -- spreads widened in")
    print("  2020 and 2022, and a 2024-only median would understate costs.")

    if errors:
        hr(f"ERRORS AFTER {MAX_RETRIES} RETRIES ({len(errors)})")
        seen = set()
        for e_ in errors:
            k = e_.split(":")[-1].strip()[:60]
            if k not in seen:
                seen.add(k)
                print(f"  {e_}")
        print("\n  A schema erroring for every window usually means it is not")
        print("  available on this dataset or plan, not that it is free.")
    else:
        hr("NO ERRORS — every figure above resolved")

    print("\n" + "=" * 96 + "\n")


if __name__ == "__main__":
    main()
