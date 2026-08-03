"""
cost_check.py — print Databento cost estimates for every instrument that
                would be downloaded on first run, WITHOUT spending any credits.

Usage:
    set DATABENTO_API_KEY=<your-key>
    python cost_check.py

Output: one line per symbol showing the cost of its 1m download window.
Only symbols whose cache is MISSING are shown (nothing to do for cached ones).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# ── resolve project root ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import (
    DATASET, DOWNLOAD_END, DOWNLOAD_START, INSTRUMENTS,
    SCHEMA_1M, STYPE, DATA_DIR,
)

_DATA = _ROOT / DATA_DIR


def _cache_path(sym: str) -> Path:
    return _DATA / f"{sym}_1m.parquet"


def main() -> None:
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: DATABENTO_API_KEY not set.")
        sys.exit(1)

    try:
        import databento as db
    except ImportError:
        print("ERROR: pip install databento")
        sys.exit(1)

    client = db.Historical(key=api_key)
    today  = pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")

    print(f"\n{'Symbol':<6}  {'Asset class':<14}  {'Window':<24}  {'Cost (USD)':>10}")
    print("-" * 62)

    total = 0.0
    any_missing = False

    for sym, instr in INSTRUMENTS.items():
        cache = _cache_path(sym)
        if cache.exists():
            print(f"{sym:<6}  {'(cached)':>38}")
            continue

        any_missing = True
        has_local = instr.get("has_local_data", False)
        if has_local:
            # Same as legacy path: only download the gap that local data doesn't cover
            start, end = DOWNLOAD_START, DOWNLOAD_END
        else:
            start = instr.get("data_start", DOWNLOAD_START)
            end   = today

        window_str = f"{start[:10]}  →  {end[:10]}"
        asset_class = instr.get("asset_class", "?")

        try:
            cost = client.metadata.get_cost(
                dataset=DATASET,
                symbols=[instr["continuous_symbol"]],
                schema=SCHEMA_1M,
                start=start,
                end=end,
                stype_in=STYPE,
            )
        except Exception as exc:
            cost = float("nan")
            print(f"{sym:<6}  {asset_class:<14}  {window_str:<24}  ERROR: {exc}")
            continue

        total += cost
        print(f"{sym:<6}  {asset_class:<14}  {window_str:<24}  ${cost:>9.2f}")

    print("-" * 62)
    if any_missing:
        print(f"{'TOTAL new spend':>44}  ${total:>9.2f}")
    else:
        print("All caches present — no download needed.")
    print()


if __name__ == "__main__":
    main()
