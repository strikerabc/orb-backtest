"""
diagnose_gc.py — is GC sparse because .c.0 tracks illiquid serial months?

Hypothesis
----------
Databento continuous symbology:
    .c.0  = calendar roll   (front month by expiry date)
    .v.0  = volume roll     (most actively traded contract)
    .n.0  = open-interest roll

For ES/NQ/CL the calendar front month IS the liquid month, so .c.0 is fine.
COMEX gold liquidity concentrates in Feb/Apr/Jun/Aug/Oct/Dec; the serial
months (Jan/Mar/May/Jul/Sep/Nov) trade very thinly. If GC.c.0 follows those
serial contracts, we would see:
    - few distinct contracts, or contracts that are serial-month coded
    - very low bars/day, roughly uniform across years (not a truncated tail)

Contract month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun
                      N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
Gold liquid months  : G(Feb) J(Apr) M(Jun) Q(Aug) V(Oct) Z(Dec)

Usage:
    python diagnose_gc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import DATA_DIR, INSTRUMENTS

_DATA = _ROOT / DATA_DIR

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)

MONTH_CODE = {
    "F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
    "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec",
}
GOLD_LIQUID = {"G", "J", "M", "Q", "V", "Z"}


def hr(t: str) -> None:
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def load(sym: str) -> pd.DataFrame | None:
    p = _DATA / f"{sym}_1m.parquet"
    if not p.exists():
        print(f"  {sym}: MISSING")
        return None
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def contract_report(sym: str, df: pd.DataFrame) -> None:
    hr(f"{sym} — contract composition")
    if "_contract" not in df.columns:
        print("  no _contract column")
        return

    vc = df["_contract"].value_counts(dropna=False)
    print(f"  distinct contracts: {vc.shape[0]}")
    print(f"  total bars        : {len(df):,}")
    print("\n  top 25 contracts by bar count:")
    for name, n in vc.head(25).items():
        s = str(name)
        # month code is typically the char before the trailing year digit(s)
        code = ""
        for ch in reversed(s):
            if ch.isalpha():
                code = ch
                break
        month = MONTH_CODE.get(code, "?")
        tag = ""
        if sym == "GC" and code:
            tag = "LIQUID" if code in GOLD_LIQUID else "serial/thin"
        print(f"    {s:<16} {n:>10,}  month={month:<4} {tag}")


def density_report(sym: str, df: pd.DataFrame) -> None:
    hr(f"{sym} — bar density by year and by UTC hour")
    ts = df["timestamp"]

    by_year = ts.dt.year.value_counts().sort_index()
    print("  bars per year:")
    for y, n in by_year.items():
        print(f"    {y}  {n:>10,}")

    print("\n  bars per UTC hour (0-23):")
    by_hour = ts.dt.hour.value_counts().sort_index()
    mx = int(by_hour.max()) if len(by_hour) else 1
    for h in range(24):
        n = int(by_hour.get(h, 0))
        bar = "#" * int(50 * n / mx) if mx else ""
        print(f"    {h:02d}h {n:>9,}  {bar}")


def main() -> None:
    # GC is the suspect; CL is the control (same ~24h commodity profile)
    for sym in ("GC", "CL"):
        df = load(sym)
        if df is None:
            continue
        contract_report(sym, df)
        if sym == "GC":
            density_report(sym, df)

    # quick contract-count comparison across the whole universe
    hr("distinct contract count — all instruments")
    print(f"  {'sym':<6} {'bars':>12} {'contracts':>11}  {'bars/contract':>14}")
    for sym in INSTRUMENTS:
        p = _DATA / f"{sym}_1m.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["_contract"]) if True else None
        n_c = d["_contract"].nunique(dropna=False) if "_contract" in d.columns else -1
        print(f"  {sym:<6} {len(d):>12,} {n_c:>11}  {len(d)/max(n_c,1):>14,.0f}")

    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
