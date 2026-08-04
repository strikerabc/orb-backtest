"""
measure_spreads.py — replace ESTIMATED slippage with MEASURED bid-ask spreads.

Why
---
SLIPPAGE_TICKS_BY_SYMBOL in config.py is currently guessed (ETH 10t, BTC 2t,
rest 1t). ohlcv-1m carries no bid/ask, so those are assumptions. They matter:
a flat 1-tick assumption undercharged ETH by roughly an order of magnitude and
made it the only instrument to "survive" costs in the first 10-instrument sweep.

Schema choice (measured, not assumed)
-------------------------------------
    bbo-1m   $0.04 per instrument-month, linear.  Full history ETH $2.63 / BTC $3.61
    tbbo     ~3-6x that.                          Full history ETH $14.63 / BTC $23.97
    mbp-1    ~20x that.                           Full history ETH $91.32 / BTC $138.09

bbo-1m is a 1-minute bid/ask snapshot, roughly the same record count as
ohlcv-1m. mbp-1 carries every book change -- depth this strategy never models.

How spread maps to round-trip slippage
--------------------------------------
Backtest fills use ohlcv TRADE prices (close/high/low), which sit between bid
and ask. In reality a buy pays the ask and a sell receives the bid, so each
side gives up about half the spread against mid:

    entry crosses  : spread / 2
    exit  crosses  : spread / 2
    round trip     : one full spread

So slippage_ticks = median spread in ticks.

Two honest limitations of that mapping:
  - It is CONSERVATIVE for take-profit exits, which could rest as limit orders
    and never cross.
  - It EXCLUDES market impact. ORB entries fire on breakouts, where price is
    moving through the book. Measured spread is a FLOOR on true execution
    cost, not the whole of it.

Sampling
--------
Three Junes (2020, 2022, 2024) to span regimes: spreads widened materially in
2020 (COVID) and 2022 (rate hikes), so a 2024-only median would understate
costs. Holding the calendar month fixed keeps cross-year comparison
like-for-like; seasonal variation (thin August, holiday December) is a known
uncovered gap.

Spreads are measured ONLY inside each instrument's traded session windows,
since that is when the strategy actually pays them.

Cost gate
---------
Prices everything and exits unless --confirm is passed. Downloads cache under
data/spreads/ so re-runs never re-spend.

Usage:
    $env:DATABENTO_API_KEY = "db-..."
    python measure_spreads.py              # price only, no download
    python measure_spreads.py --confirm    # download and measure
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import (
    DATA_DIR, DATASET, INSTRUMENTS, SESSIONS,
    SLIPPAGE_TICKS_BY_SYMBOL, STYPE,
)
from src.data_layer import _roll_tag

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("orb.spread")

SCHEMA_BBO = "bbo-1m"

# Three Junes: COVID-era, rate-hike-era, recent-normal.
SAMPLE_MONTHS = ["2020-06", "2022-06", "2024-06"]

_SPREADS = _ROOT / DATA_DIR / "spreads"
_SPREADS.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 4
BASE_SLEEP = 0.6

# Discard obvious quote glitches: crossed/locked books and absurd outliers.
MAX_PLAUSIBLE_SPREAD_TICKS = 500.0


# ── helpers ────────────────────────────────────────────────────────────────

def hr(t: str) -> None:
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def _month_bounds(month: str) -> tuple[str, str]:
    """'2024-06' -> ('2024-06-01', '2024-07-01'). end is exclusive."""
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _month_is_valid(sym: str, month: str) -> tuple[bool, str]:
    """
    Skip months before the contract existed.

    ETH launched 2021-02-08, so bbo-1m for 2020-06 returns
    422 symbology_invalid_request. That is correct API behaviour, not a
    failure, and must not be retried or counted as an error.
    """
    data_start = pd.Timestamp(INSTRUMENTS[sym].get("data_start", "2019-01-01"),
                              tz="UTC")
    m_start, m_end = _month_bounds(month)
    if pd.Timestamp(m_end, tz="UTC") <= data_start:
        return False, f"pre-launch (starts {data_start.date()})"
    return True, ""


def _symbol_for(sym: str) -> str:
    root = INSTRUMENTS[sym]["continuous_symbol"].split(".")[0]
    return f"{root}.{_roll_tag(sym)}.0"


def _cache_file(sym: str, month: str) -> Path:
    """Normalised [timestamp, bid, ask] cache."""
    return _SPREADS / f"{sym}_{_roll_tag(sym)}_{month}_bbo1m.parquet"


def _raw_cache_file(sym: str, month: str) -> Path:
    """
    RAW API response cache, written before any parsing.

    Column names below are inferred, not verified against the bbo-1m schema.
    If they are wrong, normalisation fails -- and without a raw cache the
    download would be re-paid on every retry. Persisting the raw response
    first makes parsing failures free to fix and re-run.
    """
    return _SPREADS / f"{sym}_{_roll_tag(sym)}_{month}_bbo1m_RAW.parquet"


def _retry(fn, *a, **kw):
    """Call fn with backoff. Returns (result, error_first_line)."""
    last = ""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*a, **kw), ""
        except Exception as exc:
            last = str(exc).split("\n")[0][:90]
            time.sleep(BASE_SLEEP * (2 ** attempt))
    return None, last


# ── download ───────────────────────────────────────────────────────────────

_BID_CANDIDATES = ["bid_px_00", "bid_px", "bid_price", "bid"]
_ASK_CANDIDATES = ["ask_px_00", "ask_px", "ask_price", "ask"]


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalise(df: pd.DataFrame, sym: str, month: str) -> pd.DataFrame | None:
    """
    Map a raw bbo-1m frame to [timestamp, bid, ask].

    Separated from the download so a wrong column guess costs nothing to fix:
    the raw response is already cached and normalisation can be re-run offline.
    """
    if df is None or len(df) == 0:
        log.warning("  %-5s %s  empty response", sym, month)
        return None

    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()

    ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
    bid_col = _pick_col(df, _BID_CANDIDATES)
    ask_col = _pick_col(df, _ASK_CANDIDATES)
    if bid_col is None or ask_col is None:
        log.error("  %-5s %s  no bid/ask columns found.", sym, month)
        log.error("      columns present: %s", list(df.columns)[:20])
        log.error("      raw response IS cached -- fix _BID_CANDIDATES/"
                  "_ASK_CANDIDATES and re-run, no re-download needed.")
        return None

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[ts_col], utc=True),
        "bid": pd.to_numeric(df[bid_col], errors="coerce"),
        "ask": pd.to_numeric(df[ask_col], errors="coerce"),
    }).dropna()
    return out if len(out) else None


def download_month(client, sym: str, month: str) -> pd.DataFrame | None:
    """
    Fetch one month of bbo-1m for sym and return DataFrame[timestamp, bid, ask].

    Cache order, cheapest first:
      1. normalised cache  -> no download, no parsing
      2. raw cache         -> no download, re-parse only
      3. download          -> persist RAW immediately, then parse

    The raw response is written to disk BEFORE parsing is attempted. Column
    names here are inferred rather than verified, so a bad guess would
    otherwise re-charge the download on every retry.
    """
    cache = _cache_file(sym, month)
    if cache.exists():
        log.info("  %-5s %s  cached (normalised)", sym, month)
        return pd.read_parquet(cache)

    raw_cache = _raw_cache_file(sym, month)
    if raw_cache.exists():
        log.info("  %-5s %s  cached (raw) -- re-parsing, no spend", sym, month)
        out = _normalise(pd.read_parquet(raw_cache), sym, month)
        if out is not None:
            out.to_parquet(cache, index=False)
        return out

    start, end = _month_bounds(month)
    symbol = _symbol_for(sym)

    data, err = _retry(
        client.timeseries.get_range,
        dataset=DATASET, symbols=[symbol], schema=SCHEMA_BBO,
        start=start, end=end, stype_in=STYPE,
    )
    if data is None:
        log.warning("  %-5s %s  FAILED: %s", sym, month, err)
        return None

    try:
        raw = data.to_df()
    except Exception as exc:
        log.error("  %-5s %s  to_df() failed: %s", sym, month, exc)
        return None

    # Persist the paid response before anything can go wrong downstream.
    try:
        to_save = raw if isinstance(raw.index, pd.RangeIndex) else raw.reset_index()
        to_save.to_parquet(raw_cache, index=False)
        log.info("  %-5s %s  raw cached (%d rows) -> %s",
                 sym, month, len(to_save), raw_cache.name)
    except Exception as exc:
        # Keep going: the data is still in memory for this run.
        log.warning("  %-5s %s  could not cache raw (%s); parsing in-memory",
                    sym, month, str(exc)[:60])

    out = _normalise(raw, sym, month)
    if out is None:
        return None

    out.to_parquet(cache, index=False)
    log.info("  %-5s %s  %d quotes -> %s", sym, month, len(out), cache.name)
    return out


# ── measurement ────────────────────────────────────────────────────────────

def measure(sym: str, frames: list[pd.DataFrame]) -> dict | None:
    """
    Median spread in ticks for sym, overall and per traded session.

    Only quotes inside the instrument's own session windows count, since those
    are the only spreads the strategy ever pays.
    """
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    tick = INSTRUMENTS[sym]["tick_size"]

    spread_ticks = (df["ask"] - df["bid"]) / tick
    # Drop crossed/locked books and glitch outliers.
    ok = (spread_ticks > 0) & (spread_ticks <= MAX_PLAUSIBLE_SPREAD_TICKS)
    n_bad = int((~ok).sum())
    df = df.loc[ok].copy()
    df["spread_ticks"] = spread_ticks.loc[ok]
    if df.empty:
        return None

    sessions = INSTRUMENTS[sym].get("sessions", list(SESSIONS.keys()))
    per_session: dict[str, dict] = {}
    in_any = np.zeros(len(df), dtype=bool)
    sp = df["spread_ticks"].to_numpy()

    for sess_name in sessions:
        sess = SESSIONS[sess_name]
        loc = df["timestamp"].dt.tz_convert(sess["tz"])
        wall = loc.dt.hour * 60 + loc.dt.minute
        o = sess["open"][0] * 60 + sess["open"][1]
        x = sess["exit"][0] * 60 + sess["exit"][1]
        m = ((wall >= o) & (wall < x)).to_numpy()
        in_any |= m
        if m.sum() > 0:
            s = sp[m]
            per_session[sess_name] = {
                "median": float(np.median(s)),
                "p75": float(np.percentile(s, 75)),
                "p95": float(np.percentile(s, 95)),
                "quotes": int(m.sum()),
            }

    if not per_session:
        return None

    traded = sp[in_any]
    meds = [v["median"] for v in per_session.values()]

    return {
        "sym": sym,
        "quotes_total": len(df),
        "quotes_in_session": int(in_any.sum()),
        "bad_quotes": n_bad,
        # PER-SESSION IS THE PRIMARY RESULT -- see note below.
        "per_session": per_session,
        "median_worst_session": float(max(meds)),
        "median_best_session": float(min(meds)),
        # Diagnostic only. NOT for config use: pooling across sessions weights
        # by WINDOW LENGTH rather than by trades taken (LDN's 4h window
        # contributes more quotes than NY's 2.5h), so this scalar has no
        # economic meaning for a multi-session instrument. Retained purely to
        # show how far a single pooled number drifts from the per-session
        # truth.
        "pooled_median_DIAGNOSTIC": float(np.median(traded)),
        "median_all_hours_DIAGNOSTIC": float(np.median(sp)),
        "estimated": SLIPPAGE_TICKS_BY_SYMBOL.get(sym),
    }


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true",
                    help="actually download (otherwise price only)")
    args = ap.parse_args()

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

    # ── plan + price ───────────────────────────────────────────────────────
    hr(f"PLAN — {SCHEMA_BBO}, months {', '.join(SAMPLE_MONTHS)}")
    jobs: list[tuple[str, str]] = []
    skipped: list[str] = []
    cached = 0

    for sym in INSTRUMENTS:
        for month in SAMPLE_MONTHS:
            valid, why = _month_is_valid(sym, month)
            if not valid:
                skipped.append(f"{sym} {month}: {why}")
                continue
            if _cache_file(sym, month).exists():
                cached += 1
                continue
            jobs.append((sym, month))

    total = 0.0
    errors: list[str] = []
    for sym, month in jobs:
        start, end = _month_bounds(month)
        c, err = _retry(client.metadata.get_cost,
                        dataset=DATASET, symbols=[_symbol_for(sym)],
                        schema=SCHEMA_BBO, start=start, end=end, stype_in=STYPE)
        if err:
            errors.append(f"{_symbol_for(sym)} {month}: {err}")
        if c:
            total += float(c)

    print(f"  instrument-months to download : {len(jobs)}")
    print(f"  already cached (no spend)     : {cached}")
    print(f"  skipped (pre-launch)          : {len(skipped)}")
    for s in skipped:
        print(f"      {s}")
    print(f"\n  ESTIMATED SPEND              : ${total:,.2f}")

    if errors:
        print(f"\n  pricing errors ({len(errors)}):")
        for e in errors[:6]:
            print(f"      {e}")

    if not args.confirm:
        print("\n  Nothing downloaded. Re-run with --confirm to proceed:")
        print("      python measure_spreads.py --confirm")
        print("\n" + "=" * 96 + "\n")
        return

    # ── download + measure ─────────────────────────────────────────────────
    hr("DOWNLOADING")
    per_sym: dict[str, list[pd.DataFrame]] = {}
    for sym in INSTRUMENTS:
        for month in SAMPLE_MONTHS:
            valid, _ = _month_is_valid(sym, month)
            if not valid:
                continue
            df = download_month(client, sym, month)
            if df is not None and len(df):
                per_sym.setdefault(sym, []).append(df)

    hr("MEASURED SPREADS PER SESSION (ticks; round trip = one full spread)")
    print("Per-session is the primary result. A single pooled scalar is shown")
    print("only to expose how far it drifts -- it is weighted by window length,")
    print("not by trades taken, so it has no economic meaning.\n")
    print(f"  {'sym':<5} {'sess':<5} {'est':>5} {'median':>7} {'p75':>6} "
          f"{'p95':>6} {'vs est':>8} {'quotes':>9}")
    print("  " + "-" * 60)

    results: list[dict] = []
    for sym in INSTRUMENTS:
        r = measure(sym, per_sym.get(sym, []))
        if r is None:
            print(f"  {sym:<5} {'--':<5} {'NO DATA':>21}")
            continue
        results.append(r)
        est = r["estimated"]
        for sess_name, s in r["per_session"].items():
            ratio = (s["median"] / est) if est else float("nan")
            print(f"  {sym:<5} {sess_name:<5} {est:>5.1f} {s['median']:>7.2f} "
                  f"{s['p75']:>6.2f} {s['p95']:>6.2f} {ratio:>7.2f}x "
                  f"{s['quotes']:>9,}")
        print(f"  {'':<5} {'(pooled diagnostic':<5} "
              f"{r['pooled_median_DIAGNOSTIC']:>18.2f}  <- do not use)")

    if not results:
        print("\n  No measurements produced.")
        return

    # ── config block ───────────────────────────────────────────────────────
    hr("CONFIG BLOCK — paste into config.py")
    print("# Measured via " + SCHEMA_BBO + ", months " + ", ".join(SAMPLE_MONTHS))
    print("SLIPPAGE_TICKS_BY_SYMBOL_SESSION: dict[tuple[str, str], float] = {")
    for r in sorted(results, key=lambda d: d["sym"]):
        for sess_name, s in r["per_session"].items():
            # floor of one tick: you cannot cross less than the minimum increment
            v = max(1.0, round(s["median"], 2))
            print(f'    ("{r["sym"]}", "{sess_name}"): {v:>6.2f},   '
                  f'# was {r["estimated"]:.1f} (flat)')
    print("}")
    print()
    print("# Fallback when a (symbol, session) pair is absent. Worst session")
    print("# per symbol -- conservative rather than optimistic.")
    print("SLIPPAGE_TICKS_BY_SYMBOL: dict[str, float] = {")
    for r in sorted(results, key=lambda d: d["sym"]):
        v = max(1.0, round(r["median_worst_session"], 2))
        print(f'    "{r["sym"]}": {v:>6.2f},   # worst session')
    print("}")
    print()
    print("SLIPPAGE_PROVENANCE: dict[str, str] = {")
    for r in sorted(results, key=lambda d: d["sym"]):
        print(f'    "{r["sym"]}": "measured",')
    print("}")

    # ── impact flags ───────────────────────────────────────────────────────
    hr("WHERE THE ESTIMATE WAS WRONG")
    print("Judged on the WORST session per symbol, since that is the fallback.\n")
    for r in sorted(results, key=lambda d: -(
            (d["median_worst_session"] / d["estimated"]) if d["estimated"] else 0)):
        est = r["estimated"]
        if not est:
            continue
        worst = r["median_worst_session"]
        best = r["median_best_session"]
        ratio = worst / est
        if ratio >= 1.5:
            verdict = f"UNDERCHARGED {ratio:.2f}x -- costs were too optimistic"
        elif ratio <= 0.67:
            verdict = f"OVERCHARGED {1/ratio:.2f}x -- costs were too harsh"
        else:
            verdict = "estimate close enough"
        spread_range = ""
        if worst > 0 and best > 0 and worst / best >= 1.5:
            spread_range = (f"  [sessions differ {worst/best:.1f}x: "
                            f"{best:.2f}-{worst:.2f}t -- a flat scalar would "
                            f"bias cross-session ranking]")
        print(f"  {r['sym']:<5} est {est:>5.1f}t  worst {worst:>6.2f}t   "
              f"{verdict}{spread_range}")

    print("\n  Reminder: measured spread is a FLOOR on execution cost. It")
    print("  excludes market impact, and ORB entries fire on breakouts where")
    print("  price is moving through the book. Treat these as lower bounds.")

    # ── save ───────────────────────────────────────────────────────────────
    flat: list[dict] = []
    for r in results:
        for sess_name, s in r["per_session"].items():
            flat.append({
                "sym": r["sym"], "session": sess_name,
                "median_ticks": round(s["median"], 4),
                "p75_ticks": round(s["p75"], 4),
                "p95_ticks": round(s["p95"], 4),
                "quotes": s["quotes"],
                "estimated_ticks": r["estimated"],
                "bad_quotes": r["bad_quotes"],
                "months": " ".join(SAMPLE_MONTHS),
                "schema": SCHEMA_BBO,
            })
    out = _SPREADS / "measured_spreads.csv"
    pd.DataFrame(flat).to_csv(out, index=False)
    print(f"\n  saved -> {out}  ({len(flat)} symbol-session rows)")
    print("\n" + "=" * 96 + "\n")


if __name__ == "__main__":
    main()
