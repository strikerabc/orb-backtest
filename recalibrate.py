"""
recalibrate.py — redo stats + null + report from the existing trade log.

Why this exists
---------------
Two defects were found AFTER the 67-minute sweep, both downstream of the trade
log. The log itself is sound (correct .v.0 rolls, measured per-session
slippage), so re-running the 44-minute download and 15-minute sweep would be
pure waste.

  1. stats.py never applied the tp_unfillable flag. Take-profits inside one
     tick of entry -- unfillable, inside the spread -- were counted as wins.
     Measured: 75-83% of ZN trades at rr=0.25, cutting gross expectancy from
     0.2321 to 0.0855 when excluded. rr >= 1.0 was unaffected.

  2. The null comparator drew swing-derived stops 3-4x TIGHTER than the
     observed strategy (ES obs 34 ticks vs null 4). That inflated same-bar
     SL/TP collisions 140x, and trade_sim resolves those as SL by design, so
     the null lost to the tiebreak rule rather than to worse timing:

         swing-derived null, median gross : -0.1459
         matched-stop null, median gross  : +0.0084   <- a fair null sits at ~0

     null_p was therefore ANTI-conservative, giving null_p < 0.05 for 78% of
     rankable variants (5,599 of 7,140). Under a matched null, 3 of 6 spot-
     checked variants lost significance outright (ES 0.0020 -> 0.1257,
     GC 0.0020 -> 0.6687).

What it recomputes
------------------
  compute_summary        (now excluding unfillable TPs)
  compute_regime_summary (same exclusion)
  enrich_summary_with_null (matched-stop comparator)
  write_report

Session days must be rebuilt because null pools are simulated against real
bars; that costs ~4 minutes off cached parquet and spends nothing.

Usage:
    python recalibrate.py --limit 300     # sanity-check on the top 300 variants
    python recalibrate.py                 # full run (~40-50 min)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from src.config import INSTRUMENTS, OUTPUTS_DIR, SCHEMA_1M, SESSIONS
from src.data_layer import (
    _cache_path, _compute_enrichment, ensure_daily, ensure_data,
)
from src.null_calibrator import build_r_ticks_map, enrich_summary_with_null
from src.range_builder import SessionDay, build_session_days
from src.regime_sampler import filter_to_window, select_windows
from src.report import write_report
from src.stats import compute_regime_summary, compute_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orb.recal")
_OUT = Path(OUTPUTS_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="calibrate only the top N variants by gross "
                         "expectancy (sanity check; 0 = all)")
    args = ap.parse_args()

    t0 = time.perf_counter()

    tl_path = _OUT / "trade_log.parquet"
    if not tl_path.exists():
        log.error("missing %s -- run main.py first", tl_path)
        raise SystemExit(1)

    # ── every cache must already exist; this must never spend ──────────────
    missing = [s for s in INSTRUMENTS if not _cache_path(s, SCHEMA_1M).exists()]
    if missing:
        log.error("missing caches for %s -- run main.py, not this script",
                  ", ".join(missing))
        raise SystemExit(1)

    log.info("Loading trade log...")
    trade_log = pd.read_parquet(tl_path)
    log.info("trade log: %d rows x %d cols", len(trade_log), trade_log.shape[1])
    if "tp_unfillable" not in trade_log.columns:
        log.error("trade log has no tp_unfillable column -- it predates the "
                  "flag. Re-run main.py.")
        raise SystemExit(1)

    # ── 1. statistics, now excluding unfillable TPs ────────────────────────
    log.info("Computing summary (excluding unfillable TPs)...")
    summary = compute_summary(trade_log)
    log.info("summary: %d variants", len(summary))

    log.info("Computing regime summary...")
    regime_summary = compute_regime_summary(trade_log)
    log.info("regime summary: %d rows", len(regime_summary))

    # ── 2. rebuild session days for the null pools ─────────────────────────
    log.info("Rebuilding session days from cache (no spend)...")
    data: dict[str, pd.DataFrame] = {}
    for sym, instr in INSTRUMENTS.items():
        data[sym] = _compute_enrichment(
            ensure_data(sym), ensure_daily(sym),
            tick_size=instr["tick_size"])

    all_ts = pd.concat([d["timestamp"] for d in data.values()])
    windows = select_windows(all_ts.min().date(), all_ts.max().date())

    session_days_map: dict[tuple, list[SessionDay]] = {}
    for sym, df_full in data.items():
        for sess in INSTRUMENTS[sym].get("sessions", list(SESSIONS)):
            for w in windows:
                dfw = filter_to_window(df_full, w)
                if len(dfw) == 0:
                    continue
                session_days_map.setdefault((sym, sess), []).extend(
                    build_session_days(dfw, sym, sess))
    log.info("session-day map: %d (instrument, session) keys",
             len(session_days_map))

    # ── 3. matched-stop null calibration ───────────────────────────────────
    log.info("Building observed r_ticks map...")
    r_map = build_r_ticks_map(trade_log)

    target = summary
    if args.limit and args.limit < len(summary):
        target = (summary.sort_values("expectancy_gross_r", ascending=False)
                         .head(args.limit).copy())
        log.info("LIMIT MODE: calibrating top %d variants only. Output is a "
                 "sanity check, NOT a complete summary.", len(target))

    log.info("Running matched-stop null calibration on %d rows...", len(target))
    enriched = enrich_summary_with_null(target, session_days_map, r_map)

    # ── 4. write ───────────────────────────────────────────────────────────
    if args.limit:
        out = _OUT / f"summary_recal_top{args.limit}.parquet"
        enriched.to_parquet(out, index=False)
        log.info("wrote %s (limit mode -- report not regenerated)", out)
        _report_flip(enriched)
    else:
        write_report(enriched, regime_summary, trade_log, _OUT)
        _report_flip(enriched)

    log.info("Done in %.1fs", time.perf_counter() - t0)


def _report_flip(enriched: pd.DataFrame) -> None:
    """Print how the matched null changed the significance picture."""
    from src.config import MIN_TRADES_FOR_RANKING

    if "null_p_value" not in enriched.columns:
        return
    rank = enriched[enriched["trade_count"] >= MIN_TRADES_FOR_RANKING]
    if rank.empty:
        return

    net_pos = rank[rank["expectancy_net_r"] > 0]
    sig = rank[rank["null_p_value"] < 0.05]
    both = rank[(rank["expectancy_net_r"] > 0) & (rank["null_p_value"] < 0.05)]

    print("\n" + "=" * 96)
    print("MATCHED-NULL RESULT")
    print("=" * 96)
    print(f"  rankable variants           : {len(rank):,}")
    print(f"  net-positive                : {len(net_pos):,}")
    print(f"  null_p < 0.05               : {len(sig):,}  "
          f"({100.0*len(sig)/len(rank):.1f}%)")
    print(f"  BOTH net-positive AND sig   : {len(both):,}")
    print()
    print("  For reference, the swing-derived null gave 5,599 of 7,140 (78%)")
    print("  significant and 156 net-positive-and-significant. A fair")
    print("  comparator should land far nearer 5% under the null.")

    if "null_design" in rank.columns:
        print()
        print("  comparator used:")
        for k, v in rank["null_design"].value_counts().items():
            print(f"    {k:<16} {v:,}")

    if not both.empty:
        cols = [c for c in ["instrument", "session", "range_minutes",
                            "entry_mode", "closure_tf", "direction", "rr",
                            "trade_count", "n_unfillable_excluded",
                            "expectancy_gross_r", "expectancy_net_r",
                            "null_p_value", "null_design"]
                if c in both.columns]
        print("\n  surviving variants (top 25 by net expectancy):")
        print(both.sort_values("expectancy_net_r", ascending=False)
                  .head(25)[cols].round(4).to_string(index=False))
    else:
        print("\n  NO variant is both net-positive and significant under the")
        print("  matched null.")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()
