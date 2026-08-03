"""
main.py — top-level orchestration for the ORB backtest sweep.

Usage:
    python main.py

Environment:
    DATABENTO_API_KEY  — required on first run to download 2019-2022 data.
                         Not needed once data/ cache is populated.

Outputs (in outputs/):
    trade_log.parquet / trade_log.csv
    summary.parquet / summary.csv
    regime_summary.parquet / regime_summary.csv
    report.md
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

from src.config import INSTRUMENTS, SESSIONS, RR_LEVELS, OUTPUTS_DIR
from src.data_layer import ensure_data, ensure_daily, _compute_enrichment
from src.range_builder import build_session_days, SessionDay
from src.entry_detector import detect_entries
from src.trade_sim import simulate_trade
from src.journal import build_row
from src.regime_sampler import select_windows, filter_to_window
from src.null_calibrator import enrich_summary_with_null
from src.stats import compute_summary, compute_regime_summary
from src.report import write_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orb.main")
_OUT = Path(OUTPUTS_DIR)


def _get_api_key() -> str | None:
    return os.environ.get("DATABENTO_API_KEY")


def run() -> None:
    t0 = time.perf_counter()
    api_key = _get_api_key()

    # ── 1. Load / build data ───────────────────────────────────────────────
    data: dict[str, pd.DataFrame] = {}
    daily: dict[str, pd.DataFrame] = {}
    for sym, instr in INSTRUMENTS.items():
        df_1m   = ensure_data(sym, api_key)
        df_1d   = ensure_daily(sym, api_key)
        df_enr  = _compute_enrichment(df_1m, df_1d,
                                      tick_size=instr["tick_size"])
        data[sym]  = df_enr
        daily[sym] = df_1d
        log.info("%s (%s): %d bars enriched",
                 sym, instr.get("asset_class", "?"), len(df_enr))

    # ── 2. Select regime windows ───────────────────────────────────────────
    # Use the union of all data to determine span
    all_ts   = pd.concat([d["timestamp"] for d in data.values()])
    data_start = all_ts.min().date()
    data_end   = all_ts.max().date()
    windows  = select_windows(data_start, data_end)
    log.info("Selected %d regime windows", len(windows))

    # ── 3. Build per-window session-day lists ─────────────────────────────
    # session_days_map[(sym, sess)] → flat list across ALL windows (for null calibration)
    session_days_map: dict[tuple, list[SessionDay]] = {}
    window_session_days: dict[tuple, list[SessionDay]] = {}  # (window_idx, sym, sess) → days

    for sym, df_full in data.items():
        applicable_sessions = INSTRUMENTS[sym].get("sessions", list(SESSIONS.keys()))
        for sess_name in applicable_sessions:
            for w in windows:
                df_w = filter_to_window(df_full, w)
                if len(df_w) == 0:
                    continue
                sdays = build_session_days(df_w, sym, sess_name)
                window_session_days[(w.index, sym, sess_name)] = sdays
                key = (sym, sess_name)
                session_days_map.setdefault(key, []).extend(sdays)

    # ── 4. Sweep: detect entries → simulate → build journal ───────────────
    all_rows: list[dict] = []

    for (w_idx, sym, sess_name), sdays in window_session_days.items():
        w = windows[w_idx]
        log.info("Window %02d  %s %s  %d session-days",
                 w_idx, sym, sess_name, len(sdays))
        for sd in sdays:
            signals = detect_entries(sd)
            for es in signals:
                trades = simulate_trade(es, sd, RR_LEVELS)
                for tr in trades:
                    row = build_row(es, tr, sd, w,
                                    sd.bars_h, sd.bars_l)
                    all_rows.append(row)

    if not all_rows:
        log.error("No trade rows generated. Check data and config.")
        return

    trade_log = pd.DataFrame(all_rows)
    log.info("Trade log: %d rows", len(trade_log))

    # ── 5. Compute statistics ─────────────────────────────────────────────
    summary        = compute_summary(trade_log)
    regime_summary = compute_regime_summary(trade_log)

    # ── 6. Null calibration ───────────────────────────────────────────────
    log.info("Running null calibration (bootstrap + random-entry benchmark)...")
    summary = enrich_summary_with_null(summary, session_days_map)

    # ── 7. Write outputs ──────────────────────────────────────────────────
    write_report(summary, regime_summary, trade_log, _OUT)

    elapsed = time.perf_counter() - t0
    log.info("Done in %.1fs  →  %s", elapsed, _OUT.resolve())


if __name__ == "__main__":
    run()
