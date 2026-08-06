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

from src.config import INSTRUMENTS, SESSIONS, RR_LEVELS, OUTPUTS_DIR, SCHEMA_1M
from src.data_layer import (
    ensure_data, ensure_daily, _compute_enrichment, _cache_path, _roll_tag,
    vendor_boundary_diagnostics,
)
from src.range_builder import build_session_days, SessionDay
from src.entry_detector import detect_entries
from src.trade_sim import simulate_trade
from src.journal import build_row
from src.regime_sampler import select_windows, filter_to_window
from src.null_calibrator import (
    build_r_ticks_map, enrich_summary_with_null, write_null_artifacts,
)
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


def _preflight(api_key: str | None) -> None:
    """
    Resolve every cache path BEFORE doing any work, and fail fast if a
    download is required but no API key is present.

    Without this, ensure_data only raises when it REACHES a missing cache.
    Instrument order is ES, NQ, RTY, GC..., so a missing key would load three
    instruments and then die several minutes in. This makes the run plan
    explicit up front and cheap to abort.
    """
    reuse, download = [], []
    for sym in INSTRUMENTS:
        p = _cache_path(sym, SCHEMA_1M)
        (reuse if p.exists() else download).append((sym, p.name))

    log.info("── data plan ──────────────────────────────────────────────")
    for sym, name in reuse:
        log.info("  reuse     %-5s  %s", sym, name)
    for sym, name in download:
        log.info("  DOWNLOAD  %-5s  %s  (roll=%s)", sym, name, _roll_tag(sym))

    if download and not api_key:
        names = ", ".join(s for s, _ in download)
        raise SystemExit(
            f"\nDATABENTO_API_KEY is not set, but {len(download)} symbol(s) need "
            f"downloading: {names}\n\n"
            f"  PowerShell:  $env:DATABENTO_API_KEY = \"db-...\"\n\n"
            f"Run cost_check.py first to see the exact spend before downloading."
        )

    if download:
        log.info("%d symbol(s) will download; %d reused from cache.",
                 len(download), len(reuse))
    else:
        log.info("All %d symbols cached — no Databento spend.", len(reuse))


def run() -> None:
    t0 = time.perf_counter()
    api_key = _get_api_key()
    _preflight(api_key)

    # ── 1. Load / build data ───────────────────────────────────────────────
    data: dict[str, pd.DataFrame] = {}
    daily: dict[str, pd.DataFrame] = {}
    boundary_diagnostics: list[pd.DataFrame] = []
    for sym, instr in INSTRUMENTS.items():
        df_1m   = ensure_data(sym, api_key)
        df_1d   = ensure_daily(sym, api_key)
        df_enr  = _compute_enrichment(df_1m, df_1d,
                                      tick_size=instr["tick_size"])
        data[sym]  = df_enr
        daily[sym] = df_1d
        diag = vendor_boundary_diagnostics(df_1m, sym)
        if not diag.empty:
            boundary_diagnostics.append(diag)
        log.info("%s (%s): %d bars enriched",
                 sym, instr.get("asset_class", "?"), len(df_enr))

    # ── 2. Select regime windows ───────────────────────────────────────────
    windows_by_symbol = {
        sym: select_windows(df["timestamp"].min().date(),
                            df["timestamp"].max().date())
        for sym, df in data.items()
    }
    window_lookup = {
        (sym, window.index): window
        for sym, windows in windows_by_symbol.items() for window in windows
    }
    log.info("Selected realised regime counts: %s",
             {sym: len(windows) for sym, windows in windows_by_symbol.items()})

    # ── 3. Build per-window session-day lists ─────────────────────────────
    # session_days_map[(sym, sess)] → flat list across ALL windows (for null calibration)
    session_days_map: dict[tuple, list[SessionDay]] = {}
    window_session_days: dict[tuple, list[SessionDay]] = {}  # (window_idx, sym, sess) → days

    for sym, df_full in data.items():
        applicable_sessions = INSTRUMENTS[sym].get("sessions", list(SESSIONS.keys()))
        for sess_name in applicable_sessions:
            for w in windows_by_symbol[sym]:
                df_w = filter_to_window(df_full, w)
                if len(df_w) == 0:
                    continue
                sdays = build_session_days(df_w, sym, sess_name)
                for sd in sdays:
                    sd.regime_window = w.index
                window_session_days[(w.index, sym, sess_name)] = sdays
                key = (sym, sess_name)
                session_days_map.setdefault(key, []).extend(sdays)

    # ── 4. Sweep: detect entries → simulate → build journal ───────────────
    all_rows: list[dict] = []

    for (w_idx, sym, sess_name), sdays in window_session_days.items():
        w = window_lookup[(sym, w_idx)]
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
    # The matched-stop null needs the observed r_ticks distribution per variant
    # family: random entry timing is compared at the SAME stop distance, so the
    # test isolates timing rather than stop width. Without this the comparator
    # draws much tighter stops, inflates same-bar SL/TP collisions 140x, and
    # becomes anti-conservative.
    log.info("Building observed r_ticks map for matched null...")
    r_map = build_r_ticks_map(trade_log)
    log.info("Running null calibration (matched-stop random-entry benchmark)...")
    summary = enrich_summary_with_null(
        summary, session_days_map, r_map, trade_log=trade_log)
    write_null_artifacts(summary, _OUT)
    if boundary_diagnostics:
        pd.concat(boundary_diagnostics, ignore_index=True).to_csv(
            _OUT / "vendor_boundary_diagnostics.csv", index=False)

    # ── 7. Write outputs ──────────────────────────────────────────────────
    write_report(summary, regime_summary, trade_log, _OUT)

    elapsed = time.perf_counter() - t0
    log.info("Done in %.1fs  →  %s", elapsed, _OUT.resolve())


if __name__ == "__main__":
    run()
