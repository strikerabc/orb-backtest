"""
recover_report.py — regenerate report artefacts from existing parquets.

Use when main.py completed the sweep but crashed during report writing
(e.g. summary.csv locked open in Excel -> PermissionError).

Reads:   outputs/trade_log.parquet, outputs/summary.parquet
Writes:  outputs/summary.csv
         outputs/regime_summary.parquet, outputs/regime_summary.csv
         outputs/report.md

Does NOT re-run the sweep or null calibration. summary.parquet is already
null-enriched, so it is reused verbatim. regime_summary is recomputed from
the trade log (a groupby -- cheap).

Usage:
    python recover_report.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import OUTPUTS_DIR
from src.stats import compute_regime_summary
from src.report import _write_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orb.recover")

_OUT = _ROOT / OUTPUTS_DIR


def _safe_write(label: str, fn) -> bool:
    """Run a write fn, converting a lock into a warning instead of a crash."""
    try:
        fn()
        log.info("wrote %s", label)
        return True
    except PermissionError:
        log.error("LOCKED, skipped: %s  (close it in Excel and re-run)", label)
        return False
    except Exception as exc:
        log.error("FAILED %s: %s", label, exc)
        return False


def main() -> None:
    tl_path = _OUT / "trade_log.parquet"
    sm_path = _OUT / "summary.parquet"

    for p in (tl_path, sm_path):
        if not p.exists():
            log.error("missing %s -- cannot recover, re-run main.py", p)
            sys.exit(1)

    log.info("loading %s", tl_path)
    trade_log = pd.read_parquet(tl_path)
    log.info("trade log: %d rows x %d cols", len(trade_log), trade_log.shape[1])

    log.info("loading %s", sm_path)
    summary = pd.read_parquet(sm_path)
    log.info("summary: %d variants x %d cols", len(summary), summary.shape[1])

    has_null = "null_p_value" in summary.columns
    has_ci   = "ci_lo_95" in summary.columns
    log.info("null calibration present: %s | bootstrap CI present: %s",
             has_null, has_ci)

    # instrument coverage sanity check -- confirms this is the new sweep
    if "instrument" in summary.columns:
        instruments = sorted(summary["instrument"].unique())
        log.info("instruments in summary (%d): %s",
                 len(instruments), ", ".join(map(str, instruments)))

    # ── recompute regime summary from the trade log ────────────────────────
    log.info("recomputing regime summary...")
    regime_summary = compute_regime_summary(trade_log)
    log.info("regime summary: %d rows", len(regime_summary))

    # ── write artefacts (cheap + valuable first) ───────────────────────────
    ok = []
    ok.append(_safe_write("summary.csv",
              lambda: summary.to_csv(_OUT / "summary.csv", index=False)))
    ok.append(_safe_write("regime_summary.parquet",
              lambda: regime_summary.to_parquet(_OUT / "regime_summary.parquet", index=False)))
    ok.append(_safe_write("regime_summary.csv",
              lambda: regime_summary.to_csv(_OUT / "regime_summary.csv", index=False)))
    ok.append(_safe_write("report.md",
              lambda: _write_markdown(summary, regime_summary, trade_log, _OUT)))

    n_ok = sum(ok)
    log.info("done: %d/%d artefacts written -> %s", n_ok, len(ok), _OUT)
    if n_ok < len(ok):
        log.warning("some writes were skipped; close the locked files and re-run")
        sys.exit(2)


if __name__ == "__main__":
    main()
