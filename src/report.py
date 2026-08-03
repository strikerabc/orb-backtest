"""
report.py — write summary tables and plain-text/markdown report.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("orb.report")


def write_report(
    summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    trade_log: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write all output artefacts: parquet, CSV, markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── trade log ──────────────────────────────────────────────────────────
    trade_log.to_parquet(output_dir / "trade_log.parquet", index=False)
    trade_log.to_csv(output_dir / "trade_log.csv", index=False)
    log.info("Trade log: %d rows → %s", len(trade_log), output_dir / "trade_log.csv")

    # ── summary tables ─────────────────────────────────────────────────────
    summary.to_parquet(output_dir / "summary.parquet", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    regime_summary.to_parquet(output_dir / "regime_summary.parquet", index=False)
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)

    # ── markdown report ────────────────────────────────────────────────────
    _write_markdown(summary, regime_summary, trade_log, output_dir)


def _write_markdown(summary: pd.DataFrame, regime_summary: pd.DataFrame,
                    trade_log: pd.DataFrame, output_dir: Path) -> None:
    valid   = trade_log[trade_log["exit_reason"].isin(["TP", "SL", "TIME"])]
    n_total = len(valid)
    n_atr   = int(trade_log["atr_exceeds_cap"].sum()) if "atr_exceeds_cap" in trade_log else 0
    n_amb   = int(trade_log["same_bar_ambiguous"].sum()) if "same_bar_ambiguous" in trade_log else 0

    lines = [
        "# ORB Backtest — Results Summary",
        f"\n**Total valid trades:** {n_total:,}  |  "
        f"**ATR-invalidated (flagged):** {n_atr:,}  |  "
        f"**Same-bar ambiguous:** {n_amb:,}",
        "",
        "---",
        "## Top 20 Variants by Gross Expectancy",
        "",
    ]

    if not summary.empty and "expectancy_gross_r" in summary.columns:
        top = (summary
               .sort_values("expectancy_gross_r", ascending=False)
               .head(20)
               [["instrument","session","range_minutes","entry_mode","closure_tf",
                 "direction","rr","trade_count","win_rate",
                 "expectancy_gross_r","expectancy_net_r","profit_factor",
                 "max_drawdown_r","ci_lo_95","ci_hi_95"]]
               .round(4))
        lines.append(top.to_markdown(index=False))
    else:
        lines.append("*(summary not available)*")

    lines += [
        "",
        "---",
        "## Bottom 20 Variants by Gross Expectancy",
        "",
    ]
    if not summary.empty and "expectancy_gross_r" in summary.columns:
        bot = (summary
               .sort_values("expectancy_gross_r", ascending=True)
               .head(20)
               [["instrument","session","range_minutes","entry_mode","closure_tf",
                 "direction","rr","trade_count","win_rate",
                 "expectancy_gross_r","expectancy_net_r","profit_factor",
                 "max_drawdown_r","ci_lo_95","ci_hi_95"]]
               .round(4))
        lines.append(bot.to_markdown(index=False))

    lines += [
        "",
        "---",
        "## Regime Sensitivity (expectancy std across windows)",
        "",
    ]
    if not regime_summary.empty and "expectancy_gross_r" in regime_summary.columns:
        keys = ["instrument","session","range_minutes","entry_mode","closure_tf","direction","rr"]
        avail = [k for k in keys if k in regime_summary.columns]
        stability = (regime_summary
                     .groupby(avail, observed=True)["expectancy_gross_r"]
                     .agg(["mean","std","min","max","count"])
                     .rename(columns={"mean":"mean_r","std":"std_r","min":"min_r",
                                      "max":"max_r","count":"n_windows"})
                     .sort_values("std_r", ascending=True)
                     .head(20)
                     .round(4)
                     .reset_index())
        lines.append(stability.to_markdown(index=False))

    lines += [
        "",
        "---",
        "## Notes",
        "- All R values are gross unless labelled net.",
        "- `atr_exceeds_cap` flag marks trades where TP > 2.5 × 4h ATR (simulated anyway; filter in analysis).",
        "- `same_bar_ambiguous` flag: SL and TP both hit within entry bar — SL assumed first (conservative).",
        "- Net R uses default cost model: 1 tick round-trip slippage + $2.50/side commission (see config.py).",
        "- CI is 95% block-bootstrap (block=5 days). Top performers may be inflated by selection bias.",
        "- Null calibration p-values show fraction of random-entry runs beating each variant.",
    ]

    md_path = output_dir / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report → %s", md_path)
