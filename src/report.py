"""
report.py — write summary tables and plain-text/markdown report.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import json

from src.config import (
    MIN_TRADES_FOR_RANKING, SLIPPAGE_PROVENANCE,
    SLIPPAGE_TICKS_BY_SYMBOL_SESSION,
)


def _verdict_banner(output_dir: Path) -> list[str]:
    """
    Lead the report with the out-of-sample verdict, if one exists.

    Without this the report opens with a Top-20 ranked by gross expectancy, and
    a reader would reasonably conclude those variants are good. They are
    in-sample selection artefacts: the holdout test found the survivor set at or
    below the multiple-comparisons chance rate and indistinguishable from a coin
    flip out-of-sample.

    Read from holdout_verdict.json rather than hardcoded so the report always
    carries the LATEST holdout result instead of a stale claim.
    """
    vpath = output_dir / "holdout_verdict.json"
    if not vpath.exists():
        return [
            "> **NO OUT-OF-SAMPLE TEST HAS BEEN RUN.**",
            "> Every table below is in-sample. With thousands of variants swept,",
            "> in-sample rank is not evidence of edge. Run `test_holdout.py`.",
            "",
        ]
    try:
        v = json.loads(vpath.read_text(encoding="utf-8"))
    except Exception:
        return []

    is_null = v.get("verdict", "").startswith("NO EDGE")
    head = "## ⚠️ VERDICT: " + v.get("verdict", "unknown")
    lines = [head, ""]

    if is_null:
        lines += [
            "**The ranked tables below are in-sample and are best read as "
            "selection artefacts.** Two independent checks say so:",
            "",
        ]
    lines += [
        f"**1. In-sample survivors are at or below the chance rate.** "
        f"{v['survivor_families']} of {v['families_rankable']:,} signal families "
        f"were net-positive and `null_p < 0.05` "
        f"({v['survivor_pct_of_families']}%), against ~{v['expected_fp_at_5pct']} "
        f"expected from multiple comparisons alone at a 5% threshold "
        f"(5.0%). "
        + ("So the survivor set cannot be distinguished from noise before the "
           "holdout is even consulted." if v.get("below_chance_rate") else ""),
        "",
        f"**2. Out-of-sample ({v['holdout_start']} onward, never used for "
        f"selection).** Of {v['holdout_families_tested']} families, "
        f"{v['holdout_net_positive']} stayed net-positive "
        f"({v['holdout_net_positive_pct']}% — chance is 50%). "
        f"Trade-weighted mean holdout net R = "
        f"**{v['holdout_trade_weighted_net_r']:+.4f}**, bootstrap 95% CI "
        f"[{v['holdout_ci_lo']:+.4f}, {v['holdout_ci_hi']:+.4f}]"
        + (" — includes zero." if v.get("ci_includes_zero") else "."),
        "",
        f"Variant counts overstate findings ~1.9x: all six RR levels of one "
        f"entry signal share the same entries and differ only in exit "
        f"placement, so {v['survivor_variants']} \"variants\" are "
        f"{v['survivor_families']} independent families.",
        "",
        f"*Power caveat:* median {v['median_holdout_trades']} holdout trades per "
        f"family at per-trade sd ~1.0 R gives SE ~"
        f"{1.0/max(v['median_holdout_trades'],1)**0.5:.2f} R, so individual "
        f"rows are uninformative — only the pooled figure carries weight. "
        f"A null result does not prove no edge exists; it establishes that "
        f"this sweep did not find one.",
        "",
        "---",
        "",
    ]
    return lines

log = logging.getLogger("orb.report")

# Columns shown in ranked tables, in display order. Filtered to those present
# so the report still renders if null calibration was skipped.
_RANK_COLS = [
    "instrument", "session", "range_minutes", "entry_mode", "closure_tf",
    "direction", "rr", "trade_count", "win_rate",
    "expectancy_gross_r", "expectancy_net_r", "profit_factor",
    "max_drawdown_r", "ci_lo_95", "ci_hi_95", "null_p_value",
]


def _cols(df: pd.DataFrame) -> list[str]:
    return [c for c in _RANK_COLS if c in df.columns]


def _safe_write(label: str, fn) -> bool:
    """
    Run a write fn, converting a file lock into a warning instead of a crash.

    A single file held open by Excel used to abort write_report entirely,
    losing every artefact after it. Each write is now independent.
    """
    try:
        fn()
        log.info("wrote %s", label)
        return True
    except PermissionError:
        log.error("LOCKED, skipped: %s  (close it and run recover_report.py)", label)
        return False
    except Exception as exc:
        log.error("FAILED %s: %s", label, exc)
        return False


def write_report(
    summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    trade_log: pd.DataFrame,
    output_dir: Path,
    write_trade_log_csv: bool = False,
) -> None:
    """
    Write all output artefacts: parquet, CSV, markdown.

    Order matters: cheap, high-value artefacts are written FIRST so a
    failure late in the sequence cannot cost the summary or the report.

    write_trade_log_csv: the trade-log CSV is ~1.2 GB at 3.4M rows and
        exceeds Excel's 1,048,576-row limit, so it cannot be opened there
        anyway. Parquet holds identical data at ~5% of the size. Off by
        default; enable only if an external tool needs raw CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── summary tables (small, most valuable) ──────────────────────────────
    _safe_write("summary.parquet",
                lambda: summary.to_parquet(output_dir / "summary.parquet", index=False))
    _safe_write("summary.csv",
                lambda: summary.to_csv(output_dir / "summary.csv", index=False))
    _safe_write("regime_summary.parquet",
                lambda: regime_summary.to_parquet(output_dir / "regime_summary.parquet", index=False))
    _safe_write("regime_summary.csv",
                lambda: regime_summary.to_csv(output_dir / "regime_summary.csv", index=False))

    # ── markdown report ────────────────────────────────────────────────────
    _safe_write("report.md",
                lambda: _write_markdown(summary, regime_summary, trade_log, output_dir))

    # ── trade log (large; parquet is the canonical artefact) ───────────────
    _safe_write("trade_log.parquet",
                lambda: trade_log.to_parquet(output_dir / "trade_log.parquet", index=False))
    log.info("Trade log: %d rows → %s", len(trade_log), output_dir / "trade_log.parquet")

    if write_trade_log_csv:
        log.info("Writing trade_log.csv (~1.2 GB, several minutes)...")
        _safe_write("trade_log.csv",
                    lambda: trade_log.to_csv(output_dir / "trade_log.csv", index=False))
    else:
        log.info("Skipped trade_log.csv (write_trade_log_csv=False; parquet holds same data)")


def _write_markdown(summary: pd.DataFrame, regime_summary: pd.DataFrame,
                    trade_log: pd.DataFrame, output_dir: Path) -> None:
    valid   = trade_log[trade_log["exit_reason"].isin(["TP", "SL", "TIME"])]
    n_total = len(valid)
    n_atr   = int(trade_log["atr_exceeds_cap"].sum()) if "atr_exceeds_cap" in trade_log else 0
    n_amb   = int(trade_log["same_bar_ambiguous"].sum()) if "same_bar_ambiguous" in trade_log else 0

    has_summary = (not summary.empty) and ("expectancy_gross_r" in summary.columns)

    # Ranked tables use only variants with enough trades to be meaningful.
    if has_summary and "trade_count" in summary.columns:
        rankable = summary[summary["trade_count"] >= MIN_TRADES_FOR_RANKING]
        n_excl   = len(summary) - len(rankable)
    else:
        rankable, n_excl = summary, 0

    lines = ["# ORB Backtest — Results Summary", ""]
    lines += _verdict_banner(output_dir)
    lines += [
        f"**Total valid trades:** {n_total:,}  |  "
        f"**ATR-invalidated (flagged):** {n_atr:,}  |  "
        f"**Same-bar ambiguous:** {n_amb:,}",
        "",
        f"**Variants:** {len(summary):,} total  |  "
        f"{len(rankable):,} rankable (>= {MIN_TRADES_FOR_RANKING} trades)  |  "
        f"{n_excl:,} excluded as under-sampled",
        "",
    ]

    # ── data coverage ──────────────────────────────────────────────────────
    if "instrument" in trade_log.columns:
        lines += ["---", "## Data Coverage per Instrument", "",
                  "Variants with too few trades indicate a data problem, not a",
                  "strategy result. Check bar density before reading any row below.",
                  ""]
        cov = (trade_log.groupby("instrument", observed=True)
                        .agg(rows=("exit_reason", "size"))
                        .reset_index())
        if has_summary and "trade_count" in summary.columns:
            tc = (summary.groupby("instrument", observed=True)["trade_count"]
                         .agg(variants="size", median_trades="median",
                              max_trades="max")
                         .reset_index())
            cov = cov.merge(tc, on="instrument", how="outer")
            cov["status"] = np.where(
                cov["median_trades"] < MIN_TRADES_FOR_RANKING,
                "UNDER-SAMPLED - investigate data", "ok")
        lines.append(cov.sort_values("rows").to_markdown(index=False))
        lines.append("")

    # ── cost-model provenance ──────────────────────────────────────────────
    lines += ["---", "## Cost-Model Provenance", "",
              "Slippage is measured, not assumed. Values below are round-trip",
              "ticks from `bbo-1m` quotes (2020-06 / 2022-06 / 2024-06),",
              "entry-weighted by the empirical distribution of entry minutes.",
              "",
              "`provisional` means the quote data is correct but the entry-timing",
              "weights came from the pre-roll-fix trade log, so those values use",
              "max(entry-weighted, session median) to avoid understating cost.",
              ""]
    prov_rows = []
    for sym in sorted(SLIPPAGE_PROVENANCE):
        per_sess = {s: v for (sy, s), v in
                    SLIPPAGE_TICKS_BY_SYMBOL_SESSION.items() if sy == sym}
        prov_rows.append({
            "instrument": sym,
            "provenance": SLIPPAGE_PROVENANCE[sym],
            "slippage_ticks": ", ".join(f"{s}={v:g}" for s, v in per_sess.items())
                              or "(fallback)",
        })
    lines.append(pd.DataFrame(prov_rows).to_markdown(index=False))
    lines += ["",
              "Measured spread is a **floor** on execution cost: it excludes",
              "market impact, and ORB entries cross a book in motion.",
              ""]

    lines += ["---",
              f"## Top 20 Variants by Gross Expectancy (>= {MIN_TRADES_FOR_RANKING} trades)",
              ""]

    if has_summary and not rankable.empty:
        top = (rankable
               .sort_values("expectancy_gross_r", ascending=False)
               .head(20)[_cols(rankable)]
               .round(4))
        lines.append(top.to_markdown(index=False))
    elif has_summary:
        lines.append(f"*(no variant reached {MIN_TRADES_FOR_RANKING} trades)*")
    else:
        lines.append("*(summary not available)*")

    lines += [
        "",
        "---",
        f"## Bottom 20 Variants by Gross Expectancy (>= {MIN_TRADES_FOR_RANKING} trades)",
        "",
    ]
    if has_summary and not rankable.empty:
        bot = (rankable
               .sort_values("expectancy_gross_r", ascending=True)
               .head(20)[_cols(rankable)]
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
        f"- **Ranked tables require >= {MIN_TRADES_FOR_RANKING} trades.** Without this filter, "
        "variants that fired once and won sort to the top with `win_rate=1.0`, "
        "`profit_factor=9999` (sentinel for zero losing trades) and a degenerate "
        "`ci_lo == ci_hi` (bootstrap returns the point estimate twice when "
        "`trade_count < block`). Those are small-sample arithmetic, not edges.",
        "- summary.parquet/csv contain ALL variants including under-sampled ones; "
        "the filter applies only to ranked tables here.",
        "- Check the coverage table before trusting any instrument: a low median "
        "trade count means missing bars upstream, not a strategy finding.",
        "- All R values are gross unless labelled net.",
        "- `atr_exceeds_cap` flag marks trades where TP > 2.5 × 4h ATR (simulated anyway; filter in analysis).",
        "- `same_bar_ambiguous` flag: SL and TP both hit within entry bar — SL assumed first (conservative).",
        "- Net R cost model: MEASURED per-(instrument, session) slippage "
        "+ $2.50/side commission (see config.py). Slippage came from bbo-1m "
        "quotes sampled 2020-06 / 2022-06 / 2024-06, entry-weighted by the "
        "empirical distribution of entry minutes. It is NOT a flat 1 tick: "
        "measured values range from 1.00 (ES, ZN, CL, 6E, 6J) to 53.14 ticks "
        "(ETH TOK).",
        "- **Measured spread is a FLOOR on execution cost.** It excludes "
        "market impact, and ORB entries cross a book in motion. Net R is "
        "therefore optimistic by an unquantified margin.",
        "- CI is 95% block-bootstrap (block=5 days). Top performers may be inflated by selection bias.",
        "- Null calibration p-values show fraction of random-entry runs beating each variant.",
    ]

    md_path = output_dir / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report → %s", md_path)
