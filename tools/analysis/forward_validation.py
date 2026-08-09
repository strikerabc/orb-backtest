"""Frozen NQ/TOK candidate revalidation and prospective-data collector."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs" / "forward_validation"
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from src import config

config.USE_LOCAL_DATA = True

from src.config import DATASET, INSTRUMENTS, SCHEMA_1M, STYPE
from src.entry_detector import detect_entries
from src.filters import trade_eligibility
from src.range_builder import build_session_days
from src.stats import _block_bootstrap_ci
from src.trade_sim import simulate_trade

MANIFEST = WORK / "protocol.json"
COST_PATH = WORK / "cost_estimate.json"
INCREMENT = WORK / "NQ_forward_1m.parquet"
RESULT = WORK / "revalidation_result.json"
TRADES = WORK / "revalidation_trades.parquet"
REPORT = WORK / "report.md"
LEDGER = WORK / "download_ledger.jsonl"
PROSPECTIVE_PROTOCOL = WORK / "prospective_protocol.json"
PROSPECTIVE_TRADES = WORK / "prospective_trades.parquet"
PROSPECTIVE_STATUS = WORK / "prospective_status.json"
PROSPECTIVE_RESULT = WORK / "prospective_result.json"
PROSPECTIVE_REPORT = WORK / "prospective_report.md"

BASE_1M = ROOT / "data" / "NQ_c_mixed_1m.parquet"
BASE_1D = ROOT / "data" / "NQ_1d.parquet"
SUMMARY = ROOT / "outputs" / "summary.parquet"
REVALIDATION_START = "2026-03-01"
MAX_AUTHORIZED_SPEND = 32.46
PRIMARY_RR = 0.75
SECONDARY_RR = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _safe_end() -> str:
    # Databento's end is exclusive; two days avoid the historical-license edge.
    return (pd.Timestamp.now(tz="UTC").normalize()
            - pd.Timedelta(days=2)).strftime("%Y-%m-%d")


def _base_end() -> pd.Timestamp:
    values = pd.read_parquet(BASE_1M, columns=["timestamp"])
    return pd.to_datetime(values["timestamp"], utc=True).max()


def _download_start() -> str:
    if INCREMENT.exists():
        values = pd.read_parquet(INCREMENT, columns=["timestamp"])
        latest = pd.to_datetime(values["timestamp"], utc=True).max()
    else:
        latest = _base_end()
    return (latest.floor("D") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def freeze() -> dict:
    if MANIFEST.exists():
        protocol = json.loads(MANIFEST.read_text(encoding="utf-8"))
        print(f"Protocol already frozen: {MANIFEST}")
        return protocol

    summary = pd.read_parquet(SUMMARY)
    selected = summary[
        (summary["instrument"] == "NQ")
        & (summary["session"] == "TOK")
        & (summary["range_minutes"] == 15)
        & (summary["entry_mode"] == "II")
        & (summary["closure_tf"] == 1)
        & (summary["direction"] == "short")
        & summary["rr"].isin([PRIMARY_RR, SECONDARY_RR])
    ].sort_values("rr")
    if len(selected) != 2:
        raise RuntimeError(f"Expected two frozen candidate rows, found {len(selected)}")

    protocol = {
        "protocol_version": 1,
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_sha": _git_sha(),
        "selection_summary_sha256": _sha256(SUMMARY),
        "candidate": {
            "instrument": "NQ",
            "session": "TOK",
            "range_minutes": 15,
            "entry_mode": "II",
            "closure_tf": 1,
            "direction": "short",
            "primary_rr": PRIMARY_RR,
            "secondary_rr": SECONDARY_RR,
        },
        "selection_snapshot": selected[[
            "rr", "trade_count", "expectancy_net_r", "null_p_broad",
            "null_p_matched", "p_adj_maxT", "q_fdr",
        ]].to_dict("records"),
        "selection_windows_end": "2026-02-28",
        "revalidation_start_local_date": REVALIDATION_START,
        "base_1m_last_timestamp_utc": _base_end().isoformat(),
        "interpretation": (
            "Historical revalidation only: this calendar region was inspected by "
            "an earlier research pass and is not a pristine program-level holdout."
        ),
        "primary_endpoint": "mean eligible net_r at RR 0.75",
        "secondary_endpoint": "mean eligible net_r at RR 1.00 (descriptive unless primary confirms)",
        "inference": "95% percentile CI from contiguous five-session-day block bootstrap",
        "minimum_eligible_trades": 100,
        "decision_rule": {
            "confirm": "n >= 100 and primary CI lower bound > 0",
            "falsify": "n >= 100 and primary CI upper bound < 0",
            "otherwise": "inconclusive",
        },
        "rules_frozen": [
            "entry and trade simulation at git_sha",
            "current measured slippage and commission",
            "central trade_eligibility filters",
            "no parameter reselection using revalidation outcomes",
        ],
        "known_vendor_conditions": [
            {"date": "2026-05-09", "condition": "missing", "note": "Saturday"},
            {"date": "2026-05-24", "condition": "degraded", "note": "Sunday before US Memorial Day"},
        ],
    }
    MANIFEST.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Frozen protocol: {MANIFEST}")
    print(f"Protocol SHA256: {_sha256(MANIFEST)}")
    return protocol


def _client():
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY is not set")
    import databento as db
    return db.Historical(key=key)


def estimate_cost() -> dict:
    freeze()
    start, end = _download_start(), _safe_end()
    if start >= end:
        estimate = {"start": start, "end_exclusive": end, "cost_usd": 0.0,
                    "status": "up-to-date"}
    else:
        cost = float(_client().metadata.get_cost(
            dataset=DATASET,
            symbols=[INSTRUMENTS["NQ"]["continuous_symbol"]],
            schema=SCHEMA_1M,
            start=start,
            end=end,
            stype_in=STYPE,
        ))
        estimate = {"start": start, "end_exclusive": end, "cost_usd": cost,
                    "status": "quoted"}
    estimate["quoted_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    estimate["authorized_limit_usd"] = MAX_AUTHORIZED_SPEND
    COST_PATH.write_text(json.dumps(estimate, indent=2), encoding="utf-8")
    print(json.dumps(estimate, indent=2))
    return estimate


def _normalize_download(data) -> pd.DataFrame:
    frame = data.to_df()
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    frame = frame.rename(columns={
        "ts_event": "timestamp", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    if "timestamp" not in frame:
        raise KeyError(f"Downloaded frame has no timestamp: {list(frame.columns)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if "symbol" in frame:
        frame = frame.rename(columns={"symbol": "_contract"})
    elif "_contract" not in frame:
        frame["_contract"] = INSTRUMENTS["NQ"]["continuous_symbol"]
    frame["_source"] = "databento_forward"
    columns = ["timestamp", "open", "high", "low", "close", "volume",
               "_contract", "_source"]
    return frame[columns].sort_values("timestamp").drop_duplicates(
        "timestamp", keep="last").reset_index(drop=True)


def _ledger_spend() -> float:
    if not LEDGER.exists():
        return 0.0
    return float(sum(
        float(json.loads(line).get("cost_usd", 0.0))
        for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()
    ))


def download() -> dict:
    estimate = estimate_cost()
    cost = float(estimate["cost_usd"])
    spent = _ledger_spend()
    if spent + cost > MAX_AUTHORIZED_SPEND:
        raise RuntimeError(
            f"Cumulative ${spent + cost:.2f} exceeds authorized ${MAX_AUTHORIZED_SPEND:.2f}")
    if estimate["status"] == "up-to-date":
        return estimate
    client = _client()
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[INSTRUMENTS["NQ"]["continuous_symbol"]],
        schema=SCHEMA_1M,
        start=estimate["start"],
        end=estimate["end_exclusive"],
        stype_in=STYPE,
    )
    incoming = _normalize_download(data)
    if INCREMENT.exists():
        prior = pd.read_parquet(INCREMENT)
        incoming = pd.concat([prior, incoming], ignore_index=True)
        incoming["timestamp"] = pd.to_datetime(incoming["timestamp"], utc=True)
        incoming = incoming.sort_values("timestamp").drop_duplicates(
            "timestamp", keep="last").reset_index(drop=True)
    incoming.to_parquet(INCREMENT, index=False)
    estimate.update({
        "status": "downloaded",
        "bars": len(incoming),
        "first_timestamp_utc": incoming["timestamp"].min().isoformat(),
        "last_timestamp_utc": incoming["timestamp"].max().isoformat(),
        "increment_sha256": _sha256(INCREMENT),
        "vendor_condition_warnings": [
            {"date": "2026-05-09", "condition": "missing"},
            {"date": "2026-05-24", "condition": "degraded"},
        ],
    })
    COST_PATH.write_text(json.dumps(estimate, indent=2), encoding="utf-8")
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "start": estimate["start"], "end_exclusive": estimate["end_exclusive"],
            "cost_usd": cost, "bars_in_cumulative_increment": len(incoming),
            "increment_sha256": estimate["increment_sha256"],
        }) + "\n")
    print(json.dumps(estimate, indent=2))
    return estimate


def _combined_1m() -> pd.DataFrame:
    base = pd.read_parquet(BASE_1M)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    frames = [base]
    if INCREMENT.exists():
        addition = pd.read_parquet(INCREMENT)
        addition["timestamp"] = pd.to_datetime(addition["timestamp"], utc=True)
        frames.append(addition)
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values("timestamp", kind="mergesort").drop_duplicates(
        "timestamp", keep="last").reset_index(drop=True)


def _candidate_rows(start_date_text: str = REVALIDATION_START) -> pd.DataFrame:
    from src.data_layer import _compute_enrichment

    bars = _combined_1m()
    daily = pd.read_parquet(BASE_1D)
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
    enriched = _compute_enrichment(
        bars, daily, tick_size=INSTRUMENTS["NQ"]["tick_size"])
    # Retain one prior local day so pre-open context is available at the boundary.
    relevant = enriched[enriched["timestamp"] >= pd.Timestamp(
        "2026-02-28", tz="UTC")].copy()
    days = build_session_days(relevant, "NQ", "TOK")
    start_date = date.fromisoformat(start_date_text)
    rows: list[dict] = []
    for day in days:
        if day.local_date < start_date:
            continue
        for signal in detect_entries(day):
            if not (
                signal.range_minutes == 15
                and signal.mode == "II"
                and signal.closure_tf == 1
                and signal.direction == "short"
            ):
                continue
            for trade in simulate_trade(signal, day, [PRIMARY_RR, SECONDARY_RR]):
                rows.append({
                    "date": str(day.local_date),
                    "instrument": "NQ", "session": "TOK",
                    "range_minutes": 15, "entry_mode": "II",
                    "closure_tf": 1, "direction": "short", "rr": trade.rr,
                    "gross_r": trade.gross_r, "net_r": trade.net_r,
                    "exit_reason": trade.exit_reason,
                    "tp_unfillable": trade.tp_unfillable,
                    "tp_ticks": trade.tp_ticks, "r_ticks": trade.r_ticks,
                    "cost_r": trade.cost_r,
                    "contract_changed_in_session": day.contract_changed_in_session,
                    "contract_changed_since_prev_session": day.contract_changed_since_prev_session,
                    "session_bar_completeness": day.session_bar_completeness,
                })
    return pd.DataFrame(rows)


def evaluate() -> dict:
    protocol = freeze()
    rows = _candidate_rows()
    if rows.empty:
        raise RuntimeError("No candidate signals generated in revalidation period")
    marked = trade_eligibility(rows)
    marked.to_parquet(TRADES, index=False)
    eligible = marked[marked["eligible"]]
    metrics: list[dict] = []
    for rr in (PRIMARY_RR, SECONDARY_RR):
        current = eligible[eligible["rr"] == rr].sort_values("date")
        n = len(current)
        if n:
            values = current["net_r"].to_numpy(dtype=float)
            dates = current["date"].to_numpy()
            lo, hi = _block_bootstrap_ci(
                values, dates, 20_000, 5, seed=20260806 + int(rr * 100))
            record = {
                "rr": rr, "eligible_trades": n,
                "first_date": str(current["date"].min()),
                "last_date": str(current["date"].max()),
                "mean_net_r": float(values.mean()),
                "sd_net_r": float(values.std(ddof=1)) if n > 1 else None,
                "win_rate_net": float(np.mean(values > 0)),
                "ci_lo_95_block": lo, "ci_hi_95_block": hi,
            }
        else:
            record = {"rr": rr, "eligible_trades": 0}
        metrics.append(record)

    primary = metrics[0]
    minimum = int(protocol["minimum_eligible_trades"])
    if primary["eligible_trades"] < minimum:
        verdict = "INCONCLUSIVE: fewer than 100 eligible primary trades"
    elif primary["ci_lo_95_block"] > 0:
        verdict = "CONFIRMED UNDER THE FROZEN REVALIDATION RULE"
    elif primary["ci_hi_95_block"] < 0:
        verdict = "FALSIFIED UNDER THE FROZEN REVALIDATION RULE"
    else:
        verdict = "INCONCLUSIVE: primary interval includes zero"

    combined = _combined_1m()
    last_bar = pd.to_datetime(combined["timestamp"], utc=True).max()
    last_local = last_bar.tz_convert("Asia/Tokyo")
    # Data ending before the 09:00 Tokyo open has not covered that local session.
    next_session = (last_local.date() if (last_local.hour, last_local.minute) < (12, 0)
                    else (last_local.normalize() + pd.Timedelta(days=1)).date())
    result = {
        "evaluated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "protocol_sha256": _sha256(MANIFEST),
        "data_last_timestamp_utc": last_bar.isoformat(),
        "prospective_start_local_date": str(next_session),
        "raw_candidate_rows": len(marked),
        "eligible_candidate_rows": len(eligible),
        "metrics": metrics,
        "verdict": verdict,
        "scope": "historical revalidation, not pristine program-level holdout",
    }
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Frozen NQ/TOK Revalidation", "",
        f"**Verdict:** {verdict}", "",
        f"Data through: `{result['data_last_timestamp_utc']}`", "",
        "| RR | Eligible trades | Mean net R | 95% block CI | Net win rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        if item["eligible_trades"]:
            lines.append(
                f"| {item['rr']:.2f} | {item['eligible_trades']} | "
                f"{item['mean_net_r']:+.4f} | "
                f"[{item['ci_lo_95_block']:+.4f}, {item['ci_hi_95_block']:+.4f}] | "
                f"{item['win_rate_net']:.1%} |")
        else:
            lines.append(f"| {item['rr']:.2f} | 0 | n/a | n/a | n/a |")
    lines += [
        "", "This calendar interval was seen by an earlier research pass. It is",
        "revalidation evidence, not a pristine program-level holdout.", "",
        f"Prospective collection begins with Tokyo session date "
        f"`{result['prospective_start_local_date']}`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def freeze_prospective() -> dict:
    if PROSPECTIVE_PROTOCOL.exists():
        return json.loads(PROSPECTIVE_PROTOCOL.read_text(encoding="utf-8"))
    if not RESULT.exists():
        evaluate()
    revalidation = json.loads(RESULT.read_text(encoding="utf-8"))
    protocol = {
        "protocol_version": 1,
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "parent_protocol_sha256": _sha256(MANIFEST),
        "revalidation_result_sha256": _sha256(RESULT),
        "prospective_start_local_date": revalidation["prospective_start_local_date"],
        "candidate": {
            "instrument": "NQ", "session": "TOK", "range_minutes": 15,
            "entry_mode": "II", "closure_tf": 1, "direction": "short",
            "primary_rr": PRIMARY_RR, "secondary_rr": SECONDARY_RR,
        },
        "minimum_eligible_primary_trades_before_unblinding": 100,
        "unblinding_rule": "first collector run at or above 100 eligible RR 0.75 trades",
        "decision_rule": {
            "confirm": "primary five-day-block 95% CI lower bound > 0",
            "falsify": "primary five-day-block 95% CI upper bound < 0",
            "otherwise": "inconclusive",
        },
        "secondary_policy": "RR 1.00 is descriptive unless the primary confirms",
        "maximum_cumulative_databento_spend_usd": MAX_AUTHORIZED_SPEND,
        "blinding": (
            "Collector reports only eligible counts and data coverage before 100; "
            "return outcomes remain sealed in prospective_trades.parquet."
        ),
    }
    PROSPECTIVE_PROTOCOL.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Frozen prospective protocol: {PROSPECTIVE_PROTOCOL}")
    print(f"Protocol SHA256: {_sha256(PROSPECTIVE_PROTOCOL)}")
    return protocol


def collect_prospective() -> dict:
    protocol = freeze_prospective()
    download()
    start = str(protocol["prospective_start_local_date"])
    rows = _candidate_rows(start)
    if rows.empty:
        marked = rows
        eligible = rows
        primary_n = 0
    else:
        marked = trade_eligibility(rows)
        eligible = marked[marked["eligible"]]
        primary_n = int((eligible["rr"] == PRIMARY_RR).sum())
    marked.to_parquet(PROSPECTIVE_TRADES, index=False)
    combined = _combined_1m()
    last_bar = pd.to_datetime(combined["timestamp"], utc=True).max()
    status = {
        "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "protocol_sha256": _sha256(PROSPECTIVE_PROTOCOL),
        "data_last_timestamp_utc": last_bar.isoformat(),
        "eligible_primary_trades": primary_n,
        "required_before_unblinding": int(
            protocol["minimum_eligible_primary_trades_before_unblinding"]),
        "cumulative_databento_spend_usd": _ledger_spend(),
        "state": "collecting_blinded" if primary_n < 100 else "unblinded",
    }
    PROSPECTIVE_STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if primary_n < 100:
        print(json.dumps(status, indent=2))
        return status

    metrics: list[dict] = []
    for rr in (PRIMARY_RR, SECONDARY_RR):
        current = eligible[eligible["rr"] == rr].sort_values("date")
        values = current["net_r"].to_numpy(dtype=float)
        lo, hi = _block_bootstrap_ci(
            values, current["date"].to_numpy(), 20_000, 5,
            seed=20260807 + int(rr * 100))
        metrics.append({
            "rr": rr, "eligible_trades": len(current),
            "mean_net_r": float(values.mean()),
            "ci_lo_95_block": lo, "ci_hi_95_block": hi,
            "win_rate_net": float(np.mean(values > 0)),
        })
    primary = metrics[0]
    if primary["ci_lo_95_block"] > 0:
        verdict = "CONFIRMED UNDER THE FROZEN PROSPECTIVE RULE"
    elif primary["ci_hi_95_block"] < 0:
        verdict = "FALSIFIED UNDER THE FROZEN PROSPECTIVE RULE"
    else:
        verdict = "INCONCLUSIVE: primary interval includes zero"
    result = {**status, "unblinded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
              "metrics": metrics, "verdict": verdict}
    if not PROSPECTIVE_RESULT.exists():
        PROSPECTIVE_RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        lines = ["# Frozen Prospective NQ/TOK Test", "", f"**{verdict}**", ""]
        for item in metrics:
            lines.append(
                f"- RR {item['rr']:.2f}: n={item['eligible_trades']}, "
                f"mean={item['mean_net_r']:+.4f} R, 95% block CI "
                f"[{item['ci_lo_95_block']:+.4f}, {item['ci_hi_95_block']:+.4f}]")
        PROSPECTIVE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=[
        "freeze", "cost", "download", "evaluate", "freeze-prospective",
        "collect", "run",
    ])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "cost":
        estimate_cost()
    elif args.action == "download":
        download()
    elif args.action == "evaluate":
        evaluate()
    elif args.action == "freeze-prospective":
        freeze_prospective()
    elif args.action == "collect":
        collect_prospective()
    else:
        freeze()
        download()
        evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
