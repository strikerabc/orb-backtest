"""Equity curves for the 7 profitable setups, corrected costs + hybrid sizing.

Combines in-sample (2019-01 -> 2026-01, from cached trade log) with holdout
(2026-02 onward, re-simulated).  Sized under the corrected commission model:
$3.10 RT e-mini/standard, $1.40 RT micro, with hybrid greedy allocation.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.config import OUTPUTS_DIR
from src.contracts import pnl_usd, size_hybrid
from src.data_layer import _compute_enrichment, ensure_daily, ensure_data
from src.entry_detector import detect_entries
from src.range_builder import build_session_days
from src.trade_sim import simulate_trade

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
_OUT = Path(__file__).resolve().parent / OUTPUTS_DIR
HOLDOUT_START = "2026-02-01"
SEVEN = [("ES", "NY", 15, "R-CC", 15, 2.0), ("ES", "NY", 15, "CC", 15, 2.0),
         ("CL", "NY", 30, "CC", 30, 1.5), ("CL", "NY", 30, "R-CC", 30, 1.5),
         ("ES", "NY", 15, "R-CC", 15, 1.5), ("ES", "NY", 15, "CC", 15, 1.5),
         ("ES", "NY", 15, "R-CC", 15, 0.25)]

def load_in_sample(setups: list) -> pd.DataFrame:
    cols = ["instrument", "session", "range_minutes", "entry_mode", "closure_tf",
            "rr", "date", "entry_time_utc", "r_ticks", "gross_r", "tp_unfillable",
            "exit_reason"]
    df = pq.read_table(_OUT / "trade_log.parquet", columns=cols).to_pandas()
    df = df.loc[~df["tp_unfillable"].fillna(False).astype(bool)]
    df = df[df["exit_reason"].notna() & (df["exit_reason"] != "INVALID")]
    df = df[df["gross_r"].notna()]
    syms = {s[0] for s in setups}
    df = df[df["instrument"].isin(syms)]
    mask = np.zeros(len(df), dtype=bool)
    for sym, sess, rm, em, ct, rr in setups:
        mask |= ((df.instrument == sym) & (df.session == sess)
                 & (df.range_minutes == rm) & (df.entry_mode == em)
                 & (df.closure_tf == ct) & (df.rr == rr)).to_numpy()
    df = df[mask].copy()
    df["phase"] = "in_sample"
    log.info("in-sample: %d trades across %d setups", len(df),
             df.groupby(["instrument", "session", "range_minutes", "entry_mode",
                         "closure_tf", "rr"]).ngroups)
    return df


def simulate_holdout(setups: list) -> pd.DataFrame:
    from src.config import INSTRUMENTS
    rows = []
    syms = sorted({s[0] for s in setups})
    for sym in syms:
        log.info("re-simulating holdout for %s ...", sym)
        bars = _compute_enrichment(ensure_data(sym), ensure_daily(sym),
                                   tick_size=INSTRUMENTS[sym]["tick_size"])
        bars = bars[bars["timestamp"] >= pd.Timestamp(HOLDOUT_START, tz="UTC")]
        if bars.empty:
            continue
        log.info("  bars %d  %s -> %s", len(bars),
                 bars["timestamp"].min().date(), bars["timestamp"].max().date())
        sess_set = {s[1] for s in setups if s[0] == sym}
        for sess in sess_set:
            for sd in build_session_days(bars, sym, sess):
                for es in detect_entries(sd):
                    rm_set = {s[2] for s in setups if s[0] == sym and s[1] == sess}
                    if es.range_minutes not in rm_set:
                        continue
                    em_set = {s[3] for s in setups if s[0] == sym and s[1] == sess
                              and s[2] == es.range_minutes}
                    if es.mode not in em_set:
                        continue
                    ct_set = {s[4] for s in setups if s[0] == sym and s[1] == sess
                              and s[2] == es.range_minutes and s[3] == es.mode}
                    if es.closure_tf not in ct_set:
                        continue
                    rr_set = {s[5] for s in setups if s[0] == sym and s[1] == sess
                              and s[2] == es.range_minutes and s[3] == es.mode
                              and s[4] == es.closure_tf}
                    for tr in simulate_trade(es, sd, sorted(rr_set)):
                        if tr.exit_reason in ("INVALID", None):
                            continue
                        if getattr(tr, "tp_unfillable", False):
                            continue
                        rows.append({
                            "instrument": sym, "session": sess,
                            "range_minutes": es.range_minutes, "entry_mode": es.mode,
                            "closure_tf": es.closure_tf, "rr": tr.rr,
                            "date": str(sd.local_date),
                            "entry_time_utc": str(sd.bar_timestamps[es.entry_bar_idx]),
                            "r_ticks": tr.r_ticks, "gross_r": tr.gross_r,
                            "phase": "holdout",
                        })
    out = pd.DataFrame(rows)
    log.info("holdout trades: %d", len(out))
    return out


def build_curve(df: pd.DataFrame, sym: str, sess: str, label: str) -> dict:
    d = df.sort_values(["date", "entry_time_utc"]).reset_index(drop=True)
    sz = size_hybrid(d["r_ticks"].to_numpy(), sym)
    pnl = pnl_usd(d["gross_r"].to_numpy(), d["r_ticks"].to_numpy(), sz, sym, sess)
    taken = (sz["risk_usd"] > 0).to_numpy()
    pnl = np.where(taken, pnl, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        net_r = np.where(taken, pnl / np.maximum(sz["risk_usd"].to_numpy(), 1e-12), 0.0)
    d["cum_r"] = np.nancumsum(net_r)
    d["cum_usd"] = np.cumsum(pnl)
    n = len(d)
    hold = d["phase"] == "holdout"
    stats = {
        "label": label,
        "trades": n,
        "trades_taken": int(taken.sum()),
        "win_rate": float((net_r[taken] > 0).mean()) if taken.any() else 0.0,
        "exp_net_r": float(net_r[taken].mean()) if taken.any() else 0.0,
        "total_net_r": float(net_r[taken].sum()) if taken.any() else 0.0,
        "total_usd": float(pnl.sum()),
        "commission_usd": float(sz.loc[taken, "commission_usd"].sum()),
        "mean_full": float(sz.loc[taken, "n_full"].mean()) if taken.any() else 0.0,
        "mean_micro": float(sz.loc[taken, "n_micro"].mean()) if taken.any() else 0.0,
        "mean_contracts": float((sz.loc[taken, "n_full"] + sz.loc[taken, "n_micro"]).mean()) if taken.any() else 0.0,
        "cap_eff": float(sz.loc[taken, "risk_usd"].mean() / 900) if taken.any() else 0.0,
        "capital_eff": float(sz.loc[taken, "risk_usd"].mean() / 900) if taken.any() else 0.0,
        "risk_med_usd": float(sz.loc[taken, "risk_usd"].median()) if taken.any() else 0.0,
        "executable_pct": float(taken.mean()),
        "friction_r": 0.035,  # approximate for display
        "sl_med": float(d["r_ticks"].median()),
        "sl_p10": float(d["r_ticks"].quantile(0.10)),
        "sl_p90": float(d["r_ticks"].quantile(0.90)),
        "tp_med": float(d["r_ticks"].median() * (2.0 if len(d) else 1.0)),  # approx
        "max_dd_usd": 0.0,  # placeholder
        "max_dd_r": 0.0,
        "n_holdout": int(hold.sum()),
        "holdout_net_r": float(net_r[hold & taken].sum()),
        "holdout_usd": float(pnl[hold].sum()),
        "date_min": str(d["date"].min()),
        "date_max": str(d["date"].max()),
        "holdout_idx": int(hold.to_numpy().argmax()) if hold.any() else -1,
    }
    k = max(1, n // 400)
    idx = list(range(0, n, k))
    if n and idx[-1] != n - 1:
        idx.append(n - 1)
    series = {
        "i": idx,
        "r": [round(float(v), 4) for v in d["cum_r"].to_numpy()[idx]],
        "usd": [round(float(v), 2) for v in d["cum_usd"].to_numpy()[idx]],
        "date": [str(v) for v in d["date"].to_numpy()[idx]],
    }
    return {"stats": stats, "series": series}


def main() -> None:
    ins = load_in_sample(SEVEN)
    hol = simulate_holdout(SEVEN)
    allt = pd.concat([ins, hol], ignore_index=True) if len(hol) else ins
    panels = []
    for sym, sess, rm, em, ct, rr in SEVEN:
        lbl = f"{sym}/{sess}/{rm}m/{em}/{ct}m/{rr}RR"
        sub = allt[(allt.instrument == sym) & (allt.session == sess)
                   & (allt.range_minutes == rm) & (allt.entry_mode == em)
                   & (allt.closure_tf == ct) & (allt.rr == rr)]
        if sub.empty:
            continue
        built = build_curve(sub, sym, sess, lbl)
        panels.append({
            "label": lbl,
            "unavailable": False,
            "instrument": sym,
            "session": sess,
            "rr": rr,
            **built,
        })
        s = built["stats"]
        log.info("  %-38s n=%-5d wr=%.3f  totR=%+8.2f  usd=%+11.2f",
                 lbl, s["trades"], s["win_rate"], s["total_net_r"], s["total_usd"])
    log.info("\nbuilt %d curves", len(panels))
    return panels


if __name__ == "__main__":
    main()


