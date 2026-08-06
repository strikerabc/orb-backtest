"""Re-price every setup under corrected round-trip commissions.

The sweep charged $5.00 round-trip per contract everywhere.  The real figures are
$3.10 (e-mini/standard) and $1.40 (micro), so the sweep OVERCHARGED commission --
by 38% on full-size and 72% on micros in tick terms.

That matters beyond a rescaling: the set of setups that clear zero was selected
under the wrong cost model.  Re-pricing only the previously-identified 7 would
inherit a stale selection, so this recomputes the whole population and reports
how it moved.

Three sizing modes per setup, all under a $900 risk budget / 10-contract cap:

  full   -- full-size only; cheapest per R but the widest stop it can hold is
            $900 / tick_value, so wide-stop trades are skipped entirely
  micro  -- micro only; 10x wider affordable stop, but ~4.5x the commission in
            tick terms, and the contract cap binds early on tight stops
  hybrid -- full-size first, micros for the remainder ("micros where necessary")

Reads cached parquet only.  No API calls, no spend.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import OUTPUTS_DIR, INVALID_REASONS
from src.contracts import (CONTRACTS, comm_ticks, cost_ticks, max_stop_ticks,
                           pnl_usd, size_hybrid, size_single,
                           DEFAULT_RT_COMMISSION_USD)
from src.sizing import MAX_CONTRACTS, MAX_RISK_USD
from src.trade_sim import slippage_ticks_for

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_OUT = Path(__file__).resolve().parent / OUTPUTS_DIR
KEYS = ["instrument", "session", "range_minutes", "entry_mode", "closure_tf", "rr"]
MODES = ("full", "micro", "hybrid")
MIN_TRADES = 100
MIN_AFFORD = 0.80


def load(symbols: list[str] | None = None) -> pd.DataFrame:
    cols = KEYS + ["direction", "date", "entry_time_utc", "r_ticks", "tp_ticks",
                   "gross_r", "net_r", "tp_unfillable", "exit_reason"]
    df = pq.read_table(_OUT / "trade_log.parquet", columns=cols).to_pandas()
    df = df.loc[~df["tp_unfillable"].fillna(False).astype(bool)]
    # INVALID exits are non-trades: degenerate stops (r_ticks 0-1) where the
    # simulator could not place a valid bracket, so gross_r is NaN.  Every other
    # consumer in the codebase drops them; they must go here too because
    # numpy.sum propagates NaN, and a single such row nulls an entire setup's
    # total (378 poisoned group-rows == 126 setups, from 0.045% of trades).
    df = df[~df["exit_reason"].isin(INVALID_REASONS)]
    df = df[df["gross_r"].notna()]
    if symbols:
        df = df[df["instrument"].isin(symbols)]
    return df


def size_mode(r_ticks: np.ndarray, sym: str, mode: str) -> pd.DataFrame:
    if mode == "hybrid":
        return size_hybrid(r_ticks, sym)
    return size_single(r_ticks, sym, mode)


def price_group(d: pd.DataFrame, sym: str, sess: str, mode: str) -> dict:
    """Re-price one setup under corrected commissions for one sizing mode."""
    sz = size_mode(d["r_ticks"].to_numpy(), sym, mode)
    pnl = pnl_usd(d["gross_r"].to_numpy(), d["r_ticks"].to_numpy(), sz, sym, sess)
    taken = (sz["risk_usd"] > 0).to_numpy()
    # A trade with no affordable contract contributes $0 and is not a trade.
    pnl = np.where(taken, pnl, 0.0)
    n = len(d)
    # Per-trade R net of the corrected cost model, for the trades actually taken.
    with np.errstate(divide="ignore", invalid="ignore"):
        net_r = np.where(taken & (sz["risk_usd"].to_numpy() > 0),
                         pnl / np.maximum(sz["risk_usd"].to_numpy(), 1e-12), 0.0)
    return {
        "mode": mode,
        "trades": n,
        "trades_taken": int(taken.sum()),
        "afford": float(taken.mean()) if n else 0.0,
        "win_rate": float((net_r[taken] > 0).mean()) if taken.any() else 0.0,
        "exp_net_r": float(net_r[taken].mean()) if taken.any() else 0.0,
        "total_pnl_usd": float(pnl.sum()),
        "commission_usd": float(sz.loc[taken, "commission_usd"].sum()),
        "cap_eff": float(sz["risk_usd"].mean() / MAX_RISK_USD) if n else 0.0,
        "mean_full": float(sz.loc[taken, "n_full"].mean()) if taken.any() else 0.0,
        "mean_micro": float(sz.loc[taken, "n_micro"].mean()) if taken.any() else 0.0,
        "sl_med": float(d["r_ticks"].median()) if n else 0.0,
        "tp_med": float(d["tp_ticks"].median()) if n else 0.0,
        "_pnl": pnl,
        "_taken": taken,
        "_net_r": net_r,
        "_sz": sz,
    }


def main() -> None:
    log.info("sweep commission: $%.2f/side = $%.2f round-trip",
             DEFAULT_RT_COMMISSION_USD / 2, DEFAULT_RT_COMMISSION_USD)
    log.info("corrected:        e-mini/standard $3.10 RT, micro $1.40 RT\n")

    syms = sorted(CONTRACTS.keys())
    df = load(syms)
    log.info("re-pricing %d trades across %s", len(df), syms)

    rows = []
    for key, d in df.groupby(KEYS, observed=True):
        sym, sess = key[0], key[1]
        if len(d) < MIN_TRADES:
            continue
        base = dict(zip(KEYS, key))
        for mode in MODES:
            r = price_group(d, sym, sess, mode)
            for k in ("_pnl", "_taken", "_net_r", "_sz"):
                r.pop(k, None)
            rows.append({**base, **r})

    res = pd.DataFrame(rows)
    res["tradable"] = (res["afford"] >= MIN_AFFORD) & (res["trades"] >= MIN_TRADES)
    res.to_csv(_OUT / "reprice_corrected.csv", index=False)
    pd.set_option("display.width", 260)

    log.info("\n=== POPULATION SHIFT: profitable setups by mode ===")
    for mode in MODES:
        m = res[res["mode"] == mode]
        t = m[m.tradable]
        pct = 100 * (t.total_pnl_usd > 0).mean() if len(t) else 0.0
        best = t.total_pnl_usd.max() if len(t) else 0.0
        log.info(f"  {mode:<6}  setups {len(m):4d}   tradable {len(t):4d}   "
                 f"net-positive {int((m.total_pnl_usd > 0).sum()):4d}   "
                 f"of tradable {int((t.total_pnl_usd > 0).sum()):4d} ({pct:.1f}%)   "
                 f"best ${best:,.0f}")

    log.info("\n=== previously-profitable 7, re-priced (all modes) ===")
    seven = [("ES", "NY", 15, "R-CC", 15, 2.0), ("ES", "NY", 15, "CC", 15, 2.0),
             ("CL", "NY", 30, "CC", 30, 1.5), ("CL", "NY", 30, "R-CC", 30, 1.5),
             ("ES", "NY", 15, "R-CC", 15, 1.5), ("ES", "NY", 15, "CC", 15, 1.5),
             ("ES", "NY", 15, "R-CC", 15, 0.25)]
    sel = pd.concat([res[(res.instrument == a) & (res.session == b)
                         & (res.range_minutes == c) & (res.entry_mode == d)
                         & (res.closure_tf == e) & (res.rr == f)]
                     for a, b, c, d, e, f in seven])
    show = ["instrument", "session", "range_minutes", "entry_mode", "closure_tf",
            "rr", "mode", "trades", "trades_taken", "afford", "win_rate",
            "exp_net_r", "cap_eff", "mean_full", "mean_micro", "commission_usd",
            "total_pnl_usd"]
    print(sel[show].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    log.info("\n=== best mode per setup (of the 7) ===")
    best = (sel.loc[sel.groupby(KEYS, observed=True)["total_pnl_usd"].idxmax()]
              .sort_values("total_pnl_usd", ascending=False))
    print(best[show].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    log.info("\n=== TOP 20 tradable setups under corrected costs (any mode) ===")
    t = res[res.tradable]
    bm = t.loc[t.groupby(KEYS, observed=True)["total_pnl_usd"].idxmax()]
    print(bm.nlargest(20, "total_pnl_usd")[show].to_string(
        index=False, float_format=lambda v: f"{v:,.4f}"))
    log.info("\nwrote %s", _OUT / "reprice_corrected.csv")


if __name__ == "__main__":
    main()
