"""Out-of-sample check on the tradability candidates.

`tradability.py` picks 7 setups that are both executable at $900 and net-positive
in-sample.  But 7 out of 4,050 selected on in-sample P&L is exactly the shape of
an overfit: the extreme right tail of a population whose median loses money.  The
only way to tell an edge from a selection artefact is data that took no part in
the selection.

Also reports the LONG and SHORT legs separately.  A combined long+short setup
that earns everything from one leg has half the effective sample it appears to
have, and the ORB thesis (a breakout in either direction) predicts symmetry --
so asymmetry is evidence against the mechanism rather than a bonus.

Reads cached parquet only.  No API calls, no spend.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import INSTRUMENTS, OUTPUTS_DIR
from src.data_layer import _compute_enrichment, ensure_daily, ensure_data
from src.entry_detector import detect_entries
from src.range_builder import build_session_days
from src.sizing import MAX_RISK_USD, size_trades
from src.trade_sim import simulate_trade

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_OUT = _ROOT / OUTPUTS_DIR
HOLDOUT_START = "2026-02-01"
KEYS = ["instrument", "session", "range_minutes", "entry_mode", "closure_tf", "rr"]


def candidates() -> pd.DataFrame:
    """The 7 profitable+tradable, plus NQ's profitable-but-unaffordable set."""
    t = pd.read_csv(_OUT / "tradability.csv")
    good = t[t.tradable & (t.total_pnl_usd > 0)].copy()
    good["group"] = "tradable+profitable"
    nq = t[(t.instrument == "NQ") & (t.total_pnl_usd > 0)].nlargest(10, "total_pnl_usd").copy()
    nq["group"] = "NQ profitable (unaffordable)"
    return pd.concat([good, nq], ignore_index=True)


def resim(syms: list[str]) -> pd.DataFrame:
    rows = []
    for sym in syms:
        bars = _compute_enrichment(ensure_data(sym), ensure_daily(sym),
                                   tick_size=INSTRUMENTS[sym]["tick_size"])
        bars = bars[bars["timestamp"] >= pd.Timestamp(HOLDOUT_START, tz="UTC")]
        if bars.empty:
            log.info("  %s: no holdout bars", sym)
            continue
        log.info("  %s: %d bars  %s -> %s", sym, len(bars),
                 bars["timestamp"].min().date(), bars["timestamp"].max().date())
        for sess in INSTRUMENTS[sym]["sessions"]:
            for sd in build_session_days(bars, sym, sess):
                for es in detect_entries(sd):
                    for tr in simulate_trade(es, sd, [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]):
                        if tr.exit_reason in ("INVALID", None):
                            continue
                        if getattr(tr, "tp_unfillable", False):
                            continue
                        rows.append({
                            "instrument": sym, "session": sess,
                            "range_minutes": es.range_minutes, "entry_mode": es.mode,
                            "closure_tf": es.closure_tf, "direction": es.direction,
                            "rr": tr.rr, "date": str(sd.local_date),
                            "r_ticks": tr.r_ticks, "net_r": tr.net_r,
                        })
    return pd.DataFrame(rows)


def main() -> None:
    cand = candidates()
    syms = sorted(cand["instrument"].unique())
    log.info("candidates: %d setups across %s", len(cand), syms)
    log.info("\nre-simulating holdout (%s onward) ...", HOLDOUT_START)
    ho = resim(syms)
    log.info("  holdout trades: %d", len(ho))

    # In-sample legs, for the symmetry check.
    cols = KEYS + ["direction", "r_ticks", "net_r", "tp_unfillable"]
    ins = pq.read_table(_OUT / "trade_log.parquet", columns=cols).to_pandas()
    ins = ins.loc[~ins["tp_unfillable"].fillna(False).astype(bool)]

    out = []
    for _, c in cand.iterrows():
        key = {k: c[k] for k in KEYS}
        msk_i = np.ones(len(ins), dtype=bool)
        msk_h = np.ones(len(ho), dtype=bool)
        for k, v in key.items():
            msk_i &= (ins[k] == v).to_numpy()
            msk_h &= (ho[k] == v).to_numpy()
        si, sh = ins[msk_i], ho[msk_h]
        if si.empty:
            continue
        sym = c["instrument"]
        rec = {**key, "group": c["group"],
               "is_trades": len(si), "is_net_r": si["net_r"].sum(),
               "is_exp": si["net_r"].mean(), "is_win": (si["net_r"] > 0).mean(),
               "is_usd": size_trades(si, sym)["pnl_usd"].sum()}
        for d in ("long", "short"):
            leg = si[si.direction == d]
            rec[f"is_exp_{d}"] = leg["net_r"].mean() if len(leg) else np.nan
        if len(sh):
            szh = size_trades(sh, sym)
            aff = szh[szh["contracts"] > 0]
            rec.update(ho_trades=len(sh), ho_net_r=sh["net_r"].sum(),
                       ho_exp=sh["net_r"].mean(), ho_win=(sh["net_r"] > 0).mean(),
                       ho_usd=szh["pnl_usd"].sum(),
                       ho_afford=(szh["contracts"] > 0).mean(),
                       # Expectancy on the trades the account can ACTUALLY take.
                       # This is the only number that describes the $900 account:
                       # unaffordable signals contribute exactly $0, so including
                       # them measures a portfolio nobody can hold.
                       ho_trades_aff=len(aff),
                       ho_exp_aff=aff["net_r"].mean() if len(aff) else np.nan)
        else:
            rec.update(ho_trades=0, ho_net_r=np.nan, ho_exp=np.nan,
                       ho_win=np.nan, ho_usd=np.nan, ho_afford=np.nan,
                       ho_trades_aff=0, ho_exp_aff=np.nan)
        # held_up must be judged on the affordable subset, NOT on all trades.
        # Measured on all trades, NQ/NY/30m/R-CC/15m/2.0RR reads +0.1353 R and
        # "held up"; on its 9 affordable trades it is -0.6914 R and lost $4,662.
        # The edge sits in the 48 trades whose stops exceed the $900 wall.
        rec["held_up"] = (bool(rec["ho_exp_aff"] > 0)
                          if rec["ho_trades_aff"] else None)
        rec["held_up_all_trades"] = (bool(rec["ho_exp"] > 0)
                                     if rec["ho_trades"] else None)
        rec["leg_asym"] = abs(rec["is_exp_long"] - rec["is_exp_short"])
        out.append(rec)

    res = pd.DataFrame(out)
    res.to_csv(_OUT / "holdout_candidates.csv", index=False)
    pd.set_option("display.width", 260)

    for grp in res["group"].unique():
        g = res[res.group == grp].sort_values("is_usd", ascending=False)
        log.info("\n=== %s (n=%d) ===", grp, len(g))
        show = ["instrument", "session", "range_minutes", "entry_mode", "closure_tf",
                "rr", "is_trades", "is_win", "is_exp", "is_usd",
                "is_exp_long", "is_exp_short", "ho_trades", "ho_exp",
                "ho_afford", "ho_trades_aff", "ho_exp_aff", "ho_usd",
                "held_up", "held_up_all_trades"]
        print(g[show].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
        held = g["held_up"].dropna()
        if len(held):
            log.info("  held up out-of-sample: %d of %d (%.0f%%)",
                     int(held.sum()), len(held), 100 * held.mean())
        log.info("  in-sample long/short expectancy gap: median %.4f R",
                 g["leg_asym"].median())
        one_leg = ((g["is_exp_long"] > 0) != (g["is_exp_short"] > 0)).sum()
        log.info("  setups where only ONE leg is profitable: %d of %d", one_leg, len(g))

    log.info("\nwrote %s", _OUT / "holdout_candidates.csv")


if __name__ == "__main__":
    main()
