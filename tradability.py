"""Rank every ORB setup by TRADABILITY rather than by edge.

Edge is deliberately out of scope here.  The question is narrower and more
practical: given a $900 risk budget and a 10-contract cap, which setups can
actually be executed, hit a high win rate, and compound over the sample?

A "setup" combines long+short into one deployment (the user trades both), so
the 8,100 variants collapse to 4,050 setups.

Three hard gates, then a rank.  The gates are not preferences -- a setup that
fails one is not merely worse, it is untradable:

  1. TP >= 4 ticks      -- a 1-2 tick target is inside the spread's noise.
  2. SL >= 8 ticks      -- ditto for the stop; also caps the contract count.
  3. friction <= 15% R  -- above this the spread eats the trade.
  4. >= 80% of trades affordable at 1 contract within the budget.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import INSTRUMENTS
from src.sizing import (MAX_CONTRACTS, MAX_RISK_USD, MAX_FRICTION_R,
                        MIN_EXECUTABLE_SL_TICKS, MIN_EXECUTABLE_TP_TICKS,
                        contracts_for, tick_value)
from src.trade_sim import slippage_ticks_for

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

TRADE_LOG = "outputs/trade_log.parquet"
OUT_CSV = "outputs/tradability.csv"
KEYS = ["instrument", "session", "range_minutes", "entry_mode", "closure_tf", "rr"]
COLS = KEYS + ["direction", "date", "r_ticks", "tp_ticks", "tp_unfillable",
               "net_r", "gross_r", "exit_reason"]
MIN_EXECUTABLE_PCT = 0.80
MIN_TRADES = 100


def load() -> pd.DataFrame:
    log.info("reading %s ...", TRADE_LOG)
    df = pq.read_table(TRADE_LOG, columns=COLS).to_pandas()
    log.info("  %d rows", len(df))
    # Match stats.py: unfillable-TP trades are not real fills.
    mask = df["tp_unfillable"].fillna(False).astype(bool)
    if mask.any():
        log.info("  dropping %d unfillable-TP rows (%.2f%%)",
                 mask.sum(), 100 * mask.mean())
        df = df.loc[~mask]
    # INVALID exits are non-trades: degenerate stops (r_ticks 0-1) where no
    # valid bracket could be placed, so gross_r/net_r are NaN.  pandas .agg()
    # skips NaN so the SUMS were right, but these rows still landed in the
    # trade COUNT (438 setups inflated, up to 43 rows each) and, because
    # NaN > 0 is False, were counted as losses -- diluting win_rate.
    inv = df["exit_reason"].isna() | (df["exit_reason"] == "INVALID")
    if inv.any():
        log.info("  dropping %d INVALID-exit rows (%.3f%%)",
                 inv.sum(), 100 * inv.mean())
        df = df.loc[~inv]
    return df


def per_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per setup, with sizing applied per instrument."""
    # Sizing depends on the instrument's tick value, so vectorise per symbol.
    df = df.copy()
    df["tick_val"] = df["instrument"].map(lambda s: tick_value(s))
    df["risk_per_contract_usd"] = df["r_ticks"] * df["tick_val"]
    contracts = np.zeros(len(df), dtype=int)
    for sym, idx in df.groupby("instrument").groups.items():
        pos = df.index.get_indexer(idx)
        contracts[pos] = contracts_for(df.loc[idx, "r_ticks"], sym)
    df["contracts"] = contracts
    df["risk_deployed_usd"] = df["risk_per_contract_usd"] * df["contracts"]
    df["pnl_usd"] = df["net_r"] * df["risk_deployed_usd"]
    df["is_win"] = df["net_r"] > 0

    g = df.groupby(KEYS, observed=True)
    out = g.agg(
        trades=("net_r", "size"),
        win_rate=("is_win", "mean"),
        exp_gross_r=("gross_r", "mean"),
        exp_net_r=("net_r", "mean"),
        total_net_r=("net_r", "sum"),
        sl_med=("r_ticks", "median"),
        sl_p10=("r_ticks", lambda s: s.quantile(0.10)),
        tp_med=("tp_ticks", "median"),
        risk_med_usd=("risk_per_contract_usd", "median"),
        executable_pct=("contracts", lambda s: (s > 0).mean()),
        mean_contracts=("contracts", "mean"),
        risk_deployed_mean=("risk_deployed_usd", "mean"),
        total_pnl_usd=("pnl_usd", "sum"),
        n_days=("date", "nunique"),
    ).reset_index()

    # Friction is a property of (symbol, session, median stop), not of a trade.
    out["slip_ticks"] = [slippage_ticks_for(s, sess)
                         for s, sess in zip(out.instrument, out.session)]
    out["friction_r"] = out["slip_ticks"] / out["sl_med"]
    out["asset_class"] = out["instrument"].map(
        lambda s: INSTRUMENTS[s].get("asset_class", "?"))
    # Mean dollars actually at risk per trade, as a fraction of the budget.
    # Computed from per-trade deployment (NOT mean_contracts x median_risk,
    # which is a product of aggregates and does not equal the aggregate of
    # the product).  Not clipped: it cannot exceed 1.0 by construction, so a
    # value at 1.0 is informative rather than censored.
    out["capital_eff"] = out["risk_deployed_mean"] / MAX_RISK_USD
    # Dollars earned per dollar-year of risk actually deployed.
    yrs = out["n_days"] / 252.0
    out["pnl_per_yr_usd"] = out["total_pnl_usd"] / yrs.replace(0, np.nan)
    return out


def score(out: pd.DataFrame) -> pd.DataFrame:
    """Composite rank across the three stated tradability criteria.

    The criteria PULL AGAINST EACH OTHER in this data: win rate is mechanically
    a function of rr (a 0.25RR target is hit ~80% of the time, a 2.0RR target
    ~43%), and P&L runs the other way.  So a single ranking necessarily encodes
    a weighting choice.  Percentile-rank each axis among gated setups and
    average them, which at least makes the trade-off explicit and scale-free
    rather than smuggling it in via raw units.
    """
    out = out.copy()
    # Rank ONLY among setups that clear zero.  Percentile-ranking P&L across a
    # population that is 99.7% negative rewards "least-bad": an early version
    # of this put a setup losing $51,016 at rank 1, because -$51k is a high
    # percentile when almost everything loses more.  A composite is only
    # meaningful once the sign is right, so positive P&L is a gate, not an axis.
    t = out["tradable"] & (out["total_pnl_usd"] > 0)
    out["scoreable"] = t
    for col, src in [("pct_pnl", "total_pnl_usd"),
                     ("pct_wr", "win_rate"),
                     ("pct_cap", "capital_eff")]:
        out[col] = np.nan
        out.loc[t, col] = out.loc[t, src].rank(pct=True)
    out["score"] = out[["pct_pnl", "pct_wr", "pct_cap"]].mean(axis=1)
    return out


def gate(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["g_tp"] = out["tp_med"] >= MIN_EXECUTABLE_TP_TICKS
    out["g_sl"] = out["sl_med"] >= MIN_EXECUTABLE_SL_TICKS
    out["g_friction"] = out["friction_r"] <= MAX_FRICTION_R
    out["g_afford"] = out["executable_pct"] >= MIN_EXECUTABLE_PCT
    out["g_sample"] = out["trades"] >= MIN_TRADES
    gates = ["g_tp", "g_sl", "g_friction", "g_afford", "g_sample"]
    out["tradable"] = out[gates].all(axis=1)
    out["n_gates_failed"] = (~out[gates]).sum(axis=1)
    return out


def main() -> None:
    df = load()
    out = score(gate(per_setup(df)))
    out.to_csv(OUT_CSV, index=False)
    log.info("wrote %s (%d setups, %d tradable)", OUT_CSV, len(out),
             int(out.tradable.sum()))

    log.info("\n=== GATE ATTRITION (of %d setups) ===", len(out))
    for g, lbl in [("g_sample", f"trades >= {MIN_TRADES}"),
                   ("g_tp", f"median TP >= {MIN_EXECUTABLE_TP_TICKS}t"),
                   ("g_sl", f"median SL >= {MIN_EXECUTABLE_SL_TICKS}t"),
                   ("g_friction", f"friction <= {MAX_FRICTION_R:.0%} R"),
                   ("g_afford", f">= {MIN_EXECUTABLE_PCT:.0%} affordable @1 contract")]:
        log.info("  %-38s %4d pass  %4d fail", lbl,
                 int(out[g].sum()), int((~out[g]).sum()))
    log.info("  %-38s %4d pass", "ALL GATES", int(out.tradable.sum()))

    log.info("\n=== BY ASSET CLASS (all setups) ===")
    ac = out.groupby("asset_class").agg(
        setups=("tradable", "size"), tradable=("tradable", "sum"),
        win_rate=("win_rate", "mean"), sl_med=("sl_med", "median"),
        tp_med=("tp_med", "median"), friction=("friction_r", "median"),
        afford=("executable_pct", "mean"), cap_eff=("capital_eff", "mean"),
        med_pnl=("total_pnl_usd", "median"),
    ).sort_values("tradable", ascending=False)
    ac["tradable_pct"] = ac.tradable / ac.setups
    pd.set_option("display.width", 250)
    print(ac.to_string(float_format=lambda v: f"{v:,.4f}"))

    log.info("\n=== BY ASSET CLASS (TRADABLE setups only) ===")
    t = out[out.tradable]
    if len(t):
        act = t.groupby("asset_class").agg(
            setups=("tradable", "size"), win_rate=("win_rate", "mean"),
            best_wr=("win_rate", "max"), sl_med=("sl_med", "median"),
            tp_med=("tp_med", "median"), friction=("friction_r", "median"),
            cap_eff=("capital_eff", "mean"), med_pnl=("total_pnl_usd", "median"),
            best_pnl=("total_pnl_usd", "max"), tot_net_r=("total_net_r", "median"),
        ).sort_values("med_pnl", ascending=False)
        print(act.to_string(float_format=lambda v: f"{v:,.4f}"))

        log.info("\n=== TOP 25 TRADABLE SETUPS by total P&L ($900 / 10 cap) ===")
        show = ["instrument", "session", "range_minutes", "entry_mode", "closure_tf",
                "rr", "trades", "win_rate", "sl_med", "tp_med", "friction_r",
                "risk_med_usd", "mean_contracts", "capital_eff", "exp_net_r",
                "total_net_r", "total_pnl_usd"]
        print(t.nlargest(25, "total_pnl_usd")[show].to_string(
            index=False, float_format=lambda v: f"{v:,.4f}"))

        log.info("\n=== TOP 15 TRADABLE by WIN RATE (>=200 trades) ===")
        print(t[t.trades >= 200].nlargest(15, "win_rate")[show].to_string(
            index=False, float_format=lambda v: f"{v:,.4f}"))

        log.info("\n=== WIN RATE vs P&L: are the criteria compatible? ===")
        by_rr = t.groupby("rr").agg(
            setups=("win_rate", "size"), win_rate=("win_rate", "mean"),
            med_pnl=("total_pnl_usd", "median"), best_pnl=("total_pnl_usd", "max"),
            pct_profitable=("total_pnl_usd", lambda s: (s > 0).mean()),
        )
        print(by_rr.to_string(float_format=lambda v: f"{v:,.4f}"))
        c = t[["win_rate", "total_pnl_usd"]].corr(method="spearman").iloc[0, 1]
        log.info("  Spearman corr(win_rate, total_pnl_usd) among tradable = %+.4f", c)
        log.info("  tradable setups with total_pnl_usd > 0: %d of %d (%.1f%%)",
                 int((t.total_pnl_usd > 0).sum()), len(t),
                 100 * (t.total_pnl_usd > 0).mean())

        log.info("\n=== COMPOSITE SCORE — ranked among PROFITABLE tradable only ===")
        sc = out[out.scoreable]
        log.info("  scoreable population: %d setups (tradable AND pnl > 0)", len(sc))
        show_s = show + ["pct_pnl", "pct_wr", "pct_cap", "score"]
        print(sc.nlargest(min(20, len(sc)), "score")[show_s].to_string(
            index=False, float_format=lambda v: f"{v:,.4f}"))

        log.info("\n=== PARETO FRONTIER (no setup beats these on all 3 axes) ===")
        axes = ["total_pnl_usd", "win_rate", "capital_eff"]
        arr = t[axes].to_numpy()
        dominated = np.zeros(len(t), dtype=bool)
        for i in range(len(t)):
            dominated[i] = bool(((arr >= arr[i]).all(axis=1)
                                 & (arr > arr[i]).any(axis=1)).any())
        pf = t.loc[~dominated]
        log.info("  %d of %d tradable setups are Pareto-optimal", len(pf), len(t))
        print(pf.nlargest(min(20, len(pf)), "total_pnl_usd")[show].to_string(
            index=False, float_format=lambda v: f"{v:,.4f}"))


if __name__ == "__main__":
    main()
