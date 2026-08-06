"""Position sizing under a fixed risk budget and a hard contract cap.

The backtest records ``net_usd`` for exactly ONE contract (verified: for NQ the
implied tick value from ``net_usd / (net_r * r_ticks)`` is $5.000, matching
``INSTRUMENTS['NQ']['tick_value_usd']``).  Turning that into an account P&L
requires a sizing rule, which is what this module owns.

Two constraints bind, and they bind from OPPOSITE directions:

  * **Stop too WIDE**  -> one contract already risks more than the budget, so
    the trade cannot be taken at all.  This is the executability wall.
  * **Stop too NARROW** -> the contract cap is hit before the budget is spent,
    so the trade is under-deployed and earns less than a full-risk unit.

Both are real frictions and both are captured by ``size_trades``.  Expressing
results only in R hides them: R implicitly assumes every trade risks the same
amount, which is precisely what a contract cap makes untrue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INSTRUMENTS

# ── Account constraints (user-supplied) ──────────────────────────────────────
MAX_RISK_USD = 900.0
MAX_CONTRACTS = 10

# ── Executability thresholds ─────────────────────────────────────────────────
# A take-profit of 1-2 ticks is not a tradable target: it sits inside the
# spread's own noise, so the fill is a coin flip rather than a price level.
MIN_EXECUTABLE_TP_TICKS = 4.0
MIN_EXECUTABLE_SL_TICKS = 8.0

# Friction ceiling.  ``cost_r = slippage_ticks / r_ticks`` is the fraction of
# each risk unit paid to the spread on entry+exit.  Above ~15% the strategy is
# renting the broker's spread rather than trading the market.
MAX_FRICTION_R = 0.15


def tick_value(symbol: str) -> float:
    return float(INSTRUMENTS[symbol]["tick_value_usd"])


def tick_size(symbol: str) -> float:
    return float(INSTRUMENTS[symbol]["tick_size"])


def risk_per_contract(r_ticks, symbol: str):
    """Dollar risk of a single contract for a stop of ``r_ticks``."""
    return np.asarray(r_ticks, dtype=float) * tick_value(symbol)


def contracts_for(
    r_ticks,
    symbol: str,
    max_risk_usd: float = MAX_RISK_USD,
    max_contracts: int = MAX_CONTRACTS,
):
    """Integer contracts affordable for each stop distance.

    Returns 0 where a single contract would breach the risk budget -- those
    trades are *unexecutable*, not merely small, and must be dropped rather
    than sized down (you cannot trade a third of a futures contract).
    """
    rpc = risk_per_contract(r_ticks, symbol)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(rpc > 0, np.floor(max_risk_usd / rpc), 0.0)
    return np.clip(raw, 0, max_contracts).astype(int)


def size_trades(
    df: pd.DataFrame,
    symbol: str,
    max_risk_usd: float = MAX_RISK_USD,
    max_contracts: int = MAX_CONTRACTS,
) -> pd.DataFrame:
    """Attach sizing columns to a single-instrument trade frame.

    Adds ``risk_per_contract_usd``, ``contracts``, ``executable``,
    ``risk_deployed_usd``, ``capital_efficiency`` and ``pnl_usd``.
    Does not filter -- callers decide whether to drop unexecutable rows, so
    the skip rate stays visible instead of silently vanishing.
    """
    out = df.copy()
    tv = tick_value(symbol)
    out["risk_per_contract_usd"] = out["r_ticks"].astype(float) * tv
    out["contracts"] = contracts_for(out["r_ticks"], symbol, max_risk_usd, max_contracts)
    out["executable"] = out["contracts"] > 0
    out["risk_deployed_usd"] = out["risk_per_contract_usd"] * out["contracts"]
    # How much of the risk budget the trade actually puts to work.  <1.0 means
    # the contract cap bound before the budget did.
    out["capital_efficiency"] = out["risk_deployed_usd"] / float(max_risk_usd)
    # net_r is in risk units, so P&L = net_r x (dollars actually at risk).
    out["pnl_usd"] = out["net_r"].astype(float) * out["risk_deployed_usd"]
    return out


def friction_r(r_ticks, symbol: str, session: str) -> float:
    """Round-trip slippage as a fraction of one risk unit."""
    from .trade_sim import slippage_ticks_for

    slip = slippage_ticks_for(symbol, session)
    r = float(r_ticks)
    return slip / r if r > 0 else float("inf")


def max_stop_ticks(symbol: str, max_risk_usd: float = MAX_RISK_USD) -> float:
    """Widest stop still affordable at one contract."""
    return max_risk_usd / tick_value(symbol)


def full_deployment_stop_ticks(
    symbol: str,
    max_risk_usd: float = MAX_RISK_USD,
    max_contracts: int = MAX_CONTRACTS,
) -> float:
    """Stop width at which ``max_contracts`` exactly consumes the budget.

    Stops narrower than this leave the budget partly unused because the
    contract cap binds first.
    """
    return max_risk_usd / (tick_value(symbol) * max_contracts)
