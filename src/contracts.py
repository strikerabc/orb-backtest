"""Contract specifications and hybrid full-size/micro position sizing.

The sweep priced every instrument at ``COMMISSION_PER_SIDE_USD = 2.50``, i.e.
$5.00 round-trip.  Real retail futures commissions are lower and, critically,
differ between full-size and micro contracts:

    e-mini / standard : $3.10 round-trip
    micro             : $1.40 round-trip

Commission enters the cost model divided by TICK VALUE, so the cheaper micro
commission is still far more expensive *per R*:

    comm_ticks = round_trip_usd / tick_value_usd

    ES  $3.10 / $12.50 = 0.248 ticks
    MES $1.40 / $1.25  = 1.120 ticks   <- 4.5x more, despite the smaller fee

So micros are not a cost improvement.  Their value is purely that a 1/10-size
contract has a 10x wider affordable stop, which rescues trades whose stops
breach the full-size wall imposed by a fixed risk budget.

Sizing therefore prefers full-size and falls back to micros only when full-size
does not fit -- cheapest-first, with micros as the remainder filler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sizing import MAX_CONTRACTS, MAX_RISK_USD
from .trade_sim import slippage_ticks_for

# ── Contract specs ───────────────────────────────────────────────────────────
# tick_value_usd for the micro is 1/10 of the full-size in every pair below
# (MES = 1/10 ES, MCL = 1/10 CL, MNQ = 1/10 NQ, M6E = 1/10 6E).  Tick SIZE is
# identical between the pair members, so a stop measured in ticks is directly
# transferable -- only the dollars per tick change.
#
# Commissions are USER-SUPPLIED round-trip figures.  CL is a standard contract
# rather than an e-mini; the $3.10 e-mini rate is applied to it as the closest
# available estimate, which is an ASSUMPTION worth confirming with the broker.
CONTRACTS: dict[str, dict] = {
    "ES": {
        "full":  {"symbol": "ES",  "tick_value_usd": 12.500, "rt_commission_usd": 3.10},
        "micro": {"symbol": "MES", "tick_value_usd": 1.250,  "rt_commission_usd": 1.40},
    },
    "CL": {
        "full":  {"symbol": "CL",  "tick_value_usd": 10.000, "rt_commission_usd": 3.10},
        "micro": {"symbol": "MCL", "tick_value_usd": 1.000,  "rt_commission_usd": 1.40},
    },
    "NQ": {
        "full":  {"symbol": "NQ",  "tick_value_usd": 5.000,  "rt_commission_usd": 3.10},
        "micro": {"symbol": "MNQ", "tick_value_usd": 0.500,  "rt_commission_usd": 1.40},
    },
    "6E": {
        "full":  {"symbol": "6E",  "tick_value_usd": 6.250,  "rt_commission_usd": 3.10},
        "micro": {"symbol": "M6E", "tick_value_usd": 0.625,  "rt_commission_usd": 1.40},
    },
}

# Spot FX, for the forex comparison the user asked about.
SPOT_FX_RT_PER_100K_USD = 7.00
SPOT_FX_LEVERAGE = 50


def comm_ticks(sym: str, kind: str) -> float:
    """Round-trip commission expressed in ticks."""
    c = CONTRACTS[sym][kind]
    return c["rt_commission_usd"] / c["tick_value_usd"]


def cost_ticks(sym: str, kind: str, session: str) -> float:
    """Total round-trip friction in ticks: commission + measured slippage.

    Slippage was measured on the FULL-SIZE contract's book (bbo-1m).  Applying
    the same tick spread to the micro assumes the micro quotes as tightly,
    which is broadly true for MES/MNQ but is an optimistic assumption, not a
    measurement.
    """
    return comm_ticks(sym, kind) + slippage_ticks_for(sym, session)


def max_stop_ticks(sym: str, kind: str, max_risk_usd: float = MAX_RISK_USD) -> float:
    """Widest stop affordable at one contract of this kind."""
    return max_risk_usd / CONTRACTS[sym][kind]["tick_value_usd"]


def size_hybrid(
    r_ticks: np.ndarray,
    sym: str,
    max_risk_usd: float = MAX_RISK_USD,
    max_contracts: int = MAX_CONTRACTS,
) -> pd.DataFrame:
    """Greedy cheapest-first sizing: full-size, then micros with what's left.

    ``max_contracts`` is treated as a cap on the TOTAL contract count of any
    kind (10 micros counts as 10), which is the conservative reading of a
    broker position limit.

    Returns per-trade counts and the resulting risk/commission dollars.
    """
    r = np.asarray(r_ticks, dtype=float)
    tv_f = CONTRACTS[sym]["full"]["tick_value_usd"]
    tv_m = CONTRACTS[sym]["micro"]["tick_value_usd"]
    rt_f = CONTRACTS[sym]["full"]["rt_commission_usd"]
    rt_m = CONTRACTS[sym]["micro"]["rt_commission_usd"]

    risk_f = r * tv_f
    risk_m = r * tv_m
    with np.errstate(divide="ignore", invalid="ignore"):
        n_f = np.where(risk_f > 0, np.floor(max_risk_usd / risk_f), 0.0)
    n_f = np.clip(n_f, 0, max_contracts)

    left_usd = max_risk_usd - n_f * risk_f
    left_slots = max_contracts - n_f
    with np.errstate(divide="ignore", invalid="ignore"):
        n_m = np.where(risk_m > 0, np.floor(left_usd / risk_m), 0.0)
    n_m = np.clip(n_m, 0, left_slots)

    return pd.DataFrame({
        "n_full": n_f.astype(int),
        "n_micro": n_m.astype(int),
        "risk_usd": n_f * risk_f + n_m * risk_m,
        "commission_usd": n_f * rt_f + n_m * rt_m,
        "tick_value_total": n_f * tv_f + n_m * tv_m,
    })


def size_single(
    r_ticks: np.ndarray,
    sym: str,
    kind: str,
    max_risk_usd: float = MAX_RISK_USD,
    max_contracts: int = MAX_CONTRACTS,
) -> pd.DataFrame:
    """Sizing restricted to one contract kind (for the comparison table)."""
    r = np.asarray(r_ticks, dtype=float)
    tv = CONTRACTS[sym][kind]["tick_value_usd"]
    rt = CONTRACTS[sym][kind]["rt_commission_usd"]
    risk = r * tv
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(risk > 0, np.floor(max_risk_usd / risk), 0.0)
    n = np.clip(n, 0, max_contracts)
    return pd.DataFrame({
        "n_full": (n if kind == "full" else np.zeros_like(n)).astype(int),
        "n_micro": (n if kind == "micro" else np.zeros_like(n)).astype(int),
        "risk_usd": n * risk,
        "commission_usd": n * rt,
        "tick_value_total": n * tv,
    })


def pnl_usd(
    gross_r: np.ndarray,
    r_ticks: np.ndarray,
    sized: pd.DataFrame,
    sym: str,
    session: str,
) -> np.ndarray:
    """Dollar P&L from gross R, re-priced under the corrected cost model.

    Computed in dollars rather than by blending ``cost_r``, because a hybrid
    position pays two different commission rates on the same trade and there is
    no single ``cost_r`` that describes it.

        pnl = gross_r x risk_usd  -  commission_usd  -  slippage_usd

    Reduces exactly to ``risk_usd x (gross_r - cost_r)`` for a pure single-kind
    position, which is the identity the engine itself uses.
    """
    slip = slippage_ticks_for(sym, session)
    gross_usd = np.asarray(gross_r, dtype=float) * sized["risk_usd"].to_numpy()
    slip_usd = slip * sized["tick_value_total"].to_numpy()
    return gross_usd - sized["commission_usd"].to_numpy() - slip_usd
