"""Central trade eligibility rules shared by observed and null pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import INVALID_REASONS, MIN_SESSION_BAR_COMPLETENESS
from src.sizing import (
    MAX_FRICTION_R, MIN_EXECUTABLE_SL_TICKS, MIN_EXECUTABLE_TP_TICKS,
    contracts_for,
)


RULE_COLUMNS = (
    "excluded_invalid_exit",
    "excluded_unfillable_tp",
    "excluded_small_tp",
    "excluded_small_sl",
    "excluded_high_friction",
    "excluded_unaffordable",
    "excluded_roll_day",
    "excluded_incomplete_session",
)


def trade_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Attach one boolean per exclusion rule plus the combined ``eligible``."""
    out = df.copy()
    index = out.index

    reason = out.get("exit_reason", pd.Series(None, index=index))
    out["excluded_invalid_exit"] = reason.isna() | reason.isin(INVALID_REASONS)
    out["excluded_unfillable_tp"] = out.get(
        "tp_unfillable", pd.Series(False, index=index)).fillna(False).astype(bool)
    out["excluded_small_tp"] = pd.to_numeric(
        out.get("tp_ticks", pd.Series(np.inf, index=index)), errors="coerce"
    ).fillna(-np.inf) < MIN_EXECUTABLE_TP_TICKS
    out["excluded_small_sl"] = pd.to_numeric(
        out.get("r_ticks", pd.Series(np.inf, index=index)), errors="coerce"
    ).fillna(-np.inf) < MIN_EXECUTABLE_SL_TICKS
    out["excluded_high_friction"] = pd.to_numeric(
        out.get("cost_r", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(np.inf) > MAX_FRICTION_R

    if "contracts" not in out.columns and {"instrument", "r_ticks"}.issubset(out.columns):
        out["contracts"] = 0
        for sym, idx in out.groupby("instrument", observed=True).groups.items():
            out.loc[idx, "contracts"] = contracts_for(
                out.loc[idx, "r_ticks"].to_numpy(), str(sym))
    contracts = pd.to_numeric(
        out.get("contracts", pd.Series(1, index=index)), errors="coerce")
    out["excluded_unaffordable"] = contracts.fillna(0) <= 0

    changed_in = out.get(
        "contract_changed_in_session", pd.Series(False, index=index))
    changed_prev = out.get(
        "contract_changed_since_prev_session", pd.Series(False, index=index))
    out["excluded_roll_day"] = (
        changed_in.fillna(False).astype(bool)
        | changed_prev.fillna(False).astype(bool))

    completeness = pd.to_numeric(out.get(
        "session_bar_completeness", pd.Series(1.0, index=index)), errors="coerce")
    out["excluded_incomplete_session"] = (
        completeness.fillna(0.0) < MIN_SESSION_BAR_COMPLETENESS)
    out["eligible"] = ~out[list(RULE_COLUMNS)].any(axis=1)
    return out


def exclusion_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count each rule independently; counts may overlap by design."""
    marked = trade_eligibility(df)
    return {column: int(marked[column].sum()) for column in RULE_COLUMNS}
