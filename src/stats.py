"""
stats.py — compute per-variant and aggregated performance metrics.

Metrics reported:
  win_rate, expectancy_r (gross + net), profit_factor,
  trade_count, avg_mae_r, avg_mfe_r, max_drawdown_r,
  sharpe_r, sortino_r, calmar_r
  + bootstrap 95% CI on expectancy (block-bootstrap, block=5 days).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from typing import Any

from src.config import BOOTSTRAP_BLOCK_SIZE_DAYS, BOOTSTRAP_N

log = logging.getLogger("orb.stats")


VARIANT_KEYS = [
    "instrument", "session", "range_minutes",
    "entry_mode", "closure_tf", "direction", "rr",
]


def _max_drawdown_r(r_series: np.ndarray) -> float:
    """Peak-to-trough drawdown on cumulative R series."""
    if len(r_series) == 0:
        return 0.0
    cumr = np.cumsum(r_series)
    peak = np.maximum.accumulate(cumr)
    dd   = peak - cumr
    return float(dd.max())


def _block_bootstrap_ci(r_series: np.ndarray, n_boot: int, block: int,
                         alpha: float = 0.05) -> tuple[float, float]:
    """Block-bootstrap confidence interval for mean R."""
    if len(r_series) < block:
        return (float(np.mean(r_series)),) * 2
    rng     = np.random.default_rng(0)
    n       = len(r_series)
    n_blocks = max(1, n // block)
    boots   = []
    for _ in range(n_boot):
        starts  = rng.integers(0, n - block + 1, size=n_blocks)
        sample  = np.concatenate([r_series[s:s + block] for s in starts])
        boots.append(float(np.mean(sample)))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def _compute_metrics(r_gross: np.ndarray, r_net: np.ndarray) -> dict[str, Any]:
    n = len(r_gross)
    if n == 0:
        return {"trade_count": 0}

    wins   = r_gross > 0
    losses = r_gross < 0
    wrate  = float(wins.mean())
    exp_g  = float(r_gross.mean())
    exp_n  = float(r_net.mean())

    gross_wins  = r_gross[wins].sum()
    gross_loss  = abs(r_gross[losses].sum())
    pf          = (gross_wins / gross_loss) if gross_loss > 0 else np.inf

    mdd       = _max_drawdown_r(r_gross)
    calmar    = (exp_g * 252) / mdd if mdd > 0 else np.nan

    std_g = float(r_gross.std())
    sharpe = float(r_gross.mean() / std_g * np.sqrt(252)) if std_g > 0 else np.nan

    down   = r_gross[r_gross < 0]
    sortino_denom = float(np.sqrt((down**2).mean())) if len(down) > 0 else 0
    sortino = float(r_gross.mean() / sortino_denom * np.sqrt(252)) if sortino_denom > 0 else np.nan

    ci_lo, ci_hi = _block_bootstrap_ci(r_gross, BOOTSTRAP_N, BOOTSTRAP_BLOCK_SIZE_DAYS)

    return {
        "trade_count":       n,
        "win_rate":          round(wrate, 4),
        "expectancy_gross_r": round(exp_g, 4),
        "expectancy_net_r":   round(exp_n, 4),
        "profit_factor":      round(pf, 4) if not np.isinf(pf) else 9999.0,
        "max_drawdown_r":     round(mdd, 4),
        "avg_mae_r":          0.0,    # filled in separately from trade log
        "avg_mfe_r":          0.0,
        "sharpe_r":           round(sharpe, 4) if not np.isnan(sharpe) else None,
        "sortino_r":          round(sortino, 4) if not np.isnan(sortino) else None,
        "calmar_r":           round(calmar, 4) if not np.isnan(calmar) else None,
        "ci_lo_95":           round(ci_lo, 4),
        "ci_hi_95":           round(ci_hi, 4),
    }


def _drop_unfillable(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Remove trades whose take-profit sits inside MIN_TP_TICKS of entry.

    Such a TP cannot fill -- it is inside the spread -- yet the exit walk
    records it as a win, so leaving it in inflates gross expectancy and win
    rate. trade_sim flags these as tp_unfillable; this is where the flag is
    actually applied.

    It was previously flagged and never filtered, which inflated every low-rr
    variant. Measured effect at rr=0.25: 75-83% of ZN trades were unfillable,
    and excluding them cut gross expectancy from 0.2321 to 0.0855 (-63%).
    rr >= 1.0 variants were unaffected (0.0% unfillable), so the distortion was
    concentrated entirely in tight-target variants.
    """
    if "tp_unfillable" not in df.columns:
        return df
    mask = df["tp_unfillable"].fillna(False).astype(bool)
    n = int(mask.sum())
    if n:
        log.info("%sexcluded %d unfillable-TP trades (%.2f%% of %d)",
                 f"{label}: " if label else "", n, 100.0 * n / len(df), len(df))
    return df.loc[~mask]


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the full trade log DataFrame, compute per-variant metrics
    (aggregated across all regime windows) plus a per-window breakdown.

    Excludes INVALID exits and unfillable take-profits. n_unfillable_excluded
    records how many were dropped per variant so the exclusion stays visible.
    """
    # Filter out INVALID rows
    valid = df[df["exit_reason"].notna() & (df["exit_reason"] != "INVALID")].copy()

    # Count per-variant exclusions before dropping, so they can be reported.
    if "tp_unfillable" in valid.columns:
        excl = (valid.assign(_u=valid["tp_unfillable"].fillna(False).astype(bool))
                     .groupby(VARIANT_KEYS, observed=True)["_u"].sum())
    else:
        excl = None

    valid = _drop_unfillable(valid, "compute_summary")
    rows  = []

    for keys, grp in valid.groupby(VARIANT_KEYS, observed=True, sort=True):
        r_g = grp["gross_r"].to_numpy(copy=True)
        r_n = grp["net_r"].to_numpy(copy=True)
        m   = _compute_metrics(r_g, r_n)
        kt  = keys if isinstance(keys, tuple) else (keys,)
        m.update(dict(zip(VARIANT_KEYS, kt)))
        m["avg_mae_r"] = round(float(grp["mae_r"].mean()), 4)
        m["avg_mfe_r"] = round(float(grp["mfe_r"].mean()), 4)
        # How many trades this variant lost to the unfillable-TP rule. Kept
        # visible so a variant whose sample was gutted cannot look clean.
        m["n_unfillable_excluded"] = (
            int(excl.get(kt, 0)) if excl is not None else 0)
        rows.append(m)

    return pd.DataFrame(rows)


def compute_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as compute_summary but broken out per regime window.

    Applies the same unfillable-TP exclusion, otherwise per-window expectancy
    would disagree with the aggregate for no visible reason.
    """
    valid = df[df["exit_reason"].notna() & (df["exit_reason"] != "INVALID")].copy()
    valid = _drop_unfillable(valid, "compute_regime_summary")
    rows  = []
    for keys, grp in valid.groupby(VARIANT_KEYS + ["regime_window"],
                                    observed=True, sort=True):
        r_g = grp["gross_r"].to_numpy(copy=True)
        r_n = grp["net_r"].to_numpy(copy=True)
        m   = _compute_metrics(r_g, r_n)
        key_names = VARIANT_KEYS + ["regime_window"]
        m.update(dict(zip(key_names, keys if isinstance(keys, tuple) else [keys])))
        rows.append(m)
    return pd.DataFrame(rows)
