"""Per-variant performance, day-clustered inference, and breadth metrics."""
from __future__ import annotations

import logging
import zlib
from typing import Any

import numpy as np
import pandas as pd

from src.config import BOOTSTRAP_BLOCK_SIZE_DAYS, BOOTSTRAP_N, NULL_MASTER_SEED
from src.filters import RULE_COLUMNS, trade_eligibility

log = logging.getLogger("orb.stats")

VARIANT_KEYS = [
    "instrument", "session", "range_minutes",
    "entry_mode", "closure_tf", "direction", "rr",
]


def _max_drawdown_r(r_series: np.ndarray) -> float:
    if len(r_series) == 0:
        return 0.0
    cumr = np.cumsum(r_series)
    return float((np.maximum.accumulate(cumr) - cumr).max())


def _stable_seed(key: object) -> int:
    return int(np.random.SeedSequence([
        NULL_MASTER_SEED, zlib.crc32(repr(key).encode("utf-8")),
    ]).generate_state(1)[0])


def _block_bootstrap_ci(
    r_series: np.ndarray,
    dates: np.ndarray,
    n_boot: int,
    block: int,
    *,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap a per-trade mean by contiguous blocks of session-days."""
    frame = pd.DataFrame({"r": r_series, "date": pd.to_datetime(dates)})
    per_day = [g["r"].to_numpy() for _, g in frame.sort_values("date").groupby("date")]
    nd = len(per_day)
    if nd <= 1:
        mean = float(np.mean(r_series))
        return mean, mean
    width = max(1, min(block, nd))
    n_blocks = int(np.ceil(nd / width))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, nd - width + 1, size=n_blocks)
        sample = np.concatenate([
            per_day[d] for start in starts
            for d in range(start, min(start + width, nd))
        ])
        boots[b] = float(sample.mean())
    return tuple(float(x) for x in np.percentile(
        boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def day_clustered_t(r_series: np.ndarray, dates: np.ndarray) -> float:
    """Studentised mean with a session-day clustered standard error."""
    if len(r_series) == 0:
        return np.nan
    frame = pd.DataFrame({"r": r_series, "date": pd.to_datetime(dates)})
    daily = frame.groupby("date")["r"].agg(["sum", "count"])
    mean = float(np.mean(r_series))
    se = float(np.sqrt(np.square(daily["sum"] - daily["count"] * mean).sum())
               / len(r_series))
    return mean / se if se > 0 else np.nan


def _compute_metrics(
    r_gross: np.ndarray,
    r_net: np.ndarray,
    dates: np.ndarray,
    exit_reasons: np.ndarray,
    *,
    variant_key: object,
    fast: bool = False,
) -> dict[str, Any]:
    n = len(r_gross)
    if n == 0:
        return {"trade_count": 0}
    wins, losses = r_gross > 0, r_gross < 0
    gross_loss = abs(r_gross[losses].sum())
    pf = r_gross[wins].sum() / gross_loss if gross_loss > 0 else np.inf

    daily = (pd.DataFrame({"date": pd.to_datetime(dates), "gross": r_gross})
               .groupby("date", sort=True)["gross"].sum().to_numpy())
    mdd = _max_drawdown_r(daily)
    daily_std = float(daily.std())
    sharpe = float(daily.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else np.nan
    downside = daily[daily < 0]
    downside_dev = float(np.sqrt(np.mean(downside ** 2))) if len(downside) else 0.0
    sortino = float(daily.mean() / downside_dev * np.sqrt(252)) if downside_dev > 0 else np.nan
    calmar = float(daily.mean() * 252 / mdd) if mdd > 0 else np.nan

    if fast:
        ci_lo = ci_hi = np.nan
    else:
        ci_lo, ci_hi = _block_bootstrap_ci(
            r_gross, dates, BOOTSTRAP_N, BOOTSTRAP_BLOCK_SIZE_DAYS,
            seed=_stable_seed(variant_key),
        )
    return {
        "trade_count": n,
        "win_rate": round(float(wins.mean()), 4),
        "tp_hit_rate": round(float(np.mean(exit_reasons == "TP")), 4),
        "expectancy_gross_r": round(float(r_gross.mean()), 4),
        "expectancy_net_r": round(float(r_net.mean()), 4),
        "profit_factor": round(float(pf), 4) if np.isfinite(pf) else 9999.0,
        "max_drawdown_r": round(mdd, 4),
        "sharpe_r_daily_ann252": round(sharpe, 4) if np.isfinite(sharpe) else None,
        "sortino_r_daily_ann252": round(sortino, 4) if np.isfinite(sortino) else None,
        "calmar_r_daily_ann252": round(calmar, 4) if np.isfinite(calmar) else None,
        # Backward-compatible names now carry correctly daily-aggregated values.
        "sharpe_r": round(sharpe, 4) if np.isfinite(sharpe) else None,
        "sortino_r": round(sortino, 4) if np.isfinite(sortino) else None,
        "calmar_r": round(calmar, 4) if np.isfinite(calmar) else None,
        "t_cluster_net": round(day_clustered_t(r_net, dates), 4),
        "ci_lo_95": round(ci_lo, 4) if np.isfinite(ci_lo) else np.nan,
        "ci_hi_95": round(ci_hi, 4) if np.isfinite(ci_hi) else np.nan,
    }


def _max_consecutive_losing_days(daily: pd.Series) -> int:
    longest = run = 0
    for losing in daily.lt(0).to_numpy():
        run = run + 1 if losing else 0
        longest = max(longest, run)
    return longest


def _breadth_metrics(grp: pd.DataFrame, opportunity_days: int) -> dict[str, Any]:
    dates = pd.to_datetime(grp["date"])
    daily_frame = pd.DataFrame({"date": dates, "net": grp["net_r"].to_numpy()})
    daily = daily_frame.groupby("date", sort=True)["net"].sum()
    total = float(daily.sum())
    positive_total = float(daily.clip(lower=0).sum())
    denom = positive_total if positive_total > 0 else np.nan
    top_n = max(1, int(np.ceil(len(daily) * 0.05)))
    monthly = daily.groupby(daily.index.to_period("M")).sum()
    if "regime_window" in grp:
        window_exp = grp.groupby("regime_window")["net_r"].mean()
    else:
        window_exp = pd.Series(dtype=float)
    return {
        "n_distinct_days": int(len(daily)),
        "top1_day_pnl_share": float(daily.clip(lower=0).max() / denom) if np.isfinite(denom) else np.nan,
        "top5pct_day_pnl_share": float(daily.nlargest(top_n).clip(lower=0).sum() / denom) if np.isfinite(denom) else np.nan,
        "n_windows_positive": int(window_exp.gt(0).sum()),
        "n_windows_observed": int(len(window_exp)),
        "worst_window_expectancy": float(window_exp.min()) if len(window_exp) else np.nan,
        "pnl_share_from_single_month": float(monthly.clip(lower=0).max() / denom) if np.isfinite(denom) else np.nan,
        "median_daily_r": float(daily.median()),
        "mean_daily_r": float(daily.mean()),
        "frac_days_traded": float(len(daily) / opportunity_days) if opportunity_days else np.nan,
        "deployable_trade_count": int(len(grp)),
        "max_consecutive_losing_days": _max_consecutive_losing_days(daily),
        "expectancy_gross_r_optimistic": float(grp.get(
            "gross_r_optimistic", grp["gross_r"]).mean()),
    }


def _assert_unique_trades(df: pd.DataFrame) -> None:
    keys = [*VARIANT_KEYS, "date"]
    if all(k in df.columns for k in keys) and df.duplicated(keys).any():
        sample = df.loc[df.duplicated(keys, keep=False), keys].head().to_dict("records")
        raise AssertionError(f"Duplicate variant-days detected (overlapping windows?): {sample}")


def compute_summary(df: pd.DataFrame, *, fast: bool = False) -> pd.DataFrame:
    _assert_unique_trades(df)
    marked = trade_eligibility(df)
    opportunities = marked.groupby(["instrument", "session"])["date"].nunique()
    rows: list[dict[str, Any]] = []
    for keys, all_grp in marked.groupby(VARIANT_KEYS, observed=True, sort=True):
        grp = all_grp[all_grp["eligible"]]
        if grp.empty:
            continue
        r_g = grp["gross_r"].to_numpy(dtype=float)
        r_n = grp["net_r"].to_numpy(dtype=float)
        dates = grp["date"].to_numpy()
        m = _compute_metrics(
            r_g, r_n, dates, grp["exit_reason"].to_numpy(),
            variant_key=keys, fast=fast,
        )
        kt = keys if isinstance(keys, tuple) else (keys,)
        m.update(dict(zip(VARIANT_KEYS, kt)))
        m["avg_mae_r"] = round(float(grp["mae_r"].mean()), 4)
        m["avg_mfe_r"] = round(float(grp["mfe_r"].mean()), 4)
        for rule in RULE_COLUMNS:
            m[f"n_{rule}"] = int(all_grp[rule].sum())
        opp = int(opportunities.get((kt[0], kt[1]), 0))
        m.update(_breadth_metrics(grp, opp))
        if "pct_bars_from_local" in grp:
            m["pct_trades_from_local"] = float(grp["pct_bars_from_local"].mean())
        rows.append(m)
    return pd.DataFrame(rows)


def compute_regime_summary(df: pd.DataFrame, *, fast: bool = False) -> pd.DataFrame:
    _assert_unique_trades(df)
    marked = trade_eligibility(df)
    rows: list[dict[str, Any]] = []
    keys_all = VARIANT_KEYS + ["regime_window"]
    for keys, all_grp in marked.groupby(keys_all, observed=True, sort=True):
        grp = all_grp[all_grp["eligible"]]
        if grp.empty:
            continue
        m = _compute_metrics(
            grp["gross_r"].to_numpy(dtype=float),
            grp["net_r"].to_numpy(dtype=float),
            grp["date"].to_numpy(), grp["exit_reason"].to_numpy(),
            variant_key=keys, fast=fast,
        )
        m.update(dict(zip(keys_all, keys)))
        rows.append(m)
    return pd.DataFrame(rows)
