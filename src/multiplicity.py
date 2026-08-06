"""Joint-null family rollup, step-down maxT, and permutation FDR."""
from __future__ import annotations

import numpy as np
import pandas as pd


def step_down_max_t(observed: np.ndarray, null_t: np.ndarray) -> np.ndarray:
    """Westfall-Young step-down adjusted p-values with NaN eligibility."""
    observed = np.asarray(observed, dtype=float)
    null_t = np.asarray(null_t, dtype=float)
    out = np.full(len(observed), np.nan)
    eligible = np.flatnonzero(np.isfinite(observed))
    if len(eligible) == 0:
        return out
    order = eligible[np.argsort(-observed[eligible], kind="mergesort")]
    ordered_null = np.where(np.isfinite(null_t[:, order]), null_t[:, order], -np.inf)
    tail_max = np.maximum.accumulate(ordered_null[:, ::-1], axis=1)[:, ::-1]
    raw = np.empty(len(order), dtype=float)
    for rank, column in enumerate(order):
        valid_draw = np.any(np.isfinite(null_t[:, order[rank:]]), axis=1)
        k = int(valid_draw.sum())
        raw[rank] = ((1 + np.sum(tail_max[valid_draw, rank] >= observed[column]))
                     / (1 + k)) if k else np.nan
    adjusted = np.maximum.accumulate(np.nan_to_num(raw, nan=1.0))
    out[order] = adjusted
    return out


def permutation_fdr(
    observed: np.ndarray, null_t: np.ndarray, lam: float = 0.5,
) -> np.ndarray:
    """Direct statistic-scale FDR estimate from joint permutation draws."""
    observed = np.asarray(observed, dtype=float)
    null_t = np.asarray(null_t, dtype=float)
    out = np.full(len(observed), np.nan)
    eligible = np.flatnonzero(np.isfinite(observed))
    pooled = null_t[np.isfinite(null_t)]
    if len(eligible) == 0 or len(pooled) == 0:
        return out
    q_lambda = float(np.quantile(pooled, lam))
    pi0 = float(np.clip(
        np.sum(observed[eligible] <= q_lambda) / (lam * len(eligible)), 0.0, 1.0))
    order = eligible[np.argsort(-observed[eligible], kind="mergesort")]
    fdr = np.empty(len(order), dtype=float)
    for rank, column in enumerate(order):
        threshold = observed[column]
        false_per_draw = np.sum(null_t >= threshold, axis=1)
        v_hat = float(np.nanmean(false_per_draw))
        rejections = int(np.sum(observed[eligible] >= threshold))
        fdr[rank] = min(1.0, pi0 * v_hat / max(rejections, 1))
    out[order] = np.minimum.accumulate(fdr[::-1])[::-1]
    return out


def _family_rollup(
    frame: pd.DataFrame,
    null_variant: np.ndarray,
    observed_variant: np.ndarray,
    family_keys: list[str],
) -> tuple[list[tuple], np.ndarray, np.ndarray, np.ndarray]:
    families = [tuple(x) for x in frame[family_keys].itertuples(index=False, name=None)]
    unique = list(dict.fromkeys(families))
    family_index = {key: i for i, key in enumerate(unique)}
    membership = np.asarray([family_index[key] for key in families], dtype=int)
    observed = np.full(len(unique), np.nan)
    null = np.full((null_variant.shape[0], len(unique)), np.nan, dtype=float)
    for i in range(len(unique)):
        columns = np.flatnonzero(membership == i)
        values = observed_variant[columns]
        if np.isfinite(values).any():
            observed[i] = float(np.nanmax(values))
        values = null_variant[:, columns]
        finite = np.isfinite(values)
        rolled = np.max(np.where(finite, values, -np.inf), axis=1)
        null[:, i] = np.where(finite.any(axis=1), rolled, np.nan)
    return unique, membership, observed, null


def attach_multiplicity(
    frame: pd.DataFrame,
    null_variant: np.ndarray,
    observed_variant: np.ndarray,
    *,
    family_keys: list[str],
) -> pd.DataFrame:
    """Attach family-primary and variant-supplemental multiplicity results."""
    out = frame.copy()
    families, membership, family_obs, family_null = _family_rollup(
        out, null_variant, observed_variant, family_keys)
    family_p = step_down_max_t(family_obs, family_null)
    family_q = permutation_fdr(family_obs, family_null)
    variant_p = step_down_max_t(observed_variant, null_variant)

    finite = np.isfinite(family_null)
    joint_max = np.max(np.where(finite, family_null, -np.inf), axis=1)
    joint_max = joint_max[np.isfinite(joint_max)]
    hurdle = float(np.percentile(joint_max, 95)) if len(joint_max) else np.nan
    selection_adjustment = (
        float(np.nanmax(family_obs) - np.mean(joint_max))
        if len(joint_max) and np.isfinite(family_obs).any() else np.nan)

    out["t_family_obs"] = family_obs[membership]
    out["p_adj_maxT"] = family_p[membership]
    out["q_fdr"] = family_q[membership]
    out["p_adj_maxT_variant"] = variant_p
    out["maxT_hurdle_95"] = hurdle
    out["selection_adjusted_best_stat"] = selection_adjustment
    out["hypothesis_families"] = len(families)
    out["hypothesis_variants"] = len(out)
    out["multiplicity_stat"] = "bootstrap_studentized_net_mean"
    out["multiplicity_effective_k"] = np.sum(np.isfinite(family_null), axis=0)[membership]

    window_count = out["n_windows_observed"] if "n_windows_observed" in out else pd.Series(10, index=out.index)
    windows_needed = np.ceil(0.7 * pd.to_numeric(
        window_count, errors="coerce")).fillna(7)
    top_share = out["top5pct_day_pnl_share"] if "top5pct_day_pnl_share" in out else pd.Series(np.inf, index=out.index)
    positive_windows = out["n_windows_positive"] if "n_windows_positive" in out else pd.Series(0, index=out.index)
    out["breadth_pass"] = (
        pd.to_numeric(top_share, errors="coerce").le(0.40)
        & pd.to_numeric(positive_windows, errors="coerce").ge(windows_needed)
    )
    out["survivor"] = out["p_adj_maxT"].le(0.05) & out["breadth_pass"]
    out.attrs["maxT_null_distribution"] = joint_max
    out.attrs["T_family"] = family_null.astype(np.float32, copy=False)
    out.attrs["T_variant"] = np.asarray(null_variant, dtype=np.float32)
    out.attrs["family_keys"] = families
    return out
