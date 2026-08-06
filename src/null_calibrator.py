"""Matched-stop random-entry calibration over reproducibly sampled days."""
from __future__ import annotations

import logging
import hashlib
import json
import subprocess
import zlib
from dataclasses import dataclass
from typing import Callable, Hashable
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BOOTSTRAP_BLOCK_SIZE_DAYS, N_NULL_DRAWS_PER_DAY, NULL_BOOTSTRAP_N,
    NULL_MASTER_SEED, NULL_MIN_DAYS, NULL_SAMPLE_DAYS,
    NULL_STRATIFY_BY_WINDOW, RR_LEVELS, SESSIONS,
)
from src.entry_detector import EntrySignal
from src.filters import trade_eligibility
from src.range_builder import SessionDay
from src.swing_detector import find_swing_high, find_swing_low
from src.trade_sim import simulate_trade

log = logging.getLogger("orb.null")


def _seed(*parts: object) -> int:
    words = [NULL_MASTER_SEED]
    words.extend(zlib.crc32(repr(p).encode("utf-8")) for p in parts)
    return int(np.random.SeedSequence(words).generate_state(1)[0])


@dataclass
class NullPool:
    per_day: list[np.ndarray]
    n_trades: int

    @property
    def n_days(self) -> int:
        return len(self.per_day)

    def flat(self) -> np.ndarray:
        return np.concatenate(self.per_day) if self.per_day else np.array([], dtype=float)


def sample_null_days(
    session_days: list[SessionDay],
    n_target: int,
    rng: np.random.Generator,
    *,
    stratify_by_window: bool = True,
    window_of: Callable[[SessionDay], Hashable] | None = None,
    window_weights: dict[Hashable, float] | None = None,
) -> tuple[list[SessionDay], str]:
    """Sample days without replacement, optionally matching observed era mix."""
    days = sorted(session_days, key=lambda sd: sd.local_date)
    if len(days) <= n_target:
        return days, "exhausted"
    if not stratify_by_window or window_of is None:
        idx = np.sort(rng.choice(len(days), size=n_target, replace=False))
        return [days[int(i)] for i in idx], "random"

    groups: dict[Hashable, list[SessionDay]] = {}
    for day in days:
        groups.setdefault(window_of(day), []).append(day)
    if len(groups) <= 1:
        idx = np.sort(rng.choice(len(days), size=n_target, replace=False))
        return [days[int(i)] for i in idx], "random"

    if window_weights:
        weights = {k: max(0.0, float(window_weights.get(k, 0.0))) for k in groups}
    else:
        weights = {k: float(len(v)) for k, v in groups.items()}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = {k: float(len(v)) for k, v in groups.items()}
        total_weight = sum(weights.values())

    raw = {k: n_target * weights[k] / total_weight for k in groups}
    alloc = {k: min(len(groups[k]), int(np.floor(raw[k]))) for k in groups}
    remaining = n_target - sum(alloc.values())
    order = sorted(groups, key=lambda k: (raw[k] - np.floor(raw[k]), repr(k)), reverse=True)
    while remaining:
        progressed = False
        for key in order:
            if alloc[key] < len(groups[key]):
                alloc[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    sampled: list[SessionDay] = []
    for key in sorted(groups, key=repr):
        group = groups[key]
        idx = rng.choice(len(group), size=alloc[key], replace=False)
        sampled.extend(group[int(i)] for i in idx)
    return sorted(sampled, key=lambda sd: sd.local_date), "stratified"


def _random_entry_signal(
    sd: SessionDay, rm: int, direction: str, rng: np.random.Generator,
) -> EntrySignal | None:
    if rm not in sd.range_highs or len(sd.bars_o) == 0:
        return None
    sess = SESSIONS[sd.session]
    open_min = sess["open"][0] * 60 + sess["open"][1]
    first = int(np.searchsorted(sd.bar_wall_mins, open_min + rm))
    if first >= len(sd.bars_o):
        return None
    idx = int(rng.integers(first, len(sd.bars_o)))
    fill = float(sd.bars_o[idx])
    is_long = direction == "long"
    if is_long:
        sl, back, source = find_swing_low(
            sd.bars_o, sd.bars_h, sd.bars_l, sd.bars_c, idx,
            sd.tick_size, sd.range_lows[rm], entry_price=fill)
        boundary = sd.range_highs[rm]
    else:
        sl, back, source = find_swing_high(
            sd.bars_o, sd.bars_h, sd.bars_l, sd.bars_c, idx,
            sd.tick_size, sd.range_highs[rm], entry_price=fill)
        boundary = sd.range_lows[rm]
    return EntrySignal(
        "NULL", 1, rm, direction, idx, fill, idx, None, boundary,
        sl, back, source, False, False)


def _matched_entry_signal(
    sd: SessionDay, rm: int, direction: str,
    r_ticks_pool: np.ndarray, rng: np.random.Generator,
) -> EntrySignal | None:
    if rm not in sd.range_highs or len(sd.bars_o) == 0 or len(r_ticks_pool) == 0:
        return None
    sess = SESSIONS[sd.session]
    open_min = sess["open"][0] * 60 + sess["open"][1]
    first = int(np.searchsorted(sd.bar_wall_mins, open_min + rm))
    if first >= len(sd.bars_o):
        return None
    idx = int(rng.integers(first, len(sd.bars_o)))
    fill = float(sd.bars_o[idx])
    r_ticks = float(rng.choice(r_ticks_pool))
    if r_ticks < 1:
        return None
    is_long = direction == "long"
    sl = fill - r_ticks * sd.tick_size if is_long else fill + r_ticks * sd.tick_size
    boundary = sd.range_highs[rm] if is_long else sd.range_lows[rm]
    return EntrySignal(
        "NULL-MATCHED", 1, rm, direction, idx, fill, idx, None, boundary,
        sl, 0, "matched", False, False)


def _eligible_results(sd: SessionDay, trades: list) -> pd.DataFrame:
    rows = [{
        "instrument": sd.instrument,
        "exit_reason": tr.exit_reason,
        "tp_unfillable": tr.tp_unfillable,
        "tp_ticks": tr.tp_ticks,
        "r_ticks": tr.r_ticks,
        "cost_r": tr.cost_r,
        "contract_changed_in_session": sd.contract_changed_in_session,
        "contract_changed_since_prev_session": sd.contract_changed_since_prev_session,
        "session_bar_completeness": sd.session_bar_completeness,
        "rr": tr.rr,
        "net_r": tr.net_r,
    } for tr in trades]
    return trade_eligibility(pd.DataFrame(rows)) if rows else pd.DataFrame()


def build_matched_null_pool(
    session_days: list[SessionDay],
    range_minutes: int,
    direction: str,
    r_ticks_pool: np.ndarray,
    rr_levels: list[float],
    draws_per_day: int = N_NULL_DRAWS_PER_DAY,
    seed: int | None = None,
) -> dict[float, NullPool]:
    rng = np.random.default_rng(_seed("matched") if seed is None else seed)
    per_day: dict[float, list[np.ndarray]] = {rr: [] for rr in rr_levels}
    totals = {rr: 0 for rr in rr_levels}
    for sd in session_days:
        vals: dict[float, list[float]] = {rr: [] for rr in rr_levels}
        for _ in range(max(1, draws_per_day)):
            signal = _matched_entry_signal(sd, range_minutes, direction, r_ticks_pool, rng)
            if signal is None:
                continue
            marked = _eligible_results(sd, simulate_trade(signal, sd, rr_levels))
            for row in marked.loc[marked.get("eligible", False)].itertuples():
                if np.isfinite(row.net_r):
                    vals[float(row.rr)].append(float(row.net_r))
        for rr, values in vals.items():
            if values:
                arr = np.asarray(values, dtype=float)
                per_day[rr].append(arr)
                totals[rr] += len(arr)
    return {rr: NullPool(per_day[rr], totals[rr]) for rr in rr_levels}


def build_null_pool(
    session_days: list[SessionDay], range_minutes: int, direction: str, rr: float,
    draws_per_day: int = N_NULL_DRAWS_PER_DAY, seed: int | None = None,
) -> NullPool:
    rng = np.random.default_rng(_seed("swing") if seed is None else seed)
    per_day: list[np.ndarray] = []
    total = 0
    for sd in session_days:
        values: list[float] = []
        for _ in range(max(1, draws_per_day)):
            signal = _random_entry_signal(sd, range_minutes, direction, rng)
            if signal is None:
                continue
            marked = _eligible_results(sd, simulate_trade(signal, sd, [rr]))
            if not marked.empty and bool(marked.iloc[0]["eligible"]):
                value = float(marked.iloc[0]["net_r"])
                if np.isfinite(value):
                    values.append(value)
        if values:
            arr = np.asarray(values)
            per_day.append(arr)
            total += len(arr)
    return NullPool(per_day, total)


def _bootstrap_null_means(
    pool: NullPool, n_obs: int, n_boot: int, block_days: int, seed: int = 7,
) -> np.ndarray:
    if pool.n_days == 0 or n_obs <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    nd = pool.n_days
    block = max(1, min(block_days, nd))
    days_needed = int(np.ceil(n_obs / max(1.0, pool.n_trades / nd)))
    n_blocks = max(1, int(np.ceil(days_needed / block)))
    flat = pool.flat()
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, nd - block + 1, size=n_blocks)
        sample = np.concatenate([
            pool.per_day[d] for start in starts
            for d in range(start, min(start + block, nd))
        ])
        if len(sample) < n_obs:
            sample = np.concatenate([
                sample, rng.choice(flat, n_obs - len(sample), replace=True)])
        means[b] = float(sample[:n_obs].mean())
    return means


def null_p_value(observed_expectancy: float, null_means: np.ndarray) -> float:
    if null_means is None or len(null_means) == 0:
        return np.nan
    gt = int(np.sum(null_means > observed_expectancy))
    eq = int(np.sum(null_means == observed_expectancy))
    return float((1 + gt + 0.5 * eq) / (1 + len(null_means)))


def _jackknife_null_mean_se(pool: NullPool) -> float:
    if pool.n_days < 2:
        return np.nan
    sums = np.asarray([x.sum() for x in pool.per_day], dtype=float)
    counts = np.asarray([len(x) for x in pool.per_day], dtype=float)
    estimates = (sums.sum() - sums) / (counts.sum() - counts)
    center = estimates.mean()
    return float(np.sqrt((len(estimates) - 1) / len(estimates)
                         * np.square(estimates - center).sum()))


POOL_KEYS = ["instrument", "session", "range_minutes",
             "entry_mode", "closure_tf", "direction"]


def build_r_ticks_map(trade_log: pd.DataFrame) -> dict[tuple, np.ndarray]:
    marked = trade_eligibility(trade_log)
    valid = marked[marked["eligible"]]
    out: dict[tuple, np.ndarray] = {}
    for keys, grp in valid.groupby(POOL_KEYS, observed=True):
        values = grp["r_ticks"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            out[tuple(keys)] = values
    return out


def enrich_summary_with_null(
    summary_df: pd.DataFrame,
    session_days_map: dict,
    r_ticks_map: dict | None = None,
    n_null_samples: int = NULL_SAMPLE_DAYS,
    n_boot: int = NULL_BOOTSTRAP_N,
    rr_levels: list[float] | None = None,
    trade_log: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add broad-opportunity and exact-fired-day matched-stop p-values."""
    if summary_df.empty:
        return summary_df
    rr_levels = rr_levels or list(RR_LEVELS)
    observed = trade_eligibility(trade_log) if trade_log is not None else None
    if observed is not None:
        observed = observed[observed["eligible"]]

    family_meta: dict[tuple, tuple[set, dict]] = {}
    if observed is not None:
        for key, grp in observed.groupby(POOL_KEYS, observed=True):
            dates = set(pd.to_datetime(grp["date"]).dt.date)
            weights = (grp.drop_duplicates(["date", "regime_window"])
                         .groupby("regime_window").size().to_dict()
                       if "regime_window" in grp else {})
            family_meta[tuple(key)] = (dates, weights)

    caches: dict[tuple, dict[float, NullPool]] = {}
    out: list[dict] = []
    null_draws: list[np.ndarray] = []
    observed_stats: list[float] = []

    for _, row in summary_df.iterrows():
        family = tuple(row[k] for k in POOL_KEYS)
        sym, sess, rm, _, _, direction = family
        rr = float(row["rr"])
        all_days = list(session_days_map.get((sym, sess), []))
        fired_dates, weights = family_meta.get(family, (set(), {}))
        sample_seed = _seed("days", sym, sess)
        broad_days, sampling = sample_null_days(
            all_days, n_null_samples, np.random.default_rng(sample_seed),
            stratify_by_window=NULL_STRATIFY_BY_WINDOW,
            window_of=lambda sd: sd.regime_window,
            window_weights=weights,
        )
        matched_days = [sd for sd in all_days if sd.local_date in fired_dates]
        r_pool = (r_ticks_map or {}).get(family)
        design = "matched-stop" if r_pool is not None and len(r_pool) else "swing-fallback"

        def pools_for(label: str, days: list[SessionDay]) -> dict[float, NullPool]:
            key = (label, family)
            if key not in caches:
                if design == "matched-stop":
                    caches[key] = build_matched_null_pool(
                        days, int(rm), str(direction), r_pool, rr_levels,
                        seed=_seed("joint-entry", label, sym, sess))
                else:
                    caches[key] = {level: build_null_pool(
                        days, int(rm), str(direction), level,
                        seed=_seed("joint-entry", label, sym, sess)) for level in rr_levels}
            return caches[key]

        broad_pool = pools_for("broad", broad_days).get(rr, NullPool([], 0))
        matched_pool = pools_for("fired", matched_days).get(rr, NullPool([], 0))
        n_obs = int(row.get("trade_count", 0) or 0)
        obs = float(row.get("expectancy_net_r", np.nan))
        broad_means = _bootstrap_null_means(
            broad_pool, n_obs, n_boot, BOOTSTRAP_BLOCK_SIZE_DAYS,
            seed=_seed("joint-bootstrap"))
        matched_means = _bootstrap_null_means(
            matched_pool, n_obs, n_boot, BOOTSTRAP_BLOCK_SIZE_DAYS,
            seed=_seed("joint-bootstrap-fired"))

        record = row.to_dict()
        record.update({
            "null_exp_mean": float(np.mean(broad_means)) if len(broad_means) else np.nan,
            "null_exp_p95": float(np.percentile(broad_means, 95)) if len(broad_means) else np.nan,
            "null_p_broad": null_p_value(obs, broad_means),
            "null_p_matched": null_p_value(obs, matched_means),
            "null_p_value": null_p_value(obs, broad_means),
            "null_design": design,
            "null_days_requested": n_null_samples,
            "null_days_used": len(broad_days),
            "null_days_exhausted": len(all_days) < n_null_samples,
            "null_day_sampling": sampling,
            "null_pool_trades": broad_pool.n_trades,
            "null_effective_n": broad_pool.n_days,
            "null_seed": sample_seed,
            "null_unreliable": broad_pool.n_days < NULL_MIN_DAYS,
            "null_mean_jackknife_se": _jackknife_null_mean_se(broad_pool),
            "null_matched_days_used": len(matched_days),
            "null_bootstrap_k": len(broad_means),
        })
        out.append(record)
        if len(broad_means) and np.std(broad_means) > 0:
            scale = float(np.std(broad_means))
            center = float(np.mean(broad_means))
            null_draws.append((broad_means - center) / scale)
            observed_stats.append((obs - center) / scale)
        else:
            null_draws.append(np.full(n_boot, np.nan))
            observed_stats.append(np.nan)

    result = pd.DataFrame(out)
    if null_draws:
        from src.multiplicity import attach_multiplicity
        result = attach_multiplicity(
            result, np.asarray(null_draws, dtype=np.float32).T,
            np.asarray(observed_stats), family_keys=POOL_KEYS)
    return result


def write_null_artifacts(summary: pd.DataFrame, output_dir: Path) -> None:
    """Persist joint null arrays and enough provenance to reproduce them."""
    if summary.empty or "null_p_broad" not in summary:
        return
    root = Path(output_dir) / "null"
    root.mkdir(parents=True, exist_ok=True)
    t_variant = summary.attrs.get("T_variant")
    t_family = summary.attrs.get("T_family")
    if t_variant is not None:
        np.save(root / "T_variant.npy", t_variant)
        np.save(root / "eligible.npy", np.isfinite(t_variant))
    if t_family is not None:
        np.save(root / "T_family.npy", t_family)

    index_cols = [*POOL_KEYS, "rr"]
    variant_index = summary[index_cols].reset_index(names="variant_id")
    variant_index.attrs = {}
    variant_index.to_parquet(root / "variant_index.parquet", index=False)
    p_cols = [c for c in (
        *index_cols, "null_p_broad", "null_p_matched", "p_adj_maxT", "q_fdr",
        "maxT_hurdle_95", "breadth_pass", "survivor",
    ) if c in summary]
    pvalues = summary[p_cols].copy()
    pvalues.attrs = {}
    pvalues.to_parquet(root / "null_pvalues.parquet", index=False)
    provenance_cols = [c for c in (
        *POOL_KEYS, "null_days_requested", "null_days_used", "null_days_exhausted",
        "null_day_sampling", "null_pool_trades", "null_effective_n", "null_seed",
        "null_mean_jackknife_se",
    ) if c in summary]
    provenance = summary[provenance_cols].copy()
    provenance.attrs = {}
    provenance.to_parquet(root / "null_provenance.parquet", index=False)

    from src import config
    resolved = {k: repr(v) for k, v in vars(config).items() if k.isupper()}
    config_hash = hashlib.sha256(json.dumps(
        resolved, sort_keys=True).encode("utf-8")).hexdigest()
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unknown"
    completed = int(t_variant.shape[0]) if t_variant is not None else 0
    manifest = {
        "git_sha": git_sha,
        "config_hash": config_hash,
        "master_seed": NULL_MASTER_SEED,
        "completed_draws": list(range(completed)),
        "checkpoint_interval": 500,
        "null_days_policy": "cap-at-available-without-replacement",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
