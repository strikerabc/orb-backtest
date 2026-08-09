"""Ignored, restartable CPU/CUDA orchestration for the full ORB sweep."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

# Prevent each worker from starting its own BLAS thread pool.
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Repo root. This file lives at src/engine/sweep.py, so the root is two parents
# up -- parents[2], not .parent.parent. It was .parent.parent when the file sat in
# runs/; moving it without adjusting this would have put src/ on sys.path and
# written every artifact to src/outputs and src/runs instead of the repo root,
# silently, with the sweep appearing to work.
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
WORK = RUNS / "gpu_cpu_sweep"
SHARDS = WORK / "trade_shards"
STATS = WORK / "stats"
POOLS = WORK / "null_pools"
GPU = WORK / "gpu"
STATUS_PATH = RUNS / "gpu_cpu_sweep.status.json"

sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

config.USE_LOCAL_DATA = True

from src.config import (  # noqa: E402
    BOOTSTRAP_BLOCK_SIZE_DAYS, INSTRUMENTS, NULL_BOOTSTRAP_N,
    INVALID_REASONS, MIN_SESSION_BAR_COMPLETENESS, NULL_MIN_DAYS,
    NULL_SAMPLE_DAYS, NULL_STRATIFY_BY_WINDOW, OUTPUTS_DIR, RR_LEVELS,
    SESSIONS,
)
from src.entry_detector import detect_entries  # noqa: E402
from src.filters import trade_eligibility  # noqa: E402
from src.journal import build_row  # noqa: E402
from src.multiplicity import attach_multiplicity  # noqa: E402
from src.null_calibrator import (  # noqa: E402
    POOL_KEYS, NullPool, _bootstrap_null_means, _eligible_results,
    _jackknife_null_mean_se, _matched_entry_signal, _random_entry_signal,
    _seed, build_r_ticks_map, null_p_value, sample_null_days,
    write_null_artifacts,
)
from src.range_builder import SessionDay, build_session_days  # noqa: E402
from src.regime_sampler import filter_to_window, select_windows  # noqa: E402
from src.report import write_report  # noqa: E402
from src.sizing import (  # noqa: E402
    MAX_FRICTION_R, MIN_EXECUTABLE_SL_TICKS, MIN_EXECUTABLE_TP_TICKS,
    contracts_for,
)
from src.stats import compute_regime_summary, compute_summary  # noqa: E402
from src.trade_sim import simulate_trade  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(processName)-18s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("orb.gpu_cpu")


def _status(stage: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "pid": os.getpid(),
        **extra,
    }
    content = json.dumps(payload, indent=2, default=str)
    tmp = STATUS_PATH.with_name(f"{STATUS_PATH.stem}.{os.getpid()}.tmp")
    for attempt in range(4):
        try:
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, STATUS_PATH)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    try:
        STATUS_PATH.write_text(content, encoding="utf-8")
    except OSError as exc:
        log.warning("Could not update non-critical status file: %s", exc)


def _load_symbol_context(sym: str) -> tuple[dict[str, list[SessionDay]], dict[int, object], list[pd.DataFrame]]:
    # Import after config.USE_LOCAL_DATA is set because data_layer binds it at import.
    from src.data_layer import (
        _compute_enrichment, ensure_daily, ensure_data, vendor_boundary_diagnostics,
    )

    df_1m = ensure_data(sym)
    df_1d = ensure_daily(sym)
    df = _compute_enrichment(df_1m, df_1d, tick_size=INSTRUMENTS[sym]["tick_size"])
    windows = select_windows(df["timestamp"].min().date(),
                             df["timestamp"].max().date(), label=sym)
    by_session: dict[str, list[SessionDay]] = {}
    for sess in INSTRUMENTS[sym].get("sessions", list(SESSIONS)):
        days: list[SessionDay] = []
        for window in windows:
            frame = filter_to_window(df, window)
            if frame.empty:
                continue
            current = build_session_days(frame, sym, sess)
            for day in current:
                day.regime_window = window.index
            days.extend(current)
        by_session[sess] = days
    diag = vendor_boundary_diagnostics(df_1m, sym)
    del df_1m, df_1d, df
    return by_session, {window.index: window for window in windows}, ([] if diag.empty else [diag])


def _schema_reference() -> pa.Schema | None:
    candidate = ROOT / OUTPUTS_DIR / "trade_log.parquet"
    if not candidate.exists():
        return None
    schema = pq.read_schema(candidate)
    required = {
        "context_bars_available", "contract_changed_in_session",
        "contract_changed_since_prev_session", "contracts", "cost_r",
        "entry_contract", "entry_source", "fill_at_bar_close",
        "gross_r_optimistic", "pct_bars_from_local",
        "session_bar_completeness",
    }
    return schema if required.issubset(schema.names) else None


def _write_rows(
    writer: pq.ParquetWriter | None,
    rows: list[dict],
    path: Path,
    schema: pa.Schema | None,
) -> tuple[pq.ParquetWriter, pa.Schema, int]:
    if schema is None:
        table = pa.Table.from_pylist(rows)
        schema = table.schema
    else:
        table = pa.Table.from_pylist(rows, schema=schema)
    if writer is None:
        writer = pq.ParquetWriter(path, schema, compression="zstd")
    writer.write_table(table)
    return writer, schema, len(rows)


def simulate_symbol(sym: str, schema: pa.Schema | None, batch_rows: int = 50_000) -> dict:
    started = time.perf_counter()
    path = SHARDS / f"{sym}.parquet"
    by_session, windows, diagnostics = _load_symbol_context(sym)

    writer: pq.ParquetWriter | None = None
    rows: list[dict] = []
    total = 0
    try:
        for sess, days in by_session.items():
            log.info("SIM %s/%s: %d session-days", sym, sess, len(days))
            for i, day in enumerate(days, 1):
                window = windows[int(day.regime_window)]
                for signal in detect_entries(day):
                    for trade in simulate_trade(signal, day, RR_LEVELS):
                        rows.append(build_row(
                            signal, trade, day, window, day.bars_h, day.bars_l))
                if len(rows) >= batch_rows:
                    writer, schema, written = _write_rows(writer, rows, path, schema)
                    total += written
                    rows.clear()
                if i % 250 == 0:
                    log.info("SIM %s/%s: %d/%d days, %d rows", sym, sess, i, len(days), total + len(rows))
        if rows:
            writer, schema, written = _write_rows(writer, rows, path, schema)
            total += written
    finally:
        if writer is not None:
            writer.close()
    if total == 0:
        raise RuntimeError(f"No trade rows generated for {sym}")
    if diagnostics:
        pd.concat(diagnostics, ignore_index=True).to_parquet(
            STATS / f"{sym}.boundary.parquet", index=False)
    elapsed = time.perf_counter() - started
    log.info("SIM %s complete: %d rows in %.1fs", sym, total, elapsed)
    return {"symbol": sym, "rows": total, "seconds": elapsed}


def summarize_symbol(sym: str) -> dict:
    started = time.perf_counter()
    trades = pd.read_parquet(SHARDS / f"{sym}.parquet")
    summary = compute_summary(trades)
    regimes = compute_regime_summary(trades)
    summary.to_parquet(STATS / f"{sym}.summary.parquet", index=False)
    regimes.to_parquet(STATS / f"{sym}.regime.parquet", index=False)
    result = {
        "symbol": sym, "variants": len(summary), "regime_rows": len(regimes),
        "seconds": time.perf_counter() - started,
    }
    log.info("STATS %s complete: %d variants in %.1fs", sym, len(summary), result["seconds"])
    return result


def _trade_is_eligible(day: SessionDay, trade) -> bool:
    """Scalar equivalent of filters.trade_eligibility for null simulations."""
    return bool(
        trade.exit_reason is not None
        and trade.exit_reason not in INVALID_REASONS
        and not bool(trade.tp_unfillable)
        and np.isfinite(trade.tp_ticks)
        and trade.tp_ticks >= MIN_EXECUTABLE_TP_TICKS
        and np.isfinite(trade.r_ticks)
        and trade.r_ticks >= MIN_EXECUTABLE_SL_TICKS
        and np.isfinite(trade.cost_r)
        and trade.cost_r <= MAX_FRICTION_R
        and int(contracts_for([trade.r_ticks], day.instrument)[0]) > 0
        and not day.contract_changed_in_session
        and not day.contract_changed_since_prev_session
        and np.isfinite(day.session_bar_completeness)
        and day.session_bar_completeness >= MIN_SESSION_BAR_COMPLETENESS
    )


def _build_matched_pools_fast(
    days: list[SessionDay], range_minutes: int, direction: str,
    r_ticks_pool: np.ndarray, seed: int,
) -> dict[float, NullPool]:
    rng = np.random.default_rng(seed)
    per_day: dict[float, list[np.ndarray]] = {rr: [] for rr in RR_LEVELS}
    totals = {rr: 0 for rr in RR_LEVELS}
    for day in days:
        values: dict[float, list[float]] = {rr: [] for rr in RR_LEVELS}
        for _ in range(max(1, config.N_NULL_DRAWS_PER_DAY)):
            signal = _matched_entry_signal(
                day, range_minutes, direction, r_ticks_pool, rng)
            if signal is None:
                continue
            for trade in simulate_trade(signal, day, RR_LEVELS):
                if _trade_is_eligible(day, trade) and np.isfinite(trade.net_r):
                    values[float(trade.rr)].append(float(trade.net_r))
        for rr, current in values.items():
            if current:
                array = np.asarray(current, dtype=float)
                per_day[rr].append(array)
                totals[rr] += len(array)
    return {rr: NullPool(per_day[rr], totals[rr]) for rr in RR_LEVELS}


def _build_swing_pool_fast(
    days: list[SessionDay], range_minutes: int, direction: str,
    rr: float, seed: int,
) -> NullPool:
    rng = np.random.default_rng(seed)
    per_day: list[np.ndarray] = []
    total = 0
    for day in days:
        values: list[float] = []
        for _ in range(max(1, config.N_NULL_DRAWS_PER_DAY)):
            signal = _random_entry_signal(day, range_minutes, direction, rng)
            if signal is None:
                continue
            trade = simulate_trade(signal, day, [rr])[0]
            if _trade_is_eligible(day, trade) and np.isfinite(trade.net_r):
                values.append(float(trade.net_r))
        if values:
            array = np.asarray(values, dtype=float)
            per_day.append(array)
            total += len(array)
    return NullPool(per_day, total)


def _eligibility_self_test() -> None:
    rng = np.random.default_rng(2718)
    for _ in range(250):
        day = SimpleNamespace(
            instrument=str(rng.choice(["ES", "RTY", "ZN"])),
            contract_changed_in_session=bool(rng.integers(0, 8) == 0),
            contract_changed_since_prev_session=bool(rng.integers(0, 8) == 0),
            session_bar_completeness=float(rng.choice([0.5, 0.9, 1.0, np.nan])),
        )
        trade = SimpleNamespace(
            exit_reason=str(rng.choice(["TP", "SL", "TIME", "INVALID"])),
            tp_unfillable=bool(rng.integers(0, 8) == 0),
            tp_ticks=float(rng.choice([0.5, 1.0, 2.0, np.nan])),
            r_ticks=float(rng.choice([1.0, 3.0, 4.0, 10.0, np.nan])),
            cost_r=float(rng.choice([0.1, 0.5, 2.0, np.nan])),
            rr=1.0,
            net_r=0.25,
        )
        canonical = _eligible_results(day, [trade])
        expected = bool(canonical.iloc[0]["eligible"])
        actual = _trade_is_eligible(day, trade)
        if actual != expected:
            raise AssertionError(f"Scalar eligibility mismatch: {day}, {trade}")
    log.info("Scalar eligibility parity passed for 250 deterministic cases")


def build_null_pools_symbol(sym: str) -> dict:
    started = time.perf_counter()
    trades = pd.read_parquet(SHARDS / f"{sym}.parquet")
    observed = trade_eligibility(trades)
    observed = observed[observed["eligible"]]
    summary = pd.read_parquet(STATS / f"{sym}.summary.parquet")
    by_session, _, _ = _load_symbol_context(sym)
    r_map = build_r_ticks_map(trades)

    family_meta: dict[tuple, tuple[set, dict]] = {}
    for key, grp in observed.groupby(POOL_KEYS, observed=True):
        dates = set(pd.to_datetime(grp["date"]).dt.date)
        weights = (grp.drop_duplicates(["date", "regime_window"])
                     .groupby("regime_window").size().to_dict())
        family_meta[tuple(key)] = (dates, weights)

    families: dict[tuple, dict] = {}
    family_rows = summary.drop_duplicates(POOL_KEYS)
    for ordinal, row in enumerate(family_rows.itertuples(index=False), 1):
        family = tuple(getattr(row, key) for key in POOL_KEYS)
        _, sess, rm, _, _, direction = family
        all_days = list(by_session.get(str(sess), []))
        fired_dates, weights = family_meta.get(family, (set(), {}))
        sample_seed = _seed("days", sym, sess)
        broad_days, sampling = sample_null_days(
            all_days, NULL_SAMPLE_DAYS, np.random.default_rng(sample_seed),
            stratify_by_window=NULL_STRATIFY_BY_WINDOW,
            window_of=lambda day: day.regime_window,
            window_weights=weights,
        )
        matched_days = [day for day in all_days if day.local_date in fired_dates]
        r_pool = r_map.get(family)
        design = "matched-stop" if r_pool is not None and len(r_pool) else "swing-fallback"
        if design == "matched-stop":
            broad = _build_matched_pools_fast(
                broad_days, int(rm), str(direction), r_pool,
                _seed("joint-entry", "broad", sym, sess))
            matched = _build_matched_pools_fast(
                matched_days, int(rm), str(direction), r_pool,
                _seed("joint-entry", "fired", sym, sess))
        else:
            broad = {rr: _build_swing_pool_fast(
                broad_days, int(rm), str(direction), rr,
                _seed("joint-entry", "broad", sym, sess)) for rr in RR_LEVELS}
            matched = {rr: _build_swing_pool_fast(
                matched_days, int(rm), str(direction), rr,
                _seed("joint-entry", "fired", sym, sess)) for rr in RR_LEVELS}
        families[family] = {
            "broad": broad,
            "matched": matched,
            "design": design,
            "sampling": sampling,
            "sample_seed": sample_seed,
            "all_days": len(all_days),
            "broad_days": len(broad_days),
            "matched_days": len(matched_days),
        }
        if ordinal % 10 == 0 or ordinal == len(family_rows):
            log.info("NULL-POOL %s: %d/%d families", sym, ordinal, len(family_rows))

    target = POOLS / f"{sym}.pickle"
    tmp = target.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(families, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, target)
    elapsed = time.perf_counter() - started
    log.info("NULL-POOL %s complete: %d families in %.1fs", sym, len(families), elapsed)
    return {"symbol": sym, "families": len(families), "seconds": elapsed}


_BOOTSTRAP_KERNEL = r"""
extern "C" __global__
void block_means(
    const double* prefix, const long long* day_offsets,
    const int* starts, const int n_draws, const int n_blocks,
    const int block, const int n_obs, const double* fallback,
    const long long* fallback_offsets, double* output)
{
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= n_draws) return;
    int used = 0;
    double total = 0.0;
    for (int k = 0; k < n_blocks && used < n_obs; ++k) {
        int start = starts[b * n_blocks + k];
        for (int j = 0; j < block && used < n_obs; ++j) {
            int day = start + j;
            long long lo = day_offsets[day];
            long long hi = day_offsets[day + 1];
            int take = (int)(hi - lo);
            if (take > n_obs - used) take = n_obs - used;
            total += prefix[lo + take] - prefix[lo];
            used += take;
        }
    }
    long long flo = fallback_offsets[b];
    long long fhi = fallback_offsets[b + 1];
    for (long long i = flo; i < fhi && used < n_obs; ++i) {
        total += fallback[i];
        ++used;
    }
    output[b] = used == n_obs ? total / (double)n_obs : 0.0 / 0.0;
}
"""

_FAST_BOOTSTRAP_KERNEL = r"""
__device__ __forceinline__ unsigned long long next_u64(unsigned long long* state) {
    unsigned long long z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ int bounded_u64(unsigned long long* state, int bound) {
    return (int)__umul64hi(next_u64(state), (unsigned long long)bound);
}

extern "C" __global__
void block_means_device_rng(
    const double* values, const double* prefix, const long long* day_offsets,
    const int n_days, const int n_blocks, const int block, const int n_obs,
    const int n_values, const unsigned long long master_seed,
    const int first_draw, const int n_draws, double* output)
{
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b >= n_draws) return;
    unsigned long long draw = (unsigned long long)(first_draw + b);
    unsigned long long state = master_seed ^ ((draw + 1ULL) * 0xd2b74407b1ce6e93ULL);
    int used = 0;
    double total = 0.0;
    int start_bound = n_days - block + 1;
    for (int k = 0; k < n_blocks && used < n_obs; ++k) {
        int start = bounded_u64(&state, start_bound);
        for (int j = 0; j < block && used < n_obs; ++j) {
            int day = start + j;
            long long lo = day_offsets[day];
            long long hi = day_offsets[day + 1];
            int take = (int)(hi - lo);
            if (take > n_obs - used) take = n_obs - used;
            total += prefix[lo + take] - prefix[lo];
            used += take;
        }
    }
    while (used < n_obs) {
        total += values[bounded_u64(&state, n_values)];
        ++used;
    }
    output[b] = total / (double)n_obs;
}
"""


def _pool_arrays(pool: NullPool) -> tuple[np.ndarray, np.ndarray]:
    if not pool.per_day:
        return np.array([], dtype=np.float64), np.array([0], dtype=np.int64)
    values = np.concatenate(pool.per_day).astype(np.float64, copy=False)
    lengths = np.asarray([len(day) for day in pool.per_day], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    return values, offsets


def _gpu_bootstrap_exact(
    pool: NullPool,
    n_obs: int,
    n_boot: int,
    seed: int,
    *,
    batch_size: int = 500,
    checkpoint=None,
) -> np.ndarray:
    if pool.n_days == 0 or n_obs <= 0:
        return np.array([], dtype=np.float64)
    import cupy as cp

    values, offsets = _pool_arrays(pool)
    nd = pool.n_days
    block = max(1, min(BOOTSTRAP_BLOCK_SIZE_DAYS, nd))
    days_needed = int(np.ceil(n_obs / max(1.0, pool.n_trades / nd)))
    n_blocks = max(1, int(np.ceil(days_needed / block)))
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    d_prefix = cp.asarray(prefix)
    d_offsets = cp.asarray(offsets)
    kernel = cp.RawKernel(_BOOTSTRAP_KERNEL, "block_means")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    lengths = np.diff(offsets)

    for first in range(0, n_boot, batch_size):
        count = min(batch_size, n_boot - first)
        plans = np.empty((count, n_blocks), dtype=np.int32)
        fallback_offsets = np.zeros(count + 1, dtype=np.int64)
        fallback_parts: list[np.ndarray] = []
        for b in range(count):
            starts = rng.integers(0, nd - block + 1, size=n_blocks)
            plans[b] = starts
            selected = int(sum(lengths[start:start + block].sum() for start in starts))
            missing = max(0, n_obs - selected)
            if missing:
                fallback_parts.append(np.asarray(
                    rng.choice(values, missing, replace=True), dtype=np.float64))
            fallback_offsets[b + 1] = fallback_offsets[b] + missing
        fallback = (np.concatenate(fallback_parts) if fallback_parts
                    else np.empty(0, dtype=np.float64))
        d_plans = cp.asarray(plans.ravel())
        d_fallback = cp.asarray(fallback)
        d_fallback_offsets = cp.asarray(fallback_offsets)
        d_output = cp.empty(count, dtype=cp.float64)
        threads = 256
        kernel(
            ((count + threads - 1) // threads,), (threads,),
            (d_prefix, d_offsets, d_plans, np.int32(count), np.int32(n_blocks),
             np.int32(block), np.int32(n_obs), d_fallback,
             d_fallback_offsets, d_output),
        )
        means[first:first + count] = cp.asnumpy(d_output)
        if checkpoint is not None:
            checkpoint(first + count, means[:first + count])
    return means


def _gpu_bootstrap(
    pool: NullPool,
    n_obs: int,
    n_boot: int,
    seed: int,
    *,
    batch_size: int = 5_000,
    checkpoint=None,
) -> np.ndarray:
    """Exact block-bootstrap design with independent deterministic GPU streams."""
    if pool.n_days == 0 or n_obs <= 0:
        return np.array([], dtype=np.float64)
    import cupy as cp

    values, offsets = _pool_arrays(pool)
    nd = pool.n_days
    block = max(1, min(BOOTSTRAP_BLOCK_SIZE_DAYS, nd))
    days_needed = int(np.ceil(n_obs / max(1.0, pool.n_trades / nd)))
    n_blocks = max(1, int(np.ceil(days_needed / block)))
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    d_values = cp.asarray(values)
    d_prefix = cp.asarray(prefix)
    d_offsets = cp.asarray(offsets)
    kernel = cp.RawKernel(_FAST_BOOTSTRAP_KERNEL, "block_means_device_rng")
    means = np.empty(n_boot, dtype=np.float64)
    threads = 256
    for first in range(0, n_boot, batch_size):
        count = min(batch_size, n_boot - first)
        d_output = cp.empty(count, dtype=cp.float64)
        kernel(
            ((count + threads - 1) // threads,), (threads,),
            (d_values, d_prefix, d_offsets, np.int32(nd), np.int32(n_blocks),
             np.int32(block), np.int32(n_obs), np.int32(len(values)),
             np.uint64(seed), np.int32(first), np.int32(count), d_output),
        )
        means[first:first + count] = cp.asnumpy(d_output)
        if checkpoint is not None:
            checkpoint(first + count, means[:first + count])
    return means


def _gpu_self_test() -> None:
    fixtures = [
        (NullPool([np.array([1.0, -1.0]), np.array([0.5]), np.array([2.0, 3.0, -4.0])], 6), 5),
        (NullPool([np.array([0.1]), np.array([0.2]), np.array([0.3])], 3), 17),
        (NullPool([np.arange(i + 1, dtype=float) / 7 for i in range(9)], 45), 8),
    ]
    for index, (pool, n_obs) in enumerate(fixtures):
        seed = 900 + index
        expected = _bootstrap_null_means(
            pool, n_obs, 1_037, BOOTSTRAP_BLOCK_SIZE_DAYS, seed=seed)
        actual = _gpu_bootstrap_exact(pool, n_obs, 1_037, seed, batch_size=137)
        if not np.allclose(actual, expected, rtol=2e-13, atol=2e-13):
            delta = float(np.max(np.abs(actual - expected)))
            raise AssertionError(f"CPU/CUDA bootstrap mismatch in fixture {index}: {delta}")
    log.info("CPU/CUDA exact-plan parity passed for %d deterministic fixtures", len(fixtures))
    for index, (pool, n_obs) in enumerate(fixtures):
        seed = 1900 + index
        expected = _bootstrap_null_means(
            pool, n_obs, 30_000, BOOTSTRAP_BLOCK_SIZE_DAYS, seed=seed)
        actual = _gpu_bootstrap(pool, n_obs, 30_000, seed, batch_size=500)
        expected_mean, actual_mean = float(expected.mean()), float(actual.mean())
        expected_std, actual_std = float(expected.std()), float(actual.std())
        mean_tolerance = 6.0 * np.sqrt(
            (expected_std ** 2 + actual_std ** 2) / len(expected)) + 1e-12
        if abs(expected_mean - actual_mean) > mean_tolerance:
            raise AssertionError(
                f"CPU/CUDA bootstrap mean mismatch in fixture {index}: "
                f"{expected_mean} vs {actual_mean}")
        if expected_std and abs(actual_std / expected_std - 1.0) > 0.04:
            raise AssertionError(
                f"CPU/CUDA bootstrap dispersion mismatch in fixture {index}: "
                f"{expected_std} vs {actual_std}")
        for threshold in (-0.25, 0.0, 0.5):
            p_cpu = float(np.mean(expected > threshold))
            p_gpu = float(np.mean(actual > threshold))
            tolerance = 6.0 * np.sqrt(
                max(p_cpu * (1 - p_cpu), 1e-6) * 2 / len(expected)) + 0.002
            if abs(p_cpu - p_gpu) > tolerance:
                raise AssertionError(
                    f"CPU/CUDA tail mismatch in fixture {index} at {threshold}: "
                    f"{p_cpu} vs {p_gpu}")
    log.info("CPU/CUDA device-RNG distribution parity passed for %d fixtures", len(fixtures))


def _summary_hash(summary: pd.DataFrame, n_boot: int) -> str:
    keys = summary[[*POOL_KEYS, "rr", "trade_count", "expectancy_net_r"]]
    return hashlib.sha256(
        keys.to_csv(index=False).encode("utf-8") + str(n_boot).encode("ascii")
    ).hexdigest()


def _load_metadata(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    records: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            records[int(item["variant_id"])] = item
    return records


def _attach_multiplicity_gpu(
    frame: pd.DataFrame, t_variant: np.ndarray, observed: np.ndarray,
) -> pd.DataFrame:
    """Run family rollup, maxT and FDR on CUDA, matching src.multiplicity."""
    import cupy as cp

    out = frame.copy()
    family_tuples = [tuple(x) for x in out[POOL_KEYS].itertuples(index=False, name=None)]
    families = list(dict.fromkeys(family_tuples))
    family_index = {key: i for i, key in enumerate(families)}
    membership = np.asarray([family_index[key] for key in family_tuples], dtype=np.int32)
    d_variant = cp.asarray(t_variant)
    d_obs = cp.asarray(observed)
    family_null = cp.full((len(t_variant), len(families)), cp.nan, dtype=cp.float32)
    family_obs = cp.full(len(families), cp.nan, dtype=cp.float64)
    for i in range(len(families)):
        cols = np.flatnonzero(membership == i)
        family_null[:, i] = cp.nanmax(d_variant[:, cols], axis=1)
        family_obs[i] = cp.nanmax(d_obs[cols])

    tail_kernel = cp.RawKernel(r"""
    extern "C" __global__
    void tail_max(const float* values, const int rows, const int cols, float* output) {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if (row >= rows) return;
        float current = -1.0f / 0.0f;
        for (int col = cols - 1; col >= 0; --col) {
            float value = values[row * cols + col];
            if (value > current) current = value;
            output[row * cols + col] = current;
        }
    }
    """, "tail_max")

    def step_down(d_observed, d_null):
        result = cp.full(len(d_observed), cp.nan, dtype=cp.float64)
        eligible = cp.flatnonzero(cp.isfinite(d_observed))
        if len(eligible) == 0:
            return result
        order = eligible[cp.argsort(-d_observed[eligible], kind="stable")]
        ordered = cp.ascontiguousarray(cp.where(
            cp.isfinite(d_null[:, order]), d_null[:, order], -cp.inf), dtype=cp.float32)
        tail = cp.empty_like(ordered)
        threads = 256
        tail_kernel(
            ((ordered.shape[0] + threads - 1) // threads,), (threads,),
            (ordered, np.int32(ordered.shape[0]), np.int32(ordered.shape[1]), tail))
        raw = (1.0 + cp.sum(tail >= d_observed[order][None, :], axis=0)) / (1.0 + d_null.shape[0])
        result[order] = cp.asarray(np.maximum.accumulate(cp.asnumpy(raw)))
        return result

    family_p = step_down(family_obs, family_null)
    variant_p = step_down(d_obs, d_variant)
    eligible_family = cp.flatnonzero(cp.isfinite(family_obs))
    pooled = family_null[cp.isfinite(family_null)]
    family_q = cp.full(len(families), cp.nan, dtype=cp.float64)
    if len(eligible_family) and len(pooled):
        q_lambda = cp.quantile(pooled, 0.5)
        pi0 = cp.clip(cp.sum(family_obs[eligible_family] <= q_lambda)
                      / (0.5 * len(eligible_family)), 0.0, 1.0)
        order = eligible_family[cp.argsort(-family_obs[eligible_family], kind="stable")]
        thresholds = family_obs[order]
        # E[V(t)] is the pooled exceedance count divided by the number of draws.
        sorted_pooled = cp.sort(pooled)
        exceed = len(sorted_pooled) - cp.searchsorted(sorted_pooled, thresholds, side="left")
        v_hat = exceed / family_null.shape[0]
        rejections = cp.arange(1, len(order) + 1)
        raw_q = cp.minimum(1.0, pi0 * v_hat / rejections)
        family_q[order] = cp.asarray(
            np.minimum.accumulate(cp.asnumpy(raw_q)[::-1])[::-1])

    finite = cp.isfinite(family_null)
    joint_max = cp.max(cp.where(finite, family_null, -cp.inf), axis=1)
    joint_max = joint_max[cp.isfinite(joint_max)]
    hurdle = float(cp.percentile(joint_max, 95).get()) if len(joint_max) else np.nan
    fam_obs_np = cp.asnumpy(family_obs)
    fam_null_np = cp.asnumpy(family_null)
    joint_np = cp.asnumpy(joint_max)
    family_p_np = cp.asnumpy(family_p)
    family_q_np = cp.asnumpy(family_q)
    variant_p_np = cp.asnumpy(variant_p)
    cp.get_default_memory_pool().free_all_blocks()

    out["t_family_obs"] = fam_obs_np[membership]
    out["p_adj_maxT"] = family_p_np[membership]
    out["q_fdr"] = family_q_np[membership]
    out["p_adj_maxT_variant"] = variant_p_np
    out["maxT_hurdle_95"] = hurdle
    out["selection_adjusted_best_stat"] = (
        float(np.nanmax(fam_obs_np) - np.mean(joint_np)) if len(joint_np) else np.nan)
    out["hypothesis_families"] = len(families)
    out["hypothesis_variants"] = len(out)
    out["multiplicity_stat"] = "bootstrap_studentized_net_mean"
    out["multiplicity_effective_k"] = np.sum(np.isfinite(fam_null_np), axis=0)[membership]
    window_count = out.get("n_windows_observed", pd.Series(10, index=out.index))
    windows_needed = np.ceil(pd.to_numeric(window_count, errors="coerce") * 0.7).fillna(7)
    top_share = out.get("top5pct_day_pnl_share", pd.Series(np.inf, index=out.index))
    positive = out.get("n_windows_positive", pd.Series(0, index=out.index))
    out["breadth_pass"] = (
        pd.to_numeric(top_share, errors="coerce").le(0.40)
        & pd.to_numeric(positive, errors="coerce").ge(windows_needed))
    out["survivor"] = out["p_adj_maxT"].le(0.05) & out["breadth_pass"]
    out.attrs["maxT_null_distribution"] = joint_np
    out.attrs["T_family"] = fam_null_np
    out.attrs["T_variant"] = t_variant
    out.attrs["family_keys"] = families
    return out


def calibrate_gpu(summary: pd.DataFrame, symbols: list[str], n_boot: int) -> pd.DataFrame:
    GPU.mkdir(parents=True, exist_ok=True)
    signature = _summary_hash(summary, n_boot)
    state_path = GPU / "state.json"
    metadata_path = GPU / "metadata.jsonl"
    t_path = GPU / "T_variant.dat"
    shape = (n_boot, len(summary))
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state and state.get("signature") != signature:
        raise RuntimeError("GPU checkpoint belongs to a different summary/configuration; use --fresh")
    mode = "r+" if t_path.exists() else "w+"
    t_variant = np.memmap(t_path, dtype=np.float32, mode=mode, shape=shape)
    if mode == "w+":
        t_variant[:] = np.nan
        t_variant.flush()
    metadata = _load_metadata(metadata_path)
    pool_cache: dict[str, dict] = {}

    with metadata_path.open("a", encoding="utf-8", buffering=64 * 1024) as metadata_file:
        for variant_id, row in summary.iterrows():
            if int(variant_id) in metadata:
                continue
            sym = str(row["instrument"])
            if sym not in pool_cache:
                with (POOLS / f"{sym}.pickle").open("rb") as handle:
                    pool_cache = {sym: pickle.load(handle)}
            family = tuple(row[key] for key in POOL_KEYS)
            item = pool_cache[sym][family]
            rr = float(row["rr"])
            broad_pool = item["broad"].get(rr, NullPool([], 0))
            matched_pool = item["matched"].get(rr, NullPool([], 0))
            n_obs = int(row.get("trade_count", 0) or 0)
            obs = float(row.get("expectancy_net_r", np.nan))
            broad_means = _gpu_bootstrap(
                broad_pool, n_obs, n_boot, _seed("joint-bootstrap"))
            matched_means = _gpu_bootstrap(
                matched_pool, n_obs, n_boot, _seed("joint-bootstrap-fired"))
            center = float(np.mean(broad_means)) if len(broad_means) else np.nan
            scale = float(np.std(broad_means)) if len(broad_means) else np.nan
            if np.isfinite(scale) and scale > 0:
                t_variant[:, variant_id] = ((broad_means - center) / scale).astype(np.float32)
                observed_stat = (obs - center) / scale
            else:
                t_variant[:, variant_id] = np.nan
                observed_stat = np.nan
            record = {
                "variant_id": int(variant_id),
                "null_exp_mean": center,
                "null_exp_p95": float(np.percentile(broad_means, 95)) if len(broad_means) else np.nan,
                "null_p_broad": null_p_value(obs, broad_means),
                "null_p_matched": null_p_value(obs, matched_means),
                "null_p_value": null_p_value(obs, broad_means),
                "null_design": item["design"],
                "null_days_requested": NULL_SAMPLE_DAYS,
                "null_days_used": item["broad_days"],
                "null_days_exhausted": item["all_days"] < NULL_SAMPLE_DAYS,
                "null_day_sampling": item["sampling"],
                "null_pool_trades": broad_pool.n_trades,
                "null_effective_n": broad_pool.n_days,
                "null_seed": item["sample_seed"],
                "null_unreliable": broad_pool.n_days < NULL_MIN_DAYS,
                "null_mean_jackknife_se": _jackknife_null_mean_se(broad_pool),
                "null_matched_days_used": item["matched_days"],
                "null_bootstrap_k": len(broad_means),
                "observed_stat": observed_stat,
            }
            metadata_file.write(json.dumps(record, allow_nan=True) + "\n")
            metadata[int(variant_id)] = record
            if (variant_id + 1) % 25 == 0 or variant_id + 1 == len(summary):
                # A durable checkpoint every 25 variants avoids the high cost of
                # Windows replace/flush calls while bounding restart work tightly.
                t_variant.flush()
                metadata_file.flush()
                state_path.write_text(json.dumps({
                    "signature": signature,
                    "variant_id": int(variant_id),
                    "variants": len(summary),
                    "n_boot": n_boot,
                    "phase": "complete",
                }, indent=2), encoding="utf-8")
                _status("gpu_bootstrap", variant=int(variant_id + 1),
                        variants=len(summary), phase="checkpoint")
                log.info("GPU variant %d/%d complete", variant_id + 1, len(summary))

        t_variant.flush()
        metadata_file.flush()

    ordered = pd.DataFrame([metadata[i] for i in range(len(summary))]).set_index("variant_id")
    result = summary.join(ordered.drop(columns=["observed_stat"]))
    observed_stats = ordered["observed_stat"].to_numpy(dtype=float)
    result = _attach_multiplicity_gpu(result, t_variant, observed_stats)
    return result


def _run_parallel(fn, symbols: list[str], workers: int, stage: str, *extra) -> list[dict]:
    results: list[dict] = []
    _status(stage, completed=0, total=len(symbols), workers=workers)
    with ProcessPoolExecutor(max_workers=min(workers, len(symbols))) as executor:
        futures = {executor.submit(fn, sym, *extra): sym for sym in symbols}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _status(stage, completed=len(results), total=len(symbols), workers=workers, last=result)
    return results


def _combine_trade_shards(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        frames.append(pd.read_parquet(SHARDS / f"{sym}.parquet"))
    return pd.concat(frames, ignore_index=True)


def run(symbols: list[str], workers: int, n_boot: int, fresh: bool, dry_run: bool) -> None:
    started = time.perf_counter()
    if fresh and WORK.exists():
        resolved = WORK.resolve()
        if resolved.parent != RUNS.resolve():
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    for path in (SHARDS, STATS, POOLS, GPU):
        path.mkdir(parents=True, exist_ok=True)
    _status("starting", symbols=symbols, workers=workers, n_boot=n_boot)
    _gpu_self_test()
    _eligibility_self_test()

    missing_sim = [sym for sym in symbols if not (SHARDS / f"{sym}.parquet").exists()]
    if missing_sim:
        schema = _schema_reference()
        _run_parallel(simulate_symbol, missing_sim, workers, "simulation", schema)

    missing_stats = [sym for sym in symbols if not (STATS / f"{sym}.summary.parquet").exists()]
    if missing_stats:
        _run_parallel(summarize_symbol, missing_stats, workers, "statistics")
    summary = pd.concat([
        pd.read_parquet(STATS / f"{sym}.summary.parquet") for sym in symbols
    ], ignore_index=True)
    regime_summary = pd.concat([
        pd.read_parquet(STATS / f"{sym}.regime.parquet") for sym in symbols
    ], ignore_index=True)

    missing_pools = [sym for sym in symbols if not (POOLS / f"{sym}.pickle").exists()]
    if missing_pools:
        _run_parallel(build_null_pools_symbol, missing_pools, workers, "null_pool_construction")

    _status("gpu_calibration", variants=len(summary), n_boot=n_boot)
    summary = calibrate_gpu(summary, symbols, n_boot)
    output = (WORK / "dry_outputs") if dry_run else (ROOT / OUTPUTS_DIR)
    output.mkdir(parents=True, exist_ok=True)
    write_null_artifacts(summary, output)
    diagnostics = list(STATS.glob("*.boundary.parquet"))
    if diagnostics:
        pd.concat([pd.read_parquet(path) for path in diagnostics], ignore_index=True).to_csv(
            output / "vendor_boundary_diagnostics.csv", index=False)

    _status("final_report", variants=len(summary))
    trade_log = _combine_trade_shards(symbols)
    write_report(summary, regime_summary, trade_log, output)
    elapsed = time.perf_counter() - started
    _status("complete", seconds=elapsed, rows=len(trade_log), variants=len(summary))
    log.info("FULL SWEEP COMPLETE: %d rows, %d variants in %.1fs", len(trade_log), len(summary), elapsed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2 - 2))
    parser.add_argument("--symbols", nargs="+", default=list(INSTRUMENTS))
    parser.add_argument("--n-boot", type=int, default=NULL_BOOTSTRAP_N)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _gpu_self_test()
        _eligibility_self_test()
        return 0
    unknown = sorted(set(args.symbols) - set(INSTRUMENTS))
    if unknown:
        parser.error(f"unknown symbols: {unknown}")
    try:
        run(args.symbols, max(1, args.workers), max(1, args.n_boot), args.fresh, args.dry_run)
        return 0
    except BaseException as exc:
        log.exception("Sweep failed")
        _status("failed", error=repr(exc))
        return 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
