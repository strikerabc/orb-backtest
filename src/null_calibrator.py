"""
null_calibrator.py — random-entry benchmark and bootstrap CI.

Runs the same simulation machinery with randomised entry times to answer:
"could this variant's expectancy arise by chance alone?"

Why this file was rewritten
---------------------------
The original null_p_value() computed:

    mean(null_pool >= observed_expectancy)

where null_pool held ONE gross_r per session-day -- a pool of INDIVIDUAL trade
outcomes -- while observed_expectancy is a MEAN. Comparing a mean against
single draws is a category error and imposed a hard floor on the statistic.

Measured consequences on the 10-instrument sweep (6,168 rankable variants):

    corr(null_p_value, expectancy_gross_r) = -0.0977   <- ~zero
    corr(null_p_value, win_rate)           = +0.6594
    corr(null_p_value, rr)                 = -0.7384
    min null_p_value observed              =  0.1661

Applying the identical formula to each variant against its OWN trades (true
answer: "identical", valid test -> p ~ 0.5) gave corr(self_p, win_rate) =
+0.9878 and never once fell below 0.05. The statistic measured the random-entry
TP hit rate, tracking 1/(1+rr), so p < 0.05 was unreachable by construction.

Corrected design
----------------
1. Draw N_NULL_DRAWS_PER_DAY random entries per session-day -> null trade pool.
2. Block-bootstrap that pool AT THE OBSERVED VARIANT'S SAMPLE SIZE to build a
   distribution of null MEANS (like compared with like).
3. p = (1 + #{null_mean >= observed_mean}) / (1 + n_boot).
   The (r+1)/(n+1) convention avoids reporting p = 0, which finite resampling
   can never justify.

Blocks are sampled as contiguous runs of DAYS so that day-level serial
correlation survives the resample, consistent with stats._block_bootstrap_ci.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    BOOTSTRAP_BLOCK_SIZE_DAYS, N_NULL_DRAWS_PER_DAY, NULL_BOOTSTRAP_N,
    NULL_SAMPLE_DAYS, RR_LEVELS, SESSIONS,
)
from src.entry_detector import EntrySignal
from src.range_builder import SessionDay
from src.swing_detector import find_swing_high, find_swing_low
from src.trade_sim import simulate_trade

log = logging.getLogger("orb.null")


@dataclass
class NullPool:
    """Random-entry outcomes grouped by day, ready for block bootstrap."""
    per_day: list[np.ndarray]   # one array of gross_r per session-day
    n_trades: int               # total pooled trades

    @property
    def n_days(self) -> int:
        return len(self.per_day)

    def flat(self) -> np.ndarray:
        if not self.per_day:
            return np.array([], dtype=float)
        return np.concatenate(self.per_day)


def _random_entry_signal(sd: SessionDay, rm: int, direction: str,
                         rng: np.random.Generator) -> EntrySignal | None:
    """
    Random EntrySignal inside the session's post-range active window.

    Uses the same swing-based SL machinery as the real strategy so the only
    difference between null and observed is WHEN the entry occurs.
    """
    if rm not in sd.range_highs:
        return None
    n = len(sd.bars_o)
    if n == 0:
        return None

    tick = sd.tick_size
    rh = sd.range_highs[rm]
    rl = sd.range_lows[rm]
    boundary = rh if direction == "long" else rl
    is_long = direction == "long"

    sess = SESSIONS[sd.session]
    oh, om = sess["open"]
    open_min = oh * 60 + om
    range_end = int(np.searchsorted(sd.bar_wall_mins, open_min + rm))
    if range_end >= n:
        return None

    entry_idx = int(rng.integers(range_end, n))
    fill = float(sd.bars_o[entry_idx])
    if is_long:
        sl, slb, sls = find_swing_low(sd.bars_o, sd.bars_h, sd.bars_l,
                                      sd.bars_c, entry_idx, tick, rl)
    else:
        sl, slb, sls = find_swing_high(sd.bars_o, sd.bars_h, sd.bars_l,
                                       sd.bars_c, entry_idx, tick, rh)
    return EntrySignal(
        mode="NULL", closure_tf=1, range_minutes=rm,
        direction=direction, entry_bar_idx=entry_idx,
        fill_price=fill, breakout_bar_idx=entry_idx,
        tap_in_bar_idx=None, boundary=boundary,
        sl_price=sl, sl_bars_back=slb, sl_source=sls,
        gap_fill=False,
    )


def _matched_entry_signal(sd: SessionDay, rm: int, direction: str,
                          r_ticks_pool: np.ndarray,
                          rng: np.random.Generator) -> EntrySignal | None:
    """
    Random entry TIMING with the OBSERVED variant's stop distance.

    Why this replaced the swing-derived null
    ----------------------------------------
    _random_entry_signal derives its stop from the same swing machinery, but
    from a mid-session bar the nearest down-close cluster is much closer than
    it is at a range boundary. Measured on real variants:

        ES/NY/15m/CC/short    observed r=34 ticks   swing-null r=4 ticks
        NQ/NY/15m/CC/short    observed r=173        swing-null r=12
        median ratio 0.27x -- null stops are 3-4x TIGHTER

    A 4-tick stop on ES sits inside a typical 1-minute bar, so SL and TP land
    in the SAME bar constantly, and trade_sim resolves those as SL by design.
    Same-bar ambiguity ran 140x higher in the null than in the observed set
    (ES 0.05% -> 13.00%). The null was therefore losing to the conservative
    tiebreak rule, not to worse timing:

        swing-derived null, median gross : -0.1459
        matched-stop null, median gross  : +0.0084   <- a fair null sits at ~0

    That made null_p ANTI-conservative and produced null_p < 0.05 for 78% of
    rankable variants (5,599 of 7,140), which is not credible as real edges.

    Holding r fixed and randomising only the entry bar isolates the question
    the null was always meant to answer: does breakout TIMING beat arbitrary
    timing at the same risk geometry?

    r does not depend on rr (r = |entry - sl|, set before rr is applied), so a
    single pool serves every rr level for a variant.
    """
    if rm not in sd.range_highs:
        return None
    n = len(sd.bars_o)
    if n == 0 or r_ticks_pool is None or len(r_ticks_pool) == 0:
        return None

    sess = SESSIONS[sd.session]
    open_min = sess["open"][0] * 60 + sess["open"][1]
    range_end = int(np.searchsorted(sd.bar_wall_mins, open_min + rm))
    if range_end >= n:
        return None

    idx = int(rng.integers(range_end, n))
    fill = float(sd.bars_o[idx])
    r_t = float(rng.choice(r_ticks_pool))
    if r_t < 1:
        return None

    dist = r_t * sd.tick_size
    is_long = direction == "long"
    sl = fill - dist if is_long else fill + dist
    boundary = sd.range_highs[rm] if is_long else sd.range_lows[rm]

    return EntrySignal(
        mode="NULL-MATCHED", closure_tf=1, range_minutes=rm,
        direction=direction, entry_bar_idx=idx, fill_price=fill,
        breakout_bar_idx=idx, tap_in_bar_idx=None, boundary=boundary,
        sl_price=sl, sl_bars_back=0, sl_source="matched", gap_fill=False,
    )


def build_matched_null_pool(
    session_days: list[SessionDay],
    range_minutes: int,
    direction: str,
    r_ticks_pool: np.ndarray,
    rr_levels: list[float],
    draws_per_day: int = N_NULL_DRAWS_PER_DAY,
    seed: int = 99,
) -> dict[float, NullPool]:
    """
    Matched-stop null pools for ALL rr levels in one pass.

    simulate_trade evaluates every rr from a single entry, and r is
    rr-independent, so one traversal serves all six levels. Keeps the pool
    count near the swing-null design (~1350 vs ~900) instead of 6x higher.
    """
    rng = np.random.default_rng(seed)
    per_day: dict[float, list[np.ndarray]] = {rr: [] for rr in rr_levels}
    totals: dict[float, int] = {rr: 0 for rr in rr_levels}

    for sd in session_days:
        vals: dict[float, list[float]] = {rr: [] for rr in rr_levels}
        for _ in range(max(1, draws_per_day)):
            es = _matched_entry_signal(sd, range_minutes, direction,
                                       r_ticks_pool, rng)
            if es is None:
                continue
            for tr in simulate_trade(es, sd, rr_levels=rr_levels):
                if tr.exit_reason in ("INVALID", None):
                    continue
                g = tr.gross_r
                if g is not None and np.isfinite(g):
                    vals[tr.rr].append(float(g))
        for rr, v in vals.items():
            if v:
                arr = np.asarray(v, dtype=float)
                per_day[rr].append(arr)
                totals[rr] += len(arr)

    return {rr: NullPool(per_day=per_day[rr], n_trades=totals[rr])
            for rr in rr_levels}


def build_null_pool(
    session_days: list[SessionDay],
    range_minutes: int,
    direction: str,
    rr: float,
    draws_per_day: int = N_NULL_DRAWS_PER_DAY,
    seed: int = 99,
) -> NullPool:
    """
    Build a random-entry outcome pool, grouped by session-day.

    draws_per_day > 1 enriches the pool so the bootstrap has material to
    resample; with a single draw per day the pool is only as large as the
    number of session-days.
    """
    rng = np.random.default_rng(seed)
    per_day: list[np.ndarray] = []
    total = 0

    for sd in session_days:
        vals: list[float] = []
        for _ in range(max(1, draws_per_day)):
            es = _random_entry_signal(sd, range_minutes, direction, rng)
            if es is None:
                continue
            trades = simulate_trade(es, sd, rr_levels=[rr])
            if trades and trades[0].exit_reason not in ("INVALID", None):
                g = trades[0].gross_r
                if g is not None and np.isfinite(g):
                    vals.append(float(g))
        if vals:
            arr = np.asarray(vals, dtype=float)
            per_day.append(arr)
            total += len(arr)

    return NullPool(per_day=per_day, n_trades=total)


def _bootstrap_null_means(pool: NullPool, n_obs: int, n_boot: int,
                          block_days: int, seed: int = 7) -> np.ndarray:
    """
    Distribution of null MEANS at sample size n_obs.

    Resamples contiguous blocks of DAYS (preserving day-level serial
    correlation), concatenates their trades, and truncates to n_obs.
    """
    if pool.n_days == 0 or n_obs <= 0:
        return np.array([], dtype=float)

    rng = np.random.default_rng(seed)
    nd = pool.n_days
    block = max(1, min(block_days, nd))

    # trades per day varies; estimate days needed to reach n_obs
    avg_per_day = max(1.0, pool.n_trades / nd)
    days_needed = int(np.ceil(n_obs / avg_per_day))
    n_blocks = max(1, int(np.ceil(days_needed / block)))

    flat_fallback = pool.flat()
    means = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        starts = rng.integers(0, max(1, nd - block + 1), size=n_blocks)
        chunks = [pool.per_day[d]
                  for s in starts
                  for d in range(s, min(s + block, nd))]
        if not chunks:
            means[b] = np.nan
            continue
        sample = np.concatenate(chunks)
        if len(sample) < n_obs:
            # top up by iid draws from the flat pool
            extra = rng.choice(flat_fallback, size=n_obs - len(sample),
                               replace=True)
            sample = np.concatenate([sample, extra])
        means[b] = float(sample[:n_obs].mean())

    return means[np.isfinite(means)]


def null_p_value(observed_expectancy: float, null_means: np.ndarray) -> float:
    """
    One-sided Monte Carlo p-value: P(null mean >= observed mean).

    Two conventions applied:

    (r+1)/(n+1) -- p is never exactly 0, which finite resampling cannot
        justify.

    mid-p tie handling -- per-trade outcomes are discrete (+rr or -1), so
        means of n trades fall on a lattice of spacing (1+rr)/n and exact ties
        between observed and null means are common. Counting every tie as
        "null >= observed" biases p upward (conservative). Ties are therefore
        counted as one half.
    """
    if null_means is None or len(null_means) == 0:
        return float("nan")
    n = len(null_means)
    r_gt = int(np.sum(null_means > observed_expectancy))
    r_eq = int(np.sum(null_means == observed_expectancy))
    return float((1 + r_gt + 0.5 * r_eq) / (1 + n))


POOL_KEYS = ["instrument", "session", "range_minutes",
             "entry_mode", "closure_tf", "direction"]


def build_r_ticks_map(trade_log: pd.DataFrame) -> dict[tuple, np.ndarray]:
    """
    Observed r_ticks distribution per variant family, for the matched null.

    Keyed WITHOUT rr, because r = |entry - sl| is fixed before rr is applied,
    so all six rr levels of a variant share one stop distribution.

    entry_mode and closure_tf ARE in the key: a tap-in enters at the boundary
    where the nearest swing cluster is close, while a candle-close entry enters
    further out. Measured medians differ by two orders of magnitude
    (ZN TI = 2 ticks, NQ CC = 173 ticks), so collapsing them would defeat the
    matching.
    """
    df = trade_log
    if "exit_reason" in df.columns:
        df = df[df["exit_reason"].notna() & (df["exit_reason"] != "INVALID")]

    out: dict[tuple, np.ndarray] = {}
    for keys, grp in df.groupby(POOL_KEYS, observed=True):
        r = grp["r_ticks"].to_numpy(dtype=float)
        r = r[np.isfinite(r) & (r >= 1.0)]
        if len(r):
            out[tuple(keys)] = r
    log.info("r_ticks map: %d variant families", len(out))
    return out


def enrich_summary_with_null(
    summary_df: pd.DataFrame,
    session_days_map: dict,          # (sym, session) -> list[SessionDay]
    r_ticks_map: dict | None = None,  # from build_r_ticks_map(trade_log)
    n_null_samples: int = NULL_SAMPLE_DAYS,
    n_boot: int = NULL_BOOTSTRAP_N,
    rr_levels: list[float] | None = None,
) -> pd.DataFrame:
    """
    Add null_exp_mean, null_exp_p95, null_p_value, null_design columns.

    Uses the MATCHED-STOP null when r_ticks_map is supplied: random entry
    timing at the observed variant's own stop distance, so only timing differs.
    See _matched_entry_signal for why the swing-derived null was abandoned
    (140x same-bar-ambiguity inflation made it anti-conservative and produced
    null_p < 0.05 for 78% of variants).

    Falls back to the swing-derived null per row when a variant family has no
    observed r_ticks, and records which comparator was used in null_design so
    the distinction survives into the artefact.

    Pools are cached on (instrument, session, range_minutes, entry_mode,
    closure_tf, direction) with all rr levels built in one traversal.
    """
    if summary_df.empty:
        return summary_df

    rr_levels = rr_levels or list(RR_LEVELS)
    matched_cache: dict[tuple, dict[float, NullPool]] = {}
    swing_cache: dict[tuple, NullPool] = {}
    out_rows: list[dict] = []
    n_rows = len(summary_df)
    n_matched = n_swing = 0

    for i, (_, row) in enumerate(summary_df.iterrows()):
        sym = row["instrument"]
        sess = row["session"]
        rm = int(row["range_minutes"])
        rr = float(row["rr"])
        direction = row["direction"]
        pool_key = (sym, sess, rm, row["entry_mode"],
                    int(row["closure_tf"]), direction)

        r_pool = (r_ticks_map or {}).get(pool_key)
        sds = session_days_map.get((sym, sess), [])[:n_null_samples]

        if r_pool is not None and len(r_pool):
            design = "matched"
            pools = matched_cache.get(pool_key)
            if pools is None:
                pools = build_matched_null_pool(
                    sds, rm, direction, r_pool, rr_levels)
                matched_cache[pool_key] = pools
            pool = pools.get(rr, NullPool(per_day=[], n_trades=0))
            n_matched += 1
        else:
            # No observed stops for this family -- fall back, and say so.
            design = "swing-fallback"
            skey = (sym, sess, rm, direction, rr)
            pool = swing_cache.get(skey)
            if pool is None:
                pool = build_null_pool(sds, rm, direction, rr)
                swing_cache[skey] = pool
            n_swing += 1

        n_obs = int(row.get("trade_count", 0) or 0)
        obs = float(row.get("expectancy_gross_r", 0.0) or 0.0)
        means = _bootstrap_null_means(
            pool, n_obs=n_obs, n_boot=n_boot,
            block_days=BOOTSTRAP_BLOCK_SIZE_DAYS,
        )

        d = row.to_dict()
        if len(means):
            d["null_exp_mean"] = round(float(np.mean(means)), 4)
            d["null_exp_p95"] = round(float(np.percentile(means, 95)), 4)
        else:
            d["null_exp_mean"] = np.nan
            d["null_exp_p95"] = np.nan
        d["null_pool_trades"] = pool.n_trades
        d["null_design"] = design
        d["null_p_value"] = round(null_p_value(obs, means), 4)
        out_rows.append(d)

        if (i + 1) % 1000 == 0:
            log.info("null calibration %d/%d rows "
                     "(%d matched pools, %d swing pools)",
                     i + 1, n_rows, len(matched_cache), len(swing_cache))

    log.info("null calibration complete: %d rows | matched %d, swing-fallback %d "
             "| %d matched pools, %d swing pools",
             n_rows, n_matched, n_swing, len(matched_cache), len(swing_cache))
    return pd.DataFrame(out_rows)
