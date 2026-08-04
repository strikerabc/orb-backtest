"""
validate_null_test.py — is the REWRITTEN null statistic actually calibrated?

A p-value is only meaningful if, when the observed sample truly comes from the
null distribution, p is approximately uniform on [0,1]. Concretely:

    mean(p) ~ 0.50
    fraction(p < 0.05) ~ 0.05
    fraction(p < 0.10) ~ 0.10
    corr(p, win_rate)  ~ 0.00   <- must NOT track hit rate
    no dependence on rr

Synthetic construction
----------------------
For each rr, a null trade wins with probability 1/(1+rr) paying +rr R, else
loses -1 R. Expected value is exactly zero:

    E = rr/(1+rr) - 1*(1/(1+rr))*... -> rr*(1/(1+rr)) - 1*(rr/(1+rr)) = 0

Observed samples are drawn from the SAME distribution, so the true answer is
"no edge" and a calibrated test must produce uniform p.

This isolates the statistic from the trading engine entirely -- no Databento,
no parquet, no SessionDay objects.

Baseline comparison
-------------------
The OLD statistic, mean(individual_null_trades >= observed_mean), is computed
alongside to show the contrast it was rejected for.

Usage:
    python validate_null_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.config import BOOTSTRAP_BLOCK_SIZE_DAYS
from src.null_calibrator import NullPool, _bootstrap_null_means, null_p_value

RR_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
N_DAYS = 400
DRAWS_PER_DAY = 3
N_TRIALS = 300          # observed samples per rr
N_OBS = 200             # trades per observed sample
N_BOOT = 400            # bootstrap resamples (kept modest for runtime)


def hr(t: str) -> None:
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def draw_outcomes(rr: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Zero-EV outcomes: win prob 1/(1+rr) paying +rr, else -1."""
    p_win = 1.0 / (1.0 + rr)
    wins = rng.random(n) < p_win
    return np.where(wins, rr, -1.0).astype(float)


def build_synthetic_pool(rr: float, rng: np.random.Generator) -> NullPool:
    per_day = [draw_outcomes(rr, DRAWS_PER_DAY, rng) for _ in range(N_DAYS)]
    return NullPool(per_day=per_day, n_trades=N_DAYS * DRAWS_PER_DAY)


def old_statistic(observed_mean: float, pool_flat: np.ndarray) -> float:
    """The rejected formula: observed MEAN vs pool of INDIVIDUAL trades."""
    if len(pool_flat) == 0:
        return float("nan")
    return float(np.mean(pool_flat >= observed_mean))


def main() -> None:
    rng = np.random.default_rng(12345)

    hr("CALIBRATION UNDER THE NULL  (observed drawn from the null itself)")
    print(f"per rr: {N_TRIALS} observed samples of {N_OBS} trades; "
          f"pool = {N_DAYS} days x {DRAWS_PER_DAY} draws; {N_BOOT} resamples")
    print(f"block size = {BOOTSTRAP_BLOCK_SIZE_DAYS} days\n")
    print(f"  {'rr':<6} {'mean p':>8} {'p<0.05':>8} {'p<0.10':>8} "
          f"{'p<0.50':>8} | {'OLD mean p':>11} {'OLD p<0.05':>11}")
    print("  " + "-" * 76)

    all_new: list[float] = []
    all_old: list[float] = []
    all_wr: list[float] = []
    all_rr: list[float] = []

    for rr in RR_LEVELS:
        new_ps, old_ps, wrs = [], [], []
        for _ in range(N_TRIALS):
            # Fresh pool per trial -> UNCONDITIONAL calibration.
            # Reusing one fixed pool across trials measures pool-CONDITIONAL
            # calibration instead: that pool's own sampling error (sd ~0.014 at
            # rr=0.25, vs observed-mean sd ~0.035) shifts every trial's p in
            # the same direction, so mean p drifts far from 0.5 even though the
            # statistic is sound. Section 2 below shows that effect directly.
            pool = build_synthetic_pool(rr, rng)
            flat = pool.flat()

            obs = draw_outcomes(rr, N_OBS, rng)
            obs_mean = float(obs.mean())
            wr = float((obs > 0).mean())

            means = _bootstrap_null_means(
                pool, n_obs=N_OBS, n_boot=N_BOOT,
                block_days=BOOTSTRAP_BLOCK_SIZE_DAYS,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            new_ps.append(null_p_value(obs_mean, means))
            old_ps.append(old_statistic(obs_mean, flat))
            wrs.append(wr)

        new_arr = np.array(new_ps)
        old_arr = np.array(old_ps)
        print(f"  {rr:<6} {new_arr.mean():>8.4f} "
              f"{(new_arr < 0.05).mean():>8.4f} "
              f"{(new_arr < 0.10).mean():>8.4f} "
              f"{(new_arr < 0.50).mean():>8.4f} | "
              f"{old_arr.mean():>11.4f} {(old_arr < 0.05).mean():>11.4f}")

        all_new.extend(new_ps)
        all_old.extend(old_ps)
        all_wr.extend(wrs)
        all_rr.extend([rr] * N_TRIALS)

    new_arr = np.array(all_new)
    old_arr = np.array(all_old)
    wr_arr = np.array(all_wr)
    rr_arr = np.array(all_rr)

    hr("AGGREGATE — target: mean 0.50, p<0.05 = 5%, corr ~ 0")
    print(f"  NEW  mean p          : {new_arr.mean():.4f}   (target 0.5000)")
    print(f"  NEW  frac p < 0.05   : {(new_arr < 0.05).mean():.4f}   (target 0.0500)")
    print(f"  NEW  frac p < 0.10   : {(new_arr < 0.10).mean():.4f}   (target 0.1000)")
    print(f"  NEW  corr(p, win_rate): {np.corrcoef(new_arr, wr_arr)[0,1]:+.4f}   (target ~0)")
    print(f"  NEW  corr(p, rr)      : {np.corrcoef(new_arr, rr_arr)[0,1]:+.4f}   (target ~0)")
    print()
    print(f"  OLD  mean p          : {old_arr.mean():.4f}")
    print(f"  OLD  frac p < 0.05   : {(old_arr < 0.05).mean():.4f}")
    print(f"  OLD  corr(p, win_rate): {np.corrcoef(old_arr, wr_arr)[0,1]:+.4f}")
    print(f"  OLD  corr(p, rr)      : {np.corrcoef(old_arr, rr_arr)[0,1]:+.4f}")

    hr("2. POOL-CONDITIONAL SPREAD — why a fixed pool skews mean p")
    print("Same statistic, but each row reuses ONE fixed pool for all trials")
    print("(what the first version of this script measured). Spread across")
    print("pools shows the pool's own sampling error moving mean p wholesale.")
    print("This is expected behaviour, not a defect: in production each")
    print("variant IS compared against one pool, so its p carries this term.\n")
    print(f"  {'rr':<6} {'pool mean':>10} {'mean p':>8} {'p<0.05':>8}")
    print("  " + "-" * 36)
    for rr in (0.25, 1.0, 2.0):
        for _ in range(3):
            pool = build_synthetic_pool(rr, rng)
            pm = float(pool.flat().mean())
            ps = []
            for _ in range(80):
                obs = draw_outcomes(rr, N_OBS, rng)
                means = _bootstrap_null_means(
                    pool, n_obs=N_OBS, n_boot=N_BOOT,
                    block_days=BOOTSTRAP_BLOCK_SIZE_DAYS,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                ps.append(null_p_value(float(obs.mean()), means))
            a = np.array(ps)
            print(f"  {rr:<6} {pm:>+10.4f} {a.mean():>8.4f} "
                  f"{(a < 0.05).mean():>8.4f}")

    hr("3. POWER CHECK — a REAL edge must be detected")
    print("Observed shifted by +0.15 R per trade (genuine edge); a working")
    print("test should return small p most of the time.\n")
    print(f"  {'rr':<6} {'edge':>6} {'mean p':>8} {'p<0.05':>8} {'p<0.10':>8}")
    print("  " + "-" * 42)
    for rr in (0.5, 1.0, 2.0):
        for edge in (0.15, 0.30):
            ps = []
            for _ in range(120):
                # fresh pool per trial, matching section 1 -- otherwise power
                # inherits the pool-conditional variance shown in section 2
                pool = build_synthetic_pool(rr, rng)
                obs = draw_outcomes(rr, N_OBS, rng) + edge
                means = _bootstrap_null_means(
                    pool, n_obs=N_OBS, n_boot=N_BOOT,
                    block_days=BOOTSTRAP_BLOCK_SIZE_DAYS,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                ps.append(null_p_value(float(obs.mean()), means))
            a = np.array(ps)
            print(f"  {rr:<6} {edge:>6.2f} {a.mean():>8.4f} "
                  f"{(a < 0.05).mean():>8.4f} {(a < 0.10).mean():>8.4f}")

    hr("VERDICT")
    ok_mean = abs(new_arr.mean() - 0.5) < 0.08
    ok_tail = (new_arr < 0.05).mean() < 0.15
    ok_corr = abs(np.corrcoef(new_arr, wr_arr)[0, 1]) < 0.25
    print(f"  mean p near 0.5           : {'PASS' if ok_mean else 'FAIL'}")
    print(f"  tail not wildly inflated  : {'PASS' if ok_tail else 'FAIL'}")
    print(f"  decoupled from win_rate   : {'PASS' if ok_corr else 'FAIL'}")
    print()
    print("  Compare: the OLD statistic could never emit p < 0.05 at all and")
    print("  correlated +0.99 with win_rate on real data.")
    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
