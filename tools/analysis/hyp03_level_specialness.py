"""
hyp03_level_specialness.py — SHIFT and WIDTH placebo sweeps.

Tests whether the opening-range level carries information relative to
mechanically identical placebo bands.

Primary statistic: trigger rate (fraction of session-days where price
crosses the boundary after range end). If SHIFT and WIDTH match the real
level → level is not special → close HYP-05 unrun.

Secondary: E[gross_r] per instrument-session, paired against placebo.

Inference: paired permutation test at the session-day level (real/placebo
label exchangeable within each day). Joint maxT correction across
instrument-sessions for multiplicity.

Pre-registration: prereg/hyp03_level_specialness_prereg.json (committed
before this tool ran).

Usage:
    python tools/analysis/hyp03_level_specialness.py
    python tools/analysis/hyp03_level_specialness.py --full-period
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.config import (
    INSTRUMENTS, OUTPUTS_DIR, RANGE_MINUTES, SESSIONS, RR_LEVELS,
)
from src.contracts import comm_ticks
from src.data_layer import _compute_enrichment, ensure_daily, ensure_data
from src.range_builder import build_session_days, SHIFT_OFFSET_MINUTES
from src.entry_detector import detect_entries
from src.trade_sim import simulate_trade
from src.filters import trade_eligibility

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("orb.hyp03")
_OUT = _ROOT / OUTPUTS_DIR

HOLDOUT_START = "2026-02-01"
N_PERM   = 20_000
PERM_SEED = 0

ALL_INST_SESS = [
    (sym, sess)
    for sym, cfg in INSTRUMENTS.items()
    for sess in cfg["sessions"]
]


def hr(t: str) -> None:
    print("\n" + "=" * 110)
    print(t)
    print("=" * 110)


# ── lightweight trigger check ────────────────────────────────────────────────

def _check_triggered(sd, rm: int, rh: float, rl: float,
                     detection_start_wall: int) -> bool:
    """
    True if any bar after detection_start_wall crosses rh+tick or rl-tick.

    Used for the fast trigger-rate check without full simulation. The
    detection_start_wall is either open_min+rm (real/width) or
    open_min+60+rm (shift).
    """
    idx = int(np.searchsorted(sd.bar_wall_mins, detection_start_wall))
    if idx >= len(sd.bars_h):
        return False
    tick = sd.tick_size
    h = sd.bars_h[idx:]
    l = sd.bars_l[idx:]
    return bool((h >= rh + tick).any() or (l <= rl - tick).any())


def compute_trigger_rates(
    sym: str, sess: str, df: pd.DataFrame
) -> dict:
    """
    Build real, SHIFT, and WIDTH SessionDays; compute per-day trigger flags.

    Returns a dict with keys 'real', 'shift', 'width' each mapping to a
    dict {date: bool} of whether that session-day was triggered.
    """
    sess_cfg = SESSIONS[sess]
    oh, om   = sess_cfg["open"]
    open_min = oh * 60 + om

    real_days  = build_session_days(df, sym, sess, placebo_mode="real")
    shift_days = build_session_days(df, sym, sess, placebo_mode="shift")
    width_days = build_session_days(df, sym, sess, placebo_mode="width")

    # Index by date for paired comparison
    def _index(days):
        result = {}
        for sd in days:
            for rm in RANGE_MINUTES:
                rh = sd.range_highs.get(rm)
                rl = sd.range_lows.get(rm)
                if rh is None or rl is None:
                    continue
                det_start = (sd.range_end_wall_mins[rm]
                             if sd.range_end_wall_mins and rm in sd.range_end_wall_mins
                             else open_min + rm)
                if _check_triggered(sd, rm, rh, rl, det_start):
                    result[sd.local_date] = True
                    break  # at least one rm fired → day is triggered
            else:
                if sd.local_date not in result:
                    result[sd.local_date] = False
        return result

    return {
        "real":  _index(real_days),
        "shift": _index(shift_days),
        "width": _index(width_days),
    }


# ── permutation test ─────────────────────────────────────────────────────────

def _trigger_rate_delta(real: list[bool], placebo: list[bool]) -> float:
    """Mean(real) - Mean(placebo) on paired boolean arrays."""
    r = np.asarray(real, dtype=float)
    p = np.asarray(placebo, dtype=float)
    return float(r.mean() - p.mean())


def _permutation_p(real: list[bool], placebo: list[bool],
                   rng: np.random.Generator) -> tuple[float, float]:
    """
    One-sided (real > placebo) paired permutation p-value.

    Returns (observed_delta, p_value).
    """
    obs   = _trigger_rate_delta(real, placebo)
    r_arr = np.asarray(real,    dtype=float)
    p_arr = np.asarray(placebo, dtype=float)
    n     = len(r_arr)
    count = 0
    for _ in range(N_PERM):
        # Swap labels independently per day with probability 0.5
        swap  = rng.integers(0, 2, size=n).astype(float)  # 0/1 per day
        perm_r = np.where(swap, p_arr, r_arr)
        perm_p = np.where(swap, r_arr, p_arr)
        if perm_r.mean() - perm_p.mean() >= obs:
            count += 1
    return obs, count / N_PERM


def maxt_permutation(
    deltas: dict[str, float],
    real_by_key: dict[str, list[bool]],
    plac_by_key: dict[str, list[bool]],
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Joint maxT permutation: one p-value per instrument-session, corrected
    for the joint null that ALL deltas are zero.

    deltas: {key: observed_delta}. real_by_key/plac_by_key: per-key paired lists.

    Returns {key: holm_p_adj} (Holm applied to the permutation p-values).
    """
    keys   = list(deltas.keys())
    obs_arr= np.array([deltas[k] for k in keys])  # (n_keys,)

    # Null distribution of max statistic across keys
    null_max = np.zeros(N_PERM)
    arr_real = [np.array(real_by_key[k], dtype=float) for k in keys]
    arr_plac = [np.array(plac_by_key[k], dtype=float) for k in keys]
    n_per_key = [len(arr_real[k]) for k in range(len(keys))]

    for b in range(N_PERM):
        deltas_b = []
        for i, (r, p, n) in enumerate(zip(arr_real, arr_plac, n_per_key)):
            swap  = rng.integers(0, 2, size=n).astype(float)
            perm_r = np.where(swap, p, r)
            perm_p = np.where(swap, r, p)
            deltas_b.append(float(perm_r.mean() - perm_p.mean()))
        null_max[b] = max(deltas_b)

    # Per-key p-value from joint null
    raw_p = {k: float((null_max >= obs_arr[i]).mean())
             for i, k in enumerate(keys)}

    # Holm correction on the permutation p-values
    sorted_keys = sorted(raw_p, key=lambda k: raw_p[k])
    holm_p: dict[str, float] = {}
    m = len(sorted_keys)
    for rank, k in enumerate(sorted_keys):
        holm_p[k] = min(1.0, raw_p[k] * (m - rank))

    return holm_p


# ── E[gross_r] via full simulation ──────────────────────────────────────────

def simulate_gross_r(sym: str, sess: str, df: pd.DataFrame,
                     placebo_mode: str) -> dict[str, float]:
    """
    Run the full engine for (sym, sess) on df with placebo_mode.

    Returns {(rm, direction): mean_gross_r} for triggered instrument-sessions.
    Aggregates across all range sizes and direction to a single
    instrument-session mean for the summary table.
    """
    sdays = build_session_days(df, sym, sess, placebo_mode=placebo_mode)
    rows  = []
    for sd in sdays:
        for es in detect_entries(sd):
            for tr in simulate_trade(es, sd, RR_LEVELS):
                rows.append({
                    "gross_r": tr.gross_r,
                    "tp_unfillable": tr.tp_unfillable,
                    "r_ticks": tr.r_ticks,
                    "cost_r": tr.cost_r,
                    "rr": tr.rr,
                })
    if not rows:
        return {"n_trades": 0, "mean_gross_r": float("nan")}
    df_t = trade_eligibility(pd.DataFrame(rows))
    df_t = df_t[df_t["eligible"]]
    if df_t.empty:
        return {"n_trades": 0, "mean_gross_r": float("nan")}
    return {
        "n_trades": len(df_t),
        "mean_gross_r": float(df_t["gross_r"].mean()),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main(full_period: bool = False) -> dict:
    t0 = time.perf_counter()
    period_start = "2019-01-01" if full_period else HOLDOUT_START
    period_label = "full sample" if full_period else f"holdout ({HOLDOUT_START}+)"

    hr(f"HYP-03 LEVEL SPECIALNESS — SHIFT ({SHIFT_OFFSET_MINUTES}min) + WIDTH PLACEBOS")
    print(f"  Period: {period_label}")
    print(f"  Permutations: {N_PERM:,}, seed: {PERM_SEED}")
    print(f"  Pre-registration: prereg/hyp03_level_specialness_prereg.json")

    rng = np.random.default_rng(PERM_SEED)

    # ── Phase 1: trigger rates (fast) ─────────────────────────────────────

    hr("PHASE 1: TRIGGER RATES")
    print("  Does the real range fire more often than SHIFT/WIDTH placebo?")
    print("  (If neither placebo is significantly lower: level is not special.)\n")

    tr_real_all:  dict[str, list[bool]] = {}
    tr_shift_all: dict[str, list[bool]] = {}
    tr_width_all: dict[str, list[bool]] = {}

    table_rows = []
    for sym, sess in ALL_INST_SESS:
        key = f"{sym}/{sess}"
        try:
            df = _compute_enrichment(ensure_data(sym), ensure_daily(sym),
                                     tick_size=INSTRUMENTS[sym]["tick_size"])
            df = df[df["timestamp"] >= pd.Timestamp(period_start, tz="UTC")]
            if df.empty:
                continue
        except Exception as exc:
            print(f"  {key}: data load failed — {exc}")
            continue

        rates = compute_trigger_rates(sym, sess, df)
        # Align to common dates
        all_dates = sorted(
            set(rates["real"]) | set(rates["shift"]) | set(rates["width"]))
        if not all_dates:
            continue

        r = [rates["real"].get(d, False) for d in all_dates]
        s = [rates["shift"].get(d, False) for d in all_dates]
        w = [rates["width"].get(d, False) for d in all_dates]

        tr_real_all[key]  = r
        tr_shift_all[key] = s
        tr_width_all[key] = w

        n = len(r)
        table_rows.append({
            "key": key, "n_days": n,
            "real_pct":  100.0 * np.mean(r),
            "shift_pct": 100.0 * np.mean(s),
            "width_pct": 100.0 * np.mean(w),
            "delta_shift": 100.0 * (np.mean(r) - np.mean(s)),
            "delta_width": 100.0 * (np.mean(r) - np.mean(w)),
        })
        print(f"  {key:12s}: real={100*np.mean(r):.1f}%  "
              f"shift={100*np.mean(s):.1f}%  width={100*np.mean(w):.1f}%  "
              f"Δshift={100*(np.mean(r)-np.mean(s)):+.1f}pp  "
              f"Δwidth={100*(np.mean(r)-np.mean(w)):+.1f}pp  "
              f"(n={n})")

    if not tr_real_all:
        print("  ERROR: no data loaded")
        return {}

    print()

    # ── joint permutation on trigger rate deltas ───────────────────────────
    hr("PHASE 1b: PERMUTATION TEST — real vs SHIFT")
    deltas_shift = {k: _trigger_rate_delta(tr_real_all[k], tr_shift_all[k])
                    for k in tr_real_all}
    holm_shift = maxt_permutation(
        deltas_shift, tr_real_all, tr_shift_all, rng)

    hr("PHASE 1c: PERMUTATION TEST — real vs WIDTH")
    deltas_width = {k: _trigger_rate_delta(tr_real_all[k], tr_width_all[k])
                    for k in tr_real_all}
    holm_width = maxt_permutation(
        deltas_width, tr_real_all, tr_width_all, rng)

    print("\n  Trigger-rate test summary (Holm-adjusted p from joint permutation):")
    print(f"  {'key':12s}  {'Δshift':>8s}  {'p_shift':>8s}  {'Δwidth':>8s}  {'p_width':>8s}")
    sig_keys_shift, sig_keys_width = [], []
    for row in sorted(table_rows, key=lambda x: -x["delta_shift"]):
        k = row["key"]
        ps = holm_shift.get(k, 1.0)
        pw = holm_width.get(k, 1.0)
        if ps < 0.05:
            sig_keys_shift.append(k)
        if pw < 0.05:
            sig_keys_width.append(k)
        print(f"  {k:12s}  {row['delta_shift']:+7.2f}pp  {ps:.4f}     "
              f"{row['delta_width']:+7.2f}pp  {pw:.4f}"
              + ("  *SHIFT" if ps < 0.05 else "")
              + ("  *WIDTH" if pw < 0.05 else ""))

    gate_passes = bool(sig_keys_shift or sig_keys_width)
    if not gate_passes:
        hr("HYP-05 GATE: FAILS — LEVEL IS NOT SPECIAL")
        print(
            "  Neither SHIFT nor WIDTH trigger rate differs significantly from real.")
        print("  The opening-range boundary is not broken more often than an")
        print("  equivalently-sized or equivalently-timed placebo band.")
        print()
        print("  HYP-05 closes UNRUN. Characterising heterogeneity in a signal")
        print("  that does not exist is not informative.")
        print()
        print("  The correct conclusion for the project:")
        print("   - The ORB level adds no trigger-rate premium over a placebo band.")
        print("   - HYP-01/02/04 studied refinements of a signal that this test")
        print("     cannot distinguish from noise at the level of 'does it fire?'")
        print("   - Document this and close the cross-sectional programme.")
    else:
        hr("HYP-05 GATE: PASSES — level IS distinguishable from placebo")
        print(f"  Keys significantly different from SHIFT: {sig_keys_shift}")
        print(f"  Keys significantly different from WIDTH: {sig_keys_width}")
        print(f"  HYP-05 proceeds.\n")

    # ── Phase 2: E[gross_r] (only if gate passes) ─────────────────────────
    gross_table = []
    if gate_passes:
        hr("PHASE 2: E[gross_r] PER INSTRUMENT-SESSION — real vs SHIFT vs WIDTH")
        print("  Running full simulation for placebo ranges…")
        for sym, sess in ALL_INST_SESS:
            key = f"{sym}/{sess}"
            if key not in tr_real_all:
                continue
            try:
                df = _compute_enrichment(ensure_data(sym), ensure_daily(sym),
                                         tick_size=INSTRUMENTS[sym]["tick_size"])
                df = df[df["timestamp"] >= pd.Timestamp(period_start, tz="UTC")]
                if df.empty:
                    continue
            except Exception:
                continue

            gr  = simulate_gross_r(sym, sess, df, "real")
            gs  = simulate_gross_r(sym, sess, df, "shift")
            gw  = simulate_gross_r(sym, sess, df, "width")
            cost_r = comm_ticks(sym) / INSTRUMENTS[sym].get("tick_value_usd", 1)
            gross_table.append({
                "key": key, "sym": sym, "sess": sess,
                "real_n": gr["n_trades"],   "real_gross_r": gr["mean_gross_r"],
                "shift_n": gs["n_trades"],  "shift_gross_r": gs["mean_gross_r"],
                "width_n": gw["n_trades"],  "width_gross_r": gw["mean_gross_r"],
            })
            print(f"  {key:12s}: real={gr['mean_gross_r']:+.4f} (n={gr['n_trades']:,})  "
                  f"shift={gs['mean_gross_r']:+.4f}  width={gw['mean_gross_r']:+.4f}")

    elapsed = time.perf_counter() - t0

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "hypothesis": "HYP-03",
        "prereg": "prereg/hyp03_level_specialness_prereg.json",
        "period": period_label,
        "period_start": period_start,
        "shift_offset_minutes": SHIFT_OFFSET_MINUTES,
        "n_permutations": N_PERM,
        "perm_seed": PERM_SEED,
        "trigger_rate_table": table_rows,
        "sig_shift": sig_keys_shift,
        "sig_width": sig_keys_width,
        "gate_passes_for_hyp05": gate_passes,
        "gross_r_table": gross_table,
        "elapsed_s": round(elapsed, 1),
        "verdict": (
            "LEVEL IS SPECIAL — gate passes for HYP-05"
            if gate_passes else
            "LEVEL NOT SPECIAL — neither SHIFT nor WIDTH trigger rate differs from real; HYP-05 closes unrun"
        ),
    }
    out_path = _OUT / "hyp03_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out_path}")
    print(f"  Elapsed: {elapsed:.0f}s")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-period", action="store_true",
                        help="Run on full sample instead of holdout only")
    args = parser.parse_args()
    main(full_period=args.full_period)
