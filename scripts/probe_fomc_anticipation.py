"""
probe_fomc_anticipation.py -- the feasibility gate from EVENT_REGIME_REVIEW.md section E.

WHAT THIS IS
    The cheapest decisive slice of EVENT_REGIME_PLAN.md: FOMC decision dates only,
    POST_EXIT_SAME_DAY (anticipation) channel only, day-level, in-sample.
    Runs off the existing trade_log.parquet. No new SessionDay fields, no filter
    changes, no null-calibrator surgery, no pipeline re-run.

WHAT THIS IS NOT
    It is not the test of the hypothesis and its p-value is not a finding.
    ~40 usable event days per instrument-session, one channel, in-sample, no
    multiplicity correction across the channel x tier grid. Plan section 6.2's
    Level-1 pooled test is what decides the hypothesis. This gate exists only to
    decide whether the remaining ~250 calendar rows are worth a day of sourcing.
    Recorded as such in prereg/event_prereg.json BEFORE this ran.

WHY THIS CHANNEL
    Plan section 1.4: the FOMC statement (14:00 ET) and presser (14:30 ET) fall
    entirely OUTSIDE the 09:30-12:00 ET window. The operative mechanism on NY and
    LDN is anticipation, not shock -- thin participation and low-conviction
    drift-and-fade while everyone waits. Every decision in the sample carries a
    presser, so this is the purest DIFFUSE anticipation case available.

TWO CORRECTIONS TO THE PLAN, APPLIED HERE (see review sections A1, A2)
    1. Unit of analysis is (instrument, session, date) -> mean net_r pooled across
       families. Plan section 6.2 Level 1 says to shuffle day labels "within each
       family independently"; that destroys the cross-family correlation created by
       ~1190 families sharing ~24 instrument-sessions and the same price paths,
       giving a null that is far too tight. Collapsing to day means and permuting
       ONE label vector over the union date axis per draw enforces the correct
       joint structure through the data layout.
    2. MDE is taken from the permutation draws (2.486 * sd(delta_perm)), not from
       section 7's closed form, which plugs in correlated per-TRADE counts and
       overstates power by roughly 3-5x.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MIN_TRADES_FOR_RANKING  # noqa: E402
from src.filters import trade_eligibility  # noqa: E402
from src.null_calibrator import POOL_KEYS  # noqa: E402

TRADE_LOG = ROOT / "outputs" / "trade_log.parquet"
CALENDAR = ROOT / "data" / "events" / "fomc_decisions.csv"
OUT_JSON = ROOT / "prereg" / "probe_results.json"

HOLDOUT_START = "2026-02-01"
N_PERM = 20_000
SEED = 20260808
DRAW_BATCH = 2_000
EMERGENCY_DATE = "2020-03-03"   # unscheduled cut, ~10:00 ET -> IN-PATH, not anticipation

# Excluded from the pooled headline per plan sections 1.3 / 6.2, and reported
# separately as Level 0. Supplied by the user 2026-08-08: the paper-traded setup
# was EURUSD -> 6E, NY session, 5m range, CC entry, 5m closure, both directions.
# "L+S" is two family keys, not one. The observation source cannot confirm itself.
ORIGINATING_FAMILIES: list[tuple] = [
    ("6E", "NY", 5, "CC", 5, "long"),
    ("6E", "NY", 5, "CC", 5, "short"),
]

NEEDED = [
    "date", "instrument", "session", "range_minutes", "entry_mode", "closure_tf",
    "direction", "rr", "net_r", "day_of_week", "parkinson_vol_14d",
    "tap_in_bar_idx", "mfe_r", "mae_r", "exit_reason", "tp_unfillable",
    "tp_ticks", "r_ticks", "cost_r", "contracts",
    "contract_changed_in_session", "contract_changed_since_prev_session",
    "session_bar_completeness",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_eligible() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pooled, level0). Level 0 = originating families, never pooled."""
    log("loading trade log ...")
    df = pd.read_parquet(TRADE_LOG, columns=NEEDED)
    log(f"  {len(df):,} rows")
    df = trade_eligibility(df)
    df = df[df["eligible"]].copy()
    log(f"  {len(df):,} eligible")

    counts = (df.groupby(POOL_KEYS, observed=True).size()
                .rename("fam_n").reset_index())
    keep = counts[counts["fam_n"] >= MIN_TRADES_FOR_RANKING]
    df = df.merge(keep[POOL_KEYS], on=POOL_KEYS, how="inner")
    log(f"  {len(df):,} in {len(keep):,} rankable families "
        f"(>= {MIN_TRADES_FOR_RANKING} eligible trades)")

    mask = pd.Series(False, index=df.index)
    for fam in ORIGINATING_FAMILIES:
        m = pd.Series(True, index=df.index)
        for col, val in zip(POOL_KEYS, fam):
            m &= df[col] == val
        mask |= m
    level0 = df[mask].copy()
    df = df[~mask].copy()
    log(f"  {len(level0):,} rows in {len(ORIGINATING_FAMILIES)} originating "
        f"families (Level 0, reported separately, NEVER pooled)")

    df = df[df["date"] < HOLDOUT_START].copy()
    level0 = level0[level0["date"] < HOLDOUT_START].copy()
    log(f"  {len(df):,} pooled pre-holdout (date < {HOLDOUT_START})")
    return df, level0


def load_fomc_dates() -> list[str]:
    cal = pd.read_csv(CALENDAR)
    dates = sorted(cal["decision_date_local"].unique())
    return [d for d in dates if d < HOLDOUT_START]


def build_day_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (instrument, session, date). This IS the fix."""
    g = df.groupby(["instrument", "session", "date"], observed=True)
    panel = g.agg(
        mean_net_r=("net_r", "mean"),
        n_trades=("net_r", "size"),
        vol=("parkinson_vol_14d", "first"),
        weekday=("day_of_week", "first"),
    ).reset_index()
    return panel


def make_matrices(panel: pd.DataFrame) -> tuple:
    """Align every instrument-session onto a shared union date axis."""
    dates = np.array(sorted(panel["date"].unique()))
    date_pos = {d: i for i, d in enumerate(dates)}
    groups = sorted(panel.groupby(["instrument", "session"], observed=True).groups)

    n_g, n_d = len(groups), len(dates)
    vals = np.zeros((n_g, n_d), dtype=np.float64)
    present = np.zeros((n_g, n_d), dtype=np.float64)
    for gi, key in enumerate(groups):
        sub = panel[(panel["instrument"] == key[0]) & (panel["session"] == key[1])]
        cols = sub["date"].map(date_pos).to_numpy()
        vals[gi, cols] = sub["mean_net_r"].to_numpy()
        present[gi, cols] = 1.0

    # Date-level volatility rank for stratification. Averaging each group's
    # within-group percentile rank keeps ONE label axis, so the single
    # permutation-per-draw structure survives stratification.
    ranks = np.full((n_g, n_d), np.nan)
    for gi, key in enumerate(groups):
        sub = panel[(panel["instrument"] == key[0]) & (panel["session"] == key[1])]
        cols = sub["date"].map(date_pos).to_numpy()
        r = sub["vol"].rank(pct=True).to_numpy()
        ranks[gi, cols] = r
    with np.errstate(invalid="ignore"):
        date_vol = np.nanmean(ranks, axis=0)

    weekday = panel.drop_duplicates("date").set_index("date")["weekday"]
    date_weekday = np.array([weekday.get(d, "?") for d in dates])
    return dates, groups, vals, present, date_vol, date_weekday


def deltas_for(labels: np.ndarray, vals: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Pooled delta for a batch of label vectors.

    labels : (K, n_dates) float 0/1
    returns: (K,) day-count-weighted mean across instrument-sessions of
             (mean day net_r | contaminated) - (mean day net_r | clean)

    One label vector per draw applied to EVERY instrument-session, so the
    cross-instrument correlation that FOMC dates induce is preserved. This is the
    review-A1 correction, enforced structurally.
    """
    weighted = vals * present
    n_contam = labels @ present.T                       # (K, n_groups)
    sum_contam = labels @ weighted.T
    n_total = present.sum(axis=1)[None, :]
    sum_total = weighted.sum(axis=1)[None, :]
    n_clean = n_total - n_contam
    sum_clean = sum_total - sum_contam

    ok = (n_contam > 0) & (n_clean > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(ok, sum_contam / n_contam - sum_clean / n_clean, np.nan)
    w = np.where(ok, n_contam, 0.0)
    wsum = w.sum(axis=1)
    return np.where(wsum > 0,
                    np.nansum(np.where(ok, delta * w, 0.0), axis=1)
                    / np.where(wsum > 0, wsum, 1.0), np.nan)


def strata_for(scheme: str, date_vol: np.ndarray, date_weekday: np.ndarray) -> list[np.ndarray]:
    """Index groups within which labels may be exchanged."""
    n = len(date_weekday)
    if scheme == "none":
        return [np.arange(n)]
    if scheme == "weekday":
        return [np.flatnonzero(date_weekday == w) for w in np.unique(date_weekday)]
    if scheme == "weekday_vol":
        finite = date_vol[np.isfinite(date_vol)]
        if len(finite) < 3:
            return [np.arange(n)]
        cuts = np.quantile(finite, [1 / 3, 2 / 3])
        terc = np.digitize(np.where(np.isfinite(date_vol), date_vol, -1), cuts)
        terc = np.where(np.isfinite(date_vol), terc, -1)
        out = []
        for w in np.unique(date_weekday):
            for t in np.unique(terc):
                idx = np.flatnonzero((date_weekday == w) & (terc == t))
                if len(idx):
                    out.append(idx)
        return out
    raise ValueError(scheme)


def permute_batch(rng, base: np.ndarray, strata: list[np.ndarray], k: int) -> np.ndarray:
    """k permuted label vectors; contaminated count held fixed within each stratum."""
    out = np.zeros((k, len(base)), dtype=np.float64)
    for idx in strata:
        m = int(base[idx].sum())
        if m == 0 or m == len(idx):
            out[:, idx] = base[idx]
            continue
        for draw in range(k):
            out[draw, rng.permutation(idx)[:m]] = 1.0
    return out


def run_test(name: str, event_dates: set[str], dates: np.ndarray, groups, vals,
             present, date_vol, date_weekday, scheme: str, rng) -> dict:
    base = np.isin(dates, list(event_dates)).astype(np.float64)
    n_contam = int(base.sum())
    if n_contam == 0:
        return {"test": name, "stratify": scheme, "error": "no contaminated dates"}

    observed = float(deltas_for(base[None, :], vals, present)[0])
    strata = strata_for(scheme, date_vol, date_weekday)

    null = np.empty(N_PERM, dtype=np.float64)
    done = 0
    while done < N_PERM:
        k = min(DRAW_BATCH, N_PERM - done)
        null[done:done + k] = deltas_for(
            permute_batch(rng, base, strata, k), vals, present)
        done += k

    finite = null[np.isfinite(null)]
    # One-sided in the pre-registered direction (degradation), mid-p ties,
    # consistent with null_calibrator.null_p_value.
    le = int(np.sum(finite < observed))
    eq = int(np.sum(finite == observed))
    p = (1 + le + 0.5 * eq) / (1 + len(finite)) if len(finite) else float("nan")
    sd = float(np.std(finite)) if len(finite) else float("nan")

    contam_days = base @ present.T
    return {
        "test": name,
        "stratify": scheme,
        "n_event_dates_in_axis": n_contam,
        "n_strata": len(strata),
        "observed_delta_net_r": round(observed, 6),
        "perm_p_one_sided_degradation": round(p, 6),
        "null_mean": round(float(np.mean(finite)), 6) if len(finite) else None,
        "null_sd": round(sd, 6),
        "mde_80pct_power": round(2.486 * sd, 6),
        "contaminated_days_per_group_median": float(np.median(contam_days)),
        "contaminated_days_per_group_min": float(contam_days.min()),
        "n_draws_finite": int(len(finite)),
    }


def mechanism(df: pd.DataFrame, event_dates: set[str]) -> dict:
    """Trade-level mechanism metrics available without a bar re-walk.

    Descriptive only -- no permutation p-values attached, because the plan's
    section 6.3 mechanism claim is meant to be read against the vol-profile
    taxonomy validation, which needs 1m bars and is a Phase A/B prerequisite.
    """
    d = df.copy()
    d["contam"] = d["date"].isin(event_dates)
    d["reversal"] = d["tap_in_bar_idx"].notna()
    reason = d["exit_reason"].astype(str).str.lower()
    d["tp_hit"] = reason.str.contains("tp")
    with np.errstate(invalid="ignore", divide="ignore"):
        d["mfe_mae"] = d["mfe_r"] / d["mae_r"].abs().replace(0, np.nan)

    out: dict = {"exit_reasons_seen": sorted(reason.unique().tolist())[:12]}
    for label, sub in (("clean", d[~d["contam"]]), ("contaminated", d[d["contam"]])):
        out[label] = {
            "n_trades": int(len(sub)),
            "n_days": int(sub["date"].nunique()),
            "mean_net_r": round(float(sub["net_r"].mean()), 6),
            "reversal_rate": round(float(sub["reversal"].mean()), 6),
            "mfe_mae_ratio_median": round(float(sub["mfe_mae"].median()), 6),
            "tp_hit_rate": round(float(sub["tp_hit"].mean()), 6),
            "tp_hit_rate_by_rr": {
                str(rr): round(float(g["tp_hit"].mean()), 6)
                for rr, g in sub.groupby("rr", observed=True)},
        }
    out["delta"] = {
        k: round(out["contaminated"][k] - out["clean"][k], 6)
        for k in ("mean_net_r", "reversal_rate", "mfe_mae_ratio_median", "tp_hit_rate")}
    out["delta_tp_hit_rate_by_rr"] = {
        rr: round(v - out["clean"]["tp_hit_rate_by_rr"].get(rr, float("nan")), 6)
        for rr, v in out["contaminated"]["tp_hit_rate_by_rr"].items()}
    return out


def shift_dates(dates: list[str], days: int) -> set[str]:
    return {(pd.Timestamp(d) + timedelta(days=days)).strftime("%Y-%m-%d") for d in dates}


def main() -> None:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"

    df, level0 = load_eligible()
    fomc = load_fomc_dates()
    log(f"FOMC decision dates pre-holdout: {len(fomc)}")

    panel = build_day_panel(df)
    log(f"day panel: {len(panel):,} (instrument, session, date) rows")
    dates, groups, vals, present, date_vol, date_weekday = make_matrices(panel)
    log(f"union date axis: {len(dates)} dates x {len(groups)} instrument-sessions")

    fomc_in_axis = sorted(set(fomc) & set(dates.tolist()))
    log(f"FOMC dates present in axis: {len(fomc_in_axis)}")
    weekdays = pd.Series([date_weekday[list(dates).index(d)] for d in fomc_in_axis])
    log(f"  weekday mix: {weekdays.value_counts().to_dict()}")

    results: list[dict] = []
    for scheme in ("none", "weekday", "weekday_vol"):
        log(f"running primary [{scheme}] ...")
        results.append(run_test(
            "fomc_anticipation", set(fomc_in_axis), dates, groups, vals, present,
            date_vol, date_weekday, scheme, np.random.default_rng(SEED)))

    no_emerg = set(fomc_in_axis) - {EMERGENCY_DATE}
    log("running sensitivity [drop 2020-03-03] ...")
    results.append(run_test(
        "fomc_anticipation_ex_emergency", no_emerg, dates, groups, vals, present,
        date_vol, date_weekday, "weekday", np.random.default_rng(SEED + 1)))

    for shift in (7, -7):
        log(f"running placebo [{shift:+d}d] ...")
        placebo = shift_dates(fomc_in_axis, shift) & set(dates.tolist())
        results.append(run_test(
            f"placebo_shift_{shift:+d}d", placebo, dates, groups, vals, present,
            date_vol, date_weekday, "weekday", np.random.default_rng(SEED + 2)))

    # Level 0 -- the originating families, run on their own. Plan sections 1.3/6.2:
    # the observation source cannot be its own confirmation, so this is reported
    # apart and never pooled. If the effect lives ONLY here, it is noise.
    log("running Level 0 [originating families, NOT pooled] ...")
    level0_results: list[dict] = []
    level0_mech: dict = {}
    if len(level0):
        p0 = build_day_panel(level0)
        d0, g0, v0, pr0, dv0, dw0 = make_matrices(p0)
        axis0 = sorted(set(fomc_in_axis) & set(d0.tolist()))
        for scheme in ("none", "weekday"):
            level0_results.append(run_test(
                "LEVEL0_originating_fomc_anticipation", set(axis0), d0, g0, v0, pr0,
                dv0, dw0, scheme, np.random.default_rng(SEED + 3)))
        level0_mech = mechanism(level0, set(axis0))
        log(f"  Level 0: {len(level0):,} trades, {len(g0)} group(s), "
            f"{len(axis0)} FOMC dates in axis")

    log("computing mechanism metrics ...")
    mech = mechanism(df, set(fomc_in_axis))

    payload = {
        "_what_this_is": "Feasibility GATE, not a test of the hypothesis. See "
                         "prereg/event_prereg.json -> gate_run_on_this_branch and "
                         "EVENT_REGIME_REVIEW.md section E. ~40 event days per "
                         "instrument-session, one channel, IN-SAMPLE, no multiplicity "
                         "correction. Its p-value is not a finding.",
        "git_sha": sha,
        "prereg": "prereg/event_prereg.json",
        "n_perm": N_PERM,
        "seed": SEED,
        "holdout_start": HOLDOUT_START,
        "originating_families_excluded": ORIGINATING_FAMILIES or "UNRESOLVED",
        "universe": {
            "eligible_rankable_pre_holdout_trades": int(len(df)),
            "instrument_sessions": len(groups),
            "union_dates": int(len(dates)),
            "fomc_dates_pre_holdout": len(fomc),
            "fomc_dates_in_axis": len(fomc_in_axis),
            "fomc_weekday_mix": weekdays.value_counts().to_dict(),
        },
        "tests": results,
        "mechanism": mech,
        "level0_originating_families": {
            "_note": "The two paper-traded families (6E/NY/5/CC/5 long and short). "
                     "Reported apart and NEVER pooled into the headline, per plan "
                     "sections 1.3 and 6.2 -- the observation source cannot confirm "
                     "itself. If the effect appears only here, it is noise. Single "
                     "instrument-session, so the day axis is one group and this is "
                     "the most underpowered cut in the file.",
            "families": [list(f) for f in ORIGINATING_FAMILIES],
            "n_trades": int(len(level0)),
            "tests": level0_results,
            "mechanism": level0_mech,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"\nwrote {OUT_JSON}\n")

    log(f"{'test':38s} {'strat':12s} {'delta':>10s} {'p':>8s} {'MDE':>9s}")
    for r in results:
        if "error" in r:
            continue
        log(f"{r['test']:38s} {r['stratify']:12s} "
            f"{r['observed_delta_net_r']:>10.5f} "
            f"{r['perm_p_one_sided_degradation']:>8.4f} "
            f"{r['mde_80pct_power']:>9.5f}")
    log(f"\nmechanism delta (contaminated - clean): {json.dumps(mech['delta'])}")


if __name__ == "__main__":
    main()
