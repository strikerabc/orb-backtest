"""
hyp05_shrinkage.py — HYP-05 §6.1: empirical-Bayes shrinkage vs raw selection.

Does shrinking in-sample family estimates toward the population mean (using
tau-squared estimated from ALL families) predict holdout net R better than
raw in-sample selection?

This is parameter-free: tau-squared is estimated from the data (not scanned),
so the holdout is clean for this test.

Pre-registration: prereg/hyp05_prereg.json §h5e_shrinkage.
Method:
  1. Load all 976 in-sample family estimates from summary.parquet.
  2. Estimate tau-hat-sq by method of moments:
       Var(y_i) = tau-sq + mean(sigma_i-sq)
       tau-hat-sq = max(0, Var(y_i) - mean(sigma_i-sq))
     where sigma_i-sq ≈ 1 / n_trades (per-trade R SD ≈ 1.0 for ORB).
  3. Compute shrinkage factor B_i = tau-sq / (tau-sq + sigma_i-sq).
  4. Shrunk estimate: mu_shrunk_i = B_i * y_i + (1 - B_i) * y_bar.
  5. Test on survivors (94 families with holdout data):
     - OOS MSE: shrunk vs raw (lower is better)
     - Rank correlation with holdout: Spearman(shrunk, ho) vs Spearman(raw, ho)
     - Bias: shrunk mean vs raw mean vs holdout mean

A methodological result, not a strategy one. Transfers to any future sweep.

Usage:
    python tools/analysis/hyp05_shrinkage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.bootstrap import estimate, mean_stat
from src.config import MIN_TRADES_FOR_RANKING, OUTPUTS_DIR

_OUT = _ROOT / OUTPUTS_DIR
FAMILY = ["instrument", "session", "range_minutes", "entry_mode",
          "closure_tf", "direction"]
N_BOOT = 20_000
BOOT_SEED = 0


def hr(t: str) -> None:
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def main() -> None:
    # ── 1. in-sample family estimates ─────────────────────────────────────
    hr("1. IN-SAMPLE FAMILY ESTIMATES (all rankable families)")
    sm = pd.read_parquet(_OUT / "summary.parquet")
    rank = sm[sm["trade_count"] >= MIN_TRADES_FOR_RANKING].copy()
    # One row per family: take best-net rr
    best_idx = rank.groupby(FAMILY, observed=True)["expectancy_net_r"].idxmax()
    fam_is   = rank.loc[best_idx, FAMILY + [
        "expectancy_net_r", "expectancy_gross_r", "trade_count"]].copy()

    n_fam   = len(fam_is)
    y       = fam_is["expectancy_net_r"].to_numpy(float)
    n_i     = fam_is["trade_count"].to_numpy(float)
    y_bar   = float(y.mean())

    # sigma_i_sq: sampling variance per family. ORB per-trade R SD ≈ 1.0 R;
    # family mean variance ≈ 1 / n_trades.
    sigma_sq = 1.0 / n_i

    # tau-hat-sq: method of moments
    obs_var  = float(np.var(y, ddof=1))
    mean_sig = float(sigma_sq.mean())
    tau_sq   = max(0.0, obs_var - mean_sig)
    I_sq     = tau_sq / (tau_sq + mean_sig) if (tau_sq + mean_sig) > 0 else 0.0

    print(f"  rankable families          : {n_fam:,}")
    print(f"  mean in-sample net R       : {y_bar:+.4f} R")
    print(f"  observed Var(y_i)          : {obs_var:.6f}")
    print(f"  mean sampling noise E[σ²]  : {mean_sig:.6f}")
    print(f"  tau-hat-sq (MOM)           : {tau_sq:.6f}")
    print(f"  I-squared                  : {100*I_sq:.1f}%")

    if tau_sq == 0.0:
        print(f"\n  tau-hat-sq = 0: observed variance is entirely sampling noise.")
        print(f"  Shrinkage collapses all estimates to the population mean.")
        print(f"  This is the homogeneity result — no real between-family differences.")
    else:
        print(f"\n  tau-hat-sq > 0: real between-family heterogeneity detected.")

    # Per-family shrinkage
    B_i = tau_sq / (tau_sq + sigma_sq)   # shrinkage factor ∈ [0,1]
    mu_shrunk = B_i * y + (1 - B_i) * y_bar

    fam_is = fam_is.copy()
    fam_is["sigma_sq"]  = sigma_sq
    fam_is["B_i"]       = B_i
    fam_is["mu_shrunk"] = mu_shrunk

    print(f"\n  Shrinkage factor B_i (= tau-sq / (tau-sq + sigma-sq)):")
    print(f"    min={B_i.min():.3f}  median={np.median(B_i):.3f}  max={B_i.max():.3f}")
    print(f"  Median B_i={np.median(B_i):.3f} means the typical family is "
          f"{'mostly shrunk toward mean' if np.median(B_i) < 0.3 else 'retaining most of its own estimate'}")

    # ── 2. holdout outcomes ─────────────────────────────────────────────────
    hr("2. HOLDOUT OUTCOMES (survivors only)")
    ho_csv = _OUT / "holdout_test.csv"
    if not ho_csv.exists():
        print(f"  {ho_csv} not found — run tools/holdout_test.py first.")
        return

    ho_raw = pd.read_csv(ho_csv)
    # Best-net rr per family (same collapse as in-sample)
    best_ho_idx = ho_raw.groupby(FAMILY, observed=True)["expectancy_net_r"].idxmax()
    ho_fam = ho_raw.loc[best_ho_idx, FAMILY + [
        "expectancy_net_r", "ho_net", "ho_trades"]].copy()
    ho_fam = ho_fam.rename(columns={"expectancy_net_r": "is_net_r"})

    # Join shrunk estimate from in-sample table
    merged = ho_fam.merge(
        fam_is[FAMILY + ["mu_shrunk", "B_i", "sigma_sq"]],
        on=FAMILY, how="left")
    merged = merged.dropna(subset=["mu_shrunk", "ho_net"])
    n_ho = len(merged)
    if n_ho == 0:
        print("  No overlap between holdout survivors and in-sample estimates.")
        return

    print(f"  survivors with holdout data: {n_ho}")
    raw    = merged["is_net_r"].to_numpy(float)
    shrunk = merged["mu_shrunk"].to_numpy(float)
    ho     = merged["ho_net"].to_numpy(float)

    # ── 3. comparison ──────────────────────────────────────────────────────
    hr("3. OOS MSE: shrunk vs raw")
    mse_raw    = float(np.mean((raw    - ho) ** 2))
    mse_shrunk = float(np.mean((shrunk - ho) ** 2))
    mse_ratio  = mse_shrunk / mse_raw if mse_raw > 0 else float("nan")

    print(f"  OOS MSE(raw)    = {mse_raw:.6f}")
    print(f"  OOS MSE(shrunk) = {mse_shrunk:.6f}")
    print(f"  ratio (shrunk/raw): {mse_ratio:.3f}  "
          f"({'shrunk is better' if mse_ratio < 1 else 'raw is better'})")
    print()
    print(f"  Mean raw    = {raw.mean():+.4f}  vs holdout = {ho.mean():+.4f}"
          f"  bias={raw.mean()-ho.mean():+.4f}")
    print(f"  Mean shrunk = {shrunk.mean():+.4f}  vs holdout = {ho.mean():+.4f}"
          f"  bias={shrunk.mean()-ho.mean():+.4f}")

    hr("4. RANK CORRELATION: shrunk vs raw")
    rho_raw,    p_raw    = stats.spearmanr(raw,    ho)
    rho_shrunk, p_shrunk = stats.spearmanr(shrunk, ho)
    print(f"  Spearman rho (raw    vs holdout): {rho_raw:+.4f}  p={p_raw:.4f}")
    print(f"  Spearman rho (shrunk vs holdout): {rho_shrunk:+.4f}  p={p_shrunk:.4f}")
    print(f"  shrunkρ {'>' if rho_shrunk > rho_raw else '<='} rawρ  "
          f"({'shrinkage improves ranking' if rho_shrunk > rho_raw else 'no improvement from shrinkage'})")

    # Bootstrap CI on the MSE difference (shrunk - raw)
    hr("5. BOOTSTRAP CI ON MSE DIFFERENCE (shrunk - raw)")
    n_s = len(raw)
    sq_diff = (shrunk - ho) ** 2 - (raw - ho) ** 2   # negative = shrunk better
    mse_diff_est = estimate(mean_stat(sq_diff), n_s,
                            label="MSE(shrunk) - MSE(raw)",
                            n_draws=N_BOOT, seed=BOOT_SEED, unit="family")
    print(f"  Point: {mse_diff_est.point:+.6f}  "
          f"95% CI [{mse_diff_est.lo:+.6f}, {mse_diff_est.hi:+.6f}]")
    print(f"  CI includes zero: {mse_diff_est.includes_zero}")
    if not mse_diff_est.includes_zero and mse_diff_est.point < 0:
        print(f"  -> Shrinkage SIGNIFICANTLY reduces OOS MSE.")
    elif not mse_diff_est.includes_zero and mse_diff_est.point > 0:
        print(f"  -> Raw SIGNIFICANTLY outperforms shrinkage (unusual).")
    else:
        print(f"  -> CI includes zero: shrinkage and raw are indistinguishable.")

    hr("6. INTERPRETATION")
    print(f"  tau-hat-sq: {tau_sq:.6f} (I²={100*I_sq:.1f}%)")
    if tau_sq < 1e-8:
        print(f"  All observed variance is sampling noise. Optimal shrinkage")
        print(f"  collapses every estimate to the population mean ({y_bar:+.4f} R).")
        print(f"  The raw in-sample estimates contain no usable signal about")
        print(f"  which families are better — consistent with HYP-01's finding")
        print(f"  that the selection premium is zero.")
    else:
        print(f"  Between-family τ²>0 was detected. Shrinkage retains {100*np.median(B_i):.0f}%")
        print(f"  of each family's own estimate (median B_i={np.median(B_i):.3f}).")

    print()
    print(f"  This is a methodological result: it characterises whether the")
    print(f"  in-sample estimates carry ANY signal about holdout performance,")
    print(f"  after correcting for winner's curse. It is not a strategy claim.")
    print(f"  The population mean holdout net R is {y_bar:+.4f} R — negative")
    print(f"  either way.")

    # ── Save ───────────────────────────────────────────────────────────────
    results = {
        "hypothesis": "HYP-05-shrinkage",
        "section": "6.1",
        "prereg": "prereg/hyp05_prereg.json#h5e_shrinkage",
        "n_families_in_sample": int(n_fam),
        "n_survivors_with_holdout": int(n_ho),
        "tau_sq": round(tau_sq, 8),
        "I_sq": round(I_sq, 4),
        "B_i_median": round(float(np.median(B_i)), 4),
        "B_i_min": round(float(B_i.min()), 4),
        "B_i_max": round(float(B_i.max()), 4),
        "mse_raw": round(mse_raw, 8),
        "mse_shrunk": round(mse_shrunk, 8),
        "mse_ratio_shrunk_over_raw": round(float(mse_ratio), 4),
        "mse_diff_point": round(float(mse_diff_est.point), 8),
        "mse_diff_ci_lo": round(float(mse_diff_est.lo), 8),
        "mse_diff_ci_hi": round(float(mse_diff_est.hi), 8),
        "mse_diff_ci_includes_zero": bool(mse_diff_est.includes_zero),
        "rho_raw_vs_holdout": round(float(rho_raw), 4),
        "rho_shrunk_vs_holdout": round(float(rho_shrunk), 4),
        "rho_p_raw": round(float(p_raw), 4),
        "rho_p_shrunk": round(float(p_shrunk), 4),
        "verdict": (
            "SHRINKAGE SIGNIFICANTLY REDUCES OOS MSE"
            if (not mse_diff_est.includes_zero and mse_diff_est.point < 0) else
            "NO SIGNIFICANT DIFFERENCE between shrunk and raw OOS MSE"
        ),
    }
    out_path = _OUT / "hyp05_shrinkage_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
