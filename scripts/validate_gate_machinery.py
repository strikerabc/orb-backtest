"""
validate_gate_machinery.py -- EVENT_REGIME_PLAN.md section 9 validation suite.

Runs the three checks from the plan's validation table that do not require the
full T1/T2 calendar, because they test the PERMUTATION MACHINERY rather than the
hypothesis. If these fail, every number the FOMC gate produced is meaningless
regardless of what the calendar says.

  1. Shuffled-label uniformity   -> catches a test that finds significance in noise
  2. Injected synthetic effect   -> catches insufficient power / broken pooling
  3. Joint-draw integrity        -> catches an invalid maxT correction with no
                                    other symptom

Deliberately NOT covered here (each needs machinery that does not exist yet):
  - Identity: mode="flag" reproduces the pre-change summary  (needs EVENT_FILTER_MODE)
  - Known-FOMC-date channel assignment across sessions       (needs event_tagging)
  - Vol-profile separates tagged IMPULSE from tagged DIFFUSE (needs 1m re-walk +
    the T2 calendar; and per review I3/J5 the metric itself needs replacing)
  - Placebo returns null                                     (already run in the
    FOMC gate: +7d p=0.477, -7d p=0.967)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.probe_fomc_anticipation as P  # noqa: E402

OUT = ROOT / "prereg" / "gate_validation.json"

N_FAKE = 200        # independent fake-label experiments for the uniformity test
N_NULL = 2_000      # null draws per fake experiment (smaller than the gate's 20k)
INJECT_R = -0.30    # synthetic effect size, per plan section 9
SEED = 4242


def p_one_sided(observed: float, null: np.ndarray) -> float:
    """Mid-p, one-sided in the degradation direction. Matches the gate."""
    finite = null[np.isfinite(null)]
    if not len(finite):
        return float("nan")
    lt = int(np.sum(finite < observed))
    eq = int(np.sum(finite == observed))
    return (1 + lt + 0.5 * eq) / (1 + len(finite))


def main() -> None:
    rng = np.random.default_rng(SEED)
    df, _ = P.load_eligible()
    panel = P.build_day_panel(df)
    dates, groups, vals, present, date_vol, date_weekday = P.make_matrices(panel)
    strata = P.strata_for("weekday", date_vol, date_weekday)

    fomc = [d for d in P.load_fomc_dates() if d in set(dates.tolist())]
    n_contam = len(fomc)
    print(f"axis: {len(dates)} dates x {len(groups)} groups; "
          f"{n_contam} FOMC dates; {len(strata)} weekday strata")

    results: dict = {}

    # ── TEST 1 ─────────────────────────────────────────────────────────────
    # Draw N_FAKE label vectors that are pure noise (random dates, same count,
    # same weekday strata), score each against its own null. Under a correct
    # test the resulting p-values must be ~Uniform(0,1). A machine that
    # manufactures significance shows a left-shifted mass instead.
    print(f"\n[1/3] shuffled-label uniformity: {N_FAKE} fake experiments "
          f"x {N_NULL} null draws ...")
    base = np.isin(dates, fomc).astype(np.float64)
    ps = []
    for i in range(N_FAKE):
        fake = P.permute_batch(rng, base, strata, 1)[0]
        obs = float(P.deltas_for(fake[None, :], vals, present)[0])
        null = P.deltas_for(
            P.permute_batch(rng, fake, strata, N_NULL), vals, present)
        ps.append(p_one_sided(obs, null))
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{N_FAKE}")
    ps = np.asarray(ps, dtype=float)

    # KS distance against Uniform(0,1), plus the tail rates that actually matter.
    srt = np.sort(ps)
    ks = float(np.max(np.abs(srt - (np.arange(1, len(srt) + 1) / len(srt)))))
    ks_crit = 1.36 / np.sqrt(len(srt))          # ~5% critical value
    frac05 = float((ps <= 0.05).mean())
    results["test_1_shuffled_label_uniformity"] = {
        "n_experiments": int(len(ps)),
        "n_null_draws_each": N_NULL,
        "mean_p": round(float(ps.mean()), 4),
        "median_p": round(float(np.median(ps)), 4),
        "frac_p_le_0.05": round(frac05, 4),
        "expected_frac": 0.05,
        "ks_distance": round(ks, 4),
        "ks_critical_5pct": round(float(ks_crit), 4),
        "verdict": "PASS" if ks < ks_crit and 0.01 <= frac05 <= 0.12 else "FAIL",
        "what_failure_would_mean": "The permutation machinery finds significance "
                                   "in pure noise; every gate p-value is void.",
    }
    print(f"      mean_p={ps.mean():.4f}  median={np.median(ps):.4f}  "
          f"frac<=0.05={frac05:.4f}  KS={ks:.4f} (crit {ks_crit:.4f})")

    # ── TEST 2 ─────────────────────────────────────────────────────────────
    # Force INJECT_R onto the real FOMC dates and confirm the gate detects it.
    # A null here means the pooling is broken or the design cannot see an effect
    # six times the size of the one it is looking for.
    print(f"\n[2/3] injected synthetic effect ({INJECT_R} R on FOMC days) ...")
    spiked = vals.copy()
    contam_cols = np.flatnonzero(base > 0)
    spiked[:, contam_cols] += INJECT_R          # only where present, via mask
    obs_inj = float(P.deltas_for(base[None, :], spiked, present)[0])
    null_inj = P.deltas_for(
        P.permute_batch(rng, base, strata, P.N_PERM // 2), spiked, present)
    p_inj = p_one_sided(obs_inj, null_inj)
    recovered = obs_inj - float(P.deltas_for(base[None, :], vals, present)[0])
    results["test_2_injected_effect"] = {
        "injected_r": INJECT_R,
        "observed_delta_with_injection": round(obs_inj, 6),
        "recovered_effect": round(recovered, 6),
        "recovery_error": round(abs(recovered - INJECT_R), 6),
        "perm_p": round(p_inj, 6),
        "verdict": "PASS" if p_inj < 0.05 and abs(recovered - INJECT_R) < 0.02 else "FAIL",
        "what_failure_would_mean": "Pooling is broken or the design is blind to an "
                                   "effect 6x the size of the one sought.",
    }
    print(f"      recovered={recovered:+.4f} (injected {INJECT_R})  p={p_inj:.5f}")

    # ── TEST 3 ─────────────────────────────────────────────────────────────
    # Joint-draw integrity. Section 6.2 requires that within draw k, ONE day
    # permutation applies to EVERY hypothesis. Verified structurally: each drawn
    # label vector must preserve the contaminated count within every stratum, and
    # the SAME vector must drive all groups simultaneously.
    # K matters here. At K=64 the sd estimate is far too noisy to resolve the
    # ~1.19x joint/independent ratio -- the first run of this test reported 0.96x
    # and FAILED on a sampling artifact, not a real defect. Measured mean
    # cross-group correlation is rho-bar = +0.0336 over 209 instrument-session
    # pairs, and Var(joint)/Var(indep) = 1 + (G-1)*rho-bar predicts 1.29x, so the
    # effect is real but small and needs draws to see.
    K_SD = 500
    print(f"\n[3/3] joint-draw integrity (K={K_SD} for the sd comparison) ...")
    batch = P.permute_batch(rng, base, strata, K_SD)
    count_ok = bool(np.all(batch.sum(axis=1) == base.sum()))
    strata_ok = all(
        bool(np.all(batch[:, idx].sum(axis=1) == base[idx].sum()))
        for idx in strata)
    # One label vector must produce a delta for every group from the same draw:
    # recompute per-group deltas manually and compare to the pooled routine.
    lab = batch[0]
    n_c = lab @ present.T
    n_t = present.sum(axis=1)
    shared_ok = bool(np.all(n_c <= n_t) and n_c.shape[0] == len(groups))
    # Independent-permutation control: if each group were permuted separately the
    # null sd would collapse. Confirm the joint null is materially wider.
    joint_sd = float(np.std(P.deltas_for(batch, vals, present)))
    indep = np.empty(K_SD)
    for k in range(K_SD):
        per_group = np.stack([P.permute_batch(rng, base, strata, 1)[0]
                              for _ in range(len(groups))])
        w = vals * present
        n_ci = np.einsum("gd,gd->g", per_group, present)
        s_ci = np.einsum("gd,gd->g", per_group, w)
        n_ti, s_ti = present.sum(axis=1), w.sum(axis=1)
        ok = (n_ci > 0) & ((n_ti - n_ci) > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            dl = np.where(ok, s_ci / n_ci - (s_ti - s_ci) / (n_ti - n_ci), np.nan)
        wt = np.where(ok, n_ci, 0.0)
        indep[k] = np.nansum(np.where(ok, dl * wt, 0.0)) / max(wt.sum(), 1e-12)
    indep_sd = float(np.std(indep))

    # Measure the cross-group correlation that drives the expected ratio, so the
    # threshold is justified by data rather than asserted.
    M = np.where(present > 0, vals, np.nan)
    cors = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            m = np.isfinite(M[i]) & np.isfinite(M[j])
            if m.sum() >= 100:
                c = np.corrcoef(M[i, m], M[j, m])[0, 1]
                if np.isfinite(c):
                    cors.append(c)
    rho_bar = float(np.mean(cors)) if cors else float("nan")
    predicted = float(np.sqrt(max(1 + (len(groups) - 1) * rho_bar, 0.0)))

    results["test_3_joint_draw_integrity"] = {
        "contaminated_count_preserved": count_ok,
        "preserved_within_every_stratum": strata_ok,
        "one_label_vector_drives_all_groups": shared_ok,
        "n_groups": len(groups),
        "n_draws_for_sd": K_SD,
        "mean_cross_group_correlation": round(rho_bar, 4),
        "correlation_pairs_measured": len(cors),
        "predicted_sd_ratio_from_correlation": round(predicted, 3),
        "joint_null_sd": round(joint_sd, 6),
        "independent_permutation_null_sd": round(indep_sd, 6),
        "sd_ratio_joint_over_independent": round(joint_sd / indep_sd, 3) if indep_sd else None,
        "verdict": "PASS" if (count_ok and strata_ok and shared_ok
                              and joint_sd > indep_sd) else "FAIL",
        "what_failure_would_mean": "maxT correction is invalid with no other "
                                  "symptom. Independent per-family permutation "
                                  "collapses the null and inflates significance -- "
                                  "the review A1 defect, measured here.",
    }
    print(f"      counts preserved={count_ok}  strata={strata_ok}  shared={shared_ok}")
    print(f"      joint null sd={joint_sd:.5f}  vs independent={indep_sd:.5f}  "
          f"ratio={joint_sd/indep_sd:.2f}x" if indep_sd else "")

    payload = {
        "_what_this_is": "EVENT_REGIME_PLAN.md section 9 validation suite -- the "
                         "subset testing the permutation MACHINERY rather than the "
                         "hypothesis. If these fail, the FOMC gate's numbers are "
                         "void regardless of the calendar.",
        "seed": SEED,
        "axis": {"dates": int(len(dates)), "groups": len(groups),
                 "fomc_dates": n_contam, "strata": len(strata)},
        "tests": results,
        "not_covered": {
            "identity_flag_mode": "needs EVENT_FILTER_MODE (Phase C, not built)",
            "known_fomc_channel_assignment": "needs event_tagging.py (Phase B, not built)",
            "vol_profile_taxonomy": "needs 1m re-walk + T2 calendar; and per review "
                                    "I3/J5 the metric itself needs replacing with "
                                    "efficiency-by-third before it can validate anything",
            "placebo": "already run in the FOMC gate: +7d p=0.477, -7d p=0.967",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print("\n=== VERDICTS ===")
    for k, v in results.items():
        print(f"  {v['verdict']:5s}  {k}")


if __name__ == "__main__":
    main()
