"""
Regression tests for point-estimate/interval pairing.

The bug these cover (review finding R1): holdout_test.py computed
    pooled = sum(w*x)/sum(w)                    # trade-weighted mean
    boots = [mean(choice(x)) for _ in range(N)] # UNWEIGHTED resamples
    lo, hi = percentile(boots, [2.5, 97.5])
and printed `pooled` above `[lo, hi]` as a matched pair. The weighted estimate
landed at the 97.48th percentile of the unweighted distribution, so the pair read
as a one-sided interval and the published conclusion ("not distinguishable from
zero") was wrong -- the correctly weighted interval excludes zero.

Both numbers were individually right. Only the pairing was wrong, which is why no
test caught it: any test asserting "the CI brackets the simple mean" passed, and
any test asserting "pooled is negative" passed too.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.bootstrap import (
    Estimate, difference_stat, estimate, mean_stat, resample_indices,
    weighted_mean_stat,
)

N_DRAWS = 4000
SEED = 12345


def _skewed_weights(n: int, rng) -> np.ndarray:
    """Weights spread over ~2 orders of magnitude, like real holdout trade counts
    (median 28, max 103)."""
    return np.exp(rng.normal(3.0, 1.0, size=n))


# ── the defect itself ──────────────────────────────────────────────────────────

def test_weighted_and_unweighted_means_differ_enough_to_matter():
    """Guards the premise. If the two statistics agreed, R1 would be cosmetic and
    the rest of this file would be testing nothing."""
    rng = np.random.default_rng(SEED)
    n = 94
    w = _skewed_weights(n, rng)
    # Correlate value with weight, which is the real pattern: families with more
    # holdout trades held up better (-0.045 weighted vs -0.096 simple).
    x = -0.12 + 0.03 * (np.log(w) - np.log(w).mean()) + rng.normal(0, 0.15, n)
    simple = float(x.mean())
    weighted = float((w * x).sum() / w.sum())
    assert abs(weighted - simple) > 0.02, (
        f"fixture is degenerate: weighted {weighted:+.4f} vs simple {simple:+.4f}")


def _holdout_like(n: int, rng, *, gap: float = 0.08, per_trade_sd: float = 1.0):
    """Fixture matching the real holdout structure.

    Two features matter and BOTH are load-bearing:

    1. Each family's value is itself a mean over `w` trades, so its standard error
       is per_trade_sd / sqrt(w). Weighting by trade count is then approximately
       INVERSE-VARIANCE weighting -- which is the only reason trade-weighting
       reduces variance rather than just concentrating it.
    2. Value correlates with log-weight, so the weighted and unweighted means
       estimate visibly different numbers (production: -0.045 vs -0.096).

    Calibrated against the production holdout (94 families, log-weight sd 1.007,
    median 28 trades). At the defaults this reproduces all four observed quantities:

        quantity                       production   fixture
        simple mean                      -0.0962    -0.0965
        weighted mean                    -0.0451    -0.0444
        gap                              +0.0511    +0.0521
        weighted pct of unweighted dist     97.4       97.8

    per_trade_sd = 1.0 is not a fitted parameter -- it is the per-trade sd the repo
    already documents (holdout_test.py power caveat). Only `gap` was tuned.
    """
    w = _skewed_weights(n, rng)
    truth = -0.12 + gap * (np.log(w) - np.log(w).mean())
    noise = rng.normal(0, per_trade_sd / np.sqrt(w))
    return w, truth + noise


def test_weighted_point_lands_in_the_tail_of_the_unweighted_interval():
    """Reproduces R1's actual signature. In production the weighted mean sat at the
    97.48th percentile of the UNWEIGHTED resample distribution -- inside it, but far
    enough into the tail that it printed as that interval's own upper bound to three
    significant figures.

    That is the tell: a point estimate in the extreme tail of the interval it is
    published beside is either a broken interval or a mismatched statistic. Here it
    was the second.
    """
    rng = np.random.default_rng(SEED)
    n = 94
    w, x = _holdout_like(n, rng)

    wrong = estimate(mean_stat(x), n, label="unweighted", n_draws=N_DRAWS, seed=SEED)
    right = estimate(weighted_mean_stat(x, w), n, label="weighted",
                     n_draws=N_DRAWS, seed=SEED)

    draws = resample_indices(n, N_DRAWS, seed=SEED)
    unweighted_boots = np.array([x[ix].mean() for ix in draws])
    pct = 100.0 * float((unweighted_boots < right.point).mean())
    assert pct > 90.0, (
        f"weighted mean {right.point:+.4f} sits at only the {pct:.1f}th percentile "
        f"of the unweighted distribution; fixture does not reproduce the defect")

    # Far enough apart that pairing one with the other's interval is not a rounding
    # inconvenience.
    assert abs(right.point - wrong.point) > 0.25 * wrong.width
    # With per-family precision scaling as sqrt(trades), trade-weighting is
    # inverse-variance weighting and the correct interval is TIGHTER -- as in
    # production (0.0674 weighted vs 0.106 unweighted).
    assert right.width < wrong.width


def test_weight_skew_widens_the_interval_when_precision_does_not_scale():
    """The converse, and the reason 'trade-weighting is the honest statistic' is a
    conditional claim rather than a general one.

    Weighting is variance-reducing only because per-family noise falls as
    1/sqrt(trades). If family values instead carry equal noise -- which is what
    HETEROGENEOUS TRUE MEANS across families look like, since that dispersion does
    not shrink with more trades -- then weighting merely concentrates weight on
    fewer families, cutting the effective sample size and WIDENING the interval.

    Kish effective n = (sum w)^2 / sum(w^2). Under the skew used here that is a
    fraction of the nominal 94.
    """
    rng = np.random.default_rng(SEED)
    n = 94
    w = _skewed_weights(n, rng)
    x = rng.normal(-0.09, 0.15, n)          # homoskedastic: precision ignores w

    unw = estimate(mean_stat(x), n, label="unweighted", n_draws=N_DRAWS, seed=SEED)
    wtd = estimate(weighted_mean_stat(x, w), n, label="weighted",
                   n_draws=N_DRAWS, seed=SEED)

    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    assert n_eff < 0.75 * n, f"weights not skewed enough: n_eff {n_eff:.1f} of {n}"
    assert wtd.width > unw.width, (
        f"weighted {wtd.width:.4f} vs unweighted {unw.width:.4f} at n_eff "
        f"{n_eff:.1f}/{n}")


def test_estimate_point_matches_its_own_bootstrap_median():
    """The check that would have caught R1 on sight. A point estimate sitting in
    the tail of its own resample distribution is not publishable."""
    rng = np.random.default_rng(SEED)
    n = 200
    w = _skewed_weights(n, rng)
    x = rng.normal(-0.05, 0.2, n)

    for label, stat in (("simple", mean_stat(x)),
                        ("weighted", weighted_mean_stat(x, w))):
        est = estimate(stat, n, label=label, n_draws=N_DRAWS, seed=SEED)
        assert abs(est.bias) < 0.1 * est.width, (
            f"{label}: point {est.point:+.6f} vs boot median "
            f"{est.boot_median:+.6f}, width {est.width:.6f}")
        assert est.lo < est.point < est.hi


# ── ratio-estimator correctness ────────────────────────────────────────────────

def test_weighted_bootstrap_recomputes_the_denominator():
    """A weighted mean is a RATIO. Resampling values while holding the original
    weights fixed estimates a different quantity, and the error is invisible
    because the result still looks like a plausible interval."""
    rng = np.random.default_rng(SEED)
    n = 120
    w = _skewed_weights(n, rng)
    x = rng.normal(0.0, 0.25, n)

    correct = estimate(weighted_mean_stat(x, w), n, label="ratio",
                       n_draws=N_DRAWS, seed=SEED)

    # The wrong version: fixed weights, resampled values only.
    draws = resample_indices(n, N_DRAWS, seed=SEED)
    bad = np.array([(w * x[ix]).sum() / w.sum() for ix in draws])
    bad_lo, bad_hi = np.percentile(bad, [2.5, 97.5])

    assert abs((bad_hi - bad_lo) - correct.width) > 1e-6, (
        "fixed-weight resampling produced the same interval; test is not "
        "discriminating")


def test_weighted_equals_simple_when_weights_are_equal():
    rng = np.random.default_rng(SEED)
    n = 80
    x = rng.normal(0.1, 0.3, n)
    w = np.full(n, 7.0)
    a = estimate(mean_stat(x), n, label="simple", n_draws=1500, seed=SEED)
    b = estimate(weighted_mean_stat(x, w), n, label="weighted", n_draws=1500, seed=SEED)
    assert a.point == pytest.approx(b.point, abs=1e-12)
    assert a.lo == pytest.approx(b.lo, abs=1e-12)
    assert a.hi == pytest.approx(b.hi, abs=1e-12)


# ── clustering ─────────────────────────────────────────────────────────────────

def test_cluster_bootstrap_is_wider_when_clusters_carry_the_signal():
    """94 families across 9 instruments, 27 of them NQ. When family outcomes are
    correlated within instrument, the i.i.d. family bootstrap is anti-conservative
    and the cluster interval must be wider."""
    rng = np.random.default_rng(SEED)
    clusters = np.repeat(np.arange(9), 12)
    n = len(clusters)
    cluster_effect = rng.normal(0, 0.30, 9)      # dominates
    x = cluster_effect[clusters] + rng.normal(0, 0.05, n)   # tiny idiosyncratic

    iid = estimate(mean_stat(x), n, label="iid", n_draws=N_DRAWS, seed=SEED)
    clu = estimate(mean_stat(x), n, label="clustered", n_draws=N_DRAWS, seed=SEED,
                   clusters=clusters)
    assert clu.width > 1.5 * iid.width, (
        f"clustered {clu.width:.4f} vs iid {iid.width:.4f}")
    assert clu.unit.endswith("-cluster")


def test_cluster_bootstrap_matches_iid_when_clusters_are_meaningless():
    """Converse guard: if within-cluster correlation is absent, clustering must not
    inflate the interval materially. Without this the test above would pass for a
    version that simply always widens."""
    rng = np.random.default_rng(SEED)
    clusters = np.repeat(np.arange(12), 25)
    n = len(clusters)
    x = rng.normal(0.0, 0.2, n)          # no cluster effect at all
    iid = estimate(mean_stat(x), n, label="iid", n_draws=N_DRAWS, seed=SEED)
    clu = estimate(mean_stat(x), n, label="clustered", n_draws=N_DRAWS, seed=SEED,
                   clusters=clusters)
    assert 0.6 < clu.width / iid.width < 1.7


def test_cluster_resample_draws_whole_clusters():
    """A drawn cluster contributes ALL its rows or none. Note the resample length
    varies when cluster sizes differ (here 3 clusters of size 3/2/1 give totals from
    3 to 9) -- that is correct cluster-bootstrap behaviour, not a defect. The number
    of CLUSTERS is what is held fixed."""
    sizes = {0: 3, 1: 2, 2: 1}
    clusters = np.array([0, 0, 0, 1, 1, 2])
    lengths = set()
    for ix in resample_indices(6, 60, seed=SEED, clusters=clusters):
        counts = {k: int((clusters[ix] == k).sum()) for k in sizes}
        for k, size in sizes.items():
            assert counts[k] % size == 0, f"cluster {k} split: {counts[k]} of {size}"
        assert sum(counts.values()) == len(ix)
        assert sum(counts[k] // size for k, size in sizes.items()) == len(sizes)
        lengths.add(len(ix))
    assert len(lengths) > 1, "unequal cluster sizes should give varying lengths"


def test_cluster_length_mismatch_raises():
    with pytest.raises(ValueError, match="clusters has length"):
        resample_indices(10, 5, seed=SEED, clusters=np.arange(3))


# ── difference estimator (HYP-01 primary) ──────────────────────────────────────

def test_difference_stat_recovers_an_injected_gap_disjoint():
    rng = np.random.default_rng(SEED)
    n = 600
    surv = rng.random(n) < 0.25
    x = np.where(surv, 0.04, 0.0) + rng.normal(-0.09, 0.20, n)
    est = estimate(difference_stat(x, surv, ~surv), n,
                   label="premium", n_draws=N_DRAWS, seed=SEED)
    assert est.point == pytest.approx(0.04, abs=0.035)
    assert est.lo < 0.04 < est.hi


def test_difference_stat_null_covers_zero():
    rng = np.random.default_rng(SEED)
    n = 600
    surv = rng.random(n) < 0.3
    x = rng.normal(-0.09, 0.20, n)          # no real difference
    est = estimate(difference_stat(x, surv, ~surv), n,
                   label="premium", n_draws=N_DRAWS, seed=SEED)
    assert est.includes_zero


def test_nested_control_attenuates_the_measured_premium():
    """HYP-01's structure: survivors are INSIDE the baseline population. Measuring
    survivors-vs-all therefore understates the true gap relative to
    survivors-vs-non-survivors, by roughly the survivor share.

    This matters for reading the result, not just for the arithmetic: the review
    quotes a survivor figure against a population baseline, and if that baseline
    contains the survivors the premium it implies is the attenuated one.
    """
    rng = np.random.default_rng(SEED)
    n = 1000
    surv = rng.random(n) < 0.10                 # ~9.6% in production
    true_gap = 0.05
    x = np.where(surv, true_gap, 0.0) + rng.normal(-0.09, 0.18, n)
    allrows = np.ones(n, dtype=bool)

    nested = estimate(difference_stat(x, surv, allrows), n, label="vs all",
                      n_draws=N_DRAWS, seed=SEED)
    disjoint = estimate(difference_stat(x, surv, ~surv), n, label="vs non-surv",
                        n_draws=N_DRAWS, seed=SEED)

    share = float(surv.mean())
    assert nested.point < disjoint.point
    # Attenuation factor is (1 - share): E[nested] = (1-share) * E[disjoint].
    assert nested.point == pytest.approx((1.0 - share) * disjoint.point, rel=0.05)


def test_nested_draw_is_narrower_than_two_independent_bootstraps():
    """Both arms come from the same holdout rows and overlap in membership, so one
    draw must produce both. Bootstrapping them separately and combining widths in
    quadrature ignores their positive correlation and overstates the interval."""
    rng = np.random.default_rng(SEED)
    n = 800
    surv = rng.random(n) < 0.35
    x = rng.normal(-0.09, 0.20, n)
    allrows = np.ones(n, dtype=bool)

    joint = estimate(difference_stat(x, surv, allrows), n, label="joint",
                     n_draws=N_DRAWS, seed=SEED)
    a = estimate(mean_stat(x[surv]), int(surv.sum()), label="a",
                 n_draws=N_DRAWS, seed=SEED)
    b = estimate(mean_stat(x), n, label="b", n_draws=N_DRAWS, seed=SEED + 1)
    naive_width = float(np.hypot(a.width, b.width))
    assert joint.width < 0.9 * naive_width, (
        f"joint {joint.width:.4f} vs naive-independent {naive_width:.4f}")


def test_difference_stat_length_mismatch_raises():
    with pytest.raises(ValueError, match="differ in length"):
        difference_stat(np.zeros(5), np.ones(5, bool), np.ones(4, bool))


# ── Estimate semantics ─────────────────────────────────────────────────────────

def test_sign_verdict_separates_null_from_significantly_negative():
    """The distinction R1 destroyed. 'Includes zero' and 'significantly negative'
    are different findings; the second carries exploitable information."""
    null = Estimate("n", -0.045, -0.151, 0.020, -0.045, 100)
    neg = Estimate("n", -0.045, -0.079, -0.012, -0.045, 100)
    assert null.includes_zero and "not distinguishable" in null.sign_verdict()
    assert not neg.includes_zero
    assert "significantly negative" in neg.sign_verdict()
    pos = Estimate("n", 0.05, 0.01, 0.09, 0.05, 100)
    assert "significantly positive" in pos.sign_verdict()


def test_estimate_is_reproducible_under_seed():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50)
    kw = dict(label="x", n_draws=800, seed=99)
    assert estimate(mean_stat(x), 50, **kw) == estimate(mean_stat(x), 50, **kw)


def test_empty_sample_raises():
    with pytest.raises(ValueError, match="empty sample"):
        estimate(mean_stat(np.array([])), 0, label="x", n_draws=10, seed=1)


def test_all_non_finite_raises():
    with pytest.raises(ValueError, match="non-finite"):
        estimate(lambda ix: float("nan"), 10, label="x", n_draws=10, seed=1)


def test_weighted_mean_length_mismatch_raises():
    with pytest.raises(ValueError, match="differ in length"):
        weighted_mean_stat(np.zeros(5), np.zeros(4))
