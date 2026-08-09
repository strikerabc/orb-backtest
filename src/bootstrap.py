"""
bootstrap.py — point estimates and confidence intervals that cannot be mismatched.

Why this module exists
---------------------
`tools/holdout_test.py` printed a trade-weighted mean (-0.0451 R) directly above a
bootstrap CI of [-0.151, -0.045] that had been computed on the UNWEIGHTED mean
(-0.0962 R). Both numbers were individually correct. Pairing them was not: the
weighted point estimate happened to land at the 97.48th percentile of the
unweighted resample distribution, so it printed as the interval's own upper bound
to three significant figures.

That is worse than a visibly wrong number, because it looks like a bug in the
bootstrap -- a one-sided interval, or percentile indexing off by one end -- and
sends a reader hunting in the wrong place. The true weighted interval is
[-0.0793, -0.0119], which EXCLUDES zero, so the reported conclusion ("not
distinguishable from zero") was wrong in a way that mattered, and the error
propagated into `holdout_verdict.json` and from there into the generated report.

The structural fix is not "index the right percentile". It is to make the point
estimate and its interval come out of ONE call, computed from the same statistic
function and the same resample draws, bundled in a value that carries its own
label. A caller that never holds a naked (lo, hi) pair cannot mismatch one.

`stats._block_bootstrap_ci` returns a bare (lo, hi) and is the same shape of
hazard, but it serves a different purpose (per-trade day-block resampling for a
single variant) and rewriting it would move every number in summary.parquet.
It is left alone deliberately; use this module for new work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A point estimate and the interval of the SAME statistic.

    `boot_median` is retained so callers can assert the point estimate sits near
    the centre of its own resample distribution. That check is what would have
    caught the original defect immediately: a weighted mean sitting at the 97.5th
    percentile of its own bootstrap is either a broken interval or a broken
    estimator, and either way it must not be published.
    """

    label: str
    point: float
    lo: float
    hi: float
    boot_median: float
    n_draws: int
    unit: str = "row"

    @property
    def includes_zero(self) -> bool:
        return bool(self.lo <= 0.0 <= self.hi)

    @property
    def bias(self) -> float:
        """point - boot_median. Near zero for a well-behaved estimator."""
        return float(self.point - self.boot_median)

    @property
    def width(self) -> float:
        return float(self.hi - self.lo)

    def sign_verdict(self, name: str = "estimate") -> str:
        """Three-way reading. 'Includes zero' and 'significantly negative' are
        different findings and collapsing them loses real information."""
        if self.includes_zero:
            return f"{name} is not distinguishable from zero"
        side = "positive" if self.lo > 0 else "negative"
        return f"{name} is significantly {side}"

    def fmt(self) -> str:
        return (f"{self.point:+.4f}  95% CI [{self.lo:+.4f}, {self.hi:+.4f}]"
                f"  (n={self.n_draws:,} {self.unit} draws)")


def resample_indices(
    n: int,
    n_draws: int,
    *,
    seed: int,
    clusters: Sequence | None = None,
) -> list[np.ndarray]:
    """Row indices for `n_draws` bootstrap resamples.

    clusters=None resamples rows i.i.d. When `clusters` is given, whole clusters
    are drawn with replacement and every row of a drawn cluster is taken -- the
    standard cluster bootstrap.

    Why clustering is offered rather than assumed: the 94 survivor families are
    NOT independent. 27 of them are NQ and 15 instrument-sessions carry all 94, so
    families sharing an instrument share days, price paths and the same regime
    windows. An i.i.d. family bootstrap therefore understates the interval. It is
    still the pre-registered primary (HYP-01 section 2 fixes family-level
    resampling), so both are reported and the cluster version reads as the
    robustness check -- with the caveat that 9 instruments is few clusters and the
    cluster interval is itself noisy.
    """
    rng = np.random.default_rng(seed)
    if clusters is None:
        return [rng.integers(0, n, size=n) for _ in range(n_draws)]

    clusters = np.asarray(clusters)
    if len(clusters) != n:
        raise ValueError(f"clusters has length {len(clusters)}, expected {n}")
    groups = [np.flatnonzero(clusters == k) for k in np.unique(clusters)]
    g = len(groups)
    out = []
    for _ in range(n_draws):
        picks = rng.integers(0, g, size=g)
        out.append(np.concatenate([groups[p] for p in picks]))
    return out


def estimate(
    stat: Callable[[np.ndarray], float],
    n: int,
    *,
    label: str,
    n_draws: int,
    seed: int,
    clusters: Sequence | None = None,
    alpha: float = 0.05,
    unit: str = "row",
) -> Estimate:
    """Point estimate and CI of ONE statistic, from one set of resample draws.

    `stat` takes row indices and returns a scalar, so it closes over whatever
    columns it needs. That signature is deliberate: it lets a weighted mean
    recompute BOTH numerator and denominator from the resampled rows, which is
    required for a ratio estimator -- resampling values while holding the original
    weights fixed silently estimates a different quantity.

    The point estimate is `stat(arange(n))`: the statistic on the real sample,
    evaluated through the same code path as every resample. There is then no way
    for the two to drift apart.
    """
    if n <= 0:
        raise ValueError("cannot bootstrap an empty sample")
    point = float(stat(np.arange(n)))
    draws = resample_indices(n, n_draws, seed=seed, clusters=clusters)
    boots = np.fromiter((stat(ix) for ix in draws), dtype=float, count=n_draws)
    finite = boots[np.isfinite(boots)]
    if finite.size == 0:
        raise ValueError(f"all {n_draws} resamples produced non-finite values")
    lo, med, hi = np.percentile(
        finite, [100 * alpha / 2, 50.0, 100 * (1 - alpha / 2)])
    return Estimate(label=label, point=point, lo=float(lo), hi=float(hi),
                    boot_median=float(med), n_draws=n_draws,
                    unit=unit if clusters is None else f"{unit}-cluster")


def mean_stat(values: np.ndarray) -> Callable[[np.ndarray], float]:
    """Unweighted mean of `values` over the given rows."""
    v = np.asarray(values, dtype=float)
    return lambda ix: float(np.mean(v[ix]))


def weighted_mean_stat(values: np.ndarray,
                       weights: np.ndarray) -> Callable[[np.ndarray], float]:
    """Weighted mean, recomputing numerator AND denominator from resampled rows."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) != len(w):
        raise ValueError(f"values ({len(v)}) and weights ({len(w)}) differ in length")

    def _f(ix: np.ndarray) -> float:
        den = w[ix].sum()
        return float((w[ix] * v[ix]).sum() / den) if den else float("nan")

    return _f


def difference_stat(values: np.ndarray,
                    treat_mask: np.ndarray,
                    control_mask: np.ndarray) -> Callable[[np.ndarray], float]:
    """mean(values[treat]) - mean(values[control]), both from the SAME resample.

    Masks rather than a group label, because HYP-01's two arms are NESTED, not
    disjoint: the 94 survivor families are a subset of the 976 rankable families,
    and "unconditional holdout mean" is the mean over all of them -- survivors
    included. Overlapping membership is the normal case here, and a group label
    cannot express it.

    Two consequences worth stating where the code lives:

    1. Resampling must draw rows ONCE and recompute both arms from that draw.
       The arms are positively correlated through the shared rows, so
       bootstrapping them separately and combining widths overstates the interval
       on their difference.

    2. Nesting ATTENUATES the difference. With survivors inside the baseline, the
       measured premium is diluted by roughly the survivor share of the
       population (~9.6% here, so the dilution is small but not zero). Report the
       survivors-vs-non-survivors contrast alongside it; that one is disjoint and
       is the larger of the two by construction.
    """
    v = np.asarray(values, dtype=float)
    t = np.asarray(treat_mask, dtype=bool)
    c = np.asarray(control_mask, dtype=bool)
    if not (len(v) == len(t) == len(c)):
        raise ValueError(
            f"values ({len(v)}), treat_mask ({len(t)}) and control_mask "
            f"({len(c)}) differ in length")

    def _f(ix: np.ndarray) -> float:
        vi = v[ix]
        a, b = vi[t[ix]], vi[c[ix]]
        if a.size == 0 or b.size == 0:
            return float("nan")
        return float(a.mean() - b.mean())

    return _f
