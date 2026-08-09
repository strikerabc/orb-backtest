"""
Tests that the verdict banner never contradicts its own numbers.

Three defects in one file pair motivated this, all the same class -- a CLAIM
hardcoded beside a NUMBER that is computed, so the claim silently goes stale when
the data moves:

  1. holdout_test.py: `n_fam_total = 1350` from an earlier sweep, while section 5
     computed the real 976 -- one script, two different chance rates.
  2. report.py banner: "**1. In-sample survivors are at or below the chance
     rate.**" printed directly above "94 of 976 ... (9.63%) against ~48 expected",
     which is ABOVE. The conditional sentence after it correctly dropped out, so
     the numbers and the heading disagreed inside one paragraph.
  3. report.py docstring: same stale assertion, plus "Two independent checks say
     so" when only one still did.

The banner's whole design is "read from holdout_verdict.json so the report never
carries a stale claim". That only holds if EVERY claim is derived from the file.
These tests drive the banner with synthetic verdicts on both sides of each
threshold and assert the prose follows.
"""
from __future__ import annotations

import json

import pytest

from src.report import _verdict_banner


def _write(tmp_path, **over):
    """A verdict file with production-shaped defaults, overridable per test."""
    v = {
        "holdout_start": "2026-02-01",
        "holdout_end_exclusive": None,
        "families_rankable": 976,
        "expected_fp_at_5pct": 48,
        "survivor_families": 94,
        "survivor_variants": 188,
        "survivor_pct_of_families": 9.63,
        "holdout_families_tested": 94,
        "holdout_net_positive": 36,
        "holdout_net_positive_pct": 38.3,
        "holdout_trade_weighted_net_r": -0.0451,
        "holdout_weighted_ci_lo": -0.0794,
        "holdout_weighted_ci_hi": -0.0115,
        "holdout_simple_mean_net_r": -0.0962,
        "holdout_simple_ci_lo": -0.1523,
        "holdout_simple_ci_hi": -0.0449,
        "holdout_trades_pooled": 3481,
        "median_holdout_trades": 28,
        "below_chance_rate": False,
        "ci_includes_zero": False,
        "verdict": "NO EDGE ESTABLISHED",
    }
    v.update(over)
    (tmp_path / "holdout_verdict.json").write_text(json.dumps(v), encoding="utf-8")
    return v


def _text(tmp_path) -> str:
    return "\n".join(_verdict_banner(tmp_path))


# ── defect 2: the chance-rate heading ─────────────────────────────────────────

def test_above_chance_rate_is_not_described_as_at_or_below(tmp_path):
    """Production case. 94 of 976 is 9.63% against 5.0% expected."""
    _write(tmp_path, below_chance_rate=False)
    txt = _text(tmp_path)
    assert "at or below the chance rate" not in txt
    assert "ABOVE the chance rate" in txt
    # And it must not claim the in-sample check supports the null verdict.
    assert "cannot be distinguished from noise before the holdout" not in txt


def test_below_chance_rate_still_says_at_or_below(tmp_path):
    """Converse guard. Without this, a version that simply deleted the phrase
    would pass the test above."""
    _write(tmp_path, below_chance_rate=True, survivor_families=40,
           survivor_pct_of_families=4.1)
    txt = _text(tmp_path)
    assert "at or below the chance rate" in txt
    assert "ABOVE the chance rate" not in txt
    assert "cannot be distinguished from noise before the holdout" in txt


def test_check_count_matches_the_checks_that_actually_support_the_verdict(tmp_path):
    _write(tmp_path, below_chance_rate=False)
    assert "Two independent checks" not in _text(tmp_path)
    _write(tmp_path, below_chance_rate=True)
    assert "Two independent checks" in _text(tmp_path)


# ── defect 1's sibling: CI/statistic pairing (review finding R1) ──────────────

def test_excluding_zero_is_not_reported_as_indistinguishable_from_zero(tmp_path):
    """R1's consequence. The published banner said the trade-weighted mean was
    'not distinguishable from zero' while quoting an interval that excludes it."""
    _write(tmp_path, ci_includes_zero=False)
    txt = _text(tmp_path)
    assert "EXCLUDES zero" in txt
    assert "includes zero" not in txt
    assert "lose reliably" in txt


def test_including_zero_is_reported_as_no_edge_demonstrated(tmp_path):
    _write(tmp_path, ci_includes_zero=True,
           holdout_weighted_ci_lo=-0.079, holdout_weighted_ci_hi=0.012)
    txt = _text(tmp_path)
    assert "includes zero" in txt
    assert "EXCLUDES zero" not in txt
    assert "lose reliably" not in txt


def test_banner_prints_the_weighted_interval_beside_the_weighted_mean(tmp_path):
    """The exact mismatch: the weighted mean must be printed with ITS interval,
    not the simple mean's."""
    _write(tmp_path,
           holdout_trade_weighted_net_r=-0.0451,
           holdout_weighted_ci_lo=-0.0794, holdout_weighted_ci_hi=-0.0115,
           holdout_simple_mean_net_r=-0.0962,
           holdout_simple_ci_lo=-0.1523, holdout_simple_ci_hi=-0.0449)
    txt = _text(tmp_path)
    assert "-0.0451**, bootstrap 95% CI [-0.0794, -0.0115]" in txt
    # The simple mean's own interval must appear with the simple mean, not the
    # weighted one.
    assert "-0.1523" in txt and "-0.0449" in txt
    weighted_line = next(l for l in txt.splitlines() if "-0.0451**" in l)
    assert "-0.1523" not in weighted_line, (
        "the simple-mean interval is printed on the weighted mean's line")


def test_pre_fix_verdict_file_still_renders_via_deprecated_keys(tmp_path):
    """Verdict files written before the fix carry only holdout_ci_lo/_hi. They must
    still render -- degraded, not crashed -- because report.py may run against a
    stale artifact."""
    v = _write(tmp_path)
    for k in ("holdout_weighted_ci_lo", "holdout_weighted_ci_hi",
              "holdout_simple_ci_lo", "holdout_simple_ci_hi"):
        v.pop(k)
    v["holdout_ci_lo"], v["holdout_ci_hi"] = -0.151, -0.045
    (tmp_path / "holdout_verdict.json").write_text(json.dumps(v), encoding="utf-8")
    txt = _text(tmp_path)
    assert "-0.151" in txt and "-0.045" in txt
    # The simple-mean paragraph is omitted rather than rendered with missing keys.
    assert "Simple mean across families" not in txt
    assert "" == txt.split("\n\n")[-1].strip() or txt.strip().endswith("---")


# ── structural ────────────────────────────────────────────────────────────────

def test_missing_verdict_file_says_no_test_has_been_run(tmp_path):
    txt = _text(tmp_path)
    assert "NO OUT-OF-SAMPLE TEST HAS BEEN RUN" in txt
    assert "tools/holdout_test.py" in txt


def test_corrupt_verdict_file_returns_empty_rather_than_raising(tmp_path):
    (tmp_path / "holdout_verdict.json").write_text("{not json", encoding="utf-8")
    assert _verdict_banner(tmp_path) == []


def test_candidate_edge_verdict_drops_the_artefact_framing(tmp_path):
    _write(tmp_path, verdict="CANDIDATE EDGE -- requires forward testing",
           holdout_trade_weighted_net_r=0.05, holdout_weighted_ci_lo=0.01,
           holdout_weighted_ci_hi=0.09, ci_includes_zero=False,
           holdout_net_positive=60, holdout_net_positive_pct=63.8)
    txt = _text(tmp_path)
    assert "best read as selection artefacts" not in txt
    assert "CANDIDATE EDGE" in txt


def test_no_blank_paragraph_when_simple_mean_keys_absent(tmp_path):
    """The conditional paragraph was originally an inline expression evaluating to
    "", which left a doubled blank line in the markdown."""
    v = _write(tmp_path)
    for k in ("holdout_simple_ci_lo", "holdout_simple_ci_hi"):
        v.pop(k)
    (tmp_path / "holdout_verdict.json").write_text(json.dumps(v), encoding="utf-8")
    lines = _verdict_banner(tmp_path)
    for i in range(len(lines) - 2):
        assert not (lines[i] == "" and lines[i + 1] == "" and lines[i + 2] == ""), \
            f"three consecutive blank lines at index {i}"


def test_every_numeric_claim_comes_from_the_file(tmp_path):
    """Perturb every number and assert none of the originals survive in the text.
    This is the general guard: it fails if any figure is hardcoded rather than
    interpolated, which is the defect class this file exists for."""
    _write(tmp_path)
    baseline = _text(tmp_path)
    assert "976" in baseline and "94" in baseline and "38.3" in baseline

    _write(tmp_path, families_rankable=1234, survivor_families=77,
           survivor_variants=155, survivor_pct_of_families=6.24,
           expected_fp_at_5pct=61, holdout_families_tested=77,
           holdout_net_positive=30, holdout_net_positive_pct=39.0,
           median_holdout_trades=41, holdout_trades_pooled=2900)
    txt = _text(tmp_path)
    assert "1,234" in txt and "77" in txt and "39.0" in txt and "61" in txt
    for stale in ("976", "9.63", "38.3", "3481", " 28 "):
        assert stale not in txt, f"stale hardcoded value {stale!r} survived"
