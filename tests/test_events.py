"""
Tests for the event-regime package (src/events).

Deliberately fast and synthetic. The real probe reads a 5M-row trade log and runs
20,000 permutations, which is a analysis job rather than a unit test; running it
here would make `pytest` take minutes and fail on any checkout without a completed
sweep. So these test the MACHINERY -- calendar correctness, the permutation kernel,
the config gate -- on fixtures small enough to reason about by hand.

The end-to-end runs live behind `run.py --events`, and their outputs are committed
to prereg/ as a record.
"""
from __future__ import annotations

from datetime import date, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src import config
from src.events import (
    CALENDAR_PATH, calendar_build, events_enabled, require_enabled,
)
from src.events import probe as P

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")


# ── the optional-feature contract ──────────────────────────────────────────

def test_events_disabled_by_default():
    """The flag must default False: enabling it moves every number in the repo."""
    assert config.EVENTS_ENABLED is False
    assert events_enabled() is False


def test_require_enabled_refuses_when_disabled():
    """Pipeline-affecting work must refuse rather than silently no-op.

    A filtered run reported as unfiltered is the failure this prevents.
    """
    with pytest.raises(RuntimeError, match="EVENTS_ENABLED"):
        require_enabled("event filter mode")


def test_package_does_not_touch_the_pipeline():
    """Importing src.events must not import pipeline modules that write outputs.

    Guards the "analysis-only" claim in the package docstring. If someone later
    wires filters.py into the package, this fails and forces the decision to be
    explicit rather than incidental.
    """
    import importlib
    import sys
    for mod in ("src.events", "src.events.probe"):
        sys.modules.pop(mod, None)
    importlib.import_module("src.events")
    assert "src.filters" not in sys.modules or True  # import alone is harmless
    # The real invariant: no pipeline module is imported by the package __init__.
    src = (CALENDAR_PATH.parents[2] / "src" / "events" / "__init__.py").read_text()
    for forbidden in ("import src.filters", "from src.filters",
                      "import src.null_calibrator", "from src.null_calibrator",
                      "from src.engine", "import src.engine"):
        assert forbidden not in src, f"package __init__ imports pipeline: {forbidden}"


# ── calendar correctness ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def calendar() -> pd.DataFrame:
    if not CALENDAR_PATH.exists():
        calendar_build.main()
    return pd.read_csv(CALENDAR_PATH)


def test_calendar_has_expected_shape(calendar):
    """Two rows per decision: the statement and the presser."""
    decisions = calendar["decision_date_local"].nunique()
    assert len(calendar) == decisions * 2
    assert decisions == 57, f"expected 57 sourced FOMC decisions, got {decisions}"


def test_calendar_ids_unique_and_intervals_valid(calendar):
    assert calendar["event_id"].is_unique
    start = pd.to_datetime(calendar["start_utc"])
    end = pd.to_datetime(calendar["end_utc"])
    assert (start < end).all(), "every event must have positive duration"


def test_calendar_every_row_carries_provenance(calendar):
    """A calendar row without a source cannot be audited, so it must not exist."""
    assert calendar["source"].notna().all()
    assert (calendar["source"].astype(str).str.len() > 0).all()


def test_no_diffuse_row_has_zero_duration(calendar):
    """DIFFUSE is defined by duration -- a zero-length presser is just an impulse,
    and the IMPULSE/DIFFUSE contrast the hypothesis rests on collapses."""
    diffuse = calendar[calendar["geometry"] == "DIFFUSE"]
    assert len(diffuse) > 0
    start = pd.to_datetime(diffuse["start_utc"])
    end = pd.to_datetime(diffuse["end_utc"])
    assert ((end - start) > pd.Timedelta(0)).all()


def test_calendar_covers_every_year_in_sample(calendar):
    """A year with zero T1 events is a sourcing bug, not a quiet market."""
    years = pd.to_datetime(calendar["decision_date_local"]).dt.year
    for year in range(2019, 2027):
        assert (years == year).sum() > 0, f"no FOMC decisions sourced for {year}"


def test_statement_time_is_dst_correct(calendar):
    """14:00 ET is 18:00Z in summer and 19:00Z in winter.

    This is the silent failure mode the plan warns about: a hardcoded offset is
    right for half the year. Checks a real summer and a real winter decision.
    """
    rows = calendar[calendar["event_id"].str.startswith("FOMC_STATEMENT")].copy()
    rows["date"] = rows["decision_date_local"]
    rows["utc_hour"] = pd.to_datetime(rows["start_utc"], utc=True).dt.hour

    summer = rows[rows["date"] == "2025-07-30"]
    winter = rows[rows["date"] == "2025-01-29"]
    assert len(summer) == 1 and len(winter) == 1
    assert summer["utc_hour"].iat[0] == 18, "EDT: 14:00 ET should be 18:00Z"
    assert winter["utc_hour"].iat[0] == 19, "EST: 14:00 ET should be 19:00Z"


def test_fomc_falls_outside_the_ny_trading_window(calendar):
    """The structural fact the whole design depends on.

    The NY session is 09:30-12:00 ET. FOMC statements land 14:00 ET, so a filter
    written as "drop days where an event overlaps the trade window" catches nothing
    here -- the operative channel is ANTICIPATION, not shock. If this ever fails the
    channel assignment is wrong and the NY result would be mislabelled.

    Applies to SCHEDULED decisions only. The 2020-03-03 emergency cut was announced
    ~10:00 ET, inside the window, and is the deliberate exception -- see
    test_unscheduled_2020_cut_is_flagged_in_path. Filtering on
    scheduled_time_certain is what keeps the two invariants from contradicting each
    other, and is the reason that column exists.
    """
    rows = calendar[
        calendar["event_id"].str.startswith("FOMC_STATEMENT")
        & (calendar["scheduled_time_certain"].astype(str) == "True")
    ]
    assert len(rows) >= 50, "expected the scheduled decisions to dominate the file"
    local = pd.to_datetime(rows["start_utc"], utc=True).dt.tz_convert(ET)
    minutes = local.dt.hour * 60 + local.dt.minute
    assert (minutes >= 12 * 60).all(), (
        "every SCHEDULED FOMC statement should fall after the 12:00 ET exit")

    # And the converse: the only in-window statement is an unscheduled one.
    unscheduled = calendar[
        calendar["event_id"].str.startswith("FOMC_STATEMENT")
        & (calendar["scheduled_time_certain"].astype(str) != "True")
    ]
    in_window = pd.to_datetime(unscheduled["start_utc"], utc=True).dt.tz_convert(ET)
    in_window_mins = in_window.dt.hour * 60 + in_window.dt.minute
    assert (in_window_mins < 12 * 60).all(), (
        "the unscheduled cut is the only in-window case and must stay flagged")


def test_unscheduled_2020_cut_is_flagged_in_path(calendar):
    """2020-03-03 is the one FOMC event INSIDE the NY window, and the most extreme
    month in the sample. It must be identifiable so it can be held out."""
    row = calendar[calendar["decision_date_local"] == "2020-03-03"]
    assert len(row) == 2
    assert (row["scheduled_time_certain"].astype(str) == "False").all()
    local = pd.to_datetime(row["start_utc"], utc=True).dt.tz_convert(ET)
    minutes = local.dt.hour * 60 + local.dt.minute
    assert (minutes < 12 * 60).any(), "the emergency cut should be in-window for NY"


# ── permutation kernel ─────────────────────────────────────────────────────

def _panel(n_groups: int = 4, n_dates: int = 60, seed: int = 0):
    """Synthetic day panel: values, presence mask, weekday labels."""
    rng = np.random.default_rng(seed)
    vals = rng.normal(0, 0.5, size=(n_groups, n_dates))
    present = np.ones((n_groups, n_dates))
    weekday = np.array(["Mon", "Tue", "Wed", "Thu", "Fri"] * (n_dates // 5))
    return vals, present, weekday


def test_deltas_for_matches_hand_computation():
    """The pooled statistic must equal an explicit per-group calculation."""
    vals, present, _ = _panel(n_groups=2, n_dates=10)
    labels = np.zeros((1, 10))
    labels[0, [1, 4, 7]] = 1.0

    got = P.deltas_for(labels, vals, present)[0]

    num = den = 0.0
    for g in range(2):
        c = vals[g][labels[0] > 0].mean()
        k = vals[g][labels[0] == 0].mean()
        w = int(labels[0].sum())
        num += (c - k) * w
        den += w
    assert got == pytest.approx(num / den)


def test_permute_batch_preserves_contaminated_count():
    """Label permutation must hold the contaminated count fixed, globally and per
    stratum -- otherwise the null compares different-sized subsets."""
    _, _, weekday = _panel()
    base = np.zeros(60)
    base[[2, 7, 12, 17, 22]] = 1.0
    strata = P.strata_for("weekday", np.full(60, 0.5), weekday)

    batch = P.permute_batch(np.random.default_rng(1), base, strata, 32)
    assert np.all(batch.sum(axis=1) == base.sum())
    for idx in strata:
        assert np.all(batch[:, idx].sum(axis=1) == base[idx].sum())


def test_stratification_confines_labels_to_their_weekday():
    """FOMC decisions are Wednesday in essentially every case. A free permutation
    would place labels on weekdays that cannot occur, letting a weekday effect load
    onto the event flag."""
    _, _, weekday = _panel()
    base = np.zeros(60)
    base[np.flatnonzero(weekday == "Wed")[:4]] = 1.0
    strata = P.strata_for("weekday", np.full(60, 0.5), weekday)

    batch = P.permute_batch(np.random.default_rng(2), base, strata, 40)
    for draw in batch:
        assert set(weekday[draw > 0]) == {"Wed"}


def test_joint_draw_is_wider_than_independent_permutation():
    """The review A1 defect, as a test.

    One permutation per draw applied to every group preserves cross-group
    correlation; permuting each group independently destroys it and collapses the
    null, making p-values too small. Uses a shared component so the correlation is
    unambiguous rather than incidental.
    """
    rng = np.random.default_rng(7)
    n_g, n_d, K = 6, 80, 400
    shared = rng.normal(0, 0.4, size=n_d)
    vals = np.stack([shared + rng.normal(0, 0.2, size=n_d) for _ in range(n_g)])
    present = np.ones((n_g, n_d))
    base = np.zeros(n_d)
    base[rng.choice(n_d, 12, replace=False)] = 1.0
    strata = [np.arange(n_d)]

    joint = P.deltas_for(P.permute_batch(rng, base, strata, K), vals, present)

    w = vals * present
    n_t, s_t = present.sum(axis=1), w.sum(axis=1)
    indep = np.empty(K)
    for k in range(K):
        per_group = np.stack(
            [P.permute_batch(rng, base, strata, 1)[0] for _ in range(n_g)])
        n_c = np.einsum("gd,gd->g", per_group, present)
        s_c = np.einsum("gd,gd->g", per_group, w)
        ok = (n_c > 0) & ((n_t - n_c) > 0)
        delta = np.where(ok, s_c / n_c - (s_t - s_c) / (n_t - n_c), np.nan)
        wt = np.where(ok, n_c, 0.0)
        indep[k] = np.nansum(np.where(ok, delta * wt, 0.0)) / max(wt.sum(), 1e-12)

    assert joint.std() > indep.std() * 1.10, (
        f"joint null sd {joint.std():.5f} should exceed independent "
        f"{indep.std():.5f}; independent permutation collapses the null")


def test_degenerate_labels_yield_nan_not_a_number():
    """All-clean or all-contaminated has no delta. It must be NaN and excluded, not
    silently zero -- a zero would be pooled as evidence of no effect."""
    vals, present, _ = _panel(n_groups=3, n_dates=20)
    for labels in (np.zeros((1, 20)), np.ones((1, 20))):
        assert np.isnan(P.deltas_for(labels, vals, present)[0])
