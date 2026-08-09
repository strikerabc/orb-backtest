"""
src.events -- event-regime conditioning: does macro-event exposure degrade the ORB edge?

Hypothesis under test: the ORB continuation edge degrades when a session's price
path is governed by *gradually revealed* macro narrative -- pressers, Q&A,
testimony -- rather than by order-flow continuation from the opening range.
Specification in docs/EVENT_REGIME_PLAN.md; findings and corrections in
docs/EVENT_REGIME_REVIEW.md.

## What "optional" means here, precisely

This package is **analysis-only and does not touch the main pipeline.** Nothing in
`main.py`, `src/engine/sweep.py`, `src/filters.py` or `src/null_calibrator.py`
imports it, and enabling it changes no sweep number. It reads
`outputs/trade_log.parquet` after the fact and writes its own JSON into `prereg/`.

That distinction matters because the plan's later phases *would* alter the pipeline
(an `EVENT_FILTER_MODE` in `filters.py`, an event-aware null pool). Those are not
built. `config.EVENTS_ENABLED` exists to gate them when they are, and defaults to
False so merging this package cannot move an existing result.

So today "optional" means: available as a command (`run.py --events ...`) and as a
test suite, not wired into the backtest.

## Status

Built and validated:
  - FOMC T1 calendar, sourced from Federal Reserve archives (57 decisions)
  - Day-level permutation gate on the ANTICIPATION channel, with weekday and
    volatility stratification, +/-7d placebos, and a Level-0 split that keeps the
    originating families out of the pooled headline
  - Machinery validation: shuffled-label uniformity, injected-effect recovery,
    joint-draw integrity

Not built (each needs a phase of its own -- see the plan):
  - `event_tagging.py`: five exposure channels, `SessionDay` fields
  - `EVENT_FILTER_MODE` in `filters.py`, event-aware null pool
  - T2/T3 speaker calendar, which is where the reported mechanism actually lives
  - Path-shape metrics from 1m bars. Note `tap_in_bar_idx` is NOT a reversal
    detector -- it is exactly 0.0 for CC/II and 1.0 for R-CC/R-II/TI, i.e. a
    deterministic function of entry mode. The plan calls reversal rate "free"; it
    does not currently exist. See review section I3.

## Reproducibility

These figures were computed with `config.HOLDOUT_PIN_START = "2026-02-01"`. The pin is
not optional for this hypothesis: unpinned, the holdout boundary slides with `data_end`,
and after the August 2026 data extension it lands on the paper-trading period
(2026-07-25 -> 2026-08-07) that generated the hypothesis -- so the holdout would contain
its own originating observation. `tools/holdout_test.py` needs
`HOLDOUT_END_EXCLUSIVE = "2026-07-25"` for the same reason. Both default to None on
main, so they must be set deliberately when re-running this analysis.

## Headline result

Delta -0.0565 R on FOMC anticipation days, unstratified p = 0.024 but weekday-
stratified p = 0.087 -- roughly half the raw effect is Wednesday, not FOMC.
MDE 0.067 R exceeds both the observed effect and the pre-registered 0.05, so this is
**underpowered rather than null**. Separately: clean-day expectancy is -0.0885 R and
contaminated -0.1471 R. Both firmly negative, so no event filter rescues the
strategy -- the effect is real in direction and a rounding error against the level.
"""
from __future__ import annotations

__all__ = [
    "build_calendar", "run_probe", "run_validation", "run_case_study",
    "events_enabled", "require_enabled", "CALENDAR_PATH",
]

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
CALENDAR_PATH = _ROOT / "data" / "events" / "fomc_decisions.csv"


def events_enabled() -> bool:
    """Whether event conditioning may influence the pipeline.

    False by default. Gates the unbuilt Phase C work (filter mode, event-aware null
    pool), not the read-only analysis in this package -- the probe and validation
    are safe to run regardless because they cannot alter a sweep result.
    """
    from src import config
    return bool(getattr(config, "EVENTS_ENABLED", False))


def require_enabled(feature: str) -> None:
    """Guard for anything that WOULD change pipeline output.

    Kept deliberately strict: silently ignoring the flag is how a filtered run gets
    reported as an unfiltered one.
    """
    if not events_enabled():
        raise RuntimeError(
            f"{feature} requires config.EVENTS_ENABLED = True. It is False by "
            "default because enabling it changes expectancies, breadth metrics, "
            "null pools and the holdout -- see docs/EVENT_REGIME_PLAN.md section 5.2.")


def build_calendar() -> Path:
    """Regenerate the versioned FOMC calendar. Returns its path."""
    from src.events import calendar_build
    calendar_build.main()
    return CALENDAR_PATH


def run_probe() -> dict:
    """Run the FOMC anticipation gate. Returns the parsed results."""
    import json
    from src.events import probe
    probe.main()
    return json.loads((_ROOT / "prereg" / "probe_results.json").read_text())


def run_validation() -> dict:
    """Run the permutation-machinery validation suite. Returns parsed results."""
    import json
    from src.events import validation
    validation.main()
    return json.loads((_ROOT / "prereg" / "gate_validation.json").read_text())


def run_case_study() -> dict:
    """Run the 2026-08-07 path-shape case study. Returns parsed results."""
    import json
    from src.events import case_study
    case_study.main()
    return json.loads(
        (_ROOT / "prereg" / "case_study_barkin_20260807.json").read_text())
