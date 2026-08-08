"""
build_fomc_calendar.py -- generate data/events/fomc_decisions.csv.

Phase A, minimal slice: FOMC rate decisions only, 2019-04-01 -> 2026-04-30.

Why a builder script and not a hand-typed CSV: the meeting dates are sourced
facts, but the UTC timestamps are DERIVED from them via a timezone conversion
that must be DST-correct. Doing that conversion in code with `zoneinfo` and no
hardcoded offsets means the ET->UTC mapping cannot silently drift, and the
provenance of every row is auditable by re-running this file.

Sources (fetched, not recalled -- plan section 3.2):
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  https://www.federalreserve.gov/monetarypolicy/fomchistorical2019.htm
  https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm

Release convention: the policy statement is released at 14:00 ET on the second
day of each scheduled meeting, with the Chair's press conference at 14:30 ET.
Unchanged since 2013. Powell moved to a press conference at EVERY meeting from
January 2019, so every decision in this sample carries one -- which is what makes
FOMC the purest DIFFUSE anticipation case in the calendar.

Deliberately EXCLUDED (not rate decisions):
  2019-10-11  repo-operations statement (unscheduled 10-04 conference call)
  2020-03-15  Sunday -- no trading session, drops out naturally
  2020-03-18  regularly scheduled meeting, CANCELLED
  2020-08-27  notation vote, Statement on Longer-Run Goals
  2025-08-22  notation vote, strategy statement
"""
from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "events" / "fomc_decisions.csv"

STATEMENT_ET = time(14, 0)
STATEMENT_DURATION_MIN = 5      # scalar release; IMPULSE geometry
PRESSER_ET = time(14, 30)
PRESSER_DURATION_MIN = 60       # EVENT_DEFAULT_DURATION_MIN["FOMC_PRESSER"]

# (decision_date, scheduled, statement_et, presser_et, verified, note)
# decision_date = the day the STATEMENT is released = day 2 of a 2-day meeting.
MEETINGS: list[tuple[str, bool, time | None, time | None, bool, str]] = [
    # ---- 2019 (sample starts 2019-04-01; Jan 30 and Mar 20 are out of range) --
    ("2019-05-01", True, STATEMENT_ET, PRESSER_ET, True, "Apr 30-May 1"),
    ("2019-06-19", True, STATEMENT_ET, PRESSER_ET, True, "Jun 18-19, SEP"),
    ("2019-07-31", True, STATEMENT_ET, PRESSER_ET, True, "Jul 30-31"),
    ("2019-09-18", True, STATEMENT_ET, PRESSER_ET, True, "Sep 17-18, SEP"),
    ("2019-10-30", True, STATEMENT_ET, PRESSER_ET, True, "Oct 29-30"),
    ("2019-12-11", True, STATEMENT_ET, PRESSER_ET, True, "Dec 10-11, SEP"),
    # ---- 2020 -------------------------------------------------------------
    ("2020-01-29", True, STATEMENT_ET, PRESSER_ET, True, "Jan 28-29"),
    # Unscheduled emergency 50bp cut. Meeting Mar 2, statement Mar 3 ~10:00 ET,
    # which is INSIDE the 09:30-12:00 ET window. The only genuine in-path FOMC
    # event in the sample, and it sits in the most extreme month in the sample.
    # Intraday timing NOT confirmed against a primary source -> verified=False.
    ("2020-03-03", False, time(10, 0), time(11, 0), False,
     "UNSCHEDULED emergency 50bp cut; statement ~10:00 ET -- IN-PATH for NY"),
    ("2020-04-29", True, STATEMENT_ET, PRESSER_ET, True, "Apr 28-29"),
    ("2020-06-10", True, STATEMENT_ET, PRESSER_ET, True, "Jun 9-10, SEP"),
    ("2020-07-29", True, STATEMENT_ET, PRESSER_ET, True, "Jul 28-29"),
    ("2020-09-16", True, STATEMENT_ET, PRESSER_ET, True, "Sep 15-16, SEP"),
    ("2020-11-05", True, STATEMENT_ET, PRESSER_ET, True, "Nov 4-5"),
    ("2020-12-16", True, STATEMENT_ET, PRESSER_ET, True, "Dec 15-16, SEP"),
    # ---- 2021 -------------------------------------------------------------
    ("2021-01-27", True, STATEMENT_ET, PRESSER_ET, True, "Jan 26-27"),
    ("2021-03-17", True, STATEMENT_ET, PRESSER_ET, True, "Mar 16-17, SEP"),
    ("2021-04-28", True, STATEMENT_ET, PRESSER_ET, True, "Apr 27-28"),
    ("2021-06-16", True, STATEMENT_ET, PRESSER_ET, True, "Jun 15-16, SEP"),
    ("2021-07-28", True, STATEMENT_ET, PRESSER_ET, True, "Jul 27-28"),
    ("2021-09-22", True, STATEMENT_ET, PRESSER_ET, True, "Sep 21-22, SEP"),
    ("2021-11-03", True, STATEMENT_ET, PRESSER_ET, True, "Nov 2-3"),
    ("2021-12-15", True, STATEMENT_ET, PRESSER_ET, True, "Dec 14-15, SEP"),
    # ---- 2022 -------------------------------------------------------------
    ("2022-01-26", True, STATEMENT_ET, PRESSER_ET, True, "Jan 25-26"),
    ("2022-03-16", True, STATEMENT_ET, PRESSER_ET, True, "Mar 15-16, SEP"),
    ("2022-05-04", True, STATEMENT_ET, PRESSER_ET, True, "May 3-4"),
    ("2022-06-15", True, STATEMENT_ET, PRESSER_ET, True, "Jun 14-15, SEP"),
    ("2022-07-27", True, STATEMENT_ET, PRESSER_ET, True, "Jul 26-27"),
    ("2022-09-21", True, STATEMENT_ET, PRESSER_ET, True, "Sep 20-21, SEP"),
    ("2022-11-02", True, STATEMENT_ET, PRESSER_ET, True, "Nov 1-2"),
    ("2022-12-14", True, STATEMENT_ET, PRESSER_ET, True, "Dec 13-14, SEP"),
    # ---- 2023 -------------------------------------------------------------
    ("2023-02-01", True, STATEMENT_ET, PRESSER_ET, True, "Jan 31-Feb 1"),
    ("2023-03-22", True, STATEMENT_ET, PRESSER_ET, True, "Mar 21-22, SEP"),
    ("2023-05-03", True, STATEMENT_ET, PRESSER_ET, True, "May 2-3"),
    ("2023-06-14", True, STATEMENT_ET, PRESSER_ET, True, "Jun 13-14, SEP"),
    ("2023-07-26", True, STATEMENT_ET, PRESSER_ET, True, "Jul 25-26"),
    ("2023-09-20", True, STATEMENT_ET, PRESSER_ET, True, "Sep 19-20, SEP"),
    ("2023-11-01", True, STATEMENT_ET, PRESSER_ET, True, "Oct 31-Nov 1"),
    ("2023-12-13", True, STATEMENT_ET, PRESSER_ET, True, "Dec 12-13, SEP"),
    # ---- 2024 -------------------------------------------------------------
    ("2024-01-31", True, STATEMENT_ET, PRESSER_ET, True, "Jan 30-31"),
    ("2024-03-20", True, STATEMENT_ET, PRESSER_ET, True, "Mar 19-20, SEP"),
    ("2024-05-01", True, STATEMENT_ET, PRESSER_ET, True, "Apr 30-May 1"),
    ("2024-06-12", True, STATEMENT_ET, PRESSER_ET, True, "Jun 11-12, SEP"),
    ("2024-07-31", True, STATEMENT_ET, PRESSER_ET, True, "Jul 30-31"),
    ("2024-09-18", True, STATEMENT_ET, PRESSER_ET, True, "Sep 17-18, SEP"),
    ("2024-11-07", True, STATEMENT_ET, PRESSER_ET, True, "Nov 6-7"),
    ("2024-12-18", True, STATEMENT_ET, PRESSER_ET, True, "Dec 17-18, SEP"),
    # ---- 2025 -------------------------------------------------------------
    ("2025-01-29", True, STATEMENT_ET, PRESSER_ET, True, "Jan 28-29"),
    ("2025-03-19", True, STATEMENT_ET, PRESSER_ET, True, "Mar 18-19, SEP"),
    ("2025-05-07", True, STATEMENT_ET, PRESSER_ET, True, "May 6-7"),
    ("2025-06-18", True, STATEMENT_ET, PRESSER_ET, True, "Jun 17-18, SEP"),
    ("2025-07-30", True, STATEMENT_ET, PRESSER_ET, True, "Jul 29-30"),
    ("2025-09-17", True, STATEMENT_ET, PRESSER_ET, True, "Sep 16-17, SEP"),
    ("2025-10-29", True, STATEMENT_ET, PRESSER_ET, True, "Oct 28-29"),
    ("2025-12-10", True, STATEMENT_ET, PRESSER_ET, True, "Dec 9-10, SEP"),
    # ---- 2026 (data ends 2026-04-30) --------------------------------------
    ("2026-01-28", True, STATEMENT_ET, PRESSER_ET, True, "Jan 27-28"),
    ("2026-03-18", True, STATEMENT_ET, PRESSER_ET, True, "Mar 17-18, SEP -- IN HOLDOUT"),
    ("2026-04-29", True, STATEMENT_ET, PRESSER_ET, True, "Apr 28-29 -- IN HOLDOUT"),
]

SAMPLE_START = date(2019, 4, 1)
SAMPLE_END = date(2026, 4, 30)
SOURCE_SCHEDULED = "federalreserve.gov/monetarypolicy/fomccalendars.htm"
SOURCE_2019 = "federalreserve.gov/monetarypolicy/fomchistorical2019.htm"
SOURCE_2020 = "federalreserve.gov/monetarypolicy/fomchistorical2020.htm"

FIELDS = [
    "event_id", "start_utc", "end_utc", "geometry", "tier", "region",
    "speaker", "title", "affects_symbols", "scheduled_time_certain",
    "source", "verified", "decision_date_local", "note",
]


def _to_utc(day: date, wall: time) -> datetime:
    """Localise an ET wall-clock time on a given date, then convert to UTC.

    DST is resolved by zoneinfo from the date itself. No hardcoded offsets:
    14:00 ET is 18:00Z in summer and 19:00Z in winter, and getting that wrong
    is exactly the silent failure plan section 4.4 warns about.
    """
    return datetime.combine(day, wall, tzinfo=ET).astimezone(UTC)


def _source_for(day: date, scheduled: bool) -> str:
    if not scheduled or day.year == 2020:
        return SOURCE_2020 if day.year == 2020 else SOURCE_SCHEDULED
    if day.year == 2019:
        return SOURCE_2019
    return SOURCE_SCHEDULED


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for iso, scheduled, stmt_et, presser_et, verified, note in MEETINGS:
        day = date.fromisoformat(iso)
        if not (SAMPLE_START <= day <= SAMPLE_END):
            raise ValueError(f"{iso} outside sample window")
        src = _source_for(day, scheduled)
        tag = day.strftime("%Y%m%d")

        # The statement: a scalar release. IMPULSE geometry, but its
        # INTERPRETATION is diffuse (wording, dot plot, vote split), which is
        # why plan section 2.1 gives it SCHEDULED_DECISION.
        start = _to_utc(day, stmt_et)
        rows.append({
            "event_id": f"FOMC_STATEMENT_{tag}",
            "start_utc": start.isoformat().replace("+00:00", "Z"),
            "end_utc": (start + timedelta(minutes=STATEMENT_DURATION_MIN))
                        .isoformat().replace("+00:00", "Z"),
            "geometry": "SCHEDULED_DECISION",
            "tier": "T1",
            "region": "US",
            "speaker": "",
            "title": "FOMC policy statement"
                     + ("" if scheduled else " (unscheduled)"),
            "affects_symbols": "*",
            "scheduled_time_certain": str(scheduled),
            "source": src,
            "verified": str(verified),
            "decision_date_local": iso,
            "note": note,
        })

        # The presser: information arrives as prose over ~60 min. This is the
        # DIFFUSE row, and the one the hypothesis is actually about.
        start = _to_utc(day, presser_et)
        rows.append({
            "event_id": f"FOMC_PRESSER_{tag}",
            "start_utc": start.isoformat().replace("+00:00", "Z"),
            "end_utc": (start + timedelta(minutes=PRESSER_DURATION_MIN))
                        .isoformat().replace("+00:00", "Z"),
            "geometry": "DIFFUSE",
            "tier": "T1",
            "region": "US",
            "speaker": "Powell",
            "title": "FOMC Chair press conference",
            "affects_symbols": "*",
            "scheduled_time_certain": str(scheduled),
            "source": src,
            # Duration is the EVENT_DEFAULT_DURATION_MIN default, not a sourced
            # end time -> verified=False regardless of the date's confidence.
            # Plan section 6.6 sensitivity-tests this at 0.5x and 2x.
            "verified": "False",
            "decision_date_local": iso,
            "note": f"{note}; duration = 60min DEFAULT, not sourced",
        })
    return rows


def main() -> None:
    rows = build_rows()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    decisions = sorted({r["decision_date_local"] for r in rows})
    pre_holdout = [d for d in decisions if d < "2026-02-01"]
    ids = [r["event_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate event_id"
    for r in rows:
        assert r["start_utc"] < r["end_utc"], f"bad interval {r['event_id']}"
        assert r["source"], f"missing source {r['event_id']}"
    by_year: dict[int, int] = {}
    for d in decisions:
        by_year[int(d[:4])] = by_year.get(int(d[:4]), 0) + 1

    print(f"wrote {OUT_PATH}")
    print(f"  rows              : {len(rows)} ({len(decisions)} decisions x 2)")
    print(f"  decisions total   : {len(decisions)}")
    print(f"  decisions in-sample (pre-holdout 2026-02-01): {len(pre_holdout)}")
    print(f"  by year           : {dict(sorted(by_year.items()))}")
    print(f"  unverified rows   : {sum(r['verified'] == 'False' for r in rows)}")
    for year, n in sorted(by_year.items()):
        assert n > 0, f"year {year} has zero T1 events -- sourcing bug"


if __name__ == "__main__":
    main()
