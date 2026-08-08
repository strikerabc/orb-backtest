"""
case_study_barkin_20260807.py -- the section 6.3 mechanism metrics on 1m bars.

WHY THIS EXISTS
    Two things forced it.

    1. The I3 retraction (EVENT_REGIME_REVIEW.md): tap_in_bar_idx is exactly 0.0
       for CC/II and exactly 1.0 for R-CC/R-II/TI -- a deterministic function of
       entry mode, not a path measurement. Section 6.3 called reversal rate "free"
       on the reasoning that TI detection IS a reversal detector. It is not. So the
       mechanism claim needs genuine path-shape metrics from 1m bars, and those did
       not exist anywhere in the repo.

    2. The user's 2026-08-07 session: NFP at 08:30 ET (pre-open, IMPULSE), then a
       Barkin speech + Q&A during the NY window (in-path, DIFFUSE). Reported as
       initial move tradeable, everything after untradeable, including a short
       re-entry, with each unexpected shift traceable to a specific Q&A answer.

    That is DIFFUSE_IN_PATH -- the channel the FOMC gate could not test, because
    FOMC 14:00/14:30 ET falls outside 09:30-12:00 ET.

WHAT IT DOES
    Computes path efficiency, boundary crossings and vol-profile ratio for every
    6E/NY session in the sample, then percentile-ranks 2026-08-07 against that
    distribution. Also walks the target session bar by bar to locate the breakout
    and every subsequent boundary flip.

WHAT IT IS NOT
    n=1. 2026-08-07 sits INSIDE the paper period (2026-07-25 -> 2026-08-07), so it
    is observation-origin data and can never serve as confirmation -- see the
    prereg amendment. Its only legitimate uses are (a) validating that the metrics
    behave as the taxonomy predicts, and (b) sizing the effect for a real test.
    A percentile is not a p-value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import INSTRUMENTS, SESSIONS  # noqa: E402
from src.data_layer import _cache_path  # noqa: E402
from src.config import SCHEMA_1M  # noqa: E402

TARGET = "2026-08-07"
SYM, SESS = "6E", "NY"
RANGE_MIN = 5          # the user's setup: 5m opening range
OUT = ROOT / "prereg" / "case_study_barkin_20260807.json"


def session_frame() -> pd.DataFrame:
    df = pd.read_parquet(_cache_path(SYM, SCHEMA_1M),
                         columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    tz = SESSIONS[SESS]["tz"]
    loc = df["timestamp"].dt.tz_convert(tz)
    df["_date"] = loc.dt.date
    df["_min"] = loc.dt.hour * 60 + loc.dt.minute
    df["_loc"] = loc
    oh, om = SESSIONS[SESS]["open"]
    eh, em = SESSIONS[SESS]["exit"]
    return df[(df["_min"] >= oh * 60 + om) & (df["_min"] < eh * 60 + em)].copy()


def metrics(g: pd.DataFrame, open_min: int) -> dict | None:
    """Section 6.3 metrics for one session. Post-range window only for path stats."""
    rng = g[g["_min"] < open_min + RANGE_MIN]
    post = g[g["_min"] >= open_min + RANGE_MIN]
    if len(rng) < RANGE_MIN or len(post) < 30:
        return None
    rh, rl = float(rng["high"].max()), float(rng["low"].min())
    c = post["close"].to_numpy(dtype=float)
    rets = np.diff(c)
    total = float(np.abs(rets).sum())

    # Path efficiency: net displacement / total distance travelled. Low = chop.
    eff = float(abs(c[-1] - c[0]) / total) if total > 0 else np.nan

    # Boundary crossings: how often price re-crosses the opening-range edges after
    # the range closes. High = whipsaw around the reference level.
    def crossings(level: float) -> int:
        side = np.sign(c - level)
        side = side[side != 0]
        return int((np.diff(side) != 0).sum()) if len(side) > 1 else 0

    # Vol-profile ratio: mean |1m return| final third / first third of the FULL
    # session. IMPULSE decays (ratio << 1); DIFFUSE stays flat or rises (>= 1).
    cf = g["close"].to_numpy(dtype=float)
    r_all = np.abs(np.diff(cf))
    third = max(1, len(r_all) // 3)
    first, last = r_all[:third].mean(), r_all[-third:].mean()
    return {
        "range_high": rh, "range_low": rl,
        "range_width_ticks": round((rh - rl) / INSTRUMENTS[SYM]["tick_size"]),
        "path_efficiency": eff,
        "crossings_rh": crossings(rh),
        "crossings_rl": crossings(rl),
        "crossings_total": crossings(rh) + crossings(rl),
        "vol_profile_ratio": float(last / first) if first > 0 else np.nan,
        "session_range_ticks": round((float(g["high"].max()) - float(g["low"].min()))
                                     / INSTRUMENTS[SYM]["tick_size"]),
        "n_bars": len(g),
    }


def walk(g: pd.DataFrame, open_min: int, rh: float, rl: float) -> list[str]:
    """Locate the first breakout and every subsequent boundary flip, with times."""
    # itertuples mangles leading-underscore names into positional _1/_2, so rename
    # before iterating rather than indexing by position.
    post = (g[g["_min"] >= open_min + RANGE_MIN]
            .rename(columns={"_loc": "loc_ts"}))
    events, state = [], "inside"
    for row in post.itertuples():
        t = row.loc_ts.strftime("%H:%M")
        if row.high > rh and state != "above":
            events.append(f"{t}  break ABOVE {rh:.5f}  (h={row.high:.5f})")
            state = "above"
        elif row.low < rl and state != "below":
            events.append(f"{t}  break BELOW {rl:.5f}  (l={row.low:.5f})")
            state = "below"
    return events


def main() -> None:
    df = session_frame()
    oh, om = SESSIONS[SESS]["open"]
    open_min = oh * 60 + om

    rows = {}
    for date, g in df.groupby("_date", sort=True):
        m = metrics(g.sort_values("timestamp"), open_min)
        if m:
            rows[str(date)] = m
    allm = pd.DataFrame(rows).T
    print(f"6E/NY sessions with metrics: {len(allm)}")

    if TARGET not in rows:
        print(f"!! {TARGET} not present")
        return
    tgt = rows[TARGET]

    print(f"\n=== {TARGET} vs all {len(allm)} 6E/NY sessions ===")
    pct = {}
    for k in ("path_efficiency", "crossings_total", "vol_profile_ratio",
              "session_range_ticks", "range_width_ticks"):
        col = pd.to_numeric(allm[k], errors="coerce").dropna()
        p = float((col < tgt[k]).mean() * 100)
        pct[k] = round(p, 1)
        print(f"  {k:22s} {float(tgt[k]):>10.4f}   median {col.median():>9.4f}   "
              f"pctile {p:>5.1f}")

    print(f"\n=== {TARGET} path walk (5m range {tgt['range_low']:.5f}-{tgt['range_high']:.5f}) ===")
    ev = walk(df[df["_date"] == pd.Timestamp(TARGET).date()].sort_values("timestamp"),
              open_min, tgt["range_high"], tgt["range_low"])
    for e in ev:
        print("  " + e)
    print(f"  total direction flips after range close: {len(ev)}")

    # Compare against the DIFFUSE prediction: low efficiency, high crossings,
    # flat-or-rising vol profile.
    med = {k: float(pd.to_numeric(allm[k], errors="coerce").median())
           for k in ("path_efficiency", "crossings_total", "vol_profile_ratio")}
    print("\n=== taxonomy check (section 2.1 DIFFUSE prediction) ===")
    print(f"  path_efficiency   {tgt['path_efficiency']:.4f} vs median {med['path_efficiency']:.4f}"
          f"  -> {'LOWER (as predicted)' if tgt['path_efficiency'] < med['path_efficiency'] else 'NOT lower'}")
    print(f"  crossings_total   {tgt['crossings_total']:.0f} vs median {med['crossings_total']:.0f}"
          f"  -> {'HIGHER (as predicted)' if tgt['crossings_total'] > med['crossings_total'] else 'NOT higher'}")
    print(f"  vol_profile_ratio {tgt['vol_profile_ratio']:.4f} vs median {med['vol_profile_ratio']:.4f}"
          f"  -> {'FLAT/RISING (DIFFUSE)' if tgt['vol_profile_ratio'] >= 1.0 else 'FALLING (IMPULSE-like)'}")

    import json
    OUT.write_text(json.dumps({
        "_what_this_is": "n=1 case study on observation-origin data (2026-08-07 is "
                         "inside the paper period 2026-07-25 -> 2026-08-07). Cannot "
                         "serve as confirmation. Validates that the section 6.3 path "
                         "metrics behave as the taxonomy predicts, and sizes the "
                         "effect for a real test. A percentile is not a p-value.",
        "target": TARGET, "instrument": SYM, "session": SESS,
        "range_minutes": RANGE_MIN,
        "target_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                           for k, v in tgt.items()},
        "percentile_vs_all_sessions": pct,
        "n_sessions_compared": int(len(allm)),
        "medians": med,
        "direction_flips_after_range": len(ev),
        "path_walk": ev,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
