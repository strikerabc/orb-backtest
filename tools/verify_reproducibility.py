"""
verify_reproducibility.py — do the artifacts on disk match the committed baseline?

Review finding R5: config.HOLDOUT_PIN_START defaults to None on main, while every
file in prereg/ and outputs/ was produced at "2026-02-01". A reader running main
therefore gets different numbers than the committed evidence, and the only record
of that was a prose note in a review document.

The review proposed a CI job diffing against prereg/baseline_hashes_pinned.json.
That cannot work here: outputs/ is gitignored (2.2 GB) and regenerating it needs
the 841 MB raw cache, which comes from paid Databento pulls. There is nothing for
CI to check out. This is the local equivalent -- run it after a sweep, or before
trusting a number that came off disk.

What it checks
--------------
1. CONFIG. Whether the live config matches the config recorded as having produced
   the baseline. This is the check that matters, because a config divergence
   explains a hash divergence and makes it expected rather than alarming.
2. FILE BYTES. sha256 of summary.parquet and trade_log.parquet.
3. DATAFRAME CONTENT. sha256 over pandas' row hashes. This is the stronger check:
   parquet bytes change with a pyarrow version bump or a rewrite with different
   compression while the DATA is identical, so a file-byte mismatch alone is not
   evidence of a data regression, whereas a dataframe mismatch is.

Exit codes: 0 all verified; 1 divergence found; 2 cannot check (missing inputs).

Usage:
    python tools/verify_reproducibility.py
    python tools/verify_reproducibility.py --baseline prereg/baseline_hashes_pinned.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src import config as cfg
from src.config import OUTPUTS_DIR

DEFAULT_BASELINE = "prereg/baseline_hashes_pinned.json"

# The scheme the committed digests were produced with. Recovered by reproducing the
# recorded summary_dataframe digest -- the generator script is not in the repo, and
# a verifier that guessed at the scheme would report false divergence forever.
def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    args = ap.parse_args()

    bpath = _ROOT / args.baseline
    if not bpath.exists():
        print(f"FAIL  baseline not found: {bpath}")
        return 2
    base = json.loads(bpath.read_text(encoding="utf-8"))
    out = _ROOT / OUTPUTS_DIR

    print("=" * 78)
    print(f"REPRODUCIBILITY CHECK vs {args.baseline}")
    print("=" * 78)

    ok = True

    # ── 1. config ─────────────────────────────────────────────────────────────
    produced_under = base.get("config_that_produced_these", {})
    print("\n1. CONFIG")
    if not produced_under:
        print("   baseline records no producing config; cannot compare")
    else:
        for key, want in produced_under.items():
            if key.startswith("_"):
                continue
            live = getattr(cfg, key, "<absent>")
            match = live == want
            print(f"   {'ok  ' if match else 'DIFF'}  {key:22s} "
                  f"live={live!r:14s} baseline={want!r}")
            if not match:
                ok = False
        if not ok:
            print()
            print("   -> A config divergence means the artifacts below CANNOT match,")
            print("      and any hash mismatch is explained by it rather than by a")
            print("      data regression. Reconcile the config first.")

    # ── 2/3. artifacts ────────────────────────────────────────────────────────
    print("\n2. FILE BYTES")
    hashes = base.get("hashes", {})
    frames: dict[str, pd.DataFrame] = {}
    for name in ("summary.parquet", "trade_log.parquet"):
        want = hashes.get(name)
        p = out / name
        if want is None:
            print(f"   skip  {name:20s} not in baseline")
            continue
        if not p.exists():
            print(f"   MISS  {name:20s} not on disk at {p}")
            ok = False
            continue
        got = _file_hash(p)
        same = got == want
        print(f"   {'ok  ' if same else 'DIFF'}  {name:20s} {got[:16]}…"
              + ("" if same else f"  (baseline {want[:16]}…)"))
        if not same:
            ok = False
        try:
            frames[name] = pd.read_parquet(p)
        except Exception as e:                                # pragma: no cover
            print(f"         cannot read for content check: {e}")

    print("\n3. DATAFRAME CONTENT  (survives a parquet rewrite; the stronger check)")
    for name, key in (("summary.parquet", "summary_dataframe"),
                      ("trade_log.parquet", "trade_log_dataframe")):
        want = hashes.get(key)
        if want is None or name not in frames:
            print(f"   skip  {key}")
            continue
        got = _df_hash(frames[name])
        same = got == want
        print(f"   {'ok  ' if same else 'DIFF'}  {key:22s} {got[:16]}…"
              + ("" if same else f"  (baseline {want[:16]}…)"))
        if not same:
            ok = False

    # Shape facts, which localise a divergence far faster than a digest does.
    print("\n4. SHAPE")
    if "trade_log.parquet" in frames:
        tl = frames["trade_log.parquet"]
        rows, want_rows = len(tl), base.get("trade_log_rows")
        print(f"   {'ok  ' if rows == want_rows else 'DIFF'}  trade_log rows"
              f"        {rows:,}" + ("" if rows == want_rows else f"  (baseline {want_rows:,})"))
        if "date" in tl:
            d = pd.to_datetime(tl["date"])
            span = [str(d.min().date()), str(d.max().date())]
            want_span = base.get("trade_log_date_range")
            print(f"   {'ok  ' if span == want_span else 'DIFF'}  trade_log dates"
                  f"       {span[0]} -> {span[1]}"
                  + ("" if span == want_span else f"  (baseline {want_span})"))
            if span != want_span:
                ok = False
        if rows != want_rows:
            ok = False
    if "summary.parquet" in frames:
        n, want_n = len(frames["summary.parquet"]), base.get("summary_variants")
        print(f"   {'ok  ' if n == want_n else 'DIFF'}  summary variants"
              f"      {n:,}" + ("" if n == want_n else f"  (baseline {want_n:,})"))
        if n != want_n:
            ok = False

    print("\n" + "=" * 78)
    if ok:
        print("VERIFIED — artifacts on disk match the committed baseline.")
    else:
        print("DIVERGENCE — see above. This is not automatically a defect:")
        print("  * config differs      -> expected; the baseline records its own config")
        print("  * file bytes only     -> likely a pyarrow/pandas rewrite, data intact")
        print("  * dataframe or shape  -> the data really changed; investigate")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
