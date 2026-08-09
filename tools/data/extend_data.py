"""
extend_data.py -- top up the 1m/1d caches through the Databento bound.

WHY THIS EXISTS
    The trade log stopped at 2026-04-30, but the caches already ran to
    2026-07-31/08-02. The April cutoff was a STALE PIPELINE RUN, not a data limit.
    So the only thing money needs to buy is the 2026-08-01 -> 2026-08-08 gap
    (~$0.24 for all ten instruments), which is where the user's 2026-08-07 Barkin
    session lives. Everything from May through July was already on disk and is
    free to bring into the trade log by re-running main.py.

    ensure_data() short-circuits on cache existence -- it will not top up a gap --
    hence this explicit extend path. It reuses the repo's own _merge_and_cache so
    dedupe, source-priority and ATR recomputation stay identical to the normal
    build. No reimplementation of enrichment logic.

SAFETY
    - The API key is parsed from databento_key.txt with a strict `db-[A-Za-z0-9]+`
      match and NEVER printed. Every error path is scrubbed through _clean().
      A prior run leaked the key into a terminal transcript because a Databento
      400 echoed the malformed credential back verbatim; _clean() exists so a
      vendor error message can never do that again.
    - Caches are copied to data/_backup_pre_august/ before any write. The parquet
      files cost real credits to rebuild.
    - Cost is estimated and printed per symbol BEFORE each download, and the
      script aborts if the running total exceeds MAX_SPEND_USD.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import DATASET, INSTRUMENTS, SCHEMA_1D, SCHEMA_1M, STYPE  # noqa: E402
from src.data_layer import _cache_path, _download_databento, _merge_and_cache  # noqa: E402

GAP_START = "2026-08-01"
GAP_END = "2026-08-08"          # Databento availability bound; `end` is exclusive
MAX_SPEND_USD = 1.50            # hard ceiling; estimate is ~$0.24
BACKUP_DIR = ROOT / "data" / "_backup_pre_august"

# CLI override: extend_data_to_august.py [START] [SYM,SYM,...]
#
# ES and NQ need a wider window than the other eight. They carry
# has_local_data=True, so _cache_path appends a "_db" vendor tag, and the legacy
# fallback is gated on `not vendor` -- which left ES_1m.parquet / NQ_1m.parquet
# unreachable and made the preflight plan a full 7-year re-download (~$18) rather
# than a gap fill. The files were aliased to the resolvable *_c_db_1m.parquet names
# (verified pure-Databento: they carry _contract and no _source, which is exactly
# what _download_databento emits), so they only need 2026-05-01 -> 2026-08-08.
if len(sys.argv) > 1:
    GAP_START = sys.argv[1]
SYMBOLS = (sys.argv[2].split(",") if len(sys.argv) > 2 else list(INSTRUMENTS))

_KEY_RE = re.compile(r"db-[A-Za-z0-9]+")


def load_key() -> str:
    raw = (ROOT / "databento_key.txt").read_text(encoding="utf-8")
    m = _KEY_RE.search(raw)
    if not m:
        raise SystemExit("no db-... key found in databento_key.txt")
    return m.group(0)


KEY = load_key()


def _clean(obj: object) -> str:
    """Scrub the key out of anything before it reaches stdout."""
    return _KEY_RE.sub("***REDACTED***", str(obj))


def log(msg: str) -> None:
    print(_clean(msg), flush=True)


def backup_caches() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for sym in SYMBOLS:
        for schema in (SCHEMA_1M, SCHEMA_1D):
            src = _cache_path(sym, schema)
            if src.exists():
                dst = BACKUP_DIR / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    n += 1
    log(f"backed up {n} cache files -> {BACKUP_DIR}")


def main() -> None:
    import databento as db

    client = db.Historical(key=KEY)
    backup_caches()

    log(f"\n=== cost estimate: {GAP_START} -> {GAP_END} ===")
    total = 0.0
    for sym in SYMBOLS:
        for schema in (SCHEMA_1M, SCHEMA_1D):
            try:
                c = float(client.metadata.get_cost(
                    dataset=DATASET, symbols=[INSTRUMENTS[sym]["continuous_symbol"]],
                    schema=schema, start=GAP_START, end=GAP_END, stype_in=STYPE))
                total += c
            except Exception as e:
                log(f"  {sym} {schema}: cost ERR {type(e).__name__}: {_clean(e)[:120]}")
    log(f"estimated total: ${total:.4f}  (ceiling ${MAX_SPEND_USD:.2f})")
    if total > MAX_SPEND_USD:
        raise SystemExit(f"ABORT: estimate ${total:.4f} exceeds ceiling")

    log(f"\n=== extending 1m caches ===")
    for sym in SYMBOLS:
        cache = _cache_path(sym, SCHEMA_1M)
        if not cache.exists():
            log(f"  {sym:5s} SKIP: no existing cache")
            continue
        existing = pd.read_parquet(cache)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        before_max = existing["timestamp"].max()
        if "_source" not in existing.columns:
            existing["_source"] = "unknown"
        try:
            new = _download_databento(sym, GAP_START, GAP_END, KEY)
        except Exception as e:
            log(f"  {sym:5s} DOWNLOAD ERR {type(e).__name__}: {_clean(e)[:150]}")
            continue
        if new is None or new.empty:
            log(f"  {sym:5s} no new bars returned")
            continue
        # existing goes in as `existing` (source_priority 1) and the fresh pull as
        # `downloaded` (priority 2), so on any overlapping timestamp the new vendor
        # bar wins -- same precedence the normal build uses.
        merged = _merge_and_cache(sym, existing, new, KEY)
        log(f"  {sym:5s} {before_max} -> {merged['timestamp'].max()} "
            f"(+{len(merged) - len(existing):,} bars)")

    log(f"\n=== extending 1d caches ===")
    for sym in SYMBOLS:
        cache = _cache_path(sym, SCHEMA_1D)
        if not cache.exists():
            log(f"  {sym:5s} SKIP: no existing 1d cache")
            continue
        existing = pd.read_parquet(cache)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        before_max = existing["timestamp"].max()
        try:
            data = client.timeseries.get_range(
                dataset=DATASET, symbols=[INSTRUMENTS[sym]["continuous_symbol"]],
                schema=SCHEMA_1D, start=GAP_START, end=GAP_END, stype_in=STYPE)
            new = data.to_df().reset_index().rename(columns={"ts_event": "timestamp"})
        except Exception as e:
            log(f"  {sym:5s} 1d ERR {type(e).__name__}: {_clean(e)[:150]}")
            continue
        if new.empty:
            log(f"  {sym:5s} no new daily bars")
            continue
        new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True)
        out = (pd.concat([existing, new], ignore_index=True)
                 .sort_values("timestamp", kind="mergesort")
                 .drop_duplicates("timestamp", keep="last")
                 .reset_index(drop=True))
        out.to_parquet(cache, index=False)
        log(f"  {sym:5s} {before_max} -> {out['timestamp'].max()} "
            f"(+{len(out) - len(existing)} days)")

    log("\n=== final coverage ===")
    for sym in SYMBOLS:
        c1 = _cache_path(sym, SCHEMA_1M)
        if c1.exists():
            t = pd.read_parquet(c1, columns=["timestamp"])
            log(f"  {sym:5s} 1m -> {pd.to_datetime(t['timestamp'], utc=True).max()}")
    log("\ndone. Re-run main.py to rebuild the trade log across the extended span.")


if __name__ == "__main__":
    main()
