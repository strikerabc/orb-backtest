"""
run.py -- interactive entry point for the ORB backtest.

Asks two questions, states what it decided and why, then runs. The point of the
prompt is that "use the whole machine" and "leave the machine usable" are different
jobs, and the caller is the only one who knows which they want right now.

Non-interactive use is a first-class path, not an afterthought -- CI and background
runs must not block on a prompt:

    python run.py --accelerated          # skip prompts, maximum safe capacity
    python run.py --single-core          # skip prompts, one core
    python run.py --accelerated --fresh  # discard checkpoints and restart
    python run.py --self-test            # CPU/CUDA parity checks only
    python run.py --workers 8            # explicit override, warns if unsafe

Anything not answerable from flags falls back to the prompt; if stdin is not a TTY
the safe default is taken and the choice is logged, so a piped or scheduled run
never hangs waiting for input that is not coming.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.engine.capacity import Capacity, resolve, thread_env  # noqa: E402

SWEEP = ROOT / "src" / "engine" / "sweep.py"
RUNS = ROOT / "runs"
BAR = "=" * 78


def screen(title: str) -> None:
    print(f"\n{BAR}\n  {title}\n{BAR}")


def ask(question: str, options: list[tuple[str, str, str]], default_key: str) -> str:
    """Present a numbered choice. Returns the chosen key.

    Falls back to `default_key` without blocking when stdin is not interactive, so
    the same code path serves humans and schedulers.
    """
    print(f"\n{question}\n")
    for i, (_, label, detail) in enumerate(options, 1):
        print(f"  [{i}] {label}")
        if detail:
            print(f"      {detail}")
    default_idx = next(i for i, (k, _, _) in enumerate(options, 1) if k == default_key)

    if not sys.stdin.isatty():
        chosen = options[default_idx - 1]
        print(f"\n  non-interactive stdin -> defaulting to [{default_idx}] {chosen[1]}")
        return chosen[0]

    while True:
        try:
            raw = input(f"\n  choice [1-{len(options)}, Enter = {default_idx}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  cancelled")
            raise SystemExit(130)
        if not raw:
            return options[default_idx - 1][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print(f"  not a valid choice: {raw!r}")


def confirm(question: str, default: bool = False) -> bool:
    if not sys.stdin.isatty():
        print(f"  non-interactive stdin -> {question} = {default}")
        return default
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"  {question} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  cancelled")
        raise SystemExit(130)
    return default if not raw else raw.startswith("y")


def choose_mode(args: argparse.Namespace) -> bool:
    if args.accelerated:
        return True
    if args.single_core:
        return False
    screen("COMPUTE MODE")
    print("\n  Accelerated uses several CPU worker processes and, when a CUDA device")
    print("  is present, the GPU for bootstrap calibration. Results are identical")
    print("  either way -- the sweep self-tests CPU/CUDA parity before running.")
    key = ask(
        "  Enable accelerated runs?",
        [
            ("accel", "Yes -- accelerated",
             "Maximum safe capacity. Cores are held back so the machine stays usable."),
            ("single", "No -- single-core CPU",
             "One worker. Substantially slower; leaves the machine untouched."),
        ],
        default_key="accel",
    )
    return key == "accel"


def resume_or_fresh(args: argparse.Namespace) -> bool:
    """True = fresh. Resume is the default because it is the cheap, safe option."""
    if args.fresh:
        return True
    ckpt = RUNS / "gpu_cpu_sweep"
    status = RUNS / "gpu_cpu_sweep.status.json"
    if not ckpt.exists() and not status.exists():
        return True  # nothing to resume from
    screen("CHECKPOINTS FOUND")
    stage = "unknown"
    if status.exists():
        try:
            import json
            stage = json.loads(status.read_text()).get("stage", "unknown")
        except Exception:
            pass
    print(f"\n  A previous run left checkpoints (last stage: {stage}).")
    print("\n  Resume only makes sense if the DATA and CONFIG are unchanged. Window")
    print("  placement or a data extension invalidates them, and a resumed run would")
    print("  silently mix old and new results.")
    key = ask(
        "  Resume or start fresh?",
        [
            ("resume", "Resume from checkpoints", "Much faster. Requires unchanged inputs."),
            ("fresh", "Start fresh", "Discards checkpoints and recomputes every stage."),
        ],
        default_key="resume",
    )
    return key == "fresh"


def build_command(cap: Capacity, fresh: bool, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-u", str(SWEEP), "--workers", str(cap.workers)]
    if fresh:
        cmd.append("--fresh")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.n_boot is not None:
        cmd += ["--n-boot", str(args.n_boot)]
    if args.symbols:
        cmd += ["--symbols", *args.symbols]
    return cmd


def run_events(which: str, env: dict[str, str]) -> int:
    """Run the event-regime analysis package.

    Separate path from the sweep because these are read-only: they consume an
    existing outputs/trade_log.parquet and write their own JSON into prereg/. They
    cannot change a sweep result, so they take no capacity decision and need no
    accelerate/single-core prompt -- the permutation work is numpy-vectorised and
    single-process.
    """
    screen("EVENT-REGIME ANALYSIS")
    trade_log = ROOT / "outputs" / "trade_log.parquet"
    if not trade_log.exists():
        print(f"\n  Missing {trade_log.relative_to(ROOT)}")
        print("  These analyses read a completed sweep. Run one first.")
        return 1

    from src import config
    print(f"\n  read-only: consumes outputs/trade_log.parquet, writes prereg/*.json")
    print(f"  config.EVENTS_ENABLED = {config.EVENTS_ENABLED}"
          "   (gates pipeline influence only; analysis runs regardless)")

    # Single BLAS thread here too. The permutation kernels are already vectorised,
    # so a threaded BLAS adds contention without throughput.
    env.update(thread_env(1))

    steps = {
        "calendar":   ("rebuild FOMC calendar",     "src.events.calendar_build"),
        "probe":      ("FOMC anticipation gate",    "src.events.probe"),
        "validate":   ("machinery validation",      "src.events.validation"),
        "case-study": ("2026-08-07 path case study", "src.events.case_study"),
    }
    order = ["calendar", "probe", "validate", "case-study"] if which == "all" else [which]

    for key in order:
        label, module = steps[key]
        screen(f"EVENTS -- {label}")
        print()
        rc = subprocess.call([sys.executable, "-u", "-m", module], cwd=ROOT, env=env)
        if rc != 0:
            print(f"\n  {label} failed (exit {rc})")
            return rc
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Interactive entry point for the ORB backtest sweep.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--accelerated", action="store_true",
                      help="skip the prompt; use maximum safe capacity")
    mode.add_argument("--single-core", action="store_true",
                      help="skip the prompt; use one CPU core")
    p.add_argument("--workers", type=int, default=None,
                   help="override worker count (warns if above the safe limit)")
    p.add_argument("--fresh", action="store_true",
                   help="discard checkpoints and recompute every stage")
    p.add_argument("--self-test", action="store_true",
                   help="run CPU/CUDA parity checks and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="run the pipeline without overwriting outputs/")
    p.add_argument("--n-boot", type=int, default=None, help="bootstrap draws")
    p.add_argument("--symbols", nargs="+", default=None, help="restrict instruments")
    p.add_argument("--events", choices=["probe", "validate", "calendar",
                                        "case-study", "all"], default=None,
                   help="run event-regime analysis (src/events) instead of the "
                        "sweep; read-only, reads outputs/trade_log.parquet")
    args = p.parse_args()

    env = dict(os.environ)

    if args.events:
        return run_events(args.events, env)

    if args.self_test:
        screen("SELF-TEST -- CPU/CUDA PARITY")
        env.update(thread_env(1))
        return subprocess.call(
            [sys.executable, "-u", str(SWEEP), "--self-test"], cwd=ROOT, env=env)

    accelerated = choose_mode(args)
    cap = resolve(accelerated, override_workers=args.workers)

    screen("RESOLVED CAPACITY")
    print()
    print(cap.summary())

    if cap.accelerated and not cap.gpu_available:
        print("\n  Accelerated was requested but no usable CUDA device was found.")
        print("  The run will still use multiple CPU workers and produce identical")
        print("  results -- only the bootstrap stage loses its speedup.")

    fresh = resume_or_fresh(args)
    cmd = build_command(cap, fresh, args)

    screen("READY")
    print(f"\n  {'fresh run (checkpoints discarded)' if fresh else 'resuming from checkpoints'}")
    print(f"  command: {' '.join(cmd[1:])}")
    print(f"  progress: runs/gpu_cpu_sweep.stdout.log")
    print(f"  status:   runs/gpu_cpu_sweep.status.json")
    if sys.stdin.isatty() and not confirm("\n  Start now?", default=True):
        print("  aborted")
        return 130

    # Pin BLAS to one thread per worker. Without this every worker spawns its own
    # thread pool, the machine oversubscribes by workers x cores, and both
    # throughput and UI responsiveness collapse with no error to explain it.
    env.update(thread_env(cap.workers))

    RUNS.mkdir(exist_ok=True)
    screen("RUNNING")
    print()
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
