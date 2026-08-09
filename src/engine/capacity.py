"""
capacity.py -- decide how much of the machine a run may use.

The brief: when accelerated runs are declined, use a single CPU core. When they are
accepted, use "maximum available compute capacity on the device before user
experience degradation sets in."

That last clause is the whole problem, because the naive reading -- use every core
-- is what causes the degradation. Three separate effects, and only the first is
obvious:

1. **Core starvation.** With workers == cores, the OS scheduler has nothing left for
   the compositor, the shell, or the editor. The machine does not run out of
   throughput, it runs out of *responsiveness*. Reserving a fraction fixes it at a
   cost far smaller than the reservation implies, because the reserved cores were
   never the bottleneck.

2. **BLAS oversubscription.** This is the one that actually hurts and it is
   invisible. numpy/pandas link a threaded BLAS that defaults to one thread per
   core. Spawn N worker processes and you get N x cores threads fighting over N
   cores. Throughput collapses and the UI dies. The fix is to pin every worker to a
   single BLAS thread -- see thread_env(). The existing launcher set these by hand
   and it was the most important line in it.

3. **Memory pressure.** Each worker holds a full instrument frame. Once the working
   set exceeds RAM the OS swaps, and a swapping machine is unusable in a way no core
   reservation prevents. So the worker count is capped by memory as well as by cores,
   and whichever binds first wins.

stdlib only, deliberately. psutil would be tidier for the RAM query but is not
installed, and adding a dependency so a launcher can print a nicer number is a bad
trade. Where RAM cannot be determined the code degrades to a core-only decision and
says so rather than guessing.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

# Cores held back for the OS, UI, and whatever the user is doing while this runs.
# The max() of a floor and a fraction matters: on 4 cores a 25% reserve is 1, which
# is not enough to keep a desktop responsive; on 64 cores a flat 2 is far too
# little. Taking the larger of the two behaves sensibly at both ends.
RESERVE_FLOOR = 2
RESERVE_FRACTION = 0.25

# Per-worker RAM budget. A full instrument frame is ~2.6M 1m bars; with OHLCV plus
# enrichment columns in float64 that is roughly 250-400MB, and pandas peaks at
# something like 2-3x the resident size during merges and groupbys. 1.5GB is
# deliberately generous: overestimating costs a worker, underestimating costs the
# machine.
GB = 1024 ** 3
RAM_PER_WORKER_GB = 1.5
# Never let the pool consume the whole machine's memory even if the arithmetic says
# it fits -- leave headroom for the OS page cache and everything else running.
RAM_RESERVE_GB = 4.0


@dataclass
class Capacity:
    """A resolved decision about machine usage, with its reasoning attached."""
    accelerated: bool
    workers: int
    total_cores: int
    reserved_cores: int
    total_ram_gb: float | None
    ram_capped: bool
    gpu_available: bool
    gpu_name: str | None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  mode           : {'ACCELERATED' if self.accelerated else 'SINGLE-CORE CPU'}",
            f"  CPU workers    : {self.workers} of {self.total_cores} cores "
            f"({self.reserved_cores} reserved for responsiveness)",
        ]
        if self.total_ram_gb is not None:
            lines.append(f"  RAM            : {self.total_ram_gb:.1f} GB detected"
                         + ("  <- this capped the worker count" if self.ram_capped else ""))
        else:
            lines.append("  RAM            : not detected (core-only decision)")
        if self.accelerated:
            lines.append(f"  GPU bootstrap  : {self.gpu_name if self.gpu_available else 'UNAVAILABLE -- CPU fallback'}")
        lines.append("  BLAS threads   : pinned to 1 per worker (prevents oversubscription)")
        for n in self.notes:
            lines.append(f"  note           : {n}")
        return "\n".join(lines)


def total_cores() -> int:
    return os.cpu_count() or 1


def total_ram_gb() -> float | None:
    """Total physical RAM in GB, or None if it cannot be determined.

    Windows via GlobalMemoryStatusEx; POSIX via sysconf. Returns None rather than a
    guess, because a wrong RAM figure silently produces a worker count that swaps.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return None
            return stat.ullTotalPhys / GB
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / GB
    except Exception:
        return None


def gpu_status() -> tuple[bool, str | None]:
    """Whether CuPy can actually reach a CUDA device, and its name.

    Importing cupy is not sufficient -- it imports cleanly on machines with no
    driver and fails at first allocation. So this performs a real device query and
    a trivial allocation, because a launcher that promises GPU acceleration and
    then dies 20 minutes into a run is worse than one that reports CPU up front.
    """
    try:
        import cupy  # type: ignore
    except Exception:
        return False, None
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False, None
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        name = name.decode() if isinstance(name, bytes) else str(name)
        # Prove the device is usable, not merely present.
        _ = cupy.zeros(8, dtype=cupy.float32).sum()
        cupy.cuda.Stream.null.synchronize()
        return True, name
    except Exception:
        return False, None


def thread_env(workers: int) -> dict[str, str]:
    """Environment that stops N workers spawning N x cores BLAS threads.

    Set for every mode, including single-core. Effect 2 in the module docstring:
    without this the pool oversubscribes and both throughput and responsiveness
    collapse, with no error message to explain it.
    """
    return {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def resolve(accelerated: bool, override_workers: int | None = None) -> Capacity:
    """Resolve a capacity decision.

    accelerated=False is a hard single core -- the point of declining is a run that
    leaves the machine usable, so it is not negotiated against core count.
    """
    cores = total_cores()
    ram = total_ram_gb()
    notes: list[str] = []

    if not accelerated:
        gpu_ok, gpu_name = False, None
        return Capacity(
            accelerated=False, workers=1, total_cores=cores, reserved_cores=cores - 1,
            total_ram_gb=ram, ram_capped=False, gpu_available=gpu_ok,
            gpu_name=gpu_name,
            notes=["single-core by request; slowest but leaves the machine fully usable"],
        )

    reserve = max(RESERVE_FLOOR, int(round(cores * RESERVE_FRACTION)))
    by_cores = max(1, cores - reserve)

    workers = by_cores
    ram_capped = False
    if ram is not None:
        usable = max(0.0, ram - RAM_RESERVE_GB)
        by_ram = max(1, int(usable // RAM_PER_WORKER_GB))
        if by_ram < workers:
            workers, ram_capped = by_ram, True
            notes.append(
                f"RAM-bound: {ram:.0f}GB total, {RAM_RESERVE_GB:.0f}GB reserved, "
                f"~{RAM_PER_WORKER_GB}GB per worker -> {by_ram} workers "
                f"(cores alone would allow {by_cores})")

    if override_workers is not None:
        requested = max(1, override_workers)
        if requested > by_cores:
            notes.append(
                f"override {requested} exceeds the {by_cores} safe workers; "
                "expect UI stutter and possible thrashing")
        workers = requested
        notes.append(f"worker count overridden to {requested}")

    gpu_ok, gpu_name = gpu_status()
    if not gpu_ok:
        notes.append("no usable CUDA device; bootstrap runs on CPU "
                     "(correct results, materially slower)")

    return Capacity(
        accelerated=True, workers=workers, total_cores=cores,
        reserved_cores=cores - workers, total_ram_gb=ram, ram_capped=ram_capped,
        gpu_available=gpu_ok, gpu_name=gpu_name, notes=notes,
    )
