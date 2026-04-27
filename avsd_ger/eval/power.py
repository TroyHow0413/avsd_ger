"""Power / energy monitor (spec section 5.10).

Design matches the spec:
    * Sampling interval: 500 ms (default; configurable).
    * GPU power via pynvml (per-device mW -> W).
    * CPU power via psutil utilisation * SDP_WATTS, since psutil doesn't
      expose package power on all platforms. If Linux RAPL is readable,
      we use that instead (more accurate).
    * Idle correction: a baseline window is sampled before the measured
      region; the baseline mean is subtracted from each sample so that
      reported energy is *attributable to the model*, not the host.

Usage:
    mon = PowerMonitor()
    mon.calibrate_idle(duration_s=2.0)     # sample baseline
    with mon.measure("stage2_epoch"):
        ... run your workload ...
    report = mon.last_report()
    print(report.energy_wh, report.avg_power_w)

If pynvml / psutil aren't available, the monitor degrades gracefully and
returns zero-filled samples with a warning flag -- so eval pipelines never
crash in CPU-only or container environments.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

log = logging.getLogger(__name__)


# --------------------------------------------------------------- data classes

@dataclass
class PowerSample:
    """One 500 ms (default) sample."""

    t: float                  # wall-clock seconds since measure() start
    gpu_w: float              # summed GPU package power across visible devices (W)
    cpu_w: float              # CPU package power (W), RAPL if available else estimate
    total_w: float            # gpu_w + cpu_w


@dataclass
class PowerReport:
    """Aggregated over one measurement window."""

    label: str
    duration_s: float
    n_samples: int
    avg_power_w: float
    peak_power_w: float
    energy_j: float            # integral of (total_w - idle_baseline_w) * dt
    energy_wh: float           # energy_j / 3600
    idle_baseline_w: float     # what we subtracted
    degraded: bool = False     # True if pynvml/psutil/RAPL weren't available
    samples: list[PowerSample] = field(default_factory=list)


# --------------------------------------------------------------- backends

def _nvml_init() -> tuple[object | None, list[object]]:
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        return pynvml, handles
    except Exception as exc:
        log.warning("pynvml unavailable (%s); GPU power will read 0 W", exc)
        return None, []


def _read_gpu_w(pynvml_mod, handles) -> float:
    if pynvml_mod is None or not handles:
        return 0.0
    total_mw = 0.0
    for h in handles:
        try:
            total_mw += float(pynvml_mod.nvmlDeviceGetPowerUsage(h))
        except Exception:
            pass
    return total_mw / 1000.0


def _rapl_reader_linux():
    """Return a callable() -> W using Linux RAPL, or None if unavailable."""
    import os
    base = "/sys/class/powercap/intel-rapl"
    if not os.path.isdir(base):
        return None
    # Aggregate every "package-*" node we can read
    pkgs = []
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "energy_uj")):
            pkgs.append(os.path.join(p, "energy_uj"))
    if not pkgs:
        return None

    state = {"last_uj": None, "last_t": None}

    def _read() -> float:
        try:
            total_uj = 0
            for p in pkgs:
                with open(p) as f:
                    total_uj += int(f.read().strip())
            now = time.monotonic()
            if state["last_uj"] is None:
                state["last_uj"] = total_uj
                state["last_t"] = now
                return 0.0
            du = total_uj - state["last_uj"]
            dt = max(1e-6, now - state["last_t"])
            state["last_uj"] = total_uj
            state["last_t"] = now
            # uJ / s = uW ; /1e6 -> W
            return max(0.0, du / dt / 1e6)
        except Exception:
            return 0.0

    return _read


def _psutil_cpu_reader(sdp_watts: float):
    """Approximate CPU power = SDP * utilization. Falls back to 0 if psutil missing."""
    try:
        import psutil
    except Exception:
        log.warning("psutil unavailable; CPU power estimate will be 0 W")
        return None

    # Warm up psutil internal deltas so the first call doesn't return 0/100.
    psutil.cpu_percent(interval=None)

    def _read() -> float:
        try:
            util = psutil.cpu_percent(interval=None) / 100.0
            return max(0.0, sdp_watts * util)
        except Exception:
            return 0.0
    return _read


# --------------------------------------------------------------- monitor

class PowerMonitor:
    """Background sampler. One thread per measurement window.

    Thread-safety: a single monitor should be used from one thread at a time.
    Create multiple instances for concurrent regions.
    """

    def __init__(
        self,
        sample_interval_s: float = 0.5,
        cpu_sdp_watts: float = 65.0,
    ):
        self.sample_interval_s = float(sample_interval_s)
        self.cpu_sdp_watts = float(cpu_sdp_watts)

        self._nvml, self._handles = _nvml_init()
        self._cpu_reader = _rapl_reader_linux() or _psutil_cpu_reader(self.cpu_sdp_watts)
        self._degraded = (self._nvml is None) and (self._cpu_reader is None)
        self._idle_baseline_w: float = 0.0
        self._last_report: PowerReport | None = None

        # Active-window state
        self._running = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[PowerSample] = []
        self._t0: float = 0.0
        self._label: str = ""

    # ------------------------------------------------------- calibration
    def calibrate_idle(self, duration_s: float = 2.0) -> float:
        """Sample for `duration_s` with the workload idle. Store mean as baseline."""
        samples = self._collect(duration_s, label="__idle__", store=False)
        if samples:
            mean_w = sum(s.total_w for s in samples) / len(samples)
        else:
            mean_w = 0.0
        self._idle_baseline_w = float(mean_w)
        return self._idle_baseline_w

    # ------------------------------------------------------- measurement
    @contextlib.contextmanager
    def measure(self, label: str) -> Iterator[None]:
        self._start(label)
        try:
            yield
        finally:
            self._stop_and_finalize()

    def last_report(self) -> PowerReport | None:
        return self._last_report

    # ------------------------------------------------------- internals
    def _read_sample(self, t_rel: float) -> PowerSample:
        gpu_w = _read_gpu_w(self._nvml, self._handles)
        cpu_w = self._cpu_reader() if self._cpu_reader else 0.0
        return PowerSample(t=t_rel, gpu_w=gpu_w, cpu_w=cpu_w, total_w=gpu_w + cpu_w)

    def _collect(self, duration_s: float, label: str, store: bool) -> list[PowerSample]:
        out: list[PowerSample] = []
        t0 = time.monotonic()
        # Prime one read so deltas (RAPL, psutil) stabilize.
        self._read_sample(0.0)
        while True:
            now = time.monotonic()
            t_rel = now - t0
            if t_rel >= duration_s:
                break
            out.append(self._read_sample(t_rel))
            time.sleep(self.sample_interval_s)
        if store:
            self._samples = out
            self._t0 = t0
            self._label = label
        return out

    def _start(self, label: str) -> None:
        if self._running:
            raise RuntimeError("PowerMonitor already running")
        self._samples = []
        self._label = label
        self._stop.clear()
        self._t0 = time.monotonic()
        self._running = True

        def _loop():
            # Prime once.
            self._read_sample(0.0)
            while not self._stop.is_set():
                t_rel = time.monotonic() - self._t0
                self._samples.append(self._read_sample(t_rel))
                # Event.wait returns True if set; using it lets stop() interrupt sleep.
                if self._stop.wait(self.sample_interval_s):
                    break

        self._thread = threading.Thread(target=_loop, daemon=True, name=f"pwr-{label}")
        self._thread.start()

    def _stop_and_finalize(self) -> None:
        if not self._running:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        duration = time.monotonic() - self._t0
        samples = list(self._samples)
        self._running = False

        if not samples:
            self._last_report = PowerReport(
                label=self._label, duration_s=duration, n_samples=0,
                avg_power_w=0.0, peak_power_w=0.0, energy_j=0.0, energy_wh=0.0,
                idle_baseline_w=self._idle_baseline_w, degraded=self._degraded,
                samples=[],
            )
            return

        # Numerical integration via trapezoidal rule on (total_w - idle_baseline_w)
        energy_j = 0.0
        for i in range(1, len(samples)):
            dt = max(0.0, samples[i].t - samples[i - 1].t)
            p0 = max(0.0, samples[i - 1].total_w - self._idle_baseline_w)
            p1 = max(0.0, samples[i].total_w - self._idle_baseline_w)
            energy_j += 0.5 * (p0 + p1) * dt

        avg_w = sum(max(0.0, s.total_w - self._idle_baseline_w) for s in samples) / len(samples)
        peak_w = max((s.total_w - self._idle_baseline_w) for s in samples)
        self._last_report = PowerReport(
            label=self._label,
            duration_s=duration,
            n_samples=len(samples),
            avg_power_w=avg_w,
            peak_power_w=max(0.0, peak_w),
            energy_j=energy_j,
            energy_wh=energy_j / 3600.0,
            idle_baseline_w=self._idle_baseline_w,
            degraded=self._degraded,
            samples=samples,
        )
