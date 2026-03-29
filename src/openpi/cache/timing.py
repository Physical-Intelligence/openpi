"""Timing system for the Pi0/Pi0.5 inference cache pipeline.

Overview
--------
This module provides ``SystemTimer``, a pluggable, hardware-aware latency
measurement system.  It is designed to time all components in the inference +
cache pipeline without interfering with inference correctness or performance.

Architecture
------------
Measurement is *probe-based*: each pipeline component (stage1_vision,
cp1_gate, …) registers a named *probe* once at startup, then calls
``timer.measure("probe_name")`` in the hot path.  Two backends handle the
actual timing:

* ``CudaEventBackend`` — for GPU-resident operations (inference stages,
  GPU-side vector search).  Uses ``torch.cuda.Event`` to record GPU-side
  timestamps; ``end_event.synchronize()`` blocks the CPU only until *that*
  specific event is processed, without flushing the entire CUDA pipeline.
  Falls back to ``PerfCounterBackend`` when CUDA is unavailable.

* ``PerfCounterBackend`` — for CPU-only operations (gate decisions, FAISS
  CPU search, write-back threads, etc.).  Uses ``time.perf_counter_ns`` for
  sub-microsecond resolution.

Task lifecycle
--------------
A "task" corresponds to one client WebSocket connection.  Call
``on_task_begin()`` when the connection opens and ``on_task_end()`` when it
closes.  ``on_task_end()`` prints a per-probe summary to the terminal and
(optionally) flushes all records for that task to a CSV file in one batch.

All ``TimingRecord`` objects are held in an in-memory ``deque`` (ring buffer)
during the task.  The CSV write at task end is the *only* disk IO, avoiding
per-record write overhead.

Extensibility
-------------
* **New probes**: call ``register_probe(name, backend="cpu"/"cuda")`` from
  the new component's ``__init__``.  No changes to ``SystemTimer`` needed.
* **Custom CUDA streams**: pass ``stream=your_stream`` to ``register_probe``
  so cache-stream operations are timed on the correct stream (Step 4+).
* **Resource monitors**: the ``ResourceMonitor`` protocol and
  ``add_resource_monitor`` / ``record_resource_snapshot`` interfaces are
  reserved for future GPU VRAM / CPU RAM tracking.  They are stubs in this
  version (Step 2) and will be implemented when profiling needs arise.

Thread safety
-------------
``deque.append`` is atomic under CPython's GIL, so concurrent probe writes
from different threads are safe.  However, ``on_task_end`` (which snapshots
the deque) should only be called from the main thread.  When Step 4 adds
background write-back threads, review whether a lock is needed around
``_get_task_records``.

Disabling
---------
Set ``enabled=False`` to make ``measure()`` a zero-overhead no-op.
Recommended in production when latency numbers are not needed.

Usage example::

    timer = SystemTimer(enabled=True, output_csv_dir="/tmp/timing")

    # Register probes once at startup (component __init__):
    timer.register_probe("stage1_vision", backend="cuda")
    timer.register_probe("stage2_llm",    backend="cuda")
    timer.register_probe("stage3_flow",   backend="cuda")
    timer.register_probe("cp1_gate",      backend="cpu")   # Step 4+

    # Task lifecycle (driven by WebSocket server):
    timer.on_task_begin()

    # Hot path (called once per inference):
    with timer.measure("stage1_vision"):
        stage1 = model.run_stage1(obs)
    with timer.measure("stage2_llm"):
        stage2 = model.run_stage2(stage1)

    timer.on_task_end()   # prints summary, writes CSV if configured
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, runtime_checkable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimingRecord:
    """A single timing measurement captured by one ``timer.measure()`` call.

    ``resources`` is reserved for future resource-usage snapshots (GPU VRAM,
    CPU RAM, etc.) attached to the same measurement point.  It is ``None`` in
    Step 2; ``record_resource_snapshot()`` will populate it in a later step.
    """

    name: str
    """Probe name, e.g. ``"stage1_vision"`` or ``"cp2_judge"``."""

    elapsed_ms: float
    """Wall / GPU execution time in milliseconds."""

    timestamp: float
    """``time.time()`` at the moment the measurement was *completed*.
    Used as the time axis when exporting to CSV."""

    task_id: int
    """Identifier of the task (connection) this record belongs to."""

    resources: Optional[Dict[str, float]] = field(default=None, compare=False)
    """Reserved for future resource monitors.  Keys are metric names,
    e.g. ``{"gpu_alloc_mb": 4096.0}``.  Do not read this field in Step 2."""


@dataclass
class TimingStats:
    """Aggregated statistics for one probe over a set of ``TimingRecord`` objects."""

    name: str
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_records(cls, name: str, records: List[TimingRecord]) -> "TimingStats":
        """Compute statistics from a list of records for the same probe.

        Args:
            name: Probe name.
            records: All ``TimingRecord`` objects whose ``name`` matches.
                     Must be non-empty.
        """
        values = np.array([r.elapsed_ms for r in records], dtype=np.float64)
        return cls(
            name=name,
            count=len(values),
            mean_ms=float(np.mean(values)),
            p50_ms=float(np.percentile(values, 50)),
            p95_ms=float(np.percentile(values, 95)),
            p99_ms=float(np.percentile(values, 99)),
            min_ms=float(np.min(values)),
            max_ms=float(np.max(values)),
        )


# ---------------------------------------------------------------------------
# Backend protocols and implementations
# ---------------------------------------------------------------------------

class TimingBackend(Protocol):
    """Protocol for interchangeable timing backends.

    Implementors must be reusable across multiple ``measure()`` calls.  Each
    call to ``start()`` returns an opaque *handle* that is passed unchanged to
    the corresponding ``stop()`` call.

    Implementors must **not** hold mutable per-measurement state as instance
    variables; all state must live in the handle returned by ``start()``.
    This allows the same backend instance to be shared across nested
    ``measure()`` calls (which will happen when cache and inference probes are
    both active in Step 4+).
    """

    def start(self) -> Any:
        """Begin timing.  Returns an opaque handle."""
        ...

    def stop(self, handle: Any) -> float:
        """End timing.  Returns elapsed time in milliseconds.

        May block the calling thread briefly (e.g. CUDA event synchronize).
        """
        ...


class CudaEventBackend:
    """GPU timing backend using ``torch.cuda.Event``.

    Records start and end events directly onto the specified CUDA stream.
    ``stop()`` calls ``end_event.synchronize()``, which blocks the CPU thread
    only until the GPU has processed the end event — it does NOT flush the
    entire CUDA pipeline (unlike ``torch.cuda.synchronize()``).

    When CUDA is unavailable (e.g. CPU-only test environments), this backend
    transparently falls back to ``PerfCounterBackend`` behaviour so that code
    running in CI without a GPU still works correctly.

    Args:
        stream: The CUDA stream on which to record events.
                ``None`` (default) records on the *current* stream at the time
                ``measure()`` is entered, which is the default stream during
                normal inference.  Pass an explicit stream (e.g.
                ``cache_stream``) for operations running on a non-default
                stream (Step 4+).

    Note:
        CUDA Event timing measures the interval between two points on the
        *GPU timeline*, not wall-clock time.  This gives the actual GPU
        execution time, excluding Python overhead and inter-stage gaps.
        It is the preferred method for Stage 1 / 2 / 3 timing.
    """

    def __init__(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        self._stream = stream
        self._fallback = PerfCounterBackend()

    def start(self) -> Any:
        if not torch.cuda.is_available():
            # No GPU: fall back to CPU nanosecond timer.
            return ("cpu", self._fallback.start())

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        # record() inserts a marker into the stream.  If self._stream is None,
        # the marker goes onto the *current* CUDA stream (default stream during
        # inference).  Using an explicit stream is required for cache_stream
        # operations in Step 4+.
        start_evt.record(self._stream)
        return ("cuda", start_evt, end_evt)

    def stop(self, handle: Any) -> float:
        if handle[0] == "cpu":
            return self._fallback.stop(handle[1])

        _, start_evt, end_evt = handle
        end_evt.record(self._stream)
        # Block only until *this* end event is processed.  Other streams and
        # kernels queued after this event are unaffected.
        end_evt.synchronize()
        # elapsed_time returns milliseconds as a float.
        return start_evt.elapsed_time(end_evt)


class PerfCounterBackend:
    """CPU timing backend using ``time.perf_counter_ns``.

    Uses the highest-resolution timer available on the platform.  Suitable
    for CPU-only operations: gate decisions, FAISS CPU search, write-back
    threads, similarity judge, etc.

    Resolution is typically sub-microsecond; accuracy depends on OS scheduler
    jitter.  For GPU operations use ``CudaEventBackend`` instead.
    """

    def start(self) -> int:
        """Returns current time in nanoseconds."""
        return time.perf_counter_ns()

    def stop(self, handle: int) -> float:
        """Returns elapsed time in milliseconds."""
        return (time.perf_counter_ns() - handle) / 1_000_000.0


# ---------------------------------------------------------------------------
# Resource monitor protocol (reserved for future use)
# ---------------------------------------------------------------------------

@runtime_checkable
class ResourceMonitor(Protocol):
    """Protocol for pluggable resource-usage monitors.

    Implementations are **not** required in Step 2.  This protocol defines the
    interface so that ``SystemTimer.add_resource_monitor`` and
    ``record_resource_snapshot`` can be called without changes when concrete
    monitors are added (e.g. ``GpuVramMonitor``, ``CpuRamMonitor``).

    Planned implementations (future steps):
        - ``GpuVramMonitor``: reads ``torch.cuda.memory_stats()`` to report
          allocated and peak VRAM in MB.
        - ``CpuRamMonitor``: reads ``psutil.virtual_memory()`` to report RSS.

    Usage when implemented::

        class GpuVramMonitor:
            name = "gpu_vram"
            def sample(self) -> dict[str, float]:
                stats = torch.cuda.memory_stats()
                return {
                    "alloc_mb": stats["allocated_bytes.all.current"] / 1e6,
                    "peak_mb":  stats["allocated_bytes.all.peak"]    / 1e6,
                }

        timer.add_resource_monitor(GpuVramMonitor())
        with timer.measure("stage1_vision"):
            ...
        # The TimingRecord for "stage1_vision" will have resources populated
        # after record_resource_snapshot() is called (Step N).
    """

    name: str
    """Unique identifier for this monitor, used as key prefix in CSV."""

    def sample(self) -> Dict[str, float]:
        """Capture current resource metrics.

        Returns:
            Dict mapping metric names to numeric values.
            E.g. ``{"alloc_mb": 4096.0, "peak_mb": 5120.0}``.
        """
        ...


# ---------------------------------------------------------------------------
# TaskLifecycle protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TaskLifecycle(Protocol):
    """Protocol for objects that participate in task (connection) lifecycle.

    ``WebsocketPolicyServer`` calls these methods when a client connects and
    disconnects.  The server uses ``hasattr`` to check for the protocol rather
    than importing ``InferenceInterceptor`` directly, keeping server code
    decoupled from cache internals.

    ``InferenceInterceptor`` implements this protocol by forwarding calls to
    its internal ``SystemTimer``.

    Note:
        If the policy is a plain ``Policy`` (non-cache path), these methods
        are absent and the server skips the calls gracefully.
    """

    def on_task_begin(self) -> None:
        """Called when a client connection opens.

        Resets per-task state so that the summary at task end reflects only
        the current connection's measurements.
        """
        ...

    def on_task_end(self) -> None:
        """Called when a client connection closes.

        Prints a timing summary to the terminal and (if configured) flushes
        all records for this task to a CSV file in one batch write.
        """
        ...


# ---------------------------------------------------------------------------
# SystemTimer
# ---------------------------------------------------------------------------

# Default probe display order in the summary table.  Probes not listed here
# appear after these in alphabetical order.  Update this list when new
# standard probes are added (e.g. cp1_gate in Step 4).
_DISPLAY_ORDER = [
    "stage1_vision",
    "stage2_llm",
    "stage3_flow",
    "stage3_partial_flow",
    "total_inference",
]


class SystemTimer:
    """Unified timing system for the Pi0 inference + cache pipeline.

    See module docstring for an overview and usage example.

    Args:
        enabled: When ``False``, ``measure()`` is a zero-overhead no-op.
                 Use ``False`` in production when timing data is not needed.
        buffer_size: Maximum number of ``TimingRecord`` objects kept in memory.
                     Older records are dropped (ring buffer) when the limit is
                     reached.  10 000 is ample for a typical robot episode
                     (< 1 000 inferences × 5 probes = 5 000 records).
        output_csv_dir: If set, ``on_task_end()`` writes a CSV file to this
                        directory containing all records from the completed
                        task.  The file is written in a single batch at task
                        end to avoid per-record disk IO.
                        Set to ``None`` (default) to disable CSV output.
    """

    def __init__(
        self,
        enabled: bool = True,
        buffer_size: int = 10_000,
        output_csv_dir: Optional[str] = None,
    ) -> None:
        self._enabled = enabled
        self._output_csv_dir = output_csv_dir

        # Ring buffer: holds TimingRecord objects.  deque.append is atomic
        # under CPython's GIL, safe for concurrent probe writes from threads.
        self._records: deque[TimingRecord] = deque(maxlen=buffer_size)

        # Monotonic counter of total records ever appended (never decremented).
        # Used to identify the boundary between tasks without relying on deque
        # length, which can wrap when buffer_size is exceeded.
        self._total_appended: int = 0

        # Set by on_task_begin(); marks where the current task starts.
        self._task_start_total: int = 0

        # Incrementing task ID; used in CSV filenames and summary headers.
        self._task_id: int = 0

        # Maps probe name → backend instance.
        # Populated by register_probe(); read-only in the hot path.
        self._probes: Dict[str, TimingBackend] = {}

        # Resource monitors.  Empty in Step 2; populated by
        # add_resource_monitor() in future steps.
        self._resource_monitors: List[ResourceMonitor] = []

    # -----------------------------------------------------------------------
    # Probe registration  (called once at component __init__, not hot path)
    # -----------------------------------------------------------------------

    def register_probe(
        self,
        name: str,
        backend: str = "cuda",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Register a named timing probe.

        Must be called before the first ``measure(name)`` call.  Can be called
        again to update the backend (e.g. to switch a probe from "cuda" to a
        specific non-default stream in Step 4+).

        Args:
            name: Unique probe identifier.  Use the naming convention from
                  ``docs/cache_system_architecture.md`` §9:
                  ``stage1_vision``, ``cp1_gate``, ``cp2_search``, etc.
            backend: ``"cuda"`` for GPU operations (default), ``"cpu"`` for
                     CPU-only operations.  Passing an invalid string raises
                     ``ValueError``.
            stream: CUDA stream for ``"cuda"`` backend.  ``None`` means the
                    default stream (correct for inference stages).  Pass an
                    explicit ``torch.cuda.Stream`` for cache-stream operations
                    (Step 4+).  Ignored when ``backend="cpu"``.
        """
        if backend == "cuda":
            self._probes[name] = CudaEventBackend(stream=stream)
        elif backend == "cpu":
            self._probes[name] = PerfCounterBackend()
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Expected 'cuda' or 'cpu'."
            )

    # -----------------------------------------------------------------------
    # Hot-path measurement context manager
    # -----------------------------------------------------------------------

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Time the body of a ``with`` block and store the result.

        When ``enabled=False``, this is a pure no-op: no objects are created,
        no GPU events are recorded.

        If ``name`` was not registered via ``register_probe()``, a
        ``CudaEventBackend`` is used as a lenient fallback so that probes
        added during development don't require upfront registration.  A
        ``RuntimeWarning`` is emitted to catch the omission during testing.

        Args:
            name: Probe name, must match a ``register_probe`` call.

        Usage::

            with timer.measure("stage2_llm"):
                past_kv = model.run_stage2(stage1)
            # After the with-block, one TimingRecord("stage2_llm", ...) has
            # been appended to the internal ring buffer.
        """
        if not self._enabled:
            yield
            return

        backend = self._probes.get(name)
        if backend is None:
            import warnings
            warnings.warn(
                f"SystemTimer: probe '{name}' was not registered via "
                "register_probe().  Using CudaEventBackend as fallback.  "
                "Call register_probe() in the component __init__ to silence this.",
                RuntimeWarning,
                stacklevel=2,
            )
            backend = CudaEventBackend()
            self._probes[name] = backend  # cache for future calls

        handle = backend.start()
        try:
            yield
        finally:
            elapsed_ms = backend.stop(handle)
            record = TimingRecord(
                name=name,
                elapsed_ms=elapsed_ms,
                timestamp=time.time(),
                task_id=self._task_id,
            )
            self._records.append(record)
            self._total_appended += 1

    # -----------------------------------------------------------------------
    # Task lifecycle  (called by WebsocketPolicyServer via TaskLifecycle)
    # -----------------------------------------------------------------------

    def on_task_begin(self) -> None:
        """Mark the start of a new task (client connection).

        Records the current ``_total_appended`` counter so that
        ``_get_task_records()`` can isolate measurements from this task even
        if the ring buffer contains records from previous tasks.

        Called by ``InferenceInterceptor.on_task_begin()``, which is invoked
        by ``WebsocketPolicyServer._handler()`` when a connection opens.
        """
        self._task_start_total = self._total_appended

    def on_task_end(self) -> None:
        """Finalise the current task.

        Actions taken in order:
        1. Compute per-probe statistics for records collected since
           ``on_task_begin()``.
        2. Print a formatted summary table to the terminal (stdout).
        3. If ``output_csv_dir`` was set, write all task records to a CSV
           file in a single batch write (no per-record IO).
        4. Increment ``task_id`` for the next connection.

        Note: Records are *not* cleared from the ring buffer; they remain
        accessible (up to ``buffer_size``) for post-hoc inspection.

        Called by ``InferenceInterceptor.on_task_end()``, which is invoked
        by ``WebsocketPolicyServer._handler()`` when a connection closes.
        """
        task_records = self._get_task_records()
        self._print_summary(task_records)

        if self._output_csv_dir is not None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                self._output_csv_dir,
                f"timing_task_{self._task_id:04d}_{ts}.csv",
            )
            self.export_csv(path, records=task_records)

        self._task_id += 1

    # -----------------------------------------------------------------------
    # Statistics and export
    # -----------------------------------------------------------------------

    def summary(self, task_only: bool = True) -> Dict[str, TimingStats]:
        """Compute per-probe statistics.

        Args:
            task_only: If ``True`` (default), only include records collected
                       since the last ``on_task_begin()`` call.  If ``False``,
                       include all records currently in the ring buffer
                       (spanning multiple tasks).

        Returns:
            Dict mapping probe name → ``TimingStats``.  Probes with no
            records in the selected range are omitted.
        """
        records = self._get_task_records() if task_only else list(self._records)
        return _compute_stats(records)

    def export_csv(
        self,
        path: str,
        records: Optional[List[TimingRecord]] = None,
    ) -> None:
        """Write timing records to a CSV file in a single batch write.

        All rows are first assembled in memory, then written to disk at once.
        This avoids the overhead of opening/closing the file or flushing after
        each row, which is important when exporting thousands of records at
        task end.

        Args:
            path: Absolute or relative path for the output file.  Parent
                  directories must already exist.
            records: Records to export.  Defaults to the current task's
                     records (same as ``summary(task_only=True)``).

        CSV columns:
            ``timestamp``, ``task_id``, ``name``, ``elapsed_ms``
            Future columns for resource data will be appended when
            ``ResourceMonitor`` implementations are added.
        """
        if records is None:
            records = self._get_task_records()

        # Determine resource columns (present only if any record has resources).
        resource_keys: List[str] = []
        for r in records:
            if r.resources:
                resource_keys = sorted(r.resources.keys())
                break

        # Build all rows in memory first, then write once.
        rows: List[List[Any]] = []
        for r in records:
            row: List[Any] = [r.timestamp, r.task_id, r.name, r.elapsed_ms]
            for k in resource_keys:
                row.append(r.resources.get(k, "") if r.resources else "")
            rows.append(row)

        header = ["timestamp", "task_id", "name", "elapsed_ms"] + resource_keys
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)  # single batch write

    # -----------------------------------------------------------------------
    # Resource monitor interface  (stubs — implemented in future steps)
    # -----------------------------------------------------------------------

    def add_resource_monitor(self, monitor: ResourceMonitor) -> None:
        """Register a resource-usage monitor.

        Registered monitors will be sampled when ``record_resource_snapshot``
        is called.  Multiple monitors can be registered; each contributes its
        own keys to ``TimingRecord.resources``.

        This method is a stub in Step 2.  It stores the monitor but
        ``record_resource_snapshot`` does not yet call it.

        Args:
            monitor: Any object implementing the ``ResourceMonitor`` protocol.
        """
        # TODO(Step N): activate monitors once profiling needs arise.
        self._resource_monitors.append(monitor)

    def record_resource_snapshot(self, name: str) -> None:
        """Sample all registered monitors and attach results to the most
        recent ``TimingRecord`` whose ``name`` matches.

        Intended call site: immediately after a ``measure()`` block, before
        the next stage starts.

        This method is a stub in Step 2.  When resource monitors are added,
        implement by iterating ``self._resource_monitors``, calling
        ``monitor.sample()``, and merging the results into the matching
        record's ``resources`` dict.

        Args:
            name: Probe name identifying which record to annotate.
        """
        # TODO(Step N): implement once ResourceMonitor subclasses exist.
        pass

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_task_records(self) -> List[TimingRecord]:
        """Return records appended since the last ``on_task_begin()`` call.

        Uses the monotonic ``_total_appended`` counter to compute how many
        records belong to the current task, then slices from the tail of the
        deque.  This is robust when the deque wraps (i.e. ``buffer_size``
        records were added within one task); in that case only the most recent
        ``buffer_size`` records are returned with a warning.
        """
        task_count = self._total_appended - self._task_start_total
        if task_count <= 0:
            return []

        all_records = list(self._records)  # snapshot
        if task_count > len(all_records):
            import warnings
            warnings.warn(
                f"SystemTimer: ring buffer overflowed during task "
                f"(task produced {task_count} records, buffer holds "
                f"{len(all_records)}).  Only the most recent "
                f"{len(all_records)} records are available.  "
                "Increase buffer_size if full task history is needed.",
                RuntimeWarning,
                stacklevel=3,
            )
            return all_records

        return all_records[-task_count:]

    def _print_summary(self, task_records: List[TimingRecord]) -> None:
        """Print a formatted timing summary table to stdout.

        The table shows per-probe statistics (count, mean, p50, p95, p99) in
        milliseconds.  Probes are displayed in ``_DISPLAY_ORDER``; unrecognised
        probes appear afterwards in alphabetical order.

        A "total (sum)" row is appended when ``stage1_vision``,
        ``stage2_llm``, and ``stage3_flow`` all have the same call count,
        computed by summing the three stage times for each individual
        inference call (not by summing per-probe means, which would give the
        correct mean but incorrect percentiles).
        """
        if not task_records:
            print(
                f"\n=== Inference Timing Summary "
                f"(task #{self._task_id}) — no records ===\n"
            )
            return

        stats = _compute_stats(task_records)
        if not stats:
            return

        # Determine number of inferences = max probe count in this task.
        n_calls = max(s.count for s in stats.values())

        # Build display order: standard probes first, then alphabetical rest.
        ordered_names = [n for n in _DISPLAY_ORDER if n in stats]
        extra = sorted(n for n in stats if n not in _DISPLAY_ORDER)
        ordered_names.extend(extra)

        col_w = 26  # width of probe name column
        header = (
            f"  {'Probe':<{col_w}} {'N':>5} "
            f"{'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}  ms"
        )
        separator = "  " + "-" * (len(header) - 2)

        lines = [
            f"\n=== Inference Timing Summary "
            f"(task #{self._task_id}, {n_calls} inferences) ===",
            header,
            separator,
        ]

        for name in ordered_names:
            s = stats[name]
            lines.append(
                f"  {name:<{col_w}} {s.count:>5} "
                f"{s.mean_ms:>8.1f} {s.p50_ms:>8.1f} "
                f"{s.p95_ms:>8.1f} {s.p99_ms:>8.1f}"
            )

        # Per-call total row: sum stage1 + stage2 + stage3 per inference.
        stage_names = ("stage1_vision", "stage2_llm", "stage3_flow")
        if all(n in stats for n in stage_names):
            counts = [stats[n].count for n in stage_names]
            if len(set(counts)) == 1:  # all three have identical call counts
                # Reconstruct per-call totals from the raw records.
                per_probe: Dict[str, List[float]] = {
                    n: [] for n in stage_names
                }
                for r in task_records:
                    if r.name in per_probe:
                        per_probe[r.name].append(r.elapsed_ms)

                # Zip the three lists to get per-call sums.
                min_len = min(len(v) for v in per_probe.values())
                totals = [
                    per_probe["stage1_vision"][i]
                    + per_probe["stage2_llm"][i]
                    + per_probe["stage3_flow"][i]
                    for i in range(min_len)
                ]
                if totals:
                    arr = np.array(totals)
                    lines.append(separator)
                    lines.append(
                        f"  {'total (sum)':<{col_w}} {len(totals):>5} "
                        f"{float(np.mean(arr)):>8.1f} "
                        f"{float(np.percentile(arr, 50)):>8.1f} "
                        f"{float(np.percentile(arr, 95)):>8.1f} "
                        f"{float(np.percentile(arr, 99)):>8.1f}"
                    )

        lines.append("")
        print("\n".join(lines))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _compute_stats(records: List[TimingRecord]) -> Dict[str, TimingStats]:
    """Group ``records`` by probe name and compute ``TimingStats`` for each.

    Records with zero elapsed time are included (they indicate fast/cached
    paths and should not be silently dropped).
    """
    grouped: Dict[str, List[TimingRecord]] = {}
    for r in records:
        grouped.setdefault(r.name, []).append(r)

    return {
        name: TimingStats.from_records(name, recs)
        for name, recs in grouped.items()
    }
