"""Step 2 verification script.

Run with:
    uv run scripts/verify_step2.py

Tests performed (no GPU or model weights required):
    1. Syntax check — all modified files parse without error.
    2. Stale code check — removed patterns must not exist in modified files.
    3. SystemTimer CPU probe — basic measure/summary cycle.
    4. enabled=False — measure() must be a zero-overhead no-op.
    5. Ring buffer task boundary — on_task_begin/end isolates records correctly.
    6. Multi-task isolation — task_id increments; summary covers only current task.
    7. Unregistered probe warning — RuntimeWarning is emitted.
    8. export_csv — records written to a temp file with correct columns/row count.
    9. TaskLifecycle protocol — InferenceInterceptor exposes on_task_begin/end.
   10. Summary format — output contains expected header and probe name.
"""

from __future__ import annotations

import ast
import contextlib
import csv
import io
import sys
import tempfile
import time
import warnings

PASS = "\033[32m  PASS\033[0m"
FAIL = "\033[31m  FAIL\033[0m"
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"{PASS}  {label}")
    else:
        msg = f"{label}" + (f" — {detail}" if detail else "")
        print(f"{FAIL}  {msg}")
        _failures.append(msg)


def check_raises(label: str, exc_type: type, fn) -> None:
    try:
        fn()
        print(f"{FAIL}  {label} — expected {exc_type.__name__}, nothing raised")
        _failures.append(label)
    except exc_type:
        print(f"{PASS}  {label}")
    except Exception as e:
        print(f"{FAIL}  {label} — raised {type(e).__name__}: {e}")
        _failures.append(label)


# ---------------------------------------------------------------------------
# Test 1: Syntax check
# ---------------------------------------------------------------------------
print("\n[1] Syntax check")

FILES = [
    "src/openpi/cache/timing.py",
    "src/openpi/cache/interceptor.py",
    "src/openpi/cache/__init__.py",
    "src/openpi/serving/websocket_policy_server.py",
]
for path in FILES:
    try:
        ast.parse(open(path).read())
        check(path, True)
    except SyntaxError as e:
        check(path, False, str(e))

# ---------------------------------------------------------------------------
# Test 2: Stale code must not exist in modified files
# ---------------------------------------------------------------------------
print("\n[2] Stale code removal check")

def _code_lines(path: str) -> str:
    """Return only executable code lines (strip comments and docstring lines).

    This is intentionally simple: we drop lines whose first non-whitespace
    character is ``#``, and lines that consist only of triple-quote delimiters
    or content inside a module/function docstring.  It is good enough to
    distinguish ``time.monotonic`` appearing in a docstring (explaining what
    was removed) from the same string appearing as a live function call.

    A proper AST-based check would be more robust but is unnecessary here.
    """
    lines = open(path).readlines()
    result = []
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        # Toggle docstring state on triple-quote markers.
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # A line that opens AND closes a docstring on the same line
            # (e.g. '''one-liner''') stays outside.
            delim = '"""' if stripped.startswith('"""') else "'''"
            count = stripped.count(delim)
            if in_docstring:
                in_docstring = False  # closing delimiter
                continue
            elif count >= 2:
                continue  # single-line docstring
            else:
                in_docstring = True
                continue
        if in_docstring:
            continue
        if stripped.startswith("#"):
            continue
        result.append(line)
    return "".join(result)


stale_checks = [
    # (file, pattern, description, code_only)
    # code_only=True  → search only executable lines (skip comments/docstrings)
    # code_only=False → search the entire file (pattern must not appear anywhere)
    (
        "src/openpi/serving/websocket_policy_server.py",
        "stage_timing_records",
        "websocket_policy_server.py must not contain stage_timing_records",
        False,
    ),
    (
        "src/openpi/serving/websocket_policy_server.py",
        "token_prep_ms",
        "websocket_policy_server.py must not contain token_prep_ms",
        False,
    ),
    (
        "src/openpi/cache/interceptor.py",
        "time.monotonic",
        "interceptor.py code must not call time.monotonic",
        True,   # may appear in docstrings explaining removal; check code only
    ),
    (
        "src/openpi/cache/interceptor.py",
        "cuda.synchronize",
        "interceptor.py code must not call cuda.synchronize",
        True,
    ),
    (
        "src/openpi/cache/interceptor.py",
        '"stage_timing"',   # the dict key as a string literal
        'interceptor.py code must not output "stage_timing" key',
        True,
    ),
]

for path, pattern, label, code_only in stale_checks:
    content = _code_lines(path) if code_only else open(path).read()
    check(label, pattern not in content, f"found '{pattern}' in {path}")

# Positive checks: these patterns MUST be present.
required_checks = [
    (
        "scripts/serve_policy.py",
        "timing_csv_dir",
        "serve_policy.py must expose --timing_csv_dir argument",
    ),
    (
        "scripts/serve_policy.py",
        "timer=timer",
        "serve_policy.py must pass timer instance to InferenceInterceptor",
    ),
    (
        "src/openpi/cache/interceptor.py",
        "torch.compile",
        "interceptor.py must compile stage methods with torch.compile",
    ),
    (
        "src/openpi/cache/interceptor.py",
        "pytorch_compile_mode",
        "interceptor.py must read compile mode from model config",
    ),
]

for path, pattern, label in required_checks:
    content = open(path).read()
    check(label, pattern in content, f"'{pattern}' not found in {path}")

# ---------------------------------------------------------------------------
# Test 3: SystemTimer — CPU probe basic cycle
# ---------------------------------------------------------------------------
print("\n[3] SystemTimer CPU probe")

from openpi.cache.timing import (  # noqa: E402
    CudaEventBackend,
    PerfCounterBackend,
    SystemTimer,
    TaskLifecycle,
    TimingRecord,
    TimingStats,
)

timer = SystemTimer(enabled=True, buffer_size=200)
timer.register_probe("stage1_vision", backend="cpu")   # cpu fallback (no GPU needed)
timer.register_probe("stage2_llm",    backend="cpu")
timer.register_probe("stage3_flow",   backend="cpu")
timer.register_probe("total_inference", backend="cpu")

timer.on_task_begin()
for _ in range(10):
    with timer.measure("total_inference"):
        with timer.measure("stage1_vision"):
            time.sleep(0.003)
        with timer.measure("stage2_llm"):
            time.sleep(0.008)
        with timer.measure("stage3_flow"):
            time.sleep(0.005)

stats = timer.summary(task_only=True)
check("stage1_vision recorded 10 times", stats["stage1_vision"].count == 10)
check("stage2_llm recorded 10 times",    stats["stage2_llm"].count == 10)
check("stage3_flow recorded 10 times",   stats["stage3_flow"].count == 10)
check("total_inference recorded 10 times", stats["total_inference"].count == 10)

s1 = stats["stage1_vision"]
check(
    f"stage1_vision mean in [1, 20] ms (got {s1.mean_ms:.2f})",
    1.0 <= s1.mean_ms <= 20.0,
)
check("p50 <= p95 <= p99", s1.p50_ms <= s1.p95_ms <= s1.p99_ms)
check("min <= mean <= max", s1.min_ms <= s1.mean_ms <= s1.max_ms)

total_s = stats["total_inference"]
check(
    f"total_inference mean > stage1+stage2+stage3 means",
    total_s.mean_ms >= (
        stats["stage1_vision"].mean_ms
        + stats["stage2_llm"].mean_ms
        + stats["stage3_flow"].mean_ms
    ) * 0.9,  # 10% tolerance for timer overhead
)

# ---------------------------------------------------------------------------
# Test 4: enabled=False is a no-op
# ---------------------------------------------------------------------------
print("\n[4] enabled=False zero-overhead no-op")

off = SystemTimer(enabled=False)
off.register_probe("p", backend="cpu")
for _ in range(5):
    with off.measure("p"):
        time.sleep(0.001)

check("no records appended when disabled", len(list(off._records)) == 0)
check("summary returns empty dict when disabled", off.summary() == {})

# ---------------------------------------------------------------------------
# Test 5: Ring buffer task boundary
# ---------------------------------------------------------------------------
print("\n[5] Ring buffer task boundary isolation")

t = SystemTimer(enabled=True, buffer_size=500)
t.register_probe("x", backend="cpu")

# Task 0: 3 records
t.on_task_begin()
for _ in range(3):
    with t.measure("x"):
        pass
task0_stats = t.summary(task_only=True)
check("task 0 sees 3 records", task0_stats["x"].count == 3)
t.on_task_end.__func__  # just ensure callable

# Manually call on_task_end internals to advance task without printing
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    t.on_task_end()

# Task 1: 7 records
t.on_task_begin()
for _ in range(7):
    with t.measure("x"):
        pass
task1_stats = t.summary(task_only=True)
check("task 1 sees 7 records (not 10)", task1_stats["x"].count == 7,
      f"got {task1_stats['x'].count}")

# task_only=False sees all 10
all_stats = t.summary(task_only=False)
check("task_only=False sees all 10 records", all_stats["x"].count == 10,
      f"got {all_stats['x'].count}")

# ---------------------------------------------------------------------------
# Test 6: task_id increments correctly
# ---------------------------------------------------------------------------
print("\n[6] task_id increment")

t2 = SystemTimer(enabled=True)
t2.register_probe("p", backend="cpu")
check("initial task_id is 0", t2._task_id == 0)

for _ in range(3):
    t2.on_task_begin()
    with t2.measure("p"):
        pass
    with contextlib.redirect_stdout(io.StringIO()):
        t2.on_task_end()

check("task_id is 3 after 3 tasks", t2._task_id == 3, f"got {t2._task_id}")

# ---------------------------------------------------------------------------
# Test 7: Unregistered probe emits RuntimeWarning
# ---------------------------------------------------------------------------
print("\n[7] Unregistered probe warning")

t3 = SystemTimer(enabled=True)
t3.on_task_begin()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    with t3.measure("never_registered"):
        pass
check(
    "RuntimeWarning emitted for unregistered probe",
    len(w) == 1 and issubclass(w[0].category, RuntimeWarning),
    f"got {[str(x.category) for x in w]}",
)
check(
    "warning message mentions probe name",
    "never_registered" in str(w[0].message),
)
# Second call must NOT warn again (backend cached)
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    with t3.measure("never_registered"):
        pass
check("no warning on repeated call to same unregistered probe", len(w2) == 0,
      f"got {len(w2)} warnings")

# ---------------------------------------------------------------------------
# Test 8: export_csv
# ---------------------------------------------------------------------------
print("\n[8] export_csv")

t4 = SystemTimer(enabled=True)
t4.register_probe("s1", backend="cpu")
t4.register_probe("s2", backend="cpu")
t4.on_task_begin()

N = 6
for _ in range(N):
    with t4.measure("s1"):
        time.sleep(0.001)
    with t4.measure("s2"):
        time.sleep(0.001)

with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    csv_path = f.name

t4.export_csv(csv_path)

with open(csv_path) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

check(f"CSV has {N*2} data rows", len(rows) == N * 2, f"got {len(rows)}")
check("CSV has 'name' column",        "name" in rows[0])
check("CSV has 'elapsed_ms' column",  "elapsed_ms" in rows[0])
check("CSV has 'task_id' column",     "task_id" in rows[0])
check("CSV has 'timestamp' column",   "timestamp" in rows[0])
names_in_csv = {r["name"] for r in rows}
check("CSV contains s1 and s2 probes", names_in_csv == {"s1", "s2"},
      f"got {names_in_csv}")

# output_csv_dir auto-write
with tempfile.TemporaryDirectory() as tmpdir:
    t5 = SystemTimer(enabled=True, output_csv_dir=tmpdir)
    t5.register_probe("p", backend="cpu")
    t5.on_task_begin()
    for _ in range(3):
        with t5.measure("p"):
            pass
    with contextlib.redirect_stdout(io.StringIO()):
        t5.on_task_end()

    import os
    csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
    check("on_task_end auto-writes CSV when output_csv_dir is set",
          len(csv_files) == 1, f"found {csv_files}")

# ---------------------------------------------------------------------------
# Test 9: TaskLifecycle on InferenceInterceptor
# ---------------------------------------------------------------------------
print("\n[9] InferenceInterceptor TaskLifecycle")

from openpi.cache.interceptor import InferenceInterceptor  # noqa: E402

check("on_task_begin exists",  hasattr(InferenceInterceptor, "on_task_begin"))
check("on_task_end exists",    hasattr(InferenceInterceptor, "on_task_end"))
check(
    "InferenceInterceptor matches TaskLifecycle protocol (hasattr check)",
    hasattr(InferenceInterceptor, "on_task_begin")
    and hasattr(InferenceInterceptor, "on_task_end"),
)

# ---------------------------------------------------------------------------
# Test 10: Summary format
# ---------------------------------------------------------------------------
print("\n[10] Summary format")

t6 = SystemTimer(enabled=True)
t6.register_probe("stage1_vision",  backend="cpu")
t6.register_probe("stage2_llm",     backend="cpu")
t6.register_probe("stage3_flow",    backend="cpu")
t6.register_probe("total_inference", backend="cpu")
t6.on_task_begin()

for _ in range(4):
    with t6.measure("total_inference"):
        with t6.measure("stage1_vision"):
            time.sleep(0.002)
        with t6.measure("stage2_llm"):
            time.sleep(0.006)
        with t6.measure("stage3_flow"):
            time.sleep(0.003)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    t6.on_task_end()
out = buf.getvalue()

check("output contains 'Inference Timing Summary'", "Inference Timing Summary" in out)
check("output contains 'stage1_vision'",  "stage1_vision"  in out)
check("output contains 'stage2_llm'",     "stage2_llm"     in out)
check("output contains 'stage3_flow'",    "stage3_flow"    in out)
check("output contains 'total (sum)' row", "total (sum)"   in out)
check("output contains '4 inferences'",   "4 inferences"   in out)
print("\nSummary output:\n" + out)

# ---------------------------------------------------------------------------
# Test 11: register_probe ValueError for bad backend
# ---------------------------------------------------------------------------
print("\n[11] register_probe rejects unknown backend")

check_raises(
    "ValueError for backend='bad'",
    ValueError,
    lambda: SystemTimer().register_probe("x", backend="bad"),
)

# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if _failures:
    print(f"\033[31mFAILED — {len(_failures)} check(s):\033[0m")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    total = sum(1 for line in open(__file__) if line.strip().startswith("check("))
    print(f"\033[32mALL PASSED ({total} checks)\033[0m")
    sys.exit(0)
