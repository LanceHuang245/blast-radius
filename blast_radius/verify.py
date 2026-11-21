"""Dual-phase forked runtime verifier with cost-aware early stopping.

Phase 1 (Clean Fork, before parent imports sqlglot):
    Verifies commits that touch class/module-level code, closures, or test files
    using an in-memory importlib MetaPathFinder + virtual open() overlay.

Phase 2 (Pre-Warmed Fork, after parent imports sqlglot once):
    Verifies commits that only modify existing non-nested function bodies via
    in-place __code__ replacement, and pure fixture edits via virtual open().
    Test modules are lazily pre-imported once in the parent and shared via COW.

Soundness invariant: a candidate test is dropped ONLY if it explicitly ran and
passed inside an isolated forked child against the patched source. Any error,
crash, timeout, or unverified test is conservatively retained.
"""

from __future__ import annotations

import builtins
import concurrent.futures as cf
import importlib
import importlib.abc
import importlib.machinery
import inspect
import io
import json
import os
import pickle
import select
import signal
import sys
import time
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass

from blast_radius import SOURCE, log
from blast_radius.impact_map import module_name
from blast_radius.schedule import split_with_cost
from blast_radius.select import Selection

# Decision thresholds (overridable via env vars for local A/B benchmarking).
MAX_SINGLE_TEST_MS = int(os.environ.get("BR_MAX_SINGLE_MS", "2500"))
MAX_COMMIT_VERIFY_MS = int(os.environ.get("BR_MAX_COMMIT_MS", "2500"))
MIN_GAIN_MS = int(os.environ.get("BR_MIN_GAIN_MS", "350"))
VERIFY_FIXTURE_MAX_MS = int(os.environ.get("BR_FIXTURE_MAX_MS", "0"))
TOTAL_VERIFY_BUDGET_MS = int(os.environ.get("BR_TOTAL_BUDGET_MS", "35000"))
ENABLE_HOTPATCH = os.environ.get("BR_ENABLE_HOTPATCH", "1") != "0"

# Hard wall-clock limits, enforced by the parent (the run itself has 3 minutes, 4.5 in the final).
CHILD_LIMIT_BASE_S = 5.0  # per child: base + factor x estimated CPU
CHILD_LIMIT_FACTOR = 3.0
VERIFY_WALL_BUDGET_S = float(os.environ.get("BR_VERIFY_WALL_S", "60"))

# Tests that inspect fresh import state or spawn subprocesses must never run in Phase 2.
FRESH_IMPORT_TESTS = frozenset({
    "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
    "tests/test_dialect_imports.py::TestDialectImports::test_dialect_attribute_access_is_cached",
    "tests/test_dialect_imports.py::TestDialectImports::test_dialect_import_no_deadlock",
})

# Tests that spawn a fresh Python interpreter against disk cannot see an in-memory overlay.
SUBPROCESS_TESTS = frozenset({
    "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
})

LAZY_EXECUTOR_ATTRS = (
    "tpch_conn",
    "tpcds_conn",
    "tpch_tables",
    "tpcds_tables",
    "tpch_sqls",
    "tpcds_sqls",
    "cache",
)


@dataclass
class VerifyTask:
    commit: str
    chosen: set[int]
    selection: Selection
    phase: int  # 1 = clean fork, 2 = pre-warmed fork
    static_cost_ms: int
    est_cpu_ms: int


class _InProcessExecutor:
    """Replaces ProcessPoolExecutor in child so workers see in-memory patches with zero spawn cost.

    Arguments and results still go through pickle, as they would to a spawned worker:
    sqlglot expressions pickle through serde (Expression.__reduce__), so a patch that
    breaks serialization must fail here too.
    """

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, /, *args, **kwargs):
        future: cf.Future = cf.Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            future.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as error:
            future.set_exception(error)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return [self.submit(fn, *args).result() for args in zip(*iterables)]

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _OverlayLoader(importlib.abc.Loader):
    def __init__(self, abs_path: str, source_text: str):
        self._abs_path = abs_path
        self._source_text = source_text

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        code = compile(self._source_text, self._abs_path, "exec")
        exec(code, module.__dict__)


class _OverlayFinder(importlib.abc.MetaPathFinder):
    """Serves patched .py modules directly from memory before filesystem finders."""

    def __init__(self, patched_files: dict[str, str]):
        self._modules: dict[str, tuple[str, str, bool]] = {}
        for rel_path, text in patched_files.items():
            if rel_path.endswith(".py"):
                mod = module_name(rel_path)
                abs_path = str((SOURCE / rel_path).resolve())
                is_pkg = rel_path.endswith("/__init__.py") or rel_path == "__init__.py"
                self._modules[mod] = (abs_path, text, is_pkg)

    def find_spec(self, fullname: str, path=None, target=None):
        entry = self._modules.get(fullname)
        if entry is None:
            return None
        abs_path, text, is_pkg = entry
        spec = importlib.machinery.ModuleSpec(
            fullname,
            _OverlayLoader(abs_path, text),
            origin=abs_path,
            is_package=is_pkg,
        )
        if is_pkg:
            spec.submodule_search_locations = [os.path.dirname(abs_path)]
        return spec


def _install_open_overlay(patched_files: dict[str, str]) -> None:
    """Redirect builtins.open and io.open reads of patched files to in-memory StringIO/BytesIO."""
    if not patched_files:
        return
    overlay: dict[str, tuple[str, bytes]] = {}
    for rel_path, text in patched_files.items():
        abs_path = os.path.abspath(str(SOURCE / rel_path))
        overlay[abs_path] = (text, text.encode("utf-8"))

    orig_open = builtins.open

    def patched_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and "w" not in mode and "a" not in mode and "+" not in mode:
            try:
                resolved = os.path.abspath(os.fsdecode(file))
            except Exception:
                resolved = ""
            hit = overlay.get(resolved)
            if hit is not None:
                text, raw = hit
                if "b" in mode:
                    return io.BytesIO(raw)
                return io.StringIO(text, newline=kwargs.get("newline"))
        return orig_open(file, mode, *args, **kwargs)

    builtins.open = patched_open
    io.open = patched_open


def _find_unique_code(parent_code: types.CodeType, steps: list[str]) -> types.CodeType | None:
    """Walk nested co_consts to locate the unique compiled CodeType for `qualname`."""
    cur = parent_code
    for name in steps:
        matches = [c for c in cur.co_consts if isinstance(c, types.CodeType) and c.co_name == name]
        if len(matches) != 1:
            return None
        cur = matches[0]
    return cur


def _apply_code_hotpatch(patched_files: dict[str, str], targets: list[tuple[str, str]]) -> bool:
    """Swap __code__ in-place on live sqlglot functions inside the forked Phase-2 child."""
    by_path: dict[str, list[str]] = {}
    for rel_path, qualname in targets:
        by_path.setdefault(rel_path, []).append(qualname)

    for rel_path, qualnames in by_path.items():
        text = patched_files.get(rel_path)
        if text is None:
            return False
        mod_name = module_name(rel_path)
        mod = importlib.import_module(mod_name)
        mod_code = compile(text, str((SOURCE / rel_path).resolve()), "exec")
        for qualname in qualnames:
            steps = qualname.split(".")
            new_code = _find_unique_code(mod_code, steps)
            if new_code is None:
                return False
            container: object = mod
            for part in steps[:-1]:
                container = container.__dict__.get(part) if hasattr(container, "__dict__") else None
                if container is None:
                    return False
            attr = steps[-1]
            raw = container.__dict__.get(attr) if hasattr(container, "__dict__") else None
            if isinstance(raw, (staticmethod, classmethod)):
                fn = raw.__func__
            elif isinstance(raw, property):
                fn = raw.fget
            elif inspect.isfunction(raw):
                fn = raw
            else:
                return False
            if not inspect.isfunction(fn):
                return False
            old_code = fn.__code__
            if (
                old_code.co_freevars != new_code.co_freevars
                or old_code.co_cellvars != new_code.co_cellvars
                or len(fn.__closure__ or ()) != len(new_code.co_freevars)
            ):
                return False
            fn.__code__ = new_code
    return True


def _apply_attr_hotpatch(targets: list[tuple[str, str, str, object]]) -> bool:
    """Update non-import-time literal class attributes in-place inside the forked Phase-2 child."""
    for rel_path, cls_qual, attr_name, lit_val in targets:
        mod = importlib.import_module(module_name(rel_path))
        container: object = mod
        for part in cls_qual.split("."):
            container = container.__dict__.get(part) if hasattr(container, "__dict__") else None
            if container is None:
                return False
        if not isinstance(container, type) or attr_name not in container.__dict__:
            return False
        setattr(container, attr_name, lit_val)
    return True


def _install_lazy_executor_setup(cls: type) -> None:
    """Defer TestExecutor's heavy TPC-H/TPC-DS CSV + DuckDB setup until actually accessed."""
    orig_cm = cls.__dict__.get("setUpClass")
    if not isinstance(orig_cm, classmethod):
        return
    real_setup = orig_cm.__func__
    initialized = [False]

    def ensure_initialized() -> None:
        if initialized[0]:
            return
        initialized[0] = True
        for attr_name in LAZY_EXECUTOR_ATTRS:
            if attr_name in cls.__dict__:
                delattr(cls, attr_name)
        real_setup(cls)

    class _LazyAttr:
        __slots__ = ("_name",)

        def __init__(self, name: str):
            self._name = name

        def __get__(self, obj, objtype=None):
            ensure_initialized()
            return getattr(cls, self._name)

    cls.setUpClass = classmethod(lambda c: None)
    for attr_name in LAZY_EXECUTOR_ATTRS:
        setattr(cls, attr_name, _LazyAttr(attr_name))


def _patch_pool_references() -> None:
    """Ensure both concurrent.futures and already-imported test modules use _InProcessExecutor."""
    cf.ProcessPoolExecutor = _InProcessExecutor
    for mod_name in ("tests.test_optimizer", "tests.test_executor"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "ProcessPoolExecutor"):
            setattr(mod, "ProcessPoolExecutor", _InProcessExecutor)
def _install_fail_fast_subtest() -> None:
    """Abort a test immediately on the first failing subtest (while honoring SkipTest)."""

    @contextmanager
    def _ff_subtest(self, msg=unittest.case._subtest_msg_sentinel, **params):
        try:
            yield
        except unittest.SkipTest:
            pass

    unittest.TestCase.subTest = _ff_subtest


def _should_continue_after_failure(
    failed: set[int],
    unverified: set[int],
    ms_by_number: dict[int, int],
) -> bool:
    """Only continue running tests after a failure if dropping them can save strictly more than CPU."""
    if not unverified:
        return False
    cur_cost, _ = split_with_cost([(ms_by_number[n], n) for n in (failed | unverified)])
    only_fail_cost, _ = split_with_cost([(ms_by_number[n], n) for n in failed])
    max_possible_savings = cur_cost - only_fail_cost
    rem_cpu = sum(ms_by_number[n] for n in unverified)
    # Profitable mainly when dropping unverified tests removes a 2s worker overhead.
    return max_possible_savings >= 1500 and rem_cpu < max_possible_savings * 0.65


def _run_child_verification(
    task: VerifyTask,
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    out_fd: int | None,
    limit_s: float,
) -> set[int]:
    """Fork an isolated child to verify candidate tests; return the pruned test-number set."""
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        if out_fd is not None:
            try:
                os.close(out_fd)
            except OSError:
                pass
        try:
            os.chdir(SOURCE)
            if str(SOURCE) not in sys.path:
                sys.path.insert(0, str(SOURCE))
            _patch_pool_references()
            _install_fail_fast_subtest()

            if task.phase == 1:
                sys.meta_path.insert(0, _OverlayFinder(task.selection.patched_files))
                _install_open_overlay(task.selection.patched_files)
            else:
                _install_open_overlay(task.selection.patched_files)
                if task.selection.hot_patch_targets:
                    if not _apply_code_hotpatch(task.selection.patched_files, task.selection.hot_patch_targets):
                        os.write(w_fd, b"FAIL_PATCH")
                        os.close(w_fd)
                        os._exit(0)
                if task.selection.attr_hot_targets:
                    if not _apply_attr_hotpatch(task.selection.attr_hot_targets):
                        os.write(w_fd, b"FAIL_PATCH")
                        os.close(w_fd)
                        os._exit(0)

            def _alarm_handler(_signum, _frame):
                raise TimeoutError("test timed out")

            signal.signal(signal.SIGALRM, _alarm_handler)

            # Execute cheapest tests first for fastest fail-fast stopping.
            ordered = sorted(task.chosen, key=lambda n: (ms_by_number[n], id_by_number[n]))
            failed: set[int] = set()
            passed: set[int] = set()
            unverified: set[int] = set(ordered)
            loaded_classes: dict[tuple[str, str], type] = {}
            setup_done: set[type] = set()
            executor_patched = "tests/test_executor.py" in task.selection.patched_files

            for num in ordered:
                tid = id_by_number[num]
                if tid in SUBPROCESS_TESTS:
                    continue
                if failed and not _should_continue_after_failure(failed, unverified, ms_by_number):
                    break

                unverified.discard(num)
                parts = tid.split("::")
                if len(parts) != 3:
                    failed.add(num)
                    continue
                file_path, cls_name, meth_name = parts
                mod_name = file_path[:-3].replace("/", ".")
                key = (mod_name, cls_name)

                ok = False
                timeout_s = max(5, int(ms_by_number[num] / 1000 * 5) + 4)
                signal.alarm(timeout_s)
                try:
                    if key not in loaded_classes:
                        mod = importlib.import_module(mod_name)
                        cls = getattr(mod, cls_name)
                        if cls_name == "TestExecutor" and not executor_patched:
                            _install_lazy_executor_setup(cls)
                        loaded_classes[key] = cls
                        if cls not in setup_done:
                            cls.setUpClass()
                            setup_done.add(cls)
                    cls = loaded_classes[key]
                    res = unittest.TestResult()
                    cls(meth_name).run(res)
                    ok = res.wasSuccessful()
                except BaseException:
                    ok = False
                finally:
                    signal.alarm(0)

                if ok:
                    passed.add(num)
                else:
                    failed.add(num)

            payload = json.dumps({"passed": sorted(passed), "failed": sorted(failed)}).encode("utf-8")
            os.write(w_fd, payload)
        except BaseException:
            pass
        finally:
            try:
                os.close(w_fd)
            except OSError:
                pass
            os._exit(0)

    # Parent: read the pipe until the child finishes or its deadline passes. A per-test
    # SIGALRM cannot interrupt a hang inside C code or a deadlock, so the parent enforces
    # a hard limit and kills the child; its commit then keeps the static selection.
    os.close(w_fd)
    chunks: list[bytes] = []
    deadline = time.monotonic() + limit_s
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        ready, _, _ = select.select([r_fd], [], [], remaining)
        if not ready:
            continue
        chunk = os.read(r_fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(r_fd)
    if timed_out:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _, status, rusage = os.wait4(pid, 0)
    if timed_out:
        log(f"{task.commit}: verify killed after {limit_s:.1f}s p{task.phase}")
        return task.chosen
    child_cpu = rusage.ru_utime + rusage.ru_stime
    raw = b"".join(chunks)
    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0 or not raw or raw == b"FAIL_PATCH":
        log(f"{task.commit}: verify fallback p{task.phase} ({child_cpu:.3f}s, status={status})")
        return task.chosen

    try:
        data = json.loads(raw)
        passed_set = set(data["passed"])
        log(
            f"{task.commit}: p{task.phase} cpu={child_cpu:.3f}s "
            f"pass={len(passed_set)} fail={len(data['failed'])}/{len(task.chosen)}"
        )
        # Soundness: only drop tests that explicitly passed; retain everything else.
        return task.chosen - passed_set
    except Exception:
        return task.chosen


def plan_verification(
    commits: list[tuple[str, set[int], Selection]],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
) -> tuple[list[VerifyTask], list[VerifyTask]]:
    """Select and prioritize commits worth runtime verification within a deterministic budget."""
    candidates: list[VerifyTask] = []
    for commit, chosen, sel in commits:
        if not chosen or not sel.patched_files or sel.reason.startswith("everything"):
            continue
        tids = [id_by_number[n] for n in chosen]
        if any(tid in SUBPROCESS_TESTS for tid in tids):
            continue
        durations = [ms_by_number[n] for n in chosen]
        max_ms = max(durations)
        total_ms = sum(durations)
        if max_ms > MAX_SINGLE_TEST_MS or total_ms > MAX_COMMIT_VERIFY_MS:
            continue
        if sel.fixture_only and total_ms > VERIFY_FIXTURE_MAX_MS:
            continue

        static_cost_ms, _ = split_with_cost([(ms_by_number[n], n) for n in chosen])
        can_phase2 = (
            ENABLE_HOTPATCH
            and sel.hot_patch_targets is not None
            and not any(tid in FRESH_IMPORT_TESTS for tid in tids)
        )
        phase = 2 if can_phase2 else 1
        overhead_ms = 30 if phase == 2 else 300
        est_cpu_ms = overhead_ms + total_ms
        if static_cost_ms - est_cpu_ms < MIN_GAIN_MS:
            continue
        candidates.append(
            VerifyTask(
                commit=commit,
                chosen=chosen,
                selection=sel,
                phase=phase,
                static_cost_ms=static_cost_ms,
                est_cpu_ms=est_cpu_ms,
            )
        )

    # Deterministically prioritize highest ROI (net savings / CPU) within total budget.
    candidates.sort(
        key=lambda t: (-(t.static_cost_ms - t.est_cpu_ms) / max(t.est_cpu_ms, 1), t.commit)
    )
    budget_used = 0
    phase1: list[VerifyTask] = []
    phase2: list[VerifyTask] = []
    for task in candidates:
        if budget_used + task.est_cpu_ms > TOTAL_VERIFY_BUDGET_MS:
            continue
        budget_used += task.est_cpu_ms
        if task.phase == 1:
            phase1.append(task)
        else:
            phase2.append(task)
    phase1.sort(key=lambda t: t.commit)
    phase2.sort(key=lambda t: t.commit)
    return phase1, phase2


def _prewarm_for_phase2(
    phase2_tasks: list[VerifyTask],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
) -> None:
    """Import sqlglot core and shared/first-run test modules once in parent before Phase-2 forks."""
    if not phase2_tasks:
        return
    os.chdir(SOURCE)
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    # Replace ProcessPoolExecutor BEFORE importing test_optimizer/test_executor so their
    # `from concurrent.futures import ProcessPoolExecutor` binds _InProcessExecutor.
    _patch_pool_references()
    import sqlglot  # noqa: F401

    mod_task_count: dict[str, int] = {}
    needed_mods: set[str] = set()
    for task in phase2_tasks:
        for rel_path, _ in task.selection.hot_patch_targets or ():
            needed_mods.add(module_name(rel_path))
        for rel_path, _, _, _ in task.selection.attr_hot_targets or ():
            needed_mods.add(module_name(rel_path))
        ordered = sorted(task.chosen, key=lambda n: (ms_by_number[n], id_by_number[n]))
        # Always pre-import the cheapest test's module (guaranteed to run first).
        first_file = id_by_number[ordered[0]].split("::", 1)[0]
        first_mod = first_file[:-3].replace("/", ".")
        if first_mod != "tests.test_executor":
            needed_mods.add(first_mod)
        task_mods = {id_by_number[n].split("::", 1)[0][:-3].replace("/", ".") for n in task.chosen}
        for mod in task_mods:
            mod_task_count[mod] = mod_task_count.get(mod, 0) + 1

    # Also pre-import any test module shared by >= 2 Phase-2 tasks to amortize import cost.
    for mod, count in mod_task_count.items():
        if count >= 2:
            needed_mods.add(mod)

    for mod_name in sorted(needed_mods):
        if mod_name not in sys.modules:
            try:
                importlib.import_module(mod_name)
            except Exception as err:
                log(f"prewarm skip {mod_name}: {err}")


def verify_batches(
    phase1: list[VerifyTask],
    phase2: list[VerifyTask],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    out_fd: int | None,
) -> dict[str, set[int]]:
    """Run Phase-1 (clean fork) then Phase-2 (pre-warmed hot-patch fork) verification."""
    verified: dict[str, set[int]] = {}
    if not hasattr(os, "fork"):
        return verified

    # Pre-import heavy third-party C extensions once in parent BEFORE Phase 1 forks
    # (independent of sqlglot, shared via COW across both Phase 1 and Phase 2).
    all_tasks = phase1 + phase2
    needs_opt = any(id_by_number[n].startswith("tests/test_optimizer.py::") for t in all_tasks for n in t.chosen)
    needs_exec = any(id_by_number[n].startswith("tests/test_executor.py::") for t in all_tasks for n in t.chosen)
    if needs_opt or needs_exec:
        for pkg in ("duckdb", "numpy", "pandas") if needs_exec else ("duckdb",):
            try:
                importlib.import_module(pkg)
            except Exception:
                pass

    stop_at = time.monotonic() + VERIFY_WALL_BUDGET_S

    def run(task: VerifyTask) -> None:
        remaining = stop_at - time.monotonic()
        if remaining <= 0:
            log(f"{task.commit}: verify skipped, wall budget spent")
            return
        limit_s = min(CHILD_LIMIT_BASE_S + CHILD_LIMIT_FACTOR * task.est_cpu_ms / 1000, remaining)
        verified[task.commit] = _run_child_verification(task, id_by_number, ms_by_number, out_fd, limit_s)

    for task in phase1:
        run(task)

    # Phase 2: pre-warm sqlglot + needed test modules once in parent, then fork + hot-patch.
    if phase2 and time.monotonic() < stop_at:
        _prewarm_for_phase2(phase2, id_by_number, ms_by_number)
        for task in phase2:
            run(task)

    return verified
