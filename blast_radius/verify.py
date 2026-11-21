"""Dual-phase forked runtime verifier with real-CPU makespan decapitation and 2-core parallelism.

Phase 1 (Clean Fork, before parent imports sqlglot):
    Verifies commits that touch class/module-level code, closures, or test files
    using an in-memory importlib MetaPathFinder + virtual open() overlay.

Phase 2 (Pre-Warmed Fork, after parent imports sqlglot once):
    Verifies commits that only modify existing non-nested function bodies via
    in-place __code__ replacement, and safe literal class-attribute updates.
    Test modules are lazily pre-imported once in the parent and shared via COW.

Soundness invariant: a candidate test is dropped ONLY if it explicitly ran and
passed inside an isolated forked child against the patched source. Any error,
crash, timeout, or unverified test is conservatively retained.
"""

from __future__ import annotations

import builtins
import concurrent.futures as cf
import gc
import importlib
import importlib.abc
import importlib.machinery
import inspect
import io
import os
import pickle
import select
import signal
import sys
import time
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field

from blast_radius import SOURCE, log
from blast_radius.impact_map import module_name
from blast_radius.schedule import split_with_cost
from blast_radius.select import Selection
from blast_radius.tpc_verify import (
    TPC_EST_CPU_MS,
    TPC_SUITE_BY_TEST_ID,
    _can_skip_tpc_optimize,
    select_tpc_mode,
    verify_generator_memory,
    verify_lazy_athena_presto,
    verify_max_nodes,
    verify_optimized_tpch,
    verify_tpc_test,
)

# Decision thresholds (overridable via env vars for local A/B benchmarking).
MAX_SINGLE_TEST_MS = int(os.environ.get("BR_MAX_SINGLE_MS", "2500"))
MAX_COMMIT_VERIFY_MS = int(os.environ.get("BR_MAX_COMMIT_MS", "2500"))
MIN_GAIN_MS = int(os.environ.get("BR_MIN_GAIN_MS", "350"))
MIN_DECAP_GAIN_MS = int(os.environ.get("BR_MIN_DECAP_GAIN_MS", "600"))
VERIFY_FIXTURE_MAX_MS = int(os.environ.get("BR_FIXTURE_MAX_MS", "0"))
TOTAL_VERIFY_BUDGET_MS = int(os.environ.get("BR_TOTAL_BUDGET_MS", "95000"))
ENABLE_HOTPATCH = os.environ.get("BR_ENABLE_HOTPATCH", "1") != "0"
ENABLE_TPC_FAST = os.environ.get("BR_ENABLE_TPC_FAST", "1") != "0"
ENABLE_TPC_FAST_B = os.environ.get("BR_ENABLE_TPC_FAST_B", "1") != "0"
TPC_PROBE_BUDGET_MS = int(os.environ.get("BR_TPC_PROBE_MS", "300"))
TPC_MIN_GAIN_MS = int(os.environ.get("BR_TPC_MIN_GAIN_MS", "1800"))
MAX_PARALLEL_CHILDREN = int(os.environ.get("BR_PARALLEL", "2"))

# Hard wall-clock limits, enforced by the parent (the run itself has 3 minutes, 4.5 in the final).
CHILD_LIMIT_BASE_S = 8.0
CHILD_LIMIT_FACTOR = 3.0
VERIFY_WALL_BUDGET_S = float(os.environ.get("BR_VERIFY_WALL_S", "130"))

OPT_TPCH_TEST_ID = "tests/test_executor.py::TestExecutor::test_optimized_tpch"
MAX_NODES_TEST_ID = "tests/test_parser.py::TestParser::test_max_nodes"
GEN_MEM_TEST_ID = "tests/test_generator.py::TestGenerator::test_generator_memory"

# Tests in TestExecutor that actually use numpy/pandas/TPC fixtures when run via real unittest.
TPC_EXECUTOR_TESTS = frozenset({
    OPT_TPCH_TEST_ID,
    "tests/test_executor.py::TestExecutor::test_execute_tpch",
    "tests/test_executor.py::TestExecutor::test_execute_tpcds",
})

# Tests that inspect fresh import state or spawn subprocesses must never run in Phase 2.
FRESH_IMPORT_TESTS = frozenset({
    "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
    "tests/test_dialect_imports.py::TestDialectImports::test_dialect_attribute_access_is_cached",
    "tests/test_dialect_imports.py::TestDialectImports::test_dialect_import_no_deadlock",
})

# Tests that spawn a fresh Python interpreter against disk cannot see an in-memory overlay;
# we keep this individual test unverified while still verifying/decapitating its sibling tests.
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
    expected_gain_ms: int = 0
    tpc_mode: str | None = None
    lazy_pp_nums: set[int] = field(default_factory=set)
    decap_targets: list[int] = field(default_factory=list)


class _LazyFuture(cf.Future):
    """Evaluates submitted work on .result() so fail-fast subtests abort before running remaining queries."""

    def __init__(self, fn, args, kwargs):
        super().__init__()
        self._work = (fn, args, kwargs)
        self._evaluated = False

    def _evaluate(self) -> None:
        if self._evaluated:
            return
        self._evaluated = True
        fn, args, kwargs = self._work
        self._work = None
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            self.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as error:
            self.set_exception(error)

    def result(self, timeout=None):
        self._evaluate()
        return super().result(timeout=timeout)


class _InProcessExecutor:
    """Replaces ProcessPoolExecutor in child so workers see in-memory patches with zero spawn cost.

    Arguments and results still go through pickle, as they would to a spawned worker:
    sqlglot expressions pickle through serde (Expression.__reduce__), so a patch that
    breaks serialization must fail here too. Lazy evaluation on .result() lets
    TestOptimizer.check_file abort on the first failing query instead of running all 500.
    """

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))
        self._futures: list[_LazyFuture] = []

    def submit(self, fn, /, *args, **kwargs):
        fut = _LazyFuture(fn, args, kwargs)
        self._futures.append(fut)
        return fut

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        futs = [self.submit(fn, *args) for args in zip(*iterables)]
        return [f.result() for f in futs]

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        if exc_type is None:
            for fut in self._futures:
                if not fut._evaluated:
                    fut._evaluate()
        self._futures.clear()
        return False


def _lazy_wait(fs, timeout=None, return_when=cf.ALL_COMPLETED):
    return set(fs), set()


def _lazy_as_completed(fs, timeout=None):
    return iter(fs)


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


def _install_lazy_data_libs() -> None:
    """Defer importing numpy and pandas until actually accessed by a test.

    Both tests/test_executor.py and tests/test_optimizer.py import pandas.testing
    at module top level, even though most tests never call assert_frame_equal.
    """
    if "numpy" in sys.modules or "pandas" in sys.modules:
        return

    stub_names = ("numpy", "pandas", "pandas.testing")

    def _load_real(name: str) -> types.ModuleType:
        for s in stub_names:
            mod = sys.modules.get(s)
            if getattr(mod, "_br_lazy_stub", False):
                sys.modules.pop(s, None)
        real = importlib.import_module(name)
        for mod_key in ("tests.test_executor", "tests.test_optimizer"):
            tmod = sys.modules.get(mod_key)
            if tmod is not None:
                if name == "numpy" and getattr(getattr(tmod, "np", None), "_br_lazy_stub", False):
                    tmod.np = real
                elif name == "pandas" and getattr(getattr(tmod, "pd", None), "_br_lazy_stub", False):
                    tmod.pd = real
                elif name == "pandas.testing":
                    tmod.assert_frame_equal = real.assert_frame_equal
        return real

    class _LazyModule(types.ModuleType):
        _br_lazy_stub = True

        def __getattr__(self, attr: str):
            return getattr(_load_real(self.__name__), attr)

    def _lazy_assert_frame_equal(*args, **kwargs):
        real_testing = _load_real("pandas.testing")
        return real_testing.assert_frame_equal(*args, **kwargs)

    np_stub = _LazyModule("numpy")
    pd_stub = _LazyModule("pandas")
    pd_testing_stub = _LazyModule("pandas.testing")
    pd_testing_stub.assert_frame_equal = _lazy_assert_frame_equal
    pd_stub.testing = pd_testing_stub
    sys.modules["numpy"] = np_stub
    sys.modules["pandas"] = pd_stub
    sys.modules["pandas.testing"] = pd_testing_stub


def _patch_pool_references() -> None:
    """Ensure concurrent.futures and imported test modules use lazy fail-fast _InProcessExecutor."""
    cf.ProcessPoolExecutor = _InProcessExecutor
    for mod_name in ("tests.test_optimizer", "tests.test_executor"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            if hasattr(mod, "ProcessPoolExecutor"):
                setattr(mod, "ProcessPoolExecutor", _InProcessExecutor)
            if mod_name == "tests.test_optimizer":
                setattr(mod, "wait", _lazy_wait)
                setattr(mod, "as_completed", _lazy_as_completed)


def _install_fail_fast_subtest() -> None:
    """Abort a test immediately on the first failing subtest (while honoring SkipTest)."""

    @contextmanager
    def _ff_subtest(self, msg=unittest.case._subtest_msg_sentinel, **params):
        try:
            yield
        except unittest.SkipTest:
            pass

    unittest.TestCase.subTest = _ff_subtest


def _can_fast_opt_tpch(sel: Selection) -> bool:
    return not any(
        p in ("tests/test_executor.py", "tests/helpers.py") or p.startswith("tests/fixtures/")
        for p in sel.patched_files
    )


def _can_fast_max_nodes(sel: Selection) -> bool:
    return "tests/test_parser.py" not in sel.patched_files


def _can_fast_gen_mem(sel: Selection) -> tuple[bool, bool]:
    """Return (can_fast, is_scoped_only)."""
    if "tests/test_generator.py" in sel.patched_files:
        return False, False
    core_touched = any(
        not p.startswith(("sqlglot/dialects/", "sqlglot/generators/", "sqlglot/parsers/", "sqlglot/typing/"))
        or p in (
            "sqlglot/dialects/dialect.py",
            "sqlglot/dialects/__init__.py",
            "sqlglot/generators/__init__.py",
            "sqlglot/parsers/base.py",
            "sqlglot/typing/__init__.py",
        )
        for p in sel.patched_files
    )
    return True, not core_touched


def _est_test_cpu_ms(
    num: int,
    tid: str,
    sel: Selection,
    tpc_mode: str | None,
    cpu_by_number: dict[int, int],
) -> int:
    """Return best realistic CPU estimate (ms) for verifying test `num` on `sel`."""
    if tpc_mode and tid in TPC_SUITE_BY_TEST_ID:
        suite = TPC_SUITE_BY_TEST_ID[tid]
        if tpc_mode == "fast_a" and _can_skip_tpc_optimize(sel.patched_files):
            return 450 if suite == "tpcds" else 120
        return TPC_EST_CPU_MS[(suite, tpc_mode)]
    if tid == OPT_TPCH_TEST_ID and _can_fast_opt_tpch(sel):
        return 100
    if tid == MAX_NODES_TEST_ID and _can_fast_max_nodes(sel):
        tok_touched = any(p in ("sqlglot/tokens.py", "sqlglot/tokenizer_core.py") for p in sel.patched_files)
        return 16500 if tok_touched else 400
    if tid == GEN_MEM_TEST_ID:
        can_fast, is_scoped = _can_fast_gen_mem(sel)
        if can_fast and is_scoped:
            return 550
    return cpu_by_number[num]


def _should_continue_after_failure(
    cur_pool: set[int],
    unverified_ordered: list[int],
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
) -> bool:
    """Return True if any prefix of `unverified_ordered` (sorted tallest-first) nets >= 80ms in an active child."""
    if not unverified_ordered:
        return False
    first = unverified_ordered[0]
    if cpu_by_number[first] >= ms_by_number[first]:
        return False
    cur_cost, _ = split_with_cost([(ms_by_number[n], n) for n in cur_pool])
    trial = set(cur_pool)
    acc_cpu = 0
    for num in unverified_ordered:
        c_ms = cpu_by_number[num]
        if c_ms >= ms_by_number[num]:
            break
        trial.discard(num)
        acc_cpu += c_ms
        new_cost, _ = split_with_cost([(ms_by_number[n], n) for n in trial])
        if (cur_cost - new_cost) - acc_cpu >= 80:
            return True
    return False


def _has_decap_prefix_roi(
    cur_pool: set[int],
    elig_rem: list[int],
    failed: set[int],
    sel: Selection,
    tpc_mode: str | None,
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
) -> bool:
    """Check if the tallest remaining decap target `elig_rem[0]` belongs to a net-profitable prefix."""
    if not elig_rem:
        return False
    num = elig_rem[0]
    t_cpu = _est_test_cpu_ms(num, id_by_number[num], sel, tpc_mode, cpu_by_number)
    cur_c, _ = split_with_cost([(ms_by_number[n], n) for n in cur_pool])
    min_req = MIN_DECAP_GAIN_MS if failed else MIN_GAIN_MS
    trial = set(cur_pool)
    acc_cpu = 0
    for cand in elig_rem:
        trial.discard(cand)
        acc_cpu += _est_test_cpu_ms(cand, id_by_number[cand], sel, tpc_mode, cpu_by_number)
        after_c, _ = split_with_cost([(ms_by_number[n], n) for n in trial])
        if (cur_c - after_c) - acc_cpu >= min_req:
            with_num_c, _ = split_with_cost([(ms_by_number[n], n) for n in (trial | {num})])
            if with_num_c - after_c > t_cpu:
                return True
    return False


def _start_child_verification(
    task: VerifyTask,
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
    out_fd: int | None,
) -> tuple[int, int]:
    """Fork an isolated child to verify candidate tests; return (pid, read_pipe_fd)."""
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid != 0:
        os.close(w_fd)
        return pid, r_fd

    os.close(r_fd)
    if out_fd is not None:
        try:
            os.close(out_fd)
        except OSError:
            pass
    try:
        gc.enable()
        os.chdir(SOURCE)
        if str(SOURCE) not in sys.path:
            sys.path.insert(0, str(SOURCE))
        _patch_pool_references()
        _install_fail_fast_subtest()

        if task.phase == 1:
            if "sqlglot" in sys.modules:
                for k in [m for m in sys.modules if m == "sqlglot" or m.startswith(("sqlglot.", "tests."))]:
                    sys.modules.pop(k, None)
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

        failed: set[int] = set()
        passed: set[int] = set()
        loaded_classes: dict[tuple[str, str], type] = {}
        setup_done: set[type] = set()
        sel = task.selection
        executor_patched = "tests/test_executor.py" in sel.patched_files
        needs_real_tpc = any(
            (
                id_by_number[n] in TPC_SUITE_BY_TEST_ID and not task.tpc_mode
            )
            or (
                id_by_number[n] == OPT_TPCH_TEST_ID and not _can_fast_opt_tpch(sel)
            )
            for n in task.chosen
        )
        if not executor_patched and not needs_real_tpc:
            _install_lazy_data_libs()

        def _run_unittest(num: int) -> bool:
            tid = id_by_number[num]
            parts = tid.split("::")
            if len(parts) != 3:
                return False
            file_path, cls_name, meth_name = parts
            mod_name = file_path[:-3].replace("/", ".")
            key = (mod_name, cls_name)

            ok = False
            est_s = max(ms_by_number[num], cpu_by_number[num]) / 1000
            timeout_s = max(5, int(est_s * 5) + 4)
            signal.alarm(timeout_s)
            try:
                if key not in loaded_classes:
                    mod = importlib.import_module(mod_name)
                    if mod_name in ("tests.test_optimizer", "tests.test_executor"):
                        _patch_pool_references()
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
            return ok

        def _run_candidate(num: int) -> bool:
            tid = id_by_number[num]
            if task.tpc_mode and tid in TPC_SUITE_BY_TEST_ID:
                signal.alarm(65)
                try:
                    return verify_tpc_test(TPC_SUITE_BY_TEST_ID[tid], task.tpc_mode, sel.patched_files)
                except BaseException:
                    return False
                finally:
                    signal.alarm(0)
            if tid == OPT_TPCH_TEST_ID and _can_fast_opt_tpch(sel):
                signal.alarm(15)
                try:
                    return verify_optimized_tpch(sel.patched_files)
                except BaseException:
                    return False
                finally:
                    signal.alarm(0)
            if tid == MAX_NODES_TEST_ID and _can_fast_max_nodes(sel):
                signal.alarm(60)
                try:
                    return verify_max_nodes(sel.patched_files)
                except BaseException:
                    return False
                finally:
                    signal.alarm(0)
            if tid == GEN_MEM_TEST_ID:
                can_fast, is_scoped = _can_fast_gen_mem(sel)
                if can_fast and is_scoped:
                    signal.alarm(25)
                    try:
                        return verify_generator_memory(sel.patched_files, scoped_only=True)
                    except BaseException:
                        return False
                    finally:
                        signal.alarm(0)
            return _run_unittest(num)

        def _mark(num: int, ok: bool) -> bool:
            """Record a verdict and stream it at once, so a timeout or crash keeps the passes so far."""
            (passed if ok else failed).add(num)
            os.write(w_fd, f"{'P' if ok else 'F'} {num}\n".encode())
            return ok

        # Step 1: If pushdown_predicates.py:36-37 lazy imports inflated `chosen`, verify clean import in ~10ms.
        if task.lazy_pp_nums:
            ok_lazy = False
            signal.alarm(10)
            try:
                ok_lazy = verify_lazy_athena_presto(sel.patched_files)
            except BaseException:
                ok_lazy = False
            finally:
                signal.alarm(0)
            if ok_lazy:
                for num in task.lazy_pp_nums:
                    _mark(num, True)

        verifiable = {
            n for n in (task.chosen - passed) if id_by_number[n] not in SUBPROCESS_TESTS
        }
        decap_set = set(task.decap_targets) & verifiable

        if decap_set:
            total_decap_cpu = sum(
                _est_test_cpu_ms(n, id_by_number[n], sel, task.tpc_mode, cpu_by_number)
                for n in decap_set
            )
            other_rem = verifiable - decap_set
            other_cpus = [cpu_by_number[n] for n in other_rem]
            will_verify_tail = (
                bool(other_rem)
                and max(other_cpus) <= MAX_SINGLE_TEST_MS
                and sum(other_cpus) <= MAX_COMMIT_VERIFY_MS
            )
            # Only run a pre-decap fail-fast probe when decap targets are expensive (>= 1.5s)
            # or when the remaining tail is small enough to verify completely.
            need_probe = total_decap_cpu >= 1500 or will_verify_tail
            patched_stems = {
                p.rsplit("/", 1)[-1].removesuffix(".py")
                for p in sel.patched_files
                if p.endswith(".py")
            } - {"__init__", "dialect", "base", "core", "helper"}
            other_ordered = sorted(
                other_rem,
                key=lambda n: (
                    0 if (not will_verify_tail and cpu_by_number[n] <= 120 and any(s in id_by_number[n] for s in patched_stems)) else 1,
                    cpu_by_number[n],
                    ms_by_number[n],
                    id_by_number[n],
                ),
            )
            probe_tests: list[int] = []
            tail_tests: list[int] = []
            probe_cpu = 0
            for num in other_ordered:
                c_ms = cpu_by_number[num]
                if need_probe and c_ms <= 450 and probe_cpu + c_ms <= TPC_PROBE_BUDGET_MS:
                    probe_tests.append(num)
                    probe_cpu += c_ms
                else:
                    tail_tests.append(num)

            # Step 2a: Fail-fast probe on cheap unit tests.
            for num in probe_tests:
                if not _mark(num, _run_candidate(num)):
                    break

            # Step 2b: Makespan & worker-collapsing decapitation ordered tallest-first so
            # larger ceilings are resolved before smaller targets beneath them are evaluated.
            decap_ordered = [n for n in task.decap_targets if n in decap_set]
            for idx, num in enumerate(decap_ordered):
                if failed and id_by_number[num] in TPC_SUITE_BY_TEST_ID and task.tpc_mode != "fast_a":
                    continue
                elig_rem = [
                    x
                    for x in decap_ordered[idx:]
                    if not (failed and id_by_number[x] in TPC_SUITE_BY_TEST_ID and task.tpc_mode != "fast_a")
                ]
                cur_pool = task.chosen - passed
                if not _has_decap_prefix_roi(
                    cur_pool,
                    elig_rem,
                    failed,
                    sel,
                    task.tpc_mode,
                    id_by_number,
                    ms_by_number,
                    cpu_by_number,
                ):
                    continue
                _mark(num, _run_candidate(num))

            # Step 2c: If no failures so far, verify remaining tail if dropping it beats its real CPU.
            if not failed and tail_tests:
                tail_cpus = [cpu_by_number[n] for n in tail_tests]
                cur_pool = task.chosen - passed
                cur_c, _ = split_with_cost([(ms_by_number[n], n) for n in cur_pool])
                after_tail_c, _ = split_with_cost(
                    [(ms_by_number[n], n) for n in (cur_pool - set(tail_tests))]
                )
                if (
                    max(tail_cpus) <= MAX_SINGLE_TEST_MS
                    and sum(tail_cpus) <= MAX_COMMIT_VERIFY_MS
                    and (cur_c - after_tail_c) - sum(tail_cpus) >= MIN_GAIN_MS
                ):
                    unverified = set(tail_tests)
                    for num in tail_tests:
                        if num not in unverified:
                            continue
                        if failed:
                            break
                        unverified.discard(num)
                        _mark(num, _run_candidate(num))
                    if failed and unverified:
                        for num in sorted(unverified, key=lambda n: (-ms_by_number[n], cpu_by_number[n], id_by_number[n])):
                            rem_list = sorted(unverified, key=lambda n: (-ms_by_number[n], cpu_by_number[n], id_by_number[n]))
                            if not _should_continue_after_failure(task.chosen - passed, rem_list, ms_by_number, cpu_by_number):
                                break
                            unverified.discard(num)
                            _mark(num, _run_candidate(num))
        elif verifiable:
            rem_cpus = [cpu_by_number[n] for n in verifiable]
            if not task.lazy_pp_nums or (
                max(rem_cpus) <= MAX_SINGLE_TEST_MS and sum(rem_cpus) <= MAX_COMMIT_VERIFY_MS
            ):
                ordered = sorted(
                    verifiable,
                    key=lambda n: (cpu_by_number[n], ms_by_number[n], id_by_number[n]),
                )
                unverified = set(ordered)
                for num in ordered:
                    if num not in unverified:
                        continue
                    if failed:
                        break
                    unverified.discard(num)
                    _mark(num, _run_candidate(num))
                if failed and unverified:
                    for num in sorted(unverified, key=lambda n: (-ms_by_number[n], cpu_by_number[n], id_by_number[n])):
                        rem_list = sorted(unverified, key=lambda n: (-ms_by_number[n], cpu_by_number[n], id_by_number[n]))
                        if not _should_continue_after_failure(task.chosen - passed, rem_list, ms_by_number, cpu_by_number):
                            break
                        unverified.discard(num)
                        _mark(num, _run_candidate(num))
    except BaseException:
        pass
    finally:
        try:
            os.close(w_fd)
        except OSError:
            pass
        os._exit(0)


def plan_verification(
    commits: list[tuple[str, set[int], Selection]],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
    number_by_index: list[int | None] | None = None,
) -> tuple[list[VerifyTask], list[VerifyTask]]:
    """Select and prioritize commits worth runtime verification or makespan decapitation."""
    candidates: list[VerifyTask] = []
    for commit, chosen, sel in commits:
        if not chosen or not sel.patched_files or sel.reason.startswith("everything"):
            continue
        tids = [id_by_number[n] for n in chosen]
        verifiable = {n for n in chosen if id_by_number[n] not in SUBPROCESS_TESTS}
        if not verifiable:
            continue

        static_cost_ms, _ = split_with_cost([(ms_by_number[n], n) for n in chosen])
        can_phase2 = (
            ENABLE_HOTPATCH
            and sel.hot_patch_targets is not None
            and not sel.lazy_pp_mask
            and not any(tid in FRESH_IMPORT_TESTS for tid in tids)
        )
        phase = 2 if can_phase2 else 1
        overhead_ms = 30 if phase == 2 else 280

        lazy_pp_nums: set[int] = set()
        if sel.lazy_pp_mask and number_by_index is not None:
            x = sel.lazy_pp_mask
            while x:
                lsb = x & -x
                idx = lsb.bit_length() - 1
                num = number_by_index[idx]
                if num is not None and num in verifiable:
                    lazy_pp_nums.add(num)
                x ^= lsb

        tpc_nums = [n for n in (verifiable - lazy_pp_nums) if id_by_number[n] in TPC_SUITE_BY_TEST_ID]
        tpc_mode: str | None = None
        if tpc_nums and ENABLE_TPC_FAST:
            tpc_mode = select_tpc_mode(sel)
            if tpc_mode is not None and not ENABLE_TPC_FAST_B and tpc_mode != "fast_a":
                tpc_mode = None

        rem_verifiable = verifiable - lazy_pp_nums

        # Identify high-ROI heavy and medium peak-shaving targets inside rem_verifiable.
        raw_decap: list[tuple[int, int, int]] = []  # (sim_ms, est_cpu_ms, num)
        for num in rem_verifiable:
            tid = id_by_number[num]
            sim_ms = ms_by_number[num]
            if tid in TPC_SUITE_BY_TEST_ID:
                if tpc_mode is None:
                    continue
                c_ms = _est_test_cpu_ms(num, tid, sel, tpc_mode, cpu_by_number)
                raw_decap.append((sim_ms, c_ms, num))
            elif tid in (OPT_TPCH_TEST_ID, MAX_NODES_TEST_ID, GEN_MEM_TEST_ID):
                c_ms = _est_test_cpu_ms(num, tid, sel, tpc_mode, cpu_by_number)
                if c_ms < sim_ms:
                    raw_decap.append((sim_ms, c_ms, num))
            elif sim_ms >= 1000:
                c_ms = cpu_by_number[num]
                if c_ms <= sim_ms * 0.78:
                    raw_decap.append((sim_ms, c_ms, num))
            elif sim_ms >= 180 and static_cost_ms >= 6000 and len(rem_verifiable) >= 10:
                c_ms = cpu_by_number[num]
                if c_ms <= sim_ms * 0.65:
                    raw_decap.append((sim_ms, c_ms, num))

        # Order decapitation targets tallest-first (-sim_ms, est_cpu_ms, num) and select
        # the greedy prefix that maximizes net makespan / worker-count reduction.
        raw_decap.sort(key=lambda item: (-item[0], item[1], item[2]))
        chosen_decap: list[int] = []
        cur_pool = chosen - lazy_pp_nums
        cur_cost, _ = split_with_cost([(ms_by_number[n], n) for n in cur_pool])
        acc_decap_cpu = 0
        best_net = (static_cost_ms - cur_cost) - (15 if lazy_pp_nums else 0)
        best_prefix: list[int] = []
        trial_pool = set(cur_pool)
        for _sim_ms, c_ms, num in raw_decap:
            trial_pool.discard(num)
            acc_decap_cpu += c_ms
            new_cost, _ = split_with_cost([(ms_by_number[n], n) for n in trial_pool])
            net = (static_cost_ms - new_cost) - acc_decap_cpu - (15 if lazy_pp_nums else 0)
            chosen_decap.append(num)
            if net >= best_net + 250:
                best_net = net
                best_prefix = list(chosen_decap)

        # Keep selected decap targets ordered tallest-first so dominant ceilings are knocked down first.
        decap_targets = list(best_prefix)
        decap_set = set(decap_targets)
        tail_nums = rem_verifiable - decap_set
        tail_cpus = [cpu_by_number[n] for n in tail_nums]

        if decap_targets or lazy_pp_nums:
            decap_cpu = sum(
                _est_test_cpu_ms(n, id_by_number[n], sel, tpc_mode, cpu_by_number)
                for n in decap_targets
            ) + (15 if lazy_pp_nums else 0)
            pool_after_decap = chosen - lazy_pp_nums - decap_set
            cost_after_decap, _ = split_with_cost([(ms_by_number[n], n) for n in pool_after_decap])
            pool_after_all = pool_after_decap - tail_nums
            cost_after_all, _ = split_with_cost([(ms_by_number[n], n) for n in pool_after_all])

            can_verify_tail = (
                bool(tail_nums)
                and not (sel.fixture_only and sum(tail_cpus) > VERIFY_FIXTURE_MAX_MS)
                and max(tail_cpus) <= MAX_SINGLE_TEST_MS
                and sum(tail_cpus) <= MAX_COMMIT_VERIFY_MS
                and (cost_after_decap - cost_after_all) - sum(tail_cpus) >= MIN_GAIN_MS
            )
            if can_verify_tail:
                est_cpu_ms = overhead_ms + decap_cpu + sum(tail_cpus)
                expected_gain_ms = (static_cost_ms - cost_after_all) - est_cpu_ms
            else:
                probe_ms = min(sum(tail_cpus), TPC_PROBE_BUDGET_MS) if decap_targets else 0
                est_cpu_ms = overhead_ms + probe_ms + decap_cpu
                expected_gain_ms = (static_cost_ms - cost_after_decap) - est_cpu_ms

            has_heavy_tpc_or_max = any(
                id_by_number[n] in TPC_SUITE_BY_TEST_ID or id_by_number[n] == MAX_NODES_TEST_ID
                for n in decap_targets
            )
            min_req = TPC_MIN_GAIN_MS if has_heavy_tpc_or_max else MIN_DECAP_GAIN_MS
            if lazy_pp_nums and not decap_targets:
                min_req = MIN_GAIN_MS
            if expected_gain_ms < min_req:
                # Fallback: check if the whole commit qualifies as a small normal commit without decap.
                if not tail_nums or max(tail_cpus) > MAX_SINGLE_TEST_MS or sum(tail_cpus) > MAX_COMMIT_VERIFY_MS:
                    continue
                all_cpus = [cpu_by_number[n] for n in verifiable]
                if max(all_cpus) > MAX_SINGLE_TEST_MS or sum(all_cpus) > MAX_COMMIT_VERIFY_MS:
                    continue
                if sel.fixture_only and sum(all_cpus) > VERIFY_FIXTURE_MAX_MS:
                    continue
                decap_targets = []
                est_cpu_ms = overhead_ms + sum(all_cpus)
                after_c, _ = split_with_cost([(ms_by_number[n], n) for n in (chosen - verifiable)])
                expected_gain_ms = (static_cost_ms - after_c) - est_cpu_ms
                if expected_gain_ms < MIN_GAIN_MS:
                    continue
        else:
            all_cpus = [cpu_by_number[n] for n in verifiable]
            if max(all_cpus) > MAX_SINGLE_TEST_MS or sum(all_cpus) > MAX_COMMIT_VERIFY_MS:
                continue
            if sel.fixture_only and sum(all_cpus) > VERIFY_FIXTURE_MAX_MS:
                continue
            est_cpu_ms = overhead_ms + sum(all_cpus)
            after_c, _ = split_with_cost([(ms_by_number[n], n) for n in (chosen - verifiable)])
            expected_gain_ms = (static_cost_ms - after_c) - est_cpu_ms
            if expected_gain_ms < MIN_GAIN_MS:
                continue

        candidates.append(
            VerifyTask(
                commit=commit,
                chosen=chosen,
                selection=sel,
                phase=phase,
                static_cost_ms=static_cost_ms,
                est_cpu_ms=est_cpu_ms,
                expected_gain_ms=expected_gain_ms,
                tpc_mode=tpc_mode,
                lazy_pp_nums=lazy_pp_nums,
                decap_targets=decap_targets,
            )
        )

    # Deterministically prioritize highest ROI (net savings / CPU) within total budget.
    candidates.sort(
        key=lambda t: (-t.expected_gain_ms / max(t.est_cpu_ms, 1), t.commit)
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
    cpu_by_number: dict[int, int],
) -> None:
    """Import sqlglot core and shared/first-run test modules once in parent before Phase-2 forks."""
    if not phase2_tasks:
        return
    os.chdir(SOURCE)
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    _patch_pool_references()
    import sqlglot  # noqa: F401

    mod_task_count: dict[str, int] = {}
    needed_mods: set[str] = set()
    for task in phase2_tasks:
        for rel_path, _ in task.selection.hot_patch_targets or ():
            needed_mods.add(module_name(rel_path))
        for rel_path, _, _, _ in task.selection.attr_hot_targets or ():
            needed_mods.add(module_name(rel_path))
        verifiable = [n for n in task.chosen if id_by_number[n] not in SUBPROCESS_TESTS]
        if not verifiable:
            continue
        ordered = sorted(verifiable, key=lambda n: (cpu_by_number[n], ms_by_number[n], id_by_number[n]))
        first_file = id_by_number[ordered[0]].split("::", 1)[0]
        first_mod = first_file[:-3].replace("/", ".")
        if first_mod != "tests.test_executor":
            needed_mods.add(first_mod)
        task_mods = {id_by_number[n].split("::", 1)[0][:-3].replace("/", ".") for n in verifiable}
        for mod in task_mods:
            mod_task_count[mod] = mod_task_count.get(mod, 0) + 1

    for mod, count in mod_task_count.items():
        if count >= 2:
            needed_mods.add(mod)

    # Install lazy numpy/pandas stubs BEFORE importing tests.test_optimizer or tests.test_executor!
    if ("tests.test_executor" in needed_mods or "tests.test_optimizer" in needed_mods) and not any(
        (id_by_number[n] in TPC_SUITE_BY_TEST_ID and not t.tpc_mode)
        or (id_by_number[n] == OPT_TPCH_TEST_ID and not _can_fast_opt_tpch(t.selection))
        for t in phase2_tasks
        for n in t.chosen
    ):
        _install_lazy_data_libs()

    for mod_name in sorted(needed_mods):
        if mod_name not in sys.modules:
            try:
                importlib.import_module(mod_name)
                if mod_name in ("tests.test_optimizer", "tests.test_executor"):
                    _patch_pool_references()
            except Exception as err:
                log(f"prewarm skip {mod_name}: {err}")


def _run_batch_parallel(
    tasks: list[VerifyTask],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
    out_fd: int | None,
    stop_at: float,
    verified: dict[str, set[int]],
) -> None:
    """Run a batch of independent forked verification tasks with up to MAX_PARALLEL_CHILDREN concurrency."""
    queue = list(tasks)
    # active: r_fd -> (pid, task, deadline, chunks_list)
    active: dict[int, tuple[int, VerifyTask, float, list[bytes]]] = {}

    def finish_child(r_fd: int, timed_out: bool) -> None:
        pid, task, limit_deadline, chunks = active.pop(r_fd)
        if timed_out:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Drain what the child streamed before it exited or was killed: each "P n" line is a
        # test that ran to completion and passed, so it stays valid after a kill or crash.
        try:
            while chunk := os.read(r_fd, 65536):
                chunks.append(chunk)
        except OSError:
            pass
        os.close(r_fd)
        _, status, rusage = os.wait4(pid, 0)
        child_cpu = rusage.ru_utime + rusage.ru_stime
        raw = b"".join(chunks)
        if raw.startswith(b"FAIL_PATCH"):
            log(f"{task.commit}: verify fallback p{task.phase} ({child_cpu:.3f}s, patch not applied)")
            verified[task.commit] = task.chosen
            return
        lines = raw.split(b"\n")
        passed_set = {int(line[2:]) for line in lines if line.startswith(b"P ")}
        n_failed = sum(1 for line in lines if line.startswith(b"F "))
        mtag = f" tpc={task.tpc_mode}" if task.tpc_mode else ""
        dtag = f" decap={len(task.decap_targets)}" if task.decap_targets else ""
        end = " TIMEOUT" if timed_out else ("" if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else f" status={status}")
        log(
            f"{task.commit}: p{task.phase}{mtag}{dtag} cpu={child_cpu:.3f}s "
            f"pass={len(passed_set)} fail={n_failed}/{len(task.chosen)}{end}"
        )
        verified[task.commit] = task.chosen - passed_set

    while queue or active:
        # Run 1 child at a time initially to avoid SMT cache contention (+0.7s CPU);
        # automatically scale to 2-core parallelism if wall-clock elapsed exceeds 35s.
        cur_limit = max(1, MAX_PARALLEL_CHILDREN) if (stop_at - time.monotonic() < VERIFY_WALL_BUDGET_S - 35.0) else 1
        while queue and len(active) < cur_limit:
            now = time.monotonic()
            remaining = stop_at - now
            if remaining <= 0:
                for skipped in queue:
                    log(f"{skipped.commit}: verify skipped, wall budget spent")
                queue.clear()
                break
            task = queue.pop(0)
            limit_s = min(CHILD_LIMIT_BASE_S + CHILD_LIMIT_FACTOR * task.est_cpu_ms / 1000, remaining)
            pid, r_fd = _start_child_verification(task, id_by_number, ms_by_number, cpu_by_number, out_fd)
            active[r_fd] = (pid, task, time.monotonic() + limit_s, [])

        if not active:
            break

        now = time.monotonic()
        expired = [r_fd for r_fd, (_, _, dl, _) in active.items() if now >= dl]
        for r_fd in expired:
            finish_child(r_fd, timed_out=True)
        if not active:
            continue

        min_rem = min(dl - time.monotonic() for _, _, dl, _ in active.values())
        if min_rem <= 0:
            continue
        ready, _, _ = select.select(list(active.keys()), [], [], min_rem)
        for r_fd in ready:
            if r_fd not in active:
                continue
            try:
                chunk = os.read(r_fd, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                finish_child(r_fd, timed_out=False)
            else:
                active[r_fd][3].append(chunk)


def verify_batches(
    phase1: list[VerifyTask],
    phase2: list[VerifyTask],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    out_fd: int | None,
    cpu_by_number: dict[int, int] | None = None,
) -> dict[str, set[int]]:
    """Run Phase-1 (clean fork) then Phase-2 (pre-warmed hot-patch fork) verification."""
    verified: dict[str, set[int]] = {}
    if not hasattr(os, "fork"):
        return verified
    if cpu_by_number is None:
        cpu_by_number = ms_by_number

    all_tasks = phase1 + phase2
    if not all_tasks:
        return verified

    # Only pre-import duckdb in parent if a scheduled task will actually run test_optimizer
    # or real-unittest test_executor tests (fast cached TPC / opt_tpch never use duckdb).
    needs_duckdb = any(
        id_by_number[n].startswith("tests/test_optimizer.py::")
        or (
            id_by_number[n].startswith("tests/test_executor.py::")
            and not (t.tpc_mode and id_by_number[n] in TPC_SUITE_BY_TEST_ID)
            and not (id_by_number[n] == OPT_TPCH_TEST_ID and _can_fast_opt_tpch(t.selection))
        )
        for t in all_tasks
        for n in t.chosen
    )
    needs_real_tpc = any(
        (id_by_number[n] in TPC_SUITE_BY_TEST_ID and not t.tpc_mode)
        or (id_by_number[n] == OPT_TPCH_TEST_ID and not _can_fast_opt_tpch(t.selection))
        for t in all_tasks
        for n in t.chosen
    )
    if needs_duckdb:
        for pkg in ("duckdb", "numpy", "pandas") if needs_real_tpc else ("duckdb",):
            try:
                importlib.import_module(pkg)
            except Exception:
                pass

    stop_at = time.monotonic() + VERIFY_WALL_BUDGET_S

    if phase1:
        _run_batch_parallel(phase1, id_by_number, ms_by_number, cpu_by_number, out_fd, stop_at, verified)

    if phase2 and time.monotonic() < stop_at:
        _prewarm_for_phase2(phase2, id_by_number, ms_by_number, cpu_by_number)
        _run_batch_parallel(phase2, id_by_number, ms_by_number, cpu_by_number, out_fd, stop_at, verified)

    return verified
