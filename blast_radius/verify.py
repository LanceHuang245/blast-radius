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
import marshal
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
    est_tpc_fast_a_ms,
    est_tpc_mode_ms,
    select_tpc_mode,
    select_tpcds_ckpt,
    select_tpcds_span,
    verify_generator_memory,
    verify_lazy_athena_presto,
    verify_max_nodes,
    verify_opt_tpcds,
    verify_optimized_tpch,
    verify_tpc_test,
)

# Decision thresholds (overridable via env vars for local A/B benchmarking).
MAX_SINGLE_TEST_MS = int(os.environ.get("BR_MAX_SINGLE_MS", "6000"))
MAX_COMMIT_VERIFY_MS = int(os.environ.get("BR_MAX_COMMIT_MS", "6000"))
MIN_GAIN_MS = int(os.environ.get("BR_MIN_GAIN_MS", "200"))
MIN_DECAP_GAIN_MS = int(os.environ.get("BR_MIN_DECAP_GAIN_MS", "400"))
MIN_ROI_RATIO = float(os.environ.get("BR_MIN_ROI_RATIO", "1.12"))
VERIFY_FIXTURE_MAX_MS = int(os.environ.get("BR_FIXTURE_MAX_MS", "0"))
TOTAL_VERIFY_BUDGET_MS = int(os.environ.get("BR_TOTAL_BUDGET_MS", "85000"))
ENABLE_HOTPATCH = os.environ.get("BR_ENABLE_HOTPATCH", "1") != "0"
ENABLE_TPC_FAST = os.environ.get("BR_ENABLE_TPC_FAST", "1") != "0"
ENABLE_TPC_FAST_B = os.environ.get("BR_ENABLE_TPC_FAST_B", "1") != "0"
TPC_PROBE_BUDGET_MS = int(os.environ.get("BR_TPC_PROBE_MS", "300"))
TPC_MIN_GAIN_MS = int(os.environ.get("BR_TPC_MIN_GAIN_MS", "1800"))
MAX_PARALLEL_CHILDREN = int(os.environ.get("BR_PARALLEL", "2"))
# Safety factor on the dedicated verifiers' (TPC, max_nodes, ...) CPU estimates; per-test
# estimates get BR_CPU_SCALE in __main__ (practice showed those too optimistic: c624afb spent
# +35 s CPU for -22 s simulated).
SPECIAL_CPU_SCALE = float(os.environ.get("BR_SPECIAL_CPU_SCALE", "1.0"))

# Hard wall-clock limits, enforced by the parent (the run itself has 3 minutes, 4.5 in the final).
CHILD_LIMIT_BASE_S = 8.0
CHILD_LIMIT_FACTOR = 3.0
VERIFY_WALL_BUDGET_S = float(os.environ.get("BR_VERIFY_WALL_S", "130"))

OPT_TPCH_TEST_ID = "tests/test_executor.py::TestExecutor::test_optimized_tpch"
OPT_TPCDS_TEST_ID = "tests/test_optimizer.py::TestOptimizer::test_tpcds"
MAX_NODES_TEST_ID = "tests/test_parser.py::TestParser::test_max_nodes"
GEN_MEM_TEST_ID = "tests/test_generator.py::TestGenerator::test_generator_memory"
LAZY_LOAD_TEST_ID = "tests/dialects/test_dialect.py::TestDialect::test_lazy_load"

# Tests in TestExecutor that actually use numpy/pandas/TPC fixtures when run via real unittest.
TPC_EXECUTOR_TESTS = frozenset({
    OPT_TPCH_TEST_ID,
    "tests/test_executor.py::TestExecutor::test_execute_tpch",
    "tests/test_executor.py::TestExecutor::test_execute_tpcds",
})

# Tests that inspect fresh import state must never run in Phase 2.
FRESH_IMPORT_TESTS = frozenset({
    LAZY_LOAD_TEST_ID,
    "tests/test_dialect_imports.py::TestDialectImports::test_dialect_attribute_access_is_cached",
    "tests/test_dialect_imports.py::TestDialectImports::test_dialect_import_no_deadlock",
})

# test_lazy_load is verified in <1ms at the start of Phase 1 before any dialects load.
SUBPROCESS_TESTS: frozenset[str] = frozenset()

# Skip redundant pickle round-trips in _LazyFuture when serde/expressions.core are untouched.
_SKIP_POOL_PICKLE = False

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
    # Fine-grained case/subTest/Validator/TPC filters and their scaled CPU estimates.
    case_filter: dict[int, set[tuple[str, int]]] = field(default_factory=dict)
    subtest_filter: dict[int, tuple[int, int]] = field(default_factory=dict)
    validator_filter: dict[int, tuple[int, int]] = field(default_factory=dict)
    tpc_query_mask: dict[int, int] = field(default_factory=dict)
    tpc_s2b_query_mask: dict[int, int] = field(default_factory=dict)
    cpu_override: dict[int, int] = field(default_factory=dict)


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
            if _SKIP_POOL_PICKLE:
                self.set_result(fn(*args, **kwargs))
            else:
                fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
                self.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as error:
            self.set_exception(error)

    def result(self, timeout=None):
        self._evaluate()
        return super().result(timeout=timeout)


class _InProcessExecutor:
    """Replaces ProcessPoolExecutor in child so workers see in-memory patches with zero spawn cost.

    When serde/expressions.core are patched, arguments and results still round-trip through
    pickle so broken Expression.__reduce__ fails here too. Lazy evaluation on .result() lets
    TestOptimizer.check_file abort on the first failing query instead of running all 500.
    """

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        if initializer is not None:
            initializer(*(initargs if _SKIP_POOL_PICKLE else pickle.loads(pickle.dumps(initargs))))
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


_IMPORT_DIGESTS: dict[str, str] | None = None


def import_state_digests(patched_files: dict[str, str], names: list[str]) -> dict[str, str | None]:
    """Import every sqlglot module with the patch in a forked child and digest `names`
    (see blast_radius.import_state). Empty if the child fails."""
    gc.freeze()
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        payload = marshal.dumps({})
        try:
            signal.alarm(30)
            os.chdir(SOURCE)
            sys.path.insert(0, str(SOURCE))
            sys.meta_path.insert(0, _OverlayFinder(patched_files))
            _install_open_overlay(patched_files)
            from blast_radius.import_state import import_all_sqlglot, module_digest

            # Import everything, as the build did: importing only the owners builds different
            # state for some of them (generators.duckdb), which would hide every narrowing.
            import_all_sqlglot()
            payload = marshal.dumps({name: module_digest(name) for name in names})
        except BaseException:
            pass
        finally:
            os.write(w_fd, payload)
            os._exit(0)
    os.close(w_fd)
    chunks = []
    while chunk := os.read(r_fd, 65536):
        chunks.append(chunk)
    os.close(r_fd)
    os.waitpid(pid, 0)
    return marshal.loads(b"".join(chunks))


def check_import_state(patched_files: dict[str, str], owners: list[str]) -> bool:
    """True iff every owner's import-time state digest under the patch equals the pinned one."""
    global _IMPORT_DIGESTS
    if _IMPORT_DIGESTS is None:
        from blast_radius.import_state import STATE_FILE

        _IMPORT_DIGESTS = marshal.loads(STATE_FILE.read_bytes())["digests"]
    got = import_state_digests(patched_files, owners)
    return all(name in _IMPORT_DIGESTS and got.get(name) == _IMPORT_DIGESTS[name] for name in owners)


def _find_unique_code(parent_code: types.CodeType, steps: list[str]) -> types.CodeType | None:
    """Walk nested co_consts to locate the unique compiled CodeType for `qualname`."""
    cur = parent_code
    for depth, name in enumerate(steps):
        matches = [c for c in cur.co_consts if isinstance(c, types.CodeType) and c.co_name == name]
        if not matches or (len(matches) > 1 and depth < len(steps) - 1):
            return None
        # Several defs of the function itself are @overload stubs plus the real def, which comes
        # last and is the one bound at runtime (the selector only allows that shape).
        cur = max(matches, key=lambda c: c.co_firstlineno)
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


def _install_lazy_optimizer_setup(cls: type) -> None:
    """Defer TestOptimizer's DuckDB table creation until a test actually accesses self.conn."""
    if getattr(cls, "_br_lazy_opt_setup", False):
        return
    orig_cm = cls.__dict__.get("setUpClass")
    if not isinstance(orig_cm, classmethod):
        return
    cls._br_lazy_opt_setup = True
    real_setup = orig_cm.__func__
    conn_initialized = [False]

    def ensure_conn() -> None:
        if conn_initialized[0]:
            return
        conn_initialized[0] = True
        if "conn" in cls.__dict__:
            delattr(cls, "conn")
        real_setup(cls)

    class _LazyConn:
        def __get__(self, obj, objtype=None):
            ensure_conn()
            return getattr(cls, "conn")

    def light_setup(c) -> None:
        import sqlglot
        from sqlglot.schema import MappingSchema

        c.pool = _InProcessExecutor()
        sqlglot.schema = MappingSchema()

    cls.setUpClass = classmethod(light_setup)
    setattr(cls, "conn", _LazyConn())


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


_CASE_STATE: dict = {
    "allowed": None,
    "in_check": False,
    "subtest_mask": None,
    "subtest_stride": 1,
    "subtest_idx": 0,
    "val_mask": None,
    "val_stride": 1,
    "val_idx": 0,
    "val_depth": 0,
}


def _br_subtest_ok() -> bool:
    mask = _CASE_STATE["subtest_mask"]
    if mask is None:
        return True
    idx = _CASE_STATE["subtest_idx"]
    _CASE_STATE["subtest_idx"] = idx + 1
    return bool(mask >> (idx // _CASE_STATE["subtest_stride"]) & 1)


def _install_case_filter(mod) -> None:
    """While a fixture-mapped test runs, let TestOptimizer.check_file see only the allowed
    (fixture file, index) cases (see blast_radius.precompile_cases)."""
    if getattr(mod, "_br_case_filter", False):
        return
    mod._br_case_filter = True
    orig_loader = mod.load_sql_fixture_pairs

    def loader(filename):
        allowed = _CASE_STATE["allowed"] if _CASE_STATE["in_check"] else None
        for index, item in enumerate(orig_loader(filename)):
            if allowed is None or (filename, index) in allowed:
                yield item

    mod.load_sql_fixture_pairs = loader
    orig_check_file = mod.TestOptimizer.check_file

    def check_file(self, *args, **kwargs):
        _CASE_STATE["in_check"] = True
        try:
            return orig_check_file(self, *args, **kwargs)
        finally:
            _CASE_STATE["in_check"] = False

    mod.TestOptimizer.check_file = check_file


def _install_subtest_filter(mod, cls: type, meth_name: str) -> None:
    """Lazily wrap `cls.<meth_name>` so its top-level `with self.subTest(...):` blocks can filter by bitmask."""
    fn = cls.__dict__.get(meth_name)
    if fn is not None and not getattr(fn, "_br_subtest_wrapped", False):
        from blast_radius.precompile_cases import transform_subtest_method

        transform_subtest_method(mod, cls, meth_name, _br_subtest_ok)


def _install_validator_filter() -> None:
    """Wrap Validator.validate_identity/validate_all/validate_transpile once to skip unreached calls."""
    from tests.dialects.test_dialect import Validator

    if getattr(Validator, "_br_val_filter", False):
        return
    Validator._br_val_filter = True
    for attr in ("validate_identity", "validate_all", "validate_transpile"):
        orig = getattr(Validator, attr)

        def make_wrapped(fn):
            def wrapped(self, *args, **kwargs):
                mask = _CASE_STATE["val_mask"]
                if mask is None or _CASE_STATE["val_depth"] > 0:
                    return fn(self, *args, **kwargs)
                idx = _CASE_STATE["val_idx"]
                _CASE_STATE["val_idx"] = idx + 1
                if not (mask >> (idx // _CASE_STATE["val_stride"]) & 1):
                    return None
                _CASE_STATE["val_depth"] += 1
                try:
                    return fn(self, *args, **kwargs)
                finally:
                    _CASE_STATE["val_depth"] -= 1

            return wrapped

        setattr(Validator, attr, make_wrapped(orig))


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
    tpc_query_mask: dict[int, int] | None = None,
    tpc_s2b_query_mask: dict[int, int] | None = None,
) -> int:
    """Return the CPU estimate (ms) for verifying test `num` on `sel`, safety factor included."""
    special = _special_cpu_ms(num, tid, sel, tpc_mode, tpc_query_mask, tpc_s2b_query_mask)
    # cpu_by_number already carries BR_CPU_SCALE (applied in __main__); dedicated verifiers'
    # estimates are measured end to end and use their own factor.
    return cpu_by_number[num] if special is None else round(special * SPECIAL_CPU_SCALE)


def _special_cpu_ms(
    num: int,
    tid: str,
    sel: Selection,
    tpc_mode: str | None,
    tpc_query_mask: dict[int, int] | None,
    tpc_s2b_query_mask: dict[int, int] | None = None,
) -> int | None:
    """Raw CPU estimate for tests with a dedicated fast verifier, else None."""
    if tid == LAZY_LOAD_TEST_ID and "tests/dialects/test_dialect.py" not in sel.patched_files:
        return 5
    if tpc_mode and tid in TPC_SUITE_BY_TEST_ID:
        suite = TPC_SUITE_BY_TEST_ID[tid]
        q_mask = tpc_query_mask.get(num) if tpc_query_mask is not None else None
        s2b_mask = tpc_s2b_query_mask.get(num) if tpc_s2b_query_mask is not None else None
        total_q = 99 if suite == "tpcds" else 22
        return est_tpc_mode_ms(
            suite,
            tpc_mode,
            sel.patched_files,
            q_mask.bit_count() if q_mask is not None else None,
            s2b_mask.bit_count() if s2b_mask is not None else None,
            total_q,
            edited_spans=sel.edited_spans,
            s2b_mask=s2b_mask,
            s1_mask=q_mask,
        )
    if tid == OPT_TPCH_TEST_ID and _can_fast_opt_tpch(sel):
        return 100
    if tid == MAX_NODES_TEST_ID and _can_fast_max_nodes(sel):
        tok_touched = any(p in ("sqlglot/tokens.py", "sqlglot/tokenizer_core.py") for p in sel.patched_files)
        return 16500 if tok_touched else 400
    if tid == GEN_MEM_TEST_ID:
        can_fast, is_scoped = _can_fast_gen_mem(sel)
        if can_fast and is_scoped:
            return 550
    return None


def _lb_cost(total: int, biggest: int, count: int) -> int:
    """Lower bound of a commit's cost for tests summing `total` ms (largest `biggest`)."""
    if count <= 0 or total <= 0:
        return 0
    return min(max(biggest, -(-total // k)) + 2000 * k for k in range(1, min(8, count) + 1))


def _best_drop_prefix(
    pool: set[int],
    candidates,
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
    min_gain: int,
    limit: int = 200,
    id_by_number: dict[int, str] | None = None,
    per_mod_ms: int = 0,
    min_roi: float = MIN_ROI_RATIO,
) -> list[int]:
    """Candidates worth verifying to drop from `pool`: the prefix, cheapest CPU per saved ms
    first, with the best net (cost saved minus CPU spent), if net >= min_gain and saved >= spent * min_roi."""
    cand_set = set(candidates)
    if not cand_set:
        return []
    # When pool cannot be emptied to 0 workers, 0ms tests can never reduce simulated cost.
    can_empty = pool <= cand_set
    if not can_empty:
        cand_list = [n for n in cand_set if ms_by_number[n] > 0]
    else:
        cand_list = list(cand_set)
    if not cand_list:
        return []
    order = sorted(
        cand_list,
        key=lambda n: (
            1 if ms_by_number[n] == 0 else 0,
            cpu_by_number[n] / max(ms_by_number[n], 1),
            -ms_by_number[n],
            n,
        ),
    )
    desc = sorted(pool, key=lambda n: -ms_by_number[n])
    total = sum(ms_by_number[n] for n in pool)
    count = len(pool)
    current = _lb_cost(total, ms_by_number[desc[0]] if desc else 0, count)
    removed: set[int] = set()
    seen_mods: set[str] = set()
    best_net, best, prefix, spent, top = min_gain - 1, [], [], 0, 0
    for num in order[:limit]:
        removed.add(num)
        prefix.append(num)
        total -= ms_by_number[num]
        count -= 1
        spent += cpu_by_number[num]
        if per_mod_ms and id_by_number is not None:
            mod_file = id_by_number[num].split("::", 1)[0]
            if mod_file not in seen_mods:
                seen_mods.add(mod_file)
                spent += per_mod_ms
        while top < len(desc) and desc[top] in removed:
            top += 1
        saved = current - _lb_cost(total, ms_by_number[desc[top]] if top < len(desc) else 0, count)
        net = saved - spent
        if net > best_net and saved >= spent * min_roi:
            best_net, best = net, list(prefix)
    return best


def _has_decap_prefix_roi(
    cur_pool: set[int],
    elig_rem: list[int],
    failed: set[int],
    sel: Selection,
    tpc_mode: str | None,
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
    tpc_query_mask: dict[int, int] | None = None,
    tpc_s2b_query_mask: dict[int, int] | None = None,
) -> bool:
    """Check if the tallest remaining decap target `elig_rem[0]` belongs to a net-profitable prefix."""
    if not elig_rem:
        return False
    num = elig_rem[0]
    t_cpu = _est_test_cpu_ms(num, id_by_number[num], sel, tpc_mode, cpu_by_number, tpc_query_mask, tpc_s2b_query_mask)
    cur_c, _ = split_with_cost([(ms_by_number[n], n) for n in cur_pool])
    min_req = MIN_DECAP_GAIN_MS if failed else MIN_GAIN_MS
    trial = set(cur_pool)
    acc_cpu = 0
    for cand in elig_rem:
        trial.discard(cand)
        acc_cpu += _est_test_cpu_ms(
            cand, id_by_number[cand], sel, tpc_mode, cpu_by_number, tpc_query_mask, tpc_s2b_query_mask
        )
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
        global _SKIP_POOL_PICKLE
        gc.enable()
        cpu_by_number = {**cpu_by_number, **task.cpu_override}
        _SKIP_POOL_PICKLE = not any(
            p in ("sqlglot/serde.py", "sqlglot/expressions/core.py") for p in task.selection.patched_files
        )
        os.chdir(SOURCE)
        if str(SOURCE) not in sys.path:
            sys.path.insert(0, str(SOURCE))
        _patch_pool_references()
        _install_fail_fast_subtest()

        lazy_load_ok: bool | None = None
        if task.phase == 1:
            if "sqlglot" in sys.modules:
                for k in [m for m in sys.modules if m == "sqlglot" or m.startswith(("sqlglot.", "tests."))]:
                    sys.modules.pop(k, None)
            sys.meta_path.insert(0, _OverlayFinder(task.selection.patched_files))
            _install_open_overlay(task.selection.patched_files)
            if (
                "tests/dialects/test_dialect.py" not in task.selection.patched_files
                and any(id_by_number[n] == LAZY_LOAD_TEST_ID for n in task.chosen)
            ):
                try:
                    import sqlglot as _sg

                    # Check both lazy dialect loading AND that test_dialect.py imports cleanly
                    # (collection fails in pytest if a dialect imported by test_dialect crashes).
                    lazy_ok = len(_sg.Dialect._classes) == 1
                    if lazy_ok:
                        importlib.import_module("tests.dialects.test_dialect")
                    lazy_load_ok = lazy_ok
                except BaseException:
                    lazy_load_ok = False
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
        loaded_mods: dict[str, types.ModuleType] = {}
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
                    loaded_mods[mod_name] = mod
                    if mod_name in ("tests.test_optimizer", "tests.test_executor"):
                        _patch_pool_references()
                    if mod_name == "tests.test_optimizer":
                        _install_case_filter(mod)
                    cls = getattr(mod, cls_name)
                    if cls_name == "TestExecutor" and not executor_patched:
                        _install_lazy_executor_setup(cls)
                    elif cls_name == "TestOptimizer" and "tests/test_optimizer.py" not in sel.patched_files:
                        _install_lazy_optimizer_setup(cls)
                    loaded_classes[key] = cls
                    if cls not in setup_done:
                        cls.setUpClass()
                        setup_done.add(cls)
                cls = loaded_classes[key]
                sub_spec = task.subtest_filter.get(num)
                val_spec = task.validator_filter.get(num)
                if sub_spec is not None:
                    _install_subtest_filter(loaded_mods[mod_name], cls, meth_name)
                if val_spec is not None:
                    _install_validator_filter()
                res = unittest.TestResult()
                _CASE_STATE["allowed"] = task.case_filter.get(num)
                if sub_spec is not None:
                    _CASE_STATE["subtest_mask"], _CASE_STATE["subtest_stride"] = sub_spec
                else:
                    _CASE_STATE["subtest_mask"], _CASE_STATE["subtest_stride"] = None, 1
                _CASE_STATE["subtest_idx"] = 0
                if val_spec is not None:
                    _CASE_STATE["val_mask"], _CASE_STATE["val_stride"] = val_spec
                else:
                    _CASE_STATE["val_mask"], _CASE_STATE["val_stride"] = None, 1
                _CASE_STATE["val_idx"] = 0
                _CASE_STATE["val_depth"] = 0
                cls(meth_name).run(res)
                ok = res.wasSuccessful()
            except BaseException:
                ok = False
            finally:
                _CASE_STATE["allowed"] = None
                _CASE_STATE["subtest_mask"] = None
                _CASE_STATE["val_mask"] = None
                _CASE_STATE["val_depth"] = 0
                signal.alarm(0)
            return ok

        def _run_candidate(num: int) -> bool:
            tid = id_by_number[num]
            if tid == LAZY_LOAD_TEST_ID and "tests/dialects/test_dialect.py" not in sel.patched_files:
                # The test only imports sqlglot in a subprocess. Phase 1 checked that import (plus
                # test_dialect collection import) with the patch applied; Phase 2 runs only when no
                # edited code runs at import (or the built import state is proven unchanged).
                return True if lazy_load_ok is None and task.phase == 2 else bool(lazy_load_ok)
            if task.tpc_mode and tid in TPC_SUITE_BY_TEST_ID:
                signal.alarm(65)
                try:
                    return verify_tpc_test(
                        TPC_SUITE_BY_TEST_ID[tid],
                        task.tpc_mode,
                        query_mask=task.tpc_query_mask.get(num),
                        s2b_query_mask=task.tpc_s2b_query_mask.get(num),
                        patched_files=sel.patched_files,
                        edited_spans=sel.edited_spans,
                    )
                except BaseException:
                    return False
                finally:
                    signal.alarm(0)
            if (
                tid == OPT_TPCDS_TEST_ID
                and select_tpcds_span(sel.patched_files, max_start=14, for_exec=False, edited_spans=sel.edited_spans)
                is not None
            ):
                signal.alarm(25)
                try:
                    cf_set = task.case_filter.get(num)
                    indices = {idx for _, idx in cf_set} if cf_set is not None else None
                    return verify_opt_tpcds(sel.patched_files, indices, edited_spans=sel.edited_spans)
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
                _est_test_cpu_ms(
                    n, id_by_number[n], sel, task.tpc_mode, cpu_by_number, task.tpc_query_mask, task.tpc_s2b_query_mask
                )
                for n in decap_set
            )
            other_rem = verifiable - decap_set
            other_cpus = [cpu_by_number[n] for n in other_rem]
            will_verify_tail = (
                bool(other_rem)
                and max(other_cpus) <= MAX_SINGLE_TEST_MS
                and sum(other_cpus) <= MAX_COMMIT_VERIFY_MS
            )
            # Only run a pre-decap fail-fast probe when a probe failure can actually block an
            # expensive non-fast_a TPC decap target (or when the remaining tail will be verified).
            # Cap at <=5 tests and <=2 modules (preferring already-loaded modules) to avoid mass imports.
            has_blockable_decap = any(
                id_by_number[n] in TPC_SUITE_BY_TEST_ID
                and TPC_SUITE_BY_TEST_ID[id_by_number[n]] == "tpcds"
                and task.tpc_mode != "fast_a"
                for n in decap_set
            )
            need_probe = has_blockable_decap or will_verify_tail
            probe_tests: list[int] = []
            if need_probe:
                other_ordered = sorted(
                    other_rem,
                    key=lambda n: (
                        0 if id_by_number[n].split("::", 1)[0][:-3].replace("/", ".") in sys.modules else 1,
                        cpu_by_number[n],
                        ms_by_number[n],
                        id_by_number[n],
                    ),
                )
                probe_cpu = 0
                probe_mods: set[str] = set()
                for num in other_ordered:
                    c_ms = cpu_by_number[num]
                    m_name = id_by_number[num].split("::", 1)[0][:-3].replace("/", ".")
                    if c_ms <= 450 and probe_cpu + c_ms <= TPC_PROBE_BUDGET_MS:
                        if m_name not in probe_mods and len(probe_mods) >= 2:
                            continue
                        probe_mods.add(m_name)
                        probe_tests.append(num)
                        probe_cpu += c_ms
                        if len(probe_tests) >= 5:
                            break

            # Step 2a: Fail-fast probe on cheap unit tests.
            for num in probe_tests:
                if not _mark(num, _run_candidate(num)):
                    break

            # Step 2b: Makespan & worker-collapsing decapitation. Never block test_execute_tpch
            # (fast & high ROI even in full_cached) or fast_a TPC after a unit-test failure.
            def _tpc_blocked_after_fail(n: int) -> bool:
                if not failed:
                    return False
                tid_n = id_by_number[n]
                if tid_n not in TPC_SUITE_BY_TEST_ID:
                    return False
                if TPC_SUITE_BY_TEST_ID[tid_n] == "tpch" or task.tpc_mode == "fast_a":
                    return False
                return True

            decap_ordered = [n for n in task.decap_targets if n in decap_set]
            for idx, num in enumerate(decap_ordered):
                if _tpc_blocked_after_fail(num):
                    continue
                elig_rem = [x for x in decap_ordered[idx:] if not _tpc_blocked_after_fail(x)]
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
                    task.tpc_query_mask,
                    task.tpc_s2b_query_mask,
                ):
                    continue
                _mark(num, _run_candidate(num))

        # Step 2c: the tail, as the best ROI prefix of what is still unverified, re-planned after
        # each failure with adaptive throttling and a deterministic estimated CPU cap.
        if not (sel.fixture_only and not VERIFY_FIXTURE_MAX_MS):
            per_mod_ms = 20 if task.phase == 1 else 4
            step2c_t0 = time.process_time()
            step2c_est_ms = 0
            for _ in range(12):
                if step2c_est_ms > 10000 or time.process_time() - step2c_t0 > 18.0:
                    break
                if len(failed) >= 6 and len(failed) >= 2 * max(len(passed), 1):
                    break
                cur_roi = 2.0 if (len(failed) >= 3 and len(failed) >= len(passed)) else MIN_ROI_RATIO
                batch = _best_drop_prefix(
                    task.chosen - passed,
                    verifiable - passed - failed - decap_set,
                    ms_by_number,
                    cpu_by_number,
                    MIN_GAIN_MS,
                    id_by_number=id_by_number,
                    per_mod_ms=per_mod_ms,
                    min_roi=cur_roi,
                )
                if not batch:
                    break
                step2c_est_ms += sum(cpu_by_number[n] for n in batch)
                if all(_mark(num, _run_candidate(num)) for num in batch):
                    break
    except BaseException:
        pass
    finally:
        try:
            os.close(w_fd)
        except OSError:
            pass
        os._exit(0)


def _budget_scale(n_commits: int) -> float:
    """Budgets are sized for a 30-commit batch; the run's time limit grows with the batch
    (3 minutes in practice, 4.5 for the ~1.5x larger final)."""
    return max(1.0, n_commits / 30)


_CASE_MAP: dict | None = None


def _case_plan(
    chosen: set[int],
    sel: Selection,
    id_by_number: dict[int, str],
    base_cpu: dict[int, int],
) -> tuple[
    dict[int, set[tuple[str, int]]],
    dict[int, tuple[int, int]],
    dict[int, tuple[int, int]],
    dict[int, int],
    dict[int, int],
    dict[int, int],
]:
    """Compute per-case/subTest/Validator/TPC query filters and scaled CPU estimates for `chosen`."""
    global _CASE_MAP
    case_filter: dict[int, set[tuple[str, int]]] = {}
    subtest_filter: dict[int, tuple[int, int]] = {}
    validator_filter: dict[int, tuple[int, int]] = {}
    tpc_query_mask: dict[int, int] = {}
    tpc_s2b_query_mask: dict[int, int] = {}
    cpu_override: dict[int, int] = {}

    if sel.edited_spans is not None:
        if _CASE_MAP is None:
            from blast_radius.precompile_cases import CASES_FILE

            _CASE_MAP = marshal.loads(CASES_FILE.read_bytes()) if CASES_FILE.is_file() else {}
        for num in chosen:
            tid = id_by_number[num]
            entry = _CASE_MAP.get(tid)
            if entry is None:
                continue
            mask = 0
            for path, first, last in sel.edited_spans:
                per_line = entry["lines"].get(path)
                if per_line:
                    for line in range(first, last + 1):
                        mask |= per_line.get(line, 0)
            kind = entry.get("kind", "check_file")
            if kind == "tpc_stage1":
                if mask:
                    tpc_query_mask[num] = mask
                else:
                    # Only skip Stage 1 (mask=0) if Stage 2a actually executed an edited span.
                    s2a = entry.get("s2a_lines", {})
                    if any(
                        first <= ln <= last
                        for path, first, last in sel.edited_spans
                        for ln in s2a.get(path, ())
                    ):
                        tpc_query_mask[num] = 0
                # Also compute per-query Stage-2b mask for fast_b / full_cached.
                s2b_lines = entry.get("s2b_lines")
                if s2b_lines:
                    s2b_mask = 0
                    for path, first, last in sel.edited_spans:
                        per_line = s2b_lines.get(path)
                        if per_line:
                            for line in range(first, last + 1):
                                s2b_mask |= per_line.get(line, 0)
                    if s2b_mask:
                        tpc_s2b_query_mask[num] = s2b_mask
                continue
            # Soundness guard: if a selected test matched 0 recorded cases, run it unfiltered.
            if not mask:
                continue
            if kind == "check_file":
                cases = entry["cases"]
                matched = {tuple(cases[k]) for k in range(len(cases)) if mask >> k & 1}
                if len(matched) < len(cases):
                    case_filter[num] = matched
                    cpu_override[num] = max(25, base_cpu[num] * (len(matched) + 1) // (len(cases) + 1))
            elif kind == "subtest":
                count = entry["count"]
                hits = mask.bit_count()
                if hits < count:
                    subtest_filter[num] = (mask, entry.get("stride", 1))
                    cpu_override[num] = max(25, base_cpu[num] * (hits + 1) // (count + 1))
            elif kind == "validator":
                count = entry["count"]
                full_mask = mask | entry.get("always", 0)
                hits = full_mask.bit_count()
                if hits < count:
                    validator_filter[num] = (full_mask, entry.get("stride", 1))
                    cpu_override[num] = max(12, base_cpu[num] * (hits + 1) // (count + 1))

    # Apply TPC-DS optimizer rule-stage checkpoint/suffix-span CPU estimate to TestOptimizer.test_tpcds.
    span_info = select_tpcds_span(sel.patched_files, max_start=14, for_exec=False, edited_spans=sel.edited_spans)
    if span_info is not None:
        _, _, ckpt_est_ms = span_info
        for num in chosen:
            if id_by_number[num] == OPT_TPCDS_TEST_ID:
                if num in case_filter:
                    scaled = max(45, ckpt_est_ms * (len(case_filter[num]) + 1) // 100)
                else:
                    scaled = ckpt_est_ms
                cpu_override[num] = min(cpu_override.get(num, base_cpu[num]), scaled)

    return case_filter, subtest_filter, validator_filter, tpc_query_mask, tpc_s2b_query_mask, cpu_override


def plan_verification(
    commits: list[tuple[str, set[int], Selection]],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    base_cpu: dict[int, int],
    number_by_index: list[int | None] | None = None,
) -> tuple[list[VerifyTask], list[VerifyTask]]:
    """Select and prioritize commits worth runtime verification or makespan decapitation."""
    candidates: list[VerifyTask] = []
    for commit, chosen, sel in commits:
        if not chosen or not sel.patched_files or sel.reason.startswith("everything"):
            continue
        case_filter, subtest_filter, validator_filter, tpc_query_mask, tpc_s2b_query_mask, cpu_override = _case_plan(
            chosen, sel, id_by_number, base_cpu
        )
        cpu_by_number = {**base_cpu, **cpu_override} if cpu_override else base_cpu
        tids = [id_by_number[n] for n in chosen]
        verifiable = {n for n in chosen if id_by_number[n] not in SUBPROCESS_TESTS}
        if not verifiable:
            continue

        static_cost_ms, _ = split_with_cost([(ms_by_number[n], n) for n in chosen])
        # test_lazy_load only checks `import sqlglot`: safe in Phase 2 when zero import-time code changed.
        lazy_safe_p2 = not sel.import_checks and "tests/dialects/test_dialect.py" not in sel.patched_files
        can_phase2 = (
            ENABLE_HOTPATCH
            and sel.hot_patch_targets is not None
            and not sel.lazy_pp_mask
            and not any(
                tid in FRESH_IMPORT_TESTS and not (lazy_safe_p2 and tid == LAZY_LOAD_TEST_ID)
                for tid in tids
            )
        )
        phase = 2 if can_phase2 else 1
        overhead_ms = 35 if phase == 2 else 320
        per_mod_ms = 4 if phase == 2 else 20

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
                c_ms = _est_test_cpu_ms(num, tid, sel, tpc_mode, cpu_by_number, tpc_query_mask, tpc_s2b_query_mask)
                raw_decap.append((sim_ms, c_ms, num))
            elif tid in (OPT_TPCH_TEST_ID, MAX_NODES_TEST_ID, GEN_MEM_TEST_ID, LAZY_LOAD_TEST_ID):
                c_ms = _est_test_cpu_ms(num, tid, sel, tpc_mode, cpu_by_number, tpc_query_mask, tpc_s2b_query_mask)
                if c_ms < sim_ms:
                    raw_decap.append((sim_ms, c_ms, num))
            elif sim_ms >= 150:
                c_ms = cpu_by_number[num]
                if c_ms <= sim_ms * 0.85:
                    raw_decap.append((sim_ms, c_ms, num))

        # Pick the prefix with the best net makespan / worker-count reduction, trying two
        # orders: tallest-first (knock down the ceiling) and cheapest-per-saved-ms first (so a
        # nearly free test is not stuck behind expensive taller ones).
        raw_decap = sorted(raw_decap, key=lambda item: (item[1] - item[0], item[2]))[:30]
        cur_pool = chosen - lazy_pp_nums
        cur_cost, _ = split_with_cost([(ms_by_number[n], n) for n in cur_pool])
        base_net = (static_cost_ms - cur_cost) - (15 if lazy_pp_nums else 0)
        best_net = base_net
        best_prefix: list[int] = []
        orders = (
            sorted(raw_decap, key=lambda item: (-item[0], item[1], item[2])),
            sorted(raw_decap, key=lambda item: (item[1] / item[0], -item[0], item[2])),
        )
        for order in orders:
            order_best, order_prefix = base_net, []
            chosen_decap: list[int] = []
            acc_decap_cpu = 0
            trial_pool = set(cur_pool)
            for _sim_ms, c_ms, num in order:
                trial_pool.discard(num)
                acc_decap_cpu += c_ms
                new_cost, _ = split_with_cost([(ms_by_number[n], n) for n in trial_pool])
                net = (static_cost_ms - new_cost) - acc_decap_cpu - (15 if lazy_pp_nums else 0)
                chosen_decap.append(num)
                if net >= order_best + 100:
                    order_best, order_prefix = net, list(chosen_decap)
            if order_best > best_net:
                best_net, best_prefix = order_best, order_prefix

        decap_targets = list(best_prefix)
        decap_set = set(decap_targets)

        def plan_tail(pool: set[int], cands: set[int]) -> tuple[list[int], int]:
            if sel.fixture_only and not VERIFY_FIXTURE_MAX_MS:
                return [], 0  # a fixture edit almost always breaks the tests reading it
            prefix = _best_drop_prefix(
                pool,
                cands,
                ms_by_number,
                cpu_by_number,
                MIN_GAIN_MS,
                id_by_number=id_by_number,
                per_mod_ms=per_mod_ms,
                min_roi=MIN_ROI_RATIO,
            )
            mod_count = len({id_by_number[n].split("::", 1)[0] for n in prefix})
            return prefix, sum(cpu_by_number[n] for n in prefix) + mod_count * per_mod_ms

        decap_cpu = sum(
            _est_test_cpu_ms(n, id_by_number[n], sel, tpc_mode, cpu_by_number, tpc_query_mask, tpc_s2b_query_mask)
            for n in decap_targets
        ) + (15 if lazy_pp_nums else 0)
        pool_after_decap = chosen - lazy_pp_nums - decap_set
        tail_prefix, tail_cpu = plan_tail(pool_after_decap, rem_verifiable - decap_set)
        cost_after, _ = split_with_cost([(ms_by_number[n], n) for n in pool_after_decap - set(tail_prefix)])
        est_cpu_ms = overhead_ms + decap_cpu + tail_cpu
        saved_ms = static_cost_ms - cost_after
        expected_gain_ms = saved_ms - est_cpu_ms
        has_heavy_tpc_or_max = any(
            id_by_number[n] in TPC_SUITE_BY_TEST_ID or id_by_number[n] == MAX_NODES_TEST_ID for n in decap_targets
        )
        min_req = TPC_MIN_GAIN_MS if has_heavy_tpc_or_max else MIN_DECAP_GAIN_MS if decap_targets else MIN_GAIN_MS
        roi_ok = (saved_ms >= est_cpu_ms * MIN_ROI_RATIO) or (saved_ms >= 5000 and expected_gain_ms >= 2500)
        if (expected_gain_ms < min_req or not roi_ok) and decap_targets:
            # Without the decapitation targets, the tail alone may still pay off.
            decap_targets = []
            tail_prefix, tail_cpu = plan_tail(chosen - lazy_pp_nums, rem_verifiable)
            cost_after, _ = split_with_cost([(ms_by_number[n], n) for n in chosen - lazy_pp_nums - set(tail_prefix)])
            est_cpu_ms = overhead_ms + (15 if lazy_pp_nums else 0) + tail_cpu
            saved_ms = static_cost_ms - cost_after
            expected_gain_ms = saved_ms - est_cpu_ms
            min_req = MIN_GAIN_MS
            roi_ok = saved_ms >= est_cpu_ms * MIN_ROI_RATIO
        if not (decap_targets or tail_prefix or lazy_pp_nums) or expected_gain_ms < min_req or not roi_ok:
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
                case_filter=case_filter,
                subtest_filter=subtest_filter,
                validator_filter=validator_filter,
                tpc_query_mask=tpc_query_mask,
                tpc_s2b_query_mask=tpc_s2b_query_mask,
                cpu_override=cpu_override,
            )
        )

    # Deterministically prioritize highest ROI (net savings / CPU) within total budget.
    candidates.sort(
        key=lambda t: (-t.expected_gain_ms / max(t.est_cpu_ms, 1), t.commit)
    )
    budget_used = 0
    budget = TOTAL_VERIFY_BUDGET_MS * _budget_scale(len(commits))
    phase1: list[VerifyTask] = []
    phase2: list[VerifyTask] = []
    for task in candidates:
        if budget_used + task.est_cpu_ms > budget:
            continue
        budget_used += task.est_cpu_ms
        if task.phase == 1:
            phase1.append(task)
        else:
            phase2.append(task)
    phase1.sort(key=lambda t: t.commit)
    phase2.sort(key=lambda t: t.commit)
    # Free the 7MB case map before forking children (tasks already hold extracted filters).
    global _CASE_MAP
    _CASE_MAP = None
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

    # Install lazy numpy/pandas stubs unconditionally in Phase-2 parent unless a task needs real
    # TPC pandas/numpy or patches test_executor.py — saves ~1.5s in any child importing test_executor.
    if not any(
        "tests/test_executor.py" in t.selection.patched_files
        or any(
            (id_by_number[n] in TPC_SUITE_BY_TEST_ID and not t.tpc_mode)
            or (id_by_number[n] == OPT_TPCH_TEST_ID and not _can_fast_opt_tpch(t.selection))
            for n in t.chosen
        )
        for t in phase2_tasks
    ):
        _install_lazy_data_libs()

    for mod_name in sorted(needed_mods):
        if mod_name not in sys.modules:
            try:
                mod = importlib.import_module(mod_name)
                if mod_name in ("tests.test_optimizer", "tests.test_executor"):
                    _patch_pool_references()
                if mod_name == "tests.test_optimizer":
                    _install_case_filter(mod)
                    _install_lazy_optimizer_setup(mod.TestOptimizer)
            except Exception as err:
                log(f"prewarm skip {mod_name}: {err}")

    # Pre-install subTest AST wrappers and Validator filter once in parent for Phase-2 COW sharing.
    has_val_filter = any(t.validator_filter for t in phase2_tasks)
    if has_val_filter and "tests.dialects.test_dialect" in sys.modules:
        _install_validator_filter()
    for task in phase2_tasks:
        for num in task.subtest_filter:
            parts = id_by_number[num].split("::")
            if len(parts) == 3:
                mod_name = parts[0][:-3].replace("/", ".")
                mod = sys.modules.get(mod_name)
                if mod is not None and hasattr(mod, parts[1]):
                    _install_subtest_filter(mod, getattr(mod, parts[1]), parts[2])


def _run_batch_parallel(
    tasks: list[VerifyTask],
    id_by_number: dict[int, str],
    ms_by_number: dict[int, int],
    cpu_by_number: dict[int, int],
    out_fd: int | None,
    stop_at: float,
    verified: dict[str, set[int]],
    started: float,
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
        failed_ids = [id_by_number[int(line[2:])].split("::")[-1] for line in lines if line.startswith(b"F ")]
        mtag = f" tpc={task.tpc_mode}" if task.tpc_mode else ""
        dtag = f" decap={len(task.decap_targets)}" if task.decap_targets else ""
        end = " TIMEOUT" if timed_out else ("" if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 else f" status={status}")
        log(
            f"{task.commit}: p{task.phase}{mtag}{dtag} cpu={child_cpu:.3f}s est={task.est_cpu_ms / 1000:.3f}s gain={task.expected_gain_ms / 1000:.1f}s "
            f"pass={len(passed_set)} fail={len(failed_ids)}/{len(task.chosen)}{end} failed={failed_ids[:5]}"
        )
        verified[task.commit] = task.chosen - passed_set

    while queue or active:
        # Run 1 child at a time initially to avoid SMT cache contention (+0.7s CPU);
        # automatically scale to 2-core parallelism if wall-clock elapsed exceeds 35s.
        cur_limit = max(1, MAX_PARALLEL_CHILDREN) if time.monotonic() - started > 35.0 else 1
        while queue and len(active) < cur_limit:
            now = time.monotonic()
            remaining = stop_at - now
            if remaining <= 0:
                for skipped in queue:
                    log(f"{skipped.commit}: verify skipped, wall budget spent")
                queue.clear()
                break
            task = queue.pop(0)
            limit_cpu_ms = task.est_cpu_ms
            if task.tpc_mode == "fast_a" and any(id_by_number[n] in TPC_SUITE_BY_TEST_ID for n in task.decap_targets):
                # fast_a re-executes the queries whose plans changed, up to a full_cached run.
                limit_cpu_ms += TPC_EST_CPU_MS[("tpcds", "full_cached")]
            limit_s = min(CHILD_LIMIT_BASE_S + CHILD_LIMIT_FACTOR * limit_cpu_ms / 1000, remaining)
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
    n_commits: int = 30,
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

    # Only pre-import duckdb in parent if a scheduled task may run a test that imports
    # test_optimizer (not fast verify_opt_tpcds) or real-unittest test_executor.
    needs_duckdb = any(
        (
            id_by_number[n].startswith("tests/test_optimizer.py::")
            and not (
                id_by_number[n] == OPT_TPCDS_TEST_ID
                and select_tpcds_span(
                    t.selection.patched_files, max_start=14, for_exec=False, edited_spans=t.selection.edited_spans
                )
                is not None
            )
        )
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

    # Capped well inside the final's 270 s limit.
    started = time.monotonic()
    stop_at = started + min(VERIFY_WALL_BUDGET_S * _budget_scale(n_commits), 200.0)

    if phase1:
        # Freeze parent heap so forked children don't trigger a full Gen-2 GC / COW storm (~0.6s/child).
        gc.freeze()
        _run_batch_parallel(phase1, id_by_number, ms_by_number, cpu_by_number, out_fd, stop_at, verified, started)

    if phase2 and time.monotonic() < stop_at:
        _prewarm_for_phase2(phase2, id_by_number, ms_by_number, cpu_by_number)
        gc.freeze()
        _run_batch_parallel(phase2, id_by_number, ms_by_number, cpu_by_number, out_fd, stop_at, verified, started)

    return verified
