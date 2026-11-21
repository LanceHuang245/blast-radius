"""Build-time per-case, per-subTest, per-Validator-call, and TPC-DS checkpoint precompiler.

Records fine-grained sqlglot line coverage for:
  1. TestOptimizer.check_file fixture tests (kind="check_file")
  2. TestOptimizer subTest-loop tests (kind="subtest", bucketed <=64 buckets, build-time verified)
  3. Dialect Validator tests >=25ms CPU (kind="validator", bucketed <=32 buckets, build-time verified)
  4. TPC-DS / TPC-H Stage-1 per-query line coverage (kind="tpc_stage1")

Also builds map/tpcds_ckpts.pkl with intermediate AST checkpoints across optimizer.RULES
and exact sys.setprofile-recorded cumulative file sets so TestOptimizer.test_tpcds and
verify_tpc_test("tpcds", "fast_a") skip untouched prefix stages cleanly.

Build: SKIP_INTEGRATION=0 python -m blast_radius.precompile_cases
"""

from __future__ import annotations

import ast
import glob
import importlib
import inspect
import marshal
import os
import pickle
import sys
import textwrap
import time
import unittest
from collections import defaultdict
from contextlib import contextmanager

from blast_radius import APP, SOURCE
from blast_radius.precompile_tpc import TPC_STAGE1_FILE, plan_signature

CASES_FILE = APP / "map" / "cases.marshal"
TPCDS_CKPTS_META_FILE = APP / "map" / "tpcds_ckpts_meta.pkl"
TPCDS_CKPT_DIR = APP / "map"
OPT_TEST_FILE = "tests/test_optimizer.py"
TPCDS_CKPT_STAGES = (1, 3, 5, 8, 12, 14)


def tpcds_ckpt_stage_file(stage: int):
    return TPCDS_CKPT_DIR / f"tpcds_ckpt_{stage}.pkl"


class _SubTestFilterTransformer(ast.NodeTransformer):
    """Wrap top-level `with self.subTest(...):` bodies in `if _br_subtest_ok():`."""

    def __init__(self) -> None:
        self._in_subtest = False
        self.wrapped = 0

    def visit_With(self, node: ast.With) -> ast.AST:
        is_sub = any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "subTest"
            for item in node.items
        )
        if is_sub and not self._in_subtest:
            self._in_subtest = True
            self.generic_visit(node)
            self._in_subtest = False
            self.wrapped += 1
            guard = ast.If(
                test=ast.Call(func=ast.Name(id="_br_subtest_ok", ctx=ast.Load()), args=[], keywords=[]),
                body=node.body,
                orelse=[],
            )
            ast.copy_location(guard, node)
            node.body = [guard]
            return node
        self.generic_visit(node)
        return node


def transform_subtest_method(mod, cls: type, method_name: str, ok_fn) -> bool:
    """Replace `cls.<method_name>` in-place with a subTest-filterable compiled version."""
    fn = cls.__dict__.get(method_name)
    if fn is None or getattr(fn, "_br_subtest_wrapped", False):
        return bool(getattr(fn, "_br_subtest_wrapped", False))
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except Exception:
        return False
    if any(isinstance(n, ast.Name) and n.id == "super" for n in ast.walk(tree)):
        return False
    transformer = _SubTestFilterTransformer()
    tree = ast.fix_missing_locations(transformer.visit(tree))
    if transformer.wrapped == 0:
        return False
    mod._br_subtest_ok = ok_fn
    local_ns: dict = {}
    exec(compile(tree, getattr(mod, "__file__", "<subtest>"), "exec"), mod.__dict__, local_ns)
    new_fn = local_ns.get(method_name)
    if new_fn is None:
        return False
    new_fn._br_subtest_wrapped = True
    setattr(cls, method_name, new_fn)
    return True


def _collect_used_validator_returns() -> set[tuple[str, int]]:
    """Find (abs_path, lineno) of validate_identity/validate_transpile calls whose return value is used."""
    used: set[tuple[str, int]] = set()
    for rel in sorted(glob.glob("tests/dialects/test_*.py", root_dir=str(SOURCE))):
        full = os.path.abspath(str(SOURCE / rel))
        try:
            tree = ast.parse(open(full, encoding="utf-8").read())
        except SyntaxError:
            continue
        expr_calls = {
            id(n.value)
            for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("validate_identity", "validate_transpile")
                and id(node) not in expr_calls
            ):
                used.add((full, node.lineno))
    return used


def _record_optimizer_cases(mon, tool: int, src_prefix: str) -> dict[str, dict]:
    """Record check_file and bucketed subTest-loop per-case line coverage for TestOptimizer."""
    from blast_radius import verify as V
    import sqlglot.generator as generator
    import tests.test_optimizer as T

    # Must patch tests.test_optimizer.wait / as_completed AFTER importing it, otherwise
    # concurrent.futures.wait deadlocks forever on _LazyFuture.
    V._patch_pool_references()

    state = {
        "in_check": None,
        "pending": [],
        "depth": 0,
        "case": None,
        "mode": None,  # "check_file" | "subtest" | "count"
        "broken": False,
        "recorder": None,
        "sub_idx": 0,
        "stride": 1,
        "cur_bucket": -1,
    }
    current_lines: set[tuple[str, int]] = set()
    per_test: dict[str, dict] = {}
    prefix_cut = len(src_prefix) - len("sqlglot/")

    def on_line(code, line):
        if state["case"] is not None and code.co_filename.startswith(src_prefix):
            current_lines.add((code.co_filename[prefix_cut:], line))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)

    orig_loader = T.load_sql_fixture_pairs

    def recording_loader(filename):
        for index, item in enumerate(orig_loader(filename)):
            if state["in_check"] is not None:
                state["pending"].append((filename, index))
            yield item

    T.load_sql_fixture_pairs = recording_loader

    orig_check_file = T.TestOptimizer.check_file

    def recording_check_file(self, file, *args, **kwargs):
        state["in_check"], state["pending"] = file, []
        try:
            return orig_check_file(self, file, *args, **kwargs)
        finally:
            if state["pending"]:
                state["broken"] = True
            state["in_check"] = None

    T.TestOptimizer.check_file = recording_check_file

    orig_subtest = unittest.TestCase.subTest

    @contextmanager
    def recording_subtest(self, msg=unittest.case._subtest_msg_sentinel, **params):
        mode = state["mode"]
        top_cf = mode == "check_file" and state["in_check"] is not None and state["depth"] == 0
        top_st = mode == "subtest" and state["in_check"] is None and state["depth"] == 0
        if mode == "count" and state["depth"] == 0:
            state["sub_idx"] += 1
        elif top_cf:
            if not state["pending"]:
                state["broken"] = True
                with orig_subtest(self, msg, **params):
                    yield
                return
            case = state["pending"].pop(0)
            generator._DISPATCH_CACHE.clear()
            current_lines.clear()
            state["case"] = case
            mon.restart_events()
            mon.set_events(tool, mon.events.LINE)
        elif top_st:
            idx = state["sub_idx"]
            state["sub_idx"] = idx + 1
            bucket = idx // state["stride"]
            if bucket != state["cur_bucket"]:
                if state["cur_bucket"] >= 0:
                    mon.set_events(tool, 0)
                    state["recorder"](state["cur_bucket"], set(current_lines))
                generator._DISPATCH_CACHE.clear()
                current_lines.clear()
                state["cur_bucket"] = bucket
                state["case"] = bucket
                mon.restart_events()
                mon.set_events(tool, mon.events.LINE)
        state["depth"] += 1
        try:
            with orig_subtest(self, msg, **params):
                yield
        finally:
            state["depth"] -= 1
            if top_cf:
                mon.set_events(tool, 0)
                state["recorder"](state["case"], set(current_lines))
                state["case"] = None

    unittest.TestCase.subTest = recording_subtest

    cls = T.TestOptimizer
    cls.setUpClass()
    cls_tree = ast.parse(inspect.getsource(cls))
    method_kinds: dict[str, str] = {}
    for node in cls_tree.body[0].body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            has_cf = any(isinstance(n, ast.Attribute) and n.attr == "check_file" for n in ast.walk(node))
            has_sub = any(isinstance(n, ast.Attribute) and n.attr == "subTest" for n in ast.walk(node))
            if has_cf:
                method_kinds[node.name] = "check_file"
            elif has_sub:
                method_kinds[node.name] = "subtest"

    verify_state = {"mask": None, "idx": 0, "stride": 1}

    def _verify_subtest_ok() -> bool:
        idx = verify_state["idx"]
        verify_state["idx"] = idx + 1
        mask = verify_state["mask"]
        return mask is None or bool(mask >> (idx // verify_state["stride"]) & 1)

    for name, kind in sorted(method_kinds.items()):
        tid = f"{OPT_TEST_FILE}::TestOptimizer::{name}"
        if kind == "check_file":
            cases: list = []
            lines: dict[str, dict[int, int]] = defaultdict(dict)

            def record_cf(case, executed, cases=cases, lines=lines):
                bit = 1 << len(cases)
                cases.append(case)
                for rel, line in executed:
                    lines[rel][line] = lines[rel].get(line, 0) | bit

            state.update(mode="check_file", recorder=record_cf, broken=False, depth=0, case=None)
            result = unittest.TestResult()
            cls(name).run(result)
            if cases and not state["broken"] and result.wasSuccessful():
                per_test[tid] = {"kind": "check_file", "cases": cases, "lines": dict(lines)}
            continue

        # SubTest loop: first count subtests & time uninstrumented; skip tiny (<20ms) tests.
        state.update(mode="count", depth=0, sub_idx=0, case=None, cur_bucket=-1)
        t_chk = time.process_time()
        r_chk = unittest.TestResult()
        cls(name).run(r_chk)
        total_sub = state["sub_idx"]
        if not r_chk.wasSuccessful() or total_sub < 2 or (time.process_time() - t_chk) < 0.020:
            continue

        stride = max(1, (total_sub + 63) // 64)
        n_buckets = (total_sub + stride - 1) // stride
        lines = defaultdict(dict)

        def record_st(bucket, executed, lines=lines):
            bit = 1 << bucket
            for rel, line in executed:
                lines[rel][line] = lines[rel].get(line, 0) | bit

        state.update(
            mode="subtest",
            recorder=record_st,
            broken=False,
            depth=0,
            case=None,
            sub_idx=0,
            stride=stride,
            cur_bucket=-1,
        )
        result = unittest.TestResult()
        cls(name).run(result)
        mon.set_events(tool, 0)
        if state["cur_bucket"] >= 0:
            record_st(state["cur_bucket"], set(current_lines))
        state["case"] = None
        if not result.wasSuccessful() or state["sub_idx"] != total_sub:
            continue

        # Self-verify transformed method with first bucket and last bucket only.
        unittest.TestCase.subTest = orig_subtest
        try:
            if not transform_subtest_method(T, cls, name, _verify_subtest_ok):
                continue
            ok_both = True
            for b_probe in (0, n_buckets - 1):
                verify_state.update(mask=1 << b_probe, idx=0, stride=stride)
                r = unittest.TestResult()
                cls(name).run(r)
                if not r.wasSuccessful() or verify_state["idx"] != total_sub:
                    ok_both = False
                    break
            if ok_both:
                per_test[tid] = {
                    "kind": "subtest",
                    "count": n_buckets,
                    "stride": stride,
                    "lines": dict(lines),
                }
        finally:
            verify_state["mask"] = None
            unittest.TestCase.subTest = recording_subtest

    mon.set_events(tool, 0)
    T.load_sql_fixture_pairs = orig_loader
    T.TestOptimizer.check_file = orig_check_file
    unittest.TestCase.subTest = orig_subtest
    return per_test


def _record_validator_cases(mon, tool: int, src_prefix: str) -> dict[str, dict]:
    """Record bucketed per-call coverage for Validator dialect tests >=25ms CPU."""
    import sqlglot.generator as generator
    from tests.dialects.test_dialect import Validator

    used_ret = _collect_used_validator_returns()
    state = {
        "active": False,
        "idx": 0,
        "depth": 0,
        "always": 0,
        "skip_mask": None,
        "stride": 1,
        "cur_bucket": -1,
    }
    current_lines: set[tuple[str, int]] = set()
    lines_ref: list[dict[str, dict[int, int]]] = [{}]
    prefix_cut = len(src_prefix) - len("sqlglot/")

    def on_line(code, line):
        if state["active"] and code.co_filename.startswith(src_prefix):
            current_lines.add((code.co_filename[prefix_cut:], line))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)

    orig_id = Validator.validate_identity
    orig_all = Validator.validate_all
    orig_tr = Validator.validate_transpile

    def flush_bucket() -> None:
        b = state["cur_bucket"]
        if b >= 0:
            mon.set_events(tool, 0)
            bit = 1 << b
            cur_map = lines_ref[0]
            for rel, ln in current_lines:
                cur_map[rel][ln] = cur_map[rel].get(ln, 0) | bit
            state["cur_bucket"] = -1
            state["active"] = False

    def make_wrapper(orig, can_return: bool):
        def wrapped(self, *args, **kwargs):
            if state["depth"] > 0:
                return orig(self, *args, **kwargs)
            idx = state["idx"]
            state["idx"] = idx + 1
            bucket = idx // state["stride"]
            if state["skip_mask"] is not None:
                if not (state["skip_mask"] >> bucket & 1):
                    return None
                state["depth"] += 1
                try:
                    return orig(self, *args, **kwargs)
                finally:
                    state["depth"] -= 1
            if can_return:
                frame = sys._getframe(1)
                if (os.path.abspath(frame.f_code.co_filename), frame.f_lineno) in used_ret:
                    state["always"] |= 1 << bucket
            if bucket != state["cur_bucket"]:
                flush_bucket()
                generator._DISPATCH_CACHE.clear()
                current_lines.clear()
                state["cur_bucket"] = bucket
                state["active"] = True
                mon.restart_events()
                mon.set_events(tool, mon.events.LINE)
            state["depth"] += 1
            try:
                return orig(self, *args, **kwargs)
            finally:
                state["depth"] -= 1

        return wrapped

    Validator.validate_identity = make_wrapper(orig_id, True)
    Validator.validate_all = make_wrapper(orig_all, False)
    Validator.validate_transpile = make_wrapper(orig_tr, True)

    per_test: dict[str, dict] = {}
    try:
        for rel_file in sorted(glob.glob("tests/dialects/test_*.py", root_dir=str(SOURCE))):
            mod_name = rel_file[:-3].replace("/", ".")
            mod = importlib.import_module(mod_name)
            for cls_name, cls in sorted(vars(mod).items()):
                if not (isinstance(cls, type) and issubclass(cls, Validator) and cls is not Validator):
                    continue
                if cls.__module__ != mod_name:
                    continue
                for meth in sorted(dir(cls)):
                    if not meth.startswith("test_") or meth == "test_lazy_load":
                        continue
                    # Uninstrumented timing pass: only map dialect tests >=25ms CPU with >=5 calls.
                    state.update(active=False, idx=0, depth=0, always=0, skip_mask=-1, stride=1, cur_bucket=-1)
                    t_chk = time.process_time()
                    r_chk = unittest.TestResult()
                    cls(meth).run(r_chk)
                    total_calls = state["idx"]
                    if not r_chk.wasSuccessful() or (time.process_time() - t_chk) < 0.025 or total_calls < 5:
                        continue

                    stride = max(1, (total_calls + 31) // 32)
                    n_buckets = (total_calls + stride - 1) // stride
                    cur_lines: dict[str, dict[int, int]] = defaultdict(dict)
                    lines_ref[0] = cur_lines
                    state.update(active=False, idx=0, depth=0, always=0, skip_mask=None, stride=stride, cur_bucket=-1)
                    res = unittest.TestResult()
                    cls(meth).run(res)
                    flush_bucket()
                    always = state["always"]
                    if not res.wasSuccessful() or state["idx"] != total_calls:
                        continue

                    # Self-verify: running only `always` buckets must pass with identical call count.
                    state.update(active=False, idx=0, depth=0, skip_mask=always, stride=stride, cur_bucket=-1)
                    res2 = unittest.TestResult()
                    cls(meth).run(res2)
                    if res2.wasSuccessful() and state["idx"] == total_calls:
                        tid = f"{rel_file}::{cls_name}::{meth}"
                        per_test[tid] = {
                            "kind": "validator",
                            "count": n_buckets,
                            "stride": stride,
                            "always": always,
                            "lines": dict(cur_lines),
                        }
    finally:
        mon.set_events(tool, 0)
        Validator.validate_identity = orig_id
        Validator.validate_all = orig_all
        Validator.validate_transpile = orig_tr

    return per_test


def _record_tpc_and_ckpts(mon, tool: int, src_prefix: str) -> tuple[dict[str, dict], dict]:
    """Record per-query TPC Stage-1 coverage and TPC-DS intermediate rule checkpoints."""
    from sqlglot import Dialect, find_tables, parse_one
    from sqlglot.executor.python import Python
    from sqlglot.optimizer import optimize
    from sqlglot.optimizer.optimizer import RULES
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema
    from sqlglot.serde import dump as serde_dump, load as serde_load
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs
    import sqlglot.generator as generator

    per_tpc: dict[str, dict] = {}
    s1_all = pickle.load(open(TPC_STAGE1_FILE, "rb")) if TPC_STAGE1_FILE.is_file() else {}
    cur_lines: set[tuple[str, int]] = set()
    recording = [False]
    prefix_cut = len(src_prefix) - len("sqlglot/")

    def on_line(code, line):
        if recording[0] and code.co_filename.startswith(src_prefix):
            cur_lines.add((code.co_filename[prefix_cut:], line))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)
    duck_gen = Dialect.get_or_raise("duckdb").generator()
    py_gen = Python().generator(identify=True, comments=False)

    for suite, tid in (
        ("tpcds", "tests/test_executor.py::TestExecutor::test_execute_tpcds"),
        ("tpch", "tests/test_executor.py::TestExecutor::test_execute_tpch"),
    ):
        s1 = s1_all.get(suite)
        if not s1:
            continue
        ensured = ensure_schema(s1["schema"])
        q_lines: dict[str, dict[int, int]] = defaultdict(dict)
        for idx, q in enumerate(s1["queries"]):
            generator._DISPATCH_CACHE.clear()
            cur_lines.clear()
            recording[0] = True
            mon.restart_events()
            mon.set_events(tool, mon.events.LINE)
            expr = parse_one(q["sql"])
            if q["execute"]:
                _ = {t.name for t in find_tables(expr)}
                _ = duck_gen.generate(expr)
                opt_expr = optimize(expr, ensured, leave_tables_isolated=True)
                _ = plan_signature(Plan(opt_expr), serde_dump)
            mon.set_events(tool, 0)
            recording[0] = False
            bit = 1 << idx
            for rel, ln in cur_lines:
                q_lines[rel][ln] = q_lines[rel].get(ln, 0) | bit

        generator._DISPATCH_CACHE.clear()
        cur_lines.clear()
        recording[0] = True
        mon.restart_events()
        mon.set_events(tool, mon.events.LINE)
        for expr_dump, _ in s1["gen_pairs"]:
            py_gen.generate(serde_load(expr_dump))
        mon.set_events(tool, 0)
        recording[0] = False
        s2a_lines: dict[str, list[int]] = defaultdict(list)
        for rel, ln in sorted(cur_lines):
            s2a_lines[rel].append(ln)

        per_tpc[tid] = {
            "kind": "tpc_stage1",
            "count": len(s1["queries"]),
            "lines": dict(q_lines),
            "s2a_lines": dict(s2a_lines),
        }

    mon.set_events(tool, 0)

    # Build TPC-DS rule checkpoints (Rules 1, 3, 5, 8, 12, 14) with exact sys.setprofile file sets.
    pairs = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    sqls = [sql for _, sql, _ in pairs]
    expected = [exp_sql for _, _, exp_sql in pairs]
    schema = ensure_schema(TPCDS_SCHEMA)
    kw = {
        "db": None,
        "catalog": None,
        "schema": schema,
        "dialect": None,
        "sql": None,
        "isolate_tables": True,
        "quote_identifiers": False,
    }
    full_prefix = str(SOURCE.resolve()) + "/"
    stage_files: set[str] = set()

    def prof(frame, event, _arg):
        if event == "call":
            fn = frame.f_code.co_filename
            if fn.startswith(full_prefix):
                rel = fn[len(full_prefix) :]
                if rel.startswith("sqlglot/"):
                    stage_files.add(rel)
        return None

    cum_files: set[str] = {
        "sqlglot/optimizer/optimizer.py",
        "sqlglot/schema.py",
        "sqlglot/serde.py",
    }
    rule_times_ms: list[float] = []
    ckpt_data: dict[int, dict] = {}
    sys.setprofile(prof)
    try:
        stage_files.clear()
        exprs = [parse_one(sql) for sql in sqls]
        cum_files |= stage_files
        parse_files = set(stage_files) | {"sqlglot/serde.py"}

        for idx, rule in enumerate(RULES):
            if idx in TPCDS_CKPT_STAGES:
                sys.setprofile(None)
                ckpt_data[idx] = {
                    "files": sorted(cum_files),
                    "dumps": [serde_dump(e) for e in exprs],
                }
                sys.setprofile(prof)
            stage_files.clear()
            t0 = time.process_time()
            params = inspect.getfullargspec(rule).args
            rkw = {p: kw[p] for p in params if p in kw}
            exprs = [rule(e, **rkw) for e in exprs]
            rule_times_ms.append((time.process_time() - t0) * 1000)
            cum_files |= stage_files

        sys.setprofile(None)
        ckpt_data[14] = {
            "files": sorted(cum_files),
            "dumps": [serde_dump(e) for e in exprs],
        }
        t0 = time.process_time()
        for e in exprs:
            e.sql(pretty=True)
        sql_ms = (time.process_time() - t0) * 1000

        s1_tpcds = s1_all.get("tpcds")
        s1_duck_files: set[str] = set()
        s1_plan_files: set[str] = set()
        s1_pygen_files: set[str] = set()
        if s1_tpcds:
            sys.setprofile(prof)
            stage_files.clear()
            for q in s1_tpcds["queries"]:
                if q["execute"]:
                    duck_gen.generate(parse_one(q["sql"]))
            s1_duck_files = set(stage_files)

            stage_files.clear()
            for q in s1_tpcds["queries"]:
                if q["execute"]:
                    Plan(serde_load(q["opt_dump"]))
            s1_plan_files = set(stage_files)

            stage_files.clear()
            for expr_dump, _ in s1_tpcds["gen_pairs"]:
                py_gen.generate(serde_load(expr_dump))
            s1_pygen_files = set(stage_files)
            sys.setprofile(None)
    finally:
        sys.setprofile(None)

    for k, entry in ckpt_data.items():
        rem_ms = (sum(rule_times_ms[k:]) * 0.55) + sql_ms + 40.0
        entry["est_ms"] = max(80, round(rem_ms))

    ckpts_payload = {
        "sqls": sqls,
        "expected": expected,
        "schema": TPCDS_SCHEMA,
        "checkpoints": ckpt_data,
        "parse_files": sorted(parse_files),
        "s1_opt_files": sorted(cum_files),
        "s1_duck_files": sorted(s1_duck_files | {"sqlglot/serde.py"}),
        "s1_plan_files": sorted(s1_plan_files | {"sqlglot/serde.py"}),
        "s1_pygen_files": sorted(s1_pygen_files | {"sqlglot/serde.py"}),
    }
    return per_tpc, ckpts_payload


def main() -> None:
    t0 = time.perf_counter()
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    from blast_radius import verify as V
    from blast_radius.import_state import import_all_sqlglot

    import_all_sqlglot()
    V._SKIP_POOL_PICKLE = True
    V._patch_pool_references()
    src_prefix = str(SOURCE.resolve()) + "/sqlglot/"

    mon = sys.monitoring
    tool = mon.COVERAGE_ID
    mon.use_tool_id(tool, "blast_radius_cases")
    try:
        opt_cases = _record_optimizer_cases(mon, tool, src_prefix)
        val_cases = _record_validator_cases(mon, tool, src_prefix)
        tpc_cases, ckpts_payload = _record_tpc_and_ckpts(mon, tool, src_prefix)
    finally:
        mon.set_events(tool, 0)
        mon.free_tool_id(tool)

    all_cases = {**opt_cases, **val_cases, **tpc_cases}
    CASES_FILE.write_bytes(marshal.dumps(all_cases))

    # Write tiny metadata file (~5KB) for parent planning and per-stage dump files (~6MB each)
    # loaded lazily inside forked children only for the exact checkpoint stage used.
    ckpt_meta = {}
    ckpt_bytes = 0
    for k, entry in ckpts_payload["checkpoints"].items():
        stage_file = tpcds_ckpt_stage_file(k)
        with open(stage_file, "wb") as handle:
            pickle.dump(
                {
                    "dumps": entry["dumps"],
                    "expected": ckpts_payload["expected"],
                    "schema": ckpts_payload["schema"],
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        ckpt_bytes += stage_file.stat().st_size
        ckpt_meta[k] = {"files": entry["files"], "est_ms": entry["est_ms"]}

    meta_payload = {
        "checkpoints": ckpt_meta,
        "parse_files": ckpts_payload["parse_files"],
        "s1_opt_files": ckpts_payload["s1_opt_files"],
        "s1_duck_files": ckpts_payload["s1_duck_files"],
        "s1_plan_files": ckpts_payload["s1_plan_files"],
        "s1_pygen_files": ckpts_payload["s1_pygen_files"],
    }
    with open(TPCDS_CKPTS_META_FILE, "wb") as handle:
        pickle.dump(meta_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    n_cf = sum(1 for v in opt_cases.values() if v["kind"] == "check_file")
    n_st = sum(1 for v in opt_cases.values() if v["kind"] == "subtest")
    print(
        f"case map in {time.perf_counter() - t0:.1f}s: "
        f"{n_cf} check_file, {n_st} subtest, {len(val_cases)} validator, {len(tpc_cases)} tpc; "
        f"cases={CASES_FILE.stat().st_size / 1e6:.2f} MB, "
        f"ckpt_meta={TPCDS_CKPTS_META_FILE.stat().st_size / 1e3:.1f} KB, "
        f"ckpts={ckpt_bytes / 1e6:.2f} MB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
