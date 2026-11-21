"""Two-stage exact cached verifier for test_execute_tpcds and test_execute_tpch.

Soundness guarantee:
    * Fast Path A (Stage 1 + Stage 2a, ~4s CPU): only used when every edited
      function belongs to Stage 1/2a and zero edited functions execute in Stage 2b.
      Verifies every query's parse, find_tables, DuckDB transpile SQL, ensure_schema,
      optimize, Plan canonical DAG signature, and Stage-2a Python codegen output
      are bit-for-bit identical to pinned eb11aec.
    * Fast Path B (Stage 2 cached plans, ~12.5s CPU): only used when every edited
      function belongs to Stage 2b and zero edited functions execute in Stage 1/2a.
      Rebuilds Table objects with patched sqlglot and verifies exact (columns, rows)
      equality against pinned eb11aec (strictly stronger than assert_frame_equal).
    * Full Cached Path (Stage 1 + Stage 2 without DuckDB, ~16.5s CPU): runs both
      stages + DuckDB transpile string check, skipping CSV/DuckDB/spawn overhead.
    * Any mismatch or exception fails fast and conservatively keeps the test.
"""

from __future__ import annotations

import math
import pickle

from blast_radius.precompile_tpc import (
    TPC_META_FILE,
    TPC_STAGE1_FILE,
    TPC_STAGE2_FILE,
    plan_signature,
)
from blast_radius.select import Selection

TPC_SUITE_BY_TEST_ID = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": "tpcds",
    "tests/test_executor.py::TestExecutor::test_execute_tpch": "tpch",
}

# Files that define the executor entrypoint, serialization, or TPC fixtures/harness:
# if patched, never substitute cached TPC verification for the real test.
TPC_DISALLOWED_FILES = frozenset({
    "tests/test_executor.py",
    "tests/helpers.py",
    "sqlglot/executor/__init__.py",
    "sqlglot/serde.py",
})

# Measured Docker CPU estimates (ms) used for ROI gating and parent deadline sizing.
TPC_EST_CPU_MS = {
    ("tpcds", "fast_a"): 4300,
    ("tpcds", "fast_b"): 12800,
    ("tpcds", "full_cached"): 17000,
    ("tpch", "fast_a"): 600,
    ("tpch", "fast_b"): 1500,
    ("tpch", "full_cached"): 2100,
}

_meta_cache: dict | None = None


def load_tpc_meta() -> dict | None:
    global _meta_cache
    if _meta_cache is None and TPC_META_FILE.is_file():
        with open(TPC_META_FILE, "rb") as handle:
            raw = pickle.load(handle)
        _meta_cache = {
            "stage1_funcs": set(tuple(x) for x in raw["stage1_funcs"]),
            "stage2b_funcs": set(tuple(x) for x in raw["stage2b_funcs"]),
            "opt_tpch_pairs": raw.get("opt_tpch_pairs"),
        }
    return _meta_cache


def select_tpc_mode(sel: Selection) -> str | None:
    """Choose the fastest provably-sound TPC verification mode for `sel`, or None if ineligible."""
    meta = load_tpc_meta()
    if meta is None or not TPC_STAGE1_FILE.is_file() or not TPC_STAGE2_FILE.is_file():
        return None
    for path in sel.patched_files:
        if path in TPC_DISALLOWED_FILES or path.startswith("tests/fixtures/"):
            return None

    # Function-body-only edits inside sqlglot/: if zero edited functions execute in Stage 2b,
    # Stage-1/2a bit-for-bit output parity proves Stage 2b receives identical inputs and code.
    if (
        sel.hot_patch_targets
        and not sel.attr_hot_targets
        and all(p.startswith("sqlglot/") for p in sel.patched_files)
    ):
        edited_pairs = {(rel_path, qual.rsplit(".", 1)[-1]) for rel_path, qual in sel.hot_patch_targets}
        touches_s1 = any(pair in meta["stage1_funcs"] for pair in edited_pairs)
        touches_s2b = any(pair in meta["stage2b_funcs"] for pair in edited_pairs)
        all_accounted = all(
            pair in meta["stage1_funcs"] or pair in meta["stage2b_funcs"] for pair in edited_pairs
        )
        if not touches_s2b:
            return "fast_a"
        if touches_s2b and not touches_s1 and all_accounted:
            return "fast_b"

    return "full_cached"


def _cell_equal(a, b) -> bool:
    """Strict type + value equality with NaN == NaN (strictly stronger than assert_frame_equal)."""
    if type(a) is not type(b):
        return False
    if a == b:
        return True
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return True
    return False


def _rows_equal(cols_a, rows_a, cols_b, rows_b) -> bool:
    if tuple(cols_a) != tuple(cols_b) or len(rows_a) != len(rows_b):
        return False
    for ra, rb in zip(rows_a, rows_b):
        if len(ra) != len(rb):
            return False
        for ca, cb in zip(ra, rb):
            if not _cell_equal(ca, cb):
                return False
    return True


def verify_tpc_test(suite_name: str, mode: str) -> bool:
    """Verify test_execute_tpcds or test_execute_tpch in the forked child; True iff proven PASS."""
    from sqlglot import find_tables, parse_one, transpile
    from sqlglot.executor.python import Python, PythonExecutor
    from sqlglot.executor.table import Table, ensure_tables
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema
    from sqlglot.serde import dump as serde_dump
    from sqlglot.serde import load as serde_load

    if mode == "fast_a":
        with open(TPC_STAGE1_FILE, "rb") as handle:
            s1 = pickle.load(handle)[suite_name]
        schema = s1["schema"]
        # Stage 1: parse all queries + find_tables + DuckDB transpile + ensure_schema + optimize + Plan.
        for q in s1["queries"]:
            expr = parse_one(q["sql"])
            if not q["execute"]:
                continue
            if sorted({t.name for t in find_tables(expr)}) != q["table_names"]:
                return False
            if transpile(q["sql"], write="duckdb")[0] != q["duckdb_sql"]:
                return False
            ensured_schema = ensure_schema(schema)
            opt_expr = optimize(expr, ensured_schema, leave_tables_isolated=True)
            if serde_dump(opt_expr) != q["opt_dump"]:
                return False
            plan = Plan(opt_expr)
            if plan_signature(plan, serde_dump) != q["plan_sig"]:
                return False

        # Stage 2a: Python().generator codegen parity on every expression generated during execution.
        py_gen = Python().generator(identify=True, comments=False)
        for expr_dump, expected_py in s1["gen_pairs"]:
            if py_gen.generate(serde_load(expr_dump)) != expected_py:
                return False
        return True

    with open(TPC_STAGE2_FILE, "rb") as handle:
        s2 = pickle.load(handle)[suite_name]
    schema = s2["schema"]
    # Build fresh Table objects using the patched Table class and exact schema column dicts.
    tables_by_name = {
        name: Table(columns=schema[name], rows=list(rows))
        for name, rows in s2["tables_raw"].items()
    }

    if mode == "fast_b":
        # Stage 2 only: rebuild Plan from cached optimized AST and run patched PythonExecutor.
        for q in s2["queries"]:
            if not q["execute"]:
                continue
            opt_expr = serde_load(q["opt_dump"])
            plan = Plan(opt_expr)
            q_tables = {name: tables_by_name[name] for name in q["table_names"]}
            tables_ = ensure_tables(q_tables)
            res = PythonExecutor(tables=tables_).execute(plan)
            if not _rows_equal(res.columns, res.rows, q["columns"], q["rows"]):
                return False
        return True

    # Full cached path: Stage 1 + Stage 2 + DuckDB transpile parity (no CSV/DuckDB/spawn).
    with open(TPC_STAGE1_FILE, "rb") as handle:
        s1 = pickle.load(handle)[suite_name]
    for q1, q2 in zip(s1["queries"], s2["queries"]):
        expr = parse_one(q1["sql"])
        if not q1["execute"]:
            continue
        if transpile(q1["sql"], write="duckdb")[0] != q1["duckdb_sql"]:
            return False
        q_tables = {t.name: tables_by_name[t.name] for t in find_tables(expr)}
        tables_ = ensure_tables(q_tables)
        ensured_schema = ensure_schema(schema)
        if tables_.supported_table_args and tables_.supported_table_args != ensured_schema.supported_table_args:
            return False
        opt_expr = optimize(expr, ensured_schema, leave_tables_isolated=True)
        plan = Plan(opt_expr)
        res = PythonExecutor(tables=tables_).execute(plan)
        if not _rows_equal(res.columns, res.rows, q2["columns"], q2["rows"]):
            return False
    return True


def verify_optimized_tpch(patched_files: dict[str, str]) -> bool:
    """Verify test_optimized_tpch via exact DuckDB transpile parity on all 22 query pairs (~0.09s CPU)."""
    if any(
        p in ("tests/test_executor.py", "tests/helpers.py") or p.startswith("tests/fixtures/")
        for p in patched_files
    ):
        return False
    meta = load_tpc_meta()
    pairs = meta.get("opt_tpch_pairs") if meta else None
    if not pairs:
        return False
    from sqlglot import transpile

    for sql, opt_sql, exp_sql_duck, exp_opt_duck in pairs:
        if transpile(sql, write="duckdb")[0] != exp_sql_duck:
            return False
        if transpile(opt_sql, write="duckdb")[0] != exp_opt_duck:
            return False
    return True


def verify_max_nodes(patched_files: dict[str, str]) -> bool:
    """Verify TestParser.test_max_nodes with fail-fast 50k check and token reuse (~11.8s local CPU)."""
    if "tests/test_parser.py" in patched_files:
        return False
    from sqlglot import Dialect, ParseError, exp, parse_one

    if not isinstance(parse_one("SELECT 1", max_nodes=1_000_000), exp.Select):
        return False
    cols500 = [f"col{i}" for i in range(500)]
    if not isinstance(parse_one(f"SELECT {','.join(cols500)} FROM t", max_nodes=1_000_000), exp.Select):
        return False
    try:
        parse_one("SELECT " + ",".join(f"x={i}" for i in range(100)), max_nodes=5)
        return False
    except ParseError:
        pass
    if not isinstance(parse_one("SELECT 1, 2, 3"), exp.Select):
        return False

    width = 10000
    depth = 100
    expr = "x" + "=x" * depth
    sql = f"SELECT {','.join(expr for _ in range(width))}"
    dialect = Dialect.get_or_raise(None)

    tokenizer_touched = any(
        p in ("sqlglot/tokens.py", "sqlglot/tokenizer_core.py", "sqlglot/dialects/dialect.py")
        for p in patched_files
    )
    if not tokenizer_touched:
        sample_toks = dialect.tokenize(f"SELECT {expr},{expr}")
        if len(sample_toks) != 404:
            return False
        toks = sample_toks[:202] + sample_toks[202:] * (width - 1)
    else:
        toks = dialect.tokenize(sql)

    # Check max_nodes=50_000 first (~0.68s) to fail fast if max_nodes or parser is broken.
    try:
        dialect.parser(max_nodes=50_000).parse(toks, sql)
        return False
    except ParseError as err50:
        if "(50000)" not in str(err50):
            return False

    try:
        dialect.parser(max_nodes=1_000_000).parse(toks, sql)
        return False
    except ParseError as err1m:
        if "Maximum number of AST nodes" not in str(err1m):
            return False

    return True


def verify_generator_memory(patched_files: dict[str, str], scoped_only: bool = True) -> bool:
    """Verify TestGenerator.test_generator_memory across MRO-affected dialects (or all dialects)."""
    if "tests/test_generator.py" in patched_files:
        return False
    import gc
    import tracemalloc
    from blast_radius.impact_map import module_name
    from sqlglot.dialects.dialect import Dialect, Dialects

    patched_mods = {module_name(p) for p in patched_files if p.endswith(".py")}
    core_touched = any(
        not p.startswith(("sqlglot/dialects/", "sqlglot/generators/", "sqlglot/parsers/", "sqlglot/typing/"))
        or p in ("sqlglot/dialects/dialect.py", "sqlglot/dialects/__init__.py", "sqlglot/generators/__init__.py", "sqlglot/parsers/base.py", "sqlglot/typing/__init__.py")
        for p in patched_files
    )

    target_dialects: list[Dialect] = []
    for name in Dialects:
        if not name:
            continue
        d = Dialect.get_or_raise(name)
        if not core_touched and scoped_only:
            mro_mods = {
                c.__module__
                for root_cls in (type(d), d.generator_class, d.parser_class, d.tokenizer_class)
                for c in root_cls.__mro__
            }
            if not (mro_mods & patched_mods):
                continue
        target_dialects.append(d)

    if not target_dialects:
        return True

    for dialect in target_dialects:
        for _ in range(100):
            dialect.generator()
        gc.collect()
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            for _ in range(1000):
                dialect.generator()
            gc.collect()
            after = tracemalloc.get_traced_memory()[0]
        finally:
            tracemalloc.stop()
        if not (after - before < 1024):
            return False
    return True


def verify_lazy_athena_presto(patched_files: dict[str, str] | None = None) -> bool:
    """Verify that patched modules plus Athena and Presto import cleanly with unchanged MRO."""
    import importlib
    from blast_radius.impact_map import module_name

    for rel_path in patched_files or ():
        if rel_path.endswith(".py"):
            importlib.import_module(module_name(rel_path))
    from sqlglot.dialects.athena import Athena
    from sqlglot.dialects.presto import Presto

    return (
        [c.__name__ for c in Athena.__mro__] == ["Athena", "Trino", "Presto", "Dialect", "object"]
        and [c.__name__ for c in Presto.__mro__] == ["Presto", "Dialect", "object"]
    )
