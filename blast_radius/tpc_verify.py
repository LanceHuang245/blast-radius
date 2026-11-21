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

    # Function-body-only edits inside sqlglot/ with all edited functions accounted for
    # in our Stage-1/2a vs Stage-2b call sets can safely short-circuit the untouched stage.
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
        if all_accounted:
            if touches_s1 and not touches_s2b:
                return "fast_a"
            if touches_s2b and not touches_s1:
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
