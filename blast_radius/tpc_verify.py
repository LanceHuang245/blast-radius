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
import os
import pickle
import sys
import time

from blast_radius.precompile_cases import (
    TPC_S1_BLOBS_FILE,
    TPC_S2_BLOBS_DIR,
    TPCDS_CKPTS_META_FILE,
    tpc_s2_query_file,
    tpc_s2_table_file,
    tpcds_ckpt_stage_file,
)
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

# Fallback Linux CPU estimates (ms); exact per-query timings are measured on the build host
# in precompile_cases.py (s2b_query_ms) and used automatically when available.
TPC_EST_CPU_MS = {
    ("tpcds", "fast_a"): 3000,
    ("tpcds", "fast_b"): 12800,
    ("tpcds", "full_cached"): 16500,
    ("tpch", "fast_a"): 450,
    ("tpch", "fast_b"): 1500,
    ("tpch", "full_cached"): 2000,
}

# fast_a re-executes queries whose optimized plan changed; cheapest queries run first so broken
# commits fail fast while legitimate plan refactors can still pass within budget.
TPC_REEXEC_MAX_QUERIES = int(os.environ.get("BR_TPC_REEXEC_MAX", "25"))
TPC_REEXEC_BUDGET_S = float(os.environ.get("BR_TPC_REEXEC_BUDGET", "6.0"))

_meta_cache: dict | None = None
_ckpts_meta_cache: dict | None = None


def load_tpcds_ckpts_meta() -> dict | None:
    global _ckpts_meta_cache
    if _ckpts_meta_cache is None and TPCDS_CKPTS_META_FILE.is_file():
        with open(TPCDS_CKPTS_META_FILE, "rb") as handle:
            _ckpts_meta_cache = pickle.load(handle)
    return _ckpts_meta_cache


def load_tpcds_ckpt_stage(stage: int) -> dict | None:
    path = tpcds_ckpt_stage_file(stage)
    if not path.is_file():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def select_tpcds_ckpt(patched_files: dict[str, str], max_stage: int = 14) -> tuple[int, int] | None:
    """Return highest sound (rule_stage, est_ms) checkpoint for TPC-DS whose prefix files are untouched."""
    if not patched_files or not all(p.startswith("sqlglot/") for p in patched_files):
        return None
    meta = load_tpcds_ckpts_meta()
    if meta is None:
        return None
    patched_set = set(patched_files)
    for k in sorted(meta["checkpoints"], reverse=True):
        if k > max_stage:
            continue
        if patched_set.isdisjoint(meta["checkpoints"][k]["files"]):
            return k, meta["checkpoints"][k]["est_ms"]
    return None


def _touched_stage_bits(
    meta: dict,
    patched_files: dict[str, str],
    edited_spans: tuple[tuple[str, int, int], ...] | None,
) -> int | None:
    """Return exact bitmask of TPC-DS stages executed by `edited_spans`, or None to use file-level fallback."""
    if edited_spans is None or not patched_files:
        return None
    if "sqlglot/serde.py" in patched_files or "sqlglot/optimizer/optimizer.py" in patched_files:
        return None
    stage_lines = meta.get("stage_lines")
    if not stage_lines:
        return None
    bits = 0
    for path, first, last in edited_spans:
        per_line = stage_lines.get(path)
        if per_line:
            for ln in range(first, last + 1):
                bits |= per_line.get(ln, 0)
    return bits


def select_tpcds_span(
    patched_files: dict[str, str],
    max_start: int = 14,
    for_exec: bool = False,
    edited_spans: tuple[tuple[str, int, int], ...] | None = None,
) -> tuple[int | None, int | None, int] | None:
    """Return sound (start_k, stop_k, est_ms) rule span for TPC-DS, or None if neither applies.

    Uses exact per-line stage bitmasks when `edited_spans` is available (falling back to file-level sets).
    - start_k: highest checkpoint <= max_start whose prefix stages are untouched (skip RULES[:start_k]).
    - stop_k: lowest checkpoint > start_k (or 0 if start_k is None) whose suffix stages (and Plan for
      for_exec, or sql() gen_files for test_tpcds) are untouched (short-circuit RULES[stop_k:] when
      intermediate AST matches pinned ckpt[stop_k]).
    """
    if not patched_files or not all(p.startswith("sqlglot/") for p in patched_files):
        return None
    meta = load_tpcds_ckpts_meta()
    if meta is None:
        return None
    ckpts = meta["checkpoints"]
    bits = _touched_stage_bits(meta, patched_files, edited_spans)

    start_k: int | None = None
    stop_k: int | None = None
    if bits is not None:
        # Bit 21 = ensure_schema; bit 0 = parse_one; bits 1..14 = RULES[0..13];
        # bit 15 = exec RULES[7:]; bit 16 = Plan; bit 17 = e.sql(pretty=True).
        schema_bit = 1 << 21
        for k in sorted(ckpts, reverse=True):
            if k > max_start:
                continue
            prefix_mask = ((1 << (k + 1)) - 1) | schema_bit
            if (bits & prefix_mask) == 0:
                start_k = k
                break
        max_stop = min(max_start, 7) if for_exec else max_start
        tail_guard = ((1 << 15) | (1 << 16) | schema_bit) if for_exec else ((1 << 17) | schema_bit)
        min_stop = (start_k + 1) if start_k is not None else 0
        for m in sorted(ckpts):
            if m < min_stop or m > max_stop:
                continue
            suffix_mask = tail_guard
            for r in range(m, 14):
                suffix_mask |= 1 << (r + 1)
            if (bits & suffix_mask) == 0:
                stop_k = m
                break
    else:
        patched_set = set(patched_files)
        for k in sorted(ckpts, reverse=True):
            if k > max_start:
                continue
            if patched_set.isdisjoint(ckpts[k]["files"]):
                start_k = k
                break

        if for_exec:
            can_stop = patched_set.isdisjoint(meta.get("s1_plan_files", ()))
            suffix_key = "suffix_exec_files"
            max_stop = min(max_start, 7)
        else:
            can_stop = patched_set.isdisjoint(meta.get("gen_files", ()))
            suffix_key = "suffix_files"
            max_stop = max_start

        if can_stop:
            min_stop = (start_k + 1) if start_k is not None else 0
            for m in sorted(ckpts):
                if m < min_stop or m > max_stop:
                    continue
                sfiles = ckpts[m].get(suffix_key)
                if sfiles is not None and patched_set.isdisjoint(sfiles):
                    stop_k = m
                    break

    if start_k is None and stop_k is None:
        return None

    rule_times = meta.get("rule_times_ms")
    sql_ms = meta.get("sql_ms", 120.0) if not for_exec else 0.0
    if rule_times:
        lo = start_k if start_k is not None else 0
        hi = stop_k if stop_k is not None else len(rule_times)
        parse_est = 120.0 if start_k is None else 0.0
        est_ms = max(90, round(parse_est + sum(rule_times[lo:hi]) * 0.85 + sql_ms + 50.0))
    else:
        est_ms = ckpts[start_k]["est_ms"] if start_k is not None else 500
    return start_k, stop_k, est_ms


def est_tpc_fast_a_ms(
    suite_name: str,
    patched_files: dict[str, str],
    matched_queries: int | None,
    total_queries: int,
    edited_spans: tuple[tuple[str, int, int], ...] | None = None,
) -> int:
    """Estimate fast_a CPU (ms) accounting for untouched sub-stages, checkpoints, and per-query filtering."""
    base = TPC_EST_CPU_MS[(suite_name, "fast_a")]
    if suite_name == "tpcds" and patched_files and all(p.startswith("sqlglot/") for p in patched_files):
        meta = load_tpcds_ckpts_meta()
        if meta is not None:
            bits = _touched_stage_bits(meta, patched_files, edited_spans)
            if bits is not None:
                opt_mask = ((1 << 16) - 1) | (1 << 21)
                if (bits & opt_mask) == 0:
                    base = 350 if (bits & (1 << 16)) == 0 else base
                else:
                    span = select_tpcds_span(patched_files, max_start=7, for_exec=True, edited_spans=edited_spans)
                    if span is not None:
                        base = min(base, span[2])
            else:
                patched_set = set(patched_files)
                if patched_set.isdisjoint(meta["s1_opt_files"]):
                    base = 350 if patched_set.isdisjoint(meta["s1_plan_files"]) else base
                else:
                    span = select_tpcds_span(patched_files, max_start=7, for_exec=True)
                    if span is not None:
                        base = min(base, span[2])
    if matched_queries is not None and total_queries > 0:
        base = max(120, base * (matched_queries + 1) // (total_queries + 1))
    return base


def est_tpc_mode_ms(
    suite_name: str,
    mode: str,
    patched_files: dict[str, str],
    s1_matched: int | None,
    s2b_matched: int | None,
    total_queries: int,
    edited_spans: tuple[tuple[str, int, int], ...] | None = None,
    s2b_mask: int | None = None,
    s1_mask: int | None = None,
) -> int:
    """Estimate TPC CPU (ms) accounting for untouched sub-stages, checkpoints, and exact per-query timings."""
    if mode == "fast_a":
        return est_tpc_fast_a_ms(suite_name, patched_files, s1_matched, total_queries, edited_spans)
    meta = load_tpcds_ckpts_meta()
    q_ms = (meta.get("s2b_query_ms") or {}).get(suite_name) if meta else None
    if q_ms and len(q_ms) == total_queries:
        if mode == "fast_b":
            if s2b_mask is not None:
                total = sum(q_ms[i] for i in range(total_queries) if s2b_mask >> i & 1)
                return max(150, total + 120)
            return max(500, sum(q_ms) + 250)
        if mode == "full_cached":
            s1_est = est_tpc_fast_a_ms(suite_name, patched_files, s1_matched, total_queries, edited_spans)
            if s1_mask is not None and s2b_mask is not None:
                fm = s1_mask | s2b_mask
                s2_est = sum(q_ms[i] for i in range(total_queries) if fm >> i & 1)
                return max(250, s1_est + s2_est + 150)
            return max(800, s1_est + sum(q_ms) + 250)
    base = TPC_EST_CPU_MS[(suite_name, mode)]
    if mode == "fast_b" and s2b_matched is not None and total_queries > 0:
        return max(150, base * (s2b_matched + 1) // (total_queries + 1))
    if mode == "full_cached" and total_queries > 0:
        if s1_matched is not None and s2b_matched is not None:
            hits = max(s1_matched, s2b_matched)
            return max(250, base * (hits + 1) // (total_queries + 1))
    return base


def verify_opt_tpcds(
    patched_files: dict[str, str],
    case_indices: set[int] | None = None,
    edited_spans: tuple[tuple[str, int, int], ...] | None = None,
) -> bool:
    """Verify TestOptimizer.test_tpcds using untouched prefix checkpoint and/or suffix short-circuit."""
    span = select_tpcds_span(patched_files, max_start=14, for_exec=False, edited_spans=edited_spans)
    if span is None:
        return False
    start_k, stop_k, _ = span
    start_data = load_tpcds_ckpt_stage(start_k) if start_k is not None else load_tpcds_ckpt_stage(0)
    if start_data is None:
        return False
    stop_data = load_tpcds_ckpt_stage(stop_k) if stop_k is not None else None
    if stop_k is not None and stop_data is None:
        return False

    from sqlglot import parse_one
    from sqlglot.optimizer import optimize
    from sqlglot.optimizer.optimizer import RULES
    from sqlglot.schema import ensure_schema
    from sqlglot.serde import dump as serde_dump
    from sqlglot.serde import load as serde_load

    expected = start_data["expected"]
    sqls = start_data.get("sqls")
    if start_k is None and sqls is None:
        return False
    start_dumps_b = start_data.get("dumps_b")
    start_dumps = start_data.get("dumps")
    stop_dumps_b = stop_data.get("dumps_b") if stop_data is not None else None
    stop_dumps = stop_data.get("dumps") if stop_data is not None else None
    ensured = ensure_schema(start_data["schema"]) if (start_k is None or start_k < 14) else None
    indices = sorted(case_indices) if case_indices is not None else range(len(expected))

    for index in indices:
        if start_k is None:
            expr = parse_one(sqls[index])
            if stop_k == 0:
                stop_raw = pickle.loads(stop_dumps_b[index]) if stop_dumps_b is not None else stop_dumps[index]
                if serde_dump(expr) == stop_raw:
                    continue
                expr = optimize(expr, ensured, copy=False)
            elif stop_k is not None:
                expr = optimize(expr, ensured, rules=RULES[:stop_k], copy=False)
                stop_raw = pickle.loads(stop_dumps_b[index]) if stop_dumps_b is not None else stop_dumps[index]
                if serde_dump(expr) == stop_raw:
                    continue
                expr = optimize(expr, ensured, rules=RULES[stop_k:], copy=False)
            else:
                expr = optimize(expr, ensured, copy=False)
        else:
            raw_dump = pickle.loads(start_dumps_b[index]) if start_dumps_b is not None else start_dumps[index]
            expr = serde_load(raw_dump)
            if stop_k is not None:
                if start_k < stop_k:
                    expr = optimize(expr, ensured, rules=RULES[start_k:stop_k], copy=False)
                stop_raw = pickle.loads(stop_dumps_b[index]) if stop_dumps_b is not None else stop_dumps[index]
                if serde_dump(expr) == stop_raw:
                    continue
                if stop_k < 14:
                    expr = optimize(expr, ensured, rules=RULES[stop_k:], copy=False)
            elif start_k < 14:
                expr = optimize(expr, ensured, rules=RULES[start_k:], copy=False)

        if expr.sql(pretty=True) != expected[index]:
            return False
    return True


def load_tpc_meta() -> dict | None:
    global _meta_cache
    if _meta_cache is None and TPC_META_FILE.is_file():
        with open(TPC_META_FILE, "rb") as handle:
            raw = pickle.load(handle)
        _meta_cache = {
            "stage1_funcs": set(tuple(x) for x in raw["stage1_funcs"]),
            "stage2b_funcs": set(tuple(x) for x in raw["stage2b_funcs"]),
            "stage2b_files": {x[0] for x in raw["stage2b_funcs"]},
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

    # Any edit (class attributes, module level, several files) outside every file that runs code
    # in Stage 2b leaves Stage 2b's code unchanged; fast_a checks its inputs (re-executing the
    # queries whose plans changed).
    if all(p.startswith("sqlglot/") and p not in meta["stage2b_files"] for p in sel.patched_files):
        return "fast_a"

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


def _same_duckdb_results(suite_name: str, pairs: list[tuple[str, str]]) -> bool:
    """True if each patched DuckDB SQL returns exactly what the pinned one does.

    The real test's expected rows come from DuckDB running transpile(sql, write="duckdb"):
    when the patched and pinned SQL return the same columns, types and rows on the test's
    own views, the expected side of every comparison is unchanged.
    """
    if len(pairs) > TPC_REEXEC_MAX_QUERIES:
        return False
    import duckdb
    from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, TPCH_SCHEMA

    schema, folder = (TPCDS_SCHEMA, "tpc-ds") if suite_name == "tpcds" else (TPCH_SCHEMA, "tpc-h")
    t0 = time.process_time()
    conn = duckdb.connect()
    try:
        # Same views as TestExecutor.setUpClass.
        for table, columns in schema.items():
            conn.execute(
                f"CREATE VIEW {table} AS SELECT * FROM READ_CSV("
                f"'{FIXTURES_DIR}/optimizer/{folder}/{table}.csv.gz', delim='|', header=True, columns={columns})"
            )
        for new_sql, old_sql in pairs:
            if time.process_time() - t0 > TPC_REEXEC_BUDGET_S:
                return False
            new = conn.execute(new_sql)
            new_desc, new_rows = [d[:2] for d in new.description], new.fetchall()
            old = conn.execute(old_sql)
            if new_desc != [d[:2] for d in old.description] or new_rows != old.fetchall():
                return False
        return True
    finally:
        conn.close()


def _why(msg: str) -> None:
    print(f"tpc_verify: {msg}", file=sys.stderr)


def verify_tpc_test(
    suite_name: str,
    mode: str,
    query_mask: int | None = None,
    s2b_query_mask: int | None = None,
    patched_files: dict[str, str] | None = None,
    edited_spans: tuple[tuple[str, int, int], ...] | None = None,
) -> bool:
    """Verify test_execute_tpcds or test_execute_tpch in the forked child; True iff proven PASS."""
    from sqlglot import Dialect, find_tables, parse_one
    from sqlglot.executor.python import Python, PythonExecutor
    from sqlglot.executor.table import Table, ensure_tables
    from sqlglot.optimizer import optimize
    from sqlglot.optimizer.optimizer import RULES
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema
    from sqlglot.serde import dump as serde_dump
    from sqlglot.serde import load as serde_load

    duck_gen = Dialect.get_or_raise("duckdb").generator()
    has_s2_blobs = (TPC_S2_BLOBS_DIR / suite_name / "meta.pkl").is_file()

    def _load_s1() -> dict:
        if TPC_S1_BLOBS_FILE.is_file():
            with open(TPC_S1_BLOBS_FILE, "rb") as handle:
                return pickle.load(handle)[suite_name]
        with open(TPC_STAGE1_FILE, "rb") as handle:
            return pickle.load(handle)[suite_name]

    def _q_opt_dump(q: dict):
        return pickle.loads(q["opt_dump_b"]) if "opt_dump_b" in q else q["opt_dump"]

    if mode == "fast_a":
        s1 = _load_s1()
        ensured_schema = ensure_schema(s1["schema"])
        s2_meta: dict | None = None
        s2_tables_cache: dict[str, list] = {}
        s2_full_cache: list = []

        skip_opt = False
        skip_duck = False
        skip_plan = False
        skip_pygen = False
        plan_untouched = False
        scope_patched = bool(
            patched_files and ("sqlglot/optimizer/scope.py" in patched_files or "sqlglot/__init__.py" in patched_files)
        )
        start_k: int | None = None
        stop_k: int | None = None
        start_dumps_b: list | None = None
        start_dumps: list | None = None
        stop_dumps_b: list | None = None
        stop_dumps: list | None = None
        if suite_name == "tpcds" and patched_files and all(p.startswith("sqlglot/") for p in patched_files):
            meta = load_tpcds_ckpts_meta()
            if meta is not None:
                bits = _touched_stage_bits(meta, patched_files, edited_spans)
                if bits is not None:
                    opt_mask = ((1 << 16) - 1) | (1 << 21)
                    skip_opt = (bits & opt_mask) == 0
                    skip_duck = (bits & ((1 << 18) | (1 << 21))) == 0
                    plan_untouched = (bits & ((1 << 16) | (1 << 21))) == 0
                    skip_plan = skip_opt and plan_untouched
                    skip_pygen = (bits & ((1 << 19) | (1 << 21))) == 0
                    scope_patched = (bits & (1 << 20)) != 0
                else:
                    patched_set = set(patched_files)
                    skip_opt = patched_set.isdisjoint(meta["s1_opt_files"])
                    skip_duck = patched_set.isdisjoint(meta["s1_duck_files"])
                    plan_untouched = patched_set.isdisjoint(meta["s1_plan_files"])
                    skip_plan = skip_opt and plan_untouched
                    skip_pygen = patched_set.isdisjoint(meta["s1_pygen_files"])
                if not skip_opt:
                    span = select_tpcds_span(
                        patched_files, max_start=7, for_exec=True, edited_spans=edited_spans
                    )
                    if span is not None:
                        s_k, e_k, _ = span
                        if s_k is not None:
                            s_data = load_tpcds_ckpt_stage(s_k)
                            if s_data is not None:
                                start_k = s_k
                                start_dumps_b = s_data.get("dumps_b")
                                start_dumps = s_data.get("dumps")
                        if e_k is not None:
                            e_data = load_tpcds_ckpt_stage(e_k)
                            if e_data is not None:
                                stop_k = e_k
                                stop_dumps_b = e_data.get("dumps_b")
                                stop_dumps = e_data.get("dumps")

        # Stage 2a FIRST (~0.15s): fail fast on broken Python codegen before running Stage-1 optimize!
        if not skip_pygen:
            py_gen = Python().generator(identify=True, comments=False)
            gen_pairs = pickle.loads(s1["gen_pairs_b"]) if "gen_pairs_b" in s1 else s1["gen_pairs"]
            for expr_dump, expected_py in gen_pairs:
                if py_gen.generate(serde_load(expr_dump)) != expected_py:
                    _why(f"{suite_name}: python codegen changed")
                    return False

        def same_rows_as_pinned(index: int, sql: str, opt_expr=None) -> bool:
            """Run one query the way the real test does and compare its rows to the pinned run."""
            from sqlglot.executor import execute
            nonlocal s2_meta

            if has_s2_blobs:
                if s2_meta is None:
                    with open(TPC_S2_BLOBS_DIR / suite_name / "meta.pkl", "rb") as handle:
                        s2_meta = pickle.load(handle)
                with open(tpc_s2_query_file(suite_name, index), "rb") as handle:
                    q2 = pickle.load(handle)
                schema_map = s2_meta["schema"]
                tables = {}
                for name in q2["table_names"]:
                    if name not in s2_tables_cache:
                        with open(tpc_s2_table_file(suite_name, name), "rb") as handle:
                            s2_tables_cache[name] = pickle.load(handle)
                    tables[name] = Table(columns=schema_map[name], rows=list(s2_tables_cache[name]))
                if opt_expr is not None:
                    res = PythonExecutor(tables=ensure_tables(tables)).execute(Plan(opt_expr))
                else:
                    res = execute(parse_one(sql), schema=schema_map, tables=tables)
                return _rows_equal(res.columns, res.rows, q2["columns"], q2["rows"])

            if not s2_full_cache:
                with open(TPC_STAGE2_FILE, "rb") as handle:
                    s2_full_cache.append(pickle.load(handle)[suite_name])
            s2 = s2_full_cache[0]
            q2 = s2["queries"][index]
            tables = {
                name: Table(columns=s2["schema"][name], rows=list(s2["tables_raw"][name]))
                for name in q2["table_names"]
            }
            if opt_expr is not None:
                res = PythonExecutor(tables=ensure_tables(tables)).execute(Plan(opt_expr))
            else:
                res = execute(parse_one(sql), schema=s2["schema"], tables=tables)
            return _rows_equal(res.columns, res.rows, q2["columns"], q2["rows"])

        # Stage 1: fail-fast on changed plans via immediate lazy per-query same_rows_as_pinned.
        duck_changed: list[tuple[str, str]] = []
        reexec_count = 0
        reexec_cpu = 0.0

        def _check_changed_query(index: int, sql: str, opt_expr=None) -> bool:
            nonlocal reexec_count, reexec_cpu
            reexec_count += 1
            if reexec_count > TPC_REEXEC_MAX_QUERIES or reexec_cpu > TPC_REEXEC_BUDGET_S:
                _why(f"{suite_name}: reexec limit ({reexec_count} queries, {reexec_cpu:.2f}s)")
                return False
            t0 = time.process_time()
            ok = same_rows_as_pinned(index, sql, opt_expr)
            reexec_cpu += time.process_time() - t0
            if not ok:
                _why(f"{suite_name} q{index}: rows differ")
            return ok

        # Order queries by ascending Stage-2b runtime so broken commits fail fast on cheap queries.
        meta_for_order = load_tpcds_ckpts_meta()
        q_cost_ms = (meta_for_order.get("s2b_query_ms") or {}).get(suite_name) if meta_for_order else None
        q_order = list(range(len(s1["queries"])))
        if q_cost_ms and len(q_cost_ms) == len(q_order):
            q_order.sort(key=lambda i: (q_cost_ms[i], i))

        if not (skip_opt and skip_duck and skip_plan):
            for index in q_order:
                q = s1["queries"][index]
                if query_mask is not None and not (query_mask >> index & 1):
                    continue
                if skip_opt:
                    if not q["execute"]:
                        continue
                    if not skip_duck:
                        duck_sql = duck_gen.generate(parse_one(q["sql"]))
                        if duck_sql != q["duckdb_sql"]:
                            duck_changed.append((duck_sql, q["duckdb_sql"]))
                    if not skip_plan:
                        opt_expr = optimize(parse_one(q["sql"]), ensured_schema, leave_tables_isolated=True)
                        if plan_signature(Plan(opt_expr), serde_dump) != q["plan_sig"]:
                            if not _check_changed_query(index, q["sql"], opt_expr):
                                return False
                    continue

                if start_k is not None and (start_dumps_b is not None or start_dumps is not None):
                    if not q["execute"]:
                        continue
                    if not skip_duck:
                        duck_sql = duck_gen.generate(parse_one(q["sql"]))
                        if duck_sql != q["duckdb_sql"]:
                            duck_changed.append((duck_sql, q["duckdb_sql"]))
                    raw_ckpt = pickle.loads(start_dumps_b[index]) if start_dumps_b is not None else start_dumps[index]
                    expr = serde_load(raw_ckpt)
                    if start_k == 0 and scope_patched:
                        if sorted({t.name for t in find_tables(expr)}) != q["table_names"]:
                            _why(f"{suite_name} q{index}: tables changed")
                            return False
                    if stop_k is not None and (stop_dumps_b is not None or stop_dumps is not None):
                        expr = optimize(
                            expr,
                            ensured_schema,
                            rules=RULES[start_k:stop_k],
                            leave_tables_isolated=True,
                            copy=False,
                        )
                        stop_raw = pickle.loads(stop_dumps_b[index]) if stop_dumps_b is not None else stop_dumps[index]
                        if serde_dump(expr) == stop_raw:
                            continue
                        opt_expr = optimize(
                            expr,
                            ensured_schema,
                            rules=RULES[stop_k:],
                            leave_tables_isolated=True,
                            copy=False,
                        )
                    else:
                        opt_expr = optimize(
                            expr,
                            ensured_schema,
                            rules=RULES[start_k:],
                            leave_tables_isolated=True,
                            copy=False,
                        )
                    if serde_dump(opt_expr) != _q_opt_dump(q):
                        if not _check_changed_query(index, q["sql"], opt_expr):
                            return False
                        continue
                    if plan_untouched or plan_signature(Plan(opt_expr), serde_dump) == q["plan_sig"]:
                        continue
                    opt_expr = optimize(parse_one(q["sql"]), ensured_schema, leave_tables_isolated=True)
                else:
                    expr = parse_one(q["sql"])
                    if not q["execute"]:
                        continue
                    if sorted({t.name for t in find_tables(expr)}) != q["table_names"]:
                        _why(f"{suite_name} q{index}: tables changed")
                        return False
                    if not skip_duck:
                        duck_sql = duck_gen.generate(expr)
                        if duck_sql != q["duckdb_sql"]:
                            duck_changed.append((duck_sql, q["duckdb_sql"]))
                    if stop_k is not None and (stop_dumps_b is not None or stop_dumps is not None):
                        if stop_k == 0:
                            stop_raw = pickle.loads(stop_dumps_b[index]) if stop_dumps_b is not None else stop_dumps[index]
                            if serde_dump(expr) == stop_raw:
                                continue
                            opt_expr = optimize(expr, ensured_schema, leave_tables_isolated=True)
                        else:
                            mid = optimize(expr, ensured_schema, rules=RULES[:stop_k], leave_tables_isolated=True)
                            stop_raw = pickle.loads(stop_dumps_b[index]) if stop_dumps_b is not None else stop_dumps[index]
                            if serde_dump(mid) == stop_raw:
                                continue
                            opt_expr = optimize(
                                mid,
                                ensured_schema,
                                rules=RULES[stop_k:],
                                leave_tables_isolated=True,
                                copy=False,
                            )
                    else:
                        opt_expr = optimize(expr, ensured_schema, leave_tables_isolated=True)
                if serde_dump(opt_expr) == _q_opt_dump(q) and (
                    plan_untouched or plan_signature(Plan(opt_expr), serde_dump) == q["plan_sig"]
                ):
                    continue
                if not _check_changed_query(index, q["sql"], opt_expr):
                    return False

        if duck_changed and not _same_duckdb_results(suite_name, duck_changed):
            _why(f"{suite_name}: {len(duck_changed)} duckdb sql changed, results differ")
            return False
        _why(f"{suite_name}: pass ({reexec_count} reexec, {len(duck_changed)} duck changed)")
        return True

    if mode == "fast_b":
        # Stage 2 only: rebuild Plan from cached optimized AST and run patched PythonExecutor,
        # loading queries & tables lazily in ascending runtime order so broken commits fail fast.
        if has_s2_blobs:
            with open(TPC_S2_BLOBS_DIR / suite_name / "meta.pkl", "rb") as handle:
                s2_meta = pickle.load(handle)
            schema = s2_meta["schema"]
            q_ms = s2_meta.get("s2b_query_ms")
            order = list(range(s2_meta["n_queries"]))
            if q_ms and len(q_ms) == len(order):
                order.sort(key=lambda i: (q_ms[i], i))
            tables_by_name: dict[str, Table] = {}
            for idx in order:
                if s2b_query_mask is not None and not (s2b_query_mask >> idx & 1):
                    continue
                with open(tpc_s2_query_file(suite_name, idx), "rb") as handle:
                    q = pickle.load(handle)
                if not q["execute"]:
                    continue
                for name in q["table_names"]:
                    if name not in tables_by_name:
                        with open(tpc_s2_table_file(suite_name, name), "rb") as handle:
                            rows = pickle.load(handle)
                        tables_by_name[name] = Table(columns=schema[name], rows=list(rows))
                opt_expr = serde_load(q["opt_dump"])
                plan = Plan(opt_expr)
                tables_ = ensure_tables({name: tables_by_name[name] for name in q["table_names"]})
                res = PythonExecutor(tables=tables_).execute(plan)
                if not _rows_equal(res.columns, res.rows, q["columns"], q["rows"]):
                    return False
            return True

        with open(TPC_STAGE2_FILE, "rb") as handle:
            s2 = pickle.load(handle)[suite_name]
        schema = s2["schema"]
        tables_by_name = {
            name: Table(columns=schema[name], rows=list(rows))
            for name, rows in s2["tables_raw"].items()
        }
        for idx, q in enumerate(s2["queries"]):
            if s2b_query_mask is not None and not (s2b_query_mask >> idx & 1):
                continue
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

    # Full cached path: Stage 1 + Stage 2 + DuckDB transpile parity (lazy blobs, cheapest queries first).
    s1 = _load_s1()
    full_mask = (query_mask | s2b_query_mask) if (query_mask is not None and s2b_query_mask is not None) else None
    if has_s2_blobs:
        with open(TPC_S2_BLOBS_DIR / suite_name / "meta.pkl", "rb") as handle:
            s2_meta = pickle.load(handle)
        schema = s2_meta["schema"]
        ensured_schema = ensure_schema(schema)
        q_ms = s2_meta.get("s2b_query_ms")
        order = list(range(len(s1["queries"])))
        if q_ms and len(q_ms) == len(order):
            order.sort(key=lambda i: (q_ms[i], i))
        tables_by_name = {}
        for idx in order:
            if full_mask is not None and not (full_mask >> idx & 1):
                continue
            q1 = s1["queries"][idx]
            expr = parse_one(q1["sql"])
            if not q1["execute"]:
                continue
            if duck_gen.generate(expr) != q1["duckdb_sql"]:
                return False
            with open(tpc_s2_query_file(suite_name, idx), "rb") as handle:
                q2 = pickle.load(handle)
            q_tables = {}
            for t in find_tables(expr):
                if t.name not in tables_by_name:
                    with open(tpc_s2_table_file(suite_name, t.name), "rb") as handle:
                        rows = pickle.load(handle)
                    tables_by_name[t.name] = Table(columns=schema[t.name], rows=list(rows))
                q_tables[t.name] = tables_by_name[t.name]
            tables_ = ensure_tables(q_tables)
            if tables_.supported_table_args and tables_.supported_table_args != ensured_schema.supported_table_args:
                return False
            opt_expr = optimize(expr, ensured_schema, leave_tables_isolated=True)
            plan = Plan(opt_expr)
            res = PythonExecutor(tables=tables_).execute(plan)
            if not _rows_equal(res.columns, res.rows, q2["columns"], q2["rows"]):
                return False
        return True

    with open(TPC_STAGE2_FILE, "rb") as handle:
        s2 = pickle.load(handle)[suite_name]
    schema = s2["schema"]
    tables_by_name = {
        name: Table(columns=schema[name], rows=list(rows))
        for name, rows in s2["tables_raw"].items()
    }
    ensured_schema = ensure_schema(schema)
    for q1, q2 in zip(s1["queries"], s2["queries"]):
        expr = parse_one(q1["sql"])
        if not q1["execute"]:
            continue
        if duck_gen.generate(expr) != q1["duckdb_sql"]:
            return False
        q_tables = {t.name: tables_by_name[t.name] for t in find_tables(expr)}
        tables_ = ensure_tables(q_tables)
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
    """Verify TestParser.test_max_nodes without parsing its 10000-column query (~0.2s vs ~12s).

    Every column of the test's `SELECT x=x=...=x, ...` is the same 202-token pattern, so the
    parser counts the same number of nodes per column (100 at the pinned commit; the final
    Select makes it 1_000_001, one over the 1M limit). Node counts of 1-3 column queries give
    that number and show it is constant. Each limit is then checked by starting the node
    counter a whole number of columns in and parsing only the few columns where the full
    query crosses the limit, so the error trips at the same node as in the full parse.
    """
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
    dialect = Dialect.get_or_raise(None)

    def prefix_sql(cols: int) -> str:
        return f"SELECT {','.join(expr for _ in range(cols))}"

    if any(p in ("sqlglot/tokens.py", "sqlglot/tokenizer_core.py") for p in patched_files):
        # A patched tokenizer: tokenize the whole query as the real test does, and check every
        # column tokenizes to the same 202-token pattern.
        toks = dialect.tokenize(prefix_sql(width))
        if len(toks) != 202 * width:
            return False
        pattern = [(tok.token_type, tok.text) for tok in toks[202:404]]
        for col in range(2, width):
            if [(tok.token_type, tok.text) for tok in toks[202 * col : 202 * (col + 1)]] != pattern:
                return False

    def node_count(cols: int) -> int:
        parser = dialect.parser(max_nodes=10**12)
        sql = prefix_sql(cols)
        parser.parse(dialect.tokenize(sql), sql)
        return parser._node_count

    # Nodes per column must be constant, and enough for 10000 columns to exceed 1M.
    c1, c2, c3 = node_count(1), node_count(2), node_count(3)
    per_col = c2 - c1
    if per_col <= 0 or c3 - c2 != per_col or c1 + (width - 1) * per_col <= 1_000_000:
        return False

    parser_cls = type(dialect.parser())
    own_reset = parser_cls.__dict__.get("reset")
    orig_reset = parser_cls.reset

    def limit_error(skip_cols: int, parse_cols: int, max_nodes: int) -> str | None:
        """parse_one of `parse_cols` columns with the counter starting `skip_cols` columns in."""

        def fast_forward_reset(self) -> None:
            orig_reset(self)
            self._node_count = skip_cols * per_col

        parser_cls.reset = fast_forward_reset
        try:
            parse_one(prefix_sql(parse_cols), max_nodes=max_nodes)
            return None
        except ParseError as error:
            return str(error)
        finally:
            if own_reset is None:
                del parser_cls.reset
            else:
                parser_cls.reset = own_reset

    # 50k limit: the full query crosses it a few columns after 50_000 // per_col.
    skip = max(0, 50_000 // per_col - 3)
    err50 = limit_error(skip, min(6, width - skip), 50_000)
    if err50 is None or "(50000)" not in err50:
        return False

    # 1M limit: the last columns of the full query, so trailing nodes (the Select) count too.
    err1m = limit_error(width - 6, 6, 1_000_000)
    if err1m is None or "Maximum number of AST nodes" not in err1m:
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
