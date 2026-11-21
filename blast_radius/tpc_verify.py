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
import time

from blast_radius.precompile_cases import TPCDS_CKPTS_META_FILE, tpcds_ckpt_stage_file
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
    ("tpcds", "fast_a"): 3000,
    ("tpcds", "fast_b"): 12800,
    ("tpcds", "full_cached"): 16500,
    ("tpch", "fast_a"): 450,
    ("tpch", "fast_b"): 1500,
    ("tpch", "full_cached"): 2000,
}

# fast_a re-executes queries whose optimized plan changed; beyond these it keeps the test instead.
TPC_REEXEC_MAX_QUERIES = 40
TPC_REEXEC_BUDGET_S = 10.0

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


def est_tpc_fast_a_ms(
    suite_name: str,
    patched_files: dict[str, str],
    matched_queries: int | None,
    total_queries: int,
) -> int:
    """Estimate fast_a CPU (ms) accounting for untouched sub-stages, checkpoints, and per-query filtering."""
    base = TPC_EST_CPU_MS[(suite_name, "fast_a")]
    if suite_name == "tpcds" and patched_files and all(p.startswith("sqlglot/") for p in patched_files):
        meta = load_tpcds_ckpts_meta()
        if meta is not None:
            patched_set = set(patched_files)
            if patched_set.isdisjoint(meta["s1_opt_files"]):
                base = 350
            else:
                ckpt = select_tpcds_ckpt(patched_files, max_stage=5)
                if ckpt is not None:
                    base = min(base, ckpt[1])
    if matched_queries is not None and total_queries > 0:
        base = max(120, base * (matched_queries + 1) // (total_queries + 1))
    return base


def verify_opt_tpcds(patched_files: dict[str, str], case_indices: set[int] | None = None) -> bool:
    """Verify TestOptimizer.test_tpcds from the highest untouched optimizer.RULES checkpoint."""
    ckpt_info = select_tpcds_ckpt(patched_files, max_stage=14)
    if ckpt_info is None:
        return False
    k, _ = ckpt_info
    stage_data = load_tpcds_ckpt_stage(k)
    if stage_data is None:
        return False
    from sqlglot.optimizer import optimize
    from sqlglot.optimizer.optimizer import RULES
    from sqlglot.schema import ensure_schema
    from sqlglot.serde import load as serde_load

    dumps = stage_data["dumps"]
    expected = stage_data["expected"]
    ensured = ensure_schema(stage_data["schema"]) if k < 14 else None
    indices = sorted(case_indices) if case_indices is not None else range(len(expected))
    for index in indices:
        expr = serde_load(dumps[index])
        if k < 14:
            expr = optimize(expr, ensured, rules=RULES[k:], copy=False)
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
    import duckdb
    from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, TPCH_SCHEMA

    schema, folder = (TPCDS_SCHEMA, "tpc-ds") if suite_name == "tpcds" else (TPCH_SCHEMA, "tpc-h")
    conn = duckdb.connect()
    try:
        # Same views as TestExecutor.setUpClass.
        for table, columns in schema.items():
            conn.execute(
                f"CREATE VIEW {table} AS SELECT * FROM READ_CSV("
                f"'{FIXTURES_DIR}/optimizer/{folder}/{table}.csv.gz', delim='|', header=True, columns={columns})"
            )
        for new_sql, old_sql in pairs:
            new = conn.execute(new_sql)
            new_desc, new_rows = [d[:2] for d in new.description], new.fetchall()
            old = conn.execute(old_sql)
            if new_desc != [d[:2] for d in old.description] or new_rows != old.fetchall():
                return False
        return True
    finally:
        conn.close()


def verify_tpc_test(
    suite_name: str,
    mode: str,
    query_mask: int | None = None,
    patched_files: dict[str, str] | None = None,
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

    if mode == "fast_a":
        with open(TPC_STAGE1_FILE, "rb") as handle:
            s1 = pickle.load(handle)[suite_name]
        ensured_schema = ensure_schema(s1["schema"])
        s2_cache: list = []

        skip_opt = False
        skip_duck = False
        skip_plan = False
        skip_pygen = False
        ckpt_k: int | None = None
        ckpt_dumps: list | None = None
        if suite_name == "tpcds" and patched_files and all(p.startswith("sqlglot/") for p in patched_files):
            meta = load_tpcds_ckpts_meta()
            if meta is not None:
                patched_set = set(patched_files)
                skip_opt = patched_set.isdisjoint(meta["s1_opt_files"])
                skip_duck = patched_set.isdisjoint(meta["s1_duck_files"])
                skip_plan = skip_opt and patched_set.isdisjoint(meta["s1_plan_files"])
                skip_pygen = patched_set.isdisjoint(meta["s1_pygen_files"])
                if not skip_opt:
                    ckpt_info = select_tpcds_ckpt(patched_files, max_stage=5)
                    if ckpt_info is not None:
                        stage_data = load_tpcds_ckpt_stage(ckpt_info[0])
                        if stage_data is not None:
                            ckpt_k = ckpt_info[0]
                            ckpt_dumps = stage_data["dumps"]

        def same_rows_as_pinned(index: int, sql: str) -> bool:
            """Run one query the way the real test does and compare its rows to the pinned run."""
            from sqlglot.executor import execute

            if not s2_cache:
                with open(TPC_STAGE2_FILE, "rb") as handle:
                    s2_cache.append(pickle.load(handle)[suite_name])
            s2 = s2_cache[0]
            q2 = s2["queries"][index]
            tables = {
                name: Table(columns=s2["schema"][name], rows=list(s2["tables_raw"][name]))
                for name in q2["table_names"]
            }
            res = execute(parse_one(sql), schema=s2["schema"], tables=tables)
            return _rows_equal(res.columns, res.rows, q2["columns"], q2["rows"])

        # Stage 1: parse queries + find_tables + DuckDB transpile + ensure_schema + optimize + Plan.
        # Only queries matching `query_mask` (when provided) and stages touching patched files run.
        duck_changed: list[tuple[str, str]] = []
        reexec: list[tuple[int, str]] = []
        if not (skip_opt and skip_duck and skip_plan):
            for index, q in enumerate(s1["queries"]):
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
                        opt_expr = serde_load(q["opt_dump"])
                        if plan_signature(Plan(opt_expr), serde_dump) != q["plan_sig"]:
                            reexec.append((index, q["sql"]))
                    continue

                if ckpt_k is not None and ckpt_dumps is not None:
                    if not q["execute"]:
                        continue
                    if not skip_duck:
                        duck_sql = duck_gen.generate(parse_one(q["sql"]))
                        if duck_sql != q["duckdb_sql"]:
                            duck_changed.append((duck_sql, q["duckdb_sql"]))
                    opt_expr = optimize(
                        serde_load(ckpt_dumps[index]),
                        ensured_schema,
                        rules=RULES[ckpt_k:],
                        leave_tables_isolated=True,
                        copy=False,
                    )
                else:
                    expr = parse_one(q["sql"])
                    if not q["execute"]:
                        continue
                    if sorted({t.name for t in find_tables(expr)}) != q["table_names"]:
                        return False
                    if not skip_duck:
                        duck_sql = duck_gen.generate(expr)
                        if duck_sql != q["duckdb_sql"]:
                            duck_changed.append((duck_sql, q["duckdb_sql"]))
                    opt_expr = optimize(expr, ensured_schema, leave_tables_isolated=True)
                if serde_dump(opt_expr) == q["opt_dump"] and plan_signature(Plan(opt_expr), serde_dump) == q["plan_sig"]:
                    continue
                reexec.append((index, q["sql"]))

        # Stage 2a: Python().generator codegen parity on every expression generated during execution.
        if not skip_pygen:
            py_gen = Python().generator(identify=True, comments=False)
            for expr_dump, expected_py in s1["gen_pairs"]:
                if py_gen.generate(serde_load(expr_dump)) != expected_py:
                    return False

        # Re-executing changed queries costs CPU close to what dropping the test saves: give up
        # (keeping the test) when many changed or the budget runs out.
        if len(reexec) > TPC_REEXEC_MAX_QUERIES:
            return False
        if duck_changed and not _same_duckdb_results(suite_name, duck_changed):
            return False
        start = time.process_time()
        for index, sql in reexec:
            if time.process_time() - start > TPC_REEXEC_BUDGET_S or not same_rows_as_pinned(index, sql):
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
