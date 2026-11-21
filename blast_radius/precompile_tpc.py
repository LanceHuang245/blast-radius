"""Build-time TPC-H / TPC-DS baseline precompiler (run inside Dockerfile, 0 repo bytes).

Captures exact pinned-sqlglot baseline tables, optimized plans, Stage-2a Python
codegen outputs, DuckDB transpiled SQL, exact execution outputs, and exact Stage-1
vs Stage-2b function call sets for sound two-stage runtime verification.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
import time
from pathlib import Path

from blast_radius import MAP_FILE, SOURCE

TPC_STAGE1_FILE = MAP_FILE.parent / "tpc_stage1.pkl"
TPC_STAGE2_FILE = MAP_FILE.parent / "tpc_stage2.pkl"
TPC_META_FILE = MAP_FILE.parent / "tpc_meta.pkl"

# Files whose code may run on a Plan (planner, Python codegen, executor and what they call).
# Every quoted literal in them counts as a meta key the executor might read; keys never named
# there (e.g. normalize_identifiers' "dot_parts") cannot change what a plan computes.
EXEC_SCAN_PATHS = (
    "sqlglot/executor/",
    "sqlglot/expressions/",
    "sqlglot/generators/python.py",
    "sqlglot/generators/__init__.py",
    "sqlglot/generator.py",
    "sqlglot/planner.py",
    "sqlglot/dialects/dialect.py",
    "sqlglot/dialects/__init__.py",
    "sqlglot/helper.py",
    "sqlglot/schema.py",
    "sqlglot/serde.py",
    "sqlglot/trie.py",
    "sqlglot/time.py",
    "sqlglot/errors.py",
    "sqlglot/tokens.py",
    "sqlglot/tokenizer_core.py",
    "sqlglot/__init__.py",
    "sqlglot/optimizer/scope.py",
    "sqlglot/optimizer/eliminate_joins.py",
)


def exec_meta_keys(source: Path) -> frozenset[str]:
    """Every string literal in EXEC_SCAN_PATHS, plus each dialect's ORIGINAL_NAME_META_KEY
    (read through a variable by Generator.function_name)."""
    import re

    lit = re.compile(r"""(?<![A-Za-z0-9_])[rbuRBU]?(["'])([A-Za-z0-9_.\-]+)\1""")
    keys: set[str] = set()
    for path in sorted((source / "sqlglot").rglob("*.py")):
        rel = path.relative_to(source).as_posix()
        if rel.startswith(EXEC_SCAN_PATHS):
            keys.update(m.group(2) for m in lit.finditer(path.read_text(encoding="utf-8")))
        if rel.startswith("sqlglot/dialects/"):
            for m in re.finditer(r"ORIGINAL_NAME_META_KEY\s*=\s*([\"'])(\w+)\1", path.read_text(encoding="utf-8")):
                keys.add(m.group(2))
    return frozenset(keys)


def _reduce_meta(payloads: list, keep_keys: frozenset[str]) -> list:
    out = []
    for payload in payloads:
        meta, typ = payload.get("m"), payload.get("t")
        if meta is not None or typ is not None:
            payload = dict(payload)
            if meta is not None:
                kept = {k: v for k, v in meta.items() if k in keep_keys}
                if kept:
                    payload["m"] = kept
                else:
                    del payload["m"]
            if typ is not None:
                payload["t"] = _reduce_meta(typ, keep_keys)
        out.append(payload)
    return out


def exec_dump_fn(dump_fn, keep_keys: frozenset[str]):
    """dump_fn with each node's (and its type's) meta reduced to keys the executor side may read."""
    return lambda expression: _reduce_meta(dump_fn(expression), keep_keys)


def plan_signature(plan, dump_fn) -> tuple:
    """Order-independent canonical signature of a sqlglot Plan DAG."""
    visited: dict[int, bytes] = {}

    def encode_step(step) -> bytes:
        sid = id(step)
        if sid in visited:
            return visited[sid]
        # Canonicalize dependency set by child content digest so memory addresses never leak in.
        dep_digests = tuple(sorted(encode_step(d) for d in step.dependencies))
        attrs: list[tuple] = [
            ("type", type(step).__name__),
            ("type_name", getattr(step, "type_name", type(step).__name__)),
            ("name", step.name),
            ("limit", step.limit),
            ("offset", step.offset),
            ("deps", dep_digests),
            ("projections", tuple(dump_fn(p) for p in step.projections)),
            ("condition", dump_fn(step.condition) if step.condition is not None else None),
        ]
        for extra in (
            "source",
            "source_name",
            "left",
            "right",
            "distinct",
            "side",
            "key",
            "group",
            "aggregations",
            "operands",
        ):
            if hasattr(step, extra):
                val = getattr(step, extra)
                if val is None or isinstance(val, (str, int, float, bool)):
                    attrs.append((extra, val))
                elif isinstance(val, dict):
                    attrs.append((extra, tuple((k, dump_fn(v) if hasattr(v, "args") else repr(v)) for k, v in val.items())))
                elif isinstance(val, (list, tuple)):
                    attrs.append(
                        (
                            extra,
                            tuple(
                                dump_fn(x) if hasattr(x, "args")
                                else (dump_fn(x[0]), x[1]) if isinstance(x, tuple) and len(x) == 2 and hasattr(x[0], "args")
                                else repr(x)
                                for x in val
                            ),
                        )
                    )
                elif hasattr(val, "args"):
                    attrs.append((extra, dump_fn(val)))
                else:
                    attrs.append((extra, repr(val)))
        if hasattr(step, "joins"):
            joins_sig = []
            for jname, jspec in step.joins.items():
                joins_sig.append((
                    jname,
                    jspec.get("side"),
                    tuple(dump_fn(x) for x in jspec.get("source_key", ())),
                    tuple(dump_fn(x) for x in jspec.get("join_key", ())),
                    dump_fn(jspec["condition"]) if jspec.get("condition") is not None else None,
                ))
            attrs.append(("joins", tuple(joins_sig)))
        digest = hashlib.blake2b(pickle.dumps(tuple(attrs), protocol=4), digest_size=16).digest()
        visited[sid] = digest
        return digest

    root_digest = encode_step(plan.root)
    cte_sig = dump_fn(plan.ctes) if plan.ctes is not None else None
    return (root_digest, cte_sig, len(plan.dag))


def _record_suite(
    suite_name: str,
    schema: dict,
    tables: dict,
    sqls: list,
    stage1_funcs: set[tuple[str, str]],
    stage2b_funcs: set[tuple[str, str]],
    exec_files: set[str] | None = None,
    exec_dump=None,
) -> tuple[dict, dict]:
    from sqlglot import find_tables, parse_one, transpile
    from sqlglot.executor.python import PythonExecutor
    from sqlglot.executor.table import Table, ensure_tables
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema
    from sqlglot.serde import dump as serde_dump

    src_prefix = str(SOURCE.resolve()) + "/"
    phase = [""]  # "s1", "s2a", "s2b"

    def profiler(frame, event, _arg):
        if event != "call":
            return None
        fn = frame.f_code.co_filename
        if fn.startswith(src_prefix):
            rel = fn[len(src_prefix) :]
            if rel.startswith("sqlglot/"):
                pair = (rel, frame.f_code.co_name)
                if exec_files is not None and phase[0] in ("s2a", "s2b"):
                    exec_files.add(rel)
                if phase[0] == "s2b":
                    stage2b_funcs.add(pair)
                elif phase[0] in ("s1", "s2a"):
                    stage1_funcs.add(pair)
        return None

    # Hook PythonExecutor.generate to capture exact Stage-2a expression inputs & outputs
    # and separate Stage-2a codegen from Stage-2b row execution.
    orig_generate = PythonExecutor.generate
    gen_seen: dict[bytes, tuple[list, str]] = {}
    gen_in_depth = [0]

    def hooked_generate(self, expression):
        prev = phase[0]
        try:
            if expression:
                replaced = self._replace_subqueries(expression)
                # Only tag Stage-2a around generator.generate so _replace_subqueries stays in s2b_funcs.
                gen_in_depth[0] += 1
                phase[0] = "s2a"
                try:
                    py_str = self.generator.generate(replaced)
                finally:
                    gen_in_depth[0] -= 1
                    phase[0] = prev
                key = py_str.encode("utf-8")
                if key not in gen_seen:
                    gen_seen[key] = (serde_dump(replaced), py_str)
                return compile(py_str, py_str, "eval", optimize=2)
            return None
        finally:
            phase[0] = prev

    PythonExecutor.generate = hooked_generate

    queries_s1 = []
    queries_s2 = []
    try:
        sys.setprofile(profiler)
        # Exercise Table construction under s2b so Table.__init__ is always tracked in stage2b_funcs.
        phase[0] = "s2b"
        for tname, tbl in tables.items():
            Table(columns=schema[tname], rows=list(tbl.rows[:1]))
        phase[0] = ""

        for args, sql, _ in sqls:
            is_exec = bool(args.get("execute"))
            phase[0] = "s1"
            expr = parse_one(sql)
            if not is_exec:
                queries_s1.append({"sql": sql, "execute": False})
                queries_s2.append({"execute": False})
                continue

            duck_sql = transpile(sql, write="duckdb")[0]
            table_names = sorted({t.name for t in find_tables(expr)})
            ensured_schema = ensure_schema(schema)
            opt_expr = optimize(expr, ensured_schema, leave_tables_isolated=True)
            plan = Plan(opt_expr)
            phase[0] = ""
            opt_dump = serde_dump(opt_expr)
            p_sig = plan_signature(plan, serde_dump)
            p_sig_x = None
            if exec_dump is not None:
                sys.setprofile(None)  # pure Python dumping: profiling it would only slow the build
                p_sig_x = plan_signature(plan, exec_dump)
                sys.setprofile(profiler)

            # Stage 2b covers ensure_tables, PythonExecutor init, and row execution.
            phase[0] = "s2b"
            q_tables = {name: tables[name] for name in table_names}
            tables_ = ensure_tables(q_tables)
            executor = PythonExecutor(tables=tables_)
            result = executor.execute(plan)
            phase[0] = ""

            queries_s1.append({
                "sql": sql,
                "execute": True,
                "table_names": table_names,
                "duckdb_sql": duck_sql,
                "opt_dump": opt_dump,
                "plan_sig": p_sig,
                "plan_sig_x": p_sig_x,
            })
            queries_s2.append({
                "execute": True,
                "table_names": table_names,
                "opt_dump": opt_dump,
                "columns": tuple(result.columns),
                "rows": list(result.rows),
            })
    finally:
        sys.setprofile(None)
        PythonExecutor.generate = orig_generate

    tables_raw = {name: list(tbl.rows) for name, tbl in tables.items()}
    s1_payload = {
        "name": suite_name,
        "schema": schema,
        "queries": queries_s1,
        "gen_pairs": list(gen_seen.values()),
    }
    s2_payload = {
        "name": suite_name,
        "schema": schema,
        "tables_raw": tables_raw,
        "queries": queries_s2,
    }
    return s1_payload, s2_payload


def main() -> None:
    t0 = time.perf_counter()
    os.chdir(SOURCE)
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))

    from tests.helpers import TPCDS_SCHEMA, TPCH_SCHEMA
    from tests.test_executor import TestExecutor

    from sqlglot.serde import dump as serde_dump

    keep_keys = exec_meta_keys(SOURCE)
    exec_dump = exec_dump_fn(serde_dump, keep_keys)
    exec_files: set[str] = set()
    TestExecutor.setUpClass()
    try:
        stage1_funcs: set[tuple[str, str]] = set()
        stage2b_funcs: set[tuple[str, str]] = set()

        tpch_s1, tpch_s2 = _record_suite(
            "tpch", TPCH_SCHEMA, TestExecutor.tpch_tables, TestExecutor.tpch_sqls, stage1_funcs, stage2b_funcs,
            exec_files, exec_dump,
        )
        tpcds_s1, tpcds_s2 = _record_suite(
            "tpcds", TPCDS_SCHEMA, TestExecutor.tpcds_tables, TestExecutor.tpcds_sqls, stage1_funcs, stage2b_funcs,
            exec_files, exec_dump,
        )
    finally:
        TestExecutor.tearDownClass()

    # The reduced signatures are only sound if every file that ran on a plan was scanned for keys.
    unscanned = sorted(f for f in exec_files if not f.startswith(EXEC_SCAN_PATHS))
    if unscanned:
        print(f"precompile_tpc: exec files not scanned for meta keys {unscanned}: no reduced plan signatures", file=sys.stderr)
        for s1 in (tpch_s1, tpcds_s1):
            for q in s1["queries"]:
                q.pop("plan_sig_x", None)

    # Save small metadata, Stage-1 (~2MB) and Stage-2 (~33MB) caches separately for instant fast_a loads.
    meta = {
        "stage1_funcs": sorted(stage1_funcs),
        "stage2b_funcs": sorted(stage2b_funcs),
        "exec_files": sorted(exec_files),
        "exec_meta_keys": None if unscanned else sorted(keep_keys),
    }
    with open(TPC_META_FILE, "wb") as handle:
        pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(TPC_STAGE1_FILE, "wb") as handle:
        pickle.dump({"tpch": tpch_s1, "tpcds": tpcds_s1}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(TPC_STAGE2_FILE, "wb") as handle:
        pickle.dump({"tpch": tpch_s2, "tpcds": tpcds_s2}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"precompiled TPC caches in {time.perf_counter() - t0:.1f}s: "
        f"s1={TPC_STAGE1_FILE.stat().st_size / 1e6:.2f} MB, "
        f"s2={TPC_STAGE2_FILE.stat().st_size / 1e6:.2f} MB "
        f"(s1_funcs={len(stage1_funcs)}, s2b_funcs={len(stage2b_funcs)}, "
        f"tpcds_gen_pairs={len(tpcds_s1['gen_pairs'])})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
