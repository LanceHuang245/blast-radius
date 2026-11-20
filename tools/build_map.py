"""Turns one instrumented run of sqlglot's suite into map/impact.json.gz.

Usage (inside the mapping image):
    python build_map.py <sqlglot checkout> <map_out dir> <tests.json> <output file>

Everything here is derived from the pinned sqlglot commit alone: line coverage
per test, files each test read, classes each test used, and the module import
graph. Nothing is derived from any patch.
"""

from __future__ import annotations

import ast
import gzip
import json
import os
import sys
from collections import defaultdict

from coverage import CoverageData


def main() -> None:
    src, run_dir, tests_json, output = sys.argv[1:5]
    test_ids = [t["id"] for t in json.load(open(tests_json))["tests"]]
    index = {test_id: i for i, test_id in enumerate(test_ids)}

    by_class: dict[str, int] = defaultdict(int)
    by_file: dict[str, int] = defaultdict(int)
    for test_id, i in index.items():
        by_class[test_id.rsplit("::", 1)[0]] |= 1 << i
        by_file[test_id.split("::", 1)[0]] |= 1 << i

    def tests_for(context: str) -> int:
        """A context's tests: run -> that test; setup/teardown -> its class; collect -> its file."""
        if not context:
            return 0
        if context.startswith("collect|"):
            return by_file.get(context[len("collect|"):], 0)
        nodeid, _, phase = context.rpartition("|")
        if phase == "run":
            return 1 << index[nodeid] if nodeid in index else 0
        return by_class.get(nodeid.rsplit("::", 1)[0], 0)

    data = CoverageData(basename=os.path.join(run_dir, "coverage.db"))
    data.read()

    lines: dict[str, dict[int, int]] = {}
    import_lines: dict[str, list[int]] = {}
    covered_tests = 0
    for measured in data.measured_files():
        rel = os.path.relpath(measured, src).replace(os.sep, "/")
        per_line: dict[int, int] = {}
        at_import: list[int] = []
        for lineno, contexts in (data.contexts_by_lineno(measured) or {}).items():
            mask = 0
            for context in contexts:
                if context == "":
                    at_import.append(lineno)
                else:
                    mask |= tests_for(context)
            if mask:
                per_line[lineno] = mask
                covered_tests |= mask
        lines[rel] = per_line
        import_lines[rel] = sorted(at_import)

    usage = json.load(open(os.path.join(run_dir, "usage.json")))
    paths: dict[str, int] = defaultdict(int)
    for context, entries in usage["opens"].items():
        mask = tests_for(context)
        for entry in entries:
            paths[entry] |= mask
    classes: dict[str, int] = defaultdict(int)
    for context, names in usage["classes"].items():
        mask = tests_for(context)
        for name in names:
            classes[name] |= mask

    importers = import_graph(src)

    # Interning: each distinct test set is stored once, as a hex bitset.
    sets: list[str] = []
    seen: dict[int, int] = {}

    def intern(mask: int) -> int:
        if mask not in seen:
            seen[mask] = len(sets)
            sets.append(format(mask, "x"))
        return seen[mask]

    result = {
        "tests": test_ids,
        "lines": {f: {str(n): intern(m) for n, m in sorted(per.items())} for f, per in sorted(lines.items())},
        "import_lines": import_lines,
        "paths": {p: intern(m) for p, m in sorted(paths.items()) if m},
        "classes": {c: intern(m) for c, m in sorted(classes.items()) if m},
        "importers": importers,
        "uncovered": [t for t, i in index.items() if not covered_tests >> i & 1],
        "outcomes": {k: v for k, v in usage.get("outcomes", {}).items() if v != "passed"},
    }
    result["sets"] = sets
    with gzip.open(output, "wt", compresslevel=9) as handle:
        json.dump(result, handle, separators=(",", ":"))
    print(
        f"{len(test_ids)} tests, {len(result['uncovered'])} with no coverage, "
        f"{len(sets)} distinct sets, {os.path.getsize(output) / 1e6:.2f} MB",
        file=sys.stderr,
    )


def module_name(rel: str) -> str:
    name = rel[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def import_graph(src: str) -> dict[str, list[str]]:
    """file -> files that import it with a module-level import (TYPE_CHECKING blocks skipped)."""
    files = {}
    for top in ("sqlglot", "tests"):
        for root, _, names in os.walk(os.path.join(src, top)):
            for name in names:
                if name.endswith(".py"):
                    rel = os.path.relpath(os.path.join(root, name), src).replace(os.sep, "/")
                    files[module_name(rel)] = rel

    importers: dict[str, set[str]] = defaultdict(set)
    for module, rel in files.items():
        try:
            tree = ast.parse(open(os.path.join(src, rel), encoding="utf-8").read())
        except SyntaxError:
            continue
        package = module if rel.endswith("__init__.py") else module.rpartition(".")[0]
        for node in module_level_imports(tree.body):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    parts = parts[: len(parts) - node.level + 1]
                    base = ".".join(parts + ([base] if base else []))
                targets = [base] + [f"{base}.{alias.name}" for alias in node.names]
            for target in targets:
                # importing a.b.c also runs a and a.b
                pieces = target.split(".")
                for k in range(1, len(pieces) + 1):
                    hit = files.get(".".join(pieces[:k]))
                    if hit and hit != rel:
                        importers[hit].add(rel)
    return {k: sorted(v) for k, v in sorted(importers.items())}


def module_level_imports(body):
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.If):
            test = ast.unparse(node.test)
            if "TYPE_CHECKING" not in test:
                yield from module_level_imports(node.body)
            yield from module_level_imports(node.orelse)
        elif isinstance(node, ast.Try):
            yield from module_level_imports(node.body)
            for handler in node.handlers:
                yield from module_level_imports(handler.body)
            yield from module_level_imports(node.orelse)
            yield from module_level_imports(node.finalbody)


if __name__ == "__main__":
    main()
