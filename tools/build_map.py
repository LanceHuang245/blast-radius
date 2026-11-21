"""Turns one instrumented run of sqlglot's suite into map/impact.json.gz.

Usage (inside the mapping image):
    python build_map.py <sqlglot checkout> <map_out dir> <tests.json> <output file>

Everything here is derived from the pinned sqlglot commit alone: line coverage
per test, files each test read, classes/dicts/attributes each test used, static
class references, and module/lazy import graphs. Nothing is derived from any patch.
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
    collect_lines: dict[str, list[int]] = {}
    covered_tests = 0
    for measured in data.measured_files():
        rel = os.path.relpath(measured, src).replace(os.sep, "/")
        per_line: dict[int, int] = {}
        at_import: list[int] = []
        at_collect: list[int] = []
        for lineno, contexts in (data.contexts_by_lineno(measured) or {}).items():
            mask = 0
            for context in contexts:
                if context == "":
                    at_import.append(lineno)
                else:
                    mask |= tests_for(context)
                    if context.startswith("collect|") and (not at_collect or at_collect[-1] != lineno):
                        at_collect.append(lineno)
            if mask:
                per_line[lineno] = mask
                covered_tests |= mask
        lines[rel] = per_line
        import_lines[rel] = sorted(at_import)
        if at_collect:
            collect_lines[rel] = sorted(set(at_collect))

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

    dict_keys: dict[str, int] = defaultdict(int)
    for context, entries in usage.get("dict_reads", {}).items():
        mask = tests_for(context)
        for entry in entries:
            dict_keys[entry] |= mask

    attrs: dict[str, int] = defaultdict(int)
    collect_attrs: set[str] = set()
    for context, entries in usage.get("attr_reads", {}).items():
        mask = tests_for(context)
        for entry in entries:
            attrs[entry] |= mask
            if context.startswith("collect|"):
                collect_attrs.add(entry)

    trees = parse_all_python(src)
    importers, lazy_importers = import_graphs(trees)
    class_refs = static_class_refs(trees, lines)
    import_any_attrs, import_value_attrs = analyze_import_attr_reads(trees, import_lines)
    import_reads = class_import_reads(trees, usage.get("import_reads", []))
    for key, token in class_body_self_reads(trees).items():
        import_reads[key] = sorted(set(import_reads.get(key, [])) | {token})

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
        # Lines and class attributes used while a test module imports. A verifier that
        # imports test modules once and then patches would see stale results for these.
        "collect_lines": collect_lines,
        "collect_attrs": sorted(collect_attrs),
        "paths": {p: intern(m) for p, m in sorted(paths.items()) if m},
        "classes": {c: intern(m) for c, m in sorted(classes.items()) if m},
        "class_refs": {c: intern(m) for c, m in sorted(class_refs.items()) if m},
        "tracked_dicts": usage.get("tracked_dicts", []),
        "dict_keys": {k: intern(m) for k, m in sorted(dict_keys.items()) if m},
        "tracked_attrs": usage.get("tracked_attrs", []),
        "attrs": {k: intern(m) for k, m in sorted(attrs.items()) if m},
        "import_any_attrs": sorted(import_any_attrs),
        "import_value_attrs": sorted(import_value_attrs),
        # Per-class import-time reads observed at run time; the name-based sets above
        # only apply to pre_hook_classes, which existed before recording started.
        "import_reads": import_reads,
        "pre_hook_classes": usage.get("pre_hook_classes", []),
        "importers": importers,
        "lazy_importers": lazy_importers,
        "uncovered": [t for t, i in index.items() if not covered_tests >> i & 1],
        "outcomes": {k: v for k, v in usage.get("outcomes", {}).items() if v != "passed"},
    }
    result["sets"] = sets
    with gzip.open(output, "wt", compresslevel=9) as handle:
        json.dump(result, handle, separators=(",", ":"))
    print(
        f"{len(test_ids)} tests, {len(result['uncovered'])} with no coverage, "
        f"{len(sets)} distinct sets, {len(dict_keys)} dict_keys, {len(attrs)} attrs, "
        f"{len(import_value_attrs)} import_value_attrs, {os.path.getsize(output) / 1e6:.2f} MB",
        file=sys.stderr,
    )
    if result["outcomes"]:
        print(f"Non-passed outcomes: {result['outcomes']}", file=sys.stderr)


def module_name(rel: str) -> str:
    name = rel[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def parse_all_python(src: str) -> dict[str, ast.Module]:
    """Parse every .py file under sqlglot/ and tests/ once."""
    trees: dict[str, ast.Module] = {}
    for top in ("sqlglot", "tests"):
        for root, _, names in os.walk(os.path.join(src, top)):
            for name in names:
                if name.endswith(".py"):
                    rel = os.path.relpath(os.path.join(root, name), src).replace(os.sep, "/")
                    try:
                        trees[rel] = ast.parse(open(os.path.join(src, rel), encoding="utf-8").read())
                    except SyntaxError:
                        continue
    return trees


def _resolve_import_targets(node: ast.Import | ast.ImportFrom, package: str, files: dict[str, str]) -> set[str]:
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
    hits: set[str] = set()
    for target in targets:
        pieces = target.split(".")
        for k in range(1, len(pieces) + 1):
            hit = files.get(".".join(pieces[:k]))
            if hit:
                hits.add(hit)
    return hits


def import_graphs(trees: dict[str, ast.Module]) -> tuple[dict[str, list[str]], dict[str, list[list]]]:
    """Build both module-level and function-level (lazy) import reverse graphs."""
    files = {module_name(rel): rel for rel in trees}
    importers: dict[str, set[str]] = defaultdict(set)
    lazy_importers: dict[str, set[tuple[str, int]]] = defaultdict(set)

    for rel, tree in trees.items():
        mod = module_name(rel)
        package = mod if rel.endswith("__init__.py") else mod.rpartition(".")[0]
        mod_nodes = list(module_level_imports(tree.body))
        mod_node_ids = {id(n) for n in mod_nodes}
        for node in mod_nodes:
            for hit in _resolve_import_targets(node, package, files):
                if hit != rel:
                    importers[hit].add(rel)

        # Also walk function/class bodies for lazy imports, skipping TYPE_CHECKING blocks.
        for node in all_runtime_imports(tree):
            if id(node) in mod_node_ids:
                continue
            for hit in _resolve_import_targets(node, package, files):
                if hit != rel:
                    lazy_importers[hit].add((rel, node.lineno))

    return (
        {k: sorted(v) for k, v in sorted(importers.items())},
        {k: [[f, ln] for f, ln in sorted(v)] for k, v in sorted(lazy_importers.items())},
    )


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


def all_runtime_imports(node: ast.AST):
    """Yield every Import/ImportFrom in the AST outside TYPE_CHECKING guards."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield node
        return
    if isinstance(node, ast.If):
        if "TYPE_CHECKING" in ast.unparse(node.test):
            for child in node.orelse:
                yield from all_runtime_imports(child)
            return
    for child in ast.iter_child_nodes(node):
        yield from all_runtime_imports(child)


def static_class_refs(trees: dict[str, ast.Module], lines: dict[str, dict[int, int]]) -> dict[str, int]:
    """Map each Expression class to tests covering runtime `exp.<Cls>.<member>` accesses.

    Scoped strictly to attribute accesses on `exp.<Cls>` (excluding type annotations and
    `isinstance` checks) so unrelated names like `t.Any`, `TokenType.LT`, or `exp.Table`
    for `executor.table.Table` never inflate blast radius.
    """
    defs_by_name: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)
    for rel, tree in trees.items():
        if not rel.startswith("sqlglot/expressions/"):
            continue
        mod = module_name(rel)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                defs_by_name[node.name].append((f"{mod}:{node.name}", rel, node.lineno, node.end_lineno))

    refs: dict[str, int] = defaultdict(int)
    for rel, tree in trees.items():
        per_file = lines.get(rel)
        if not per_file:
            continue

        imported_exp: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (
                (node.module and (node.module == "sqlglot.expressions" or node.module.startswith("sqlglot.expressions.")))
                or (rel.startswith("sqlglot/expressions/") and node.level >= 1)
            ):
                for al in node.names:
                    if al.name in defs_by_name:
                        imported_exp[al.asname or al.name] = al.name
        if rel.startswith("sqlglot/expressions/"):
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name in defs_by_name:
                    imported_exp[node.name] = node.name

        def record(cls_name: str, ln: int | None) -> None:
            mask = per_file.get(ln, 0) if ln is not None else 0
            if not mask:
                return
            for ckey, def_rel, start, end in defs_by_name[cls_name]:
                if rel == def_rel and start <= ln <= end:
                    continue
                refs[ckey] |= mask

        def walk(node: ast.AST) -> None:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return
            # Match `exp.<Cls>.<member>` or directly imported `<Cls>.<member>`.
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                if (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "exp"
                    and node.value.attr in defs_by_name
                ):
                    record(node.value.attr, getattr(node, "lineno", None))
                elif isinstance(node.value, ast.Name) and node.value.id in imported_exp:
                    record(imported_exp[node.value.id], getattr(node, "lineno", None))
            # Skip type annotations so `exp.DataType.Type` hints are not treated as runtime reads.
            if isinstance(node, ast.arg):
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    walk(dec)
                for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                    walk(d)
                for stmt in node.body:
                    walk(stmt)
                return
            if isinstance(node, ast.AnnAssign):
                walk(node.target)
                if node.value is not None:
                    walk(node.value)
                return
            for ch in ast.iter_child_nodes(node):
                walk(ch)

        walk(tree)
    return refs


def class_import_reads(trees: dict[str, ast.Module], records: list[list]) -> dict[str, list[str]]:
    """`cls|attr` -> key tokens read at import ("-" for a plain attribute read).

    Reads made by `ATTR = {**Base.ATTR, ...}` in a subclass body are skipped: the
    subclass's copy is tracked on its own and its reads already propagate to Base.
    """
    statements: dict[str, dict[int, ast.stmt]] = {}
    parent_maps: dict[str, dict[int, ast.AST]] = {}
    copy_only: dict[tuple[str, int, str], bool] = {}
    out: dict[str, set[str]] = defaultdict(set)
    for cls_key, attr, token, file, lineno in records:
        probe = (file, lineno, attr)
        if probe not in copy_only:
            copy_only[probe] = _only_mro_copies(trees.get(file), file, lineno, attr, statements, parent_maps)
        if not copy_only[probe]:
            out[f"{cls_key}|{attr}"].add(token)
    return {k: sorted(v) for k, v in sorted(out.items())}


def class_body_self_reads(trees: dict[str, ast.Module]) -> dict[str, str]:
    """`cls|attr` -> "*" for attributes a class body reads by bare name (`Y = f(X)`),
    or "k" when every such read only looks at keys (`for k in X`, `k in X`, len(X)).

    That code runs before the class object exists, so the plugin cannot observe it.
    Function and lambda bodies are skipped: they cannot see class scope.
    """
    found: dict[str, str] = {}
    parents: dict[int, ast.AST] = {}

    def visit_class(node: ast.ClassDef, module: str, qualname: str) -> None:
        stack: list[ast.AST] = list(node.body)
        while stack:
            cur = stack.pop()
            if isinstance(cur, ast.ClassDef):
                visit_class(cur, module, f"{qualname}.{cur.name}")
                continue
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stack.extend(cur.decorator_list + cur.args.defaults + [d for d in cur.args.kw_defaults if d])
                continue
            if isinstance(cur, ast.Lambda):
                stack.extend(cur.args.defaults + [d for d in cur.args.kw_defaults if d])
                continue
            if isinstance(cur, ast.Name) and isinstance(cur.ctx, ast.Load):
                key = f"{module}:{qualname}|{cur.id}"
                parent = parents.get(id(cur))
                grandparent = parents.get(id(parent)) if parent is not None else None
                # Skip `ATTR = ATTR.copy()` module-to-class copy in class body.
                if (
                    isinstance(parent, ast.Attribute)
                    and parent.attr == "copy"
                    and isinstance(grandparent, ast.Call)
                ):
                    assign = parents.get(id(grandparent))
                    if (
                        isinstance(assign, ast.Assign)
                        and len(assign.targets) == 1
                        and isinstance(assign.targets[0], ast.Name)
                        and assign.targets[0].id == cur.id
                    ):
                        continue
                token = "k" if _is_key_only_use(cur, parent, grandparent) else "*"
                if found.get(key) != "*":
                    found[key] = token
            stack.extend(ast.iter_child_nodes(cur))

    for rel, tree in trees.items():
        if not rel.startswith("sqlglot/"):
            continue
        module = module_name(rel)
        parents.update({id(c): p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)})
        stack: list[tuple[ast.AST, str]] = [(n, "") for n in tree.body]
        while stack:
            cur, prefix = stack.pop()
            if isinstance(cur, ast.ClassDef):
                visit_class(cur, module, prefix + cur.name)
            elif isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stack.extend((n, f"{prefix}{cur.name}.<locals>.") for n in cur.body)
            else:
                stack.extend((n, prefix) for n in ast.iter_child_nodes(cur) if isinstance(n, ast.stmt))
    return found


def _only_mro_copies(tree, file, lineno, attr, statements, parent_maps) -> bool:
    if tree is None or not lineno:
        return False
    if file not in statements:
        # line -> innermost statement covering it (later-starting / shorter statements win)
        innermost: dict[int, ast.stmt] = {}
        nodes = sorted((n for n in ast.walk(tree) if isinstance(n, ast.stmt)), key=lambda n: (n.lineno, -n.end_lineno))
        for node in nodes:
            for line in range(node.lineno, node.end_lineno + 1):
                innermost[line] = node
        statements[file] = innermost
        parent_maps[file] = {id(c): p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    best = statements[file].get(lineno)
    if best is None:
        return False
    refs = [n for n in ast.walk(best) if isinstance(n, ast.Attribute) and n.attr == attr and isinstance(n.ctx, ast.Load)]
    return bool(refs) and all(_is_mro_dict_copy(n, attr, parent_maps[file]) for n in refs)


def _is_key_only_use(node: ast.AST, parent: ast.AST | None, grandparent: ast.AST | None) -> bool:
    """True when a dict reference only inspects keys/membership/length/truthiness, never values."""
    if parent is None:
        return False
    # `k in d` or `k not in d`
    if isinstance(parent, ast.Compare) and node in parent.comparators:
        if all(isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops):
            return True
    # `for k in d:` or `( ... for k in d )`
    if isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is node:
        return True
    # `bool(d)`, `len(d)`, `set(d)`, `frozenset(d)`, `list(d)`, `tuple(d)`
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name):
        if parent.func.id in ("bool", "len", "set", "frozenset", "list", "tuple") and parent.args == [node]:
            return True
    # `d.keys()`
    if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr == "keys":
        if isinstance(grandparent, ast.Call) and grandparent.func is parent and not grandparent.args:
            return True
    return False


def _is_mro_dict_copy(
    node: ast.AST,
    attr_name: str,
    parent_map: dict[int, ast.AST],
) -> bool:
    """True when `Base.ATTR` is spread or key/value-copied into a subclass's own `ATTR = {...}`."""
    if not isinstance(node, ast.Attribute):
        return False
    src_name = (
        node.value.id
        if isinstance(node.value, ast.Name)
        else node.value.attr if isinstance(node.value, ast.Attribute) else ""
    )
    if not src_name:
        return False

    # Walk up to check two patterns inside a ClassDef:
    # 1. `ATTR = {**Base.ATTR, ...}`
    # 2. `ATTR = {**{k: v for k, v in Base.ATTR.items() if ...}, ...}` or `{k: v for k, v in Base.ATTR.items() ...}`
    p = parent_map.get(id(node))
    assign: ast.AST | None = None
    if isinstance(p, ast.Dict) and node in p.values:
        idx = p.values.index(node)
        if p.keys[idx] is None:
            assign = parent_map.get(id(p))
    elif isinstance(p, ast.Attribute) and p.value is node and p.attr == "items":
        call = parent_map.get(id(p))
        comp = parent_map.get(id(call)) if isinstance(call, ast.Call) else None
        dict_comp = parent_map.get(id(comp)) if isinstance(comp, ast.comprehension) else None
        if isinstance(dict_comp, ast.DictComp):
            outer = parent_map.get(id(dict_comp))
            if isinstance(outer, ast.Dict) and dict_comp in outer.values:
                idx = outer.values.index(dict_comp)
                if outer.keys[idx] is None:
                    assign = parent_map.get(id(outer))
            else:
                assign = outer

    if assign is None:
        return False
    if isinstance(assign, ast.Assign) and len(assign.targets) == 1 and isinstance(assign.targets[0], ast.Name):
        tgt_name = assign.targets[0].id
    elif isinstance(assign, ast.AnnAssign) and isinstance(assign.target, ast.Name):
        tgt_name = assign.target.id
    else:
        return False
    if tgt_name != attr_name:
        return False

    enclosing = parent_map.get(id(assign))
    if not isinstance(enclosing, ast.ClassDef):
        return False
    base_names = {
        (b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else "")
        for b in enclosing.bases
    }
    return src_name in base_names


def analyze_import_attr_reads(
    trees: dict[str, ast.Module],
    import_lines: dict[str, list[int]],
) -> tuple[set[str], set[str]]:
    """Find class attribute names read at import/metaclass/class-definition time.

    Returns:
        import_any_attrs: attributes read at import/metaclass time (excluding MRO dict copies).
        import_value_attrs: attributes whose values (not merely keys) may be read at import/metaclass time.
    """
    any_attrs: set[str] = set()
    value_attrs: set[str] = set()

    for rel, tree in trees.items():
        if not rel.startswith("sqlglot/"):
            continue
        parent_map: dict[int, ast.AST] = {}
        for p in ast.walk(tree):
            for ch in ast.iter_child_nodes(p):
                parent_map[id(ch)] = p

        # Collect nodes that execute at import/class-creation time:
        # module-level code, class bodies (outside methods), and metaclass bodies (including methods).
        import_nodes: list[ast.AST] = []
        metaclass_funcs: set[int] = set()

        def is_metaclass(cls_node: ast.ClassDef) -> bool:
            return any(isinstance(b, ast.Name) and b.id == "type" for b in cls_node.bases) or any(
                isinstance(m, ast.FunctionDef) and m.name in ("__new__", "__init_subclass__")
                for m in cls_node.body
            )

        def collect_import_nodes(node: ast.AST, in_meta: bool) -> None:
            import_nodes.append(node)
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, ast.ClassDef):
                    collect_import_nodes(ch, in_meta or is_metaclass(ch))
                elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if in_meta:
                        metaclass_funcs.add(id(ch))
                        collect_import_nodes(ch, True)
                    else:
                        # Decorators and default argument expressions run at import time; body runs later.
                        for dec in ch.decorator_list:
                            collect_import_nodes(dec, False)
                        for default in ch.args.defaults + [d for d in ch.args.kw_defaults if d is not None]:
                            collect_import_nodes(default, False)
                elif isinstance(ch, ast.Lambda):
                    for default in ch.args.defaults + [d for d in ch.args.kw_defaults if d is not None]:
                        collect_import_nodes(default, False)
                else:
                    collect_import_nodes(ch, in_meta)

        collect_import_nodes(tree, False)

        # Ordinary functions can run at import too (e.g. module-level `exp.maybe_parse(...)`
        # constants construct dialects, tokenizers and parsers). Coverage saw those lines
        # execute with no test context; treat every read on them as an import-time read.
        executed = set(import_lines.get(rel, ()))
        if executed:
            known = {id(n) for n in import_nodes}
            for n in ast.walk(tree):
                if (
                    isinstance(n, (ast.Attribute, ast.Name))
                    and getattr(n, "lineno", None) in executed
                    and id(n) not in known
                ):
                    import_nodes.append(n)

        # Track local alias uses inside metaclass methods (e.g. `numeric_escapes = klass.tokenizer_class.NUMERIC_ESCAPES`).
        func_of: dict[int, int] = {}

        def tag_func(node: ast.AST, fid: int) -> None:
            func_of[id(node)] = fid
            for ch in ast.iter_child_nodes(node):
                next_fid = id(ch) if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fid
                tag_func(ch, next_fid)

        tag_func(tree, 0)

        alias_uses: dict[tuple[int, str], list[tuple[ast.AST, ast.AST | None, ast.AST | None]]] = defaultdict(list)
        for node in import_nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                p = parent_map.get(id(node))
                gp = parent_map.get(id(p)) if p else None
                alias_uses[(func_of.get(id(node), 0), node.id)].append((node, p, gp))

        for node in import_nodes:
            attr_name = None
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                attr_name = node.attr
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id.isupper() or node.id == "arg_types":
                    attr_name = node.id
            if not attr_name or attr_name.startswith("_"):
                continue

            if _is_mro_dict_copy(node, attr_name, parent_map):
                continue

            p = parent_map.get(id(node))
            gp = parent_map.get(id(p)) if p else None

            any_attrs.add(attr_name)

            if _is_key_only_use(node, p, gp):
                continue

            if isinstance(p, ast.Assign) and p.value is node and len(p.targets) == 1 and isinstance(p.targets[0], ast.Name):
                fid = func_of.get(id(p), 0)
                if fid in metaclass_funcs:
                    uses = alias_uses.get((fid, p.targets[0].id), [])
                    if uses and all(_is_key_only_use(u, up, ugp) for u, up, ugp in uses):
                        continue

            value_attrs.add(attr_name)

    return any_attrs, value_attrs


if __name__ == "__main__":
    main()
