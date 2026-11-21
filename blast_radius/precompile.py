"""Build-time precompiler run inside Dockerfile (never at scoring time).

Reads map/impact.json.gz + /app/sqlglot-src, recomputes clean scoped Expression
class_refs (only runtime `exp.<Cls>.<member>` accesses), converts hex bitsets to
raw bytes, and writes /app/map/impact.marshal for fast startup with 0 repo bloat.
"""

from __future__ import annotations

import ast
import gzip
import json
import marshal
import os
import sys
from collections import defaultdict
from pathlib import Path

from blast_radius import MAP_FILE, SOURCE
from blast_radius.impact_map import module_name


def _scoped_class_refs(
    src: Path,
    lines_idx: dict[str, dict[str, int]],
    masks: list[int],
) -> dict[str, int]:
    """Compute Expression class_refs strictly from runtime `exp.<Cls>.<member>` accesses."""
    trees: dict[str, ast.Module] = {}
    for top in ("sqlglot", "tests"):
        top_dir = src / top
        if not top_dir.is_dir():
            continue
        for root, _, names in os.walk(top_dir):
            for name in names:
                if name.endswith(".py"):
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, src).replace(os.sep, "/")
                    try:
                        trees[rel] = ast.parse(open(full, encoding="utf-8").read())
                    except SyntaxError:
                        continue

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
        per_file = lines_idx.get(rel)
        if not per_file:
            continue

        # Track local aliases explicitly imported from sqlglot.expressions* (plus same-package names).
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
            idx = per_file.get(str(ln)) if ln is not None else None
            if idx is None:
                return
            mask = masks[idx]
            for ckey, def_rel, start, end in defs_by_name[cls_name]:
                if rel == def_rel and start <= ln <= end:
                    continue
                refs[ckey] |= mask

        def walk(node: ast.AST) -> None:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return
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


def main() -> None:
    with gzip.open(MAP_FILE, "rt") as handle:
        raw = json.load(handle)

    old_masks = [int(s, 16) for s in raw["sets"]]
    # Recompute clean scoped class_refs and intern any newly formed masks.
    scoped_refs = _scoped_class_refs(SOURCE, raw["lines"], old_masks)

    seen: dict[int, int] = {m: i for i, m in enumerate(old_masks)}
    sets_bytes: list[bytes] = [
        m.to_bytes((m.bit_length() + 7) // 8, "little") for m in old_masks
    ]

    def intern(mask: int) -> int:
        idx = seen.get(mask)
        if idx is None:
            idx = len(sets_bytes)
            seen[mask] = idx
            sets_bytes.append(mask.to_bytes((mask.bit_length() + 7) // 8, "little"))
        return idx

    raw["class_refs"] = {c: intern(m) for c, m in sorted(scoped_refs.items()) if m}
    raw["sets"] = sets_bytes
    raw["sets_binary"] = True

    out_path = MAP_FILE.parent / "impact.marshal"
    with open(out_path, "wb") as handle:
        marshal.dump(raw, handle)
    print(
        f"precompiled {out_path}: {len(raw['class_refs'])} scoped class_refs, "
        f"{out_path.stat().st_size / 1e6:.2f} MB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
