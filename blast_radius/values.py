"""Narrow a condition-site edit with the value map (map/values.bin).

For a patched sqlglot file whose every AST difference is a rearranged condition
site (and/or swapped, not added/removed, comparison operator changed) over the
same operands, replay each test's recorded vectors through the original and the
mutated site: a test whose every evaluation yields the same truth (and the same
object where the value is used), and evaluates the same side-effecting operands
in the same order, cannot behave differently. Tests that covered the site's line
but left no vectors stay selected.
"""

from __future__ import annotations

import ast
import gzip
import marshal
from pathlib import Path

from blast_radius import valuesite as vs


class ValueMap:
    def __init__(self, path: Path, test_ids: list[str]):
        raw = marshal.loads(gzip.decompress(path.read_bytes()))
        pos = {t: i for i, t in enumerate(test_ids)}
        self._remap = [pos.get(t) for t in raw["tests"]]
        self._identity = self._remap == list(range(len(test_ids)))
        self.files = raw["files"]
        self.imports = raw["import"]

    def _mask(self, mask: int) -> int:
        if self._identity:
            return mask
        out = 0
        i = 0
        while mask:
            if mask & 1 and self._remap[i] is not None:
                out |= 1 << self._remap[i]
            mask >>= 1
            i += 1
        return out

    def narrow(self, path: str, old_tree: ast.Module, new_tree: ast.Module, impact, hunks=None) -> int | None:
        muts = site_mutations(old_tree, new_tree, touched_old_lines(hunks) if hunks is not None else None)
        if muts is None or not muts:
            return None
        sites = self.files.get(path, {})
        imports = self.imports.get(path, {})
        acc = 0  # value-map test space
        extra = 0  # impact test space
        for old, new, parent, field in muts:
            spec_o, atoms, nslots = vs.build(old)
            spec_n, _, _ = vs.build(new)
            truth, links = vs.slot_maps(spec_o)
            offs = vs.offsets(spec_o, nslots)
            impure = {i for i, a in enumerate(atoms) if not vs.is_pure(a)}
            boolctx = vs.bool_context(parent, field)
            key = vs.site_key(old)

            def differs(vec: str) -> bool:
                try:
                    vo, io, so = vs.replay(spec_o, vec, truth, links, offs)
                    vn, in_, sn = vs.replay(spec_n, vec, truth, links, offs)
                except vs.Unknown:
                    return True
                if vo != vn:
                    return True
                if not boolctx and io != in_ and not (io == ("b",) and in_ == ("b",)):
                    return True
                return [a for a in so if a in impure] != [a for a in sn if a in impure]

            if any(differs(v) for v in imports.get(key, ())):
                return None
            recorded = 0
            for vec, mask in sites.get(key, ()):
                recorded |= mask
                if differs(vec):
                    acc |= mask
            extra |= impact.span(path, old.lineno, old.end_lineno) & ~self._mask(recorded)
        return self._mask(acc) | extra


def _dump(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=False)


def touched_old_lines(hunks) -> set[int]:
    """Old lines a hunk removes, plus the old lines on both sides of every insertion."""
    out: set[int] = set()
    for old_start, _, _, body in hunks:
        ln = old_start
        for line in body:
            tag = line[:1]
            if tag == "-":
                out.add(ln)
                ln += 1
            elif tag == "+":
                out.update((ln - 1, ln))
            else:
                ln += 1
    return out


def site_mutations(old_tree: ast.AST, new_tree: ast.AST, touched: set[int] | None = None):
    """[(old site, new site, parent, field)] when every difference is a site rearrangement, else None.
    With `touched` (old line numbers the patch changed), nodes wholly outside it are taken as equal."""
    muts: list = []

    def same(o, n, depth: int, parent, field) -> bool:
        if type(o) is not type(n) and not (vs.is_logic(o) and vs.is_logic(n)):
            return False
        if touched is not None and isinstance(o, (ast.stmt, ast.expr)):
            if not any(ln in touched for ln in range(o.lineno, o.end_lineno + 1)):
                return True
        if isinstance(o, ast.AST):
            if vs.is_logic(o):
                if _dump(o) == _dump(n):
                    return True
                if depth == 0:
                    return False
                _, atoms_o, _ = vs.build(o)
                _, atoms_n, _ = vs.build(n)
                if len(atoms_o) != len(atoms_n) or not vs.instrumentable(atoms_o):
                    return False
                if any(_dump(a) != _dump(b) for a, b in zip(atoms_o, atoms_n)):
                    return False
                muts.append((o, n, parent, field))
                return True
            if (
                isinstance(o, (ast.stmt, ast.expr))
                and not isinstance(o, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and (o.end_lineno - o.lineno) <= 5
                and _dump(o) == _dump(n)
            ):
                return True
            inner = depth + 1 if isinstance(o, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) else depth
            for name, vo in ast.iter_fields(o):
                vn = getattr(n, name, None)
                if isinstance(vo, list):
                    if not isinstance(vn, list) or len(vo) != len(vn):
                        return False
                    for a, b in zip(vo, vn):
                        if isinstance(a, ast.AST):
                            if not same(a, b, inner, o, name):
                                return False
                        elif a != b:
                            return False
                elif isinstance(vo, ast.AST):
                    if not isinstance(vn, ast.AST) or not same(vo, vn, inner, o, name):
                        return False
                elif vo != vn:
                    return False
            return True
        return o == n

    return muts if same(old_tree, new_tree, 0, None, None) else None
