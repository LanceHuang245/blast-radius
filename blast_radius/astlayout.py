"""Lossless companion to sqlglot.serde for checkpointed ASTs.

serde.dump drops args whose value is None or an empty list and serde.load re-adds the rest
in dump order, so a loaded AST can differ from the live one in ways later code reads: the
planner turns `joins: []` into a Join step and `joins` missing into a plain Scan. A layout
records, per node in serde's own order, every arg key in order and whether it held None, an
empty list or a value; `restore` puts those args back on a loaded AST so it matches the live
one arg for arg
(and comments: None and [] are both dropped by dump, and optimizer rules tell them apart).
"""

from __future__ import annotations

_NONE, _EMPTY, _VALUE = 0, 1, 2


def _nodes(expression) -> list:
    """Nodes in serde.dump's order (its payload index is the position here)."""
    out = []
    stack = [expression]
    while stack:
        node = stack.pop()
        out.append(node)
        if hasattr(node, "parent") and node.args:
            for vs in reversed(list(node.args.values())):
                if type(vs) is list:
                    stack.extend(reversed(vs))
                elif vs is not None:
                    stack.append(vs)
    return out


def layout(expression) -> list:
    out = []
    for node in _nodes(expression):
        if not hasattr(node, "parent"):
            out.append(None)
            continue
        keys = tuple(
            (k, _NONE if v is None else _EMPTY if type(v) is list and not v else _VALUE)
            for k, v in node.args.items()
        )
        typ = node.type
        comments = node.comments
        out.append((
            keys,
            layout(typ) if typ and typ is not node else None,
            _NONE if comments is None else _EMPTY if not comments else _VALUE,
        ))
    return out


def _plan_restore(expression, lay: list, out: list) -> bool:
    nodes = _nodes(expression)
    if len(nodes) != len(lay):
        return False
    for node, entry in zip(nodes, lay):
        if entry is None:
            if hasattr(node, "parent"):
                return False
            continue
        if not hasattr(node, "parent"):
            return False
        keys, type_lay, comments_kind = entry
        if comments_kind == _VALUE and not node.comments:
            return False
        args = node.args
        if sorted(k for k, kind in keys if kind == _VALUE) != sorted(args):
            return False
        out.append((
            node,
            {k: None if kind == _NONE else [] if kind == _EMPTY else args[k] for k, kind in keys},
            None if comments_kind == _NONE else [] if comments_kind == _EMPTY else node.comments,
        ))
        typ = node.type
        if type_lay is not None:
            if not typ or typ is node or not _plan_restore(typ, type_lay, out):
                return False
        elif typ and typ is not node:
            return False
    return True


def restore(expression, lay: list) -> bool:
    """Rebuild every node's args from `lay`; on any mismatch leave the AST untouched, return False."""
    planned: list = []
    if not _plan_restore(expression, lay, planned):
        return False
    for node, args, comments in planned:
        node.args = args
        node.comments = comments
    return True
