"""Static Python analysis: scopes, statements, binding shape, dict entries, comment-only checks."""

from __future__ import annotations

import ast
import io
import re
import tokenize

FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
COMPOUND_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Match,
)
if hasattr(ast, "TryStar"):
    COMPOUND_NODES += (ast.TryStar,)

ENCODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")


class Scopes:
    """Where each line of a module sits: functions, classes, statements, class assigns."""

    def __init__(self, tree: ast.Module):
        self.functions: list[tuple[int, int, int, str, ast.AST]] = []
        self.lambdas: list[tuple[int, int]] = []
        self.classes: list[tuple[int, int, str]] = []
        self.by_qualname: dict[str, list[tuple[int, int]]] = {}
        self.statements: list[tuple[int, int, ast.stmt]] = []
        self.class_assigns: dict[tuple[str, str], ast.stmt] = {}
        self.docstrings: set[int] = set()
        self._walk(tree, "")

    def _walk(self, node: ast.AST, prefix: str) -> None:
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and _is_docstring(body[0]):
            self.docstrings.update(range(body[0].lineno, body[0].end_lineno + 1))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self.statements.append((child.lineno, child.end_lineno, child))
            if isinstance(child, FUNCTION_NODES):
                qualname = prefix + child.name
                head = min([child.lineno] + [d.lineno for d in child.decorator_list])
                body_start = child.body[0].lineno
                self.functions.append((body_start, child.end_lineno, head, qualname, child))
                self.by_qualname.setdefault(qualname, []).append((body_start, child.end_lineno))
                self._walk(child, qualname + ".<locals>.")
            elif isinstance(child, ast.ClassDef):
                qualname = prefix + child.name
                head = min([child.lineno] + [d.lineno for d in child.decorator_list])
                self.classes.append((head, child.end_lineno, qualname))
                # Record direct class-body attribute assignments for fine-grained dict/attr tracking.
                for stmt in child.body:
                    attr = _single_assign_target(stmt)
                    if attr is not None:
                        self.class_assigns[(qualname, attr)] = stmt
                self._walk(child, qualname + ".")
            elif isinstance(child, ast.Lambda):
                self.lambdas.append((child.body.lineno, child.body.end_lineno))
                self._walk(child, prefix)
            else:
                self._walk(child, prefix)

    def function_at(self, lineno: int):
        """Innermost function whose body or header holds the line."""
        best = None
        for body_start, end, head, qualname, node in self.functions:
            if head <= lineno <= end and (best is None or head >= best[2]):
                best = (body_start, end, head, qualname, node)
        return best

    def lambda_at(self, lineno: int):
        best = None
        for start, end in self.lambdas:
            if start <= lineno <= end and (best is None or start >= best[0]):
                best = (start, end)
        return best

    def class_at(self, lineno: int):
        best = None
        for head, end, qualname in self.classes:
            if head <= lineno <= end and (best is None or head >= best[0]):
                best = (head, end, qualname)
        return best

    def statement_at(self, lineno: int):
        best = None
        for start, end, node in self.statements:
            if start <= lineno <= end and (best is None or start >= best[0]):
                best = (start, end, node)
        return best

    def class_assign_at(self, lineno: int) -> tuple[str, str, ast.stmt] | None:
        """If lineno lies inside a direct class-body attribute assignment, return (cls, attr, stmt)."""
        cls = self.class_at(lineno)
        if cls is None:
            return None
        cls_qualname = cls[2]
        for (cqual, attr), stmt in self.class_assigns.items():
            if cqual == cls_qualname and stmt.lineno <= lineno <= stmt.end_lineno:
                return cls_qualname, attr, stmt
        return None


def _single_assign_target(stmt: ast.stmt) -> str | None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        return stmt.targets[0].id
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
        return stmt.target.id
    return None


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def header_end(node: ast.stmt) -> int:
    """Last line of a compound statement's header (before its first body line)."""
    first_body = min(
        (
            child.lineno
            for field in ("body", "orelse", "handlers", "finalbody", "cases")
            for child in getattr(node, field, []) or []
            if hasattr(child, "lineno")
        ),
        default=node.end_lineno + 1,
    )
    return max(node.lineno, first_body - 1)


def _function_binding_shape(func_node: ast.AST) -> tuple:
    """Capture lexical scope bindings and generator/coroutine flags of a function body.

    An edit in an unexecuted branch that adds a local assignment, `global`/`nonlocal`,
    or `yield` changes the whole function's semantics at compile time.
    """
    locals_bound: set[str] = set()
    globals_decl: set[str] = set()
    nonlocals_decl: set[str] = set()
    has_yield = False

    def collect_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            locals_bound.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                collect_target(elt)
        elif isinstance(target, ast.Starred):
            collect_target(target.value)

    stack: list[ast.AST] = list(getattr(func_node, "body", []))
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            locals_bound.add(cur.name)
            continue
        if isinstance(cur, ast.Lambda):
            continue
        if isinstance(cur, (ast.Yield, ast.YieldFrom)):
            has_yield = True
        elif isinstance(cur, ast.Global):
            globals_decl.update(cur.names)
        elif isinstance(cur, ast.Nonlocal):
            nonlocals_decl.update(cur.names)
        elif isinstance(cur, ast.Name) and isinstance(cur.ctx, (ast.Store, ast.Del)):
            locals_bound.add(cur.id)
        elif isinstance(cur, ast.ExceptHandler) and cur.name:
            locals_bound.add(cur.name)
        elif isinstance(cur, (ast.Import, ast.ImportFrom)):
            for alias in cur.names:
                locals_bound.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(cur, ast.NamedExpr):
            collect_target(cur.target)
        stack.extend(ast.iter_child_nodes(cur))

    return (
        isinstance(func_node, ast.AsyncFunctionDef),
        has_yield,
        frozenset(locals_bound),
        frozenset(globals_decl),
        frozenset(nonlocals_decl),
    )


def _ast_dict_key_tokens(key_node: ast.AST) -> tuple[str, ...] | None:
    """Convert an AST dict key node into candidate _TrackedDict lookup tokens."""
    if isinstance(key_node, ast.Constant):
        if isinstance(key_node.value, str):
            return ("s:" + key_node.value,)
        if isinstance(key_node.value, int) and not isinstance(key_node.value, bool):
            return (f"i:{key_node.value}",)
        return None
    if isinstance(key_node, ast.Attribute):
        parent = (
            key_node.value.id
            if isinstance(key_node.value, ast.Name)
            else key_node.value.attr if isinstance(key_node.value, ast.Attribute) else None
        )
        if parent:
            return (f"e:{parent}.{key_node.attr}", "t:" + key_node.attr)
        return ("t:" + key_node.attr,)
    return None


def _extract_dict_entries(dict_node: ast.AST) -> tuple[dict[tuple[str, ...], ast.AST], list[str]] | None:
    """Extract statically known key -> value AST pairs and spread expressions from an ast.Dict."""
    if not isinstance(dict_node, ast.Dict):
        return None
    entries: dict[tuple[str, ...], ast.AST] = {}
    spreads: list[str] = []
    for k, v in zip(dict_node.keys, dict_node.values):
        if k is None:
            spreads.append(ast.dump(v))
            continue
        toks = _ast_dict_key_tokens(k)
        if toks is None or toks in entries:
            return None
        entries[toks] = v
    return entries, spreads


def _encoding_cookie(text: str) -> str | None:
    for line in text.split("\n", 2)[:2]:
        match = ENCODING_RE.match(line)
        if match:
            return match.group(1).lower()
    return None


def same_code(old_text: str, new_text: str) -> bool:
    """True when the two sources differ only in comments and blank lines (and same encoding cookie)."""
    if _encoding_cookie(old_text) != _encoding_cookie(new_text):
        return False

    def significant(text):
        tokens = []
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in (tokenize.COMMENT, tokenize.NL):
                continue
            tokens.append((token.type, token.string))
        return tokens

    try:
        return significant(old_text) == significant(new_text)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return False


def comment_lines_only(lines: list[str]) -> bool:
    return all(not line.strip() or line.strip().startswith("#") for line in lines)


def indent(text: str) -> int:
    return len(text) - len(text.lstrip(" \t"))
