"""Blast Radius selector: pick the tests a patch can break, using a map built
from the pinned sqlglot commit (see tools/), then split them across workers.

How a patch is mapped to tests:

* A Python file is patched in memory against the pinned source. A change that
  only touches comments or whitespace selects nothing; one that no longer
  compiles selects everything.
* A one-for-one line edit inside a function body selects the tests that ran
  the edited statement. Other edits inside a function select the tests that
  ran any line of that function.
* An edit at class level selects the tests that used that class (or a
  subclass) plus the tests that ran any line of the file. An edit at module
  level also pulls in every module that imports this one, transitively.
* A fixture file selects the tests that opened it (or listed its folder).
* Anything the map cannot explain selects every test.
"""

from __future__ import annotations

import ast
import gzip
import io
import json
import sys
import tokenize
from pathlib import Path

DATA = Path("/data")
APP = Path(__file__).resolve().parent
SOURCE = APP / "sqlglot-src"
MAP_FILE = APP / "map" / "impact.json.gz"
MAX_WORKERS = 8
WORKER_COST_MS = 2000

FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
COMPOUND_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try,
                  ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Match)
if hasattr(ast, "TryStar"):
    COMPOUND_NODES += (ast.TryStar,)


def log(message: str) -> None:
    print(message, file=sys.stderr)


class Everything(Exception):
    """Raised when a change cannot be mapped; the commit then runs every test."""


# --------------------------------------------------------------------------- map


class ImpactMap:
    def __init__(self, path: Path):
        with gzip.open(path, "rt") as handle:
            raw = json.load(handle)
        self.test_ids: list[str] = raw["tests"]
        self._sets: list[str] = raw["sets"]
        self._decoded: dict[int, int] = {}
        self.lines: dict[str, dict[str, int]] = raw["lines"]
        self.import_lines = {f: set(v) for f, v in raw["import_lines"].items()}
        self.paths: dict[str, int] = raw["paths"]
        self.classes: dict[str, int] = raw["classes"]
        self.importers: dict[str, list[str]] = raw["importers"]
        self.always = 0
        self.everything = (1 << len(self.test_ids)) - 1
        self.by_file: dict[str, int] = {}
        for i, test_id in enumerate(self.test_ids):
            file = test_id.split("::", 1)[0]
            self.by_file[file] = self.by_file.get(file, 0) | 1 << i
        self._touch: dict[str, int] = {}
        self._module_level: dict[str, int] = {}

    def mask(self, set_index: int) -> int:
        value = self._decoded.get(set_index)
        if value is None:
            value = self._decoded[set_index] = int(self._sets[set_index], 16)
        return value

    def line(self, file: str, lineno: int) -> int:
        index = self.lines.get(file, {}).get(str(lineno))
        return 0 if index is None else self.mask(index)

    def span(self, file: str, first: int, last: int) -> int:
        per_file = self.lines.get(file, {})
        mask = 0
        for lineno in range(first, last + 1):
            index = per_file.get(str(lineno))
            if index is not None:
                mask |= self.mask(index)
        return mask

    def touch(self, file: str) -> int:
        """Tests that ran any line of the file during a test."""
        if file not in self._touch:
            mask = 0
            for index in set(self.lines.get(file, {}).values()):
                mask |= self.mask(index)
            self._touch[file] = mask
        return self._touch[file]

    def class_usage(self, module: str, qualname: str) -> int | None:
        index = self.classes.get(f"{module}:{qualname}")
        return None if index is None else self.mask(index)

    def module_classes(self, module: str) -> int:
        prefix = module + ":"
        mask = 0
        for name, index in self.classes.items():
            if name.startswith(prefix):
                mask |= self.mask(index)
        return mask

    def module_level(self, file: str) -> int:
        """Everything an import-time change to `file` can reach."""
        if file in self._module_level:
            return self._module_level[file]
        self._module_level[file] = 0  # cycle guard
        mask = self.touch(file) | self.by_file.get(file, 0) | self.module_classes(module_name(file))
        for importer in self.importers.get(file, []):
            mask |= self.module_level(importer)
        self._module_level[file] = mask
        return mask


def module_name(file: str) -> str:
    name = file[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


# --------------------------------------------------------------------------- patches


class FilePatch:
    def __init__(self):
        self.old: str | None = None
        self.new: str | None = None
        self.hunks: list[tuple[int, list[str]]] = []
        self.unusual = False  # binary, mode change, rename, ...


def parse_patch(text: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    current: FilePatch | None = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("diff --git "):
            current = FilePatch()
            files.append(current)
            parts = line.split(" ")
            if len(parts) == 4:
                current.old, current.new = parts[2][2:], parts[3][2:]
            continue
        if current is None:
            continue
        if line.startswith("--- "):
            name = line[4:].split("\t")[0]
            current.old = None if name == "/dev/null" else name[2:]
        elif line.startswith("+++ "):
            name = line[4:].split("\t")[0]
            current.new = None if name == "/dev/null" else name[2:]
        elif line.startswith("@@ "):
            header = line.split("@@")[1].split()
            old_start, old_count = _range(header[0][1:])
            _, new_count = _range(header[1][1:])
            body: list[str] = []
            while (old_count > 0 or new_count > 0) and i < len(lines):
                hunk_line = lines[i]
                i += 1
                tag = hunk_line[:1]
                if tag == "\\":
                    continue
                if tag in (" ", ""):
                    old_count -= 1
                    new_count -= 1
                elif tag == "-":
                    old_count -= 1
                elif tag == "+":
                    new_count -= 1
                else:
                    raise Everything("malformed hunk")
                body.append(hunk_line if hunk_line else " ")
            current.hunks.append((old_start, body))
        elif line.startswith(("Binary files", "GIT binary patch", "old mode", "new mode",
                              "rename from", "copy from", "similarity index")):
            current.unusual = True
    return files


def _range(text: str) -> tuple[int, int]:
    if "," in text:
        start, count = text.split(",")
        return int(start), int(count)
    return int(text), 1


def apply(old_text: str, hunks: list[tuple[int, list[str]]]) -> str:
    old_lines = old_text.split("\n")
    out: list[str] = []
    cursor = 0
    for old_start, body in hunks:
        start = old_start - 1 if old_start > 0 else 0
        if start < cursor:
            raise Everything("overlapping hunks")
        out.extend(old_lines[cursor:start])
        cursor = start
        for line in body:
            tag, content = line[0], line[1:]
            if tag in (" ", "-"):
                if cursor >= len(old_lines) or old_lines[cursor].rstrip("\r") != content.rstrip("\r"):
                    raise Everything("patch does not apply")
                cursor += 1
                if tag == " ":
                    out.append(content)
            else:
                out.append(content)
    out.extend(old_lines[cursor:])
    return "\n".join(out)


# --------------------------------------------------------------------------- python analysis


class Scopes:
    """Where each line of a module sits: function bodies, def headers, classes, statements."""

    def __init__(self, tree: ast.Module):
        self.functions: list[tuple[int, int, int, str, ast.AST]] = []  # body_start, end, head_start, qualname, node
        self.lambdas: list[tuple[int, int]] = []
        self.classes: list[tuple[int, int, str]] = []
        self.by_qualname: dict[str, list[tuple[int, int]]] = {}
        self.statements: list[tuple[int, int, ast.stmt]] = []
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


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def header_end(node: ast.stmt) -> int:
    """Last line of a compound statement's header (before its first body line)."""
    first_body = min(
        (child.lineno for field in ("body", "orelse", "handlers", "finalbody", "cases")
         for child in getattr(node, field, []) or [] if hasattr(child, "lineno")),
        default=node.end_lineno + 1,
    )
    return max(node.lineno, first_body - 1)


def same_code(old_text: str, new_text: str) -> bool:
    """True when the two sources differ only in comments and blank lines."""
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


# --------------------------------------------------------------------------- selection


class Selector:
    def __init__(self, impact: ImpactMap):
        self.impact = impact
        self._old: dict[str, tuple[str, Scopes]] = {}
        docs = [i for i, t in enumerate(impact.test_ids) if t.startswith("tests/test_docs.py")]
        self.docs_mask = sum(1 << i for i in docs)

    def old_source(self, path: str) -> tuple[str, Scopes]:
        if path not in self._old:
            file = SOURCE / path
            if not file.is_file():
                raise Everything(f"no pinned source for {path}")
            text = file.read_text(encoding="utf-8")
            self._old[path] = (text, Scopes(ast.parse(text)))
        return self._old[path]

    def select(self, patch_text: str) -> tuple[int, str]:
        impact = self.impact
        try:
            files = parse_patch(patch_text)
            if not files:
                raise Everything("empty patch")
            mask = 0
            for file_patch in files:
                mask |= self.select_file(file_patch)
            return mask, "mapped"
        except Everything as reason:
            return impact.everything, f"everything ({reason})"
        except Exception as error:  # never let an unexpected input crash the run
            return impact.everything, f"everything (error {type(error).__name__}: {error})"

    def select_file(self, fp: FilePatch) -> int:
        impact = self.impact
        if fp.unusual or fp.old is None or fp.new is None or fp.old != fp.new:
            raise Everything("added, deleted, renamed or binary file")
        path = fp.new
        if path.startswith("tests/fixtures/"):
            return self.select_fixture(path)
        if path.endswith(".py") and path.startswith(("sqlglot/", "tests/")):
            if path.startswith("tests/") and (path.endswith("__init__.py") or path.endswith("conftest.py")):
                raise Everything("test package plumbing")
            if path.startswith("sqlglot/") and path not in impact.lines and path not in impact.importers:
                raise Everything(f"{path} unknown to the map")
            return self.select_python(path, fp.hunks)
        raise Everything(f"unmapped path {path}")

    def select_fixture(self, path: str) -> int:
        impact = self.impact
        mask = 0
        index = impact.paths.get("f:" + path)
        if index is not None:
            mask |= impact.mask(index)
        folder = path.rsplit("/", 1)[0]
        index = impact.paths.get("d:" + folder)
        if index is not None:
            mask |= impact.mask(index)
        if not mask:
            raise Everything(f"no test reads {path}")
        return mask

    def select_python(self, path: str, hunks) -> int:
        impact = self.impact
        old_text, old_scopes = self.old_source(path)
        new_text = apply(old_text, hunks)

        changed = [line[1:] for _, body in hunks for line in body if line[0] in "+-"]
        if comment_lines_only(changed) and same_code(old_text, new_text):
            return 0
        try:
            new_tree = ast.parse(new_text)
            compile(new_tree, path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            raise Everything(f"{path} no longer compiles")
        new_scopes = Scopes(new_tree)

        module = module_name(path)
        mask = 0
        for removed, added in change_blocks(hunks):
            mask |= self.select_block(path, module, old_scopes, new_scopes, removed, added)
        return mask

    def select_block(self, path, module, old_scopes, new_scopes, removed, added) -> int:
        impact = self.impact
        # One-for-one edit of statements inside one function: the tests that ran them.
        if removed and len(removed) == len(added):
            mask = self.statement_level(path, old_scopes, new_scopes, removed, added)
            if mask is not None:
                return mask

        mask = 0
        for lineno, _ in removed:
            mask |= self.old_line(path, module, old_scopes, lineno)
        for new_lineno, anchor, _ in added:
            mask |= self.new_line(path, module, old_scopes, new_scopes, new_lineno, anchor)
        return mask

    def statement_level(self, path, old_scopes, new_scopes, removed, added):
        impact = self.impact
        mask = 0
        for (old_no, old_text), (new_no, _anchor, new_text) in zip(removed, added):
            if indent(old_text) != indent(new_text):
                return None
            function = old_scopes.function_at(old_no)
            new_function = new_scopes.function_at(new_no)
            if function is None or new_function is None or function[3] != new_function[3]:
                return None
            body_start, end, head, qualname, node = function
            if old_no < body_start:
                return None  # def header or decorator
            if self.runs_at_import(path, body_start, end):
                return None
            if old_no in old_scopes.docstrings:
                mask |= self.docs_mask
                continue
            statement = old_scopes.statement_at(old_no)
            if statement is None:
                return None
            start, stop, stmt = statement
            if isinstance(stmt, COMPOUND_NODES):
                last = header_end(stmt)
                if old_no > last:
                    return None  # else:/finally: lines and the like
                stop = last
            mask |= impact.span(path, start, stop)
        return mask

    def runs_at_import(self, path: str, first: int, last: int) -> bool:
        lines = self.impact.import_lines.get(path)
        return bool(lines) and any(first <= n <= last for n in lines)

    def function_level(self, path, module, body_start, end) -> int:
        mask = self.impact.span(path, body_start, end)
        if self.runs_at_import(path, body_start, end):
            mask |= self.impact.module_level(path)
        return mask

    def container_level(self, path, module, scopes, lineno) -> int:
        """An import-time change at this line: its class's users, or the whole module's reach."""
        impact = self.impact
        cls = scopes.class_at(lineno)
        if cls is not None:
            usage = impact.class_usage(module, cls[2])
            if usage is not None:
                return usage | impact.touch(path) | impact.by_file.get(path, 0)
        return impact.module_level(path)

    def old_line(self, path, module, scopes, lineno) -> int:
        function = scopes.function_at(lineno)
        if function is not None:
            body_start, end, head, qualname, node = function
            mask = self.function_level(path, module, body_start, end)
            if lineno < body_start:  # signature or decorator: callers and overriding both change
                mask |= self.container_level(path, module, scopes, head)
            if lineno in scopes.docstrings:
                mask |= self.docs_mask
            return mask
        span = scopes.lambda_at(lineno)
        if span is not None:
            mask = self.impact.span(path, span[0], span[1])
            if lineno == span[0]:
                mask |= self.container_level(path, module, scopes, lineno)
            return mask
        mask = self.container_level(path, module, scopes, lineno)
        if lineno in scopes.docstrings:
            mask |= self.docs_mask
        return mask

    def new_line(self, path, module, old_scopes, new_scopes, new_lineno, anchor) -> int:
        function = new_scopes.function_at(new_lineno)
        if function is not None:
            body_start, end, head, qualname, node = function
            spans = old_scopes.by_qualname.get(qualname)
            if spans and new_lineno >= body_start:
                mask = 0
                for start, stop in spans:
                    mask |= self.function_level(path, module, start, stop)
                if new_lineno in new_scopes.docstrings:
                    mask |= self.docs_mask
                return mask
            # new function or changed signature: what defines it changed
            mask = self.container_level(path, module, new_scopes, head)
            for start, stop in spans or []:
                mask |= self.function_level(path, module, start, stop)
            return mask
        if new_scopes.lambda_at(new_lineno) is not None:
            for probe in (anchor - 1, anchor):
                span = old_scopes.lambda_at(probe)
                if span is not None and span[0] < probe <= span[1]:
                    return self.impact.span(path, span[0], span[1])
        mask = self.container_level(path, module, new_scopes, new_lineno)
        if new_lineno in new_scopes.docstrings:
            mask |= self.docs_mask
        return mask


def change_blocks(hunks):
    """Runs of -/+ lines as ([(old_no, text)], [(new_no, anchor_old_no, text)]).

    anchor_old_no is the old line the insertion sits in front of.
    """
    offset = 0  # new line number minus old line number, from earlier hunks
    for old_start, body in hunks:
        old_no = old_start if old_start > 0 else 1
        new_no = old_no + offset
        removed: list[tuple[int, str]] = []
        added: list[tuple[int, int, str]] = []
        for line in body:
            tag, content = line[0], line[1:]
            if tag == " ":
                if removed or added:
                    yield removed, added
                    removed, added = [], []
                old_no += 1
                new_no += 1
            elif tag == "-":
                removed.append((old_no, content))
                old_no += 1
            else:
                added.append((new_no, old_no, content))
                new_no += 1
        if removed or added:
            yield removed, added
        offset = new_no - old_no


def indent(text: str) -> int:
    return len(text) - len(text.lstrip(" \t"))


# --------------------------------------------------------------------------- workers


def split(tests: list[tuple[int, int]]) -> list[list[int]]:
    """Best worker count for these (ms, number) tests, longest first onto the lightest."""
    if not tests:
        return []
    ordered = sorted(tests, reverse=True)
    best = None
    for count in range(1, MAX_WORKERS + 1):
        loads = [0] * count
        workers: list[list[int]] = [[] for _ in range(count)]
        for ms, number in ordered:
            lightest = loads.index(min(loads))
            loads[lightest] += ms
            workers[lightest].append(number)
        cost = max(loads) + WORKER_COST_MS * count
        if best is None or cost < best[0]:
            best = (cost, workers)
        if count >= len(ordered):
            break
    return best[1]


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    number_by_id = {t["id"]: t["number"] for t in tests}
    ms_by_number = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in tests}

    impact = ImpactMap(MAP_FILE)
    selector = Selector(impact)
    numbers = [number_by_id.get(test_id) for test_id in impact.test_ids]
    unmapped = [n for i, n in number_by_id.items() if i not in set(impact.test_ids)]

    for patch in sorted((DATA / "commits").glob("*.patch")):
        mask, reason = selector.select(patch.read_text(encoding="utf-8", errors="replace"))
        chosen = {numbers[i] for i in range(len(numbers)) if mask >> i & 1 and numbers[i] is not None}
        if mask:
            chosen.update(unmapped)
        workers = split([(ms_by_number[n], n) for n in chosen])
        print(json.dumps({"commit": patch.stem, "workers": workers}), flush=True)
        log(f"{patch.stem}: {len(chosen)} tests, {len(workers)} workers, {reason}")


if __name__ == "__main__":
    main()
