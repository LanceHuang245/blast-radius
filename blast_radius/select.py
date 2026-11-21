"""Per-file and per-change-block test selection rules."""

from __future__ import annotations

import ast

from blast_radius import SOURCE, Everything
from blast_radius.impact_map import ImpactMap, module_name
from blast_radius.patches import FilePatch, apply, change_blocks, parse_patch
from blast_radius.scopes import (
    COMPOUND_NODES,
    Scopes,
    _extract_dict_entries,
    _function_binding_shape,
    comment_lines_only,
    header_end,
    indent,
    same_code,
)


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
            all_noop = True
            for file_patch in files:
                file_mask, is_noop = self.select_file(file_patch)
                mask |= file_mask
                all_noop = all_noop and is_noop
            return mask, "noop" if all_noop and mask == 0 else "mapped"
        except Everything as reason:
            return impact.everything, f"everything ({reason})"
        except Exception as error:  # never let an unexpected input crash the run
            return impact.everything, f"everything (error {type(error).__name__}: {error})"

    def select_file(self, fp: FilePatch) -> tuple[int, bool]:
        impact = self.impact
        if fp.unusual or fp.old is None or fp.new is None or fp.old != fp.new:
            raise Everything("added, deleted, renamed or binary file")
        path = fp.new
        if path.startswith("tests/fixtures/"):
            return self.select_fixture(path), False
        if path.endswith(".py") and path.startswith(("sqlglot/", "tests/")):
            if path.startswith("tests/") and (path.endswith("__init__.py") or path.endswith("conftest.py")):
                raise Everything("test package plumbing")
            if path.startswith("sqlglot/") and path not in impact.lines and path not in impact.importers:
                raise Everything(f"{path} unknown to the map")
            # Always include tests that directly opened this .py file (e.g. test_typing.py).
            readers = impact.file_readers(path)
            py_mask, py_noop = self.select_python(path, fp.hunks)
            return readers | py_mask, py_noop and readers == 0
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

    def select_python(self, path: str, hunks) -> tuple[int, bool]:
        old_text, old_scopes = self.old_source(path)
        new_text = apply(old_text, hunks)

        changed = [line[1:] for _, _, _, body in hunks for line in body if line[0] in "+-"]
        if comment_lines_only(changed) and same_code(old_text, new_text):
            return 0, True
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
        return mask, False

    def select_block(self, path, module, old_scopes, new_scopes, removed, added) -> int:
        # Try fine-grained class dict-key or class attribute tracking first.
        attr_mask = self.class_attr_level(path, module, old_scopes, new_scopes, removed, added)
        if attr_mask is not None:
            return attr_mask | self._test_file_own_mask(path, old_scopes, new_scopes, removed, added)

        # One-for-one edit of statements inside one function: the tests that ran them.
        if removed and len(removed) == len(added):
            mask = self.statement_level(path, old_scopes, new_scopes, removed, added)
            if mask is not None:
                return mask | self._test_file_own_mask(path, old_scopes, new_scopes, removed, added)

        mask = 0
        for lineno, _ in removed:
            mask |= self.old_line(path, module, old_scopes, lineno)
        for new_lineno, anchor, _ in added:
            mask |= self.new_line(path, module, old_scopes, new_scopes, new_lineno, anchor)
        return mask | self._test_file_own_mask(path, old_scopes, new_scopes, removed, added)

    def _test_file_own_mask(self, path: str, old_scopes: Scopes, new_scopes: Scopes, removed, added) -> int:
        """When editing a test file, always include the touched test method(s) or test class."""
        if not path.startswith("tests/"):
            return 0
        impact = self.impact
        mask = 0
        for scopes, lines in ((old_scopes, [n for n, _ in removed]), (new_scopes, [n for n, _, _ in added])):
            for lineno in lines:
                fn = scopes.function_at(lineno)
                if fn is not None:
                    qual = fn[3]
                    bit = impact.by_test_qualname.get((path, qual))
                    if bit:
                        mask |= bit
                        continue
                cls = scopes.class_at(lineno)
                if cls is not None:
                    cls_mask = impact.by_class.get(f"{path}::{cls[2].replace('.', '::')}", 0)
                    mask |= cls_mask
        return mask

    def class_attr_level(self, path, module, old_scopes: Scopes, new_scopes: Scopes, removed, added) -> int | None:
        """Resolve an edit confined to one direct class-body attribute assignment."""
        impact = self.impact
        if not removed and not added:
            return None

        # Verify every edited line in this block sits inside the exact same (class, attr) assignment.
        target_old = None
        for old_no, _ in removed:
            info = old_scopes.class_assign_at(old_no)
            if info is None:
                return None
            if target_old is None:
                target_old = info
            elif (info[0], info[1]) != (target_old[0], target_old[1]):
                return None

        target_new = None
        for new_no, _anchor, _ in added:
            info = new_scopes.class_assign_at(new_no)
            if info is None:
                return None
            if target_new is None:
                target_new = info
            elif (info[0], info[1]) != (target_new[0], target_new[1]):
                return None

        if target_old is None and target_new is not None:
            cls_qual, attr, new_stmt = target_new
            old_stmt = old_scopes.class_assigns.get((cls_qual, attr))
            if old_stmt is None:
                return None
        elif target_new is None and target_old is not None:
            cls_qual, attr, old_stmt = target_old
            new_stmt = new_scopes.class_assigns.get((cls_qual, attr))
            if new_stmt is None:
                return None
        elif target_old is not None and target_new is not None:
            if (target_old[0], target_old[1]) != (target_new[0], target_new[1]):
                return None
            cls_qual, attr, old_stmt = target_old
            new_stmt = target_new[2]
        else:
            return None

        # If every edited line actually lives inside a lambda body within the assignment,
        # let the normal lambda/statement coverage handle it (often even narrower).
        if removed and all(old_scopes.lambda_at(n) is not None and old_scopes.lambda_at(n)[0] < n for n, _ in removed):
            if all(new_scopes.lambda_at(n) is not None for n, _, _ in added):
                return None

        cls_key = f"{module}:{cls_qual}"
        key_id = f"{cls_key}|{attr}"

        # Case 1: Dict-literal attribute with identical key set and no import-time value reads.
        old_val = old_stmt.value
        new_val = new_stmt.value
        if isinstance(old_val, ast.Dict) and isinstance(new_val, ast.Dict):
            if attr not in impact.import_value_attrs and key_id in impact.tracked_dicts:
                old_parsed = _extract_dict_entries(old_val)
                new_parsed = _extract_dict_entries(new_val)
                if old_parsed is not None and new_parsed is not None:
                    old_entries, old_spreads = old_parsed
                    new_entries, new_spreads = new_parsed
                    # Same keys in the same order: iteration order is observable, and
                    # key-only iteration is not recorded by the map.
                    if old_spreads == new_spreads and list(old_entries) == list(new_entries):
                        changed_keys = [
                            toks for toks in old_entries if ast.dump(old_entries[toks]) != ast.dump(new_entries[toks])
                        ]
                        if not changed_keys:
                            return 0
                        mask = impact.dict_key_usage(cls_key, attr, "*")
                        for toks in changed_keys:
                            for tok in toks:
                                mask |= impact.dict_key_usage(cls_key, attr, tok)
                        return mask
            return None

        # Case 2: Non-dict class data attribute not read at import/metaclass time.
        if attr not in impact.import_any_attrs and key_id in impact.tracked_attrs:
            return impact.attr_usage(cls_key, attr)

        return None

    def statement_level(self, path, old_scopes, new_scopes, removed, added):
        impact = self.impact
        mask = 0
        checked_funcs: set[str] = set()
        for (old_no, old_text), (new_no, _anchor, new_text) in zip(removed, added):
            if indent(old_text) != indent(new_text):
                return None
            function = old_scopes.function_at(old_no)
            new_function = new_scopes.function_at(new_no)
            if function is None or new_function is None or function[3] != new_function[3]:
                return None
            body_start, end, head, qualname, old_node = function
            if old_no < body_start:
                return None  # def header or decorator
            if self.runs_at_import(path, body_start, end):
                return None
            # Reject statement-level if local bindings or generator flags changed anywhere in the function.
            if qualname not in checked_funcs:
                if _function_binding_shape(old_node) != _function_binding_shape(new_function[4]):
                    return None
                checked_funcs.add(qualname)
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
        """An import-time change at this line: its class's users & refs, or the whole module's reach."""
        impact = self.impact
        cls = scopes.class_at(lineno)
        if cls is not None:
            head, end, qualname = cls
            usage = impact.class_usage(module, qualname)
            if usage is not None:
                return (
                    usage
                    | impact.class_ref_usage(module, qualname)
                    | impact.span(path, head, end)
                    | impact.by_file.get(path, 0)
                )
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
