"""Per-file and per-change-block test selection rules."""

from __future__ import annotations

import ast
import marshal
from dataclasses import dataclass, field

import os

from blast_radius import APP, SOURCE, Everything, log
from blast_radius.impact_map import ImpactMap, module_name
from blast_radius.import_state import STATE_FILE
from blast_radius.patches import FilePatch, apply, change_blocks, parse_patch
from blast_radius.scopes import (
    COMPOUND_NODES,
    Scopes,
    _extract_dict_entries,
    _is_docstring,
    canon_ast,
    comment_lines_only,
    header_end,
    indent,
    same_ast,
    same_code,
    same_local_binding_semantics,
)

SAFE_HOTPATCH_DECORATORS = frozenset({"staticmethod", "classmethod", "property"})
# Files that configure how pytest runs every test.
PYTEST_CONFIG_FILES = frozenset({"pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini", "conftest.py"})


@dataclass
class Selection:
    """Static selection result plus in-memory patched files and hot-patch eligibility."""

    mask: int
    reason: str
    patched_files: dict[str, str]
    # None -> requires Phase 1 clean import; list -> Phase 2 function __code__ hot-patch targets.
    hot_patch_targets: list[tuple[str, str]] | None
    # Phase 2 safe non-import-time literal class-attribute updates: (rel_path, cls_qual, attr, val).
    attr_hot_targets: list[tuple[str, str, str, object]] | None = None
    fixture_only: bool = False
    # Bitmask of tests in `mask` pulled in solely by pushdown_predicates.py:36-37 lazy imports.
    lazy_pp_mask: int = 0
    # Modules whose import ran an edited statement: `mask` holds only if their import-time
    # state digests are unchanged under the patch, else `coarse_mask` applies.
    import_checks: list[str] = field(default_factory=list)
    coarse_mask: int = 0
    # Every change resolved to these sqlglot statement spans (path, first, last), or None when
    # some change was resolved otherwise: lets the verifier rerun only fixture cases that reach them.
    edited_spans: list[tuple[str, int, int]] | None = None


def _decorator_name(node: ast.expr) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _is_overload(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "overload") or (
        isinstance(node, ast.Attribute) and node.attr == "overload"
    )


def _runtime_def(scopes: Scopes, qualname: str):
    """The def a qualname is bound to at runtime: the only one, or the last after @overload stubs."""
    defs = [f for f in scopes.functions if f[3] == qualname]
    if not defs or not all(any(_is_overload(d) for d in f[4].decorator_list) for f in defs[:-1]):
        return None
    return defs[-1]


def _matching_compound_subblocks(
    so: ast.stmt,
    sn: ast.stmt,
    ignore_ann: bool,
) -> list[tuple[list[ast.stmt], list[ast.stmt]]] | None:
    """If two same-type compound statements share an identical header, return their paired sub-blocks."""
    if isinstance(so, (ast.If, ast.While)):
        if canon_ast(so.test, ignore_ann) == canon_ast(sn.test, ignore_ann):
            return [(so.body, sn.body), (so.orelse, sn.orelse)]
    elif isinstance(so, (ast.For, ast.AsyncFor)):
        if canon_ast(so.target, ignore_ann) == canon_ast(sn.target, ignore_ann) and canon_ast(
            so.iter, ignore_ann
        ) == canon_ast(sn.iter, ignore_ann):
            return [(so.body, sn.body), (so.orelse, sn.orelse)]
    elif isinstance(so, (ast.With, ast.AsyncWith)):
        if canon_ast(so.items, ignore_ann) == canon_ast(sn.items, ignore_ann):
            return [(so.body, sn.body)]
    elif isinstance(so, ast.Try):
        if len(so.handlers) == len(sn.handlers) and all(
            canon_ast(ho.type, ignore_ann) == canon_ast(hn.type, ignore_ann) and ho.name == hn.name
            for ho, hn in zip(so.handlers, sn.handlers)
        ):
            pairs = [(so.body, sn.body), (so.orelse, sn.orelse), (so.finalbody, sn.finalbody)]
            pairs.extend((ho.body, hn.body) for ho, hn in zip(so.handlers, sn.handlers))
            return pairs
    return None


def _same_compound_bodies(so: ast.stmt, sn: ast.stmt, ignore_ann: bool) -> bool:
    """True when all sub-block bodies of two same-type compound statements are canonically identical."""
    for field in ("body", "orelse", "finalbody"):
        bo = getattr(so, field, None)
        bn = getattr(sn, field, None)
        if bo is not None or bn is not None:
            if canon_ast(bo, ignore_ann) != canon_ast(bn, ignore_ann):
                return False
    ho = getattr(so, "handlers", None)
    hn = getattr(sn, "handlers", None)
    if ho is not None or hn is not None:
        if canon_ast(ho, ignore_ann) != canon_ast(hn, ignore_ann):
            return False
    return True


def _extract_attr_hot_targets(
    path: str,
    module: str,
    old_scopes: Scopes,
    new_scopes: Scopes,
    hunks,
    impact: ImpactMap,
) -> list[tuple[str, str, str, object]] | None:
    """Return [(path, cls_qual, attr, literal_val)] if every edit only updates non-import-time literal class attrs."""
    targets: list[tuple[str, str, str, object]] = []
    seen: set[tuple[str, str]] = set()
    for removed, added in change_blocks(hunks):
        if not removed and not added:
            continue
        target_cls = target_attr = None
        for old_no, _ in removed:
            info = old_scopes.class_assign_at(old_no)
            if info is None:
                return None
            if target_cls is None:
                target_cls, target_attr = info[0], info[1]
            elif (info[0], info[1]) != (target_cls, target_attr):
                return None
        for new_no, _anchor, _ in added:
            info = new_scopes.class_assign_at(new_no)
            if info is None:
                return None
            if target_cls is None:
                target_cls, target_attr = info[0], info[1]
            elif (info[0], info[1]) != (target_cls, target_attr):
                return None
        if target_cls is None or target_attr is None:
            return None
        if impact.read_at_import(f"{module}:{target_cls}", target_attr):
            return None
        # A test module read it while importing: the pre-warmed copy would be stale.
        if f"{module}:{target_cls}|{target_attr}" in impact.collect_attrs:
            return None
        if f"{module}:{target_cls}|{target_attr}" not in impact.tracked_attrs:
            return None
        new_stmt = new_scopes.class_assigns.get((target_cls, target_attr))
        if new_stmt is None or new_stmt.value is None:
            return None
        try:
            lit_val = ast.literal_eval(new_stmt.value)
        except Exception:
            return None
        if (target_cls, target_attr) not in seen:
            seen.add((target_cls, target_attr))
            targets.append((path, target_cls, target_attr, lit_val))
    return targets if targets else None


def _extract_hot_patch_targets(
    path: str,
    old_scopes: Scopes,
    new_scopes: Scopes,
    hunks,
    runs_at_import,
) -> list[tuple[str, str]] | None:
    """Return [(path, qualname)] if every edit is strictly inside existing non-nested function bodies."""
    targets: set[tuple[str, str]] = set()
    for removed, added in change_blocks(hunks):
        for old_no, _ in removed:
            fn = old_scopes.function_at(old_no)
            if fn is None or old_no < fn[0]:
                return None
            body_start, end, _head, qual, old_node = fn
            if "<locals>" in qual or runs_at_import(path, body_start, end):
                return None
            old_rt, new_fn = _runtime_def(old_scopes, qual), _runtime_def(new_scopes, qual)
            if old_rt is None or old_rt[4] is not old_node or new_fn is None:
                return None
            new_node = new_fn[4]
            if not all(_decorator_name(d) in SAFE_HOTPATCH_DECORATORS for d in old_node.decorator_list):
                return None
            if [ast.dump(d) for d in old_node.decorator_list] != [ast.dump(d) for d in new_node.decorator_list]:
                return None
            if ast.dump(old_node.args) != ast.dump(new_node.args):
                return None
            targets.add((path, qual))

        for new_no, _anchor, _ in added:
            fn = new_scopes.function_at(new_no)
            if fn is None or new_no < fn[0]:
                return None
            _body_start, _end, _head, qual, new_node = fn
            if "<locals>" in qual:
                return None
            old_fn, new_rt = _runtime_def(old_scopes, qual), _runtime_def(new_scopes, qual)
            if new_rt is None or new_rt[4] is not new_node:
                return None
            if old_fn is None or runs_at_import(path, old_fn[0], old_fn[1]):
                return None
            old_node = old_fn[4]
            if not all(_decorator_name(d) in SAFE_HOTPATCH_DECORATORS for d in old_node.decorator_list):
                return None
            if [ast.dump(d) for d in old_node.decorator_list] != [ast.dump(d) for d in new_node.decorator_list]:
                return None
            if ast.dump(old_node.args) != ast.dump(new_node.args):
                return None
            targets.add((path, qual))
    return sorted(targets)


class Selector:
    def __init__(self, impact: ImpactMap):
        self.impact = impact
        self._old: dict[str, tuple[str, Scopes]] = {}
        self._no_pp_lazy: bool = False
        self._hit_pp_lazy: bool = False
        self._coarse_block: bool = False
        self._no_import_check: bool = False
        self._no_verify: bool = False
        self._spans: list[tuple[str, int, int]] = []
        self._spans_ok: bool = True
        self._owner_checks: set[str] = set()
        self._state = marshal.loads(STATE_FILE.read_bytes()) if STATE_FILE.is_file() else None
        self.values = None
        values_file = APP / "map" / "values.bin"
        if values_file.is_file() and os.environ.get("BR_VALUES", "1") != "0":
            from blast_radius.values import ValueMap

            self.values = ValueMap(values_file, impact.test_ids)
        docs = [i for i, t in enumerate(impact.test_ids) if t.startswith("tests/test_docs.py")]
        self.docs_mask = sum(1 << i for i in docs)

    def old_source(self, path: str) -> tuple[str, Scopes]:
        if path not in self._old:
            file = SOURCE / path
            if not file.is_file():
                raise Everything(f"no pinned source for {path}")
            text = file.read_text(encoding="utf-8")
            self._old[path] = (text, Scopes(ast.parse(text), has_lambda="lambda" in text))
        return self._old[path]

    def select(self, patch_text: str) -> tuple[int, str]:
        sel = self.select_detailed(patch_text)
        return sel.mask, sel.reason

    def select_detailed(self, patch_text: str) -> Selection:
        impact = self.impact
        try:
            files = parse_patch(patch_text)
            if not files:
                raise Everything("empty patch")
            mask = 0
            lazy_pp_mask = 0
            all_noop = True
            all_fixture_or_noop = True
            patched_files: dict[str, str] = {}
            hot_targets: list[tuple[str, str]] | None = []
            attr_targets: list[tuple[str, str, str, object]] | None = []
            self._hit_pp_lazy = False
            self._owner_checks = set()
            self._no_verify = False
            self._spans, self._spans_ok = [], True
            for file_patch in files:
                file_mask, is_noop, is_fixture_or_comment, f_targets, a_targets = self.select_file_detailed(
                    file_patch, patched_files
                )
                mask |= file_mask
                all_noop = all_noop and is_noop
                all_fixture_or_noop = all_fixture_or_noop and is_fixture_or_comment
                if f_targets is None and a_targets is None:
                    hot_targets = None
                    attr_targets = None
                else:
                    if hot_targets is not None and f_targets:
                        hot_targets.extend(f_targets)
                    if attr_targets is not None and a_targets:
                        attr_targets.extend(a_targets)
            # The extra passes below re-select with other settings: keep this pass's spans.
            spans = list(dict.fromkeys(self._spans)) if self._spans_ok else None
            if self._hit_pp_lazy and mask:
                self._no_pp_lazy = True
                try:
                    mask_no_pp = 0
                    for file_patch in files:
                        fm, _, _, _, _ = self.select_file_detailed(file_patch, {})
                        mask_no_pp |= fm
                    lazy_pp_mask = mask & ~mask_no_pp
                finally:
                    self._no_pp_lazy = False
            owners = sorted(self._owner_checks)
            coarse_mask = 0
            if owners:
                self._no_import_check = True
                try:
                    for file_patch in files:
                        fm, _, _, _, _ = self.select_file_detailed(file_patch, {})
                        coarse_mask |= fm
                finally:
                    self._no_import_check = False
                if self._no_verify:
                    # The import check runs the patched code, which this commit cannot overlay.
                    mask, owners, lazy_pp_mask = coarse_mask, [], 0
            reason = "noop" if all_noop and mask == 0 else "mapped"
            return Selection(
                mask=mask,
                reason=reason,
                # No patched files means no runtime verification (deleted/renamed/binary files).
                patched_files={} if self._no_verify else patched_files,
                hot_patch_targets=hot_targets,
                attr_hot_targets=attr_targets,
                fixture_only=all_fixture_or_noop and not all_noop,
                lazy_pp_mask=lazy_pp_mask,
                import_checks=owners,
                coarse_mask=coarse_mask,
                edited_spans=spans,
            )
        except Everything as reason:
            return Selection(impact.everything, f"everything ({reason})", {}, None, None, False)
        except Exception as error:  # never let an unexpected input crash the run
            return Selection(
                impact.everything,
                f"everything (error {type(error).__name__}: {error})",
                {},
                None,
                None,
                False,
            )

    def select_file_detailed(
        self,
        fp: FilePatch,
        patched_files: dict[str, str],
    ) -> tuple[
        int,
        bool,
        bool,
        list[tuple[str, str]] | None,
        list[tuple[str, str, str, object]] | None,
    ]:
        impact = self.impact
        if not (fp.new and fp.new == fp.old and not fp.binary and fp.new.startswith("sqlglot/") and fp.new.endswith(".py")):
            self._spans_ok = False  # only in-place sqlglot .py edits resolve to statement spans
        if fp.binary or (fp.old is not None and fp.new is not None and fp.old != fp.new) or fp.new is None:
            # Binary, renamed or deleted: the in-memory overlay cannot reproduce these, so this
            # commit is not verified; select what imports or reads the old path (a module that
            # is gone fails its importers like one that cannot import) and lists its folder.
            self._no_verify = True
            mask = 0
            for p in {fp.old, fp.new} - {None}:
                mask |= self._readers(p)
                if p.endswith(".py") and p.startswith(("sqlglot/", "tests/")) and p == fp.old:
                    mask |= impact.module_level(p)
            return mask, False, False, None, None
        path = fp.new
        if fp.old is None:
            # New file: no pinned code imports or opens it; tests that list its folder may see it.
            # The overlay serves it, so verification still works.
            patched_files[path] = apply("", fp.hunks)
            return self._readers(path), False, False, None, None
        if path.startswith("tests/fixtures/"):
            file = SOURCE / path
            if not file.is_file():
                raise Everything(f"no pinned fixture for {path}")
            old_text = file.read_text(encoding="utf-8")
            patched_files[path] = apply(old_text, fp.hunks)
            return self.select_fixture(path), False, True, [], []
        if path in PYTEST_CONFIG_FILES:
            raise Everything(f"pytest configuration {path}")
        if path.endswith(".py") and path.startswith(("sqlglot/", "tests/")):
            if path.startswith("tests/") and (path.endswith("__init__.py") or path.endswith("conftest.py")):
                # Package plumbing: a real change reaches every test collected under its folder.
                py_mask, py_noop, new_text, _, _ = self.select_python_detailed(path, fp.hunks)
                if py_noop:
                    return 0, True, True, [], []
                patched_files[path] = new_text
                folder = path.rsplit("/", 1)[0] + "/"
                under = sum(1 << i for i, t in enumerate(impact.test_ids) if t.startswith(folder))
                return under | py_mask, False, False, None, None
            if path.startswith("sqlglot/") and path not in impact.lines and path not in impact.importers:
                # Never imported or run by any test: only tests that open the file can notice.
                return self._readers(path), self._readers(path) == 0, False, [], []
            # Always include tests that directly opened this .py file (e.g. test_typing.py),
            # except when a sqlglot/typing/*.py change is a pure comment or AST-identical noop
            # (test_no_redundant_expression_metadata only inspects AST imports and EXPRESSION_METADATA).
            readers = impact.file_readers(path)
            py_mask, py_noop, new_text, targets, attr_targets = self.select_python_detailed(path, fp.hunks)
            if py_noop and path.startswith("sqlglot/typing/"):
                readers = 0
            if not py_noop or readers:
                patched_files[path] = new_text
            return readers | py_mask, py_noop and readers == 0, py_noop, targets, attr_targets
        # Docs, packaging, CI config, the Rust tokenizer (not installed) ...: only tests that
        # open the file or list its folder can notice a change.
        readers = self._readers(path)
        if readers:
            file = SOURCE / path
            if not file.is_file():
                raise Everything(f"no pinned source for {path}")
            patched_files[path] = apply(file.read_text(encoding="utf-8"), fp.hunks)
        return readers, readers == 0, readers == 0, [], []

    def _readers(self, path: str) -> int:
        """Tests that opened the file or listed its folder in the pinned run."""
        impact = self.impact
        mask = 0
        folder = path.rsplit("/", 1)[0] if "/" in path else ""
        for key in ("f:" + path, "d:" + folder):
            index = impact.paths.get(key)
            if index is not None:
                mask |= impact.mask(index)
        return mask

    def select_fixture(self, path: str) -> int:
        # A fixture no test opened (nor listed its folder) cannot change any test's outcome.
        return self._readers(path)

    def select_python_detailed(
        self,
        path: str,
        hunks,
    ) -> tuple[
        int,
        bool,
        str,
        list[tuple[str, str]] | None,
        list[tuple[str, str, str, object]] | None,
    ]:
        old_text, old_scopes = self.old_source(path)
        new_text = apply(old_text, hunks)

        changed = [line[1:] for _, _, _, body in hunks for line in body if line[0] in "+-"]
        if comment_lines_only(changed) and same_code(old_text, new_text):
            return 0, True, new_text, [], []
        try:
            new_tree = ast.parse(new_text)
            compile(new_tree, path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            # The module cannot import: exactly what an import-time change reaches fails
            # (plus test_docs.py for sqlglot/ modules walked dynamically by pkgutil).
            self._spans_ok = False
            mask = self.impact.module_level(path)
            if path.startswith("sqlglot/"):
                mask |= self.docs_mask
            return mask, False, new_text, None, None
        new_scopes = Scopes(new_tree, has_lambda="lambda" in new_text)

        # Normalized AST equivalence catches quote/paren/wrapping refactors, TYPE_CHECKING-only
        # edits, and pure type-annotation edits when `from __future__ import annotations` holds.
        blocks = list(change_blocks(hunks))
        changed_old_lines = {ln for rem, _ in blocks for ln, _ in rem}
        changed_new_lines = {ln for _, add in blocks for ln, _, _ in add}
        ignore_ann = (
            old_scopes.has_future_annotations
            and new_scopes.has_future_annotations
            and self.impact.file_readers(path) == 0
        )
        if same_ast(
            old_scopes.tree,
            new_tree,
            old_text,
            new_text,
            ignore_annotations=ignore_ann,
            old_lines=changed_old_lines,
            new_lines=changed_new_lines,
        ):
            return 0, True, new_text, [], []

        module = module_name(path)
        mask = 0
        self._coarse_block = False
        owners_before = set(self._owner_checks)
        for removed, added in blocks:
            mask |= self.select_block(path, module, old_scopes, new_scopes, removed, added, ignore_ann)

        if self.values is not None and path.startswith("sqlglot/"):
            # Every difference is a rearranged condition over the same operands: replay the
            # pinned run's recorded outcomes (import-time ones included) through the edit.
            try:
                vmask = self.values.narrow(path, old_scopes.tree, new_tree, self.impact, hunks)
            except Exception as error:  # noqa: BLE001 - fall back to the rules above
                log(f"value map error on {path}: {type(error).__name__}: {error}")
                vmask = None
            if vmask is not None:
                log(f"{path}: value map narrows {bin(mask).count('1')} -> {bin(vmask).count('1')} tests")
                mask = vmask
                self._owner_checks = owners_before
                self._coarse_block = False

        targets: list[tuple[str, str]] | None = None
        attr_targets: list[tuple[str, str, str, object]] | None = None
        if path.startswith("sqlglot/"):
            # Every block resolved per statement means no edited statement ran before the tests, or
            # only in imports whose built state the runtime proves unchanged: either way, swapping
            # __code__ after a pre-warmed import matches importing the patched module.
            ran_before = self.runs_before_tests if self._coarse_block else (lambda *_: False)
            targets = _extract_hot_patch_targets(path, old_scopes, new_scopes, hunks, ran_before)
            if targets is None:
                attr_targets = _extract_attr_hot_targets(path, module, old_scopes, new_scopes, hunks, self.impact)
        return mask, False, new_text, targets, attr_targets

    def select_block(
        self,
        path,
        module,
        old_scopes: Scopes,
        new_scopes: Scopes,
        removed,
        added,
        ignore_ann: bool = False,
    ) -> int:
        # Edits strictly inside `if TYPE_CHECKING:` bodies never execute at runtime.
        if (removed or added) and (
            all(old_scopes.in_type_checking(n) for n, _ in removed)
            and all(new_scopes.in_type_checking(n) for n, _, _ in added)
        ):
            return 0

        # Try fine-grained module-level EXPRESSION_METADATA dict-key tracking for sqlglot/typing/*.py.
        if path in self.impact.typing_dialects:
            t_mask = self.typing_dict_level(path, old_scopes, new_scopes, removed, added)
            if t_mask is not None:
                self._spans_ok = False
                return t_mask

        # Try fine-grained class dict-key or class attribute tracking first.
        attr_mask = self.class_attr_level(path, module, old_scopes, new_scopes, removed, added)
        if attr_mask is not None:
            self._spans_ok = False
            return attr_mask | self._test_file_own_mask(path, old_scopes, new_scopes, removed, added)

        # One-for-one edit of statements inside one function: the tests that ran them.
        if removed and len(removed) == len(added):
            mask = self.statement_level(path, old_scopes, new_scopes, removed, added)
            if mask is not None:
                return mask | self._test_file_own_mask(path, old_scopes, new_scopes, removed, added)

        # Multi-line edits or insertions confined to a single function body: AST statement diff.
        ast_mask = self.ast_statement_level(path, old_scopes, new_scopes, removed, added, ignore_ann)
        if ast_mask is not None:
            return ast_mask | self._test_file_own_mask(path, old_scopes, new_scopes, removed, added)

        self._coarse_block = True
        self._spans_ok = False
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
                    qual = fn[3].split(".<locals>.", 1)[0]
                    bit = impact.by_test_qualname.get((path, qual))
                    if bit:
                        mask |= bit
                        continue
                    # Helper method inside a test class that already has line coverage: keep narrow coverage.
                    if impact.span(path, fn[0], fn[1]):
                        continue
                cls = scopes.class_at(lineno)
                if cls is not None:
                    cls_mask = impact.by_class.get(f"{path}::{cls[2].replace('.', '::')}", 0)
                    mask |= cls_mask
        return mask

    def typing_dict_level(self, path: str, old_scopes: Scopes, new_scopes: Scopes, removed, added) -> int | None:
        """Resolve an edit confined to top-level EXPRESSION_METADATA in sqlglot/typing/*.py by dict key."""
        if not removed and not added:
            return None
        for old_no, _ in removed:
            info = old_scopes.module_assign_at(old_no)
            if info is None or info[0] != "EXPRESSION_METADATA":
                return None
        for new_no, _anchor, _ in added:
            info = new_scopes.module_assign_at(new_no)
            if info is None or info[0] != "EXPRESSION_METADATA":
                return None
        old_stmt = old_scopes.module_assigns.get("EXPRESSION_METADATA")
        new_stmt = new_scopes.module_assigns.get("EXPRESSION_METADATA")
        if old_stmt is None or new_stmt is None:
            return None
        old_parsed = _extract_dict_entries(old_stmt.value)
        new_parsed = _extract_dict_entries(new_stmt.value)
        if old_parsed is None or new_parsed is None:
            return None
        old_entries, old_spreads = old_parsed
        new_entries, new_spreads = new_parsed
        if old_spreads != new_spreads:
            return None
        # _TrackedDict records key hits, not misses: key removals, reorderings, and value
        # edits are exact; newly added keys fall back conservatively.
        if not (set(new_entries) <= set(old_entries)):
            return None
        changed_keys = [
            toks
            for toks in old_entries
            if toks not in new_entries or ast.dump(old_entries[toks]) != ast.dump(new_entries[toks])
        ]
        same_order = list(old_entries) == list(new_entries)
        if not changed_keys and same_order:
            return 0
        impact = self.impact
        mask = 0
        changed_tokens = [t for toks in changed_keys for t in toks]
        for cls_key in impact.typing_dialects.get(path, ()):
            if not same_order and (cls_key in impact.pre_hook_classes or impact.read_at_import(cls_key, "EXPRESSION_METADATA")):
                return None
            if changed_tokens and impact.read_at_import(cls_key, "EXPRESSION_METADATA", changed_tokens):
                return None
            mask |= impact.dict_key_usage(cls_key, "EXPRESSION_METADATA", "*")
            for toks in changed_keys:
                for tok in toks:
                    mask |= impact.dict_key_usage(cls_key, "EXPRESSION_METADATA", tok)
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
            if key_id in impact.tracked_dicts:
                old_parsed = _extract_dict_entries(old_val)
                new_parsed = _extract_dict_entries(new_val)
                if old_parsed is not None and new_parsed is not None:
                    old_entries, old_spreads = old_parsed
                    new_entries, new_spreads = new_parsed
                    # _TrackedDict records key hits, not misses: key removals, reorderings (when
                    # no import-time key iteration exists), and value edits are exact; added keys
                    # fall back conservatively.
                    if old_spreads == new_spreads and set(new_entries) <= set(old_entries):
                        same_order = list(old_entries) == list(new_entries)
                        if not same_order and (
                            cls_key in impact.pre_hook_classes or impact.read_at_import(cls_key, attr)
                        ):
                            return None
                        changed_keys = [
                            toks
                            for toks in old_entries
                            if toks not in new_entries or ast.dump(old_entries[toks]) != ast.dump(new_entries[toks])
                        ]
                        if not changed_keys and same_order:
                            return 0
                        changed_tokens = [tok for toks in changed_keys for tok in toks]
                        if changed_tokens and impact.read_at_import(cls_key, attr, changed_tokens):
                            return None
                        mask = impact.dict_key_usage(cls_key, attr, "*")
                        for toks in changed_keys:
                            for tok in toks:
                                mask |= impact.dict_key_usage(cls_key, attr, tok)
                        return mask
            return None

        # Case 2: Non-dict class data attribute not read at import/metaclass time.
        if key_id in impact.tracked_attrs and not impact.read_at_import(cls_key, attr):
            return impact.attr_usage(cls_key, attr)

        return None

    def statement_level(self, path, old_scopes, new_scopes, removed, added):
        mask = 0
        checked_funcs: set[str] = set()
        old_lines = {n for n, _ in removed}
        new_lines = {n for n, _, _ in added}
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
            # Reject statement-level only if non-local lexical bindings or generator flags changed.
            if qualname not in checked_funcs:
                if not same_local_binding_semantics(old_node, new_function[4], old_lines, new_lines):
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
            span = self._test_time_span(path, start, stop)
            if span is None:
                return None
            mask |= span
        return mask

    def _test_time_span(self, path: str, first: int, last: int) -> int | None:
        """Tests that ran an edited statement, or None if it also ran while importing/collecting.

        The map imports every sqlglot module before any test, so a statement that never ran
        then only ever ran inside tests, and is credited to them. Execution only diverges
        once it reaches an edited statement: if that never happened at import or collection
        time, import-time state is unchanged even when the enclosing function ran then.
        """
        if self.runs_before_tests(path, first, last):
            owners = self._import_owners(path, first, last)
            if owners is None:
                return None
            # Ran while other modules imported: tests only see that through the state those
            # imports built, which the runtime re-checks (else the coarse selection is used,
            # and hot-patching is dropped).
            self._owner_checks |= owners
        self._spans.append((path, first, last))
        return self.impact.span(path, first, last)

    def _import_owners(self, path: str, first: int, last: int) -> set[str] | None:
        """Modules whose import ran these lines, if all have a recorded state digest; else None."""
        if self._no_import_check or not self._state:
            return None
        if any(first <= n <= last for n in self.impact.collect_lines.get(path, ())):
            return None
        per_line = self._state["owners"].get(path, {})
        owners: set[str] = set()
        for n in self.impact.import_lines.get(path, ()):
            if first <= n <= last:
                names = per_line.get(n)
                if not names or any(name not in self._state["digests"] for name in names):
                    return None
                owners.update(names)
        return owners

    def ast_statement_level(
        self,
        path: str,
        old_scopes: Scopes,
        new_scopes: Scopes,
        removed,
        added,
        ignore_ann: bool,
    ) -> int | None:
        """Map multi-line edits or insertions inside one function body via recursive AST statement diff."""
        if not removed and not added:
            return None
        qualname = None
        old_fn = None
        new_fn = None
        for old_no, _ in removed:
            fn = old_scopes.function_at(old_no)
            if fn is None or old_no < fn[0]:
                return None
            if qualname is None:
                qualname, old_fn = fn[3], fn
            elif fn[3] != qualname:
                return None
        for new_no, _anchor, _ in added:
            fn = new_scopes.function_at(new_no)
            if fn is None or new_no < fn[0]:
                return None
            if qualname is None:
                qualname, new_fn = fn[3], fn
            elif fn[3] != qualname:
                return None
            else:
                new_fn = fn
        if qualname is None:
            return None
        old_rt, new_rt = _runtime_def(old_scopes, qualname), _runtime_def(new_scopes, qualname)
        if old_rt is None or new_rt is None:
            return None
        if (old_fn is not None and old_fn[4] is not old_rt[4]) or (new_fn is not None and new_fn[4] is not new_rt[4]):
            return None
        old_fn, new_fn = old_rt, new_rt
        old_node = old_fn[4]
        new_node = new_fn[4]
        old_lines = {n for n, _ in removed}
        new_lines = {n for n, _, _ in added}
        if not same_local_binding_semantics(old_node, new_node, old_lines, new_lines):
            return None
        return self._diff_stmt_lists(path, old_node.body, new_node.body, ignore_ann)

    def _diff_stmt_lists(
        self,
        path: str,
        old_stmts: list[ast.stmt],
        new_stmts: list[ast.stmt],
        ignore_ann: bool,
    ) -> int | None:
        """Return coverage mask for the minimal changed statements/sub-blocks between two statement lists."""
        old_c = [canon_ast(s, ignore_ann) for s in old_stmts]
        new_c = [canon_ast(s, ignore_ann) for s in new_stmts]

        prefix = 0
        limit = min(len(old_c), len(new_c))
        while prefix < limit and old_c[prefix] == new_c[prefix]:
            prefix += 1

        suffix = 0
        rem_old = len(old_c) - prefix
        rem_new = len(new_c) - prefix
        while suffix < min(rem_old, rem_new) and old_c[len(old_c) - 1 - suffix] == new_c[len(new_c) - 1 - suffix]:
            suffix += 1

        mid_old = old_stmts[prefix : len(old_stmts) - suffix]
        mid_new = new_stmts[prefix : len(new_stmts) - suffix]

        if not mid_old and not mid_new:
            return 0

        # Pure insertion into this block: any test reaching the insertion point must pass through
        # an adjacent dominating non-docstring sibling statement in the same block.
        if not mid_old:
            sib = next((s for s in reversed(old_stmts[:prefix]) if not _is_docstring(s)), None)
            if sib is None:
                sib = next((s for s in old_stmts[prefix:] if not _is_docstring(s)), None)
            if sib is None:
                return None
            stop = header_end(sib) if isinstance(sib, COMPOUND_NODES) else sib.end_lineno
            return self._test_time_span(path, sib.lineno, stop)

        # Single compound statement edited in place: recurse into sub-blocks if its header is unchanged.
        if len(mid_old) == 1 and len(mid_new) == 1 and type(mid_old[0]) is type(mid_new[0]):
            so, sn = mid_old[0], mid_new[0]
            sub_pairs = _matching_compound_subblocks(so, sn, ignore_ann)
            if sub_pairs is not None:
                mask = 0
                for bo, bn in sub_pairs:
                    sub_mask = self._diff_stmt_lists(path, bo, bn, ignore_ann)
                    if sub_mask is None:
                        return self._test_time_span(path, so.lineno, so.end_lineno)
                    mask |= sub_mask
                return mask
            # Only the compound header changed while all sub-blocks are identical.
            if isinstance(so, COMPOUND_NODES) and _same_compound_bodies(so, sn, ignore_ann):
                return self._test_time_span(path, so.lineno, header_end(so))

        mask = 0
        for s in mid_old:
            if _is_docstring(s):
                mask |= self.docs_mask
                continue
            span = self._test_time_span(path, s.lineno, s.end_lineno)
            if span is None:
                return None
            mask |= span
        return mask

    def runs_at_import(self, path: str, first: int, last: int) -> bool:
        lines = self.impact.import_lines.get(path)
        return bool(lines) and any(first <= n <= last for n in lines)

    def runs_before_tests(self, path: str, first: int, last: int) -> bool:
        """Runs while sqlglot or a test module imports. The verifier imports both once,
        before patching, so a hot-patched function like this would leave stale results."""
        if self.runs_at_import(path, first, last):
            return True
        lines = self.impact.collect_lines.get(path)
        return bool(lines) and any(first <= n <= last for n in lines)

    def _module_reach(self, path: str) -> int:
        impact = self.impact
        if self._no_pp_lazy:
            return impact.module_level_without_pp_lazy(path)
        full = impact.module_level(path)
        if full != impact.module_level_without_pp_lazy(path):
            self._hit_pp_lazy = True
        return full

    def function_level(self, path, module, body_start, end) -> int:
        mask = self.impact.span(path, body_start, end)
        if self.runs_at_import(path, body_start, end):
            mask |= self._module_reach(path)
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
        return self._module_reach(path)

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
