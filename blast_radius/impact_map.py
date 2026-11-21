"""Query interface over the precomputed impact map (map/impact.json.gz)."""

from __future__ import annotations

import gzip
import json
import marshal
from pathlib import Path


class ImpactMap:
    """In-memory query interface over the precomputed pinned-sqlglot impact map."""

    def __init__(self, path: Path):
        marshal_path = path.parent / "impact.marshal"
        if marshal_path.is_file():
            with open(marshal_path, "rb") as handle:
                raw = marshal.load(handle)
        else:
            with gzip.open(path, "rt") as handle:
                raw = json.load(handle)
        self.test_ids: list[str] = raw["tests"]
        self._sets = raw["sets"]
        self._sets_binary: bool = bool(raw.get("sets_binary", False))
        self._decoded: dict[int, int] = {}
        self.lines: dict[str, dict[str, int]] = raw["lines"]
        self.import_lines = {f: set(v) for f, v in raw["import_lines"].items()}
        self.collect_lines = {f: set(v) for f, v in raw.get("collect_lines", {}).items()}
        self.collect_attrs: set[str] = set(raw.get("collect_attrs", []))
        self.paths: dict[str, int] = raw["paths"]
        self.classes: dict[str, int] = raw["classes"]
        self.class_refs: dict[str, int] = raw.get("class_refs", {})
        self.tracked_dicts: set[str] = set(raw.get("tracked_dicts", []))
        self.dict_keys: dict[str, int] = raw.get("dict_keys", {})
        self.tracked_attrs: set[str] = set(raw.get("tracked_attrs", []))
        self.attrs: dict[str, int] = raw.get("attrs", {})
        self.import_any_attrs: set[str] = set(raw.get("import_any_attrs", []))
        self.import_value_attrs: set[str] = set(raw.get("import_value_attrs", []))
        self.import_reads: dict[str, set[str]] = {k: set(v) for k, v in raw.get("import_reads", {}).items()}
        self.pre_hook_classes: set[str] = set(raw.get("pre_hook_classes", []))
        self.importers: dict[str, list[str]] = raw["importers"]
        self.lazy_importers: dict[str, list[list]] = raw.get("lazy_importers", {})
        self.everything = (1 << len(self.test_ids)) - 1

        self.by_file: dict[str, int] = {}
        self.by_class: dict[str, int] = {}
        self.by_test_qualname: dict[tuple[str, str], int] = {}
        for i, test_id in enumerate(self.test_ids):
            bit = 1 << i
            parts = test_id.split("::")
            file = parts[0]
            self.by_file[file] = self.by_file.get(file, 0) | bit
            if len(parts) >= 3:
                cls_key = "::".join(parts[:-1])
                self.by_class[cls_key] = self.by_class.get(cls_key, 0) | bit
                self.by_test_qualname[(file, ".".join(parts[1:]))] = bit
            elif len(parts) == 2:
                self.by_test_qualname[(file, parts[1])] = bit

        self._touch: dict[str, int] = {}
        self._module_level: dict[str, int] = {}

    def mask(self, set_index: int) -> int:
        value = self._decoded.get(set_index)
        if value is None:
            entry = self._sets[set_index]
            value = int.from_bytes(entry, "little") if self._sets_binary else int(entry, 16)
            self._decoded[set_index] = value
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

    def file_readers(self, path: str) -> int:
        index = self.paths.get("f:" + path)
        return 0 if index is None else self.mask(index)

    def class_usage(self, module: str, qualname: str) -> int | None:
        index = self.classes.get(f"{module}:{qualname}")
        return None if index is None else self.mask(index)

    def class_ref_usage(self, module: str, qualname: str) -> int:
        index = self.class_refs.get(f"{module}:{qualname}")
        return 0 if index is None else self.mask(index)

    def dict_key_usage(self, cls_key: str, attr: str, key_tok: str) -> int:
        index = self.dict_keys.get(f"{cls_key}|{attr}|{key_tok}")
        return 0 if index is None else self.mask(index)

    def attr_usage(self, cls_key: str, attr: str) -> int:
        index = self.attrs.get(f"{cls_key}|{attr}")
        return 0 if index is None else self.mask(index)

    def read_at_import(self, cls_key: str, attr: str, tokens: list[str] | None = None) -> bool:
        """Was this class attribute read while sqlglot imported? For a dict, pass the
        changed key tokens: only those keys (or a read of all values) count.

        Classes created after recording started have exact per-class reads; the base
        classes that existed before fall back to the name-based sets.
        """
        if cls_key in self.pre_hook_classes or not self.import_reads:
            return attr in (self.import_value_attrs if tokens is not None else self.import_any_attrs)
        read = self.import_reads.get(f"{cls_key}|{attr}")
        if not read:
            return False
        if tokens is None:
            return True
        return "*" in read or "-" in read or any(token in read for token in tokens)

    def module_classes(self, module: str) -> int:
        prefix = module + ":"
        mask = 0
        for table in (self.classes, self.class_refs):
            for name, index in table.items():
                if name.startswith(prefix):
                    mask |= self.mask(index)
        return mask

    def module_level(self, file: str) -> int:
        """Everything an import-time change to `file` can reach (eager + lazy)."""
        if file in self._module_level:
            return self._module_level[file]
        self._module_level[file] = 0  # cycle guard
        mask = (
            self.touch(file)
            | self.by_file.get(file, 0)
            | self.module_classes(module_name(file))
            | self.file_readers(file)
        )
        for importer in self.importers.get(file, []):
            mask |= self.module_level(importer)
        for lazy_file, lazy_lineno in self.lazy_importers.get(file, []):
            mask |= self.line(lazy_file, lazy_lineno)
            if lazy_lineno in self.import_lines.get(lazy_file, ()):
                mask |= self.module_level(lazy_file)
        self._module_level[file] = mask
        return mask


def module_name(file: str) -> str:
    name = file[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name
