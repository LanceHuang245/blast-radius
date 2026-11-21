"""pytest plugin used only while building the test impact map (never at scoring time).

It records, for every test, five things from one run of sqlglot's suite at the
pinned commit:

* line coverage of sqlglot/ and tests/, split by test and phase via coverage
  dynamic contexts ("<nodeid>|setup", "<nodeid>|run", "<nodeid>|teardown",
  "collect|<module nodeid>");
* which files and directories each test opened or listed (fixture and source reads);
* which sqlglot classes each test used (dialects, tokenizers, parsers,
  generators and expression nodes, with their whole MRO);
* which keys of class-level config dicts each test read by value;
* which non-dict class attributes on Dialect/Tokenizer/Parser/Generator each test read.

Process pools are replaced by an in-process executor that pickles arguments and
results the same way a spawned worker would, so work the optimizer and executor
tests hand to a pool is still attributed to the test that submitted it.
"""

from __future__ import annotations

import builtins
import concurrent.futures as cf
import inspect
import io
import json
import os
import pickle
import sys
from enum import Enum

import coverage
import pytest

OUT = os.environ.get("MAP_OUT", "/out")
ROOT = os.getcwd()

_cov: coverage.Coverage | None = None
_ctx = ""
_opens: dict[str, set[str]] = {}
_types: dict[str, set[type]] = {}
_dict_reads: dict[str, set[tuple[int, str]]] = {}
_dict_meta: list[tuple[type, str]] = []
_attr_reads: dict[str, set[tuple[type, str]]] = {}
_tracked_attrs: list[str] = []

# Reads that happen while sqlglot imports (no test context), with the source line that read.
_import_phase = False
_import_dict_reads: set[tuple[int, str, str, int]] = set()
_import_attr_reads: set[tuple[type, str, str, int]] = set()
_pre_hook_classes: set[type] = set()
_PLUGIN_FILE = os.path.abspath(__file__)


def _reader_location() -> tuple[str, int]:
    """First frame outside this plugin: the sqlglot line that performed the read."""
    frame = sys._getframe(2)
    while frame is not None and os.path.abspath(frame.f_code.co_filename) == _PLUGIN_FILE:
        frame = frame.f_back
    if frame is None:
        return "?", 0
    path = os.path.abspath(frame.f_code.co_filename)
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/") if path.startswith(ROOT) else "?"
    return rel, frame.f_lineno


def _set_ctx(ctx: str) -> None:
    global _ctx
    _ctx = ctx
    if _cov is not None:
        _cov.switch_context(ctx)


def _note_path(path, kind: str) -> None:
    if not _ctx:
        return
    try:
        resolved = os.path.abspath(os.fsdecode(path))
    except Exception:
        return
    if not resolved.startswith(ROOT):
        return
    rel = os.path.relpath(resolved, ROOT).replace(os.sep, "/")
    # Skip Python bytecode caches and pytest reading a test module while collecting it.
    if "/__pycache__/" in f"/{rel}" or rel.endswith(".pyc"):
        return
    if kind == "f:" and _ctx == "collect|" + rel:
        return
    _opens.setdefault(_ctx, set()).add(kind + rel)


def _wrap_open(original):
    def opener(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            _note_path(file, "f:")
        return original(file, *args, **kwargs)

    return opener


def _wrap_listing(original):
    def lister(path=".", *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)):
            _note_path(path, "d:")
        return original(path, *args, **kwargs)

    return lister


class _InProcessExecutor:
    """Stands in for ProcessPoolExecutor: runs each call now, through pickle."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, /, *args, **kwargs):
        future: cf.Future = cf.Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            future.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as error:  # noqa: BLE001 - mirror a worker's failure
            future.set_exception(error)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return [self.submit(fn, *args).result() for args in zip(*iterables)]

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def _record_type(obj) -> None:
    if _ctx:
        _types.setdefault(_ctx, set()).add(type(obj))


def _wrap_method(cls, name: str) -> None:
    original = cls.__dict__.get(name)
    if not inspect.isfunction(original):
        return

    def wrapper(self, *args, **kwargs):
        _record_type(self)
        return original(self, *args, **kwargs)

    wrapper.__name__ = original.__name__
    wrapper.__qualname__ = original.__qualname__
    wrapper.__doc__ = original.__doc__
    setattr(cls, name, wrapper)


def _key_str(key) -> str:
    """Normalize a dict key so runtime lookups match AST-extracted keys."""
    if isinstance(key, Enum):
        return f"e:{type(key).__name__}.{key.name}"
    if isinstance(key, str):
        return "s:" + key
    if isinstance(key, type):
        return "t:" + key.__name__
    if isinstance(key, int) and not isinstance(key, bool):
        return f"i:{key}"
    return "*"


class _TrackedDict(dict):
    """Dict subclass that records value lookups per key and whole-dict value reads."""

    __slots__ = ("_did",)

    def __init__(self, initial: dict, did: int):
        super().__init__(initial)
        self._did = did

    def _hit(self, key_token: str) -> None:
        if _ctx:
            _dict_reads.setdefault(_ctx, set()).add((self._did, key_token))
        elif _import_phase:
            _import_dict_reads.add((self._did, key_token, *_reader_location()))

    def __iter__(self):
        # Overriding __iter__ makes CPython take the slow path for {**d}, dict(d),
        # x.update(d) and f(**d): keys() then __getitem__, so those value reads are recorded.
        return super().__iter__()

    def __getitem__(self, key):
        self._hit(_key_str(key))
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._hit(_key_str(key))
        return super().get(key, default)

    def pop(self, key, *args):
        self._hit(_key_str(key))
        return super().pop(key, *args)

    def setdefault(self, key, default=None):
        self._hit(_key_str(key))
        return super().setdefault(key, default)

    def values(self):
        self._hit("*")
        return super().values()

    def items(self):
        self._hit("*")
        return super().items()

    def copy(self):
        self._hit("*")
        return super().copy()

    def __eq__(self, other):
        self._hit("*")
        return super().__eq__(other)

    def __ne__(self, other):
        self._hit("*")
        return super().__ne__(other)

    def __or__(self, other):
        self._hit("*")
        return super().__or__(other)

    def __ror__(self, other):
        self._hit("*")
        return super().__ror__(other)

    def __repr__(self):
        self._hit("*")
        return super().__repr__()


class _TrackedAttr:
    """Non-data descriptor that records per-class reads of class-level data attributes."""

    __slots__ = ("_val", "_name")

    def __init__(self, val, name: str):
        self._val = val
        self._name = name

    def __get__(self, obj, objtype=None):
        if _ctx or _import_phase:
            actual = type(obj) if obj is not None else objtype
            if actual is not None:
                if _ctx:
                    _attr_reads.setdefault(_ctx, set()).add((actual, self._name))
                else:
                    _import_attr_reads.add((actual, self._name, *_reader_location()))
        return self._val


def _all_subclasses(root: type) -> list[type]:
    seen: list[type] = []
    stack = [root]
    visited: set[type] = set()
    while stack:
        cls = stack.pop()
        if cls in visited:
            continue
        visited.add(cls)
        seen.append(cls)
        stack.extend(cls.__subclasses__())
    return seen


_SCALAR_TYPES = (int, float, bool, str, bytes, type(None), tuple, frozenset, set, list, Enum)


def _instrument_one(cls: type) -> None:
    """Wrap one class's own plain dicts and simple data attributes (idempotent)."""
    if not cls.__module__.startswith("sqlglot"):
        return
    ckey = _class_key(cls)
    for name, val in list(cls.__dict__.items()):
        if name.startswith("_") or name in ("classes",):
            continue
        if type(val) is dict and not name.endswith("_TRIE"):
            did = len(_dict_meta)
            _dict_meta.append((cls, name))
            setattr(cls, name, _TrackedDict(val, did))
        elif isinstance(val, _SCALAR_TYPES) and not callable(val):
            _tracked_attrs.append(f"{ckey}|{name}")
            setattr(cls, name, _TrackedAttr(val, name))


def _instrument_class_attrs(roots: tuple[type, ...]) -> None:
    """Wrap class dicts and simple data attributes across sqlglot's class hierarchies."""
    for root in roots:
        for cls in _all_subclasses(root):
            _instrument_one(cls)


def _install_creation_hooks(roots: tuple[type, ...]) -> None:
    """Instrument every later subclass the moment it is created.

    type.__new__ calls __init_subclass__ before the rest of a metaclass __new__
    runs, so reads made while the class is being finished (and by any import-time
    code after it) go through the tracked dicts/descriptors and are recorded per class.
    """
    for root in roots:
        original = root.__dict__.get("__init_subclass__")

        def hook(cls, __root=root, __original=original, **kwargs):
            _instrument_one(cls)
            if __original is not None:
                __original.__func__(cls, **kwargs)
            else:
                super(__root, cls).__init_subclass__(**kwargs)

        root.__init_subclass__ = classmethod(hook)


def _instrument_sqlglot() -> None:
    from sqlglot import generator, parser, tokens
    from sqlglot.dialects import dialect
    from sqlglot.expressions import core

    _wrap_method(core.Expr, "__init__")
    _wrap_method(tokens.Tokenizer, "__init__")
    _wrap_method(tokens.Tokenizer, "tokenize")
    _wrap_method(parser.Parser, "__init__")
    _wrap_method(parser.Parser, "parse")
    _wrap_method(parser.Parser, "parse_into")
    _wrap_method(generator.Generator, "__init__")
    _wrap_method(generator.Generator, "generate")

    # Wrap every public instance method on Dialect plus __init__ so cached dialect
    # instances still attribute every dialect method call to the active test.
    for name in list(dialect.Dialect.__dict__):
        if name == "__init__" or not name.startswith("_"):
            _wrap_method(dialect.Dialect, name)

    # Also record cached Dialect instances returned by Dialect.get_or_raise.
    orig_get_or_raise = dialect.Dialect.__dict__.get("get_or_raise")
    if isinstance(orig_get_or_raise, classmethod):
        fn = orig_get_or_raise.__func__

        def wrapped_get_or_raise(cls, *args, **kwargs):
            result = fn(cls, *args, **kwargs)
            if isinstance(result, dialect.Dialect):
                _record_type(result)
            return result

        dialect.Dialect.get_or_raise = classmethod(wrapped_get_or_raise)

    _instrument_class_attrs((dialect.Dialect, tokens.Tokenizer, parser.Parser, generator.Generator))


def pytest_configure(config):
    global _cov
    cf.ProcessPoolExecutor = _InProcessExecutor
    opener = _wrap_open(builtins.open)
    builtins.open = opener
    io.open = opener
    os.listdir = _wrap_listing(os.listdir)
    os.scandir = _wrap_listing(os.scandir)

    _cov = coverage.Coverage(
        data_file=os.path.join(OUT, "coverage.db"),
        source=[os.path.join(ROOT, "sqlglot"), os.path.join(ROOT, "tests")],
        config_file=False,
    )
    _cov.start()

    # Instrument the base classes, then every subclass as it is created, so reads made
    # while the rest of sqlglot imports are recorded per class (not just by name).
    global _import_phase
    from sqlglot import generator, parser, tokens
    from sqlglot.dialects import dialect

    roots = (dialect.Dialect, tokens.Tokenizer, parser.Parser, generator.Generator)
    _instrument_class_attrs(roots)
    _pre_hook_classes.update(c for root in roots for c in _all_subclasses(root))
    _install_creation_hooks(roots)
    _import_phase = True
    try:
        _import_everything()
    finally:
        _import_phase = False
    _instrument_sqlglot()


def _import_everything() -> None:
    """Import every sqlglot module now, under the empty context.

    Dialects and optimizer rules load lazily; otherwise their import-time code
    would be credited to whichever test happened to import them first.
    """
    import importlib
    import pkgutil

    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name != "sqlglot.__main__":
            try:
                importlib.import_module(info.name)
            except Exception as err:
                print(f"map_plugin: failed to import {info.name}: {err}", file=sys.stderr)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    if isinstance(collector, pytest.Module):
        _set_ctx("collect|" + collector.nodeid)
        yield
        _set_ctx("")
    else:
        yield


def _phase(name):
    @pytest.hookimpl(hookwrapper=True)
    def hook(item, *args, **kwargs):
        _set_ctx(f"{item.nodeid}|{name}")
        yield
        _set_ctx("")

    return hook


pytest_runtest_setup = _phase("setup")
pytest_runtest_call = _phase("run")
pytest_runtest_teardown = _phase("teardown")


_outcomes: dict[str, str] = {}


def pytest_runtest_logreport(report):
    if report.when == "call" or report.outcome != "passed":
        _outcomes.setdefault(report.nodeid, report.outcome)


def _class_key(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def pytest_unconfigure(config):
    if _cov is None:
        return
    _cov.stop()
    _cov.save()

    used = {
        ctx: sorted({_class_key(c) for t in types for c in t.__mro__ if c.__module__.startswith("sqlglot")})
        for ctx, types in _types.items()
    }

    # Propagate dict key reads up MRO to ancestor classes defining the same dict attribute.
    dict_usage: dict[str, list[str]] = {}
    for ctx, pairs in _dict_reads.items():
        expanded: set[str] = set()
        for did, key_tok in pairs:
            cls, attr = _dict_meta[did]
            for base in cls.__mro__:
                if base.__module__.startswith("sqlglot") and attr in base.__dict__:
                    expanded.add(f"{_class_key(base)}|{attr}|{key_tok}")
        if expanded:
            dict_usage[ctx] = sorted(expanded)

    # Propagate scalar/data attribute reads up MRO to ancestor classes defining that attribute.
    attr_usage: dict[str, list[str]] = {}
    for ctx, pairs in _attr_reads.items():
        expanded_attrs: set[str] = set()
        for cls, attr in pairs:
            for base in cls.__mro__:
                if base.__module__.startswith("sqlglot") and attr in base.__dict__:
                    expanded_attrs.add(f"{_class_key(base)}|{attr}")
        if expanded_attrs:
            attr_usage[ctx] = sorted(expanded_attrs)

    tracked_dicts = sorted({f"{_class_key(cls)}|{attr}" for cls, attr in _dict_meta})

    # Import-time reads, expanded up the MRO like runtime reads, keeping the reading line.
    import_reads: set[tuple[str, str, str, str, int]] = set()
    for did, key_tok, file, lineno in _import_dict_reads:
        cls, attr = _dict_meta[did]
        for base in cls.__mro__:
            if base.__module__.startswith("sqlglot") and attr in base.__dict__:
                import_reads.add((_class_key(base), attr, key_tok, file, lineno))
    for cls, attr, file, lineno in _import_attr_reads:
        for base in cls.__mro__:
            if base.__module__.startswith("sqlglot") and attr in base.__dict__:
                import_reads.add((_class_key(base), attr, "-", file, lineno))

    with open(os.path.join(OUT, "usage.json"), "w") as handle:
        json.dump(
            {
                "opens": {k: sorted(v) for k, v in _opens.items()},
                "classes": used,
                "dict_reads": dict_usage,
                "tracked_dicts": tracked_dicts,
                "attr_reads": attr_usage,
                "tracked_attrs": sorted(set(_tracked_attrs)),
                "import_reads": [list(r) for r in sorted(import_reads)],
                "pre_hook_classes": sorted({_class_key(c) for c in _pre_hook_classes}),
                "outcomes": _outcomes,
            },
            handle,
        )
