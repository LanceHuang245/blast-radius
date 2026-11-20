"""pytest plugin used only while building the test impact map (never at scoring time).

It records, for every test, three things from one run of sqlglot's suite at the
pinned commit:

* line coverage of sqlglot/ and tests/, split by test and phase via coverage
  dynamic contexts ("<nodeid>|setup", "<nodeid>|run", "<nodeid>|teardown",
  "collect|<module nodeid>");
* which files and directories each test opened or listed (fixture usage);
* which sqlglot classes each test used (dialects, tokenizers, parsers,
  generators and expression nodes, with their whole MRO).

Process pools are replaced by an in-process executor that pickles arguments and
results the same way a spawned worker would, so work the optimizer and executor
tests hand to a pool is still attributed to the test that submitted it.
"""

from __future__ import annotations

import builtins
import concurrent.futures as cf
import io
import json
import os
import pickle

import coverage
import pytest

OUT = os.environ.get("MAP_OUT", "/out")
ROOT = os.getcwd()

_cov: coverage.Coverage | None = None
_ctx = ""
_opens: dict[str, set[str]] = {}
_types: dict[str, set[type]] = {}


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
    if resolved.startswith(ROOT):
        _opens.setdefault(_ctx, set()).add(kind + os.path.relpath(resolved, ROOT).replace(os.sep, "/"))


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
    if original is None or isinstance(original, (staticmethod, classmethod, property)):
        return

    def wrapper(self, *args, **kwargs):
        _record_type(self)
        return original(self, *args, **kwargs)

    wrapper.__name__ = original.__name__
    wrapper.__qualname__ = original.__qualname__
    wrapper.__doc__ = original.__doc__
    setattr(cls, name, wrapper)


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
    for name in ("__init__", "parse", "parse_into", "generate", "transpile", "tokenize",
                 "tokenizer", "jsonpath_tokenizer", "parser", "generator"):
        _wrap_method(dialect.Dialect, name)


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
    _import_everything()
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
            except Exception:
                pass


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
    with open(os.path.join(OUT, "usage.json"), "w") as handle:
        json.dump(
            {"opens": {k: sorted(v) for k, v in _opens.items()}, "classes": used, "outcomes": _outcomes},
            handle,
        )
