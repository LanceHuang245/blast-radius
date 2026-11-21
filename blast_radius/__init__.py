"""Blast Radius selector: pick the tests a patch can break, using a map built
from the pinned sqlglot commit (see tools/), then split them across workers.

How a patch is mapped to tests:

* Every file also unions any test that opened the file directly (e.g. fixture
  files and source-inspecting tests like tests/test_typing.py).
* A Python file is patched in memory against the pinned source. A change that
  only touches comments or whitespace (without altering encoding cookies) selects
  only direct file readers; one that no longer compiles selects everything.
* A class-level dict value edit with unchanged keys selects tests that looked up
  those keys (or read all values), provided the dict's values are not consumed at
  import/metaclass time. A class-level data attribute edit not read at import
  time selects tests that read that attribute on the class or a subclass.
* A one-for-one line edit inside a function body selects the tests that ran the
  edited statement, provided the function's local binding / generator shape is
  unchanged. Other edits inside a function select tests that ran the function.
* Other class-level edits select tests that used or statically referenced that
  class plus lines inside that class. Module-level edits propagate through both
  eager and lazy import graphs.
* Anything the map cannot explain selects every test.
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA = Path("/data")
APP = Path(__file__).resolve().parent.parent
SOURCE = APP / "sqlglot-src"
MAP_FILE = APP / "map" / "impact.json.gz"


def log(message: str) -> None:
    print(message, file=sys.stderr)


class Everything(Exception):
    """Raised when a change cannot be mapped; the commit then runs every test."""
