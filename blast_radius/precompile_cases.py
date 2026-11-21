"""Per-case coverage of TestOptimizer.check_file fixture cases (build time).

check_file runs one fixture case per subTest: optimize the SQL, render it, compare it, and
sometimes execute it. Recording which sqlglot lines each case runs lets the verifier rerun only
the cases an edit reaches and still prove the whole test passes: a case that never reaches an
edited statement repeats its pinned (passing) run. The real test spreads cases over pool workers
in no fixed order, so a case's outcome cannot depend on the others.

Output map/cases.marshal:
    {test_id: {"cases": [(fixture_file, index), ...], "lines": {rel_path: {line: case_mask}}}}

Build: python -m blast_radius.precompile_cases
"""

from __future__ import annotations

import marshal
import os
import sys
import unittest
from collections import defaultdict
from contextlib import contextmanager

from blast_radius import APP, SOURCE

CASES_FILE = APP / "map" / "cases.marshal"
TEST_FILE = "tests/test_optimizer.py"


def main() -> None:
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    from blast_radius import verify as V
    from blast_radius.import_state import import_all_sqlglot

    import_all_sqlglot()
    import sqlglot.generator as generator
    import tests.test_optimizer as T

    V._patch_pool_references()
    src_prefix = str(SOURCE.resolve()) + "/sqlglot/"

    state = {"in_check": None, "pending": [], "depth": 0, "case": None}
    current_lines: set[tuple[str, int]] = set()
    per_test: dict[str, dict] = {}

    mon = sys.monitoring
    tool = mon.COVERAGE_ID
    mon.use_tool_id(tool, "blast_radius_cases")

    def on_line(code, line):
        if state["case"] is not None and code.co_filename.startswith(src_prefix):
            current_lines.add((code.co_filename[len(src_prefix) - len("sqlglot/") :], line))
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)

    orig_loader = T.load_sql_fixture_pairs

    def recording_loader(filename):
        for index, item in enumerate(orig_loader(filename)):
            if state["in_check"] is not None:
                state["pending"].append((filename, index))
            yield item

    T.load_sql_fixture_pairs = recording_loader

    orig_check_file = T.TestOptimizer.check_file

    def recording_check_file(self, file, *args, **kwargs):
        state["in_check"], state["pending"] = file, []
        try:
            return orig_check_file(self, file, *args, **kwargs)
        finally:
            if state["pending"]:
                state["broken"] = True  # cases yielded but never checked: mapping unusable
            state["in_check"] = None

    T.TestOptimizer.check_file = recording_check_file

    orig_subtest = unittest.TestCase.subTest

    @contextmanager
    def recording_subtest(self, msg=unittest.case._subtest_msg_sentinel, **params):
        top = state["in_check"] is not None and state["depth"] == 0
        if top:
            if not state["pending"]:
                state["broken"] = True
                with orig_subtest(self, msg, **params):
                    yield
                return
            case = state["pending"].pop(0)
            generator._DISPATCH_CACHE.clear()  # rebuilt per case so each case covers it
            current_lines.clear()
            state["case"] = case
            mon.restart_events()
            mon.set_events(tool, mon.events.LINE)
        state["depth"] += 1
        try:
            with orig_subtest(self, msg, **params):
                yield
        finally:
            state["depth"] -= 1
            if top:
                mon.set_events(tool, 0)
                state["recorder"](state["case"], set(current_lines))
                state["case"] = None

    unittest.TestCase.subTest = recording_subtest

    cls = T.TestOptimizer
    cls.setUpClass()
    for name in sorted(n for n in dir(cls) if n.startswith("test_")):
        cases: list[tuple[str, int]] = []
        lines: dict[str, dict[int, int]] = defaultdict(dict)

        def record(case, executed, cases=cases, lines=lines):
            bit = 1 << len(cases)
            cases.append(case)
            for rel, line in executed:
                lines[rel][line] = lines[rel].get(line, 0) | bit

        state.update(recorder=record, broken=False, depth=0, case=None)
        result = unittest.TestResult()
        cls(name).run(result)
        if cases and not state["broken"] and result.wasSuccessful():
            per_test[f"{TEST_FILE}::TestOptimizer::{name}"] = {"cases": cases, "lines": dict(lines)}

    mon.set_events(tool, 0)
    mon.free_tool_id(tool)
    CASES_FILE.write_bytes(marshal.dumps(per_test))
    print(
        f"case map: {len(per_test)} tests, {sum(len(v['cases']) for v in per_test.values())} cases, "
        f"{CASES_FILE.stat().st_size / 1e6:.1f} MB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
