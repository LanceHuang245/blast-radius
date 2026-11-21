"""Entry point: read /data, run static selection + dual-phase runtime verification, emit JSON."""

from __future__ import annotations

import gc
import json
import os
import sys

from blast_radius import DATA, MAP_FILE, log
from blast_radius.impact_map import ImpactMap
from blast_radius.schedule import split
from blast_radius.select import Selection, Selector


def main() -> None:
    gc.disable()
    # Dedicate a private fd to JSON answers and redirect fd 1 -> fd 2 so imported
    # libraries or executed tests can never pollute stdout.
    out_fd = os.dup(1)
    os.dup2(2, 1)
    out = os.fdopen(out_fd, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr

    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    number_by_id = {t["id"]: t["number"] for t in tests}
    id_by_number = {t["number"]: t["id"] for t in tests}
    ms_by_number = {t["number"]: round(t["seconds"] * 1000) for t in tests}

    impact = ImpactMap(MAP_FILE)
    # Uncalibrated tests: in-process CPU is ~1.3x the recorded seconds (median of calibrated ones).
    cpu_by_number = {
        t["number"]: impact.real_cpu_ms.get(t["id"], round(t["seconds"] * 1300)) for t in tests
    }
    selector = Selector(impact)
    numbers = [number_by_id.get(test_id) for test_id in impact.test_ids]
    mapped_ids = set(impact.test_ids)
    unmapped = [n for tid, n in number_by_id.items() if tid not in mapped_ids]

    # Phase 0: static selection across all commits.
    entries: list[tuple[str, set[int], Selection]] = []
    for patch in sorted((DATA / "commits").glob("*.patch")):
        sel = selector.select_detailed(patch.read_text(encoding="utf-8", errors="replace"))
        if sel.import_checks:
            # An edited statement ran during other modules' imports: keep the narrow selection
            # only if those modules build the same import-time state under the patch.
            from blast_radius.verify import check_import_state

            same = os.environ.get("BR_VERIFY", "1") != "0" and check_import_state(sel.patched_files, sel.import_checks)
            log(f"{patch.stem}: import state of {sel.import_checks} {'unchanged' if same else 'CHANGED'}")
            if not same:
                sel.mask, sel.lazy_pp_mask = sel.coarse_mask, 0
                # Import-time state differs: a pre-warmed import would be stale, and the
                # changes reach tests beyond the edited statements.
                sel.hot_patch_targets, sel.attr_hot_targets, sel.edited_spans = None, None, None
        chosen = {numbers[i] for i in range(len(numbers)) if sel.mask >> i & 1 and numbers[i] is not None}
        if unmapped and sel.reason != "noop":
            chosen.update(unmapped)
        entries.append((patch.stem, chosen, sel))

    # Phase 1 & 2: cost-aware runtime verification in isolated forked children.
    # BR_VERIFY=0 skips it entirely (no verify imports, no pre-warm, no TPC cache).
    verified: dict[str, set[int]] = {}
    if os.environ.get("BR_VERIFY", "1") != "0":
        from blast_radius.verify import plan_verification, verify_batches

        phase1, phase2 = plan_verification(entries, id_by_number, ms_by_number, cpu_by_number, numbers)
        verified = verify_batches(phase1, phase2, id_by_number, ms_by_number, out_fd, cpu_by_number, len(entries))

    for commit, chosen, sel in entries:
        final_chosen = verified.get(commit, chosen)
        workers = split([(ms_by_number[n], n) for n in final_chosen])
        out.write(json.dumps({"commit": commit, "workers": workers}) + "\n")
        tag = f"{sel.reason}->verified({len(chosen)}->{len(final_chosen)})" if commit in verified else sel.reason
        log(f"{commit}: {len(final_chosen)} tests, {len(workers)} workers, {tag}")

    out.flush()
    out.close()


if __name__ == "__main__":
    main()
