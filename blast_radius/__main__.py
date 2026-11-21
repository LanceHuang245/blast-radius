"""Entry point: read /data, print one JSON line per commit to stdout."""

from __future__ import annotations

import json

from blast_radius import DATA, MAP_FILE, log
from blast_radius.impact_map import ImpactMap
from blast_radius.schedule import split
from blast_radius.select import Selector


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    number_by_id = {t["id"]: t["number"] for t in tests}
    ms_by_number = {t["number"]: round(t["seconds"] * 1000) for t in tests}

    impact = ImpactMap(MAP_FILE)
    selector = Selector(impact)
    numbers = [number_by_id.get(test_id) for test_id in impact.test_ids]
    mapped_ids = set(impact.test_ids)
    unmapped = [n for tid, n in number_by_id.items() if tid not in mapped_ids]

    for patch in sorted((DATA / "commits").glob("*.patch")):
        mask, reason = selector.select(patch.read_text(encoding="utf-8", errors="replace"))
        chosen = {numbers[i] for i in range(len(numbers)) if mask >> i & 1 and numbers[i] is not None}
        if unmapped and reason != "noop":
            chosen.update(unmapped)
        workers = split([(ms_by_number[n], n) for n in chosen])
        print(json.dumps({"commit": patch.stem, "workers": workers}), flush=True)
        log(f"{patch.stem}: {len(chosen)} tests, {len(workers)} workers, {reason}")


if __name__ == "__main__":
    main()
