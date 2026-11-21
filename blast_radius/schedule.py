"""Worker count and makespan packing for a selected set of tests."""

from __future__ import annotations

MAX_WORKERS = 8
WORKER_COST_MS = 2000


def _lpt_pack(ordered: list[tuple[int, int]], count: int) -> tuple[int, list[list[int]]]:
    loads = [0] * count
    workers: list[list[int]] = [[] for _ in range(count)]
    for ms, number in ordered:
        lightest = loads.index(min(loads))
        loads[lightest] += ms
        workers[lightest].append(number)
    return max(loads), workers


def _exact_pack(ordered: list[tuple[int, int]], count: int, upper: int, lb: int) -> tuple[int, list[list[int]]] | None:
    """Bounded branch-and-bound on the heaviest tests to beat LPT makespan, then LPT-fill the tail."""
    head = ordered[:18]
    tail = ordered[18:]
    n = len(head)
    loads = [0] * count
    assign = [0] * n
    best_max = upper
    best_assign: list[int] | None = None
    nodes = 0
    node_limit = 3000

    def dfs(idx: int, cur_max: int) -> None:
        nonlocal best_max, best_assign, nodes
        if nodes >= node_limit or best_max <= lb:
            return
        if idx == n:
            if cur_max < best_max:
                best_max = cur_max
                best_assign = assign.copy()
            return
        nodes += 1
        ms = head[idx][0]
        seen_loads: set[int] = set()
        # Try emptier bins first; skip symmetric bins with identical current load.
        for b in sorted(range(count), key=loads.__getitem__):
            ld = loads[b]
            if ld in seen_loads:
                continue
            new_ld = ld + ms
            if new_ld >= best_max:
                break
            seen_loads.add(ld)
            loads[b] = new_ld
            assign[idx] = b
            dfs(idx + 1, new_ld if new_ld > cur_max else cur_max)
            loads[b] = ld
            if nodes >= node_limit or best_max <= lb:
                return

    dfs(0, 0)
    if best_assign is None:
        return None
    workers: list[list[int]] = [[] for _ in range(count)]
    final_loads = [0] * count
    for (ms, num), b in zip(head, best_assign):
        workers[b].append(num)
        final_loads[b] += ms
    for ms, num in tail:
        lightest = final_loads.index(min(final_loads))
        workers[lightest].append(num)
        final_loads[lightest] += ms
    return max(final_loads), workers


def split(tests: list[tuple[int, int]]) -> list[list[int]]:
    """Optimal worker count and near-exact makespan schedule for (ms, number) tests."""
    if not tests:
        return []
    ordered = sorted(tests, reverse=True)
    total = sum(ms for ms, _ in ordered)
    heaviest = ordered[0][0]
    best_cost = total + WORKER_COST_MS
    best_workers: list[list[int]] = [[num for _, num in ordered]]

    for count in range(2, min(MAX_WORKERS, len(ordered)) + 1):
        lb = max(heaviest, (total + count - 1) // count)
        if lb + WORKER_COST_MS * count >= best_cost:
            continue
        makespan, workers = _lpt_pack(ordered, count)
        if makespan > lb:
            improved = _exact_pack(ordered, count, makespan, lb)
            if improved is not None and improved[0] < makespan:
                makespan, workers = improved
        cost = makespan + WORKER_COST_MS * count
        if cost < best_cost:
            best_cost = cost
            best_workers = workers
    return best_workers
