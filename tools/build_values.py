"""Fold value_plugin shards into map/values.bin (gzip'd marshal).

Usage: build_values.py <out dir with values_*.pkl> <tests.json> <map/values.bin>

Contexts are credited like the coverage map: a test's run phase to the test, its
setup/teardown (which hosts setUpClass/tearDownClass) to every test of its class,
collection to every test of the module, and anything outside a test to "import".
"""

import collections
import glob
import gzip
import marshal
import json
import pickle
import sys

out_dir, tests_path, dest = sys.argv[1:4]
ids = [t["id"] for t in json.load(open(tests_path))["tests"]]
index = {t: i for i, t in enumerate(ids)}
by_prefix = collections.defaultdict(list)
for i, t in enumerate(ids):
    parts = t.split("::")
    for k in range(1, len(parts)):
        by_prefix["::".join(parts[:k])].append(i)

# (file, key) -> vec -> set(test index); import vectors separately.
sites: dict = collections.defaultdict(lambda: collections.defaultdict(set))
imports: dict = collections.defaultdict(set)
outcomes = {}
unknown_ctx = collections.Counter()
for path in sorted(glob.glob(f"{out_dir}/values_*.pkl")):
    data = pickle.load(open(path, "rb"))
    outcomes.update(data["outcomes"])
    where = data["sites"]
    for ctx, recs in data["rec"].items():
        if ctx == "":
            targets = None
        elif ctx.startswith("collect|"):
            targets = by_prefix.get(ctx[len("collect|"):], [])
        else:
            nodeid, phase = ctx.rsplit("|", 1)
            if phase == "run":
                targets = [index[nodeid]] if nodeid in index else []
            else:
                parent = nodeid.rsplit("::", 1)[0]
                targets = by_prefix.get(parent, []) or ([index[nodeid]] if nodeid in index else [])
            if not targets:
                unknown_ctx[ctx] += 1
        for sid, vecs in recs.items():
            site = where[sid]
            if targets is None:
                imports[site].update(vecs)
                continue
            bucket = sites[site]
            for v in vecs:
                bucket[v].update(targets)

files: dict = collections.defaultdict(dict)
nvec = 0
for (file, key), bucket in sites.items():
    files[file][key] = [(v, sum(1 << i for i in tests)) for v, tests in sorted(bucket.items())]
    nvec += len(bucket)
imp: dict = collections.defaultdict(dict)
for (file, key), vecs in imports.items():
    imp[file][key] = sorted(vecs)

blob = gzip.compress(marshal.dumps({"tests": ids, "files": dict(files), "import": dict(imp)}), 9)
open(dest, "wb").write(blob)
bad = {k: v for k, v in outcomes.items() if v != "passed"}
print(f"{len(sites)} sites with test vectors, {nvec} (site, vec) entries, {sum(len(v) for v in imports.values())} import vectors")
print(f"{len(blob) / 1e6:.2f} MB -> {dest}; {len(outcomes)} outcomes, {len(bad)} not passed: {sorted(bad)[:20]}")
print(f"unattributed contexts: {list(unknown_ctx)[:10]}")
