"""Run sqlglot's suite under tools/value_plugin.py in parallel shards (by test file)."""
import json, os, subprocess, sys, time

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
tests = json.load(open("/tests.json"))["tests"]
cost = {}
for t in tests:
    cost[t["file"]] = cost.get(t["file"], 0) + t["seconds"]
shards = [[0.0, []] for _ in range(n)]
for f in sorted(cost, key=lambda f: -cost[f]):
    s = min(shards, key=lambda s: s[0])
    s[0] += cost[f] + 0.3
    s[1].append(f)
procs = []
t0 = time.time()
for i, (_, files) in enumerate(shards):
    env = dict(os.environ, VALUE_SHARD=str(i))
    log = open(f"/out/log_{i}.txt", "w")
    procs.append(subprocess.Popen(
        [sys.executable, "-m", "pytest", *files, "-p", "value_plugin", "-p", "no:cacheprovider", "-q", "-o", "addopts="],
        env=env, stdout=log, stderr=subprocess.STDOUT))
for i, p in enumerate(procs):
    p.wait()
    print(f"shard {i} rc={p.returncode} t={time.time()-t0:.0f}s", flush=True)
