"""Run given test files split into N shards by node id (for the slow files)."""
import json, os, subprocess, sys, time
n = int(sys.argv[1]); prefix = sys.argv[2]; files = sys.argv[3:]
skip = set(os.environ.get("SKIP_IDS", "").split(",")) - {""}
tests = [t for t in json.load(open("/tests.json"))["tests"] if t["file"] in files and t["id"] not in skip]
shards = [[0.0, []] for _ in range(n)]
for t in sorted(tests, key=lambda t: -t["seconds"]):
    s = min(shards, key=lambda s: s[0]); s[0] += t["seconds"] + 0.05; s[1].append(t["id"])
procs = []
t0 = time.time()
for i, (_, ids) in enumerate(shards):
    env = dict(os.environ, VALUE_SHARD=f"{prefix}{i}")
    procs.append(subprocess.Popen([sys.executable, "-m", "pytest", *ids, "-p", "value_plugin", "-p", "no:cacheprovider", "-q", "-o", "addopts="],
                                  env=env, stdout=open(f"/out/log_{prefix}{i}.txt", "w"), stderr=subprocess.STDOUT))
for i, p in enumerate(procs):
    p.wait(); print(f"shard {prefix}{i} rc={p.returncode} t={time.time()-t0:.0f}s", flush=True)
