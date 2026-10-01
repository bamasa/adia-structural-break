"""079: six more boundary cuts per series (AUG6) — on top of the triple augmentation, nine in total.

Built in parallel shards: python build_aug6.py <shard> <n_shards>; each
process holds only its own series. Groups 200000 + sid*10 + j, j = 0..5.
Merge: python build_aug6.py merge <n_shards>."""
import sys, time, os
import numpy as np

if sys.argv[1] == "merge":
    n = int(sys.argv[2])
    parts = [np.load(f"aug6_parts/X_{i}.npy", mmap_mode="r") for i in range(n)]
    total = sum(p.shape[0] for p in parts)
    out = np.lib.format.open_memmap("AUG6_X.npy", mode="w+", dtype="float32", shape=(total, 200))
    pos = 0
    for p in parts:
        out[pos:pos + p.shape[0]] = p; pos += p.shape[0]
    out.flush(); del out
    for name, dt in (("Y", "int8"), ("G", "int64"), ("S", "int32")):
        np.save(f"AUG6_{name}.npy", np.concatenate([np.load(f"aug6_parts/{name}_{i}.npy") for i in range(n)]).astype(dt))
    print(f"AUG6: {total} rows, {len(np.unique(np.load('AUG6_G.npy')))} pseudo-series", flush=True)
    sys.exit(0)

shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
time.sleep(shard * 15)  # spread out the memory peaks while loading parquet
sys.path.insert(0, "repo/src")
import importlib.util, pandas as pd
spec = importlib.util.spec_from_file_location("sub", "repo/submissions/053-spectral/main.py")
sub = importlib.util.module_from_spec(spec); sys.modules["sub"] = sub; spec.loader.exec_module(sub)
import lightgbm as lgb, joblib
base_fc = lgb.Booster(model_str=joblib.load("resources053/model.joblib")["forecaster"])

t0 = time.time()
x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
ids = x.index.get_level_values("id")
x = x[(ids % n_shards) == shard]
yl = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
print(f"shard {shard}: loading {time.time()-t0:.0f}s", flush=True)

rows, labels, groups, steps = [], [], [], []
count = 0
for sid, part in x.groupby(level="id"):
    hist = part.loc[part.period == 1, "value"].to_numpy()
    online = part.loc[part.period == 2, "value"].to_numpy()
    if len(online) < 60:
        continue
    lab = yl.loc[sid, "target"].to_numpy()
    tau = int(lab.argmax()) if lab.max() > 0 else None
    limit = tau if tau is not None else len(online)
    if limit < 30:
        continue
    hi = max(11, min(limit, len(online) // 2))
    rng = np.random.default_rng(2_000_000 + int(sid))
    cuts = sorted({int(c) for c in rng.integers(10, hi, size=6)})
    for j, k in enumerate(cuts):
        h2 = np.concatenate([hist, online[:k]]); o2 = online[k:]
        if len(o2) < 20:
            continue
        lab2 = lab[k:]
        mon = sub.TriMonitor(h2, base_fc)
        for i, v in enumerate(o2):
            rows.append(mon.update(float(v))); labels.append(int(lab2[i]))
            groups.append(200000 + int(sid) * 10 + j); steps.append(i)
    count += 1
    if count % 200 == 0:
        print(f"shard {shard}: {count} series, {time.time()-t0:.0f}s", flush=True)

os.makedirs("aug6_parts", exist_ok=True)
np.save(f"aug6_parts/X_{shard}.npy", np.asarray(rows, dtype="float32"))
np.save(f"aug6_parts/Y_{shard}.npy", np.asarray(labels, dtype="int8"))
np.save(f"aug6_parts/G_{shard}.npy", np.asarray(groups, dtype="int64"))
np.save(f"aug6_parts/S_{shard}.npy", np.asarray(steps, dtype="int32"))
print(f"shard {shard} done {time.time()-t0:.0f}s: {len(rows)} rows from {count} series", flush=True)
