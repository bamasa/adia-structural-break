"""091: все 206 каналов для конвертированных рядов через TriMonitor посылки #30 (шардами).
python build_channels_fe.py <shard> <n_shards>   |   python build_channels_fe.py merge <n_shards>
Выход: FE_X206.npy, FE_Y.npy, FE_G.npy, FE_S.npy — выровнены по строкам, как X40/Y40/G40/S40.
"""
import importlib.util, sys, time, os, numpy as np, pandas as pd
if sys.argv[1] == "merge":
    n = int(sys.argv[2]); parts = [np.load(f"fe_parts/part_{i}.npz", allow_pickle=True) for i in range(n)]
    Xs, Ys, Gs, Ss = [], [], [], []
    for p in parts:
        for gid, X, Y in zip(p["gids"], p["Xs"], p["Ys"]):
            Xs.append(X); Ys.append(Y); Gs.append(np.full(len(Y), gid)); Ss.append(np.arange(len(Y)))
    np.save("FE_X206.npy", np.vstack(Xs).astype("float32")); np.save("FE_Y.npy", np.concatenate(Ys).astype("int8"))
    np.save("FE_G.npy", np.concatenate(Gs).astype("int64")); np.save("FE_S.npy", np.concatenate(Ss).astype("int32"))
    print(f"FE: {sum(len(y) for y in Ys)} строк, {len(Ys)} рядов"); sys.exit(0)
shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
spec = importlib.util.spec_from_file_location("sub", "repo/submissions/075-bocpd-clf/main.py")
sub = importlib.util.module_from_spec(spec); sys.modules["sub"] = sub; spec.loader.exec_module(sub)
import joblib, lightgbm as lgb
forecaster = lgb.Booster(model_str=joblib.load("resources075/model.joblib")["forecaster"])
X = pd.read_parquet("FE_series.parquet"); ix = pd.read_parquet("FE_index.parquet")
t0 = time.time(); gids, Xs, Ys = [], [], []
for i, (sid, part) in enumerate(X.groupby(level="id")):
    if int(sid) % n_shards != shard: continue
    hist = part[part.period == 1].value.to_numpy("float64"); online = part[part.period == 2].value.to_numpy("float64")
    mon = sub.TriMonitor(hist, forecaster)
    rows = np.array([mon.update(float(p)) for p in online], dtype="float32")
    tau = int(ix.loc[sid, "tau_index"]); lab = np.zeros(len(online), int)
    if tau >= 0: lab[tau:] = 1
    gids.append(int(sid)); Xs.append(rows); Ys.append(lab)
    if len(gids) % 100 == 0: print(f"шард {shard}: {len(gids)} рядов, {time.time()-t0:.0f}s", flush=True)
os.makedirs("fe_parts", exist_ok=True)
np.savez(f"fe_parts/part_{shard}.npz", gids=np.array(gids), Xs=np.array(Xs, dtype=object), Ys=np.array(Ys, dtype=object))
print(f"шард {shard} готов: {len(gids)} рядов, {time.time()-t0:.0f}s", flush=True)
