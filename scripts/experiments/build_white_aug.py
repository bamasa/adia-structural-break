"""151: whitened channels (90) + Shiryaev-Roberts odds (21) for the AUG3 pseudo-series, in AUG3_G/AUG3_S row order.
The pseudo-series are regenerated exactly as build_aug3.py made them (rng seed 1, three cuts per qualifying
series, boundary moved right), then whitened against their own extended history.
python build_white_aug.py <shard> <n> | merge <n> | check"""
import sys, time, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_white import white_channels, NCH as NW
from build_sr import sr_channels, NCH as NS
NCH = NW + NS; OUT = "AUG3_WHITE111"; PARTS = "aug3_white_parts"

def pseudo_series():
    """Yield (group, h2, o2, labels) exactly as build_aug3.py generated them."""
    x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    yl = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
    rng = np.random.default_rng(1)
    for sid, part in x.groupby(level="id"):
        hist = part.loc[part.period == 1, "value"].to_numpy(); online = part.loc[part.period == 2, "value"].to_numpy()
        if len(online) < 60: continue
        lab = yl.loc[sid, "target"].to_numpy(); tau = int(lab.argmax()) if lab.max() > 0 else None
        limit = tau if tau is not None else len(online)
        if limit < 30: continue
        hi = max(11, min(limit, len(online) // 2))
        cuts = sorted({int(c) for c in rng.integers(10, hi, size=3)})
        for j, k in enumerate(cuts):
            h2 = np.concatenate([hist, online[:k]]); o2 = online[k:]
            if len(o2) < 20: continue
            yield 100000 + int(sid) * 10 + j, h2.astype("float64"), o2.astype("float64"), lab[k:]

if __name__ == "__main__":
    if sys.argv[1] == "check":
        AG = np.load("AUG3_G.npy"); AS = np.load("AUG3_S.npy"); AY = np.load("AUG3_Y.npy"); R = np.load("AUG3_RAW2.npy", mmap_mode="r")
        st = np.flatnonzero(np.concatenate([[True], AG[1:] != AG[:-1]])); bd = np.append(st, len(AG)); pos = {int(AG[a]): (a, b) for a, b in zip(st, bd[1:])}
        n = bad = 0
        for grp, h2, o2, lab in pseudo_series():
            a, b = pos[grp]; z = (o2 - h2.mean()) / (h2.std() + 1e-12)
            ok = (b - a == len(o2)) and np.array_equal(AY[a:b], lab) and np.abs(np.clip(z, -20, 20) - np.asarray(R[a:b, 0])).max() < 1e-3
            bad += not ok; n += 1
            if n >= 3000: break
        print(f"alignment check: {n} pseudo-series, {bad} mismatches; groups in AUG3 {len(st)}")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); AG = np.load("AUG3_G.npy"); AS = np.load("AUG3_S.npy")
        st = np.flatnonzero(np.concatenate([[True], AG[1:] != AG[:-1]])); bd = np.append(st, len(AG)); pos = {int(AG[a]): (a, b) for a, b in zip(st, bd[1:])}
        out = np.lib.format.open_memmap(f"{OUT}.npy", mode="w+", dtype="float32", shape=(len(AG), NCH)); filled = 0
        for i in range(n):
            p = np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True)
            for grp, arr in zip(p["groups"], p["arrs"]):
                a, b = pos[int(grp)]; out[a:b] = arr[AS[a:b]]; filled += b - a
        out.flush(); print(f"{OUT}: {out.shape}, filled {filled} of {len(AG)}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time(); groups, arrs = [], []
    for grp, h2, o2, lab in pseudo_series():
        sid = (grp - 100000) // 10
        if sid % n_shards != shard: continue
        arrs.append(np.hstack([white_channels(h2, o2), sr_channels(h2, o2)]).astype("float32")); groups.append(grp)
        if len(groups) % 500 == 0: print(f"shard {shard}: {len(groups)} pseudo-series, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", groups=np.array(groups), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(groups)} pseudo-series", flush=True)
