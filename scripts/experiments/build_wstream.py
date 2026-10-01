"""158: the whitened innovation stream itself as a network input — five channels per step: the unconditional
normal score, the conditional one, its square minus one, the lag-1 product, the log conditional-scale process.
For the originals (WSTREAM5.npy) and the AUG3 pseudo-series (AUG3_WSTREAM5.npy). python build_wstream.py <shard> <n> [aug] | merge <n> [aug]"""
import sys, time, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_white import fit_history, normal_scores_online
NCH = 5
def stream_channels(hist, online):
    h = np.asarray(hist, float); mu, sd = h.mean(), h.std() + 1e-12; zh = (h - mu) / sd; zo = (np.asarray(online, float) - mu) / sd
    fit = fit_history(zh); nc, nu, lz = normal_scores_online(fit, zh, zo)
    prev = np.concatenate([[fit["nhu"][-1]], nu[:-1]])
    return np.clip(np.nan_to_num(np.column_stack([nu, nc, nu ** 2 - 1, nu * prev, lz])), -30, 30).astype("float32")
if __name__ == "__main__":
    aug = "aug" in sys.argv; OUT = "AUG3_WSTREAM5" if aug else "WSTREAM5"; PARTS = OUT.lower() + "_parts"
    if sys.argv[1] == "merge":
        n = int(sys.argv[2])
        if aug:
            AG = np.load("AUG3_G.npy"); AS = np.load("AUG3_S.npy")
            st = np.flatnonzero(np.concatenate([[True], AG[1:] != AG[:-1]])); bd = np.append(st, len(AG)); pos = {int(AG[a]): (a, b) for a, b in zip(st, bd[1:])}
            out = np.lib.format.open_memmap(f"{OUT}.npy", mode="w+", dtype="float32", shape=(len(AG), NCH))
            for i in range(n):
                p = np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True)
                for grp, arr in zip(p["groups"], p["arrs"]): a, b = pos[int(grp)]; out[a:b] = arr[AS[a:b]]
            out.flush()
        else:
            g = np.load("G40.npy"); s = np.load("S40.npy"); st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
            out = np.empty((len(g), NCH), dtype="float32"); block = {}
            for i in range(n):
                p = np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True)
                for sid, arr in zip(p["groups"], p["arrs"]): block[int(sid)] = arr
            for a_, b_ in zip(st, bd[1:]): out[a_:b_] = block[int(g[a_])][s[a_:b_]]
            np.save(f"{OUT}.npy", out)
        print(f"{OUT}: {out.shape}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time(); groups, arrs = [], []
    if aug:
        from build_white_aug import pseudo_series
        for grp, h2, o2, lab in pseudo_series():
            if ((grp - 100000) // 10) % n_shards != shard: continue
            groups.append(grp); arrs.append(stream_channels(h2, o2))
    else:
        import pandas as pd
        x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet"); ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]
        for sid, part in x.groupby(level="id"):
            groups.append(int(sid)); arrs.append(stream_channels(part.loc[part.period == 1, "value"].to_numpy("float64"), part.loc[part.period == 2, "value"].to_numpy("float64")))
    os.makedirs(PARTS, exist_ok=True); np.savez(f"{PARTS}/part_{shard}.npz", groups=np.array(groups), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(groups)}", flush=True)
