"""149: per-series null calibration of the whitened battery — the second pillar of the forum recipe.
The history's own normal scores are run through the same battery as a pseudo-online stream (the last
L = min(1000, half) points against the earlier part), giving each of the 76 full-battery channels a
series-specific null mean and sd; the online channels are reported as z-scores against that null.
python build_whitez.py <shard> <n> | merge <n>"""
import sys, time, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_white import fit_history, normal_scores_online, _battery, NFULL
NCH = NFULL; OUT = "WHITEZ76"; PARTS = f"{OUT.lower()}_parts"

def whitez_channels(hist, online):
    h = np.asarray(hist, float); mu, sd = h.mean(), h.std() + 1e-12; zh = (h - mu) / sd; zo = (np.asarray(online, float) - mu) / sd
    fit = fit_history(zh); _, n_u, _ = normal_scores_online(fit, zh, zo); nhu = fit["nhu"]
    L = min(1000, len(nhu) // 2); ref, pseudo = nhu[:-L], np.clip(nhu[-L:], -4.5, 4.5)
    null = np.zeros((L, NFULL), dtype="float32"); _battery(pseudo, ref, null, 0)
    null = np.clip(np.nan_to_num(null), -60, 60); m, s = null.mean(0), null.std(0) + 1e-6
    out = np.zeros((len(zo), NFULL), dtype="float32"); _battery(n_u, nhu, out, 0)
    out = np.clip(np.nan_to_num(out), -60, 60)
    return np.clip((out - m) / s, -30, 30).astype("float32")

if __name__ == "__main__":
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); g = np.load("G40.npy"); s = np.load("S40.npy")
        st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
        out = np.empty((len(g), NCH), dtype="float32"); block = {}
        for i in range(n):
            p = np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True)
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a_, b_ in zip(st, bd[1:]): out[a_:b_] = block[int(g[a_])][s[a_:b_]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for i, (sid, part) in enumerate(x.groupby(level="id")):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(whitez_channels(hist, online))
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"шард {shard} готов {time.time()-t0:.0f}s: {len(sids)} рядов", flush=True)
