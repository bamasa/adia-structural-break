"""116: the raw window as features — the last 32 values and their order statistics.

All our channels (and the mass battery) are summaries. Here the tree sees the piece of the series itself:
the last 32 z-values in arrival order + the same 32 sorted
(the window's order statistics — permutation-invariant, robust to noise).
64 channels, O(1) per step via a ring buffer + sorting 32 elements.
python build_window.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
K = 32; OUT = os.environ.get("WIN_OUT", "WIN64"); PARTS = f"{OUT.lower()}_parts"

def window_channels(hist, online):
    h = np.asarray(hist, float); mu, sd = h.mean(), h.std() + 1e-12
    n = len(online); out = np.zeros((n, 2 * K), dtype="float32")
    buf = np.zeros(K)
    for t, x in enumerate(online):
        buf[:-1] = buf[1:]; buf[-1] = (x - mu) / sd
        m = min(t + 1, K)
        out[t, :K] = buf
        srt = np.sort(buf[-m:]) if m < K else np.sort(buf)
        out[t, K:K + m] = srt
    return np.clip(out, -20, 20)

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0); hist = rng.normal(0, 1, 2000)
        online = np.concatenate([rng.normal(0, 1, 200), rng.normal(0.5, 1, 200)])
        t0 = time.time(); o = window_channels(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print(f"{o.shape[1]} channels, {dt:.3f} ms/step; window median before the break {np.median(o[150:200, K + K//2]):+.2f}, after {np.median(o[300:400, K + K//2]):+.2f}")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
        g = np.load("G40.npy"); s = np.load("S40.npy")
        st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
        out = np.empty((len(g), 2 * K), dtype="float32"); block = {}
        for p in parts:
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a, b in zip(st, bd[1:]): out[a:b] = block[int(g[a])][s[a:b]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for sid, part in x.groupby(level="id"):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(window_channels(hist, online))
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(sids)} series", flush=True)
