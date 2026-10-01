"""097: distribution divergences history vs prefix — the feature block of the 2025 winners (Alphabot).

Bins — 32 history quantiles. Channels (8): JS, Hellinger, Wasserstein-1 (in history sd),
entropy difference — for the whole prefix and for the window of the last 100 points. O(bins) per step.
python build_divergence.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
NB = 32; WIN = 100
OUT = os.environ.get("DIV_OUT", "DIV8"); PARTS = f"{OUT.lower()}_parts"

def divergences(hist, online):
    h = np.asarray(hist, float); edges = np.quantile(h, np.linspace(0, 1, NB + 1)[1:-1])
    ph = np.bincount(np.searchsorted(edges, h), minlength=NB) / len(h) + 1e-9; ph /= ph.sum()
    sd = h.std() + 1e-12; centers = np.array([np.median(h[np.searchsorted(edges, h) == b]) if (np.searchsorted(edges, h) == b).any() else 0.0 for b in range(NB)]) / sd
    Hh = -(ph * np.log(ph)).sum(); ch = np.cumsum(ph)
    def divs(counts, n):
        p = counts / n + 1e-9; p /= p.sum(); m = 0.5 * (p + ph)
        js = 0.5 * (p * np.log(p / m)).sum() + 0.5 * (ph * np.log(ph / m)).sum()
        hel = np.sqrt(max(0.0, 1.0 - (np.sqrt(p * ph)).sum()))
        w1 = np.abs(np.cumsum(p) - ch)[:-1] @ np.diff(centers)
        ent = -(p * np.log(p)).sum() - Hh
        return js, hel, w1, ent
    n = len(online); out = np.empty((n, 8), dtype="float32")
    cp = np.zeros(NB); cw = np.zeros(NB); win = []
    for t, x in enumerate(online):
        b = int(np.searchsorted(edges, x)); cp[b] += 1; cw[b] += 1; win.append(b)
        if len(win) > WIN: cw[win.pop(0)] -= 1
        out[t, :4] = divs(cp, t + 1); out[t, 4:] = divs(cw, len(win))
    return out

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0); hist = rng.normal(0, 1, 2000)
        online = np.concatenate([rng.normal(0, 1, 400), rng.standard_t(3, 400) * 0.6])  # shape change, roughly the same variance
        t0 = time.time(); o = divergences(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print(f"test: window JS before {o[300:400,4].mean():.4f} after {o[450:550,4].mean():.4f}; Hellinger {o[300:400,5].mean():.3f}->{o[450:550,5].mean():.3f}; W1 {o[300:400,6].mean():.3f}->{o[450:550,6].mean():.3f}; entropy {o[300:400,7].mean():+.3f}->{o[450:550,7].mean():+.3f}; {dt:.3f} ms/step")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
        g = np.load("G40.npy"); s = np.load("S40.npy")
        starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bounds = np.append(starts, len(g))
        out = np.empty((len(g), 8), dtype="float32"); block = {}
        for p in parts:
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a_, b_ in zip(starts, bounds[1:]): out[a_:b_] = block[int(g[a_])][s[a_:b_]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for i, (sid, part) in enumerate(x.groupby(level="id")):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(divergences(hist, online))
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(sids)} series", flush=True)
