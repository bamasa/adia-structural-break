"""141 (140 rebuilt): window novelty against the EMPIRICAL distribution of history windows — a different reference point.

All members compare the window with the history's mean/variance. If the history itself is heterogeneous
(regimes, heteroscedasticity), the mean is a poor zero: a window typical for the history looks
anomalous, while a break into a "calm" regime does not. Here the window is described by a vector of summaries
(mean, sd, skew, kurt, ac1, q05, q95) and compared with ALL history windows of the same length:
distance to the nearest, median distance, and the rank of the current distance among the
"history vs history" distances (empirical p-value). 4 windows x 3 = 12 channels + 4 channels
of p-values for individual summaries (sd and ac1) = 20. O(history windows x 7) per step.
python build_knn.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
WINS = (25, 50, 100, 250); NCH = len(WINS) * 5
OUT = os.environ.get("KNN_OUT", "KNN20B"); PARTS = f"{OUT.lower()}_parts"

def summ(w):
    m = w.mean(); sd = w.std() + 1e-9; z = (w - m) / sd
    return np.array([m, np.log(sd), (z ** 3).mean(), (z ** 4).mean() - 3.0, np.corrcoef(w[:-1], w[1:])[0, 1] if len(w) > 4 else 0.0, np.quantile(w, 0.05), np.quantile(w, 0.95)])

def knn_channels(hist, online):
    h = np.asarray(hist, float); mu, sd = h.mean(), h.std() + 1e-12; zh = (h - mu) / sd; zo = (np.asarray(online, float) - mu) / sd
    n = len(zo); out = np.zeros((n, NCH), dtype="float32")
    full = np.concatenate([zh[-max(WINS):], zo]); off = len(zh[-max(WINS):])
    for i, W in enumerate(WINS):
        step = max(W // 2, 1)
        H = np.array([summ(zh[j:j + W]) for j in range(0, len(zh) - W + 1, step)])            # history windows
        scale = H.std(0) + 1e-6; Hn = H / scale
        # "history vs history" distances (excluding itself) for the reference
        D = np.sqrt(((Hn[:, None, :] - Hn[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
        d_nn_h = D.min(1); d_med_h = np.nanmedian(np.where(np.isinf(D), np.nan, D), 1)  # 141: nanmedian — the median over a row with one NaN was NaN, and the channel constant
        nn_ref = np.sort(d_nn_h); med_ref = np.sort(np.nan_to_num(d_med_h, nan=np.nanmedian(d_med_h)))
        sd_ref = np.sort(np.abs(Hn[:, 1] - np.median(Hn[:, 1]))); ac_ref = np.sort(np.abs(Hn[:, 4] - np.median(Hn[:, 4])))
        for t in range(n):
            if t + 1 < 5: continue
            w = full[off + t + 1 - W: off + t + 1] if t + 1 >= W else full[off: off + t + 1]
            if len(w) < 5: continue
            v = summ(w) / scale
            d = np.sqrt(((Hn - v) ** 2).sum(1)); dnn = d.min(); dmed = np.median(d)
            p_nn = np.searchsorted(nn_ref, dnn) / len(nn_ref); p_med = np.searchsorted(med_ref, dmed) / len(med_ref)
            p_sd = np.searchsorted(sd_ref, abs(v[1] - np.median(Hn[:, 1]))) / len(sd_ref); p_ac = np.searchsorted(ac_ref, abs(v[4] - np.median(Hn[:, 4]))) / len(ac_ref)
            out[t, i * 5:(i + 1) * 5] = (np.log(dnn / (np.median(nn_ref) + 1e-9) + 1e-9), p_nn - 0.5, p_med - 0.5, p_sd - 0.5, p_ac - 0.5)
    return np.clip(np.nan_to_num(out), -20, 20)

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0)
        # heterogeneous history: alternating regimes σ=0.6 and σ=1.4; the break goes into a third regime σ=1.0 with AR 0.5
        hist = np.concatenate([rng.normal(0, 0.6 if (k // 200) % 2 == 0 else 1.4, 200) for k in range(0, 2000, 200)])
        def ar(phi, n): 
            x = np.zeros(n); e = rng.normal(0, 1, n)
            for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
            return x
        online = np.concatenate([rng.normal(0, 0.6, 200), ar(0.5, 200)])
        t0 = time.time(); o = knn_channels(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print(f"{o.shape[1]} channels, {dt:.3f} ms/step; heterogeneous history, break into a new regime: p_nn/window100 {o[150:200,11].mean():+.2f} -> {o[300:400,11].mean():+.2f}, p_ac {o[150:200,14].mean():+.2f} -> {o[300:400,14].mean():+.2f}")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
        g = np.load("G40.npy"); s = np.load("S40.npy")
        st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
        out = np.empty((len(g), NCH), dtype="float32"); block = {}
        for p in parts:
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a_, b_ in zip(st, bd[1:]): out[a_:b_] = block[int(g[a_])][s[a_:b_]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for i, (sid, part) in enumerate(x.groupby(level="id")):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(knn_channels(hist, online))
        if (i + 1) % 200 == 0: print(f"shard {shard}: {i+1} series, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(sids)} series", flush=True)
