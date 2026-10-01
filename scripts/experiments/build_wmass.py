"""122: mass battery on the WHITENED series (innovations).

The same construction that gave +0.0039 in the cloud, but with a different input: the historical
AR(1) dependence is removed from the series, (x_t - rho*x_{t-1}) / sqrt(1-rho^2). The channels
diverge from the mass ones wherever the dependence structure changes — our weakest
zone per 109. Expectation: independence from a different input, not from a different model.

Original description of 114: mass battery — many simple statistics over many windows and representations.

Not a new "smart" modality (four of those in a row gave zero), but volume: the
2nd-place solution of 2025 had 2408 features versus our 200. We build the same cheaply:
6 representations of the series x 6 sliding windows x several comparisons with the history.
All via sliding sums, O(representations x windows) per step.

Representations: z, |z|, z^2, increment, cumulative-sum deviation, sign.
Windows: 10, 25, 50, 100, 250, 500.
For each: (mean_W - mu_hist)/se, log(var_W/var_hist) -> 72 channels.
Plus the share of exceedances of historical quantiles q75/q95/q99 over the windows -> 18.
Total 90. python build_mass.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
WINS = (10, 25, 50, 100, 250, 500); NREP = 6
OUT = os.environ.get("MASS_OUT", "WMASS90"); PARTS = f"{OUT.lower()}_parts"
NCH = NREP * len(WINS) * 2 + len(WINS) * 3

def mass_channels(hist, online):
    h0 = np.asarray(hist, float); o0 = np.asarray(online, float)
    rho = float(np.clip(np.corrcoef(h0[:-1], h0[1:])[0, 1], -0.95, 0.95)); sc = np.sqrt(max(1 - rho * rho, 1e-6))
    hist = (h0[1:] - rho * h0[:-1]) / sc                                   # history innovations
    prev = np.concatenate([h0[-1:], o0[:-1]]); online = (o0 - rho * prev) / sc   # online-part innovations
    h = np.asarray(hist, float); mu_h, sd_h = h.mean(), h.std() + 1e-12
    zh = (h - mu_h) / sd_h
    reps_h = [zh, np.abs(zh), zh * zh, np.diff(zh, prepend=zh[0]), np.zeros_like(zh), np.sign(zh)]
    hm = np.array([r.mean() for r in reps_h]); hv = np.array([r.var() + 1e-9 for r in reps_h])
    q = np.quantile(np.abs(zh), [0.75, 0.95, 0.99])
    n = len(online); out = np.empty((n, NCH), dtype="float32")
    W = np.array(WINS, float)
    s1 = np.zeros((NREP, len(WINS))); s2 = np.zeros((NREP, len(WINS)))   # sliding EWMA sums
    qs = np.zeros((3, len(WINS)))
    prev = zh[-1]; cum = 0.0
    for t, x in enumerate(online):
        z = (x - mu_h) / sd_h
        cum += z
        vals = np.array([z, abs(z), z * z, z - prev, cum / np.sqrt(t + 1), np.sign(z)])
        a = 1.0 / W                                   # decay = 1/window
        s1 = (1 - a) * s1 + a * vals[:, None]
        s2 = (1 - a) * s2 + a * (vals * vals)[:, None]
        var = np.maximum(s2 - s1 * s1, 1e-9)
        se = np.sqrt(hv[:, None] / W[None, :])
        out[t, :NREP * len(WINS)] = ((s1 - hm[:, None]) / se).ravel()
        out[t, NREP * len(WINS):NREP * len(WINS) * 2] = np.log(var / hv[:, None]).ravel()
        exc = (np.abs(z) > q[:, None]).astype(float)
        qs = (1 - a[None, :]) * qs + a[None, :] * exc
        out[t, NREP * len(WINS) * 2:] = (qs - np.array([0.25, 0.05, 0.01])[:, None]).ravel()
        prev = z
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20)

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0); hist = rng.normal(0, 1, 2000)
        online = np.concatenate([rng.normal(0, 1, 300), rng.normal(0.3, 1.2, 300)])
        t0 = time.time(); o = mass_channels(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print(f"{o.shape[1]} channels, {dt:.3f} ms/step; before/after the break: "
              f"mean-window100 {o[250:300,3].mean():+.2f}->{o[450:600,3].mean():+.2f}, "
              f"variance-window100 {o[250:300,NREP*len(WINS)+3].mean():+.2f}->{o[450:600,NREP*len(WINS)+3].mean():+.2f}")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
        g = np.load("G40.npy"); s = np.load("S40.npy")
        st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
        out = np.empty((len(g), NCH), dtype="float32"); block = {}
        for p in parts:
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a, b in zip(st, bd[1:]): out[a:b] = block[int(g[a])][s[a:b]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for i, (sid, part) in enumerate(x.groupby(level="id")):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(mass_channels(hist, online))
        if (i + 1) % 300 == 0: print(f"shard {shard}: {i+1} series, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(sids)} series", flush=True)
