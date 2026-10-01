"""117: rank battery — the same trick that gave +0.004, but in non-parametric form.

The mass battery describes a point in sigmas of the history. Here — by its place in the history:
u = fraction of historical points below. Under the null hypothesis u is uniform on [0,1],
so deviations can be read without assumptions about the distribution shape.
Representations: u-0.5, |u-0.5|, sign, normal quantile of u (inverse of Phi),
and the "extremeness" min(u,1-u). Same windows (10..500), comparisons: mean against 0,
variance against uniform, fraction in the top/bottom deciles. ~66 channels.
python build_rankbat.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
from scipy.special import ndtri
WINS = (10, 25, 50, 100, 250, 500); NREP = 5
OUT = os.environ.get("RB_OUT", "RANKBAT"); PARTS = f"{OUT.lower()}_parts"
NCH = NREP * len(WINS) * 2 + len(WINS) * 2

def rank_channels(hist, online):
    h = np.sort(np.asarray(hist, float)); nh = len(h)
    n = len(online); out = np.empty((n, NCH), dtype="float32")
    W = np.array(WINS, float); a = 1.0 / W
    s1 = np.zeros((NREP, len(WINS))); s2 = np.zeros((NREP, len(WINS))); dec = np.zeros((2, len(WINS)))
    # reference values under uniform u
    hm = np.array([0.0, 0.25, 0.0, 0.0, 0.25])
    hv = np.array([1/12, 1/48, 1.0, 1.0, 1/48]) + 1e-9
    for t, x in enumerate(online):
        u = (np.searchsorted(h, x) + 0.5) / (nh + 1.0)
        u = min(max(u, 1e-6), 1 - 1e-6)
        vals = np.array([u - 0.5, abs(u - 0.5), np.sign(u - 0.5), float(ndtri(u)), min(u, 1 - u)])
        s1 = (1 - a) * s1 + a * vals[:, None]
        s2 = (1 - a) * s2 + a * (vals * vals)[:, None]
        var = np.maximum(s2 - s1 * s1, 1e-9)
        out[t, :NREP * len(WINS)] = ((s1 - hm[:, None]) / np.sqrt(hv[:, None] / W[None, :])).ravel()
        out[t, NREP * len(WINS):NREP * len(WINS) * 2] = np.log(var / hv[:, None]).ravel()
        d = np.array([float(u > 0.9), float(u < 0.1)])
        dec = (1 - a[None, :]) * dec + a[None, :] * d[:, None]
        out[t, NREP * len(WINS) * 2:] = (dec - 0.1).ravel()
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20)

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0); hist = rng.normal(0, 1, 2000)
        online = np.concatenate([rng.normal(0, 1, 200), rng.standard_t(3, 200) * 0.7])
        t0 = time.time(); o = rank_channels(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print(f"channels {o.shape[1]}, {dt:.3f} ms/step; shape change: variance-window100 {o[150:200, NREP*len(WINS)+3].mean():+.2f} -> {o[300:400, NREP*len(WINS)+3].mean():+.2f}, "
              f"extremeness {o[150:200,-2]. mean():+.3f} -> {o[300:400,-2].mean():+.3f}")
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
    for sid, part in x.groupby(level="id"):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(rank_channels(hist, online))
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(sids)} series", flush=True)
