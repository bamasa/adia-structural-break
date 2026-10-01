"""147: Shiryaev-Roberts statistics on the whitened stream — the accumulated posterior odds of a
change at some tau <= t (uniform tau prior) for the break types the forensics found: innovation
variance (increase and decrease), mean, and lag-1 dependence, each over a grid of alternatives
with a mixture over the grid. log R_t = L_t + logcumsumexp_{tau<=t}(-L_{tau-1}), L = cumsum of
per-point log likelihood ratios; fully vectorised per series. 22 channels.
python build_sr.py <shard> <n> | merge <n>"""
import sys, time, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_white import fit_history, normal_scores_online
VAR_UP = (1.25, 1.5, 2.0, 3.0, 5.0); VAR_DOWN = (0.7, 0.5); MEANS = (0.3, 0.6, 1.0); PHIS = (0.2, 0.4)
NCH = len(VAR_UP) + len(VAR_DOWN) + 2 * len(MEANS) + 2 * len(PHIS) + 4
OUT = os.environ.get("SR_OUT", "SR22"); PARTS = f"{OUT.lower()}_parts"

def log_sr(ell):
    """log Shiryaev-Roberts statistic from per-point log likelihood ratios."""
    L = np.cumsum(ell); Lprev = np.concatenate([[0.0], L[:-1]])
    return L + np.logaddexp.accumulate(-Lprev)

def sr_channels(hist, online):
    h = np.asarray(hist, float); mu, sd = h.mean(), h.std() + 1e-12; zh = (h - mu) / sd; zo = (np.asarray(online, float) - mu) / sd
    fit = fit_history(zh); _, n, _ = normal_scores_online(fit, zh, zo)      # the unconditional normal scores
    prev = np.concatenate([[fit["nhu"][-1]], n[:-1]])
    T = len(n); out = np.zeros((T, NCH), dtype="float32"); c = 0; fam = {}
    for v in VAR_UP + VAR_DOWN:
        ell = -0.5 * np.log(v) - 0.5 * n ** 2 * (1.0 / v - 1.0); out[:, c] = log_sr(ell); c += 1
    fam["var_up"] = out[:, :len(VAR_UP)].copy(); fam["var_down"] = out[:, len(VAR_UP):c].copy()
    for d in MEANS:
        for sign in (1.0, -1.0):
            ell = sign * d * n - 0.5 * d * d; out[:, c] = log_sr(ell); c += 1
    fam["mean"] = out[:, c - 2 * len(MEANS):c].copy()
    for phi in PHIS:
        for sign in (1.0, -1.0):
            p = sign * phi; ell = -0.5 * np.log(1 - p * p) - 0.5 * ((n - p * prev) ** 2 / (1 - p * p) - n ** 2); out[:, c] = log_sr(ell); c += 1
    fam["dep"] = out[:, c - 2 * len(PHIS):c].copy()
    for key in ("var_up", "var_down", "mean", "dep"):
        F = fam[key]; out[:, c] = np.logaddexp.reduce(F, axis=1) - np.log(F.shape[1]); c += 1   # equal-weight mixture over the grid
    assert c == NCH
    return np.clip(np.nan_to_num(out), -50, 200)

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
        sids.append(int(sid)); arrs.append(sr_channels(hist, online))
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(sids)} series", flush=True)
