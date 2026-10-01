"""026 (fold 0): what is actually shippable — blend of classifier and ranker + hold."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
mask = assignment == 0
yf, sf, gf = y[mask], s[mask], g[mask]
clf = np.load("oof_cfg5.npy")[mask].astype("float64")
rnk = np.load("oof_rank.npy")[mask].astype("float64")
sig = 1.0 / (1.0 + np.exp(-rnk))
print(f"classifier: {ts_auc(clf, yf, sf):.4f}", flush=True)
print(f"ranker (sigmoid): {ts_auc(sig, yf, sf):.4f}", flush=True)

starts = np.flatnonzero(np.concatenate([[True], gf[1:] != gf[:-1]]))
bounds = np.append(starts, len(gf))
def hold(p, alpha):
    out = np.empty_like(p)
    for a, b in zip(bounds[:-1], bounds[1:]):
        acc = 0.0
        for i in range(a, b):
            acc = max(float(p[i]), alpha * acc)
            out[i] = acc
    return out

best = (0.0, None)
for w in (0.5, 0.6, 0.7, 0.8, 1.0):
    mix = w * sig + (1 - w) * clf
    v0 = ts_auc(mix, yf, sf)
    line = f"blend w_ranker={w}: {v0:.4f}"
    for alpha in (0.99, 0.995):
        vh = ts_auc(hold(mix, alpha), yf, sf)
        line += f" | +hold {alpha}: {vh:.4f}"
        if vh > best[0]:
            best = (vh, (w, alpha))
    if v0 > best[0]:
        best = (v0, (w, None))
    print(line, flush=True)
print(f"BEST: {best[0]:.4f} at {best[1]}", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
