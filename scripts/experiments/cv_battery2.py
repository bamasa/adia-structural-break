"""020 verdict: 186 (146 + battery v2) vs 146, plus the hold re-check."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
from structural_break.combiners import Boosted, split_by_series, step_weights, ts_auc

t0 = time.time()
X186 = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
                  np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
                  np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
names = [f"c{i}" for i in range(X186.shape[1])]
oof = np.zeros(len(y), dtype="float32")
scores = []
for fold in range(5):
    tr, va = assignment != fold, assignment == fold
    m = Boosted.fit(X186[tr], y[tr], names, step_weights(s[tr]))
    oof[va] = m.predict(X186[va])
    scores.append(ts_auc(oof[va], y[va], s[va]))
    print(f"  фолд {fold}: {scores[-1]:.4f}  [{time.time()-t0:.0f}s]", flush=True)
print(f"186 (146 + батарея v2): {np.mean(scores):.4f}  " + " ".join(f"{v:.4f}" for v in scores), flush=True)
np.save("oof186.npy", oof)

starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
def fold_scores(p):
    return [ts_auc(p[assignment == f], y[assignment == f], s[assignment == f]) for f in range(5)]
def peak_hold(p, alpha):
    out = np.empty_like(p)
    for a, b in zip(bounds[:-1], bounds[1:]):
        acc = 0.0
        for i in range(a, b):
            acc = max(float(p[i]), alpha * acc)
            out[i] = acc
    return out
for alpha in (0.99, 0.995, 0.999):
    sc = fold_scores(peak_hold(oof, alpha))
    print(f"186 + пик-холд α={alpha}: {np.mean(sc):.4f}  " + " ".join(f"{v:.4f}" for v in sc), flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
