"""The tail of the sweep: magnitude forecaster and the everything-at-once combo."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
from structural_break.combiners import Boosted, split_by_series, step_weights, ts_auc

t0 = time.time()
X100 = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
                  np.load("N9.npy").astype("float32"), np.load("X50a.npy")])
E4 = np.load("E4.npy"); E8 = np.load("E8.npy"); D50 = np.load("X50d.npy"); M4 = np.load("M4.npy")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)

def cv(mat, tag):
    names = [f"c{i}" for i in range(mat.shape[1])]
    scores = []
    for fold in range(5):
        tr, va = assignment != fold, assignment == fold
        m = Boosted.fit(mat[tr], y[tr], names, step_weights(s[tr]))
        scores.append(ts_auc(m.predict(mat[va]), y[va], s[va]))
    print(f"{tag}: {np.mean(scores):.4f}  " + " ".join(f"{v:.4f}" for v in scores)
          + f"   [{time.time()-t0:.0f}s]", flush=True)

print("соло амплитудный, пик:", f"{ts_auc(M4[:, 2], y, s):.4f}", flush=True)
cv(np.hstack([X100, E4, M4]), "108 (104 + амплитудный)")
cv(np.hstack([X100, E4, E8, D50, M4]), "170 (ПОЛНАЯ комбинация всего)")
print(f"всего {time.time()-t0:.0f}s", flush=True)
