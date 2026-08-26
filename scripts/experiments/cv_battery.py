"""018 verdict: the battery alone, and the battery over the 104 base."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
from structural_break.combiners import Boosted, split_by_series, step_weights, ts_auc

t0 = time.time()
B = np.load("B40.npy")
X104 = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
                  np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
                  np.load("E4.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)

def cv(mat, tag):
    names = [f"c{i}" for i in range(mat.shape[1])]
    scores = []
    for fold in range(5):
        tr, va = assignment != fold, assignment == fold
        m = Boosted.fit(mat[tr], y[tr], names, step_weights(s[tr]))
        scores.append(ts_auc(m.predict(mat[va]), y[va], s[va]))
        print(f"  фолд {fold}: {scores[-1]:.4f}", flush=True)
    print(f"{tag}: {np.mean(scores):.4f}  " + " ".join(f"{v:.4f}" for v in scores)
          + f"   [{time.time()-t0:.0f}s]", flush=True)

cv(B, "40 (батарея СОЛО)")
cv(np.hstack([X104, B]), "144 (104 + батарея)")
print(f"всего {time.time()-t0:.0f}s", flush=True)
