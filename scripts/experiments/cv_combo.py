"""The combination sweep: channel sets, combiners, asymmetric post-processing."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
from structural_break.combiners import Boosted, Weighted, split_by_series, step_weights, ts_auc

t0 = time.time()
X100 = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
                  np.load("N9.npy").astype("float32"), np.load("X50a.npy")])
E4 = np.load("E4.npy"); E8 = np.load("E8.npy"); D50 = np.load("X50d.npy")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)

def cv_boosted(mat, tag, save_oof=False):
    names = [f"c{i}" for i in range(mat.shape[1])]
    scores, oof = [], np.zeros(len(y), dtype="float32")
    for fold in range(5):
        tr, va = assignment != fold, assignment == fold
        m = Boosted.fit(mat[tr], y[tr], names, step_weights(s[tr]))
        pred = m.predict(mat[va])
        if save_oof:
            oof[va] = pred
        scores.append(ts_auc(pred, y[va], s[va]))
    print(f"{tag}: {np.mean(scores):.4f}  " + " ".join(f"{v:.4f}" for v in scores)
          + f"   [{time.time()-t0:.0f}s]", flush=True)
    return np.mean(scores), oof

# 1) standalone quality of the new blocks
print("D-view alone, max over 50:", f"{ts_auc(D50.max(axis=1), y, s):.4f}", flush=True)
print("E8 alone, horizon-5 peak:  ", f"{ts_auc(E8[:, 5], y, s):.4f}", flush=True)

# 2) channel sets
best_mean, best_mat, best_tag, best_oof = 0.5779, None, "104 (baseline 013)", None
for tag, mat in [
    ("108 (100 + E8 forecaster)", np.hstack([X100, E8])),
    ("154 (104 + D-view)", np.hstack([X100, E4, D50])),
    ("162 (100 + E8 + D-view)", np.hstack([X100, E8, D50])),
]:
    mean, oof = cv_boosted(mat, tag, save_oof=True)
    if mean > best_mean:
        best_mean, best_mat, best_tag, best_oof = mean, mat, tag, oof
print(f"best set: {best_tag} ({best_mean:.4f})", flush=True)

if best_mat is None:
    print("the new sets did not beat 104 — post-processing is measured on 104", flush=True)
    _, best_oof = cv_boosted(np.hstack([X100, E4]), "104 (recomputed for OOF)", save_oof=True)
    best_mat = np.hstack([X100, E4])

# 3) weighted sum and max on the best set (one fold for speed, then the full run if close)
names = [f"c{i}" for i in range(best_mat.shape[1])]
tr, va = assignment != 0, assignment == 0
w_model = Weighted.fit(best_mat[tr], y[tr], names, step_weights(s[tr]))
print(f"weighted sum, fold 0: {ts_auc(w_model.predict(best_mat[va]), y[va], s[va]):.4f}", flush=True)
print(f"channel max, fold 0: {ts_auc(best_mat[va].max(axis=1), y[va], s[va]):.4f}", flush=True)

# 4) asymmetric post-processing of the best model's OOF
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
fast_pred = E8[:, 0]  # fast EWMA of the forecast error: ~1 = the series is predictable

def fold_scores(p):
    return [ts_auc(p[assignment == f], y[assignment == f], s[assignment == f]) for f in range(5)]

base_scores = fold_scores(best_oof)
print(f"post-processing, baseline: {np.mean(base_scores):.4f}", flush=True)

def peak_hold(p, alpha):
    out = np.empty_like(p)
    for a, b in zip(bounds[:-1], bounds[1:]):
        acc = 0.0
        for i in range(a, b):
            acc = max(float(p[i]), alpha * acc)
            out[i] = acc
    return out

def gated(p, a_calm, a_loud, thr):
    out = np.empty_like(p)
    for a, b in zip(bounds[:-1], bounds[1:]):
        acc = 0.0
        for i in range(a, b):
            alpha = a_calm if fast_pred[i] < thr else a_loud
            acc = max(float(p[i]), alpha * acc)
            out[i] = acc
    return out

for alpha in (0.98, 0.995, 0.999):
    sc = fold_scores(peak_hold(best_oof, alpha))
    print(f"peak-hold α={alpha}: {np.mean(sc):.4f}  " + " ".join(f"{v:.4f}" for v in sc), flush=True)
for a_calm, a_loud, thr in [(0.95, 0.999, 1.05), (0.90, 0.999, 1.10), (0.97, 1.0, 1.05)]:
    sc = fold_scores(gated(best_oof, a_calm, a_loud, thr))
    print(f"forecaster gate ({a_calm}/{a_loud}, threshold {thr}): {np.mean(sc):.4f}  "
          + " ".join(f"{v:.4f}" for v in sc), flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
