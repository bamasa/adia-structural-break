"""021 final: OOF of config 5, peak-hold on top."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import lightgbm as lgb
from structural_break.combiners import split_by_series, step_weights, ts_auc

t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=500,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0,
              n_estimators=300, random_state=0, n_jobs=6, deterministic=True,
              force_row_wise=True, verbose=-1)
oof = np.zeros(len(y), dtype="float32")
for fold in range(5):
    tr, va = assignment != fold, assignment == fold
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(X[tr], y[tr], sample_weight=step_weights(s[tr]))
    oof[va] = m.predict_proba(X[va])[:, 1]
    print(f"  fold {fold} done [{time.time()-t0:.0f}s]", flush=True)
np.save("oof_cfg5.npy", oof)
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
base = fold_scores(oof)
print(f"config5 without hold: {np.mean(base):.4f}  " + " ".join(f"{v:.4f}" for v in base), flush=True)
for alpha in (0.99, 0.995):
    sc = fold_scores(peak_hold(oof, alpha))
    print(f"config5 + hold α={alpha}: {np.mean(sc):.4f}  " + " ".join(f"{v:.4f}" for v in sc), flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
