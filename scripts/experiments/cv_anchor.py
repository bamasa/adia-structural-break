"""031 (fold 0): anchor view on top of 186 — ranker and blend."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import lightgbm as lgb
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("A50.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
tr, va = assignment != 0, assignment == 0
Xtr, ytr, str_ = X[tr], y[tr], s[tr]
order = np.argsort(str_, kind="stable")
Xtr, ytr, str_ = Xtr[order], ytr[order], str_[order]
_, sizes = np.unique(str_, return_counts=True)
yf, sf = y[va], s[va]
r = lgb.LGBMRanker(
    objective="lambdarank", learning_rate=0.03, num_leaves=31,
    min_child_samples=500, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=10.0, n_estimators=600,
    random_state=0, n_jobs=8, deterministic=True, force_row_wise=True,
    verbose=-1, lambdarank_truncation_level=2000, label_gain=[0, 1])
r.fit(Xtr, ytr, group=sizes)
sc = r.predict(X[va])
sig = 1.0 / (1.0 + np.exp(-sc))
clf = np.load("oof_cfg5.npy")[va].astype("float64")
solo = ts_auc(sc, yf, sf)
mix = ts_auc(0.6 * sig + 0.4 * clf, yf, sf)
print(f"236 channels (186+anchor), ranker-600: alone {solo:.4f}, blend {mix:.4f} "
      f"(references 0.6016 / 0.6045)", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
