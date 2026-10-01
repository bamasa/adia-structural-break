"""025 (fold 0): lambdarank truncation depth — 500 / 2000 (reference) / 8000."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import lightgbm as lgb
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
tr, va = assignment != 0, assignment == 0
Xtr, ytr, str_ = X[tr], y[tr], s[tr]
order = np.argsort(str_, kind="stable")
Xtr, ytr, str_ = Xtr[order], ytr[order], str_[order]
_, sizes = np.unique(str_, return_counts=True)
for trunc in (500, 8000):
    r = lgb.LGBMRanker(
        objective="lambdarank", learning_rate=0.05, num_leaves=31,
        min_child_samples=500, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=10.0, n_estimators=300,
        random_state=0, n_jobs=8, deterministic=True, force_row_wise=True,
        verbose=-1, lambdarank_truncation_level=trunc, label_gain=[0, 1])
    r.fit(Xtr, ytr, group=sizes)
    print(f"truncation {trunc}: fold 0 {ts_auc(r.predict(X[va]), y[va], s[va]):.4f}  "
          f"(reference 2000: 0.6006)  [{time.time()-t0:.0f}s]", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
