"""Final bagged rankers: each trained on four folds of five (fold 0 included)."""
import sys, time, os
sys.path.insert(0, "repo/src")
import numpy as np, joblib
import lightgbm as lgb
from structural_break.combiners import split_by_series

drop = int(sys.argv[1])
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
tr = assignment != drop
Xtr, ytr, str_ = X[tr], y[tr], s[tr]
order = np.argsort(str_, kind="stable")
Xtr, ytr, str_ = Xtr[order], ytr[order], str_[order]
_, sizes = np.unique(str_, return_counts=True)
r = lgb.LGBMRanker(
    objective="lambdarank", learning_rate=0.03, num_leaves=31,
    min_child_samples=500, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=10.0, n_estimators=600,
    random_state=0, n_jobs=6, deterministic=True, force_row_wise=True,
    verbose=-1, lambdarank_truncation_level=2000, label_gain=[0, 1])
r.fit(Xtr, ytr, group=sizes)
os.makedirs("resources039", exist_ok=True)
joblib.dump(r, f"resources039/ranker_drop{drop}.joblib")
print(f"ранкер без фолда {drop} готов {time.time()-t0:.0f}s", flush=True)
