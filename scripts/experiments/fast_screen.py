"""Fast screening: half the series, 150 trees, max_bin 63. Calibration against known results."""
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
tr_full, va = assignment != 0, assignment == 0
rng = np.random.default_rng(7)
train_sids = np.unique(g[tr_full])
half = rng.choice(train_sids, size=len(train_sids) // 2, replace=False)
tr = tr_full & np.isin(g, half)
print(f"training rows: {tr.sum()} (was {tr_full.sum()})", flush=True)
Xtr, ytr, str_ = X[tr], y[tr], s[tr]
order = np.argsort(str_, kind="stable")
Xtr, ytr, str_ = Xtr[order], ytr[order], str_[order]
_, sizes = np.unique(str_, return_counts=True)
yf, sf = y[va], s[va]

def screen_rank(trunc):
    r = lgb.LGBMRanker(
        objective="lambdarank", learning_rate=0.05, num_leaves=31,
        min_child_samples=250, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=10.0, n_estimators=150,
        random_state=0, n_jobs=8, deterministic=True, force_row_wise=True,
        verbose=-1, lambdarank_truncation_level=trunc, label_gain=[0, 1],
        max_bin=63)
    r.fit(Xtr, ytr, group=sizes)
    return ts_auc(r.predict(X[va]), yf, sf)

def screen_clf():
    m = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=250,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0,
        n_estimators=150, random_state=0, n_jobs=8, deterministic=True,
        force_row_wise=True, verbose=-1, max_bin=63)
    m.fit(X[tr], y[tr], sample_weight=step_weights(s[tr]))
    return ts_auc(m.predict_proba(X[va])[:, 1], yf, sf)

print(f"screen ranker t2000:  {screen_rank(2000):.4f}  (full: 0.6006)  [{time.time()-t0:.0f}s]", flush=True)
print(f"screen classifier:    {screen_clf():.4f}  (full: 0.5952)  [{time.time()-t0:.0f}s]", flush=True)
print(f"screen ranker t500:   {screen_rank(500):.4f}  (full: 0.5919)  [{time.time()-t0:.0f}s]", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
