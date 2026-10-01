"""050: training with metric weights — row weight = pairs_at_step / rows_at_step."""
import sys, time, os
sys.path.insert(0, "repo/src")
import numpy as np, joblib
import lightgbm as lgb
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)

def metric_weights(step_arr, lab_arr):
    """Row weight: the metric's weight share at its step, divided by the number of rows at that step."""
    steps, inv = np.unique(step_arr, return_inverse=True)
    pos = np.bincount(inv, weights=(lab_arr == 1).astype("float64"))
    neg = np.bincount(inv, weights=(lab_arr == 0).astype("float64"))
    cnt = np.bincount(inv).astype("float64")
    pair_w = pos * neg
    per_row = pair_w / np.maximum(cnt, 1)
    w = per_row[inv]
    return w / w.mean()

tr, va = assignment != 0, assignment == 0
yf, sf = y[va], s[va]

# 1) Classifier with metric weights
w_tr = metric_weights(s[tr], y[tr])
clf = lgb.LGBMClassifier(
    objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=500,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0,
    n_estimators=300, random_state=0, n_jobs=8, deterministic=True,
    force_row_wise=True, verbose=-1)
clf.fit(X[tr], y[tr], sample_weight=w_tr)
p_clf = clf.predict_proba(X[va])[:, 1]
print(f"classifier with metric weights: fold 0 {ts_auc(p_clf, yf, sf):.4f} "
      f"(old 0.5952) [{time.time()-t0:.0f}s]", flush=True)
np.save("oof_clf_mw.npy", p_clf)

# 2) Ranker: the group (step) weight is proportional to the number of pairs
Xtr, ytr, str_ = X[tr], y[tr], s[tr]
order = np.argsort(str_, kind="stable")
Xtr, ytr, str_sorted = Xtr[order], ytr[order], str_[order]
uniq, sizes = np.unique(str_sorted, return_counts=True)
pos_g = np.array([((ytr[str_sorted == u]) == 1).sum() for u in uniq], dtype="float64")
neg_g = sizes - pos_g
gw = pos_g * neg_g
gw = gw / gw.mean()
# LightGBM does not accept group weights directly — emulate them via row weights
row_w = np.repeat(gw, sizes)
rk = lgb.LGBMRanker(
    objective="lambdarank", learning_rate=0.03, num_leaves=31,
    min_child_samples=500, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=10.0, n_estimators=600,
    random_state=0, n_jobs=8, deterministic=True, force_row_wise=True,
    verbose=-1, lambdarank_truncation_level=2000, label_gain=[0, 1])
rk.fit(Xtr, ytr, group=sizes, sample_weight=row_w)
p_rk = rk.predict(X[va])
print(f"ranker with group weights: fold 0 {ts_auc(p_rk, yf, sf):.4f} (old 0.6016) "
      f"[{time.time()-t0:.0f}s]", flush=True)
np.save("oof_rank_mw.npy", p_rk)

sig = 1.0 / (1.0 + np.exp(-p_rk))
print(f"0.6/0.4 pair with metric weights: {ts_auc(0.6*sig + 0.4*p_clf, yf, sf):.4f} "
      f"(old pair 0.6045)", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
