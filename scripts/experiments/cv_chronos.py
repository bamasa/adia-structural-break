"""030b (fold 0): does Chronos add to the ensemble — 186+3 and 236+3."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import lightgbm as lgb
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
base = [np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
        np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
        np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")]
C3 = np.load("C3.npy"); A50 = np.load("A50.npy")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
tr, va = assignment != 0, assignment == 0
yf, sf = y[va], s[va]
clf = np.load("oof_cfg5.npy")[va].astype("float64")

def screen(mats, tag, ref):
    X = np.hstack(mats)
    Xtr, ytr, str_ = X[tr], y[tr], s[tr]
    order = np.argsort(str_, kind="stable")
    Xtr, ytr, str_ = Xtr[order], ytr[order], str_[order]
    _, sizes = np.unique(str_, return_counts=True)
    r = lgb.LGBMRanker(
        objective="lambdarank", learning_rate=0.03, num_leaves=31,
        min_child_samples=500, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=10.0, n_estimators=600,
        random_state=0, n_jobs=8, deterministic=True, force_row_wise=True,
        verbose=-1, lambdarank_truncation_level=2000, label_gain=[0, 1])
    r.fit(Xtr, ytr, group=sizes)
    sc = r.predict(X[va])
    sig = 1.0 / (1.0 + np.exp(-sc))
    print(f"{tag}: alone {ts_auc(sc, yf, sf):.4f}, blend {ts_auc(0.6*sig+0.4*clf, yf, sf):.4f} ({ref})  [{time.time()-t0:.0f}s]", flush=True)

screen(base + [C3], "189 (186 + Chronos)", "references 0.6016 / 0.6045")
screen(base + [A50, C3], "239 (186 + anchor + Chronos)", "anchor reference will come from cv_anchor")
print(f"total {time.time()-t0:.0f}s", flush=True)
