"""145 (rebuilt matrix): the whitened member under the slow learner on fold 2, alone and at a 0.30 share."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time(); D = np.load("WHITE90.npy"); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
s39 = np.where(sf < 100, 0.55 * base + 0.25 * mass + 0.20 * fq, 0.50 * base + 0.25 * mass + 0.25 * un)
Dtr, Dte = D[tr], D[te]; del D
p = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.015, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=300, verbose=-1, n_jobs=8).fit(Dtr, y[tr]).predict_proba(Dte)[:, 1]
np.save("fold2_clf_white_slow.npy", p)
print(f"whitened member (rebuilt matrix, step without clipping): alone {ts_auc(p, yf, sf):.4f} | #39 {ts_auc(s39, yf, sf):.4f} | " + " | ".join(f"weight {w:.2f}: {ts_auc((1 - w) * s39 + w * p, yf, sf):.4f}" for w in (0.25, 0.30, 0.35)) + f" [{time.time()-t0:.0f}s]", flush=True)
