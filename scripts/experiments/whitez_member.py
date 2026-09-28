"""149: the calibrated battery as the member's input — z-only and raw+z — against the raw one."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time(); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
s39 = np.where(sf < 100, 0.55 * base + 0.25 * mass + 0.20 * fq, 0.50 * base + 0.25 * mass + 0.25 * un)
ref = np.load("fold2_clf_white_sr_slower.npy")
W = np.load("WHITE90.npy"); Z = np.load("WHITEZ76.npy"); S = np.load("SR22.npy")
slow = dict(n_estimators=3000, learning_rate=0.0075, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=500, reg_lambda=10.0, verbose=-1, n_jobs=8)
sp = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
print(f"опора: сырые 90 + SR (медленный клф): соло {ts_auc(ref, yf, sf):.4f} | #39 + 0.30: {ts_auc(0.7 * s39 + 0.3 * ref, yf, sf):.4f}", flush=True)
for name, D in (("только z (76 z + 14 прочих + SR)", np.hstack([Z, W[:, 76:], S])), ("сырые + z + SR", np.hstack([W, Z, S]))):
    Dtr, Dte = D[tr], D[te]; del D
    p = lgb.LGBMClassifier(**slow).fit(Dtr, y[tr]).predict_proba(Dte)[:, 1]; np.save(f"fold2_clf_whitez_{'z' if name.startswith('только') else 'rawz'}.npy", p)
    print(f"{name}: соло {ts_auc(p, yf, sf):.4f} | Spearman с опорой {sp(p, ref):.3f} | #39 + 0.30: {ts_auc(0.7 * s39 + 0.3 * p, yf, sf):.4f} | + 0.40: {ts_auc(0.6 * s39 + 0.4 * p, yf, sf):.4f} [{time.time()-t0:.0f}s]", flush=True)
    del Dtr, Dte
print(f"готово [{time.time()-t0:.0f}s]")
