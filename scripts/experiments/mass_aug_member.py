"""119: mass member trained with boundary augmentation (AUG3) — against member #36."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
M = np.load("MASS90.npy"); AM = np.load("AUG3_MASS90.npy")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
AY = np.load("AUG3_Y.npy"); AG = np.load("AUG3_G.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
fold_of = {int(g[a]): int(f[a]) for a in starts}
keep = np.array([fold_of.get((int(v) - 100000) // 10, 0) != 2 for v in AG])
print(f"pseudo-series in training: {keep.sum()} rows out of {len(AG)} [{time.time()-t0:.0f}s]", flush=True)
params = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8,
              subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8)
Xtr = np.vstack([M[tr], AM[keep]]); ytr = np.concatenate([y[tr], AY[keep]])
clf = lgb.LGBMClassifier(**params).fit(Xtr, ytr)
p = clf.predict_proba(M[te])[:, 1]; np.save("fold2_clf_mass_aug.npy", p)
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0])
print(f"mass with augmentation: alone {ts_auc(p, yf, sf):.4f} (without augmentation 0.5830) | correlation with the ensemble {pd.Series(p).corr(pd.Series(base), method='spearman'):.3f} [{time.time()-t0:.0f}s]")
print(f"#36 (base + 0.25·mass): {ts_auc(0.75*base + 0.25*mass, yf, sf):.4f}")
best = (0, None)
for w in (0.20, 0.25, 0.30, 0.35):
    v = ts_auc((1 - w) * base + w * p, yf, sf)
    if v > best[0]: best = (v, w)
    print(f"  base + {w:.2f}·mass-with-augmentation: {v:.4f}")
print(f"BEST: {best[0]:.4f} at share {best[1]}  (#36 = 0.6205)")
