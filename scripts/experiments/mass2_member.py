"""118: расширенная батарея как член — против массовой батареи 114."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
M2 = np.load("MASS216.npy"); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8,
                         subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8).fit(M2[tr], y[tr])
p2 = clf.predict_proba(M2[te])[:, 1]; np.save("fold2_clf_mass216.npy", p2)
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0])
print(f"расширенная батарея: соло {ts_auc(p2, yf, sf):.4f} | корреляция с ансамблем {pd.Series(p2).corr(pd.Series(base), method='spearman'):.3f} "
      f"| с батареей-90 {pd.Series(p2).corr(pd.Series(mass), method='spearman'):.3f} [{time.time()-t0:.0f}s]")
print(f"база {ts_auc(base, yf, sf):.4f}; батарея-90 в смеси (#36) {ts_auc(0.75*base + 0.25*mass, yf, sf):.4f}")
best = (0, None)
for w in (0.15, 0.20, 0.25, 0.30, 0.35):
    v = ts_auc((1 - w) * base + w * p2, yf, sf)
    if v > best[0]: best = (v, ("одна 216", w))
    print(f"  база + {w:.2f}·216: {v:.4f}")
for w2 in (0.10, 0.15, 0.20):
    v = ts_auc(0.75 * base + 0.25 * ((1 - w2 / 0.25) * mass + (w2 / 0.25) * p2) if w2 <= 0.25 else 0, yf, sf)
for wm, w2 in ((0.15, 0.15), (0.20, 0.15), (0.15, 0.20), (0.125, 0.125)):
    v = ts_auc((1 - wm - w2) * base + wm * mass + w2 * p2, yf, sf)
    if v > best[0]: best = (v, ("обе", wm, w2))
    print(f"  база + {wm:.3f}·90 + {w2:.3f}·216: {v:.4f}")
print(f"ЛУЧШЕЕ: {best[0]:.4f} — {best[1]}")
