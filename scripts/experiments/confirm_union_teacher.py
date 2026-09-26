"""Подтвердить, что fold2_clf_freqdep_knn.npy получен медленным учителем 131b."""
import sys, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
D = np.hstack([np.load("MSPEC32.npy"), np.load("MSPEC60.npy"), np.load("ACF8.npy"), np.load("KNN20.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2
clf = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.015, num_leaves=31, colsample_bytree=0.5, subsample=0.8,
                         subsample_freq=1, min_child_samples=300, verbose=-1, n_jobs=5).fit(D[tr], y[tr])
p = clf.predict_proba(D[te])[:, 1]; c = np.load("fold2_clf_freqdep_knn.npy")
print(f"медленный учитель заново: соло {ts_auc(p, y[te], s[te]):.4f} | кэш {ts_auc(c, y[te], s[te]):.4f} | Spearman {pd.Series(p).corr(pd.Series(c), method='spearman'):.4f} | max|diff| {np.abs(p - c).max():.3f} [{time.time()-t0:.0f}s]")
