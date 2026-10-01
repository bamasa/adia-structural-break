"""115: the "rejected" families as an independent member.

GLR6, ACF8, DIV8, TESTS8, BOCPDAR7 — each lost as an addition to the 206 channels.
Principle 114 says: test not the addition but a separate member. 37 channels.
"""
import sys, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
R = np.hstack([np.load(f) for f in ("RANKBAT.npy",)])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
print(f"rank battery channels: {R.shape[1]}")
clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8,
                         subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8).fit(R[tr], y[tr])
p = clf.predict_proba(R[te])[:, 1]; np.save("fold2_clf_rankbat.npy", p)
base = np.load("fold2_base30.npy")
mass = np.load([q for q in __import__("glob").glob("fold2_clf_mass_90*.npy")][0])
cur = 0.75 * base + 0.25 * mass                      # what was shipped as #36
print(f"member on the rank battery: alone {ts_auc(p, yf, sf):.4f} | correlation with #36 {pd.Series(p).corr(pd.Series(cur), method='spearman'):.3f} [{time.time()-t0:.0f}s]")
print(f"#36 (base + 0.25·mass): {ts_auc(cur, yf, sf):.4f}")
best = (0, None)
for w in (0.05, 0.10, 0.15, 0.20, 0.25):
    v = ts_auc((1 - w) * cur + w * p, yf, sf)
    if v > best[0]: best = (v, w)
    print(f"  + rank battery share {w:.2f}: {v:.4f}  ({v - ts_auc(cur, yf, sf):+.4f})")
print(f"BEST: {best[0]:.4f} at share {best[1]}")
