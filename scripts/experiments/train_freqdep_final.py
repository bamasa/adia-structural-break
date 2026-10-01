"""Final frequency-dependence member (131b) on all 10,000 series: 100 channels, 1500 trees."""
import time, numpy as np, lightgbm as lgb, joblib
t0 = time.time()
D = np.hstack([np.load("MSPEC32.npy"), np.load("MSPEC60.npy"), np.load("ACF8.npy")]); y = np.load("Y40.npy")
clf = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.015, num_leaves=31, colsample_bytree=0.5, subsample=0.8,
                         subsample_freq=1, min_child_samples=300, verbose=-1, n_jobs=6).fit(D, y)
joblib.dump(clf, "resources083_freqdep.joblib")
print(f"freqdep: {clf.booster_.num_feature()} features, {len(y)} rows [{time.time()-t0:.0f}s]")
