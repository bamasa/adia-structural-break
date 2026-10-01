"""Final frequency+novelty union (120 channels) on all 10,000 series, slow teacher 131b."""
import time, numpy as np, lightgbm as lgb, joblib
t0 = time.time()
D = np.hstack([np.load("MSPEC32.npy"), np.load("MSPEC60.npy"), np.load("ACF8.npy"), np.load("KNN20.npy")]); y = np.load("Y40.npy")
clf = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.015, num_leaves=31, colsample_bytree=0.5, subsample=0.8,
                         subsample_freq=1, min_child_samples=300, verbose=-1, n_jobs=6).fit(D, y)
joblib.dump(clf, "resources084_union.joblib")
print(f"union: {clf.booster_.num_feature()} features, {len(y)} rows [{time.time()-t0:.0f}s]")
