"""Final mass-battery classifier on all 10,000 series."""
import time, numpy as np, lightgbm as lgb, joblib
t0 = time.time()
M = np.load("MASS90.npy"); y = np.load("Y40.npy")
clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5,
                         subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8).fit(M, y)
joblib.dump(clf, "resources081_massclf.joblib")
print(f"mass-battery classifier: {clf.booster_.num_feature()} features, {len(y)} rows [{time.time()-t0:.0f}s]")
