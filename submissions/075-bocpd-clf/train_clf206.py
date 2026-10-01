"""Final classifier on all 10,000 series: 200 channels + 6 BOCPD (hazard 1/50)."""
import time, numpy as np, lightgbm as lgb, joblib
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"),
               np.load("X50a.npy"), np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy"), np.load("BOCPD6_H50.npy")]).astype("float32")
y = np.load("Y40.npy")
params = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5,
              subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8)
clf = lgb.LGBMClassifier(**params).fit(X, y)
joblib.dump(clf, "resources075/clf206.joblib")
print(f"clf206: {clf.booster_.num_feature()} features, {X.shape[0]} rows [{time.time()-t0:.0f}s]", flush=True)
