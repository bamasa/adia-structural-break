"""Final wide LightGBM and ExtraTrees on all series (206 channels) for the classifier triple of #33."""
import time, numpy as np, lightgbm as lgb, joblib
from sklearn.ensemble import ExtraTreesClassifier
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"), np.load("SPEC14.npy"), np.load("BOCPD6_H50.npy")]).astype("float32")
y = np.load("Y40.npy")
wide = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=255, max_depth=8, colsample_bytree=0.4, subsample=0.7, subsample_freq=1, min_child_samples=300, verbose=-1, n_jobs=8).fit(X, y)
joblib.dump(wide, "resources077_wide206.joblib"); print(f"wide: {wide.booster_.num_feature()} features [{time.time()-t0:.0f}s]", flush=True)
rng = np.random.default_rng(0); idx = rng.choice(len(y), 1_500_000, replace=False)
et = ExtraTreesClassifier(n_estimators=300, max_features=0.3, min_samples_leaf=50, n_jobs=8).fit(X[idx], y[idx])
et.n_jobs = 1   # single-row inference: no thread pools (lesson of #109121)
joblib.dump(et, "resources077_et206.joblib", compress=3); print(f"extratrees: {et.n_features_in_} features, {len(et.estimators_)} trees [{time.time()-t0:.0f}s]", flush=True)
