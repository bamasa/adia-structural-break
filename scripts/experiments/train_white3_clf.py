"""Final classifier of the v3 member (90 + 21 SR + 8 context = 119) on all series."""
import time, numpy as np, lightgbm as lgb, joblib
t0 = time.time(); D = np.hstack([np.load("WHITE90.npy"), np.load("SR22.npy"), np.load("HISTCTX8.npy")]); y = np.load("Y40.npy")
clf = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.0075, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                         min_child_samples=500, reg_lambda=10.0, verbose=-1, n_jobs=4).fit(D, y)
joblib.dump(clf, "resources088_white_clf.joblib"); print(f"clf v3: {clf.booster_.num_feature()} features [{time.time()-t0:.0f}s]", flush=True)
