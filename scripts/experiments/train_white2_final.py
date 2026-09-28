"""Финальный отбелённый член v2 для #42: ранкер по шагам + медленный классификатор на 90 + 21 SR каналах, все ряды."""
import time, numpy as np, lightgbm as lgb, joblib
t0 = time.time(); D = np.hstack([np.load("WHITE90.npy"), np.load("SR22.npy")]); y = np.load("Y40.npy"); s = np.load("S40.npy")
clf = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.0075, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                         min_child_samples=500, reg_lambda=10.0, verbose=-1, n_jobs=6).fit(D, y)
joblib.dump(clf, "resources087_white_clf.joblib"); print(f"clf: {clf.booster_.num_feature()} признаков [{time.time()-t0:.0f}s]", flush=True)
order = np.argsort(s, kind="stable"); _, sizes = np.unique(s[order], return_counts=True)
rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.5, reg_lambda=10.0, lambdarank_truncation_level=2000, label_gain=[0, 1], verbose=-1, n_jobs=6)
rk.fit(D[order], y[order], group=sizes)
joblib.dump(rk, "resources087_white_rank.joblib"); print(f"rank: {rk.booster_.num_feature()} признаков [{time.time()-t0:.0f}s]", flush=True)
