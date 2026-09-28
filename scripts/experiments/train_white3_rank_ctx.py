"""Финальный второй ранкер члена v3 (с контекстом, 119 входов) на всех рядах — для бэга с ранкером #42."""
import time, numpy as np, lightgbm as lgb, joblib
t0 = time.time(); D = np.hstack([np.load("WHITE90.npy"), np.load("SR22.npy"), np.load("HISTCTX8.npy")]); y = np.load("Y40.npy"); s = np.load("S40.npy")
order = np.argsort(s, kind="stable"); _, sizes = np.unique(s[order], return_counts=True)
rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.5, reg_lambda=10.0, lambdarank_truncation_level=2000, label_gain=[0, 1], verbose=-1, n_jobs=6)
rk.fit(D[order], y[order], group=sizes); joblib.dump(rk, "resources088_white_rank_ctx.joblib"); print(f"rank ctx: {rk.booster_.num_feature()} признаков [{time.time()-t0:.0f}s]", flush=True)
