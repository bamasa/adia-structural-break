"""102: веса положительных строк по наблюдаемости слома — w = min(1, (t - tau + 1) / W).

Сразу после tau улик нет; метка 1 там — шум для обучения. Отрицательные строки — вес 1.
Парно на классификаторе 206: без весов / W=20 / W=50 / W=100 / W=200.
"""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"), np.load("SPEC14.npy"), np.load("BOCPD6_H50.npy")]).astype("float32")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2
yi = pd.read_parquet("structural-break-real-time-test/data/y_train_index.parquet")
tau = yi.loc[g, "tau_index"].to_numpy()
since = np.where(tau >= 0, s - tau + 1, 0)            # число точек после слома (для положительных строк >= 1)
params = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8)
for W in (0, 20, 50, 100, 200):
    w = None
    if W:
        w = np.ones(tr.sum()); pos = y[tr] == 1; w[pos] = np.clip(since[tr][pos] / W, 0.05, 1.0)
    clf = lgb.LGBMClassifier(**params).fit(X[tr], y[tr], sample_weight=w); p = clf.predict_proba(X[te])[:, 1]
    np.save(f"fold2_clf_evw_{W}.npy", p)
    print(f"clf 206, W={W or 'нет'}: фолд-2 {ts_auc(p, y[te], s[te]):.4f} [{time.time()-t0:.0f}s]", flush=True)
