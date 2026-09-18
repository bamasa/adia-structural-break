"""104: древесные специалисты по диапазонам шагов — отдельный классификатор на t<50, 50<=t<200, t>=200.

Шаг как признак вредил (учит базовую долю), но специализация — другое: на ранних шагах работают
короткие окна, на поздних — длинные; одна модель делит ёмкость между режимами. Каждый специалист
обучается на своём диапазоне (с перекрытием 20% для гладкости) и скорит только его. Парно с clf 206.
"""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"), np.load("SPEC14.npy"), np.load("BOCPD6_H50.npy")]).astype("float32")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2
params = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8)
base = np.load("fold2_clf_evw_0.npy")
print(f"единая модель: фолд-2 {ts_auc(base, y[te], s[te]):.4f}", flush=True)
for name, edges in (("3 диапазона", [0, 50, 200, 10**6]), ("2 диапазона", [0, 150, 10**6])):
    out = np.zeros(te.sum()); st, ste = s[tr], s[te]
    for lo, hi in zip(edges[:-1], edges[1:]):
        fit = (st >= lo * 0.8) & (st < hi * 1.2); use = (ste >= lo) & (ste < hi)
        m = lgb.LGBMClassifier(**params).fit(X[tr][fit], y[tr][fit]); out[use] = m.predict_proba(X[te][use])[:, 1]
        print(f"  [{lo},{hi}): строк {fit.sum()}, AUC диапазона специалист {ts_auc(out[use], y[te][use], ste[use]):.4f} vs единая {ts_auc(base[use], y[te][use], ste[use]):.4f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"специалисты ({name}): фолд-2 {ts_auc(out, y[te], s[te]):.4f}", flush=True)
    np.save(f"fold2_clf_spec_{len(edges)-1}.npy", out)
