"""085: checking the BOCPD channels with the classifier on fold 2 (200 vs 206)."""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"),
               np.load("X50a.npy"), np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy")]).astype("float32")
B = np.load("BOCPD6.npy")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0)
tr, te = f != 2, f == 2
params = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5,
              subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8)
res = {}
for name, M in (("200", X), ("206 (+BOCPD)", np.hstack([X, B]))):
    clf = lgb.LGBMClassifier(**params).fit(M[tr], y[tr])
    p = clf.predict_proba(M[te])[:, 1]
    res[name] = ts_auc(p, y[te], s[te])
    np.save(f"fold2_clf_bocpd_{name[:3]}.npy", p)
    print(f"clf {name}: fold 2 {res[name]:.4f} [{time.time()-t0:.0f}s]", flush=True)
    if name.startswith("206"):
        imp = clf.booster_.feature_importance("gain"); order = np.argsort(-imp)
        ranks = [int(np.where(order == j)[0][0]) + 1 for j in range(200, 206)]
        print(f"  gain ranks of the BOCPD channels out of 206: {ranks}", flush=True)
print(f"RESULT: gain {res['206 (+BOCPD)'] - res['200']:+.4f}  (acceptance: ≥ +0.002)", flush=True)
