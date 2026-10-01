"""096: classifier 206 (with NG-BOCPD) against 213 (+AR-BOCPD, 7 channels) on fold 2."""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"), np.load("SPEC14.npy"), np.load("BOCPD6_H50.npy")]).astype("float32")
A = np.load("BOCPDAR7.npy")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2
params = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=6)
res = {}
for name, M in (("206", X), ("213 (+AR)", np.hstack([X, A])), ("207 (+only P(break happened))", np.hstack([X, A[:, 6:7]]))):
    clf = lgb.LGBMClassifier(**params).fit(M[tr], y[tr]); p = clf.predict_proba(M[te])[:, 1]
    res[name] = ts_auc(p, y[te], s[te]); np.save(f"fold2_clf_ar_{name[:3]}.npy", p)
    print(f"clf {name}: fold 2 {res[name]:.4f} [{time.time()-t0:.0f}s]", flush=True)
    if name.startswith("213"):
        imp = clf.booster_.feature_importance("gain"); order = np.argsort(-imp)
        print("  ranks of the AR channels by gain out of 213:", [int(np.where(order == j)[0][0]) + 1 for j in range(206, 213)], flush=True)
print(f"RESULT: +AR {res['213 (+AR)'] - res['206']:+.4f}; only P(break happened) {res['207 (+only P(break happened))'] - res['206']:+.4f}", flush=True)
