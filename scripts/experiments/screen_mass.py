"""114: classifier 206 against 296 (+MASS90), and MASS90 on its own."""
import sys, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"), np.load("SPEC14.npy"), np.load("BOCPD6_H50.npy")]).astype("float32")
M = np.load("MASS90.npy")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
params = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8)
res = {}
for name, D in (("206", X), ("90 (MASS only)", M), ("296 (206+MASS)", np.hstack([X, M]))):
    clf = lgb.LGBMClassifier(**params).fit(D[tr], y[tr]); p = clf.predict_proba(D[te])[:, 1]
    res[name] = p; np.save(f"fold2_clf_mass_{name[:3].strip()}.npy", p)
    print(f"clf {name}: fold 2 {ts_auc(p, yf, sf):.4f} [{time.time()-t0:.0f}s]", flush=True)
    if name.startswith("296"):
        imp = clf.booster_.feature_importance("gain"); order = np.argsort(-imp)
        r = [int(np.where(order == j)[0][0]) + 1 for j in range(206, 296)]
        print(f"  ranks of the MASS channels: best {min(r)}, median {int(np.median(r))}, in top-50: {sum(x <= 50 for x in r)}", flush=True)
base = np.load("fold2_base30.npy"); sig = lambda a: 1/(1+np.exp(-a.astype("float64")))
rank = sig(np.load("fold2_rank_bocpd_200.npy"))
print(f"\nensemble #30: {ts_auc(base, yf, sf):.4f}")
for name in ("90 (MASS only)", "296 (206+MASS)"):
    p = res[name]
    print(f"  clf={name}: trees 0.7/0.3 {ts_auc(0.7*rank + 0.3*p, yf, sf):.4f} | correlation with the ensemble {pd.Series(p).corr(pd.Series(base), method='spearman'):.3f}")
    for w in (0.1, 0.2):
        print(f"    ensemble + {w}·clf: {ts_auc((1-w)*base + w*p, yf, sf):.4f}")
