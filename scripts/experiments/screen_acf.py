"""110: classifier 206 versus 214 (+ACF8), and separately — on dependence breaks."""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"), np.load("SPEC14.npy"), np.load("BOCPD6_H50.npy")]).astype("float32")
A = np.load("ACF8.npy")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2
params = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8)
res = {}
for name, M in (("206", X), ("214 (+ACF)", np.hstack([X, A]))):
    clf = lgb.LGBMClassifier(**params).fit(M[tr], y[tr]); p = clf.predict_proba(M[te])[:, 1]
    res[name] = p; np.save(f"fold2_clf_acf_{name[:3]}.npy", p)
    print(f"clf {name}: fold 2 {ts_auc(p, y[te], s[te]):.4f} [{time.time()-t0:.0f}s]", flush=True)
    if name.startswith("214"):
        imp = clf.booster_.feature_importance("gain"); order = np.argsort(-imp)
        print("  ranks of the ACF channels by gain out of 214:", [int(np.where(order == j)[0][0]) + 1 for j in range(206, 214)], flush=True)
print(f"RESULT: {ts_auc(res['214 (+ACF)'], y[te], s[te]) - ts_auc(res['206'], y[te], s[te]):+.4f}", flush=True)
# separately on the series where the break changes the dependence
yi = pd.read_parquet("structural-break-real-time-test/data/y_train_index.parquet")
Xd = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
gf, yf, sf = g[te], y[te], s[te]
st = np.flatnonzero(np.concatenate([[True], gf[1:] != gf[:-1]])); bd = np.append(st, len(gf))
ar1 = lambda v: np.corrcoef(v[:-1], v[1:])[0, 1] if len(v) > 5 else 0.0
dep, rest = [], []
for a, b in zip(st, bd[1:]):
    sid = int(gf[a]); tau = int(yi.loc[sid, "tau_index"])
    if tau < 0: rest.append((a, b)); continue
    part = Xd.loc[sid]; v = part.value.to_numpy("float64"); p = part.period.to_numpy()
    hist, online = v[p == 1], v[p == 2]
    if (b - a) - tau < 100: rest.append((a, b)); continue
    pre = np.concatenate([hist, online[:tau]])[-300:]; post = online[tau:][:300]
    (dep if abs(ar1(post) - ar1(pre)) > 0.12 else rest).append((a, b))
mask = np.zeros(len(gf), bool)
for a, b in dep: mask[a:b] = True
print(f"\nseries with a dependence break: {len(dep)}; rows {mask.sum()}")
for name in ("206", "214 (+ACF)"):
    sub = res[name][mask | np.isin(np.arange(len(gf)), [a for a, _ in rest])]
    print(f"  {name}: only these series + clean ones {ts_auc(res[name][mask], yf[mask], sf[mask]):.4f}", flush=True)
