"""085: проверка BOCPD-каналов классификатором на фолде-2 (200 против 206)."""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"),
               np.load("X50a.npy"), np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy")]).astype("float32")
B = np.hstack([np.load("BOCPD6_H50.npy"), np.load("BOCPD6_H20.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0)
tr, te = f != 2, f == 2
params = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5,
              subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8)
res = {}
for name, M in (("200", X), ("212 (+BOCPD 1/50 и 1/20)", np.hstack([X, B]))):
    clf = lgb.LGBMClassifier(**params).fit(M[tr], y[tr])
    p = clf.predict_proba(M[te])[:, 1]
    res[name] = ts_auc(p, y[te], s[te])
    np.save(f"fold2_clf_bocpdboth_{name[:3]}.npy", p)
    print(f"clf {name}: фолд-2 {res[name]:.4f} [{time.time()-t0:.0f}s]", flush=True)
    if name.startswith("206"):
        imp = clf.booster_.feature_importance("gain"); order = np.argsort(-imp)
        ranks = [int(np.where(order == j)[0][0]) + 1 for j in range(200, 212)]
        print(f"  ранги BOCPD-каналов по gain из 212: {ranks}", flush=True)
print(f"ИТОГ: прирост {res['212 (+BOCPD 1/50 и 1/20)'] - res['200']:+.4f}  (принятие: ≥ +0.002)", flush=True)
