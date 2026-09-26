"""137: учитель частотного члена — сетка вокруг 131b (1500 деревьев, lr 0.015, 31 лист, min_child 300)."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0])
D = np.hstack([np.load("MSPEC32.npy"), np.load("MSPEC60.npy"), np.load("ACF8.npy")])
def blend(fd, w): return (0.75 - w) * base + 0.25 * mass + w * fd
ref = np.load("fold2_clf_freqdep_1500.npy"); print(f"131b: соло {ts_auc(ref,yf,sf):.4f}, смесь 0.20 {ts_auc(blend(ref,0.20),yf,sf):.4f}", flush=True)
grid = [("3000 деревьев lr 0.0075", dict(n_estimators=3000, learning_rate=0.0075, num_leaves=31, min_child_samples=300)),
        ("1500, 15 листьев", dict(n_estimators=1500, learning_rate=0.015, num_leaves=15, min_child_samples=300)),
        ("1500, min_child 600", dict(n_estimators=1500, learning_rate=0.015, num_leaves=31, min_child_samples=600)),
        ("1500, colsample 0.3", dict(n_estimators=1500, learning_rate=0.015, num_leaves=31, min_child_samples=300, colsample_bytree=0.3)),
        ("2500 lr 0.01, 15 листьев, min_child 600", dict(n_estimators=2500, learning_rate=0.01, num_leaves=15, min_child_samples=600))]
for name, P in grid:
    params = dict(colsample_bytree=0.5, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=6); params.update(P)
    clf = lgb.LGBMClassifier(**params).fit(D[tr], y[tr]); p = clf.predict_proba(D[te])[:, 1]
    tag = name.replace(" ", "_").replace(",", "")[:24]; np.save(f"fold2_clf_fdsweep_{tag}.npy", p)
    best = max((ts_auc(blend(p, w), yf, sf), w) for w in (0.15, 0.20, 0.25))
    print(f"{name}: соло {ts_auc(p,yf,sf):.4f} | корр. с базой {pd.Series(p).corr(pd.Series(base), method='spearman'):.3f} | смесь {best[0]:.4f} при {best[1]} [{time.time()-t0:.0f}s]", flush=True)
