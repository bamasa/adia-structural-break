"""131: усиление члена 130 — (a) + каналы AR-фильтра, (b) больше деревьев на том же наборе."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); cur = 0.75 * base + 0.25 * mass
D0 = np.hstack([np.load("MSPEC32.npy"), np.load("MSPEC60.npy"), np.load("ACF8.npy")])
D1 = np.hstack([D0, np.load("BOCPDAR7.npy")])
P600 = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=6)
P1500 = dict(P600, n_estimators=1500, learning_rate=0.015, num_leaves=31, min_child_samples=300)
def run(name, D, P, tag):
    clf = lgb.LGBMClassifier(**P).fit(D[tr], y[tr]); p = clf.predict_proba(D[te])[:, 1]; np.save(f"fold2_clf_{tag}.npy", p)
    best = max((ts_auc((1 - w) * cur + w * p, yf, sf), w) for w in (0.10, 0.15, 0.20, 0.25))
    print(f"{name}: соло {ts_auc(p, yf, sf):.4f} | корр. с #36 {pd.Series(p).corr(pd.Series(cur), method='spearman'):.3f} | лучшее в смеси {best[0]:.4f} при {best[1]} ({best[0] - ts_auc(cur, yf, sf):+.4f}) [{time.time()-t0:.0f}s]", flush=True)
print(f"#36: {ts_auc(cur, yf, sf):.4f}; член 130 (100 каналов, 600 деревьев): соло 0.5357, +0.0018", flush=True)
run("(a) +AR-фильтр, 107 каналов, 600 деревьев", D1, P600, "freqdep_ar")
run("(b) 100 каналов, 1500 деревьев lr 0.015, 31 лист", D0, P1500, "freqdep_1500")
run("(c) 107 каналов, 1500 деревьев", D1, P1500, "freqdep_ar_1500")
