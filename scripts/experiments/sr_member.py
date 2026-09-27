"""147: the whitened member with the Shiryaev-Roberts channels appended, against the whitened member alone."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time(); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
s39 = np.where(sf < 100, 0.55 * base + 0.25 * mass + 0.20 * fq, 0.50 * base + 0.25 * mass + 0.25 * un)
wh = np.load("fold2_clf_white_slow.npy")
D = np.hstack([np.load("WHITE90.npy"), np.load("SR22.npy")]); Dtr, Dte = D[tr], D[te]; del D
slow = dict(n_estimators=1500, learning_rate=0.015, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=300, verbose=-1, n_jobs=8)
p = lgb.LGBMClassifier(**slow).fit(Dtr, y[tr]).predict_proba(Dte)[:, 1]; np.save("fold2_clf_white_sr.npy", p)
sp = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
print(f"отбелённый+SR: соло {ts_auc(p, yf, sf):.4f} (отбелённый один {ts_auc(wh, yf, sf):.4f}) | Spearman между ними {sp(p, wh):.3f} [{time.time()-t0:.0f}s]")
print(f"#39: {ts_auc(s39, yf, sf):.4f} | " + " | ".join(f"#39 + доля {w:.2f}: бел {ts_auc((1 - w) * s39 + w * wh, yf, sf):.4f} / бел+SR {ts_auc((1 - w) * s39 + w * p, yf, sf):.4f}" for w in (0.30, 0.35, 0.40)))
print("--- по диапазонам: #39+0.30·бел -> #39+0.30·(бел+SR) ---")
for a, b in ((0, 30), (30, 100), (100, 300), (300, 700), (700, 3000)):
    m = (sf >= a) & (sf < b); print(f"  шаги {a}-{b}: {ts_auc(0.7 * s39[m] + 0.3 * wh[m], yf[m], sf[m]):.4f} -> {ts_auc(0.7 * s39[m] + 0.3 * p[m], yf[m], sf[m]):.4f}")
S = np.load("SR22.npy")[te]
print("SR-каналы сами (без обучения), смесь по семействам: " + " | ".join(f"{k} {ts_auc(S[:, j], yf, sf):.4f}" for k, j in (("var_up", 18), ("var_down", 19), ("mean", 20), ("dep", 21))))
print(f"готово [{time.time()-t0:.0f}s]")
