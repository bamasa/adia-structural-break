"""146: the tail-referenced mass battery as a member on fold 2 — alone, and against the whole-history mass member."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
D = np.load("MASSTAIL512.npy"); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
ref = 0.55 * base + 0.25 * mass + 0.20 * fq; late = 0.50 * base + 0.25 * mass + 0.25 * un; s39 = np.where(sf < 100, ref, late)
sp = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
p = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8).fit(D[tr], y[tr]).predict_proba(D[te])[:, 1]
np.save("fold2_clf_masstail512.npy", p)
print(f"масса по хвосту истории (512): соло {ts_auc(p, yf, sf):.4f} | масса по всей истории соло {ts_auc(mass, yf, sf):.4f} | Spearman между ними {sp(p, mass):.3f}, с #39 {sp(p, s39):.3f} [{time.time()-t0:.0f}s]")
print(f"#39: {ts_auc(s39, yf, sf):.4f}")
for w in (0.10, 0.15, 0.20, 0.25):
    print(f"  #39 + хвостовая масса доля {w:.2f}: {ts_auc((1 - w) * s39 + w * p, yf, sf):.4f}")
for w in (0.10, 0.15):
    ref2 = 0.55 * base + (0.25 - w) * mass + w * p + 0.20 * fq; late2 = 0.50 * base + (0.25 - w) * mass + w * p + 0.25 * un
    print(f"  часть массовой доли ({w:.2f}) отдана хвостовой: {ts_auc(np.where(sf < 100, ref2, late2), yf, sf):.4f}")
print(f"готово [{time.time()-t0:.0f}s]")
