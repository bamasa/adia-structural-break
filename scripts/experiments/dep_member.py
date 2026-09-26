"""144: the dependence-score CUSUM battery as its own member on fold 2."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
D = np.load("DEP23.npy"); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
ref = 0.55 * base + 0.25 * mass + 0.20 * fq; late = 0.50 * base + 0.25 * mass + 0.25 * un; s39 = np.where(sf < 100, ref, late)
slow = dict(n_estimators=1500, learning_rate=0.015, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=300, verbose=-1, n_jobs=8)
fast = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8)
sp = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
res = {}
for name, prm in (("медленный", slow), ("быстрый", fast)):
    p = lgb.LGBMClassifier(**prm).fit(D[tr], y[tr]).predict_proba(D[te])[:, 1]; res[name] = p
    np.save(f"fold2_clf_dep_{'slow' if name == 'медленный' else 'fast'}.npy", p)
    print(f"зависимость-CUSUM ({name}): соло {ts_auc(p, yf, sf):.4f} | Spearman с #39 {sp(p, s39):.3f}, с частотным {sp(p, fq):.3f}, со связкой {sp(p, un):.3f}, с массой {sp(p, mass):.3f} [{time.time()-t0:.0f}s]", flush=True)
p = res["медленный"]
print(f"#39: {ts_auc(s39, yf, sf):.4f}")
for w in (0.10, 0.15, 0.20, 0.25):
    print(f"  #39 + член зависимости доля {w:.2f}: {ts_auc((1 - w) * s39 + w * p, yf, sf):.4f}")
print("--- по диапазонам шагов: #39 против #39 + 0.15·зависимость ---")
for a, b in ((0, 30), (30, 100), (100, 300), (300, 700), (700, 3000)):
    m = (sf >= a) & (sf < b)
    print(f"  шаги {a}-{b}: {ts_auc(s39[m], yf[m], sf[m]):.4f} -> {ts_auc(0.85 * s39[m] + 0.15 * p[m], yf[m], sf[m]):.4f}")
# один классификатор на зависимость + частоту (семья зависимости, 123 канала)
F = np.hstack([np.load("MSPEC32.npy"), np.load("MSPEC60.npy"), np.load("ACF8.npy"), D])
pf = lgb.LGBMClassifier(**slow).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]; np.save("fold2_clf_freqdep_dep.npy", pf)
print(f"частота+зависимость-CUSUM одним классификатором: соло {ts_auc(pf, yf, sf):.4f} (частотный один: {ts_auc(fq, yf, sf):.4f}) | Spearman с #39 {sp(pf, s39):.3f}")
for wm, wf in ((0.25, 0.20), (0.25, 0.25), (0.20, 0.30)):
    ref2 = (1 - wm - wf) * base + wm * mass + wf * pf; late2 = (1 - wm - 0.25 - wf * 0.5) * base + wm * mass + 0.25 * un + wf * 0.5 * pf
    print(f"  вместо частотного члена (масса {wm}, доля {wf}): без гейта {ts_auc(ref2, yf, sf):.4f} | с гейтом и связкой {ts_auc(np.where(sf < 100, ref2, late2), yf, sf):.4f}")
print(f"готово [{time.time()-t0:.0f}s]")
