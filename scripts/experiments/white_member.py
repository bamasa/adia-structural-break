"""145: the whitened-stream battery as its own member on fold 2."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
D = np.load("WHITE90.npy"); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy"); dep = np.load("fold2_clf_dep_slow.npy")
ref = 0.55 * base + 0.25 * mass + 0.20 * fq; late = 0.50 * base + 0.25 * mass + 0.25 * un; s39 = np.where(sf < 100, ref, late)
s40 = np.where(sf < 300, s39, 0.8 * s39 + 0.2 * dep)
sp = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
teachers = {"slow": dict(n_estimators=1500, learning_rate=0.015, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=300, verbose=-1, n_jobs=8),
            "medium": dict(n_estimators=1000, learning_rate=0.03, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=200, verbose=-1, n_jobs=8)}
Dtr, Dte = D[tr], D[te]; del D
res = {}
for name, prm in teachers.items():
    p = lgb.LGBMClassifier(**prm).fit(Dtr, y[tr]).predict_proba(Dte)[:, 1]; res[name] = p; np.save(f"fold2_clf_white_{name}.npy", p)
    print(f"отбелённый член ({name}): соло {ts_auc(p, yf, sf):.4f} | Spearman с #39 {sp(p, s39):.3f}, с ядром {sp(p, base):.3f}, с массой {sp(p, mass):.3f}, с частотным {sp(p, fq):.3f}, с зависимостью {sp(p, dep):.3f} [{time.time()-t0:.0f}s]", flush=True)
p = res["slow"] if ts_auc(res["slow"], yf, sf) >= ts_auc(res["medium"], yf, sf) else res["medium"]
print(f"#39: {ts_auc(s39, yf, sf):.4f} | #40: {ts_auc(s40, yf, sf):.4f}")
for w in (0.15, 0.25, 0.35, 0.45, 0.55):
    print(f"  #39 + отбелённый доля {w:.2f}: {ts_auc((1 - w) * s39 + w * p, yf, sf):.4f} | #40 + доля {w:.2f}: {ts_auc((1 - w) * s40 + w * p, yf, sf):.4f}", flush=True)
print("--- по диапазонам шагов: #39 -> #39 + 0.35·отбелённый ---")
for a, b in ((0, 30), (30, 100), (100, 300), (300, 700), (700, 3000)):
    m = (sf >= a) & (sf < b); print(f"  шаги {a}-{b}: {ts_auc(s39[m], yf[m], sf[m]):.4f} -> {ts_auc(0.65 * s39[m] + 0.35 * p[m], yf[m], sf[m]):.4f} (член сам {ts_auc(p[m], yf[m], sf[m]):.4f})")
print("--- без ядра: только члены ---")
print(f"  масса+частота+связка+зависимость+отбелённый (равные доли): {ts_auc((mass + fq + un + dep + p) / 5, yf, sf):.4f}")
print(f"готово [{time.time()-t0:.0f}s]", flush=True)
