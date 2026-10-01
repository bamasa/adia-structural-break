"""153b: the per-step ranker with the history context appended (119 inputs)."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time(); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
s39 = np.where(sf < 100, 0.55 * base + 0.25 * mass + 0.20 * fq, 0.50 * base + 0.25 * mass + 0.25 * un)
r0 = np.load("fold2_rank_white_sr.npy"); c0 = np.load("fold2_clf_white_sr_slower.npy"); c1 = np.load("fold2_clf_white_ctx.npy")
D = np.hstack([np.load("WHITE90.npy"), np.load("SR22.npy"), np.load("HISTCTX8.npy")]); Dtr, Dte = D[tr], D[te]; del D
str_ = s[tr]; order = np.argsort(str_, kind="stable"); _, sizes = np.unique(str_[order], return_counts=True)
rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0, lambdarank_truncation_level=2000, label_gain=[0, 1], verbose=-1, n_jobs=6)
rk.fit(Dtr[order], y[tr][order], group=sizes); r1 = 1.0 / (1.0 + np.exp(-rk.predict(Dte))); np.save("fold2_rank_white_ctx.npy", r1)
sp = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
print(f"ranker + context: alone {ts_auc(r1, yf, sf):.4f} (without context {ts_auc(r0, yf, sf):.4f}) | Spearman {sp(r1, r0):.3f} [{time.time()-t0:.0f}s]")
for rn, r in (("ranker w/o ctx", r0), ("ranker with ctx", r1)):
    for cn, c in (("clf w/o ctx", c0), ("clf with ctx", c1)):
        m = 0.7 * r + 0.3 * c; print(f"  {rn} + {cn}: #39 + 0.40 → {ts_auc(0.6 * s39 + 0.4 * m, yf, sf):.4f} | + 0.45 → {ts_auc(0.55 * s39 + 0.45 * m, yf, sf):.4f}")
print(f"done [{time.time()-t0:.0f}s]")
