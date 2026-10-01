"""157e: the t-PIT whitening as a replacement of WHITE90 in the member's classifier, full slow recipe."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time(); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
r0 = np.load("fold2_rank_white_sr.npy"); r1 = np.load("fold2_rank_white_ctx.npy"); c1 = np.load("fold2_clf_white_ctx.npy")
sig = lambda a: 1.0 / (1.0 + np.exp(-a)); pool = sum(sig(np.load(f"nets_white_w/fold2_logits_w{i}.npy")) for i in range(3)) / 3
s39 = np.where(sf < 100, 0.55 * base + 0.25 * mass + 0.20 * fq, 0.50 * base + 0.25 * mass + 0.25 * un)
def blend(c): return ts_auc(0.85 * (0.6 * s39 + 0.4 * (0.7 * 0.5 * (r0 + r1) + 0.3 * c)) + 0.15 * pool, yf, sf)
D = np.hstack([np.load("SCREEN_TPIT.npy"), np.load("SR22.npy"), np.load("HISTCTX8.npy")]); Dtr, Dte = D[tr], D[te]; del D
p = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.0075, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=500, reg_lambda=10.0, verbose=-1, n_jobs=8).fit(Dtr, y[tr]).predict_proba(Dte)[:, 1]
np.save("fold2_clf_white_tpit.npy", p); sp = pd.Series(p).corr(pd.Series(c1), method="spearman")
print(f"t-PIT: alone {ts_auc(p, yf, sf):.4f} (ECDF {ts_auc(c1, yf, sf):.4f}) | Spearman {sp:.3f} | blend: c1 {blend(c1):.4f} → replacement {blend(p):.4f}, 0.5c1+0.5new {blend(0.5 * c1 + 0.5 * p):.4f} [{time.time()-t0:.0f}s]")
for a, bb in ((0, 100), (100, 300), (300, 700), (700, 3000)):
    m = (sf >= a) & (sf < bb); print(f"  steps {a}-{bb}: alone {ts_auc(c1[m], yf[m], sf[m]):.4f} → {ts_auc(p[m], yf[m], sf[m]):.4f}")
