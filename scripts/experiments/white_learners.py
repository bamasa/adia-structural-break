"""148: stronger learners for the whitened member — a per-step ranker and a slower classifier, blended."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time(); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
s39 = np.where(sf < 100, 0.55 * base + 0.25 * mass + 0.20 * fq, 0.50 * base + 0.25 * mass + 0.25 * un)
wh = np.load("fold2_clf_white_slow.npy")
D = np.hstack([np.load("WHITE90.npy"), np.load("SR22.npy")]); Dtr, Dte = D[tr], D[te]; ytr, str_ = y[tr], s[tr]; del D
sp = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
print(f"отбелённый (slow, 90 каналов): соло {ts_auc(wh, yf, sf):.4f} | #39 + 0.30: {ts_auc(0.7 * s39 + 0.3 * wh, yf, sf):.4f}", flush=True)
# 1. slower, more regularised classifier on 90+SR
p_slower = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.0075, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=500, reg_lambda=10.0, verbose=-1, n_jobs=8).fit(Dtr, ytr).predict_proba(Dte)[:, 1]
np.save("fold2_clf_white_sr_slower.npy", p_slower)
print(f"ещё медленнее (3000 деревьев, lr 0.0075, min_child 500, L2 10): соло {ts_auc(p_slower, yf, sf):.4f} | #39 + 0.30: {ts_auc(0.7 * s39 + 0.3 * p_slower, yf, sf):.4f} [{time.time()-t0:.0f}s]", flush=True)
# 2. per-step lambdarank on 90+SR
order = np.argsort(str_, kind="stable"); _, sizes = np.unique(str_[order], return_counts=True)
rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0, lambdarank_truncation_level=2000, label_gain=[0, 1], verbose=-1, n_jobs=8)
rk.fit(Dtr[order], ytr[order], group=sizes); r = 1.0 / (1.0 + np.exp(-rk.predict(Dte))); np.save("fold2_rank_white_sr.npy", r)
print(f"ранкер по шагам: соло {ts_auc(r, yf, sf):.4f} | Spearman с классификатором {sp(r, p_slower):.3f} | #39 + 0.30: {ts_auc(0.7 * s39 + 0.3 * r, yf, sf):.4f} [{time.time()-t0:.0f}s]", flush=True)
for a in (0.3, 0.5, 0.7):
    m = a * r + (1 - a) * p_slower
    print(f"  член = {a:.1f}·ранкер + {1-a:.1f}·клф: соло {ts_auc(m, yf, sf):.4f} | #39 + 0.30: {ts_auc(0.7 * s39 + 0.3 * m, yf, sf):.4f} | + 0.40: {ts_auc(0.6 * s39 + 0.4 * m, yf, sf):.4f}")
print(f"готово [{time.time()-t0:.0f}s]")
