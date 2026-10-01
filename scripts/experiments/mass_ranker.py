"""114b: ranker on MASS90 as an independent member + fine weight tuning."""
import sys, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
M = np.load("MASS90.npy")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
ytr, str_ = y[tr], s[tr]
rng = np.random.default_rng(7); MAXQ = 8000
chunk = np.empty(len(str_), dtype="int64")
for st in np.unique(str_):
    idx = np.flatnonzero(str_ == st); rng.shuffle(idx); nc = int(np.ceil(len(idx) / MAXQ))
    for c in range(nc): chunk[idx[c::nc]] = int(st) * 10 + c
order = np.argsort(chunk, kind="stable"); _, sizes = np.unique(chunk[order], return_counts=True)
r = lgb.LGBMRanker(objective="lambdarank", learning_rate=0.03, num_leaves=63, min_child_samples=300, subsample=0.7,
                   subsample_freq=1, colsample_bytree=0.5, reg_lambda=20.0, n_estimators=600, random_state=7, n_jobs=8,
                   deterministic=True, force_row_wise=True, verbose=-1, lambdarank_truncation_level=2000, label_gain=[0, 1])
Mtr = M[tr]; r.fit(Mtr[order], ytr[order], group=sizes)
rs = r.predict(M[te]); np.save("fold2_rank_mass.npy", rs)
import joblib; joblib.dump(r, "rank_mass90.joblib")
sig = lambda a: 1/(1+np.exp(-np.asarray(a, dtype="float64")))
rank_mass = sig(rs)
clf_mass = np.load("fold2_clf_mass_90 .npy") if False else np.load([p for p in __import__("glob").glob("fold2_clf_mass_90*.npy")][0])
base = np.load("fold2_base30.npy")
print(f"ranker on MASS90 alone: {ts_auc(rank_mass, yf, sf):.4f} | classifier on MASS90: {ts_auc(clf_mass, yf, sf):.4f} [{time.time()-t0:.0f}s]")
print(f"correlation with the ensemble: ranker {pd.Series(rank_mass).corr(pd.Series(base), method='spearman'):.3f}, classifier {pd.Series(clf_mass).corr(pd.Series(base), method='spearman'):.3f}")
mass_pair = 0.7 * rank_mass + 0.3 * clf_mass
print(f"MASS pair (0.7 ranker + 0.3 clf): {ts_auc(mass_pair, yf, sf):.4f}")
print(f"\nbaseline {ts_auc(base, yf, sf):.4f}; tuning the MASS member share:")
best = (0, None)
for name, member in (("classifier", clf_mass), ("ranker", rank_mass), ("pair", mass_pair)):
    for w in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
        v = ts_auc((1 - w) * base + w * member, yf, sf)
        if v > best[0]: best = (v, (name, w))
        print(f"  {name:13s} share {w:.2f}: {v:.4f}  ({v - ts_auc(base, yf, sf):+.4f})")
print(f"\nBEST: {best[0]:.4f} — {best[1]}")
