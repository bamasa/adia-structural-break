"""155: the whitened member's learners trained with the boundary augmentation (one cut of three, fold-2 parents excluded).
The augmentation moves breaks early -- where the whitened member is weakest (0.586 alone on steps 100-300)."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time(); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
s39 = np.where(sf < 100, 0.55 * base + 0.25 * mass + 0.20 * fq, 0.50 * base + 0.25 * mass + 0.25 * un)
r0 = np.load("fold2_rank_white_sr.npy"); r1 = np.load("fold2_rank_white_ctx.npy"); c1 = np.load("fold2_clf_white_ctx.npy")
sig = lambda a: 1.0 / (1.0 + np.exp(-a)); pool = sum(sig(np.load(f"nets_white_w/fold2_logits_w{i}.npy")) for i in range(3)) / 3
s44 = 0.85 * (0.6 * s39 + 0.4 * (0.7 * 0.5 * (r0 + r1) + 0.3 * c1)) + 0.15 * pool
# originals (fold != 2) + one cut per parent from the triple augmentation (parents in fold != 2)
W = np.hstack([np.load("WHITE90.npy"), np.load("SR22.npy")]); Dtr, Dte = W[tr], W[te]; del W
AG = np.load("AUG3_G.npy"); AS = np.load("AUG3_S.npy"); AY = np.load("AUG3_Y.npy")
fold_by_sid = dict(zip(g[np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))], f[np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))]))
sid = (AG - 100000) // 10; cut = AG % 10
keep = np.flatnonzero((cut == 0) & np.array([fold_by_sid.get(int(x), 0) != 2 for x in sid]))
AW = np.load("AUG3_WHITE111.npy", mmap_mode="r"); Atr = np.asarray(AW[keep]); del AW
Xtr = np.vstack([Dtr, Atr]); ytr = np.concatenate([y[tr], AY[keep]]); str_ = np.concatenate([s[tr], AS[keep]]); del Dtr, Atr
print(f"обучение: {tr.sum()} оригинальных строк + {len(keep)} аугментированных = {len(ytr)} [{time.time()-t0:.0f}s]", flush=True)
slow = dict(n_estimators=3000, learning_rate=0.0075, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=500, reg_lambda=10.0, verbose=-1, n_jobs=8)
import os
if os.path.exists("fold2_clf_white_aug.npy"): c_aug = np.load("fold2_clf_white_aug.npy")
else: c_aug = lgb.LGBMClassifier(**slow).fit(Xtr, ytr).predict_proba(Dte)[:, 1]; np.save("fold2_clf_white_aug.npy", c_aug)
c0 = np.load("fold2_clf_white_sr_slower.npy")
print(f"клф с аугментацией: соло {ts_auc(c_aug, yf, sf):.4f} (без {ts_auc(c0, yf, sf):.4f}) [{time.time()-t0:.0f}s]", flush=True)
# lambdarank caps a query at 10,000 rows; with the augmentation a step's cross-section exceeds it,
# so every step is split into random chunks of at most 9,000 rows, each its own query.
rng = np.random.default_rng(0); key = str_.astype("int64") * 8
for t_ in np.unique(str_):
    idx = np.flatnonzero(str_ == t_); k = int(np.ceil(len(idx) / 9000))
    if k > 1: key[idx] += rng.integers(0, k, size=len(idx))
order = np.argsort(key, kind="stable"); _, sizes = np.unique(key[order], return_counts=True)
rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0, lambdarank_truncation_level=2000, label_gain=[0, 1], verbose=-1, n_jobs=8)
rk.fit(Xtr[order], ytr[order], group=sizes); r_aug = sig(rk.predict(Dte)); np.save("fold2_rank_white_aug.npy", r_aug)
print(f"ранкер с аугментацией: соло {ts_auc(r_aug, yf, sf):.4f} (без {ts_auc(r0, yf, sf):.4f}) [{time.time()-t0:.0f}s]", flush=True)
print(f"#44-рецепт: {ts_auc(s44, yf, sf):.4f}")
for name, rb, cc in (("бэг r0+r1 / c1 (#44)", 0.5 * (r0 + r1), c1), ("бэг r0+r1+r_aug / c1", (r0 + r1 + r_aug) / 3, c1), ("бэг r0+r1+r_aug / c_aug", (r0 + r1 + r_aug) / 3, c_aug), ("r_aug / c_aug", r_aug, c_aug)):
    m = 0.7 * rb + 0.3 * cc; b = 0.85 * (0.6 * s39 + 0.4 * m) + 0.15 * pool
    print(f"  {name}: член {ts_auc(m, yf, sf):.4f} | смесь {ts_auc(b, yf, sf):.4f}")
print("--- по диапазонам: #44 -> бэг с аугментацией ---")
m = 0.7 * (r0 + r1 + r_aug) / 3 + 0.3 * c_aug; b = 0.85 * (0.6 * s39 + 0.4 * m) + 0.15 * pool
for a, bb in ((0, 30), (30, 100), (100, 300), (300, 700), (700, 3000)):
    mm = (sf >= a) & (sf < bb); print(f"  шаги {a}-{bb}: {ts_auc(s44[mm], yf[mm], sf[mm]):.4f} -> {ts_auc(b[mm], yf[mm], sf[mm]):.4f}")
print(f"готово [{time.time()-t0:.0f}s]")
