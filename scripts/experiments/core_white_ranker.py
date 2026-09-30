"""156: a per-step ranker over the core's 206 channels and the whitened 111 together, with the boundary
augmentation (one cut of three) -- the trees see both families at once and can read their interactions."""
import sys, glob, time, os, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time(); y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
CORE = ["X40.npy", "C_cnn.npy", "N9.npy", "X50a.npy", "E4.npy", "B40.npy", "B2.npy", "SPEC14.npy", "BOCPD6_H50.npy", "WHITE90.npy", "SR22.npy"]
def stack(names, rows):
    parts = []
    for n in names:
        m = np.load(n, mmap_mode="r"); b = np.asarray(m[rows]).astype("float32"); parts.append(b[:, None] if b.ndim == 1 else b)
    return np.hstack(parts)
tr_idx, te_idx = np.flatnonzero(tr), np.flatnonzero(te)
AG = np.load("AUG3_G.npy"); AS = np.load("AUG3_S.npy"); AY = np.load("AUG3_Y.npy")
st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); fold_by_sid = dict(zip(g[st], f[st]))
sid = (AG - 100000) // 10; cut = AG % 10
keep = np.flatnonzero((cut == 0) & np.array([fold_by_sid.get(int(x), 0) != 2 for x in sid]))
Xtr = np.vstack([stack(CORE, tr_idx), np.hstack([stack(["AUG3_X.npy", "AUG3_BOCPD6_H50.npy"], keep), stack(["AUG3_WHITE111.npy"], keep)])])
Dte = stack(CORE, te_idx); ytr = np.concatenate([y[tr], AY[keep]]); str_ = np.concatenate([s[tr], AS[keep]])
print(f"обучение {Xtr.shape}, тест {Dte.shape} [{time.time()-t0:.0f}s]", flush=True)
rng = np.random.default_rng(0); key = str_.astype("int64") * 8
for t_ in np.unique(str_):
    idx = np.flatnonzero(str_ == t_); k = int(np.ceil(len(idx) / 9000))
    if k > 1: key[idx] += rng.integers(0, k, size=len(idx))
order = np.argsort(key, kind="stable"); _, sizes = np.unique(key[order], return_counts=True)
rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0, lambdarank_truncation_level=2000, label_gain=[0, 1], verbose=-1, n_jobs=8)
rk.fit(Xtr[order], ytr[order], group=sizes); del Xtr
r = 1.0 / (1.0 + np.exp(-rk.predict(Dte))); np.save("fold2_rank_core_white.npy", r)
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy"); un = np.load("fold2_clf_freqdep_knn.npy")
s39 = np.where(sf < 100, 0.55 * base + 0.25 * mass + 0.20 * fq, 0.50 * base + 0.25 * mass + 0.25 * un)
r0 = np.load("fold2_rank_white_sr.npy"); r1 = np.load("fold2_rank_white_ctx.npy"); c1 = np.load("fold2_clf_white_ctx.npy")
sig = lambda a: 1.0 / (1.0 + np.exp(-a)); pool = sum(sig(np.load(f"nets_white_w/fold2_logits_w{i}.npy")) for i in range(3)) / 3
s44 = 0.85 * (0.6 * s39 + 0.4 * (0.7 * 0.5 * (r0 + r1) + 0.3 * c1)) + 0.15 * pool
sp = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")
print(f"ранкер ядро+отбелённые (317, с аугментацией): соло {ts_auc(r, yf, sf):.4f} | Spearman с ядром {sp(r, base):.3f}, с отбелённым ранкером {sp(r, r0):.3f} | #44-рецепт {ts_auc(s44, yf, sf):.4f} [{time.time()-t0:.0f}s]")
for w in (0.10, 0.20, 0.30, 0.40):
    print(f"  #44 + доля {w:.2f}: {ts_auc((1 - w) * s44 + w * r, yf, sf):.4f}")
print(f"готово [{time.time()-t0:.0f}s]")
