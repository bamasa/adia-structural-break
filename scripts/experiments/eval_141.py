"""141: связка частота+новизна с исправленным каналом медианного расстояния (KNN20B) на фолде 2."""
import sys, glob, time, numpy as np, lightgbm as lgb, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
KB = np.load("KNN20B.npy"); KA = np.load("KNN20.npy")
print("p_med уникальных (B):", [len(np.unique(KB[:, 2 + 5 * i])) for i in range(4)], "| прочие каналы совпадают с KNN20:", float(np.abs(np.delete(KB, [2, 7, 12, 17], 1) - np.delete(KA, [2, 7, 12, 17], 1)).max()))
F = np.hstack([np.load("MSPEC32.npy"), np.load("MSPEC60.npy"), np.load("ACF8.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); tr, te = f != 2, f == 2; yf, sf = y[te], s[te]
slow = dict(n_estimators=1500, learning_rate=0.015, num_leaves=31, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=300, verbose=-1, n_jobs=8)
pk = lgb.LGBMClassifier(**slow).fit(KB[tr], y[tr]).predict_proba(KB[te])[:, 1]; np.save("fold2_clf_knnB_slow.npy", pk)
old_k = np.load("fold2_clf_knn_slow.npy")
print(f"новизна сама: исправленная {ts_auc(pk, yf, sf):.4f} | прежняя {ts_auc(old_k, yf, sf):.4f} [{time.time()-t0:.0f}s]", flush=True)
D = np.hstack([F, KB])
pu = lgb.LGBMClassifier(**slow).fit(D[tr], y[tr]).predict_proba(D[te])[:, 1]; np.save("fold2_clf_freqdep_knnB.npy", pu)
old_u = np.load("fold2_clf_freqdep_knn.npy")
base = np.load("fold2_base30.npy"); mass = np.load(glob.glob("fold2_clf_mass_90*.npy")[0]); fq = np.load("fold2_clf_freqdep_1500.npy")
ref = 0.55 * base + 0.25 * mass + 0.20 * fq
print(f"связка сама: исправленная {ts_auc(pu, yf, sf):.4f} | прежняя {ts_auc(old_u, yf, sf):.4f} | Spearman между ними {pd.Series(pu).corr(pd.Series(old_u), method='spearman'):.3f}")
print(f"#38: {ts_auc(ref, yf, sf):.4f}")
for name, u in (("прежняя (#39)", old_u), ("исправленная", pu)):
    for M, U in ((0.25, 0.25), (0.20, 0.25), (0.20, 0.30)):
        late = (1 - M - U) * base + M * mass + U * u
        print(f"  {name:14s} mass {M:.2f} union {U:.2f}: flat {ts_auc(late, yf, sf):.4f} | gate<100 {ts_auc(np.where(sf < 100, ref, late), yf, sf):.4f}")
print(f"готово [{time.time()-t0:.0f}s]")
