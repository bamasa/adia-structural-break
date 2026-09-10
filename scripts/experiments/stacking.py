"""099: классический стекинг (по Alphabot): блочные модели первого уровня + мета-модель.

Уровень 0: LightGBM-классификаторы, каждый на своём блоке каналов; OOF по четырём
внутренним фолдам (0,1,3,4); для фолда-2 — модель на всех четырёх.
Уровень 1: мета-модель на OOF-предсказаниях (и вариант «предсказания + признаки»).
Замер: фолд-2, TS-AUC; сравнение с полным классификатором и со смесью #30.
"""
import sys, time, os, numpy as np, lightgbm as lgb
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"), np.load("SPEC14.npy"), np.load("BOCPD6_H50.npy")]).astype("float32")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); te = f == 2; tr = ~te
BLOCKS = {"X40": range(0, 40), "cnn_now": range(40, 50), "X50a": range(50, 100), "forecast": range(100, 104),
          "battery_raw": range(104, 146), "battery_asinh": range(146, 186), "spectral": range(186, 200), "bocpd": range(200, 206),
          "full206": range(0, 206), "raw_half": list(range(0, 20)) + list(range(50, 75)) + list(range(104, 146)),
          "asinh_half": list(range(20, 40)) + list(range(75, 100)) + list(range(146, 186))}
params = dict(n_estimators=400, learning_rate=0.04, num_leaves=63, colsample_bytree=0.6, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8)
os.makedirs("stack", exist_ok=True)
inner = [0, 1, 3, 4]
P_tr = np.zeros((tr.sum(), len(BLOCKS)), dtype="float32"); P_te = np.zeros((te.sum(), len(BLOCKS)), dtype="float32")
f_tr = f[tr]; ytr = y[tr]
for j, (name, cols) in enumerate(BLOCKS.items()):
    cols = np.array(list(cols)); Xb = X[:, cols]
    cache = f"stack/oof_{name}.npz"
    if os.path.exists(cache):
        z = np.load(cache); P_tr[:, j], P_te[:, j] = z["tr"], z["te"]
        print(f"[{name}] из кэша", flush=True); continue
    oof = np.zeros(tr.sum(), dtype="float32")
    Xtr = Xb[tr]
    for k in inner:
        fit, hold = f_tr != k, f_tr == k
        m = lgb.LGBMClassifier(**params).fit(Xtr[fit], ytr[fit]); oof[hold] = m.predict_proba(Xtr[hold])[:, 1]
    m = lgb.LGBMClassifier(**params).fit(Xtr, ytr); pte = m.predict_proba(Xb[te])[:, 1]
    P_tr[:, j], P_te[:, j] = oof, pte; np.savez(cache, tr=oof, te=pte)
    print(f"[{name}] {len(cols)} каналов: OOF (фолды 0/1/3/4) {ts_auc(oof, ytr, s[tr]):.4f} | фолд-2 {ts_auc(pte, y[te], s[te]):.4f} [{time.time()-t0:.0f}s]", flush=True)
names = list(BLOCKS)
print("--- уровень 1 ---", flush=True)
sf, yf = s[te], y[te]
print(f"среднее блоков (равные веса): фолд-2 {ts_auc(P_te.mean(1), yf, sf):.4f}", flush=True)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0, max_iter=500).fit(P_tr, ytr); meta_lin = lr.decision_function(P_te)
print(f"мета линейная (11 предсказаний): фолд-2 {ts_auc(meta_lin, yf, sf):.4f}; веса {dict(zip(names, lr.coef_[0].round(2)))}", flush=True)
mp = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=500, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=8)
m = lgb.LGBMClassifier(**mp).fit(P_tr, ytr); meta_gb = m.predict_proba(P_te)[:, 1]
print(f"мета LightGBM (11 предсказаний): фолд-2 {ts_auc(meta_gb, yf, sf):.4f}", flush=True)
Ftr = np.hstack([P_tr, X[tr]]); Fte = np.hstack([P_te, X[te]])
m2 = lgb.LGBMClassifier(**dict(mp, num_leaves=31, colsample_bytree=0.5)).fit(Ftr, ytr); meta_gbf = m2.predict_proba(Fte)[:, 1]
print(f"мета LightGBM (предсказания + 206 признаков): фолд-2 {ts_auc(meta_gbf, yf, sf):.4f}", flush=True)
np.savez("stack/meta_fold2.npz", mean=P_te.mean(1), lin=meta_lin, gb=meta_gb, gbf=meta_gbf)
# в смеси #30: мета вместо классификатора
sig = lambda a: 1/(1+np.exp(-a.astype("float64")))
rank = sig(np.load("fold2_rank_bocpd_200.npy")); clf = np.load("fold2_clf_bocpdh50_206.npy").astype("float64")
net = np.mean([np.load(f"fold2_sig_nets_aug3_member_p{i}.pt.npy") for i in range(6)] + [np.load(f"fold2_sig_nets_aug3_last_member_z{i}.pt.npy") for i in range(6)], 0)
print(f"смесь #30 (0.7·ранкер+0.3·clf, сети 0.55): {ts_auc(0.45*(0.7*rank+0.3*clf)+0.55*net, yf, sf):.4f}", flush=True)
for nm, meta in (("среднее", P_te.mean(1)), ("линейная", sig(meta_lin)), ("gb", meta_gb), ("gb+признаки", meta_gbf)):
    for cw in (0.3, 0.5, 0.7):
        trees = (1-cw)*rank + cw*meta
        print(f"  мета={nm:12s} доля {cw}: деревья {ts_auc(trees, yf, sf):.4f} | смесь с сетями 0.55 {ts_auc(0.45*trees+0.55*net, yf, sf):.4f} | 0.45 {ts_auc(0.55*trees+0.45*net, yf, sf):.4f}", flush=True)
print("done", flush=True)
