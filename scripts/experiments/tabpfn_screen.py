"""042 (фолд-0): TabPFN поверх батарейных двухвыборочных векторов."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
B = np.hstack([np.load("B40.npy"), np.load("B2.npy")]).astype("float32")  # 82 признака
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
tr, va = assignment != 0, assignment == 0

# Обучающая подвыборка: TabPFN любит до ~10k строк; стратифицируем по шагам.
rng = np.random.default_rng(0)
tr_idx = np.flatnonzero(tr)
sub = rng.choice(tr_idx, size=10000, replace=False)
Xtr, ytr = B[sub], y[sub]

import os
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
from tabpfn import TabPFNClassifier
import torch
device = "cpu"
clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
clf.fit(Xtr, ytr)
print(f"обучен на 10k строк [{time.time()-t0:.0f}s]", flush=True)

# Каденс: предсказываем только там, где батарея пересчитывалась (значения
# между пересчётами всё равно повторяются), потом растягиваем удержанием.
va_idx = np.flatnonzero(va)
Bva = B[va_idx]
change = np.ones(len(va_idx), dtype=bool)
change[1:] = np.any(Bva[1:] != Bva[:-1], axis=1)
pts = np.flatnonzero(change)
print(f"каденс-точек: {len(pts)} из {len(va_idx)}", flush=True)
# Для вердикта хватит подвыборки точек: равномерно каждая третья.
pts = pts[::3]
print(f"скрин-подвыборка: {len(pts)} точек", flush=True)
pred_pts = np.zeros(len(pts), dtype="float64")
BS = 500
for k in range(0, len(pts), BS):
    pred_pts[k:k+BS] = clf.predict_proba(Bva[pts[k:k+BS]])[:, 1]
    if (k // BS) % 5 == 0:
        print(f"  {k}/{len(pts)} [{time.time()-t0:.0f}s]", flush=True)
preds = np.full(len(va_idx), 0.5, dtype="float64")
last = 0.5
pset = {int(q): k for k, q in enumerate(pts)}
for i in range(len(va_idx)):
    if i in pset:
        last = pred_pts[pset[i]]
    preds[i] = last
np.save("tabpfn_fold0.npy", preds)
yf, sf = y[va], s[va]
print(f"TabPFN соло, фолд-0: {ts_auc(preds, yf, sf):.4f}", flush=True)
clf_oof = np.load("oof_cfg5.npy")[va].astype("float64")
rnk = np.load("oof_rank.npy")[va].astype("float64")
blend = 0.6/(1+np.exp(-rnk)) + 0.4*clf_oof
net_ens = np.load("tcn_foldens_fold0.npy").astype("float64")
bag = np.load("rank_foldbag_fold0.npy").astype("float64")
full = 0.35*bag + 0.15*clf_oof + 0.5*net_ens
for w in (0.1, 0.2, 0.3):
    print(f"девятка + {w:.0%} TabPFN: {ts_auc((1-w)*full + w*preds, yf, sf):.4f} (эталон 0.6099)", flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
