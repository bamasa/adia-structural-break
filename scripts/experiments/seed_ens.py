"""027 (фолд-0): сид-ансамбль ранкеров + смесь с классификатором."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import lightgbm as lgb
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
tr, va = assignment != 0, assignment == 0
Xtr, ytr, str_ = X[tr], y[tr], s[tr]
order = np.argsort(str_, kind="stable")
Xtr, ytr, str_ = Xtr[order], ytr[order], str_[order]
_, sizes = np.unique(str_, return_counts=True)
yf, sf = y[va], s[va]

sigs = []
for seed in (0, 1, 2):
    r = lgb.LGBMRanker(
        objective="lambdarank", learning_rate=0.05, num_leaves=31,
        min_child_samples=500, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=10.0, n_estimators=300,
        random_state=seed, n_jobs=8, deterministic=True, force_row_wise=True,
        verbose=-1, lambdarank_truncation_level=2000, label_gain=[0, 1])
    r.fit(Xtr, ytr, group=sizes)
    sc = r.predict(X[va])
    sigs.append(1.0 / (1.0 + np.exp(-sc)))
    print(f"сид {seed}: {ts_auc(sc, yf, sf):.4f}  [{time.time()-t0:.0f}s]", flush=True)

mean_sig = np.mean(sigs, axis=0)
print(f"ансамбль 3 сидов: {ts_auc(mean_sig, yf, sf):.4f}", flush=True)
clf = np.load("oof_cfg5.npy")[va].astype("float64")
for w in (0.5, 0.6, 0.7):
    print(f"смесь {w:.0%} сид-ансамбля + классификатор: "
          f"{ts_auc(w * mean_sig + (1 - w) * clf, yf, sf):.4f}", flush=True)
print(f"(эталон текущей посылки #15: 0.6035)", flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
