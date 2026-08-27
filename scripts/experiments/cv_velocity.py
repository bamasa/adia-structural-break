"""034 (фолд-0): скорость каналов — приращения топ-каналов за 10 и 30 шагов."""
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
# Топ-каналы по gain из прунинга 024.
TOP = [5, 82, 58, 102, 8, 74, 72, 52, 22, 78, 2, 81]
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
bounds = np.append(starts, len(g))
vel = np.zeros((len(g), len(TOP) * 2), dtype="float32")
for a, b in zip(starts, bounds[1:]):
    seg = X[a:b][:, TOP]
    for j, lag in enumerate((10, 30)):
        d = np.zeros_like(seg)
        if b - a > lag:
            d[lag:] = seg[lag:] - seg[:-lag]
        vel[a:b, j * len(TOP):(j + 1) * len(TOP)] = d
print(f"скоростных признаков: {vel.shape[1]}, подготовка {time.time()-t0:.0f}s", flush=True)

X2 = np.hstack([X, vel])
assignment = split_by_series(g, folds=5, seed=0)
tr, va = assignment != 0, assignment == 0
yf, sf = y[va], s[va]
clf = np.load("oof_cfg5.npy")[va].astype("float64")
Xtr, ytr, str_ = X2[tr], y[tr], s[tr]
order = np.argsort(str_, kind="stable")
Xtr, ytr, str_ = Xtr[order], ytr[order], str_[order]
_, sizes = np.unique(str_, return_counts=True)
r = lgb.LGBMRanker(
    objective="lambdarank", learning_rate=0.03, num_leaves=31,
    min_child_samples=500, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=10.0, n_estimators=600,
    random_state=0, n_jobs=8, deterministic=True, force_row_wise=True,
    verbose=-1, lambdarank_truncation_level=2000, label_gain=[0, 1])
r.fit(Xtr, ytr, group=sizes)
sc = r.predict(X2[va])
sig = 1.0 / (1.0 + np.exp(-sc))
print(f"210 (186+скорости): соло {ts_auc(sc, yf, sf):.4f}, "
      f"смесь {ts_auc(0.6*sig+0.4*clf, yf, sf):.4f} (эталоны 0.6016 / 0.6045)", flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
