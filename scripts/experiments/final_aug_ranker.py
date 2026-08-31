"""Финальный ранкер для посылки: все данные + аугментация, конфиг 63/0.5."""
import sys, time, os
sys.path.insert(0, "repo/src")
import numpy as np, joblib
import lightgbm as lgb

t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy")])
y = np.load("Y40.npy"); s = np.load("S40.npy")
AX = np.load("AUG_X.npy"); AY = np.load("AUG_Y.npy"); AS = np.load("AUG_S.npy")
Xa = np.vstack([X, AX]); ya = np.concatenate([y, AY]); sa = np.concatenate([s, AS])
del X, AX
print(f"строк: {len(ya):,} [{time.time()-t0:.0f}s]", flush=True)

rng = np.random.default_rng(0)
MAXQ = 8000
chunk = np.empty(len(sa), dtype="int64")
for st in np.unique(sa):
    idx = np.flatnonzero(sa == st)
    rng.shuffle(idx)
    n_ch = int(np.ceil(len(idx) / MAXQ))
    for c in range(n_ch):
        chunk[idx[c::n_ch]] = int(st) * 10 + c
order = np.argsort(chunk, kind="stable")
_, sizes = np.unique(chunk[order], return_counts=True)
r = lgb.LGBMRanker(
    objective="lambdarank", learning_rate=0.03, num_leaves=63,
    min_child_samples=500, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.5, reg_lambda=10.0, n_estimators=600,
    random_state=0, n_jobs=8, deterministic=True, force_row_wise=True,
    verbose=-1, lambdarank_truncation_level=2000, label_gain=[0, 1])
r.fit(Xa[order], ya[order], group=sizes)
os.makedirs("resources070", exist_ok=True)
joblib.dump(r, "resources070/rank_aug.joblib")
print(f"готово {time.time()-t0:.0f}s", flush=True)
