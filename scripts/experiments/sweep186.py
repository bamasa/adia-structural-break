"""021: hyperparameter sweep at 186 channels, 3 folds, then confirm best on 5."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import lightgbm as lgb
from structural_break.combiners import split_by_series, step_weights, ts_auc

t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)

BASE = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=500,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0,
            n_estimators=300, random_state=0, n_jobs=8, deterministic=True,
            force_row_wise=True, verbose=-1)
GRID = [
    dict(),                                                   # текущая
    dict(num_leaves=63),
    dict(num_leaves=63, n_estimators=500, learning_rate=0.04),
    dict(num_leaves=127, n_estimators=500, learning_rate=0.03, min_child_samples=1000),
    dict(num_leaves=31, n_estimators=600, learning_rate=0.03),
    dict(num_leaves=63, colsample_bytree=0.5),
    dict(num_leaves=63, min_child_samples=2000),
    dict(num_leaves=63, reg_lambda=30.0),
]

def run(params, folds):
    scores = []
    for fold in folds:
        tr, va = assignment != fold, assignment == fold
        w = step_weights(s[tr])
        m = lgb.LGBMClassifier(**{**BASE, **params})
        m.fit(X[tr], y[tr], sample_weight=w)
        scores.append(ts_auc(m.predict_proba(X[va])[:, 1], y[va], s[va]))
    return scores

results = []
for i, params in enumerate(GRID):
    sc = run(params, [0, 1, 2])
    results.append((float(np.mean(sc)) - float(np.std(sc)), i, params))
    print(f"конфиг {i} {params}: {np.mean(sc):.4f} (±{np.std(sc):.4f})  [{time.time()-t0:.0f}s]", flush=True)
results.sort(reverse=True)
_, best_i, best = results[0]
print(f"лучший на 3 фолдах: конфиг {best_i} {best}", flush=True)
sc = run(best, [0, 1, 2, 3, 4])
print(f"подтверждение на 5: {np.mean(sc):.4f}  " + " ".join(f"{v:.4f}" for v in sc), flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
