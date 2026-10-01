"""024: importance-based pruning of the 186 channels (gain importances)."""
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
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=500,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0,
              n_estimators=300, random_state=0, n_jobs=4, deterministic=True,
              force_row_wise=True, verbose=-1)

# Importances from fold 0 (we do not look at its validation during selection — selection by gain).
tr = assignment != 0
m = lgb.LGBMClassifier(**PARAMS)
m.fit(X[tr], y[tr], sample_weight=step_weights(s[tr]))
imp = m.booster_.feature_importance(importance_type="gain")
order = np.argsort(imp)[::-1]
print("top-15 channels by gain:", order[:15].tolist(), flush=True)
print("bottom-15 (candidates for removal):", order[-15:].tolist(), flush=True)

for keep in (150, 120, 90):
    cols = np.sort(order[:keep])
    scores = []
    for fold in range(3):
        tr, va = assignment != fold, assignment == fold
        mm = lgb.LGBMClassifier(**PARAMS)
        mm.fit(X[tr][:, cols], y[tr], sample_weight=step_weights(s[tr]))
        scores.append(ts_auc(mm.predict_proba(X[va][:, cols])[:, 1], y[va], s[va]))
    print(f"top-{keep}: {np.mean(scores):.4f}  " + " ".join(f"{v:.4f}" for v in scores)
          + f"  [{time.time()-t0:.0f}s]", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
