"""022b: lambdarank OOF (for the ensemble) and the final ranker (for the ship)."""
import sys, time, os
sys.path.insert(0, "repo/src")
import numpy as np, joblib
import lightgbm as lgb
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
PARAMS = dict(objective="lambdarank", learning_rate=0.05, num_leaves=31,
              min_child_samples=500, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=10.0, n_estimators=300,
              random_state=0, n_jobs=8, deterministic=True, force_row_wise=True,
              verbose=-1, lambdarank_truncation_level=2000, label_gain=[0, 1])

def fit(tr_mask):
    Xtr, ytr, str_ = X[tr_mask], y[tr_mask], s[tr_mask]
    order = np.argsort(str_, kind="stable")
    Xtr, ytr, str_ = Xtr[order], ytr[order], str_[order]
    _, sizes = np.unique(str_, return_counts=True)
    r = lgb.LGBMRanker(**PARAMS)
    r.fit(Xtr, ytr, group=sizes)
    return r

oof = np.zeros(len(y), dtype="float32")
for fold in range(5):
    tr, va = assignment != fold, assignment == fold
    r = fit(tr)
    oof[va] = r.predict(X[va])
    print(f"  фолд {fold}: {ts_auc(oof[va], y[va], s[va]):.4f}  [{time.time()-t0:.0f}s]", flush=True)
np.save("oof_rank.npy", oof)
final = fit(np.ones(len(y), dtype=bool))
art = joblib.load("resources013/model.joblib")
os.makedirs("resources022", exist_ok=True)
joblib.dump({"booster": final, "forecaster": art["forecaster"], "ranker": True},
            "resources022/model.joblib")
print(f"готово {time.time()-t0:.0f}s", flush=True)
