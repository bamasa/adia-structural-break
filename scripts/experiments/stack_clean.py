"""040b: stacking done properly — clean members (without fold 0 and without fold d), saved."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np, joblib, os
import lightgbm as lgb
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
os.makedirs("resources040", exist_ok=True)

# Clean predictions: P[i, d-1] are valid on folds 0 and d.
P = np.full((len(y), 4), np.nan, dtype="float32")
for j, d in enumerate((1, 2, 3, 4)):
    path = f"resources040/clean_drop{d}.joblib"
    if os.path.exists(path):
        r = joblib.load(path)
    else:
        tr = (assignment != 0) & (assignment != d)
        Xtr, ytr, str_ = X[tr], y[tr], s[tr]
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
        joblib.dump(r, path)
    m = (assignment == 0) | (assignment == d)
    P[m, j] = r.booster_.predict(X[m], num_threads=8)
    print(f"clean member drop{d} done [{time.time()-t0:.0f}s]", flush=True)

sig = 1.0 / (1.0 + np.exp(-P))
clf = np.load("oof_cfg5.npy").astype("float64")
rnk_sig = 1.0 / (1.0 + np.exp(-np.load("oof_rank.npy").astype("float64")))

# Rows of fold d: the clean drop-d member; fold 0: mean and spread of all four.
bag = np.zeros(len(y)); spread = np.zeros(len(y))
for d in (1, 2, 3, 4):
    m = assignment == d
    bag[m] = sig[m, d - 1]
    spread[m] = 0.0  # train rows have a single clean member — no spread
m0 = assignment == 0
bag[m0] = np.nanmean(sig[m0], axis=1)
spread[m0] = np.nanstd(sig[m0], axis=1)
# The spread is not comparable between train and val -> NOT used as a meta-feature.
F = np.column_stack([bag, clf, rnk_sig]).astype("float32")

tr, va = assignment != 0, assignment == 0
yf, sf = y[va], s[va]
Ftr, ytr, str_ = F[tr], y[tr], s[tr]
order = np.argsort(str_, kind="stable")
Ftr, ytr, str_ = Ftr[order], ytr[order], str_[order]
_, sizes = np.unique(str_, return_counts=True)
meta = lgb.LGBMRanker(
    objective="lambdarank", learning_rate=0.05, num_leaves=7,
    min_child_samples=2000, reg_lambda=10.0, n_estimators=100,
    random_state=0, n_jobs=8, deterministic=True, force_row_wise=True,
    verbose=-1, lambdarank_truncation_level=2000, label_gain=[0, 1])
meta.fit(Ftr, ytr, group=sizes)
msig = 1.0 / (1.0 + np.exp(-meta.predict(F[va])))
net_ens = np.load("tcn_foldens_fold0.npy").astype("float64")
print(f"meta (clean), tree leg: {ts_auc(msig, yf, sf):.4f}", flush=True)
for w in (0.5,):
    print(f"meta + {w:.0%} nets: {ts_auc((1-w)*msig + w*net_ens, yf, sf):.4f} (nine-member reference 0.6099)", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
