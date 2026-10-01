"""040 (fold 0): stacking v2 — a meta-ranker over the members' clean predictions."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import lightgbm as lgb
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
P = np.load("member_rank_preds.npy").astype("float64")  # drop1..drop4
clf = np.load("oof_cfg5.npy").astype("float64")
rnk = np.load("oof_rank.npy").astype("float64")

# Clean bagging score on every row: for a row of fold d — the prediction
# of the drop-d member; for fold 0 — the mean of all four (none of them saw it).
sig = 1.0 / (1.0 + np.exp(-P))
bag_clean = np.zeros(len(y))
for d in (1, 2, 3, 4):
    m = assignment == d
    bag_clean[m] = sig[m, d - 1]
m0 = assignment == 0
bag_clean[m0] = sig[m0].mean(axis=1)

# Meta-features: clean bagging, classifier OOF, single-ranker OOF,
# spread of the members' opinions (disagreement is a signal in itself).
spread = sig.std(axis=1)
F = np.column_stack([bag_clean, clf, 1.0/(1.0+np.exp(-rnk)), spread]).astype("float32")

tr, va = assignment != 0, assignment == 0
yf, sf = y[va], s[va]
Ftr, ytr, str_ = F[tr], y[tr], s[tr]
order = np.argsort(str_, kind="stable")
Ftr, ytr, str_ = Ftr[order], ytr[order], str_[order]
_, sizes = np.unique(str_, return_counts=True)
meta = lgb.LGBMRanker(
    objective="lambdarank", learning_rate=0.05, num_leaves=15,
    min_child_samples=1000, subsample=0.8, subsample_freq=1,
    reg_lambda=10.0, n_estimators=200, random_state=0, n_jobs=8,
    deterministic=True, force_row_wise=True, verbose=-1,
    lambdarank_truncation_level=2000, label_gain=[0, 1])
meta.fit(Ftr, ytr, group=sizes)
mp = meta.predict(F[va])
net_ens = np.load("tcn_foldens_fold0.npy").astype("float64")
msig = 1.0/(1.0+np.exp(-mp))
print(f"meta-ranker (tree leg): {ts_auc(mp, yf, sf):.4f}", flush=True)
for w in (0.4, 0.5, 0.6):
    print(f"meta + {w:.0%} nets: {ts_auc((1-w)*msig + w*net_ens, yf, sf):.4f} (reference 0.6099)", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
