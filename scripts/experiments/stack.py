"""029 (фолд-0): стекинг — мета-модель поверх OOF-скоров базовых моделей."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import lightgbm as lgb
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
clf = np.load("oof_cfg5.npy").astype("float64")
sig = 1.0 / (1.0 + np.exp(-np.load("oof_rank.npy").astype("float64")))
E4 = np.load("E4.npy").astype("float64")
B_slope = np.load("B2.npy")[:, [18, 19, 38, 39]].astype("float64")  # t-статистики трендов, оба вида

META = {
    "2 скора": np.column_stack([clf, sig]),
    "скоры+прогнозист": np.column_stack([clf, sig, E4]),
    "скоры+прогнозист+тренды": np.column_stack([clf, sig, E4, B_slope]),
}
tr, va = assignment != 0, assignment == 0
yf, sf = y[va], s[va]
for name, F in META.items():
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
    print(f"мета [{name}]: {ts_auc(meta.predict(F[va]), yf, sf):.4f} "
          f"(эталон смеси 60/40: 0.6035)  [{time.time()-t0:.0f}s]", flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
