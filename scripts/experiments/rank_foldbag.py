"""039 (фолд-0): фолд-бэггинг ранкеров — разнообразие через данные."""
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
va = assignment == 0
yf, sf = y[va], s[va]
clf = np.load("oof_cfg5.npy")[va].astype("float64")

sigs = []
for drop in (1, 2, 3, 4):
    tr = (assignment != 0) & (assignment != drop)
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
    sc = r.predict(X[va])
    print(f"ранкер без фолда {drop}: соло {ts_auc(sc, yf, sf):.4f}  [{time.time()-t0:.0f}s]", flush=True)
    sigs.append(1.0 / (1.0 + np.exp(-sc)))
ens = np.mean(sigs, axis=0)
print(f"фолд-бэггинг ранкеров: соло {ts_auc(ens, yf, sf):.4f} (одиночный: 0.6016)", flush=True)
np.save("rank_foldbag_fold0.npy", ens)
net_ens = np.load("tcn_foldens_fold0.npy").astype("float64")
for w_net in (0.4, 0.5):
    for w_clf in (0.15, 0.2):
        mix = (1 - w_net - w_clf) * ens + w_clf * clf + w_net * net_ens
        print(f"тройка bag-ранкер {1-w_net-w_clf:.2f}/клф {w_clf:.2f}/сети {w_net:.2f}: "
              f"{ts_auc(mix, yf, sf):.4f} (эталон #19: 0.6090)", flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
