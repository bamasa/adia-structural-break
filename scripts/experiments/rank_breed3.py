"""074: третья порода ранкера — глубокая, на аугментированных данных.

Гипотеза: древесная половина держится на двух близких по конфигурации
ранкерах; порода с другой геометрией (127 листьев, lr 0.02, colsample 0.7,
truncation 1000, другой сид) добавит разнообразия. Kill: смесь трёх ранкеров
не лучше смеси двух на фолде-2.
"""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import lightgbm as lgb
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
               np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
tr, va = assignment != 2, assignment == 2
yf, sf = y[va], s[va]

AX = np.load("AUG_X.npy"); AY = np.load("AUG_Y.npy")
AG = np.load("AUG_G.npy"); AS = np.load("AUG_S.npy")
orig_sid = AG - 100000
starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
fold_of = {int(g[a]): int(assignment[a]) for a in starts}
keep = np.array([fold_of.get(int(sid), 0) != 2 for sid in orig_sid])
AX, AY, AS = AX[keep], AY[keep], AS[keep]

Xtr = np.vstack([X[tr], AX]); ytr = np.concatenate([y[tr], AY])
str_ = np.concatenate([s[tr], AS])
rng = np.random.default_rng(7)
MAXQ = 8000
chunk_id = np.empty(len(str_), dtype="int64")
for st in np.unique(str_):
    idx = np.flatnonzero(str_ == st)
    rng.shuffle(idx)
    n_chunks = int(np.ceil(len(idx) / MAXQ))
    for c in range(n_chunks):
        chunk_id[idx[c::n_chunks]] = int(st) * 10 + c
order = np.argsort(chunk_id, kind="stable")
_, sizes = np.unique(chunk_id[order], return_counts=True)
print(f"строк {len(ytr)}, групп {len(sizes)} [{time.time()-t0:.0f}s]", flush=True)

r = lgb.LGBMRanker(
    objective="lambdarank", learning_rate=0.02, num_leaves=127,
    min_child_samples=300, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.7, reg_lambda=20.0, n_estimators=600,
    random_state=7, n_jobs=8, deterministic=True, force_row_wise=True,
    verbose=-1, lambdarank_truncation_level=1000, label_gain=[0, 1])
r.fit(Xtr[order], ytr[order], group=sizes)
rs = r.predict(X[va])
np.save("fold2_rank_deep.npy", rs)
print(f"глубокий ранкер соло: {ts_auc(rs, yf, sf):.4f} (rank_aug 0.6034) [{time.time()-t0:.0f}s]", flush=True)

import joblib
joblib.dump(r, "rank_deep.joblib")

# Смесь трёх ранкеров + классификатор + сети (кэш топ-10 из свипа).
r1 = 1/(1+np.exp(-np.load("fold2_rank_aug.npy").astype("float64")))
r2 = 1/(1+np.exp(-np.load("fold2_rank_aug3.npy").astype("float64")))
r3 = 1/(1+np.exp(-rs.astype("float64")))
cp = np.load("fold2_clf_эталон.npy").astype("float64")
net = np.load("fold2_net_top10.npy")
best = (0, None)
for a3 in (0.0, 0.1, 0.15, 0.2, 0.25):
    rest = 0.70 - a3
    pair = (rest/2)*r1 + (rest/2)*r2 + a3*r3 + 0.30*cp
    for w in (0.45, 0.5, 0.55):
        v = ts_auc((1-w)*pair + w*net, yf, sf)
        if v > best[0]:
            best = (v, (a3, w))
        print(f"доля породы-3 {a3:.2f}, вес сетей {w}: {v:.4f}", flush=True)
print(f"\nИТОГ: {best[0]:.4f} при {best[1]}  (рекорд без неё 0.6116)", flush=True)
