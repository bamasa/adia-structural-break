"""097r: ранкер (конфиг #28) на оригиналах(≠2) БЕЗ аугментации: 200 против 208 (дивергенции) — парно."""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"),
               np.load("X50a.npy"), np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"),
               np.load("SPEC14.npy")]).astype("float32")
B = np.load("DIV8.npy")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
tr, va = assignment != 2, assignment == 2
yf, sf = y[va], s[va]
ytr = y[tr]; str_ = s[tr]
rng = np.random.default_rng(7); MAXQ = 8000
chunk_id = np.empty(len(str_), dtype="int64")
for st in np.unique(str_):
    idx = np.flatnonzero(str_ == st); rng.shuffle(idx)
    n_chunks = int(np.ceil(len(idx) / MAXQ))
    for c in range(n_chunks):
        chunk_id[idx[c::n_chunks]] = int(st) * 10 + c
order = np.argsort(chunk_id, kind="stable")
_, sizes = np.unique(chunk_id[order], return_counts=True)
print(f"строк {len(ytr)}, групп {len(sizes)} [{time.time()-t0:.0f}s]", flush=True)
res = {}
for name, Xtr, Xva in (("200", X[tr], X[va]),
                       ("208", np.hstack([X[tr], B[tr]]), np.hstack([X[va], B[va]]))):
    r = lgb.LGBMRanker(objective="lambdarank", learning_rate=0.03, num_leaves=63, min_child_samples=300,
                       subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=20.0, n_estimators=600,
                       random_state=7, n_jobs=8, deterministic=True, force_row_wise=True, verbose=-1,
                       lambdarank_truncation_level=2000, label_gain=[0, 1])
    r.fit(Xtr[order], ytr[order], group=sizes)
    rs = r.predict(Xva); res[name] = ts_auc(rs, yf, sf)
    np.save(f"fold2_rank_div_{name}.npy", rs)
    print(f"ранкер {name}: фолд-2 {res[name]:.4f} [{time.time()-t0:.0f}s]", flush=True)
    del Xtr
print(f"ИТОГ ранкер: прирост {res['208'] - res['200']:+.4f}", flush=True)
