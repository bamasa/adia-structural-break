"""099c: ранкер (lambdarank, конфиг #28) как член первого уровня — OOF по фолдам 0/1/3/4 на оригиналах, затем общая мета со всеми членами."""
import sys, time, os, numpy as np, lightgbm as lgb
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
from sklearn.linear_model import LogisticRegression
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"), np.load("SPEC14.npy")]).astype("float32")
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); te = f == 2; tr = ~te; f_tr = f[tr]; ytr = y[tr]; s_tr = s[tr]; inner = [0, 1, 3, 4]
def groups(str_):
    rng = np.random.default_rng(7); MAXQ = 8000; chunk_id = np.empty(len(str_), dtype="int64")
    for st in np.unique(str_):
        idx = np.flatnonzero(str_ == st); rng.shuffle(idx); n_chunks = int(np.ceil(len(idx) / MAXQ))
        for c in range(n_chunks): chunk_id[idx[c::n_chunks]] = int(st) * 10 + c
    order = np.argsort(chunk_id, kind="stable"); _, sizes = np.unique(chunk_id[order], return_counts=True); return order, sizes
def ranker():
    return lgb.LGBMRanker(objective="lambdarank", learning_rate=0.03, num_leaves=63, min_child_samples=300, subsample=0.7, subsample_freq=1,
                          colsample_bytree=0.5, reg_lambda=20.0, n_estimators=600, random_state=7, n_jobs=8, deterministic=True,
                          force_row_wise=True, verbose=-1, lambdarank_truncation_level=2000, label_gain=[0, 1])
cache = "stack/oof_ranker.npz"
if not os.path.exists(cache):
    Xtr = X[tr]; oof = np.zeros(tr.sum(), dtype="float32")
    for k in inner:
        fit, hold = f_tr != k, f_tr == k
        order, sizes = groups(s_tr[fit]); Xf, yf_ = Xtr[fit], ytr[fit]
        m = ranker().fit(Xf[order], yf_[order], group=sizes); oof[hold] = m.predict(Xtr[hold])
        print(f"[ranker] внутренний фолд {k}: {ts_auc(oof[hold], ytr[hold], s_tr[hold]):.4f} [{time.time()-t0:.0f}s]", flush=True)
    order, sizes = groups(s_tr); m = ranker().fit(Xtr[order], ytr[order], group=sizes); pte = m.predict(X[te])
    np.savez(cache, tr=oof, te=pte)
    print(f"[ranker] OOF {ts_auc(oof, ytr, s_tr):.4f} | фолд-2 {ts_auc(pte, y[te], s[te]):.4f} [{time.time()-t0:.0f}s]", flush=True)
sig = lambda a: 1/(1+np.exp(-a))
names = [p[4:-4] for p in sorted(os.listdir("stack")) if p.startswith("oof_")]
P_tr = np.column_stack([np.load(f"stack/oof_{n}.npz")["tr"] for n in names]).astype("float64")
P_te = np.column_stack([np.load(f"stack/oof_{n}.npz")["te"] for n in names]).astype("float64")
for j, n in enumerate(names):
    if n in ("linear", "ranker"): P_tr[:, j] = sig(P_tr[:, j]); P_te[:, j] = sig(P_te[:, j])
sf, yf = s[te], y[te]
print(f"--- общая мета над {len(names)} членами: {names} ---", flush=True)
print(f"среднее всех: {ts_auc(P_te.mean(1), yf, sf):.4f}", flush=True)
lr = LogisticRegression(C=1.0, max_iter=2000).fit(P_tr, ytr); ml = lr.decision_function(P_te)
print(f"мета линейная: {ts_auc(ml, yf, sf):.4f}; веса {dict(zip(names, lr.coef_[0].round(2)))}", flush=True)
mg = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=500, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=8).fit(P_tr, ytr).predict_proba(P_te)[:, 1]
print(f"мета LightGBM: {ts_auc(mg, yf, sf):.4f}", flush=True)
np.savez("stack/meta3_fold2.npz", mean=P_te.mean(1), lin=ml, gb=mg)
net = np.mean([np.load(f"fold2_sig_nets_aug3_member_p{i}.pt.npy") for i in range(6)] + [np.load(f"fold2_sig_nets_aug3_last_member_z{i}.pt.npy") for i in range(6)], 0)
rank = sig(np.load("fold2_rank_bocpd_200.npy").astype("float64")); clf = np.load("fold2_clf_bocpdh50_206.npy").astype("float64")
print(f"смесь #30: {ts_auc(0.45*(0.7*rank+0.3*clf)+0.55*net, yf, sf):.4f}", flush=True)
for nm, meta in (("среднее", P_te.mean(1)), ("линейная", sig(ml)), ("gb", mg)):
    for w in (0.35, 0.45, 0.55):
        print(f"  деревья=мета({nm}) как вся древесная половина, вес сетей {w}: {ts_auc((1-w)*meta + w*net, yf, sf):.4f}", flush=True)
print("done", flush=True)
