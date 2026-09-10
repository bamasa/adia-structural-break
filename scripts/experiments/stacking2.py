"""099b: второй ярус разнообразия — другие семейства моделей на полных 206 (+ блок тестов), тот же OOF-протокол; общая мета над всеми.
"""
import sys, time, os, numpy as np, lightgbm as lgb
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
t0 = time.time()
X = np.hstack([np.load("X40.npy"), np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
               np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy"), np.load("SPEC14.npy"), np.load("BOCPD6_H50.npy")]).astype("float32")
T = np.load("TESTS8.npy") if os.path.exists("TESTS8.npy") else None
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
f = split_by_series(g, folds=5, seed=0); te = f == 2; tr = ~te; f_tr = f[tr]; ytr = y[tr]; inner = [0, 1, 3, 4]
rng = np.random.default_rng(0)
def oof_for(name, make, Xall, sub=None):
    cache = f"stack/oof_{name}.npz"
    if os.path.exists(cache):
        z = np.load(cache); print(f"[{name}] из кэша", flush=True); return z["tr"], z["te"]
    Xtr = Xall[tr]; oof = np.zeros(tr.sum(), dtype="float32")
    for k in inner:
        fit, hold = np.flatnonzero(f_tr != k), np.flatnonzero(f_tr == k)
        if sub: fit = rng.choice(fit, min(sub, len(fit)), replace=False)
        m = make().fit(Xtr[fit], ytr[fit]); oof[hold] = (m.decision_function(Xtr[hold]) if hasattr(m, "decision_function") and not hasattr(m, "predict_proba") else m.predict_proba(Xtr[hold])[:, 1])
    fit = np.arange(tr.sum())
    if sub: fit = rng.choice(fit, min(sub, len(fit)), replace=False)
    m = make().fit(Xtr[fit], ytr[fit]); pte = (m.decision_function(Xall[te]) if hasattr(m, "decision_function") and not hasattr(m, "predict_proba") else m.predict_proba(Xall[te])[:, 1])
    np.savez(cache, tr=oof, te=pte)
    print(f"[{name}] OOF {ts_auc(oof, ytr, s[tr]):.4f} | фолд-2 {ts_auc(pte, y[te], s[te]):.4f} [{time.time()-t0:.0f}s]", flush=True)
    return oof, pte
class Lin:
    def __init__(self): self.sc = StandardScaler(); self.m = LogisticRegression(C=0.05, max_iter=300)
    def fit(self, X, y): self.m.fit(self.sc.fit_transform(X), y); return self
    def decision_function(self, X): return self.m.decision_function(self.sc.transform(X))
members = {}
members["lgb_wide"] = oof_for("lgb_wide", lambda: lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=255, max_depth=8, colsample_bytree=0.4, subsample=0.7, subsample_freq=1, min_child_samples=300, verbose=-1, n_jobs=8), X)
members["lgb_dart"] = oof_for("lgb_dart", lambda: lgb.LGBMClassifier(boosting_type="dart", n_estimators=300, learning_rate=0.08, num_leaves=63, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8), X)
members["extratrees"] = oof_for("extratrees", lambda: ExtraTreesClassifier(n_estimators=300, max_features=0.3, min_samples_leaf=50, n_jobs=8), X, sub=1_200_000)
members["rf"] = oof_for("rf", lambda: RandomForestClassifier(n_estimators=200, max_features=0.3, min_samples_leaf=100, n_jobs=8), X, sub=1_000_000)
members["linear"] = oof_for("linear", Lin, X, sub=1_500_000)
if T is not None:
    members["tests8"] = oof_for("tests8", lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=63, colsample_bytree=0.6, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8), T)
    members["full214"] = oof_for("full214", lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=63, colsample_bytree=0.6, subsample=0.8, subsample_freq=1, min_child_samples=100, verbose=-1, n_jobs=8), np.hstack([X, T]))
# общая мета: блоки 099 + семейства 099b
blocks = ["X40", "cnn_now", "X50a", "forecast", "battery_raw", "battery_asinh", "spectral", "bocpd", "full206", "raw_half", "asinh_half"]
P_tr, P_te, names = [], [], []
for nm in blocks + list(members):
    z = np.load(f"stack/oof_{nm}.npz"); P_tr.append(z["tr"]); P_te.append(z["te"]); names.append(nm)
P_tr = np.column_stack(P_tr).astype("float64"); P_te = np.column_stack(P_te).astype("float64")
sf, yf = s[te], y[te]
sig = lambda a: 1/(1+np.exp(-a))
# линейные выходы -> в сигмоиду для сопоставимости
for j, nm in enumerate(names):
    if nm == "linear": P_tr[:, j] = sig(P_tr[:, j]); P_te[:, j] = sig(P_te[:, j])
print(f"--- общая мета над {len(names)} членами ---", flush=True)
print(f"среднее всех: {ts_auc(P_te.mean(1), yf, sf):.4f}", flush=True)
lr = LogisticRegression(C=1.0, max_iter=1000).fit(P_tr, ytr); ml = lr.decision_function(P_te)
print(f"мета линейная: {ts_auc(ml, yf, sf):.4f}; веса {dict(zip(names, lr.coef_[0].round(2)))}", flush=True)
mp = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=500, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=8)
mg = lgb.LGBMClassifier(**mp).fit(P_tr, ytr).predict_proba(P_te)[:, 1]
print(f"мета LightGBM: {ts_auc(mg, yf, sf):.4f}", flush=True)
np.savez("stack/meta2_fold2.npz", mean=P_te.mean(1), lin=ml, gb=mg)
rank = sig(np.load("fold2_rank_bocpd_200.npy").astype("float64")); clf = np.load("fold2_clf_bocpdh50_206.npy").astype("float64")
net = np.mean([np.load(f"fold2_sig_nets_aug3_member_p{i}.pt.npy") for i in range(6)] + [np.load(f"fold2_sig_nets_aug3_last_member_z{i}.pt.npy") for i in range(6)], 0)
print(f"смесь #30: {ts_auc(0.45*(0.7*rank+0.3*clf)+0.55*net, yf, sf):.4f}", flush=True)
for nm, meta in (("среднее", P_te.mean(1)), ("линейная", sig(ml)), ("gb", mg)):
    for cw in (0.3, 0.5, 0.7, 1.0):
        trees = (1-cw)*rank + cw*meta
        print(f"  мета={nm:9s} доля {cw}: деревья {ts_auc(trees, yf, sf):.4f} | с сетями 0.55 {ts_auc(0.45*trees+0.55*net, yf, sf):.4f} | 0.45 {ts_auc(0.55*trees+0.45*net, yf, sf):.4f}", flush=True)
print("done", flush=True)
