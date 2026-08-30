"""056: спектр на нетронутом фолде 2 — честная проверка переносимости."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import lightgbm as lgb
from structural_break.combiners import split_by_series, step_weights, ts_auc

t0 = time.time()
base_mats = [np.load("X40.npy"), np.load("C_cnn.npy")[:, None],
             np.load("N9.npy").astype("float32"), np.load("X50a.npy"),
             np.load("E4.npy"), np.load("B40.npy"), np.load("B2.npy")]
X186 = np.hstack(base_mats)
X200 = np.hstack(base_mats + [np.load("SPEC14.npy")])
y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)

VAL = 2  # фолд, на который мы никогда не смотрели
tr, va = assignment != VAL, assignment == VAL
yf, sf = y[va], s[va]

def run(X, tag):
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
    c = lgb.LGBMClassifier(
        objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=500,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0,
        n_estimators=300, random_state=0, n_jobs=8, deterministic=True,
        force_row_wise=True, verbose=-1)
    c.fit(X[tr], y[tr], sample_weight=step_weights(s[tr]))
    pc = c.predict_proba(X[va])[:, 1]
    sig = 1.0 / (1.0 + np.exp(-sc))
    print(f"{tag}: ранкер {ts_auc(sc, yf, sf):.4f}, классификатор {ts_auc(pc, yf, sf):.4f}, "
          f"пара {ts_auc(0.7*sig + 0.3*pc, yf, sf):.4f}  [{time.time()-t0:.0f}s]", flush=True)
    return ts_auc(0.7*sig + 0.3*pc, yf, sf)

a = run(X186, "186 каналов (без спектра), фолд-2")
b = run(X200, "200 каналов (со спектром), фолд-2")
print(f"РАЗНИЦА на нетронутом фолде: {b - a:+.4f}", flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
