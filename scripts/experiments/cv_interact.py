"""032 (фолд-0): интеракции «внешнее отличие x отсутствие внутреннего перелома»."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import lightgbm as lgb
from structural_break.combiners import split_by_series, ts_auc

t0 = time.time()
X40 = np.load("X40.npy"); B40 = np.load("B40.npy"); B2 = np.load("B2.npy")
base = [X40, np.load("C_cnn.npy")[:, None], np.load("N9.npy").astype("float32"),
        np.load("X50a.npy"), np.load("E4.npy"), B40, B2]

# Внешнее отличие (история vs префикс): KS, |сдвиг среднего|, |лог-отношение СКО| — оба вида.
ks_r, ks_a = B40[:, 9], B40[:, 30]
mean_r, mean_a = np.abs(B40[:, 5]), np.abs(B40[:, 26])
std_r, std_a = np.abs(B40[:, 6]), np.abs(B40[:, 27])
ext_r = np.maximum(ks_r * 3.0, np.maximum(mean_r, std_r))
ext_a = np.maximum(ks_a * 3.0, np.maximum(mean_a, std_a))
# Внутренний перелом: полусплиты батареи + ретроскан лучшего разбиения.
half_r = np.abs(B40[:, 19]) + np.abs(B40[:, 20])
half_a = np.abs(B40[:, 40]) + np.abs(B40[:, 41])
scan = X40[:, 33]  # scan_best, 0..1
internal = np.maximum(np.maximum(half_r, half_a), 2.0 * (scan - 0.5).clip(0))

I = np.column_stack([
    ext_r * (1.0 - scan),            # внешнее есть, скан молчит -> подозрение ложное
    ext_a * (1.0 - scan),
    ext_r - internal,                # чистое превышение внешнего над внутренним
    ext_a - internal,
    ext_r * np.exp(-2.0 * half_r),
    ext_a * np.exp(-2.0 * half_a),
    internal,                        # сводный внутренний
    ext_r, ext_a,                    # сводные внешние
    internal - 0.5 * (ext_r + ext_a),
]).astype("float32")
print(f"интеракций: {I.shape[1]}", flush=True)

y = np.load("Y40.npy"); g = np.load("G40.npy"); s = np.load("S40.npy")
assignment = split_by_series(g, folds=5, seed=0)
tr, va = assignment != 0, assignment == 0
yf, sf = y[va], s[va]
clf = np.load("oof_cfg5.npy")[va].astype("float64")
X = np.hstack(base + [I])
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
sig = 1.0 / (1.0 + np.exp(-sc))
print(f"196 (186+интеракции): соло {ts_auc(sc, yf, sf):.4f}, "
      f"смесь {ts_auc(0.6*sig+0.4*clf, yf, sf):.4f} (эталоны 0.6016 / 0.6045)", flush=True)
print(f"всего {time.time()-t0:.0f}s", flush=True)
