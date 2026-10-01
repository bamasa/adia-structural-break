"""015b: a magnitude forecaster — predict |deviation|, error channels."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import pandas as pd
import lightgbm as lgb
from structural_break.features import Normalisation
from structural_break.stream import iter_series

K, LAGS = 10, 8
t0 = time.time()
x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
y = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
print(f"loading {time.time()-t0:.0f}s", flush=True)

def deviations(z):
    csum = np.concatenate([[0.0], np.cumsum(z)])
    out = np.full(len(z), np.nan)
    for i in range(K, len(z)):
        out[i] = z[i] - (csum[i] - csum[i - K]) / K
    return out

def feats(a, idx):
    lags = np.stack([a[idx - j] for j in range(1, LAGS + 1)], axis=1)
    roll5 = np.stack([a[idx - j] for j in range(1, 6)], axis=1)
    return np.hstack([lags, roll5.mean(1, keepdims=True), roll5.std(1, keepdims=True)])

cache, fa, ta = [], [], []
for sid, hist, online, labels in iter_series(x, y):
    norm = Normalisation.fit(hist)
    n = len(hist)
    z_h = np.asarray([norm.clip(norm.standardise(float(v), i - n)) for i, v in enumerate(hist)])
    z_o = np.asarray([norm.clip(norm.standardise(float(v), i)) for i, v in enumerate(online)])
    cache.append((z_h, z_o))
    m = np.abs(deviations(z_h))
    idx = np.arange(LAGS + K, n)
    if len(idx) > 8:
        fa.append(feats(m, idx)[::2]); ta.append(m[idx][::2])
base = lgb.train(
    dict(objective="l2", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200,
         feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
         deterministic=True, force_row_wise=True, num_threads=4, seed=0),
    lgb.Dataset(np.vstack(fa).astype("float32"), np.concatenate(ta).astype("float32")),
    num_boost_round=200)
del fa, ta
print(f"amplitude forecaster ready, {time.time()-t0:.0f}s", flush=True)

rows, count = [], 0
for z_h, z_o in cache:
    m_full = np.abs(deviations(np.concatenate([z_h, z_o])))
    n_h = len(z_h)
    idx_h = np.arange(LAGS + K, n_h)
    model, sigma = base, 1.0
    if len(idx_h) > 40:
        F = feats(m_full, idx_h).astype("float32"); T = m_full[idx_h].astype("float32")
        model = lgb.train(
            dict(objective="l2", learning_rate=0.03, num_leaves=7, min_data_in_leaf=20,
                 verbose=-1, deterministic=True, force_row_wise=True, num_threads=1, seed=0),
            lgb.Dataset(F, T), num_boost_round=30, init_model=base)
        sigma = float(np.std(T - model.predict(F, num_threads=1))) + 1e-6
    idx_o = np.arange(n_h, n_h + len(z_o))
    F = feats(m_full, idx_o)
    ok = np.isfinite(F).all(axis=1) & np.isfinite(m_full[idx_o])
    err = np.zeros(len(z_o))
    if ok.any():
        pred = model.predict(F[ok], num_threads=1)
        err[ok] = np.abs(m_full[idx_o][ok] - pred) / sigma
    fast, slow, peak = 1.0, 1.0, 0.0
    for e in err:
        e = min(float(e), 8.0)
        fast += 0.10 * (e - fast); slow += 0.02 * (e - slow); peak = max(peak, fast)
        rows.append([fast, slow, peak, e])
    count += 1
    if count % 2000 == 0:
        print(f"  {count} series, {time.time()-t0:.0f}s", flush=True)
a = np.asarray(rows, dtype="float32")
np.save("M4.npy", a)
print(f"done {time.time()-t0:.0f}s: {a.shape}", flush=True)
