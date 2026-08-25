"""015: forecaster extensions — horizons 1 and 5, signed error."""
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
print(f"загрузка {time.time()-t0:.0f}s", flush=True)

def deviations(z):
    csum = np.concatenate([[0.0], np.cumsum(z)])
    out = np.full(len(z), np.nan)
    for i in range(K, len(z)):
        out[i] = z[i] - (csum[i] - csum[i - K]) / K
    return out

def feats(d, idx, h):
    lags = np.stack([d[idx - h + 1 - j] for j in range(1, LAGS + 1)], axis=1)
    roll5 = np.stack([d[idx - h + 1 - j] for j in range(1, 6)], axis=1)
    return np.hstack([lags, roll5.mean(1, keepdims=True), roll5.std(1, keepdims=True)])

HORIZONS = (1, 5)
cache = []
pre = {h: ([], []) for h in HORIZONS}
for sid, hist, online, labels in iter_series(x, y):
    norm = Normalisation.fit(hist)
    n = len(hist)
    z_h = np.asarray([norm.clip(norm.standardise(float(v), i - n)) for i, v in enumerate(hist)])
    z_o = np.asarray([norm.clip(norm.standardise(float(v), i)) for i, v in enumerate(online)])
    cache.append((z_h, z_o))
    d = deviations(z_h)
    for h in HORIZONS:
        idx = np.arange(LAGS + K + h - 1, n)
        if len(idx) > 8:
            pre[h][0].append(feats(d, idx, h)[::2])
            pre[h][1].append(d[idx][::2])
bases = {}
for h in HORIZONS:
    bases[h] = lgb.train(
        dict(objective="l2", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200,
             feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
             deterministic=True, force_row_wise=True, num_threads=8, seed=0),
        lgb.Dataset(np.vstack(pre[h][0]).astype("float32"),
                    np.concatenate(pre[h][1]).astype("float32")),
        num_boost_round=200)
    print(f"горизонт {h}: модель готова, {time.time()-t0:.0f}s", flush=True)
del pre

rows, count = [], 0
for z_h, z_o in cache:
    d = deviations(np.concatenate([z_h, z_o]))
    n_h = len(z_h)
    models, sigmas = {}, {}
    for h in HORIZONS:
        idx = np.arange(LAGS + K + h - 1, n_h)
        models[h], sigmas[h] = bases[h], 1.0
        if len(idx) > 40:
            F = feats(d, idx, h).astype("float32"); T = d[idx].astype("float32")
            models[h] = lgb.train(
                dict(objective="l2", learning_rate=0.03, num_leaves=7, min_data_in_leaf=20,
                     verbose=-1, deterministic=True, force_row_wise=True, num_threads=1, seed=0),
                lgb.Dataset(F, T), num_boost_round=30, init_model=bases[h])
            sigmas[h] = float(np.std(T - models[h].predict(F, num_threads=1))) + 1e-6
    idx_o = np.arange(n_h, n_h + len(z_o))
    errs, signed = {}, None
    for h in HORIZONS:
        F = feats(d, idx_o, h)
        ok = np.isfinite(F).all(axis=1) & np.isfinite(d[idx_o])
        e = np.zeros(len(z_o)); raw_e = np.zeros(len(z_o))
        if ok.any():
            pred = models[h].predict(F[ok], num_threads=1)
            raw = (d[idx_o][ok] - pred) / sigmas[h]
            e[ok] = np.abs(raw)
            if h == 1:
                raw_e[ok] = raw
        errs[h] = e
        if h == 1:
            signed = raw_e
    f1, s1, p1, f5, s5, p5, sg = 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0
    for e1, e5, r1 in zip(errs[1], errs[5], signed):
        e1 = min(float(e1), 8.0); e5 = min(float(e5), 8.0); r1 = float(np.clip(r1, -8, 8))
        f1 += 0.10 * (e1 - f1); s1 += 0.02 * (e1 - s1); p1 = max(p1, f1)
        f5 += 0.10 * (e5 - f5); s5 += 0.02 * (e5 - s5); p5 = max(p5, f5)
        sg += 0.05 * (r1 - sg)
        rows.append([f1, s1, p1, f5, s5, p5, sg, e1])
    count += 1
    if count % 2000 == 0:
        print(f"  {count} рядов, {time.time()-t0:.0f}s", flush=True)

a = np.asarray(rows, dtype="float32")
np.save("E8.npy", a)
print(f"готово {time.time()-t0:.0f}s: {a.shape}", flush=True)
