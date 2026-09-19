"""112: полный GLR-скан позиции слома — перебор всех k на каждом шаге.

Все прежние каналы сравнивают «последние W точек» с историей при фиксированных W.
Здесь на каждом шаге t перебираются ВСЕ возможные позиции слома k in [0, t] и для
каждой считается логарифм отношения правдоподобия сегмента [k..t] против истории
(нормальная модель: своё среднее и дисперсия против исторических). Берётся максимум.
Это классический GLR-детектор, который мы никогда не считали честно — не хватало
бюджета. Сейчас используем 3% лимита, так что можно.

6 каналов: max LR (норм.), позиция аргмаксa / t, max LR только по среднему,
max LR только по дисперсии, max LR на хвосте 200, превышение над нулевым ожиданием.
O(t) numpy на шаг. python build_glr.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
OUT = os.environ.get("GLR_OUT", "GLR6"); PARTS = f"{OUT.lower()}_parts"

def glr_channels(hist, online):
    h = np.asarray(hist, float); mu_h = h.mean(); v_h = h.var() + 1e-12
    n = len(online); z = (np.asarray(online, float) - mu_h) / np.sqrt(v_h)
    c1 = np.concatenate([[0.0], np.cumsum(z)]); c2 = np.concatenate([[0.0], np.cumsum(z * z)])
    out = np.empty((n, 6), dtype="float32")
    for t in range(n):
        k = np.arange(0, t + 1)                       # начало сегмента
        m = (t + 1) - k                               # длина сегмента [k..t]
        s1 = c1[t + 1] - c1[k]; s2 = c2[t + 1] - c2[k]
        mean = s1 / m
        var = np.maximum(s2 / m - mean * mean, 1e-9)
        lr_mean = 0.5 * m * mean * mean                       # сдвиг среднего
        lr_var = 0.5 * m * (var - 1.0 - np.log(var))          # изменение дисперсии
        lr = lr_mean + lr_var
        ok = m >= 10                                          # короткие сегменты шумят
        if not ok.any():
            out[t] = 0.0; continue
        lr_ok = np.where(ok, lr, -np.inf); j = int(np.argmax(lr_ok)); best = float(lr_ok[j])
        tail = m <= 200
        lr_tail = np.where(ok & tail, lr, -np.inf)
        best_tail = float(lr_tail.max()) if np.isfinite(lr_tail).any() else 0.0
        null = 0.5 * np.log(max(t + 1, 2))                    # грубое ожидание максимума под нулём
        out[t] = (np.tanh(best / 20.0), j / max(t, 1),
                  np.tanh(float(np.where(ok, lr_mean, -np.inf).max()) / 20.0),
                  np.tanh(float(np.where(ok, lr_var, -np.inf).max()) / 20.0),
                  np.tanh(best_tail / 20.0), np.tanh((best - null) / 20.0))
    return out

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0); hist = rng.normal(0, 1, 2000)
        online = np.concatenate([rng.normal(0, 1, 300), rng.normal(0.4, 1.0, 300)])
        t0 = time.time(); o = glr_channels(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print(f"тест (сдвиг +0.4σ на шаге 300): max LR до {o[250:300,0].mean():.3f} после {o[400:600,0].mean():.3f}; "
              f"позиция аргмаксa после {o[400:600,1].mean():.3f} (истина {300/500:.2f}); {dt:.2f} мс/шаг")
        online2 = np.concatenate([rng.normal(0, 1, 300), rng.normal(0, 1.4, 300)])
        o2 = glr_channels(hist, online2)
        print(f"тест (дисперсия ×1.4 на шаге 300): LR-дисперсия до {o2[250:300,3].mean():.3f} после {o2[400:600,3].mean():.3f}")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
        g = np.load("G40.npy"); s = np.load("S40.npy")
        st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
        out = np.empty((len(g), 6), dtype="float32"); block = {}
        for p in parts:
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a, b in zip(st, bd[1:]): out[a:b] = block[int(g[a])][s[a:b]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for i, (sid, part) in enumerate(x.groupby(level="id")):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(glr_channels(hist, online))
        if (i + 1) % 200 == 0: print(f"шард {shard}: {i+1} рядов, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"шард {shard} готов {time.time()-t0:.0f}s: {len(sids)} рядов", flush=True)
