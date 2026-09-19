"""110: прямое сравнение автокорреляций онлайн-части с историей — 8 каналов.

109 показал: сломы зависимостей читаются на 0.583 против 0.686 у сломов дисперсии.
В 200 каналах зависимость входит лишь косвенно (отбеливание по историческому rho),
прямого «текущий rho против исторического» нет. Здесь: EWMA-оценки автокорреляции
на лагах 1,2,5,10 для двух окон (быстрое ~50, медленное ~200), минус исторические.
O(1) на шаг: EWMA от z_t*z_{t-k} и от z_t^2.
python build_acf.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
LAGS = (1, 2, 5, 10); WINS = (50.0, 200.0)
OUT = os.environ.get("ACF_OUT", "ACF8"); PARTS = f"{OUT.lower()}_parts"

def acf_channels(hist, online):
    h = np.asarray(hist, float); mu, sd = h.mean(), h.std() + 1e-12
    zh = (h - mu) / sd
    hist_acf = {k: float(np.corrcoef(zh[:-k], zh[k:])[0, 1]) for k in LAGS}
    buf = list(zh[-max(LAGS):])                      # хвост истории для первых лагов
    n = len(online); out = np.empty((n, len(LAGS) * len(WINS)), dtype="float32")
    num = {(k, w): 0.0 for k in LAGS for w in WINS}   # EWMA от z_t * z_{t-k}
    den = {w: 1.0 for w in WINS}                      # EWMA от z_t^2
    for t, x in enumerate(online):
        z = (x - mu) / sd
        for w in WINS:
            a = 1.0 / w
            den[w] = (1 - a) * den[w] + a * z * z
        for j, k in enumerate(LAGS):
            zk = buf[-k] if len(buf) >= k else 0.0
            for i, w in enumerate(WINS):
                a = 1.0 / w
                num[(k, w)] = (1 - a) * num[(k, w)] + a * z * zk
                r = num[(k, w)] / max(den[w], 1e-9)
                out[t, i * len(LAGS) + j] = np.clip(r, -1.5, 1.5) - hist_acf[k]
        buf.append(z)
        if len(buf) > max(LAGS) + 1: buf.pop(0)
    return out

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0)
        def ar(phi, n, s=1.0):
            x = np.zeros(n); e = rng.normal(0, s, n)
            for i in range(1, n): x[i] = phi * x[i-1] + e[i]
            return x
        hist = ar(0.1, 2000); online = np.concatenate([ar(0.1, 400), ar(0.6, 400, 0.8)])
        t0 = time.time(); o = acf_channels(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print(f"тест (rho 0.1 -> 0.6 на шаге 400): лаг1/окно200 до {o[300:400,4].mean():+.3f} после {o[600:800,4].mean():+.3f}; "
              f"лаг1/окно50 до {o[300:400,0].mean():+.3f} после {o[500:600,0].mean():+.3f}; {dt:.3f} мс/шаг")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
        g = np.load("G40.npy"); s = np.load("S40.npy")
        st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
        out = np.empty((len(g), len(LAGS) * len(WINS)), dtype="float32"); block = {}
        for p in parts:
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a, b in zip(st, bd[1:]): out[a:b] = block[int(g[a])][s[a:b]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for sid, part in x.groupby(level="id"):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(acf_channels(hist, online))
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"шард {shard} готов {time.time()-t0:.0f}s: {len(sids)} рядов", flush=True)
