"""118: расширенная массовая батарея — восемь окон, четыре сравнения на каждое.

114 дал +0.004 на 90 каналах (6 представлений x 6 окон x 2 сравнения). Здесь то же
самое, но шире: окна 5..1000, плюс третий и четвёртый моменты на окне (асимметрия и
эксцесс против исторических). 6 x 8 x 4 = 192 + 8 x 3 квантильных = 216 каналов.
Всё через скользящие моменты, O(представлений x окон) на шаг.
python build_mass2.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
WINS = (5, 10, 25, 50, 100, 250, 500, 1000); NREP = 6
OUT = os.environ.get("M2_OUT", "MASS216"); PARTS = f"{OUT.lower()}_parts"
NCH = NREP * len(WINS) * 4 + len(WINS) * 3

def mass2(hist, online):
    h = np.asarray(hist, float); mu, sd = h.mean(), h.std() + 1e-12
    zh = (h - mu) / sd
    reps_h = [zh, np.abs(zh), zh * zh, np.diff(zh, prepend=zh[0]), np.zeros_like(zh), np.sign(zh)]
    hm = np.array([r.mean() for r in reps_h]); hv = np.array([r.var() + 1e-9 for r in reps_h])
    hs = np.array([((r - r.mean()) ** 3).mean() / (r.std() + 1e-9) ** 3 for r in reps_h])
    hk = np.array([((r - r.mean()) ** 4).mean() / (r.std() + 1e-9) ** 4 for r in reps_h])
    q = np.quantile(np.abs(zh), [0.75, 0.95, 0.99])
    W = np.array(WINS, float); a = 1.0 / W
    m1 = np.zeros((NREP, len(WINS))); m2 = np.zeros((NREP, len(WINS)))
    m3 = np.zeros((NREP, len(WINS))); m4 = np.zeros((NREP, len(WINS)))
    qs = np.zeros((3, len(WINS)))
    n = len(online); out = np.empty((n, NCH), dtype="float32")
    prev = zh[-1]; cum = 0.0
    se = np.sqrt(hv[:, None] / W[None, :])
    for t, x in enumerate(online):
        z = (x - mu) / sd; cum += z
        v = np.array([z, abs(z), z * z, z - prev, cum / np.sqrt(t + 1), np.sign(z)])[:, None]
        m1 = (1 - a) * m1 + a * v; m2 = (1 - a) * m2 + a * v ** 2
        m3 = (1 - a) * m3 + a * v ** 3; m4 = (1 - a) * m4 + a * v ** 4
        var = np.maximum(m2 - m1 ** 2, 1e-9); s = np.sqrt(var)
        skew = (m3 - 3 * m1 * var - m1 ** 3) / (s ** 3 + 1e-9)
        kurt = (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / (var ** 2 + 1e-9)
        k = NREP * len(WINS)
        out[t, :k] = ((m1 - hm[:, None]) / se).ravel()
        out[t, k:2*k] = np.log(var / hv[:, None]).ravel()
        out[t, 2*k:3*k] = (skew - hs[:, None]).ravel()
        out[t, 3*k:4*k] = (kurt - hk[:, None]).ravel()
        exc = (abs(z) > q[:, None]).astype(float)
        qs = (1 - a[None, :]) * qs + a[None, :] * exc
        out[t, 4*k:] = (qs - np.array([0.25, 0.05, 0.01])[:, None]).ravel()
        prev = z
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20)

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0); hist = rng.normal(0, 1, 2000)
        online = np.concatenate([rng.normal(0, 1, 300), rng.normal(0.3, 1.2, 300)])
        t0 = time.time(); o = mass2(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print(f"каналов {o.shape[1]}, {dt:.3f} мс/шаг; среднее-окно100 {o[250:300,4].mean():+.2f} -> {o[450:600,4].mean():+.2f}")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
        g = np.load("G40.npy"); s = np.load("S40.npy")
        st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
        out = np.empty((len(g), NCH), dtype="float32"); block = {}
        for p in parts:
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a_, b_ in zip(st, bd[1:]): out[a_:b_] = block[int(g[a_])][s[a_:b_]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for sid, part in x.groupby(level="id"):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(mass2(hist, online))
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"шард {shard} готов {time.time()-t0:.0f}s: {len(sids)} рядов", flush=True)
