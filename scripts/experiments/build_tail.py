"""139: хвостовой член — новое сырьё. «Поведение хвостов» в списке типов слома, у нас на него 3 канала из ~400.

На окнах 25/50/100/250/500 (EWMA): доля |z| выше исторических q90/q97.5/q99.5; средний размер
превышения над q90 (хвостовая шкала); доля точек в верхнем и нижнем 2.5% отдельно (асимметрия
хвостов); скользящий максимум |z| относительно ожидаемого максимума в истории на таком окне;
кластеризация превышений (доля превышений, у которых предыдущая точка тоже превышение).
8 x 5 = 40 каналов, O(1) на шаг. python build_tail.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
WINS = (25.0, 50.0, 100.0, 250.0, 500.0); NF = 8
OUT = os.environ.get("TAIL_OUT", "TAIL40"); PARTS = f"{OUT.lower()}_parts"
NCH = NF * len(WINS)

def tail_channels(hist, online):
    h = np.asarray(hist, float); mu, sd = h.mean(), h.std() + 1e-12; zh = (h - mu) / sd; az = np.abs(zh)
    q90, q975, q995 = np.quantile(az, [0.90, 0.975, 0.995]); qhi, qlo = np.quantile(zh, [0.975, 0.025])
    exc_scale_h = (az[az > q90] - q90).mean() if (az > q90).any() else 1.0
    W = np.array(WINS); a = 1.0 / W
    # ожидаемый максимум |z| на окне длины W по истории: медиана максимумов по блокам
    exp_max = np.array([np.median([az[i:i + int(w)].max() for i in range(0, max(len(az) - int(w), 1), max(int(w) // 2, 1))]) for w in W])
    n = len(online); out = np.zeros((n, NCH), dtype="float32")
    r90 = np.zeros(len(W)); r975 = np.zeros(len(W)); r995 = np.zeros(len(W)); esc = np.zeros(len(W))
    rhi = np.zeros(len(W)); rlo = np.zeros(len(W)); clu = np.zeros(len(W)); mx = np.zeros(len(W))
    prev_exc = 0.0
    for t, x in enumerate(online):
        z = (x - mu) / sd; v = abs(z)
        e90 = float(v > q90)
        r90 = (1 - a) * r90 + a * e90; r975 = (1 - a) * r975 + a * float(v > q975); r995 = (1 - a) * r995 + a * float(v > q995)
        esc = (1 - a) * esc + a * (max(v - q90, 0.0) / (exc_scale_h + 1e-9))
        rhi = (1 - a) * rhi + a * float(z > qhi); rlo = (1 - a) * rlo + a * float(z < qlo)
        clu = (1 - a) * clu + a * (e90 * prev_exc)
        mx = np.maximum(mx * (1 - a / 2), v)                       # медленно спадающий максимум
        out[t] = np.concatenate([r90 - 0.10, r975 - 0.025, r995 - 0.005, esc - 0.10, rhi - 0.025, rlo - 0.025, clu - 0.01, mx / exp_max - 1.0])
        prev_exc = e90
    return np.clip(np.nan_to_num(out), -20, 20)

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0); hist = rng.normal(0, 1, 2000)
        online = np.concatenate([rng.normal(0, 1, 300), rng.standard_t(2.5, 300) / np.sqrt(5.0)])   # тяжелее хвосты, та же дисперсия
        t0 = time.time(); o = tail_channels(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print(f"каналов {o.shape[1]}, {dt:.3f} мс/шаг; хвосты t(2.5) при той же σ: q99.5-доля/окно100 {o[250:300,12].mean():+.3f} -> {o[450:600,12].mean():+.3f}, макс/окно100 {o[250:300,37].mean():+.2f} -> {o[450:600,37].mean():+.2f}")
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
        sids.append(int(sid)); arrs.append(tail_channels(hist, online))
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"шард {shard} готов {time.time()-t0:.0f}s: {len(sids)} рядов", flush=True)
