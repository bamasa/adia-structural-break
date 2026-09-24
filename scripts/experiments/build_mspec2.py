"""126: спектр на ДЛИННЫХ окнах (64..1024) и 10 полосах — усиление 124 (корр. 0.12, соло 0.5285).

SPEC14 в основном наборе считает спектр на одном окне (SPEC_NFFT) по геометрической
сетке шагов. Здесь та же массовость по окнам, что дала прирост в 114, но над
частотной областью: на окнах 32/64/128/256 точек — мощность в 6 логарифмических
полосах относительно истории (в сигмах исторического разброса), спектральная
энтропия и центроид. 4 окна x 8 = 32 канала. FFT на каждом шаге, O(W log W).
python build_mspec.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
WINS = (64, 128, 256, 512, 1024); NB = 10
OUT = os.environ.get("MSPEC_OUT", "MSPEC60"); PARTS = f"{OUT.lower()}_parts"
NCH = len(WINS) * (NB + 2)

def band_edges(w):
    k = w // 2
    e = np.unique(np.round(np.geomspace(1, k, NB + 1)).astype(int)); 
    while len(e) < NB + 1: e = np.append(e, e[-1] + 1)
    return e[:NB + 1]

def spec_feats(seg, edges):
    w = len(seg); win = np.hanning(w); sp = np.abs(np.fft.rfft((seg - seg.mean()) * win))[1:] ** 2
    tot = sp.sum() + 1e-12; p = sp / tot
    bands = np.array([p[edges[i]-1:edges[i+1]-1].sum() for i in range(NB)])
    ent = -(p * np.log(p + 1e-12)).sum(); freqs = np.arange(1, len(p) + 1) / w
    cen = (p * freqs).sum()
    return np.concatenate([np.log(bands + 1e-6), [ent, cen]])

def mspec_channels(hist, online):
    h = np.asarray(hist, float); mu, sd = h.mean(), h.std() + 1e-12
    zh = (h - mu) / sd; zo = (np.asarray(online, float) - mu) / sd
    n = len(zo); out = np.zeros((n, NCH), dtype="float32")
    edges = {w: band_edges(w) for w in WINS}
    ref = {}
    for w in WINS:                                   # исторический профиль: среднее и разброс по непересекающимся окнам
        segs = [spec_feats(zh[i:i + w], edges[w]) for i in range(0, len(zh) - w + 1, w)]
        S = np.array(segs); ref[w] = (S.mean(0), S.std(0) + 1e-3)
    full = np.concatenate([zh[-max(WINS):], zo])
    for t in range(n):
        pos = len(zh[-max(WINS):]) + t + 1
        for i, w in enumerate(WINS):
            f = spec_feats(full[pos - w:pos], edges[w]); m, s = ref[w]
            out[t, i * (NB + 2):(i + 1) * (NB + 2)] = np.clip((f - m) / s, -20, 20)
    return out

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0)
        def ar(phi, n): 
            x = np.zeros(n); e = rng.normal(0, 1, n)
            for i in range(1, n): x[i] = phi * x[i-1] + e[i]
            return x
        hist = ar(0.0, 2000); online = np.concatenate([ar(0.0, 300), ar(0.7, 300)])
        t0 = time.time(); o = mspec_channels(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print(f"каналов {o.shape[1]}, {dt:.3f} мс/шаг; смена спектра (белый -> AR 0.7): нижняя полоса/окно128 {o[250:300,16].mean():+.2f} -> {o[450:600,16].mean():+.2f}, центроид {o[250:300,23].mean():+.2f} -> {o[450:600,23].mean():+.2f}")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
        g = np.load("G40.npy"); s = np.load("S40.npy")
        st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
        out = np.empty((len(g), NCH), dtype="float32"); block = {}
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
        sids.append(int(sid)); arrs.append(mspec_channels(hist, online))
        if (i + 1) % 200 == 0: print(f"шард {shard}: {i+1} рядов, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"шард {shard} готов {time.time()-t0:.0f}s: {len(sids)} рядов", flush=True)
