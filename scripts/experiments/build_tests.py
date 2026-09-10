"""099t: явные двухвыборочные тесты история vs префикс / окно, потоково: KS, Манн–Уитни (AUC-форма), Левен (по |x - медиана|), Флигнер-подобный (ранги |x - медиана|).
Каналы (8): KS_prefix, MW_prefix, Levene_prefix, Fligner_prefix, KS_win100, MW_win100, Levene_win100, Fligner_win100.
Реализация через 64 квантильных бина истории: O(бины) на шаг.
"""
import sys, time, os, numpy as np
NB = 64; WIN = 100
OUT = os.environ.get("TESTS_OUT", "TESTS8"); PARTS = f"{OUT.lower()}_parts"

def tests(hist, online):
    h = np.asarray(hist, float); edges = np.quantile(h, np.linspace(0, 1, NB + 1)[1:-1])
    hb = np.searchsorted(edges, h); ph = np.bincount(hb, minlength=NB) / len(h); Fh = np.cumsum(ph)
    med = np.median(h); dev_h = np.abs(h - med); dedges = np.quantile(dev_h, np.linspace(0, 1, NB + 1)[1:-1])
    pdh = np.bincount(np.searchsorted(dedges, dev_h), minlength=NB) / len(h); Fdh = np.cumsum(pdh); mean_dev_h = dev_h.mean()
    # Манн–Уитни в форме AUC: P(x_online > x_hist) ≈ сумма по бинам p_o[b]·(F_h[b-1] + 0.5·p_h[b])
    mw_w = np.concatenate([[0.0], Fh[:-1]]) + 0.5 * ph; mwd_w = np.concatenate([[0.0], Fdh[:-1]]) + 0.5 * pdh
    def stats(cnt, dcnt, dsum, n):
        p = cnt / n; F = np.cumsum(p)
        ks = np.abs(F - Fh).max(); mw = (p * mw_w).sum() - 0.5
        lev = (dsum / n) / (mean_dev_h + 1e-12) - 1.0                  # Левен: отношение средних |x - med|
        pd_ = dcnt / n; fl = (pd_ * mwd_w).sum() - 0.5                  # Флигнер-подобный: MW на |x - med|
        return ks, mw, lev, fl
    n = len(online); out = np.empty((n, 8), dtype="float32")
    cp = np.zeros(NB); cd = np.zeros(NB); dsum = 0.0; wb, wd, wdv = [], [], []
    cw = np.zeros(NB); cwd = np.zeros(NB); wsum = 0.0
    for t, x in enumerate(online):
        b = int(np.searchsorted(edges, x)); d = abs(x - med); db = int(np.searchsorted(dedges, d))
        cp[b] += 1; cd[db] += 1; dsum += d
        cw[b] += 1; cwd[db] += 1; wsum += d; wb.append(b); wd.append(db); wdv.append(d)
        if len(wb) > WIN: cw[wb.pop(0)] -= 1; cwd[wd.pop(0)] -= 1; wsum -= wdv.pop(0)
        out[t, :4] = stats(cp, cd, dsum, t + 1); out[t, 4:] = stats(cw, cwd, wsum, len(wb))
    return out

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0); hist = rng.normal(0, 1, 2000)
        online = np.concatenate([rng.normal(0, 1, 400), rng.normal(0.3, 1.3, 400)])
        t0 = time.time(); o = tests(hist, online); dt = (time.time() - t0) / len(online) * 1000
        print("тест (сдвиг 0.3σ + дисперсия ×1.3 на шаге 400): окно до/после — " + ", ".join(f"{n} {o[300:400,4+i].mean():+.3f}->{o[450:550,4+i].mean():+.3f}" for i, n in enumerate(("KS", "MW", "Levene", "Fligner"))) + f"; {dt:.3f} мс/шаг")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); parts = [np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True) for i in range(n)]
        g = np.load("G40.npy"); s = np.load("S40.npy")
        starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bounds = np.append(starts, len(g))
        out = np.empty((len(g), 8), dtype="float32"); block = {}
        for p in parts:
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a_, b_ in zip(starts, bounds[1:]): out[a_:b_] = block[int(g[a_])][s[a_:b_]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for sid, part in x.groupby(level="id"):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(tests(hist, online))
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"шард {shard} готов {time.time()-t0:.0f}s: {len(sids)} рядов", flush=True)
