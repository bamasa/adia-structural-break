"""145: the Rosenblatt-whitened stream as a member's input.

The history fits AR(p<=12) by BIC, a conditional scale (EWMA variance, lambda by
Gaussian quasi-likelihood), and the empirical CDF of the standardised innovations;
the online stream is mapped through the same fit to normal scores n_t that are
i.i.d. N(0,1) under the null for a far wider family of histories than an AR(1)
whitening covers. On that stream: CUSUMs and EWMAs for mean, scale, dependence
(lags 1,2,3,5,10 + portmanteau), volatility clustering, shape and tails; GLR over
dyadic windows 8..1024 for mean, scale and lag-1 dependence (per scale and max);
one-sample KS/CvM/AD against N(0,1) over windows 32..512 and the prefix, at the
geometric cadence -- all on the unconditional stream (history innovation variance),
where scale breaks stay visible; mean/dependence/test extras on the conditional stream;
the conditional scale process against its history level; and the step index. 90 channels.
python build_white.py <shard> <n> | merge <n> | test
"""
import sys, time, os, numpy as np
from scipy.signal import lfilter
from scipy.special import ndtri, ndtr
PMAX = 12; LAMBDAS = (0.90, 0.94, 0.97, 0.99, 1.0); DRIFT = 0.25
LAGS = (1, 2, 3, 5, 10); PORT = 10; DYADIC = (8, 16, 32, 64, 128, 256, 512, 1024); TESTW = (32, 64, 128, 256, 512)
NFULL = 8 + len(LAGS) * 2 + 1 + 2 + 7 + 3 * len(DYADIC) + 3 + 3 * (len(TESTW) + 1) + 3   # 76 on the unconditional stream
NCH = NFULL + 10 + 3 + 1                                                                     # + conditional extras, scale process, step
OUT = os.environ.get("WHITE_OUT", "WHITE90"); PARTS = f"{OUT.lower()}_parts"

def ewma(a, alpha, init=0.0):
    return lfilter([alpha], [1, -(1 - alpha)], a, zi=[(1 - alpha) * init])[0]

def cusum_peak(a, drift=DRIFT):
    cp = np.concatenate([[0.0], np.cumsum(a - drift)]); cm = np.concatenate([[0.0], np.cumsum(-a - drift)])
    return np.maximum(cp[1:] - np.minimum.accumulate(cp)[1:], cm[1:] - np.minimum.accumulate(cm)[1:])

def fit_history(h):
    """AR order by BIC on a common sample, conditional-scale lambda by quasi-likelihood, innovation ECDF."""
    n = len(h); y = h[PMAX:]; best = (np.inf, 0, np.zeros(0))
    for p in range(0, PMAX + 1):
        if p == 0:
            e = y; coef = np.zeros(0)
        else:
            X = np.column_stack([h[PMAX - j - 1: n - j - 1] for j in range(p)]); coef = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ coef
        bic = len(e) * np.log((e ** 2).mean() + 1e-12) + p * np.log(len(e))
        if bic < best[0]: best = (bic, p, coef)
    p, coef = best[1], best[2]
    e = h[p:] - (np.column_stack([h[p - j - 1: n - j - 1] for j in range(p)]) @ coef if p else 0.0)
    e2 = e ** 2; v0 = max(float(e2.mean()), 1e-12); bestl = (-np.inf, 1.0, np.full(len(e), v0))
    for lam in LAMBDAS:
        s2 = np.full(len(e), v0) if lam >= 1.0 else np.concatenate([[v0], ewma(e2[:-1], 1 - lam, init=v0)])
        s2 = np.maximum(s2, 1e-12); ql = -0.5 * np.sum(np.log(s2) + e2 / s2)
        if np.isfinite(ql) and ql > bestl[0]: bestl = (ql, lam, s2)
    lam, s2 = bestl[1], bestl[2]
    def scores(u, su): return ndtri(np.clip((np.searchsorted(su, u) + 0.5) / (len(su) + 1), 1e-6, 1 - 1e-6))
    uc = e / np.sqrt(s2); suc = np.sort(uc); uu = e / np.sqrt(v0); suu = np.sort(uu)
    l = np.log(s2 / v0)
    return dict(p=p, coef=coef, lam=lam, s2_last=float(lam * s2[-1] + (1 - lam) * e2[-1]) if lam < 1.0 else float(v0), v0=float(v0),
                suc=suc, suu=suu, nhc=scores(uc, suc), nhu=scores(uu, suu), lm=float(l.mean()), ls=float(l.std()) + 1e-6)

def normal_scores_online(fit, h, zo):
    p, coef, lam = fit["p"], fit["coef"], fit["lam"]
    full = np.concatenate([h[-PMAX:], zo]); m = len(full)
    e = full[PMAX:] - (np.column_stack([full[PMAX - j - 1: m - j - 1] for j in range(p)]) @ coef if p else 0.0)
    e2 = e ** 2
    if lam >= 1.0: s2 = np.full(len(e), fit["v0"])
    else: s2 = np.concatenate([[fit["s2_last"]], ewma(e2[:-1], 1 - lam, init=fit["s2_last"])])
    def scores(u, su): return np.clip(ndtri(np.clip((np.searchsorted(su, u) + 0.5) / (len(su) + 1), 1e-6, 1 - 1e-6)), -4.5, 4.5)
    return scores(e / np.sqrt(s2), fit["suc"]), scores(e / np.sqrt(fit["v0"]), fit["suu"]), (np.log(s2 / fit["v0"]) - fit["lm"]) / fit["ls"]

def one_sample_tests(w):
    """KS, Cramer-von Mises and Anderson-Darling of a window against N(0,1)."""
    x = np.sort(w); m = len(x); F = ndtr(x); i = np.arange(1, m + 1)
    ks = np.sqrt(m) * max((i / m - F).max(), (F - (i - 1) / m).max())
    cvm = 1.0 / (12 * m) + ((F - (2 * i - 1) / (2 * m)) ** 2).sum()
    Fc = np.clip(F, 1e-10, 1 - 1e-10)
    ad = -m - ((2 * i - 1) * (np.log(Fc) + np.log(1 - Fc[::-1]))).sum() / m
    return ks, cvm, ad

def _battery(n_on, nh, out, c):
    """The full battery on one normal-score stream, written into out[:, c:]; returns the next column."""
    T = len(n_on)
    tail = nh[-1024:]; ext = np.concatenate([tail, n_on]); off = len(tail)   # extended stream: history scores then online
    n = n_on
    # mean
    out[:, 0] = cusum_peak(n); out[:, 1] = ewma(n, 0.02); out[:, 2] = ewma(n, 0.005); out[:, 3] = np.cumsum(n) / np.sqrt(np.arange(1, T + 1))
    # scale
    q = (n ** 2 - 1) / np.sqrt(2); out[:, 4] = cusum_peak(q); out[:, 5] = ewma(q, 0.02); out[:, 6] = ewma(q, 0.005)
    out[:, 7] = (np.cumsum(n ** 2) / np.arange(1, T + 1) - 1) * np.sqrt(np.arange(1, T + 1) / 2); c = 8
    # dependence
    port = []
    for k in range(1, PORT + 1):
        a = ext[off:] * ext[off - k: len(ext) - k]; port.append(ewma(a, 0.01))
        if k in LAGS:
            out[:, c] = cusum_peak(a); out[:, c + 1] = port[-1]; c += 2
    out[:, c] = 199 * (np.stack(port) ** 2).sum(0); c += 1
    # volatility clustering, against the history's own level
    ah = np.abs(nh); mh, sh = (ah[1:] * ah[:-1]).mean(), (ah[1:] * ah[:-1]).std() + 1e-9
    a = (np.abs(ext[off:]) * np.abs(ext[off - 1: len(ext) - 1]) - mh) / sh; out[:, c] = cusum_peak(a); out[:, c + 1] = ewma(a, 0.01); c += 2
    # shape and tails, against the history's own frequencies
    out[:, c] = ewma(n ** 3, 0.01); out[:, c + 1] = ewma(n ** 4 - 3, 0.01)
    for j, ev in enumerate(((np.abs(n) > 2.5, np.abs(nh) > 2.5), (np.abs(n) > 1.5, np.abs(nh) > 1.5), (np.abs(n) < 0.3, np.abs(nh) < 0.3))):
        out[:, c + 2 + j] = ewma(ev[0].astype(float) - ev[1].mean(), 0.01)
    sc = (np.sign(ext[off:]) != np.sign(ext[off - 1: len(ext) - 1])).astype(float); out[:, c + 5] = ewma(sc - (np.sign(nh[1:]) != np.sign(nh[:-1])).mean(), 0.01)
    out[:, c + 6] = ewma(np.abs(n) - np.abs(nh).mean(), 0.01); c += 7
    # GLR over dyadic windows on the extended stream
    C1 = np.concatenate([[0.0], np.cumsum(ext)]); C2 = np.concatenate([[0.0], np.cumsum(ext ** 2)]); Cx = np.concatenate([[0.0], np.cumsum(ext[1:] * ext[:-1])])
    idx = np.arange(off, off + T) + 1     # exclusive end index into cumsums
    stats = {"mean": [], "scale": [], "dep": []}
    for m in DYADIC:
        lo = np.maximum(idx - m, 0); me = idx - lo          # the window is the last m values, or all there are
        s1 = C1[idx] - C1[lo]; s2 = C2[idx] - C2[lo]; sx = Cx[idx - 1] - Cx[np.maximum(lo - 1, 0)]
        var = np.maximum(s2 / me, 1e-6); r1 = np.clip(sx / np.maximum(s2, 1e-9), -0.99, 0.99)
        g_mean = np.abs(s1) / np.sqrt(me); g_scale = 0.5 * me * (var - 1 - np.log(var)); g_dep = -0.5 * me * np.log(1 - r1 ** 2)
        stats["mean"].append(g_mean); stats["scale"].append(g_scale); stats["dep"].append(g_dep)
        out[:, c] = g_mean; out[:, c + 1] = g_scale; out[:, c + 2] = g_dep; c += 3
    for key in ("mean", "scale", "dep"):
        out[:, c] = np.max(np.stack(stats[key]), 0); c += 1
    # distribution tests at the geometric cadence, held between
    cur = np.zeros(3 * (len(TESTW) + 1) + 3); nxt = 1
    for t in range(T):
        if t + 1 >= nxt:
            nxt = max(nxt + 1, int(nxt * 1.12)); vals = []
            for W in TESTW:
                vals.extend(one_sample_tests(ext[off + t + 1 - W: off + t + 1]))
            vals.extend(one_sample_tests(n[:t + 1]) if t + 1 >= 8 else (0.0, 0.0, 0.0))
            v = np.array(vals); cur = np.concatenate([v, v.reshape(-1, 3).max(0)])
        out[t, c: c + len(cur)] = cur
    c += len(cur)
    return c

def white_channels(hist, online):
    h = np.asarray(hist, float); mu, sd = h.mean(), h.std() + 1e-12; zh = (h - mu) / sd; zo = (np.asarray(online, float) - mu) / sd
    fit = fit_history(zh); n_c, n_u, lz = normal_scores_online(fit, zh, zo)
    T = len(zo); out = np.zeros((T, NCH), dtype="float32")
    c = _battery(n_u, fit["nhu"], out, 0)
    assert c == NFULL, (c, NFULL)
    # The conditional stream: sharper for mean and dependence when the history is heteroskedastic,
    # blind to scale by construction (the scale estimate adapts), so no scale statistics on it.
    nh = fit["nhc"]; tail = nh[-1024:]; ext = np.concatenate([tail, n_c]); off = len(tail); n = n_c
    out[:, c] = cusum_peak(n); out[:, c + 1] = ewma(n, 0.02); c += 2
    port = []
    for k in range(1, PORT + 1):
        a = ext[off:] * ext[off - k: len(ext) - k]; port.append(ewma(a, 0.01))
        if k in (1, 2, 5): out[:, c] = cusum_peak(a); c += 1
    out[:, c] = 199 * (np.stack(port) ** 2).sum(0); c += 1
    C1 = np.concatenate([[0.0], np.cumsum(ext)]); C2 = np.concatenate([[0.0], np.cumsum(ext ** 2)]); Cx = np.concatenate([[0.0], np.cumsum(ext[1:] * ext[:-1])])
    idx = np.arange(off, off + T) + 1; gm, gd = [], []
    for m in DYADIC:
        lo = np.maximum(idx - m, 0); me = idx - lo; s1 = C1[idx] - C1[lo]; s2 = C2[idx] - C2[lo]; sx = Cx[idx - 1] - Cx[np.maximum(lo - 1, 0)]
        r1 = np.clip(sx / np.maximum(s2, 1e-9), -0.99, 0.99); gm.append(np.abs(s1) / np.sqrt(me)); gd.append(-0.5 * me * np.log(1 - r1 ** 2))
    out[:, c] = np.max(np.stack(gm), 0); out[:, c + 1] = np.max(np.stack(gd), 0); c += 2
    cur = np.zeros(2); nxt = 1
    for t in range(T):
        if t + 1 >= nxt:
            nxt = max(nxt + 1, int(nxt * 1.12)); ks, ad = [], []
            for W in TESTW:
                k_, _, a_ = one_sample_tests(ext[off + t + 1 - W: off + t + 1]); ks.append(k_); ad.append(a_)
            cur = np.array([max(ks), max(ad)])
        out[t, c: c + 2] = cur
    c += 2
    # The scale process itself: a persistent shift of the conditional variance against the history's level.
    out[:, c] = lz; out[:, c + 1] = ewma(lz, 0.02); out[:, c + 2] = cusum_peak(lz); c += 3
    out[:, c] = np.arange(T); c += 1
    assert c == NCH, (c, NCH)
    out[:, :-1] = np.clip(out[:, :-1], -60, 60)          # the step index stays as it is
    return np.nan_to_num(out)

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0)
        def ar(phi, n, sd=1.0):
            x = np.zeros(n); e = rng.normal(0, sd, n)
            for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
            return x
        hist = ar(0.6, 3000) * (1 + 0.5 * np.sin(np.arange(3000) / 300))   # heteroskedastic AR history
        online = np.concatenate([ar(0.6, 300), ar(0.6, 300, sd=1.5)])
        t0 = time.time(); o = white_channels(hist, online); dt = (time.time() - t0) * 1000
        print(f"{o.shape[1]} channels (expected {NCH}), {dt:.0f} ms per series; scale break: scale CUSUM {o[250:300,4].mean():.2f} -> {o[450:600,4].mean():.2f}, scale GLR-max {o[250:300,53].mean():.1f} -> {o[450:600,53].mean():.1f}, KS-max {o[250:300,73].mean():.2f} -> {o[450:600,73].mean():.2f}, log-scale {o[250:300,86].mean():+.2f} -> {o[450:600,86].mean():+.2f}")
        online2 = np.concatenate([ar(0.6, 300), ar(0.9, 300, sd=np.sqrt((1 - 0.9 ** 2) / (1 - 0.6 ** 2)))]); o2 = white_channels(hist, online2)
        print(f"dependence break at equal variance: CUSUM lag1 {o2[250:300,8].mean():.2f} -> {o2[450:600,8].mean():.2f}, dependence GLR-max {o2[250:300,54].mean():.1f} -> {o2[450:600,54].mean():.1f}")
        sys.exit(0)
    if sys.argv[1] == "merge":
        n = int(sys.argv[2]); g = np.load("G40.npy"); s = np.load("S40.npy")
        st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g))
        out = np.empty((len(g), NCH), dtype="float32"); block = {}
        for i in range(n):
            p = np.load(f"{PARTS}/part_{i}.npz", allow_pickle=True)
            for sid, arr in zip(p["sids"], p["arrs"]): block[int(sid)] = arr
        for a_, b_ in zip(st, bd[1:]): out[a_:b_] = block[int(g[a_])][s[a_:b_]]
        np.save(f"{OUT}.npy", out); print(f"{OUT}: {out.shape}, nan {np.isnan(out).sum()}", flush=True); sys.exit(0)
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    import pandas as pd
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    ids = x.index.get_level_values("id"); x = x[(ids % n_shards) == shard]; sids, arrs = [], []
    for i, (sid, part) in enumerate(x.groupby(level="id")):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        sids.append(int(sid)); arrs.append(white_channels(hist, online))
        if (i + 1) % 250 == 0: print(f"shard {shard}: {i+1} series, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    np.savez(f"{PARTS}/part_{shard}.npz", sids=np.array(sids), arrs=np.array(arrs, dtype=object), allow_pickle=True)
    print(f"shard {shard} done {time.time()-t0:.0f}s: {len(sids)} series", flush=True)
