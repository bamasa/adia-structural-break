"""144: dependence-score CUSUM — the sequential statistic the dependence breaks were missing.

The history fits an AR(5); its residuals are white by construction. If the online
dependence changes, residuals computed with the history's coefficients become
autocorrelated, and the score for a change in the lag-k coefficient is the sum of
lagged residual products. Page's two-sided CUSUM on standardised products (drift
0.25) is the sequential test with the best power for a small sustained change,
exactly as a CUSUM on squares is for variance. Per lag 1,2,3,5,10: CUSUM peak and
two EWMAs of the product; a portmanteau over lags 1..10 at two horizons; absolute
products at lags 1,2,5 (volatility clustering) with CUSUM peak and EWMA. 23 channels.
"""
import sys, time, numpy as np, pandas as pd
from scipy.signal import lfilter
P = 5; LAGS = (1, 2, 3, 5, 10); ALAGS = (1, 2, 5); DRIFT = 0.25; PORT = 10
NCH = len(LAGS) * 3 + 2 + len(ALAGS) * 2

def ewma(a, alpha):
    return lfilter([alpha], [1, -(1 - alpha)], a)

def cusum_peak(a, drift=DRIFT):
    cp = np.concatenate([[0.0], np.cumsum(a - drift)]); cm = np.concatenate([[0.0], np.cumsum(-a - drift)])
    return np.maximum(cp[1:] - np.minimum.accumulate(cp)[1:], cm[1:] - np.minimum.accumulate(cm)[1:])

def dep_channels(hist, online):
    h = np.asarray(hist, float); mu, sd = h.mean(), h.std() + 1e-12
    zh = (h - mu) / sd; zo = (np.asarray(online, float) - mu) / sd
    X = np.column_stack([zh[P - k - 1: len(zh) - k - 1] for k in range(P)])
    coef = np.linalg.lstsq(X, zh[P:], rcond=None)[0]
    full = np.concatenate([zh, zo])
    e = full[P:] - sum(coef[k] * full[P - k - 1: len(full) - k - 1] for k in range(P))
    nh = len(zh) - P                      # history residuals: e[:nh]; online: e[nh:]
    u = e / (e[:nh].std() + 1e-9)
    n = len(zo); out = np.zeros((n, NCH), dtype="float32"); c = 0; port = []
    for k in range(1, PORT + 1):
        q = u[k:] * u[:-k]                # q[i] pairs u[i+k] with u[i]; q index i+k-k ... aligned to u[k:]
        qh, qo = q[:nh - k], q[nh - k:]   # products whose later member is a history / online residual
        m, s = qh.mean(), qh.std() + 1e-9
        a = (qo - m) / s
        port.append(a)
        if k in LAGS:
            out[:, c] = cusum_peak(a); out[:, c + 1] = ewma(a, 0.02); out[:, c + 2] = ewma(a, 0.005); c += 3
    E1 = np.stack([ewma(a, 0.01) for a in port]); E2 = np.stack([ewma(a, 0.003) for a in port])
    out[:, c] = 199 * (E1 ** 2).sum(0); out[:, c + 1] = 666 * (E2 ** 2).sum(0); c += 2
    au = np.abs(u)
    for k in ALAGS:
        r = au[k:] * au[:-k]; rh, ro = r[:nh - k], r[nh - k:]
        a = (ro - rh.mean()) / (rh.std() + 1e-9)
        out[:, c] = cusum_peak(a); out[:, c + 1] = ewma(a, 0.01); c += 2
    assert c == NCH
    return np.clip(np.nan_to_num(out), -50, 50)

if __name__ == "__main__":
    if sys.argv[1] == "test":
        rng = np.random.default_rng(0)
        def ar(phi, n, sd=1.0):
            x = np.zeros(n); e = rng.normal(0, sd, n)
            for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
            return x
        hist = ar(0.3, 3000); online = np.concatenate([ar(0.3, 300), ar(0.6, 300) * np.sqrt((1 - 0.6 ** 2) / (1 - 0.3 ** 2))])  # same marginal variance
        t0 = time.time(); o = dep_channels(hist, online); dt = (time.time() - t0) * 1000
        print(f"{o.shape[1]} channels, {dt:.0f} ms per series; dependence break at equal variance: CUSUM lag1 {o[250:300,0].mean():.2f} -> {o[450:600,0].mean():.2f}, EWMA lag1 {o[250:300,1].mean():+.2f} -> {o[450:600,1].mean():+.2f}, portmanteau {o[250:300,15].mean():.1f} -> {o[450:600,15].mean():.1f}")
        sys.exit(0)
    t0 = time.time(); x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
    g = np.load("G40.npy"); s = np.load("S40.npy"); out = np.empty((len(g), NCH), dtype="float32")
    st = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); bd = np.append(st, len(g)); pos = {int(g[a]): (a, b) for a, b in zip(st, bd[1:])}
    for i, (sid, part) in enumerate(x.groupby(level="id")):
        hist = part.loc[part.period == 1, "value"].to_numpy("float64"); online = part.loc[part.period == 2, "value"].to_numpy("float64")
        a, b = pos[int(sid)]; out[a:b] = dep_channels(hist, online)[s[a:b]]
        if (i + 1) % 2000 == 0: print(f"{i+1} series, {time.time()-t0:.0f}s", flush=True)
    np.save("DEP23.npy", out); print(f"DEP23: {out.shape}, nan {np.isnan(out).sum()} [{time.time()-t0:.0f}s]", flush=True)
