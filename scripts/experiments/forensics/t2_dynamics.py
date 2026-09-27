"""Task 2: dynamic structure of histories. Per-series features -> t2_features.csv
ACF/PACF to lag 20, ARCH (acf of squared residuals), unit-root (ADF-like, variance ratio), seasonality
(Fisher g on levels and on AR residuals), regime switching (rolling-variance bimodality), model fits:
AR(p) by BIC (p<=6), MA(1) via lfilter grid, ARMA(1,1) via Hannan-Rissanen. Multiprocessing with 4 workers."""
import os, sys, time
import numpy as np, pandas as pd
from scipy import stats, signal
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import acf, pacf_from_acf, ar_ols, periodogram, spectral_slope

OUT = os.path.dirname(os.path.abspath(__file__))
vals = None; off = None; meta = None

def init():
    global vals, off, meta
    vals = np.load(os.path.join(OUT, "values.npy"), mmap_mode="r")
    off = np.load(os.path.join(OUT, "offsets.npy"))
    meta = pd.read_csv(os.path.join(OUT, "meta.csv"))

def ljung_box(r, n, lags):
    r = np.asarray(r[1:lags + 1]); k = np.arange(1, lags + 1)
    return n * (n + 2) * np.sum(r ** 2 / (n - k))

def fisher_g(x):
    fr, P = periodogram(x)
    m = len(P)
    if m < 8: return np.nan, np.nan, np.nan, np.nan
    i = int(np.argmax(P)); g = P[i] / P.sum()
    logp = np.log10(m) + (m - 1) * np.log10(max(1 - g, 1e-300))   # log10 of Fisher p-value (approx)
    # peak concentration: mass within +-2 bins of peak relative to +-20 bins
    lo, hi = max(0, i - 2), min(m, i + 3); lo2, hi2 = max(0, i - 20), min(m, i + 21)
    conc = P[lo:hi].sum() / P[lo2:hi2].sum()
    return g, logp, 1.0 / fr[i], conc

def adf_t(x, k=4):
    d = np.diff(x); n = len(d)
    Y = d[k:]
    X = [x[k:-1]] + [d[k - j - 1: n - j - 1] for j in range(k)] + [np.ones(len(Y))]
    X = np.column_stack(X)
    beta, res, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    e = Y - X @ beta
    s2 = e @ e / (len(Y) - X.shape[1])
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta[0] / np.sqrt(cov[0, 0])

def var_ratio(x, q):
    d = np.diff(x)
    dq = x[q:] - x[:-q]
    return dq.var() / (q * d.var() + 1e-300)

def ma1_fit(x):
    """MA(1) by conditional least squares on a theta grid, refined once."""
    best = (np.inf, 0.0)
    for th in np.linspace(-0.95, 0.95, 39):
        e = signal.lfilter([1.0], [1.0, th], x)
        v = e[20:].var()
        if v < best[0]: best = (v, th)
    th0 = best[1]
    for th in np.linspace(th0 - 0.05, th0 + 0.05, 11):
        if abs(th) >= 0.999: continue
        e = signal.lfilter([1.0], [1.0, th], x)
        v = e[20:].var()
        if v < best[0]: best = (v, th)
    return best[1], best[0]

def feats(k):
    a = off[k]; h = int(meta.hist_len[k])
    x = np.asarray(vals[a:a + h], dtype=np.float64)
    n = len(x)
    r = dict(id=k, n=n)
    ac = acf(x, 60); pac = pacf_from_acf(ac, 20)
    for L in range(1, 21): r[f"acf{L}"] = ac[L]; r[f"pacf{L}"] = pac[L]
    r["acf30"] = ac[30]; r["acf50"] = ac[50]
    r["lb10"] = ljung_box(ac, n, 10)
    # AR(p) by BIC
    bics = []; fits = {}
    for p in range(0, 11):
        if p == 0:
            res = x - x.mean(); s2 = res.var(); coef = np.array([])
        else:
            coef, c, res, s2 = ar_ols(x, p)
        bics.append(n * np.log(s2) + (p + 1) * np.log(n)); fits[p] = (coef, res, s2)
    p_bic = int(np.argmin(bics)); r["ar_p_bic"] = p_bic
    aics = [n * np.log(fits[p][2]) + 2 * (p + 1) for p in range(11)]; r["ar_p_aic"] = int(np.argmin(aics))
    r["bic_gain_p1"] = bics[0] - bics[1]; r["bic_gain_best"] = bics[0] - bics[p_bic]
    for p in (1, 2, 3):
        for j in range(p): r[f"ar{p}_c{j+1}"] = fits[p][0][j]
        r[f"ar{p}_s2"] = fits[p][2]
    r["ar1_c1_of_best"] = fits[p_bic][0][0] if p_bic >= 1 else 0.0
    coef_b, res_b, s2_b = fits[p_bic]
    r["s2_best"] = s2_b
    # residual diagnostics
    r["res_skew"] = stats.skew(res_b); r["res_kurt"] = stats.kurtosis(res_b)
    acr = acf(res_b, 60); r["res_lb10"] = ljung_box(acr, len(res_b), 10)
    r["res_acf1"] = acr[1]; r["res_acf2"] = acr[2]
    sq = res_b ** 2; acs = acf(sq, 60)
    for L in (1, 2, 3, 5, 10, 20, 50): r[f"sqacf{L}"] = acs[L]
    r["sq_lb10"] = ljung_box(acs, len(sq), 10); r["sq_lb50"] = ljung_box(acs, len(sq), 50)
    aab = acf(np.abs(res_b), 60); r["absacf1"] = aab[1]; r["absacf10"] = aab[10]; r["absacf50"] = aab[50]
    # standardised residuals by rolling sd (centred window 21) -> kurtosis
    w = 21; c = np.convolve(sq, np.ones(w) / w, mode="same"); zs = res_b / np.sqrt(c + 1e-12)
    r["res_kurt_rollstd"] = stats.kurtosis(zs[w:-w])
    # ARCH-LM style: regress sq on 5 lags
    coef_a, c_a, res_a, s2_a = ar_ols(sq, 5); r["arch_r2"] = 1 - s2_a / sq.var()
    # unit root / trend
    r["adf_t"] = adf_t(x); r["vr5"] = var_ratio(x, 5); r["vr20"] = var_ratio(x, 20); r["vr100"] = var_ratio(x, 100)
    r["d_acf1"] = acf(np.diff(x), 2)[1]
    # linear trend
    t = np.arange(n); sl = np.polyfit(t, x, 1)[0]; r["trend_slope_sd"] = sl * n  # total drift over history in sd units
    # spectral
    r["g_lev"], r["g_lev_logp"], r["peak_period_lev"], r["peak_conc_lev"] = fisher_g(x)
    r["g_res"], r["g_res_logp"], r["peak_period_res"], r["peak_conc_res"] = fisher_g(res_b)
    r["spec_slope"] = spectral_slope(x)
    # rolling variance regime statistics (non-overlapping windows of 50 on residuals and on x)
    for name, series in (("x", x), ("res", res_b)):
        w = 50; m = len(series) // w
        rv = series[: m * w].reshape(m, w).var(axis=1)
        lrv = np.log(rv + 1e-12)
        g1 = stats.skew(lrv); k1 = stats.kurtosis(lrv)
        bc = (g1 ** 2 + 1) / (k1 + 3 * (m - 1) ** 2 / ((m - 2) * (m - 3)))
        r[f"rv_bc_{name}"] = bc
        r[f"rv_cv_{name}"] = rv.std() / rv.mean()
        r[f"rv_q90q10_{name}"] = np.quantile(rv, 0.9) / (np.quantile(rv, 0.1) + 1e-12)
        r[f"rv_acf1_{name}"] = np.corrcoef(rv[:-1], rv[1:])[0, 1] if m > 5 else np.nan
        r[f"rv_maxmin_{name}"] = rv.max() / (rv.min() + 1e-12)
    # 2-means split of log rolling variance (window 50, on x): separation and share of high regime
    w = 50; m = n // w; lrv = np.log(x[: m * w].reshape(m, w).var(axis=1) + 1e-12)
    s = np.sort(lrv); best = (np.inf, 0)
    for j in range(2, m - 2):
        ssw = s[:j].var() * j + s[j:].var() * (m - j)
        if ssw < best[0]: best = (ssw, j)
    j = best[1]; r["rv_split_gap"] = (s[j:].mean() - s[:j].mean()) / (np.sqrt(best[0] / m) + 1e-12); r["rv_split_hi_share"] = 1 - j / m
    # MA(1) and ARMA(1,1) (Hannan-Rissanen with long AR(20))
    th, v_ma = ma1_fit(x); r["ma1_theta"] = th; r["ma1_s2"] = v_ma
    r["bic_ma1"] = n * np.log(v_ma) + 2 * np.log(n)
    coef20, c20, e20, s20 = ar_ols(x, 20)
    e_full = np.concatenate([np.zeros(20), e20])
    Y = x[21:]; X = np.column_stack([x[20:-1], e_full[20:-1], np.ones(len(Y))])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None); eh = Y - X @ beta
    r["arma_phi"] = beta[0]; r["arma_theta"] = beta[1]; r["arma_s2"] = eh.var()
    r["bic_arma11"] = n * np.log(eh.var()) + 3 * np.log(n)
    r["bic_ar_best"] = bics[p_bic]; r["bic_wn"] = bics[0]
    return r

if __name__ == "__main__":
    init()
    N = len(meta); t0 = time.time()
    with Pool(4, initializer=init) as pool:
        rows = []
        for i, r in enumerate(pool.imap(feats, range(N), chunksize=50)):
            rows.append(r)
            if i % 1000 == 0: print(i, round(time.time() - t0, 1), "s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "t2_features.csv"), index=False)
    print("done", round(time.time() - t0, 1), "s")
