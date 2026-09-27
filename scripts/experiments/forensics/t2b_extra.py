"""Task 2b: extra per-series diagnostics on histories (4 workers) -> t2b_features.csv
- robust ARCH test: Ljung-Box(10) on ranks of |AR residuals| (tail-robust)
- quantile-based Student-t df estimate of AR residuals (and of rolling-standardised residuals)
- GARCH(1,1) grid QMLE on AR residuals (alpha, beta), for every series (cheap enough via lfilter)
- half-history stability: sd ratio, mean diff, acf1 diff between first/second half
- rolling-mean bimodality (w=50): 2-means split gap
- run-length / sign statistics
Also a simulation calibration of thresholds under iid Gaussian and t(3) noise -> t2b_calibration.txt
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy import stats, signal
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import acf, ar_ols

OUT = os.path.dirname(os.path.abspath(__file__))
vals = off = meta = None

def init():
    global vals, off, meta
    vals = np.load(os.path.join(OUT, "values.npy"), mmap_mode="r")
    off = np.load(os.path.join(OUT, "offsets.npy"))
    meta = pd.read_csv(os.path.join(OUT, "meta.csv"))

def ljung_box(r, n, lags):
    r = np.asarray(r[1:lags + 1]); k = np.arange(1, lags + 1)
    return n * (n + 2) * np.sum(r ** 2 / (n - k))

# quantile ratio -> t df lookup (theoretical (q0.99-q0.01)/(q0.75-q0.25))
DFS = np.concatenate([np.linspace(1.2, 6, 49), np.linspace(6.2, 30, 60), [40, 60, 100, 1e6]])
RATIOS = np.array([(stats.t.ppf(0.99, d) - stats.t.ppf(0.01, d)) / (stats.t.ppf(0.75, d) - stats.t.ppf(0.25, d)) for d in DFS])
def df_from_quantiles(e):
    q = np.quantile(e, [0.01, 0.25, 0.75, 0.99]); rr = (q[3] - q[0]) / (q[2] - q[1] + 1e-12)
    if rr <= RATIOS[-1]: return 1e6
    if rr >= RATIOS[0]: return DFS[0]
    return float(np.interp(-rr, -RATIOS, DFS))   # RATIOS decreasing in df

def garch_grid(e):
    """Gaussian QML GARCH(1,1) on zero-mean residuals e; omega = (1-a-b)*var. Grid then local refine."""
    e2 = e ** 2; v = e2.mean(); n = len(e)
    def nll(a, b):
        if a < 0 or b < 0 or a + b >= 0.999: return np.inf
        om = (1 - a - b) * v
        s2 = om / (1 - b) + signal.lfilter([a], [1.0, -b], np.concatenate([[v], e2[:-1]]))
        s2 = np.maximum(s2, 1e-10)
        return 0.5 * np.sum(np.log(s2) + e2 / s2)
    best = (nll(0.0, 0.0), 0.0, 0.0)
    for a in np.arange(0.02, 0.62, 0.04):
        for b in np.arange(0.0, 0.99, 0.05):
            f = nll(a, b)
            if f < best[0]: best = (f, a, b)
    f0, a0, b0 = best
    for a in np.arange(a0 - 0.03, a0 + 0.031, 0.01):
        for b in np.arange(b0 - 0.04, b0 + 0.041, 0.01):
            f = nll(a, b)
            if f < best[0]: best = (f, a, b)
    return best[1], best[2], (nll(0.0, 0.0) - best[0])   # log-lik gain vs constant variance

def feats(k):
    a = off[k]; h = int(meta.hist_len[k])
    x = np.asarray(vals[a:a + h], dtype=np.float64); n = len(x)
    r = dict(id=k)
    # AR by BIC (p<=10) as in t2
    bics = []; fits = {}
    for p in range(0, 11):
        if p == 0: res = x - x.mean(); s2 = res.var(); coef = np.array([])
        else: coef, c, res, s2 = ar_ols(x, p)
        bics.append(n * np.log(s2) + (p + 1) * np.log(n)); fits[p] = (coef, res, s2)
    p = int(np.argmin(bics)); coef, e, s2 = fits[p]
    e = e - e.mean()
    ae = np.abs(e); rk = stats.rankdata(ae)
    r["rk_lb10"] = ljung_box(acf(rk, 12), len(rk), 10)
    r["rk_lb50"] = ljung_box(acf(rk, 52), len(rk), 50)
    r["rk_acf1"] = acf(rk, 2)[1]
    r["df_q"] = df_from_quantiles(e / e.std())
    w = 21; c = np.convolve(e ** 2, np.ones(w) / w, mode="same"); zs = (e / np.sqrt(c + 1e-12))[w:-w]
    r["df_q_roll"] = df_from_quantiles(zs / zs.std())
    r["res_kurt"] = stats.kurtosis(e); r["res_skew"] = stats.skew(e)
    # tail asymmetry of residuals
    q = np.quantile(e, [0.01, 0.5, 0.99]); r["res_tailasym"] = (q[2] - q[1]) / (q[1] - q[0] + 1e-12)
    # GARCH
    r["g_alpha"], r["g_beta"], r["g_llgain"] = garch_grid(e)
    # half-history stability
    h2 = n // 2; x1, x2 = x[:h2], x[h2:]
    r["half_sd_ratio"] = x2.std() / (x1.std() + 1e-12); r["half_mean_diff"] = (x2.mean() - x1.mean())
    r["half_acf1_diff"] = acf(x2, 2)[1] - acf(x1, 2)[1]
    # thirds: max/min sd across 3 blocks
    b3 = np.array_split(x, 3); sds = [b.std() for b in b3]; r["third_sd_maxmin"] = max(sds) / (min(sds) + 1e-12)
    # rolling mean bimodality (w=50 non-overlapping) on x
    w = 50; m = n // w; rm = x[: m * w].reshape(m, w).mean(axis=1)
    s = np.sort(rm); best = (np.inf, 0)
    for j in range(2, m - 2):
        ssw = s[:j].var() * j + s[j:].var() * (m - j)
        if ssw < best[0]: best = (ssw, j)
    j = best[1]; r["rm_split_gap"] = (s[j:].mean() - s[:j].mean()) / (np.sqrt(best[0] / m) + 1e-12)
    r["rm_sd_over_expected"] = rm.std() / (x.std() / np.sqrt(w) * np.sqrt(max(1e-3, (1 + acf(x, 2)[1]) / (1 - acf(x, 2)[1]))))
    r["rm_acf1"] = np.corrcoef(rm[:-1], rm[1:])[0, 1]
    # CUSUM of mean and of squares on history (max abs standardised)
    cs = np.cumsum(x - x.mean()) / (x.std() * np.sqrt(n)); r["cusum_mean"] = np.abs(cs).max()
    x2c = x ** 2; cs2 = np.cumsum(x2c - x2c.mean()) / (x2c.std() * np.sqrt(n)); r["cusum_var"] = np.abs(cs2).max()
    # sign runs on residuals
    sgn = e > 0; runs = 1 + np.sum(sgn[1:] != sgn[:-1]); r["runs_z"] = (runs - (n / 2 + 1)) / np.sqrt(n / 4)
    r["ar_p_bic"] = p
    return r

def calib():
    rng = np.random.default_rng(0); out = []
    for name, gen in (("gauss", lambda n: rng.standard_normal(n)), ("t3", lambda n: rng.standard_t(3, n)), ("t5", lambda n: rng.standard_t(5, n))):
        rows = []
        for i in range(600):
            n = int(rng.integers(1000, 5001)); e = gen(n); e = e - e.mean()
            rk = stats.rankdata(np.abs(e))
            w = 50; m = n // w; rv = e[: m * w].reshape(m, w).var(axis=1); lrv = np.log(rv)
            g1 = stats.skew(lrv); k1 = stats.kurtosis(lrv); bc = (g1 ** 2 + 1) / (k1 + 3 * (m - 1) ** 2 / ((m - 2) * (m - 3)))
            s = np.sort(lrv); best = (np.inf, 0)
            for j in range(2, m - 2):
                ssw = s[:j].var() * j + s[j:].var() * (m - j)
                if ssw < best[0]: best = (ssw, j)
            gap = (s[best[1]:].mean() - s[:best[1]].mean()) / np.sqrt(best[0] / m)
            h2 = n // 2
            rows.append(dict(rk_lb10=ljung_box(acf(rk, 12), n, 10), rv_bc=bc, rv_gap=gap, rv_acf1=np.corrcoef(rv[:-1], rv[1:])[0, 1],
                             kurt=stats.kurtosis(e), skew=stats.skew(e), df_q=df_from_quantiles(e / e.std()),
                             half_sd_ratio=e[h2:].std() / e[:h2].std(), third_sd_maxmin=max(b.std() for b in np.array_split(e, 3)) / min(b.std() for b in np.array_split(e, 3)),
                             cusum_var=np.abs(np.cumsum(e ** 2 - (e ** 2).mean()) / ((e ** 2).std() * np.sqrt(n))).max(),
                             sq_lb10=ljung_box(acf(e ** 2, 12), n, 10)))
        d = pd.DataFrame(rows)
        out.append(f"== iid {name} (600 sims, n~U(1000,5000)) quantiles 50/95/99/99.9:")
        for c in d.columns:
            out.append(f"  {c:16s} " + " ".join(f"{v:9.4g}" for v in np.quantile(d[c], [0.5, 0.95, 0.99, 0.999])))
    txt = "\n".join(out); print(txt); open(os.path.join(OUT, "t2b_calibration.txt"), "w").write(txt)

if __name__ == "__main__":
    calib()
    init(); N = len(meta); t0 = time.time()
    with Pool(4, initializer=init) as pool:
        rows = []
        for i, r in enumerate(pool.imap(feats, range(N), chunksize=50)):
            rows.append(r)
            if i % 1000 == 0: print(i, round(time.time() - t0, 1), "s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "t2b_features.csv"), index=False)
    print("done", round(time.time() - t0, 1), "s")
