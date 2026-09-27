"""Task 4: broad statistic set on matched windows for every break with post_len>=30 and 2 length-matched controls each
(non-break series cut so that the control post window has the same length; control rows carry target=id of the break).
Windows: post = all post-break points (<=999); pre = last min(post_len,1000) points before the cut. Both standardised by the
FULL pre segment (history + online pre) mean/sd. -> t4_stats.csv"""
import os, sys, time, math, numpy as np, pandas as pd
from scipy import stats
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import acf, ar_ols, spectral_slope
OUT = os.path.dirname(os.path.abspath(__file__))
vals = off = meta = fam = None
def init():
    global vals, off, meta, fam
    vals = np.load(os.path.join(OUT, "values.npy"), mmap_mode="r"); off = np.load(os.path.join(OUT, "offsets.npy"))
    meta = pd.read_csv(os.path.join(OUT, "meta.csv")); fam = pd.read_csv(os.path.join(OUT, "families.csv"), usecols=["id", "ar_p_bic", "lin", "vol", "innov"])

def perm_entropy(x, m):
    n = len(x) - m + 1
    if n < 10: return np.nan
    idx = np.arange(m)[None, :] + np.arange(n)[:, None]
    pat = np.argsort(x[idx], axis=1)
    codes = (pat * (m ** np.arange(m))).sum(axis=1)
    _, c = np.unique(codes, return_counts=True); p = c / c.sum()
    return -np.sum(p * np.log(p)) / np.log(math.factorial(m))

def sample_entropy(x, m=2, r=0.2):
    n = len(x)
    if n < 50: return np.nan
    x = x[:1000]; n = len(x)
    def count(mm):
        N = n - m   # same number of templates for m and m+1 (standard definition)
        tot = 0
        for i0 in range(0, N, 250):
            i1 = min(N, i0 + 250)
            d = np.zeros((i1 - i0, N), bool) | True
            for j in range(mm):
                d &= np.abs(x[i0 + j:i1 + j][:, None] - x[j:N + j][None, :]) <= r
            # exclude self matches
            for i in range(i0, i1): d[i - i0, i] = False
            tot += d.sum()
        return tot
    B = count(m); A = count(m + 1)
    if A == 0 or B == 0: return np.nan
    return -np.log(A / B)

def dfa(x):
    n = len(x)
    if n < 64: return np.nan
    y = np.cumsum(x - x.mean()); sizes = np.unique(np.geomspace(8, n // 4, 8).astype(int)); F = []
    for s in sizes:
        nb = n // s; segs = y[: nb * s].reshape(nb, s); t = np.arange(s)
        A = np.column_stack([t, np.ones(s)]); coef = np.linalg.lstsq(A, segs.T, rcond=None)[0]
        res = segs.T - A @ coef; F.append(np.sqrt(np.mean(res ** 2)))
    return np.polyfit(np.log(sizes), np.log(np.array(F) + 1e-12), 1)[0]

def hurst_rs(x):
    n = len(x)
    if n < 64: return np.nan
    sizes = np.unique(np.geomspace(16, n // 2, 6).astype(int)); rs = []
    for s in sizes:
        nb = n // s; segs = x[: nb * s].reshape(nb, s); v = []
        for seg in segs:
            z = np.cumsum(seg - seg.mean()); R = z.max() - z.min(); S = seg.std()
            if S > 0: v.append(R / S)
        rs.append(np.mean(v))
    return np.polyfit(np.log(sizes), np.log(np.array(rs) + 1e-12), 1)[0]

def bds_proxy(x, eps=1.0):
    x = x[:600]; n = len(x) - 1
    d1 = np.abs(x[:n][:, None] - x[:n][None, :]) <= eps; d2 = d1 & (np.abs(x[1:n + 1][:, None] - x[1:n + 1][None, :]) <= eps)
    iu = np.triu_indices(n, 1); c1 = d1[iu].mean(); c2 = d2[iu].mean()
    return c2 - c1 ** 2

def win_stats(w, pre_full, p, coef, const, s2pre, prefix):
    """w: window (standardised by full pre); pre_full: standardised full pre segment for quantile thresholds"""
    r = {}; n = len(w); r[prefix + "n"] = n
    r[prefix + "mean"] = w.mean(); r[prefix + "sd"] = w.std(); r[prefix + "mad"] = 1.4826 * np.median(np.abs(w - np.median(w)))
    ac = acf(w, 55)
    for L in (1, 2, 3, 4, 5, 10): r[prefix + f"acf{L}"] = ac[L]
    aw = np.abs(w - w.mean()); aa = acf(aw, 55); r[prefix + "absacf_1_5"] = aa[1:6].mean(); r[prefix + "absacf_6_20"] = aa[6:21].mean(); r[prefix + "absacf_21_50"] = aa[21:51].mean() if n > 120 else np.nan
    r[prefix + "sqacf1"] = acf((w - w.mean()) ** 2, 2)[1]
    r[prefix + "kurt"] = stats.kurtosis(w); r[prefix + "skew"] = stats.skew(w)
    r[prefix + "hurst"] = hurst_rs(w); r[prefix + "dfa"] = dfa(w)
    r[prefix + "pe3"] = perm_entropy(w, 3); r[prefix + "pe4"] = perm_entropy(w, 4)
    r[prefix + "sampen"] = sample_entropy(w, 2, 0.2 * pre_full.std())
    r[prefix + "spec_slope"] = spectral_slope(w) if n >= 64 else np.nan
    d = np.diff(w); thr = 4 * 1.4826 * np.median(np.abs(np.diff(pre_full) - np.median(np.diff(pre_full))))
    r[prefix + "jumpfrac"] = np.mean(np.abs(d) > thr)
    med = np.median(pre_full); sgn = w > med; runs = 1 + np.sum(sgn[1:] != sgn[:-1]); n1 = sgn.sum(); n2 = n - n1
    mu = 1 + 2 * n1 * n2 / n; var = (mu - 1) * (mu - 2) / (n - 1) if n > 1 else 1
    r[prefix + "runs_z"] = (runs - mu) / np.sqrt(var + 1e-12)
    r[prefix + "trev1"] = np.mean(d ** 3) / (np.mean(d ** 2) ** 1.5 + 1e-12); r[prefix + "trev2"] = np.mean(w[1:] ** 2 * w[:-1]) - np.mean(w[1:] * w[:-1] ** 2)
    r[prefix + "bds"] = bds_proxy(w, 1.0)
    q10, q50, q90 = np.quantile(pre_full, [0.1, 0.5, 0.9]); r[prefix + "up90"] = np.mean(w > q90); r[prefix + "lo10"] = np.mean(w < q10); r[prefix + "up50"] = np.mean(w > q50)
    r[prefix + "zc"] = np.mean(np.sign(w[1:] - med) != np.sign(w[:-1] - med))
    r[prefix + "tail2"] = np.mean(np.abs(w) > 2)
    b = 10; nb = n // b; bm = w[: nb * b].reshape(nb, b).mean(axis=1); r[prefix + "aggvar"] = bm.var() * b / (w.var() + 1e-12)
    # residuals under pre-fitted AR(p)
    q = max(p, 1)
    if n > q + 10:
        X = np.column_stack([w[q - j - 1: n - j - 1] for j in range(q)] + [np.ones(n - q)]); e = w[q:] - X @ np.append(coef, const)
        r[prefix + "res_var"] = e.var() / s2pre; r[prefix + "res_kurt"] = stats.kurtosis(e); r[prefix + "res_skew"] = stats.skew(e); r[prefix + "res_acf1"] = acf(e, 2)[1]
        r[prefix + "res_sqacf1"] = acf(e ** 2, 2)[1]
    return r

def pair(k, cut, is_break, target):
    a = off[k]; h = int(meta.hist_len[k]); o = int(meta.onl_len[k]); x = np.asarray(vals[a:a + h + o], float)
    pre_full = x[: h + cut]; post = x[h + cut:]; mu, sd = pre_full.mean(), pre_full.std()
    zpre = (pre_full - mu) / sd; zpost = (post - mu) / sd
    L = min(len(post), 1000); wpre = zpre[-L:]
    p = int(fam.ar_p_bic[k]); q = max(p, 1); coef, const, res, s2 = ar_ols(zpre, q)
    r = dict(id=k, is_break=is_break, target=target, cut=cut, onl_len=o, post_len=len(post), lin=fam.lin[k], vol=fam.vol[k], innov=fam.innov[k])
    r.update(win_stats(wpre, zpre, p, coef, const, s2, "pre_")); r.update(win_stats(zpost, zpre, p, coef, const, s2, "post_"))
    r["ks"] = stats.ks_2samp(zpre, zpost).statistic; r["wass"] = stats.wasserstein_distance(zpre, zpost)
    r["ks_win"] = stats.ks_2samp(wpre, zpost).statistic
    return r

def work(job): return pair(*job)

if __name__ == "__main__":
    init(); rng = np.random.default_rng(1); jobs = []
    nb_ids = meta.index[meta.tau_index < 0].to_numpy(); nb_len = meta.onl_len.to_numpy()
    for k in range(len(meta)):
        ti = int(meta.tau_index[k]); o = int(meta.onl_len[k])
        if ti < 0: continue
        pl = o - ti
        if pl < 30: continue
        jobs.append((k, ti, 1, k))
        cand = nb_ids[nb_len[nb_ids] > pl]
        for c in rng.choice(cand, 2, replace=False):
            jobs.append((int(c), int(nb_len[c] - pl), 0, k))
    print("jobs", len(jobs)); t0 = time.time()
    with Pool(4, initializer=init) as pool:
        rows = []
        for i, r in enumerate(pool.imap(work, jobs, chunksize=20)):
            rows.append(r)
            if i % 2000 == 0: print(i, round(time.time() - t0, 1), "s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "t4_stats.csv"), index=False); print("done", round(time.time() - t0, 1), "s")
