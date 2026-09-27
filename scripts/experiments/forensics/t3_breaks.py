"""Task 3: break characterisation. pre = full history + online[:tau], post = online[tau:] (all points).
Controls: non-break series cut at 2 random points of the online part (uniform relative position, seed 0).
Per segment pair compute changes in mean, sd, AR(1..3) OLS coefficients, kurtosis, skew, acf1, acf of squared
residuals, Hill tail index, spectral slope, innovation-variance ratio under the pre-fitted AR(p_bic) model,
KS / Wasserstein distances (post vs pre, standardised by pre). -> t3_stats.csv"""
import os, sys, time, numpy as np, pandas as pd
from scipy import stats
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import acf, ar_ols, spectral_slope, hill
OUT = os.path.dirname(os.path.abspath(__file__))
vals = off = meta = fam = None
def init():
    global vals, off, meta, fam
    vals = np.load(os.path.join(OUT, "values.npy"), mmap_mode="r"); off = np.load(os.path.join(OUT, "offsets.npy"))
    meta = pd.read_csv(os.path.join(OUT, "meta.csv")); fam = pd.read_csv(os.path.join(OUT, "families.csv"), usecols=["id", "ar_p_bic", "lin", "vol", "innov"])

def seg_stats(x, prefix, p):
    r = {}
    n = len(x); r[prefix + "n"] = n
    r[prefix + "mean"] = x.mean(); r[prefix + "sd"] = x.std()
    r[prefix + "kurt"] = stats.kurtosis(x); r[prefix + "skew"] = stats.skew(x)
    ac = acf(x, 5); r[prefix + "acf1"] = ac[1]; r[prefix + "acf2"] = ac[2]
    for q in (1, 2, 3):
        c, _, res, s2 = ar_ols(x, q)
        for j in range(q): r[f"{prefix}ar{q}_c{j+1}"] = c[j]
        if q == 1: r[prefix + "ar1_s2"] = s2
    c, _, res, s2 = ar_ols(x, max(p, 1)); e = res - res.mean()
    r[prefix + "res_kurt"] = stats.kurtosis(e); r[prefix + "res_skew"] = stats.skew(e)
    r[prefix + "sqacf1"] = acf(e ** 2, 2)[1]; r[prefix + "absacf1"] = acf(np.abs(e), 2)[1]
    r[prefix + "hill"] = hill(x, 0.05); r[prefix + "hill_res"] = hill(e, 0.05)
    r[prefix + "spec_slope"] = spectral_slope(x) if n >= 64 else np.nan
    # tail quantiles standardised
    z = (x - x.mean()) / (x.std() + 1e-12); q = np.quantile(z, [0.01, 0.05, 0.95, 0.99]); r[prefix + "q99_01"] = q[3] - q[0]; r[prefix + "q95_05"] = q[2] - q[1]
    r[prefix + "maxabs"] = np.abs(z).max()
    r[prefix + "jumps4"] = int(np.sum(np.abs(np.diff(x)) > 4 * 1.4826 * np.median(np.abs(np.diff(x) - np.median(np.diff(x)))) + 1e-12))
    return r

def pair(k, cut, is_break):
    a = off[k]; h = int(meta.hist_len[k]); o = int(meta.onl_len[k])
    x = np.asarray(vals[a:a + h + o], float)
    pre = x[: h + cut]; post = x[h + cut:]
    p = int(fam.ar_p_bic[k])
    r = dict(id=k, is_break=is_break, cut=cut, onl_len=o, hist_len=h, post_len=len(post), lin=fam.lin[k], vol=fam.vol[k], innov=fam.innov[k], p_bic=p)
    if len(post) < 20: return r
    r.update(seg_stats(pre, "pre_", p)); r.update(seg_stats(post, "post_", p))
    # online pre-break part alone (sanity: does it match history?)
    if cut >= 30:
        r.update({("onlpre_" + kk[4:]): v for kk, v in seg_stats(x[h:h + cut], "pre_", p).items() if kk in ("pre_mean", "pre_sd", "pre_acf1", "pre_kurt", "pre_n")})
    # innovation variance ratio under pre-fitted AR(p) model (p = max(p_bic,1)) with constant
    q = max(p, 1); c, const, res_pre, s2_pre = ar_ols(pre, q)
    X = np.column_stack([post[q - j - 1: len(post) - j - 1] for j in range(q)] + [np.ones(len(post) - q)])
    res_post = post[q:] - X @ np.append(c, const)
    r["innov_var_ratio"] = res_post.var() / s2_pre; r["innov_mean_shift"] = res_post.mean() / np.sqrt(s2_pre)
    # distances: post vs pre standardised by pre
    zpre = (pre - pre.mean()) / pre.std(); zpost = (post - pre.mean()) / pre.std()
    r["ks"] = stats.ks_2samp(zpre, zpost).statistic; r["wass"] = stats.wasserstein_distance(zpre, zpost)
    # derived changes
    r["d_mean"] = (post.mean() - pre.mean()) / pre.std(); r["sd_ratio"] = post.std() / pre.std()
    r["d_ar1"] = r["post_ar1_c1"] - r["pre_ar1_c1"]; r["d_kurt"] = r["post_kurt"] - r["pre_kurt"]; r["d_skew"] = r["post_skew"] - r["pre_skew"]
    r["d_res_kurt"] = r["post_res_kurt"] - r["pre_res_kurt"]; r["d_sqacf1"] = r["post_sqacf1"] - r["pre_sqacf1"]
    r["hill_ratio"] = r["post_hill"] / r["pre_hill"] if r["pre_hill"] > 0 else np.nan; r["d_spec_slope"] = r["post_spec_slope"] - r["pre_spec_slope"]
    r["d_ar2"] = np.hypot(r["post_ar2_c1"] - r["pre_ar2_c1"], r["post_ar2_c2"] - r["pre_ar2_c2"])
    r["d_ar3"] = np.sqrt(sum((r[f"post_ar3_c{j}"] - r[f"pre_ar3_c{j}"]) ** 2 for j in (1, 2, 3)))
    return r

def work(job):
    k, cut, is_break = job
    return pair(k, cut, is_break)

if __name__ == "__main__":
    init(); rng = np.random.default_rng(0); jobs = []
    for k in range(len(meta)):
        ti = int(meta.tau_index[k]); o = int(meta.onl_len[k])
        if ti >= 0: jobs.append((k, ti, 1))
        else:
            for rep in range(2):
                jobs.append((k, int(rng.integers(0, o)), 0))
    t0 = time.time()
    with Pool(4, initializer=init) as pool:
        rows = list(pool.imap(work, jobs, chunksize=50))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "t3_stats.csv"), index=False)
    print("done", len(df), "rows", round(time.time() - t0, 1), "s")
