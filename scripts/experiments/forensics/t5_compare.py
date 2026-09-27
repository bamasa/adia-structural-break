"""Task 5b: compare 500 synthetic series (synth.py) with 500 random real series on summary statistics.
History statistics (per series) and break statistics (post vs pre; breaks and non-break controls cut at a random point).
Prints side-by-side quantile tables and KS p-values (real vs synthetic). -> t5_compare.txt"""
import os, sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import acf, ar_ols, spectral_slope
from t2b_extra import df_from_quantiles
OUT = os.path.dirname(os.path.abspath(__file__))

def ljung_box(r, n, lags):
    r = np.asarray(r[1:lags + 1]); k = np.arange(1, lags + 1); return n * (n + 2) * np.sum(r ** 2 / (n - k))

def hist_stats(x):
    n = len(x); r = dict(kurt=stats.kurtosis(x), skew=stats.skew(x))
    ac = acf(x, 20); r["acf1"], r["acf2"], r["acf5"], r["acf10"] = ac[1], ac[2], ac[5], ac[10]
    bics = []; fits = {}
    for p in range(0, 11):
        if p == 0: res = x - x.mean(); s2 = res.var()
        else: _, _, res, s2 = ar_ols(x, p)
        bics.append(n * np.log(s2) + (p + 1) * np.log(n)); fits[p] = res
    p = int(np.argmin(bics)); e = fits[p] - fits[p].mean(); r["ar_p_bic"] = p
    r["res_kurt"] = stats.kurtosis(e); r["res_df"] = min(df_from_quantiles(e / e.std()), 100)
    r["rk_lb10"] = ljung_box(acf(stats.rankdata(np.abs(e)), 12), len(e), 10)
    r["sqacf1"] = acf(e ** 2, 2)[1]
    w = 50; m = len(e) // w; rv = e[: m * w].reshape(m, w).var(axis=1); r["rv_acf1"] = np.corrcoef(rv[:-1], rv[1:])[0, 1]
    r["spec_slope"] = spectral_slope(x); r["maxabs_z"] = np.abs((x - x.mean()) / x.std()).max()
    r["third_sd_maxmin"] = max(b.std() for b in np.array_split(x, 3)) / min(b.std() for b in np.array_split(x, 3))
    return r

def break_stats(x, h, cut):
    pre = x[: h + cut]; post = x[h + cut:]
    if len(post) < 100: return None
    r = dict(post_len=len(post), log2_sd_ratio=np.log2(post.std() / pre.std()), d_mean=(post.mean() - pre.mean()) / pre.std())
    a1 = lambda v: np.corrcoef(v[:-1], v[1:])[0, 1]; r["d_acf1"] = a1(post) - a1(pre)
    c, const, res, s2 = ar_ols(pre, 2); X = np.column_stack([post[1:-1], post[:-2], np.ones(len(post) - 2)]); e = post[2:] - X @ np.append(c, const)
    r["log2_res_var"] = np.log2(e.var() / s2); r["d_kurt"] = stats.kurtosis(post) - stats.kurtosis(pre)
    zpre = (pre - pre.mean()) / pre.std(); zpost = (post - pre.mean()) / pre.std(); r["ks"] = stats.ks_2samp(zpre, zpost).statistic
    return r

def collect(vals, offs, meta, rng):
    H, B = [], []
    for i in range(len(meta)):
        a, b = offs[i], offs[i + 1]; x = np.asarray(vals[a:b], float); h = int(meta.hist_len[i]); o = int(meta.onl_len[i]); tau = int(meta.tau_index[i])
        H.append(hist_stats(x[:h]))
        cut = tau if tau >= 0 else int(rng.integers(0, o)); r = break_stats(x, h, cut)
        if r is not None: r["is_break"] = int(tau >= 0); B.append(r)
    return pd.DataFrame(H), pd.DataFrame(B)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rv = np.load(os.path.join(OUT, "values.npy"), mmap_mode="r"); ro = np.load(os.path.join(OUT, "offsets.npy")); rm = pd.read_csv(os.path.join(OUT, "meta.csv"))
    sel = np.sort(rng.choice(len(rm), 500, replace=False)); rm2 = rm.iloc[sel].reset_index(drop=True)
    ro2 = [0]; rvals = []
    for k in sel: rvals.append(np.asarray(rv[ro[k]:ro[k + 1]])); ro2.append(ro2[-1] + ro[k + 1] - ro[k])
    Hr, Br = collect(np.concatenate(rvals), np.array(ro2), rm2, rng)
    sv = np.load(os.path.join(OUT, "synth_values.npy")); so = np.load(os.path.join(OUT, "synth_offsets.npy")); sm = pd.read_csv(os.path.join(OUT, "synth_meta.csv"))
    Hs, Bs = collect(sv, so, sm, rng)
    out = []
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); out.append(s)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    P("== history statistics: real (n=%d) vs synthetic (n=%d); quantiles 5/25/50/75/95 and KS p" % (len(Hr), len(Hs)))
    P(f"{'stat':16s} {'':5s} " + " ".join(f"{'q' + str(int(q * 100)):>8s}" for q in qs) + "     KS_p")
    for c in Hr.columns:
        a = Hr[c].to_numpy(float); b = Hs[c].to_numpy(float); a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        P(f"{c:16s} real  " + " ".join(f"{v:8.3f}" for v in np.quantile(a, qs)))
        P(f"{'':16s} synth " + " ".join(f"{v:8.3f}" for v in np.quantile(b, qs)) + f"  {stats.ks_2samp(a, b).pvalue:9.2e}")
    P("\n== break statistics (post_len>=100): breaks vs non-break cuts; real vs synthetic")
    for lab, m in (("BREAKS", 1), ("NO-BREAK", 0)):
        P(f"-- {lab}: real n={int((Br.is_break == m).sum())}, synth n={int((Bs.is_break == m).sum())}")
        for c in ["log2_sd_ratio", "log2_res_var", "d_acf1", "d_mean", "d_kurt", "ks"]:
            a = Br[Br.is_break == m][c].to_numpy(float); b = Bs[Bs.is_break == m][c].to_numpy(float); a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
            P(f"{c:16s} real  " + " ".join(f"{v:8.3f}" for v in np.quantile(a, qs)))
            P(f"{'':16s} synth " + " ".join(f"{v:8.3f}" for v in np.quantile(b, qs)) + f"  {stats.ks_2samp(a, b).pvalue:9.2e}")
    # excess tail fractions for log2 sd ratio (breaks minus non-breaks)
    P("\n== excess tail fractions (breaks - no-break), log2 sd ratio: real vs synth")
    for c in (0.2, 0.3, 0.5, 1.0):
        fr = lambda D, s: np.mean(D[D.is_break == 1].log2_sd_ratio > s) - np.mean(D[D.is_break == 0].log2_sd_ratio > s)
        fd = lambda D, s: np.mean(D[D.is_break == 1].log2_sd_ratio < -s) - np.mean(D[D.is_break == 0].log2_sd_ratio < -s)
        P(f"  c={c}: up real {fr(Br, c):+.3f} synth {fr(Bs, c):+.3f} | down real {fd(Br, c):+.3f} synth {fd(Bs, c):+.3f}")
    P("  |d_acf1|>0.1 excess: real %+.3f synth %+.3f" % (np.mean(Br[Br.is_break == 1].d_acf1.abs() > 0.1) - np.mean(Br[Br.is_break == 0].d_acf1.abs() > 0.1), np.mean(Bs[Bs.is_break == 1].d_acf1.abs() > 0.1) - np.mean(Bs[Bs.is_break == 0].d_acf1.abs() > 0.1)))
    P("  family mix synth:", sm.lin.value_counts(normalize=True).round(2).to_dict(), " vol:", sm.vol.value_counts(normalize=True).round(2).to_dict())
    open(os.path.join(OUT, "t5_compare.txt"), "w").write("\n".join(out))
