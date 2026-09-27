"""Task 2e: (1) ARMA order distribution from t2d; (2) fine parameter histograms per family (menu check);
(3) calibration of df_q under Gaussian iid / Gaussian AR(1) via BIC-AR residuals; (4) history stability for WN+const+gauss
series vs iid calibration, with rolling-sd profiles of the most unstable examples; (5) fine acf1 histogram (bump at 0.5?)."""
import os, sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import thist, qtab, ar_ols, acf
from t2b_extra import df_from_quantiles
OUT = os.path.dirname(os.path.abspath(__file__))
fam = pd.read_csv(os.path.join(OUT, "families.csv")); od = pd.read_csv(os.path.join(OUT, "t2d_orders.csv"))
df = fam.merge(od, on="id")
vals = np.load(os.path.join(OUT, "values.npy"), mmap_mode="r"); off = np.load(os.path.join(OUT, "offsets.npy")); meta = pd.read_csv(os.path.join(OUT, "meta.csv"))

print("== (1) Hannan-Rissanen ARMA(p,q) order selection (BIC), counts p x q:")
print(pd.crosstab(df.hr_p, df.hr_q).to_string())
print("p+q distribution:", (df.hr_p + df.hr_q).value_counts().sort_index().to_dict())
print("HR order vs BIC-AR family:\n", pd.crosstab(df.lin, df.hr_p.astype(str) + "," + df.hr_q.astype(str)).to_string())
print("HR margin (BIC gap to 2nd best) quantiles:", np.quantile(df.hr_margin, [0.1, 0.25, 0.5, 0.75, 0.9]).round(1))
# AR(p>=3) by BIC-AR: what HR says
ap = df[df.lin == "ARp"]; print("ARp(BIC-AR) series: HR (p,q) counts:", (ap.hr_p.astype(str) + "," + ap.hr_q.astype(str)).value_counts().head(10).to_dict())
print("ARp series: BIC gain of AR(pbic) over HR-best ARMA(p<=3,q<=3) (positive => AR(p) better):", np.quantile(df.loc[df.lin == "ARp", "hr_bic"] - df.loc[df.lin == "ARp", "bic_ar_best"], [0.1, 0.5, 0.9]).round(1))

print("\n== (2) fine parameter histograms")
a1 = df[df.lin == "AR1"]; print(thist(a1.ar1_c1, rng=(-1, 1), bins=80, label=f"phi, AR1 family (BIC-AR), n={len(a1)}"))
h10 = df[(df.hr_p == 1) & (df.hr_q == 0)]; print(thist(h10.hr_ar1, rng=(-1, 1), bins=80, label=f"phi, HR ARMA(1,0) n={len(h10)}"))
h01 = df[(df.hr_p == 0) & (df.hr_q == 1)]; print(thist(h01.hr_ma1, rng=(-1, 1), bins=40, label=f"theta, HR ARMA(0,1) n={len(h01)}"))
h11 = df[(df.hr_p == 1) & (df.hr_q == 1)]; print(thist(h11.hr_ar1, rng=(-1, 1), bins=40, label=f"phi, HR ARMA(1,1) n={len(h11)}")); print(thist(h11.hr_ma1, rng=(-1, 1), bins=40, label="theta, HR ARMA(1,1)"))
print("ARMA(1,1): corr(phi,theta)=%.3f ; |phi+theta| quantiles:" % np.corrcoef(h11.hr_ar1, h11.hr_ma1)[0, 1], np.quantile(np.abs(h11.hr_ar1 + h11.hr_ma1), [0.1, 0.5, 0.9]).round(3))
h20 = df[(df.hr_p == 2) & (df.hr_q == 0)]; disc = h20.hr_ar1 ** 2 + 4 * h20.hr_ar2
print(f"AR(2) HR n={len(h20)}: complex roots {int((disc < 0).sum())}, real {int((disc >= 0).sum())}")
mod = np.sqrt(-h20.hr_ar2[disc < 0]); print(thist(mod, rng=(0, 1), bins=20, label="AR2 complex: root modulus"))
print(thist(h20.hr_ar1, rng=(-2, 2), bins=40, label="AR2 HR c1")); print(thist(h20.hr_ar2, rng=(-1, 1), bins=40, label="AR2 HR c2"))
# uniformity test of phi for AR1 family within (-0.95,0.95)
ph = h10.hr_ar1[(h10.hr_ar1.abs() < 0.95)]; print("phi AR(1): KS vs uniform(-0.95,0.95): D=%.3f p=%.3g ; mean %.3f ; frac>0 %.3f" % (*stats.kstest((ph + 0.95) / 1.9, "uniform")[:2], ph.mean(), (ph > 0).mean()))

print("\n== (3) df_q calibration (lower quantiles) under Gaussian")
rng = np.random.default_rng(1); out = {"iid": [], "ar09": [], "ar-05": []}
for i in range(400):
    n = int(rng.integers(1000, 5001)); e = rng.standard_normal(n)
    out["iid"].append(df_from_quantiles((e - e.mean()) / e.std()))
    for key, phi in (("ar09", 0.9), ("ar-05", -0.5)):
        x = np.empty(n); x[0] = e[0] / np.sqrt(1 - phi ** 2)
        for t in range(1, n): x[t] = phi * x[t - 1] + e[t]
        bics = []; fits = {}
        for p in range(0, 11):
            if p == 0: res = x - x.mean(); s2 = res.var()
            else: _, _, res, s2 = ar_ols(x, p)
            bics.append(n * np.log(s2) + (p + 1) * np.log(n)); fits[p] = res
        r = fits[int(np.argmin(bics))]; out[key].append(df_from_quantiles((r - r.mean()) / r.std()))
for key in out:
    v = np.array(out[key]); print(f"  {key}: df_q quantiles 0.1/1/5/25/50%:", np.quantile(v, [0.001, 0.01, 0.05, 0.25, 0.5]).round(1), " frac<30: %.3f frac<10: %.3f" % ((v < 30).mean(), (v < 10).mean()))
print("  data: frac z_df<30: %.3f  <10: %.3f  <5: %.3f" % ((df.z_df < 30).mean(), (df.z_df < 10).mean(), (df.z_df < 5).mean()))

print("\n== (4) history stability: WN + const-vol + gauss series vs iid calibration")
g = df[(df.lin == "WN") & (df.vol == "const") & (df.innov == "gauss")]
print("n=%d" % len(g))
for c, cal in (("half_sd_ratio", "0.999/1.045/1.066/1.081"), ("third_sd_maxmin", "1.036/1.083/1.108/1.126"), ("cusum_var", "0.79/1.30/1.67/1.95"), ("cusum_mean", "iid BB: ~0.83/1.36/1.63/1.95")):
    print(f"  {c}: q50/95/99/99.9 =", np.quantile(g[c], [0.5, 0.95, 0.99, 0.999]).round(3), " calib", cal)
print("  frac third_sd_maxmin>1.126: %.3f ; frac cusum_var>1.95: %.3f ; frac cusum_mean>1.95: %.3f" % ((g.third_sd_maxmin > 1.126).mean(), (g.cusum_var > 1.95).mean(), (g.cusum_mean > 1.95).mean()))
# also GARCH-standardised? no: these are const-vol. Show rolling sd profiles (20 blocks) of the 8 most unstable
top = g.sort_values("third_sd_maxmin", ascending=False).head(8)
for k in top.id:
    a = off[k]; h = int(meta.hist_len[k]); x = np.asarray(vals[a:a + h], float); m = 20; w = h // m
    prof = x[: m * w].reshape(m, w).std(axis=1)
    print(f"  id {k} n={h} rk_lb10={df.rk_lb10[k]:.1f} third_sd_maxmin={df.third_sd_maxmin[k]:.2f} sd profile:", " ".join(f"{v:.2f}" for v in prof))
# mean profile of the most mean-unstable
top = g.sort_values("cusum_mean", ascending=False).head(6)
for k in top.id:
    a = off[k]; h = int(meta.hist_len[k]); x = np.asarray(vals[a:a + h], float); m = 20; w = h // m
    prof = x[: m * w].reshape(m, w).mean(axis=1)
    print(f"  id {k} n={h} cusum_mean={df.cusum_mean[k]:.2f} mean profile:", " ".join(f"{v:+.2f}" for v in prof))
# variance-instability of histories, all series: is instability confined to volclust series? distribution of third_sd_maxmin by vol family
print(qtab(df.assign(g=df.vol), ["third_sd_maxmin"]))
for v in ("const", "garch-like", "regime-like"):
    s = df[df.vol == v]; print(f"  vol={v}: third_sd_maxmin q50/90/99:", np.quantile(s.third_sd_maxmin, [0.5, 0.9, 0.99]).round(2), " cusum_var q50/90/99:", np.quantile(s.cusum_var, [0.5, 0.9, 0.99]).round(2))

print("\n== (5) acf1 fine histogram by family (bump near 0.5?)")
print(thist(df.acf1, rng=(0.3, 0.7), bins=40, label="acf1 in [0.3,0.7], all series"))
for f in ("WN", "AR1", "AR2", "ARp", "MA1", "ARMA11"):
    s = df[df.lin == f]; c, _ = np.histogram(s.acf1, bins=8, range=(0.3, 0.7)); print(f"  {f:7s} acf1 bins 0.3..0.7 step .05:", c)
print(thist(df.hr_ar1[(df.hr_p >= 1)], rng=(0.3, 0.7), bins=40, label="HR ar1 coefficient in [0.3,0.7] (p>=1)"))
