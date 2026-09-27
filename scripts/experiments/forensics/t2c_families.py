"""Task 2c: assign generating families (linear dynamics x volatility x innovation distribution) and test
whether parameters look like a discrete menu. Writes families.csv and prints summary (t2c_summary.txt)."""
import os, sys, numpy as np, pandas as pd
from scipy import stats, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import thist, qtab, ar_ols
from t2b_extra import df_from_quantiles
OUT = os.path.dirname(os.path.abspath(__file__))
t1 = pd.read_csv(os.path.join(OUT, "t1_features.csv")); t2 = pd.read_csv(os.path.join(OUT, "t2_features.csv")); t2b = pd.read_csv(os.path.join(OUT, "t2b_features.csv"))
df = t1.merge(t2, on="id", suffixes=("", "_t2")).merge(t2b, on="id", suffixes=("", "_t2b"))
vals = np.load(os.path.join(OUT, "values.npy"), mmap_mode="r"); off = np.load(os.path.join(OUT, "offsets.npy")); meta = pd.read_csv(os.path.join(OUT, "meta.csv"))

# ---- GARCH-standardised residual distribution (recompute residuals of BIC-AR and sigma_t from saved alpha, beta)
kg, dfg, skg, asg = [], [], [], []
for k in range(len(df)):
    a = off[k]; h = int(meta.hist_len[k]); x = np.asarray(vals[a:a + h], float); n = len(x)
    p = int(df.ar_p_bic[k])
    e = (x - x.mean()) if p == 0 else ar_ols(x, p)[2]
    e = e - e.mean(); al, be = df.g_alpha[k], df.g_beta[k]
    if df.rk_lb10[k] > 30 and al > 0:
        e2 = e ** 2; v = e2.mean(); om = (1 - al - be) * v
        s2 = om / (1 - be) + signal.lfilter([al], [1.0, -be], np.concatenate([[v], e2[:-1]]))
        z = e / np.sqrt(np.maximum(s2, 1e-10))
    else:
        z = e
    z = (z - z.mean()) / z.std()
    kg.append(stats.kurtosis(z)); dfg.append(df_from_quantiles(z)); skg.append(stats.skew(z))
    q = np.quantile(z, [0.01, 0.5, 0.99]); asg.append((q[2] - q[1]) / (q[1] - q[0]))
df["z_kurt"] = kg; df["z_df"] = dfg; df["z_skew"] = skg; df["z_tailasym"] = asg

# ---- linear family by BIC among WN, AR1, AR2, AR3+, MA1, ARMA11
n = df.n.to_numpy()
bic_ar1 = n * np.log(df.ar1_s2) + 2 * np.log(n); bic_ar2 = n * np.log(df.ar2_s2) + 3 * np.log(n)
cands = pd.DataFrame(dict(WN=df.bic_wn, AR1=bic_ar1, AR2=bic_ar2, ARp=np.where(df.ar_p_bic >= 3, df.bic_ar_best, np.inf), MA1=df.bic_ma1, ARMA11=df.bic_arma11))
df["lin"] = cands.idxmin(axis=1)
df["lin_margin"] = np.sort(cands.to_numpy(), axis=1)[:, 1] - cands.min(axis=1)   # BIC gap to runner-up
# ---- volatility family
df["volclust"] = df.rk_lb10 > 30
df["vol"] = np.where(~df.volclust, "const", np.where((df.rv_acf1_res > 0.45) & (df.rv_bc_res > 0.6), "regime-like", "garch-like"))
# ---- innovation family (after GARCH standardisation where applicable)
def innov(r):
    if r.z_kurt < -0.35: return "light"
    if r.z_df >= 30 and r.z_kurt < 0.4: return "gauss"
    if r.z_df < 3: return "t<3"
    if r.z_df < 5: return "t3-5"
    if r.z_df < 10: return "t5-10"
    if r.z_df < 30: return "t10-30"
    return "gauss"
df["innov"] = df.apply(innov, axis=1)
df["asym"] = (df.z_tailasym < 0.8) | (df.z_tailasym > 1.25)
df.to_csv(os.path.join(OUT, "families.csv"), index=False)

print("linear family counts:\n", df.lin.value_counts().to_string())
print("\nlinear family, margin>10 BIC only:\n", df[df.lin_margin > 10].lin.value_counts().to_string(), "\n(unclear, margin<=10):", (df.lin_margin <= 10).sum())
print("\nvolatility family counts:\n", df.vol.value_counts().to_string())
print("\ninnovation family counts:\n", df.innov.value_counts().to_string())
print("\nasymmetric innovations (tail asym outside [0.8,1.25]):", int(df.asym.sum()))
print("\nlin x vol:\n", pd.crosstab(df.lin, df.vol).to_string())
print("\nlin x innov:\n", pd.crosstab(df.lin, df.innov).to_string())
print("\nvol x innov:\n", pd.crosstab(df.vol, df.innov).to_string())
print("\nmarginal kurt<-0.2 series by lin family:\n", df[df["kurt"] < -0.2].lin.value_counts().to_string())
print("marginal kurt<-0.2 by innov:\n", df[df["kurt"] < -0.2].innov.value_counts().to_string())
print(qtab(df[df["kurt"] < -0.2], ["acf1", "ar1_c1", "res_kurt", "z_kurt", "rm_split_gap", "rm_acf1", "cusum_mean"]))

# ---- menu checks
a1 = df[(df.lin == "AR1") & (df.lin_margin > 10)]
print(thist(a1.ar1_c1, rng=(-1, 1), bins=80, label="phi for AR1 family (fine bins 0.025)"))
m1 = df[(df.lin == "MA1") & (df.lin_margin > 10)]
print(thist(m1.ma1_theta, rng=(-1, 1), bins=40, label="theta for MA1 family"))
am = df[(df.lin == "ARMA11") & (df.lin_margin > 10)]
print(thist(am.arma_phi, rng=(-1, 1), bins=40, label="phi for ARMA11 family")); print(thist(am.arma_theta, rng=(-1, 1), bins=40, label="theta for ARMA11 family"))
print("corr(phi,theta) in ARMA11:", np.corrcoef(am.arma_phi, am.arma_theta)[0, 1])
a2 = df[(df.lin == "AR2") & (df.lin_margin > 10)]
disc = a2.ar2_c1 ** 2 + 4 * a2.ar2_c2
print("AR2 family: complex roots (disc<0):", int((disc < 0).sum()), " real roots:", int((disc >= 0).sum()))
print(thist(a2.ar2_c1, rng=(-2, 2), bins=40, label="AR2 c1")); print(thist(a2.ar2_c2, rng=(-1, 1), bins=40, label="AR2 c2"))
ap = df[(df.lin == "ARp") & (df.lin_margin > 10)]
print("ARp family p distribution:", ap.ar_p_bic.value_counts().sort_index().to_dict())
print(thist(ap.acf1, rng=(-1, 1), bins=40, label="acf1 for ARp family"))
vc = df[df.volclust]
print(thist(vc.g_alpha, rng=(0, 0.65), bins=26, label="GARCH alpha (vol-clustering series)"))
print(thist(vc.g_beta, rng=(0, 1), bins=25, label="GARCH beta (vol-clustering series)"))
print(thist(vc.g_alpha + vc.g_beta, rng=(0, 1), bins=25, label="GARCH alpha+beta (vol-clustering series)"))
print(thist(np.log10(vc.g_llgain + 1), rng=(0, 3.5), bins=35, label="log10 GARCH loglik gain +1 (vol-clustering)"))
hv = df[df.innov.isin(["t<3", "t3-5", "t5-10", "t10-30"])]
print(thist(hv.z_df, rng=(1, 30), bins=58, label="quantile-based t df (heavy-tailed innovations, after GARCH std)"))
print(thist(np.log10(hv.z_df), rng=(0, 1.5), bins=30, label="log10 t df (heavy-tailed innovations)"))
print(thist(df.z_kurt, rng=(-1.5, 3), bins=45, label="kurtosis of GARCH-standardised residuals (all)"))
# half-history stability for const-vol gaussian series vs calibration
g = df[(df.vol == "const") & (df.innov == "gauss")]
print("\nconst-vol gaussian series (n=%d): half_sd_ratio q50/95/99/99.9:" % len(g), np.quantile(g.half_sd_ratio, [0.5, 0.95, 0.99, 0.999]).round(3),
      " (iid calib 0.999/1.045/1.066/1.081)")
print("  third_sd_maxmin q50/95/99/99.9:", np.quantile(g.third_sd_maxmin, [0.5, 0.95, 0.99, 0.999]).round(3), " (calib 1.036/1.083/1.108/1.126)")
print("  cusum_var q50/95/99/99.9:", np.quantile(g.cusum_var, [0.5, 0.95, 0.99, 0.999]).round(3), " (calib 0.79/1.30/1.67/1.95)")
print("  cusum_mean q50/95/99/99.9:", np.quantile(g.cusum_mean, [0.5, 0.95, 0.99, 0.999]).round(3))
print("  |half_mean_diff| q50/95/99:", np.quantile(np.abs(g.half_mean_diff), [0.5, 0.95, 0.99]).round(3))
print("  frac with third_sd_maxmin>1.13:", (g.third_sd_maxmin > 1.13).mean().round(4), " frac cusum_var>1.95:", (g.cusum_var > 1.95).mean().round(4))
print(thist(g.third_sd_maxmin, rng=(1, 1.5), bins=25, label="third_sd_maxmin (const-vol gaussian series)"))
