"""Task 3b: effect-size distributions by deconvolution against controls.
For each change statistic: excess variance (breaks - controls), tail excess, and a 2-parameter mixture fit
  break stat = control noise + effect,  effect = 0 w.p. (1-pi), else ~ N(0, s^2) [also Laplace(0,b), Uniform(-a,a)]
fitted by maximum likelihood with the control sample as the noise kernel. Per post_len bin and per linear family
(family-matched controls). -> t3b_summary.txt"""
import os, sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT, "t3_z.csv"))
df["log2_sd_ratio"] = np.log2(df.sd_ratio); df["log2_ivr_sd"] = 0.5 * np.log2(df.innov_var_ratio)
bins = [20, 50, 100, 200, 400, 1000]; df["lbin"] = pd.cut(df.post_len, bins, right=False)
rob_sd = lambda v: np.subtract(*np.percentile(v, [75, 25])) / 1.349

def mixfit(xb, xc, kind="normal"):
    """ML fit of pi, s: f_b(x) = (1-pi) f_c(x) + pi * mean_j K_s(x - c_j); f_c = KDE of controls (bw by Silverman/2)."""
    xb = np.asarray(xb, float); xc = np.asarray(xc, float)
    if len(xc) > 1500: xc = np.random.default_rng(0).choice(xc, 1500, replace=False)
    h = 0.5 * 1.06 * rob_sd(xc) * len(xc) ** (-0.2) + 1e-6
    D = xb[:, None] - xc[None, :]
    f_c = np.mean(stats.norm.pdf(D / h) / h, axis=1)
    best = (-np.inf, 0, 0)
    for s in np.concatenate([np.linspace(0.02, 0.3, 15), np.linspace(0.35, 1.5, 24)]):
        if kind == "normal": K = np.mean(stats.norm.pdf(D / s) / s, axis=1)
        elif kind == "laplace": K = np.mean(stats.laplace.pdf(D / s) / s, axis=1)
        else: K = np.mean(stats.uniform.pdf(D, loc=-s, scale=2 * s), axis=1)
        for pi in np.linspace(0, 1, 41):
            ll = np.sum(np.log((1 - pi) * f_c + pi * K + 1e-300))
            if ll > best[0]: best = (ll, pi, s)
    ll0 = np.sum(np.log(f_c + 1e-300))
    return best[1], best[2], best[0] - ll0

out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

STATS = ["log2_sd_ratio", "log2_ivr_sd", "d_ar1", "d_mean", "d_acf2", "d_sqacf1", "d_res_kurt", "ks", "wass"]
P("== excess variance and tail excess by post_len bin (all families pooled)")
P(f"{'stat':14s} {'bin':12s} {'n_b':>5s} {'n_c':>5s} {'rsd_b':>7s} {'rsd_c':>7s} {'eff_sd':>7s} {'exc|z|>2':>9s} {'exc|z|>3':>9s} {'exc|z|>5':>9s}")
for s in STATS:
    for b, g in df.groupby("lbin", observed=True):
        xb = g[g.is_break == 1][s].dropna().to_numpy(); xc = g[g.is_break == 0][s].dropna().to_numpy()
        zb = g[g.is_break == 1]["z_" + s.replace("log2_sd_ratio", "log_sd_ratio").replace("log2_ivr_sd", "log_ivr")].dropna().abs().to_numpy()
        zc = g[g.is_break == 0]["z_" + s.replace("log2_sd_ratio", "log_sd_ratio").replace("log2_ivr_sd", "log_ivr")].dropna().abs().to_numpy()
        vb, vc = rob_sd(xb) ** 2, rob_sd(xc) ** 2
        P(f"{s:14s} {str(b):12s} {len(xb):5d} {len(xc):5d} {np.sqrt(vb):7.3f} {np.sqrt(vc):7.3f} {np.sqrt(max(vb - vc, 0)):7.3f} {np.mean(zb > 2) - np.mean(zc > 2):9.3f} {np.mean(zb > 3) - np.mean(zc > 3):9.3f} {np.mean(zb > 5) - np.mean(zc > 5):9.3f}")

P("\n== mixture deconvolution (post_len>=100): pi = fraction of breaks with a non-zero effect, s = effect scale")
big = df[df.post_len >= 100]
for s in ["log2_sd_ratio", "d_ar1", "d_mean", "d_acf2"]:
    xb = big[big.is_break == 1][s].dropna().to_numpy(); xc = big[big.is_break == 0][s].dropna().to_numpy()
    for kind in ("normal", "laplace", "uniform"):
        pi, sc, dll = mixfit(xb, xc, kind)
        P(f"  {s:14s} {kind:8s} pi={pi:.3f} scale={sc:.3f} dLL={dll:.1f}")
P("\n== mixture deconvolution for log2 sd ratio by post_len bin (normal effect)")
for b, g in df.groupby("lbin", observed=True):
    xb = g[g.is_break == 1]["log2_sd_ratio"].dropna().to_numpy(); xc = g[g.is_break == 0]["log2_sd_ratio"].dropna().to_numpy()
    pi, sc, dll = mixfit(xb, xc, "normal"); P(f"  {str(b):12s} n_b={len(xb):5d} pi={pi:.3f} s={sc:.3f} dLL={dll:.1f}")
P("\n== by linear family (post_len>=100, family-matched controls): effect sd of log2 sd ratio, d_ar1, d_mean; mixture (normal) for log2 sd ratio")
for fam, g in big.groupby("lin"):
    xb = g[g.is_break == 1]; xc = g[g.is_break == 0]
    line = f"  {fam:7s} n_b={len(xb):4d} n_c={len(xc):4d}"
    for s in ["log2_sd_ratio", "d_ar1", "d_mean", "d_acf2"]:
        vb, vc = rob_sd(xb[s].dropna()) ** 2, rob_sd(xc[s].dropna()) ** 2
        line += f" | {s}: rsd_b={np.sqrt(vb):.3f} rsd_c={np.sqrt(vc):.3f} eff={np.sqrt(max(vb - vc, 0)):.3f}"
    pi, sc, dll = mixfit(xb["log2_sd_ratio"].dropna(), xc["log2_sd_ratio"].dropna(), "normal")
    P(line + f" || mix log2sd: pi={pi:.2f} s={sc:.2f} dLL={dll:.1f}")
P("\n== by vol family (post_len>=100)")
for fam, g in big.groupby("vol"):
    xb = g[g.is_break == 1]; xc = g[g.is_break == 0]
    line = f"  {fam:11s} n_b={len(xb):4d} n_c={len(xc):4d}"
    for s in ["log2_sd_ratio", "d_ar1", "d_mean"]:
        vb, vc = rob_sd(xb[s].dropna()) ** 2, rob_sd(xc[s].dropna()) ** 2
        line += f" | {s}: rsd_b={np.sqrt(vb):.3f} rsd_c={np.sqrt(vc):.3f} eff={np.sqrt(max(vb - vc, 0)):.3f}"
    P(line)
P("\n== by innovation family (post_len>=100)")
for fam, g in big.groupby("innov"):
    xb = g[g.is_break == 1]; xc = g[g.is_break == 0]
    if len(xb) < 30: continue
    line = f"  {fam:7s} n_b={len(xb):4d} n_c={len(xc):4d}"
    for s in ["log2_sd_ratio", "d_ar1", "d_mean"]:
        vb, vc = rob_sd(xb[s].dropna()) ** 2, rob_sd(xc[s].dropna()) ** 2
        line += f" | {s}: rsd_b={np.sqrt(vb):.3f} rsd_c={np.sqrt(vc):.3f} eff={np.sqrt(max(vb - vc, 0)):.3f}"
    P(line)
# sign symmetry of variance effects: compare upper vs lower tails
P("\n== sign symmetry (post_len>=100): P(log2 sd ratio > c) - ctrl, P(< -c) - ctrl")
xb = big[big.is_break == 1]["log2_sd_ratio"]; xc = big[big.is_break == 0]["log2_sd_ratio"]
for c in (0.2, 0.3, 0.5, 1.0):
    P(f"  c={c}: up {np.mean(xb > c) - np.mean(xc > c):.3f}  down {np.mean(xb < -c) - np.mean(xc < -c):.3f}   (raw brk up {np.mean(xb > c):.3f} down {np.mean(xb < -c):.3f}; ctl up {np.mean(xc > c):.3f} down {np.mean(xc < -c):.3f})")
xb = big[big.is_break == 1]["d_ar1"]; xc = big[big.is_break == 0]["d_ar1"]
for c in (0.1, 0.2, 0.3):
    P(f"  d_ar1 c={c}: up {np.mean(xb > c) - np.mean(xc > c):.3f}  down {np.mean(xb < -c) - np.mean(xc < -c):.3f}")
# does the AR1 change go with a variance change (independent draws?)
b1 = big[big.is_break == 1]
P("\ncorr among breaks (post_len>=100): corr(|log2 sd ratio|, |d_ar1|)=%.3f ; controls %.3f" % (np.corrcoef(b1.log2_sd_ratio.abs(), b1.d_ar1.abs())[0, 1], np.corrcoef(big[big.is_break == 0].log2_sd_ratio.abs(), big[big.is_break == 0].d_ar1.abs())[0, 1]))
open(os.path.join(OUT, "t3b_summary.txt"), "w").write("\n".join(out))
