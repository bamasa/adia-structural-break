"""Task 3 summary: robust z-scores of break statistics vs matched controls (post-length bins), detectable fractions,
break-type menu, magnitude distributions (menu vs continuous), minimum magnitudes, families. -> t3_summary.txt, t3_z.csv"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import thist, qtab
OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT, "t3_stats.csv")); df = df[df.post_len >= 20].copy()
df["log_sd_ratio"] = np.log(df.sd_ratio); df["log_ivr"] = np.log(df.innov_var_ratio); df["log_hill_ratio"] = np.log(df.hill_ratio)
df["d_q99_01"] = df.post_q99_01 - df.pre_q99_01; df["d_absacf1"] = df.post_absacf1 - df.pre_absacf1; df["d_acf2"] = df.post_acf2 - df.pre_acf2
df["d_res_skew"] = df.post_res_skew - df.pre_res_skew; df["log_maxabs_ratio"] = np.log(df.post_maxabs / df.pre_maxabs)
STATS = ["d_mean", "log_sd_ratio", "log_ivr", "innov_mean_shift", "d_ar1", "d_ar2", "d_ar3", "d_acf2", "d_kurt", "d_res_kurt", "d_skew", "d_res_skew", "d_sqacf1", "d_absacf1",
         "log_hill_ratio", "d_spec_slope", "d_q99_01", "log_maxabs_ratio", "ks", "wass"]
bins = [20, 50, 100, 200, 400, 1000]; df["lbin"] = pd.cut(df.post_len, bins, right=False)
ctrl = df[df.is_break == 0]; brk = df[df.is_break == 1]
print("breaks with post_len>=20:", len(brk), " controls:", len(ctrl))
print("post_len bins (breaks):", brk.lbin.value_counts().sort_index().to_dict())
# robust z per bin
Z = pd.DataFrame(index=df.index)
for s in STATS:
    z = np.full(len(df), np.nan)
    for b, g in ctrl.groupby("lbin", observed=True):
        v = g[s].to_numpy(float); v = v[np.isfinite(v)]; med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-12
        m = (df.lbin == b).to_numpy(); z[m] = (df.loc[m, s].to_numpy(float) - med) / mad
    Z["z_" + s] = z
df = pd.concat([df, Z], axis=1); df.to_csv(os.path.join(OUT, "t3_z.csv"), index=False)
ctrl = df[df.is_break == 0]; brk = df[df.is_break == 1]
rows = []
for s in STATS:
    zb = brk["z_" + s].to_numpy(float); zc = ctrl["z_" + s].to_numpy(float); zb = zb[np.isfinite(zb)]; zc = zc[np.isfinite(zc)]
    iqr = lambda v: np.subtract(*np.percentile(v, [75, 25]))
    rows.append([s, len(zb), iqr(zb) / (iqr(zc) + 1e-12), np.mean(np.abs(zb) > 2), np.mean(np.abs(zc) > 2), np.mean(np.abs(zb) > 3), np.mean(np.abs(zc) > 3), np.mean(np.abs(zb) > 5), np.mean(np.abs(zc) > 5)])
tab = pd.DataFrame(rows, columns=["stat", "n_brk", "iqr_ratio", "brk|z|>2", "ctl|z|>2", "brk|z|>3", "ctl|z|>3", "brk|z|>5", "ctl|z|>5"])
print("\n== detectability of each statistic (robust z vs length-matched controls)\n", tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---- break-type menu with z>3 (well beyond control noise): groups
zt = 3.0
mean_b = brk.z_d_mean.abs() > zt
var_b = (brk.z_log_sd_ratio.abs() > zt) | (brk.z_log_ivr.abs() > zt)
dep_b = (brk.z_d_ar1.abs() > zt) | (brk.z_d_ar2.abs() > zt) | (brk.z_d_ar3.abs() > zt) | (brk.z_d_acf2.abs() > zt)
shape_b = (brk.z_d_res_kurt.abs() > zt) | (brk.z_d_res_skew.abs() > zt) | (brk.z_log_hill_ratio.abs() > zt) | (brk.z_d_q99_01.abs() > zt)
volc_b = (brk.z_d_sqacf1.abs() > zt) | (brk.z_d_absacf1.abs() > zt)
dist_b = (brk.z_ks.abs() > zt) | (brk.z_wass.abs() > zt)
kind = np.where(mean_b, "M", "") ; kind = np.char.add(kind.astype(str), np.where(var_b, "V", "")); kind = np.char.add(kind, np.where(dep_b, "D", "")); kind = np.char.add(kind, np.where(shape_b, "S", "")); kind = np.char.add(kind, np.where(volc_b, "C", ""))
kind = np.where(kind == "", "none", kind); brk = brk.assign(kind=kind)
print("\n== break kinds (|z|>3 on any stat of the group; M=mean V=variance D=dependence S=shape C=vol-clustering):\n", pd.Series(kind).value_counts().to_string())
print("\nnone (no |z|>3) fraction by post_len bin:\n", brk.groupby("lbin", observed=True).apply(lambda g: pd.Series(dict(n=len(g), none=(g.kind == "none").mean(), none_z2=((g[[c for c in brk.columns if c.startswith('z_')]].abs() > 2).sum(axis=1) == 0).mean()))).to_string())
# same for controls: fraction of controls with any |z|>3 (false positive rate of the union rule)
ctl_any = (ctrl[["z_" + s for s in STATS if s not in ("ks", "wass")]].abs() > zt).any(axis=1)
print("controls with any |z|>3 (false-positive rate of the union rule): %.3f" % ctl_any.mean())
brk.to_csv(os.path.join(OUT, "t3_break_kinds_full.csv"), index=False)
print("\nkind by linear family:\n", pd.crosstab(brk.lin, brk.kind.str.replace(r"[MSC]", "", regex=True).replace("", "other")).to_string())
print("\nkind by vol family:\n", pd.crosstab(brk.vol, brk.kind).to_string())

# ---- magnitude distributions: menu or continuous?
vb = brk[brk.z_log_sd_ratio.abs() > 4]
print(thist(np.log2(vb.sd_ratio), rng=(-3, 3), bins=60, label=f"log2 sd ratio, variance breaks |z|>4 (n={len(vb)})"))
print(thist(np.log2(vb.sd_ratio), rng=(-1.5, 1.5), bins=60, label="log2 sd ratio zoom (bin 0.05)"))
print("sd_ratio quantiles (variance breaks):", np.quantile(vb.sd_ratio, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(3), " frac<1:", (vb.sd_ratio < 1).mean().round(3))
print("min |log2 sd ratio| among variance breaks:", np.abs(np.log2(vb.sd_ratio)).min().round(3), " q05:", np.quantile(np.abs(np.log2(vb.sd_ratio)), 0.05).round(3))
allb = brk[brk.post_len >= 200]
print(thist(np.log2(allb.sd_ratio), rng=(-1.5, 1.5), bins=60, label=f"log2 sd ratio, ALL breaks with post_len>=200 (n={len(allb)})"))
allc = ctrl[ctrl.post_len >= 200]
print(thist(np.log2(allc.sd_ratio), rng=(-1.5, 1.5), bins=60, label=f"log2 sd ratio, controls post_len>=200 (n={len(allc)})"))
ivb = brk[brk.z_log_ivr.abs() > 4]
print(thist(0.5 * np.log2(ivb.innov_var_ratio), rng=(-3, 3), bins=60, label=f"0.5*log2 innovation var ratio (sd scale), |z|>4 (n={len(ivb)})"))
db = brk[brk.z_d_ar1.abs() > 4]
print(thist(db.d_ar1, rng=(-1.5, 1.5), bins=60, label=f"d_ar1 (post-pre), dependence breaks |z|>4 (n={len(db)})"))
print("pre_ar1 vs post_ar1 (dependence breaks): corr=%.3f ; controls corr=%.3f" % (np.corrcoef(db.pre_ar1_c1, db.post_ar1_c1)[0, 1], np.corrcoef(ctrl.pre_ar1_c1, ctrl.post_ar1_c1)[0, 1]))
print(thist(db.post_ar1_c1, rng=(-1, 1), bins=40, label="post AR1 coefficient (dependence breaks)")); print(thist(db.pre_ar1_c1, rng=(-1, 1), bins=40, label="pre AR1 coefficient (dependence breaks)"))
print("sign of d_ar1 vs pre: frac post |phi| > pre |phi|: %.3f" % (db.post_ar1_c1.abs() > db.pre_ar1_c1.abs()).mean())
mb = brk[brk.z_d_mean.abs() > 4]
print(thist(mb.d_mean, rng=(-3, 3), bins=60, label=f"mean shift (pre-sd units), mean breaks |z|>4 (n={len(mb)})"))
print(thist(allb.d_mean, rng=(-1, 1), bins=50, label="mean shift, ALL breaks post_len>=200")); print(thist(allc.d_mean, rng=(-1, 1), bins=50, label="mean shift, controls post_len>=200"))
sb = brk[brk.z_d_res_kurt.abs() > 4]
print(thist(sb.d_res_kurt, rng=(-10, 30), bins=40, label=f"d residual kurtosis, shape breaks |z|>4 (n={len(sb)})"))
print("shape breaks: post res kurt quantiles", np.quantile(sb.post_res_kurt, [0.05, 0.25, 0.5, 0.75, 0.95]).round(2), " pre:", np.quantile(sb.pre_res_kurt, [0.05, 0.25, 0.5, 0.75, 0.95]).round(2))
cb = brk[brk.z_d_sqacf1.abs() > 4]
print("vol-clustering breaks (n=%d): pre sqacf1 q:" % len(cb), np.quantile(cb.pre_sqacf1, [0.1, 0.5, 0.9]).round(3), " post sqacf1 q:", np.quantile(cb.post_sqacf1, [0.1, 0.5, 0.9]).round(3))
# ---- joint: variance x dependence correlation among breaks with both
both = brk[(brk.z_log_sd_ratio.abs() > 3) & (brk.z_d_ar1.abs() > 3)]
print("\nbreaks with both variance and AR1 change:", len(both), " corr(log sd ratio, d_ar1)=%.3f" % np.corrcoef(both.log_sd_ratio, both.d_ar1)[0, 1])
# is the innovation-variance change explaining the sd change? among V-breaks: log sd_ratio vs 0.5 log ivr
print("V-breaks: corr(log sd_ratio, 0.5*log ivr)=%.3f ; median |log sd_ratio - 0.5 log ivr| = %.3f" % (np.corrcoef(vb.log_sd_ratio, 0.5 * vb.log_ivr)[0, 1], np.median(np.abs(vb.log_sd_ratio - 0.5 * vb.log_ivr))))
# ---- sanity: online pre-break segment vs history (sd ratio) for breaks and controls
op = df[df.cut >= 30]
print("\nonline-pre-break sd (history sd=1): breaks q05/50/95:", np.quantile(op[op.is_break == 1].onlpre_sd, [0.05, 0.5, 0.95]).round(3), " controls:", np.quantile(op[op.is_break == 0].onlpre_sd, [0.05, 0.5, 0.95]).round(3))
print("online-pre-break mean: breaks q05/50/95:", np.quantile(op[op.is_break == 1].onlpre_mean, [0.05, 0.5, 0.95]).round(3), " controls:", np.quantile(op[op.is_break == 0].onlpre_mean, [0.05, 0.5, 0.95]).round(3))
# ---- per-family magnitude summary
print("\n== per linear family: median |z| and detect fraction (|z|>3) for key stats")
for s in ["d_mean", "log_sd_ratio", "log_ivr", "d_ar1", "d_res_kurt", "d_sqacf1", "ks"]:
    g = brk.groupby("lin")["z_" + s].agg(lambda v: np.mean(np.abs(v.dropna()) > 3)); print(f"  {s:14s}", " ".join(f"{k}:{v:.2f}" for k, v in g.items()))
print("\n== per innovation family")
for s in ["d_mean", "log_sd_ratio", "log_ivr", "d_ar1", "d_res_kurt", "d_sqacf1", "ks"]:
    g = brk.groupby("innov")["z_" + s].agg(lambda v: np.mean(np.abs(v.dropna()) > 3)); print(f"  {s:14s}", " ".join(f"{k}:{v:.2f}" for k, v in g.items()))
print("\n== per vol family")
for s in ["d_mean", "log_sd_ratio", "log_ivr", "d_ar1", "d_res_kurt", "d_sqacf1", "ks"]:
    g = brk.groupby("vol")["z_" + s].agg(lambda v: np.mean(np.abs(v.dropna()) > 3)); print(f"  {s:14s}", " ".join(f"{k}:{v:.2f}" for k, v in g.items()))
