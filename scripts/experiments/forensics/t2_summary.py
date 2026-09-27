import os, numpy as np, pandas as pd
from util import thist, qtab
OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT, "t2_features.csv"))
m1 = pd.read_csv(os.path.join(OUT, "t1_features.csv"))
df = df.merge(m1[["id", "kurt", "skew", "maxabs_z"]], on="id")
print(qtab(df, ["acf1", "acf2", "acf5", "acf10", "acf20", "acf50", "pacf2", "pacf3", "ar1_c1", "ar_p_bic", "ar_p_aic", "bic_gain_best",
                "res_kurt", "res_skew", "res_kurt_rollstd", "sqacf1", "sqacf5", "sqacf20", "sqacf50", "sq_lb10", "sq_lb50", "arch_r2", "absacf1", "absacf10",
                "adf_t", "vr5", "vr20", "vr100", "trend_slope_sd", "g_lev_logp", "g_res_logp", "peak_conc_lev", "peak_conc_res", "spec_slope",
                "rv_bc_x", "rv_bc_res", "rv_cv_res", "rv_q90q10_res", "rv_acf1_res", "rv_split_gap", "ma1_theta", "arma_phi", "arma_theta"]))
print("\nar_p_bic counts:\n", df.ar_p_bic.value_counts().sort_index().to_string())
print("\nar_p_aic counts:\n", df.ar_p_aic.value_counts().sort_index().to_string())
print(thist(df.acf1, rng=(-1, 1), bins=40, label="acf1 (history)"))
p1 = df[df.ar_p_bic == 1]
print(thist(p1.ar1_c1, rng=(-1, 1), bins=50, label="AR(1) coef for series with BIC p=1"))
print(thist(df.adf_t, rng=(-60, 2), bins=31, label="ADF t (const, 4 lags); reject unit root if < -2.86"))
print(thist(np.log10(df.vr20), rng=(-2, 0.5), bins=25, label="log10 VR(20) levels; WN=-1.3, RW=0"))
print(thist(df.g_lev_logp, rng=(-100, 2), bins=34, label="Fisher g log10 p (levels)"))
print(thist(df.g_res_logp, rng=(-100, 2), bins=34, label="Fisher g log10 p (AR residuals)"))
print(thist(df.sqacf1, rng=(-0.1, 0.6), bins=35, label="acf1 of squared AR residuals"))
print(thist(np.log10(df.sq_lb10 + 1), rng=(0, 4), bins=40, label="log10 LjungBox(10) squared residuals (+1); iid ~ 1"))
print(thist(df.res_kurt, rng=(-2, 10), bins=48, label="excess kurtosis of AR residuals"))
print(thist(np.log10(df.res_kurt + 3), rng=(0.2, 2), bins=36, label="log10(res_kurt+3)"))
print(thist(df.res_kurt_rollstd, rng=(-2, 6), bins=32, label="excess kurtosis of residuals standardised by rolling sd(21)"))
print(thist(df.rv_bc_res, rng=(0.1, 1), bins=36, label="bimodality coeff of log rolling var (residuals, w=50); >0.555 bimodal"))
print(thist(df.rv_split_gap, rng=(0, 8), bins=32, label="2-means split gap of log rolling var (x)"))
print(thist(df.rv_acf1_res, rng=(-0.5, 1), bins=30, label="acf1 of non-overlapping rolling var (residuals)"))
print(thist(df.ma1_theta, rng=(-1, 1), bins=40, label="MA(1) theta"))
print(thist(df.spec_slope, rng=(-3, 1.5), bins=45, label="spectral slope (levels)"))
# BIC model comparison
best = np.argmin(np.column_stack([df.bic_wn, df.bic_ar_best, df.bic_ma1, df.bic_arma11]), axis=1)
print("\nBIC best among [WN, AR(pBIC), MA1, ARMA11]:", np.bincount(best, minlength=4))
d = df.bic_ar_best - df.bic_arma11
print("ARMA11 beats AR(p) by >10 BIC:", (d > 10).sum(), " by >0:", (d > 0).sum(), " MA1 beats AR(p) by >10:", ((df.bic_ar_best - df.bic_ma1) > 10).sum())
# where does p_bic land when ARMA wins
w = df[d > 10]; print("p_bic when ARMA wins:", w.ar_p_bic.value_counts().sort_index().to_dict())
print(qtab(w, ["arma_phi", "arma_theta", "acf1", "ar_p_bic"]))
# seasonal ones: periods
s = df[df.g_res_logp < -8]
print("\nseasonal by residual g (logp<-8):", len(s), "; by level g (logp<-8):", (df.g_lev_logp < -8).sum(), " both:", ((df.g_lev_logp < -8) & (df.g_res_logp < -8)).sum())
print(thist(np.log10(s.peak_period_res), rng=(0.3, 3.5), bins=32, label="log10 peak period (residual-seasonal series)"))
print(thist(s.peak_conc_res, rng=(0, 1), bins=20, label="peak concentration (residual-seasonal)"))
print("peak periods (rounded) top 30:\n", s.peak_period_res.round(1).value_counts().head(30).to_string())
