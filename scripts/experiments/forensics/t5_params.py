"""Task 5a: per-series generator parameter table for synth.py (bootstrap source) -> synth_params.csv
Linear part: for lin in {WN, AR1, AR2, MA1, ARMA11} use the Hannan-Rissanen ARMA(p<=3,q<=3) coefficients of the
BIC-selected order (t2d); for lin == ARp use the BIC-AR(p) OLS coefficients refitted here (p<=10).
Volatility: GARCH(1,1) alpha, beta for vol != const (unit unconditional variance). Innovations: t df (z_df) or gauss/light."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import ar_ols
OUT = os.path.dirname(os.path.abspath(__file__))
fam = pd.read_csv(os.path.join(OUT, "families.csv")); od = pd.read_csv(os.path.join(OUT, "t2d_orders.csv"))
vals = np.load(os.path.join(OUT, "values.npy"), mmap_mode="r"); off = np.load(os.path.join(OUT, "offsets.npy")); meta = pd.read_csv(os.path.join(OUT, "meta.csv"))
rows = []
for k in range(len(fam)):
    r = dict(id=k, lin=fam.lin[k], vol=fam.vol[k], innov=fam.innov[k], df=fam.z_df[k], res_kurt=fam.z_kurt[k], alpha=fam.g_alpha[k] if fam.vol[k] != "const" else 0.0, beta=fam.g_beta[k] if fam.vol[k] != "const" else 0.0)
    ar = np.zeros(10); ma = np.zeros(3)
    if fam.lin[k] == "ARp":
        p = int(fam.ar_p_bic[k]); a = off[k]; h = int(meta.hist_len[k]); x = np.asarray(vals[a:a + h], float)
        coef, c, res, s2 = ar_ols(x, p); ar[:p] = coef; r["p"], r["q"] = p, 0
    else:
        p, q = int(od.hr_p[k]), int(od.hr_q[k]); r["p"], r["q"] = p, q
        ar[:3] = [od.hr_ar1[k], od.hr_ar2[k], od.hr_ar3[k]]; ma[:] = [od.hr_ma1[k], od.hr_ma2[k], od.hr_ma3[k]]
    for j in range(10): r[f"ar{j+1}"] = ar[j]
    for j in range(3): r[f"ma{j+1}"] = ma[j]
    # stationarity / invertibility check
    pa = np.r_[1, -ar[: max(1, int(r["p"]))]]; ma_poly = np.r_[1, ma[: max(1, int(r["q"]))]]
    r["ar_maxroot"] = np.max(np.abs(1 / np.roots(pa[::-1]))) if len(pa) > 1 and np.any(pa[1:] != 0) else 0.0   # max |1/z| of characteristic roots (must be <1)
    r["ma_maxroot"] = np.max(np.abs(1 / np.roots(ma_poly[::-1]))) if len(ma_poly) > 1 and np.any(ma_poly[1:] != 0) else 0.0
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "synth_params.csv"), index=False)
print("saved", len(df)); print("non-stationary AR (maxroot>=1):", int((df.ar_maxroot >= 1).sum()), " non-invertible MA:", int((df.ma_maxroot >= 1).sum()))
print(df.groupby("lin")[["p", "q"]].agg(["mean", "max"]).to_string())
print("alpha+beta>=0.999:", int((df.alpha + df.beta >= 0.999).sum()))
