"""Task 1: marginal structure of histories (period 1). Per-series features -> t1_features.csv
Then a summary pass prints quantiles / text histograms and rule-based marginal families."""
import os, sys, time
import numpy as np, pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
vals = np.load(os.path.join(OUT, "values.npy"), mmap_mode="r")
off = np.load(os.path.join(OUT, "offsets.npy"))
meta = pd.read_csv(os.path.join(OUT, "meta.csv"))

def decimals(x):
    """smallest k in 0..6 such that x*10^k is integer (float32 tolerance), else 7"""
    for k in range(7):
        y = x * (10.0 ** k)
        if np.all(np.abs(y - np.round(y)) < 1e-3 * max(1.0, np.abs(y).max()) * 1e-3 + 1e-4):
            return k
    return 7

rows = []
t0 = time.time()
for k in range(len(meta)):
    a = off[k]; h = int(meta.hist_len[k])
    x = np.asarray(vals[a:a+h], dtype=np.float64)
    n = len(x)
    mu, sd = x.mean(), x.std()
    z = (x - mu) / (sd + 1e-300)
    q = np.quantile(z, [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999])
    nu = len(np.unique(x))
    d = np.diff(np.sort(x)); dpos = d[d > 0]
    r = dict(
        id=k, n=n, mean=mu, sd=sd, min=x.min(), max=x.max(), median=np.median(x),
        skew=stats.skew(x), kurt=stats.kurtosis(x),  # excess kurtosis
        # robust shape (standardised quantiles)
        q001=q[0], q01=q[2], q05=q[4], q25=q[6], q50=q[7], q75=q[8], q95=q[10], q99=q[12], q999=q[14],
        tailratio=(q[12] - q[2]) / (q[8] - q[6] + 1e-12),        # (q99-q01)/IQR ; Gaussian = 3.45
        tailasym=(q[12] - q[7]) / (q[7] - q[2] + 1e-12),          # (q99-med)/(med-q01) ; sym = 1
        extasym=(x.max() - mu) / (mu - x.min() + 1e-12),
        maxabs_z=np.abs(z).max(),
        # discreteness
        nunique_frac=nu / n, nunique=nu, int_frac=np.mean(np.abs(x - np.round(x)) < 1e-6),
        decimals=decimals(x), min_gap=dpos.min() if len(dpos) else 0.0, med_gap=np.median(dpos) if len(dpos) else 0.0,
        zero_frac=np.mean(x == 0), mode_frac=np.max(np.unique(x, return_counts=True)[1]) / n,
        repeat_frac=np.mean(x[1:] == x[:-1]),
        # bounds
        nonneg=bool(x.min() >= 0), in01=bool(x.min() >= 0 and x.max() <= 1), in_m11=bool(x.min() >= -1 and x.max() <= 1),
        # normality
        jb=stats.jarque_bera(x)[0], ad=stats.anderson(z)[0],
        # marginal of first differences (for random-walk-like series)
        dskew=stats.skew(np.diff(x)), dkurt=stats.kurtosis(np.diff(x)),
    )
    rows.append(r)
    if k % 2000 == 0: print(k, round(time.time() - t0, 1), "s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "t1_features.csv"), index=False)
print("done", round(time.time() - t0, 1), "s")
