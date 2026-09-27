import os, numpy as np, pandas as pd
from util import thist, qtab
OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT, "t1_features.csv"))
print(qtab(df, ["mean", "sd", "median", "min", "max", "skew", "kurt", "dskew", "dkurt", "tailratio", "tailasym", "extasym", "maxabs_z",
                "nunique_frac", "int_frac", "decimals", "min_gap", "med_gap", "zero_frac", "mode_frac", "repeat_frac", "jb", "ad"]))
print("\nbounds: nonneg", df.nonneg.sum(), " in01", df.in01.sum(), " in[-1,1]", df.in_m11.sum())
print("decimals counts:\n", df.decimals.value_counts().sort_index().to_string())
print("int_frac==1:", (df.int_frac == 1).sum(), " int_frac>0.5:", (df.int_frac > 0.5).sum())
print("nunique_frac<0.5:", (df.nunique_frac < 0.5).sum(), " <0.9:", (df.nunique_frac < 0.9).sum(), " <0.99:", (df.nunique_frac < 0.99).sum())
print("zero_frac>0.01:", (df.zero_frac > 0.01).sum(), " >0:", (df.zero_frac > 0).sum(), " repeat_frac>0.01:", (df.repeat_frac > 0.01).sum())
print(thist(df["mean"], label="raw mean"))
print(thist(df["sd"], label="raw sd"))
print(thist(df["sd"], label="raw sd (log10)", log=True))
print(thist(df["kurt"], rng=(-2, 10), label="excess kurtosis (history)"))
print(thist(df["kurt"], label="excess kurtosis (log10 of kurt+3)", log=True) if False else thist(np.log10(df["kurt"] + 3), rng=(0, 2), label="log10(kurt+3)"))
print(thist(df["skew"], rng=(-3, 3), label="skew"))
print(thist(df["tailratio"], rng=(2, 8), label="tailratio (q99-q01)/IQR; Gauss=3.45"))
print(thist(df["tailasym"], rng=(0.3, 3), label="tail asym (q99-med)/(med-q01)"))
print(thist(df["maxabs_z"], rng=(2, 12), label="max |z|"))
print(thist(df["nunique_frac"], rng=(0, 1), label="unique fraction"))
print(thist(np.log10(df["min_gap"] + 1e-12), rng=(-12, 2), label="log10 min gap between sorted values"))
print(thist(df["dkurt"], rng=(-2, 10), label="excess kurtosis of first differences"))
print(thist(df["ad"], rng=(0, 20), label="Anderson-Darling stat (normal)"))
# joint: kurt vs tailratio
print("\ncorr kurt vs ad:", np.corrcoef(np.log(df["kurt"] + 3), np.log(df["ad"] + 1e-9))[0, 1])
# series with very high kurtosis: what do they look like
top = df.sort_values("kurt", ascending=False).head(15)
print(top[["id", "n", "mean", "sd", "skew", "kurt", "tailratio", "maxabs_z", "nunique_frac", "zero_frac", "dkurt"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
