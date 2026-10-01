"""109: where the ensemble is blind — breakdown by break type.

For every fold-2 series with a break, compute what exactly changed (windows of 300 before/after tau):
mean shift, variance ratio, AR(1) change, shape change (skewness/kurtosis).
Then measure the per-series AUC of ensemble #30 within each type: the series with a break against all
series without a break at the same steps. Shows which kind of change we do not see.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
g = np.load("G40.npy"); y = np.load("Y40.npy"); s = np.load("S40.npy")
m2 = split_by_series(g, folds=5, seed=0) == 2; gf, yf, sf = g[m2], y[m2], s[m2]
base = np.load("fold2_base30.npy")
D = "structural-break-real-time-test/data/"
X = pd.read_parquet(D + "X_train.parquet"); yi = pd.read_parquet(D + "y_train_index.parquet")
starts = np.flatnonzero(np.concatenate([[True], gf[1:] != gf[:-1]])); bounds = np.append(starts, len(gf))
def ar1(v): return np.corrcoef(v[:-1], v[1:])[0, 1] if len(v) > 5 else np.nan
rows = []
for a, b in zip(starts, bounds[1:]):
    sid = int(gf[a]); tau = int(yi.loc[sid, "tau_index"])
    part = X.loc[sid]; v = part.value.to_numpy("float64"); p = part.period.to_numpy()
    hist, online = v[p == 1], v[p == 2]
    r = dict(a=a, b=b, sid=sid, tau=tau, n=b - a)
    if tau >= 0 and (b - a) - tau >= 30:
        pre = np.concatenate([hist, online[:tau]])[-300:]; post = online[tau:][:300]
        r["dmean"] = abs(post.mean() - pre.mean()) / (pre.std() + 1e-12)
        r["vratio"] = post.std() / (pre.std() + 1e-12)
        r["dar1"] = abs(ar1(post) - ar1(pre))
        r["dshape"] = abs(pd.Series(post).kurt() - pd.Series(pre).kurt())
    rows.append(r)
df = pd.DataFrame(rows)
brk = df[df.tau >= 0].dropna(subset=["dmean"]).copy()
def kind(r):
    k = []
    if r.dmean > 0.3: k.append("mean")
    if r.vratio < 0.85 or r.vratio > 1.18: k.append("variance")
    if r.dar1 > 0.12: k.append("dependence")
    if r.dshape > 1.5: k.append("shape")
    return "+".join(k) if k else "none evident"
brk["kind"] = brk.apply(kind, axis=1)
brk[["a", "b", "sid", "tau", "n", "dmean", "vratio", "dar1", "dshape", "kind"]].to_csv("fold2_break_kinds.csv", index=False)
print(brk.kind.value_counts().to_string())
