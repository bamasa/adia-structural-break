"""109b: what changes in the 40% of "invisible" breaks — a wide screen of before/after statistics.

We take the fold-2 series where the break does not change the mean, variance, AR(1) or kurtosis (windows of 300),
and look for a statistic that still tells them apart. Control: series without a break, cut
at a random point: any statistic is noisy, only the excess over the control is significant.
"""
import os, sys, numpy as np, pandas as pd
WIN = int(os.environ.get("WIN", "300"))
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series
g = np.load("G40.npy"); m2 = split_by_series(g, folds=5, seed=0) == 2; gf = g[m2]
D = "structural-break-real-time-test/data/"
X = pd.read_parquet(D + "X_train.parquet"); yi = pd.read_parquet(D + "y_train_index.parquet")
rng = np.random.default_rng(0)
def stats(v):
    v = np.asarray(v, float); z = (v - v.mean()) / (v.std() + 1e-12); n = len(z)
    ac = lambda k: np.corrcoef(z[:-k], z[k:])[0, 1] if n > k + 5 else 0.0
    sq = z ** 2; av = np.abs(z); d = np.diff(z)
    sp = np.abs(np.fft.rfft(z * np.hanning(n)))[1:] ** 2; sp = sp / (sp.sum() + 1e-12)
    k = len(sp)
    run = (np.diff(np.sign(z)) != 0).mean()
    return dict(
        ac1=ac(1), ac3=ac(3), ac10=ac(10), ac30=ac(30),
        vol_ac1=np.corrcoef(sq[:-1], sq[1:])[0, 1], abs_ac5=np.corrcoef(av[:-5], av[5:])[0, 1],
        diff_sd=d.std(), diff_kurt=pd.Series(d).kurt(), rev=np.corrcoef(d[:-1], d[1:])[0, 1],
        sp_low=sp[:k//8].sum(), sp_mid=sp[k//8:k//3].sum(), sp_high=sp[2*k//3:].sum(),
        sp_ent=-(sp * np.log(sp + 1e-12)).sum(), peak=sp.max(),
        q95=np.quantile(av, 0.95), q99=np.quantile(av, 0.99), zero=run,
        nonlin=np.corrcoef(z[1:], z[:-1] ** 2)[0, 1] if n > 6 else 0.0,
        longvar=np.var([z[i:i+50].mean() for i in range(0, n - 50, 10)]) if n > 100 else 0.0)
inv, ctl = [], []
starts = np.flatnonzero(np.concatenate([[True], gf[1:] != gf[:-1]]))
for a in starts:
    sid = int(gf[a]); tau = int(yi.loc[sid, "tau_index"])
    part = X.loc[sid]; v = part.value.to_numpy("float64"); p = part.period.to_numpy()
    hist, online = v[p == 1], v[p == 2]
    if tau >= 0:
        if len(online) - tau < WIN or tau + len(hist) < WIN: continue
        pre = np.concatenate([hist, online[:tau]])[-WIN:]; post = online[tau:][:WIN]
        A, B = stats(pre), stats(post)
        if abs(B["q95"] - A["q95"]) > 9e9: continue
        dm = abs(post.mean() - pre.mean()) / (pre.std() + 1e-12); vr = post.std() / (pre.std() + 1e-12)
        if dm > 0.3 or vr < 0.85 or vr > 1.18 or abs(B["ac1"] - A["ac1"]) > 0.12 or abs(pd.Series(post).kurt() - pd.Series(pre).kurt()) > 1.5:
            continue
        inv.append({k: B[k] - A[k] for k in A})
    else:
        if len(online) < WIN: continue
        cut = len(online) // 2
        pre = np.concatenate([hist, online[:cut]])[-WIN:]; post = online[cut:][:WIN]
        A, B = stats(pre), stats(post)
        ctl.append({k: B[k] - A[k] for k in A})
I, C = pd.DataFrame(inv), pd.DataFrame(ctl)
print(f"window {WIN}: \"invisible\" breaks {len(I)}, control cuts: {len(C)}\n")
print(f"{'statistic':12s} {'σ break':>8s} {'σ ctrl':>8s} {'ratio':>10s} {'share >2σ ctrl':>16s}")
res = []
for c in I.columns:
    sb, sc = I[c].std(), C[c].std()
    if not np.isfinite(sb) or not np.isfinite(sc) or sc < 1e-12: continue
    frac_b = (I[c].abs() > 2 * sc).mean(); frac_c = (C[c].abs() > 2 * sc).mean()
    res.append((sb / sc, c, sb, sc, frac_b, frac_c))
for ratio, c, sb, sc, fb, fc in sorted(res, reverse=True):
    print(f"{c:12s} {sb:8.4f} {sc:8.4f} {ratio:10.2f} {fb:8.2f} vs {fc:.2f}")
