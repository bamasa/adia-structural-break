"""092: entry/exit on top of the ensemble score trajectory — rules (no training)."""
import sys, numpy as np
sys.path.insert(0, "repo/src")
from structural_break.combiners import split_by_series, ts_auc
g = np.load("G40.npy"); y = np.load("Y40.npy"); s = np.load("S40.npy")
m2 = split_by_series(g, folds=5, seed=0) == 2
gf, yf, sf = g[m2], y[m2], s[m2]
sig = lambda a: 1/(1+np.exp(-a.astype("float64")))
trees = 0.7*sig(np.load("fold2_rank_bocpd_200.npy")) + 0.3*np.load("fold2_clf_bocpdh50_206.npy").astype("float64")
net = np.mean([np.load(f"fold2_sig_nets_aug3_member_p{i}.pt.npy") for i in range(6)] + [np.load(f"fold2_sig_nets_aug3_last_member_z{i}.pt.npy") for i in range(6)], 0)
base = 0.45*trees + 0.55*net
np.save("fold2_base30.npy", base)
starts = np.flatnonzero(np.concatenate([[True], gf[1:] != gf[:-1]])); bounds = np.append(starts, len(gf))
print(f"base (#30 on fold 2): {ts_auc(base, yf, sf):.4f}")

def apply(rule):
    out = np.empty_like(base)
    for a, b in zip(starts, bounds[1:]):
        out[a:b] = rule(base[a:b])
    return out

def peak_hold(decay):
    def r(x):
        o = np.empty_like(x); h = 0.0
        for t, v in enumerate(x):
            h = max(v, h * decay); o[t] = h
        return o
    return r

def entry_exit(decay, w, beta):
    """Entry: hold the peak with decay. Exit: if the mean of the last w steps < beta·held value — reset to the mean."""
    def r(x):
        o = np.empty_like(x); h = 0.0
        for t, v in enumerate(x):
            h = max(v, h * decay)
            recent = x[max(0, t - w + 1):t + 1].mean()
            if recent < beta * h:
                h = recent
            o[t] = h
        return o
    return r

def ema_mix(alpha, wmix):
    def r(x):
        o = np.empty_like(x); e = x[0]
        for t, v in enumerate(x):
            e = alpha * e + (1 - alpha) * v; o[t] = (1 - wmix) * v + wmix * e
        return o
    return r

def cummax_mix(wmix):
    return lambda x: (1 - wmix) * x + wmix * np.maximum.accumulate(x)

def running_mean_mix(w, wmix):
    def r(x):
        c = np.cumsum(np.insert(x, 0, 0.0)); n = np.arange(1, len(x) + 1); lo = np.maximum(0, n - w)
        rm = (c[n] - c[lo]) / (n - lo); return (1 - wmix) * x + wmix * rm
    return r

res = {}
for d in (0.99, 0.995, 0.999):
    res[f"peak hold decay={d}"] = ts_auc(apply(peak_hold(d)), yf, sf)
for d in (0.995, 0.999):
    for w in (20, 50, 100):
        for beta in (0.7, 0.85):
            res[f"entry/exit decay={d} w={w} beta={beta}"] = ts_auc(apply(entry_exit(d, w, beta)), yf, sf)
for a in (0.9, 0.97):
    for wm in (0.3, 0.5):
        res[f"EMA alpha={a} share={wm}"] = ts_auc(apply(ema_mix(a, wm)), yf, sf)
for wm in (0.3, 0.5):
    res[f"cummax share={wm}"] = ts_auc(apply(cummax_mix(wm)), yf, sf)
for w in (20, 60, 200):
    for wm in (0.3, 0.5):
        res[f"running mean w={w} share={wm}"] = ts_auc(apply(running_mean_mix(w, wm)), yf, sf)
for k, v in sorted(res.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {v:.4f}  {k}")
print(f"  ...worst: {min(res.values()):.4f}")
