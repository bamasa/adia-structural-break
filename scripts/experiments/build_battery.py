"""018: per-prefix two-sample battery, recomputed on a geometric cadence."""
import sys, time
sys.path.insert(0, "repo/src")
import numpy as np
import pandas as pd
from structural_break.features import Normalisation
from structural_break.stream import iter_series

t0 = time.time()
x = pd.read_parquet("structural-break-real-time-test/data/X_train.parquet")
y = pd.read_parquet("structural-break-real-time-test/data/y_train.parquet")
print(f"loading {time.time()-t0:.0f}s", flush=True)

QS = (0.05, 0.25, 0.50, 0.75, 0.95)

def acf(v, lag):
    if len(v) <= lag + 2:
        return 0.0
    a, b = v[:-lag], v[lag:]
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))

def two_sample(hist_sorted, hist_stats, prefix):
    """Drift-free two-sample features: history vs the online prefix so far."""
    p = prefix
    n = len(p)
    hq, hmean, hstd, hmad, hacf1, hacf2, hacf5, habs = hist_stats
    out = []
    pq = np.quantile(p, QS)
    out.extend((pq - hq).tolist())                                   # 5 quantiles
    out.append(float(p.mean() - hmean))                              # mean shift
    out.append(float(np.log((p.std() + 1e-9) / (hstd + 1e-9))))     # log std ratio
    c = p - p.mean()
    s2 = float((c ** 2).mean()) + 1e-12
    out.append(float((c ** 3).mean()) / s2 ** 1.5)                   # prefix skewness
    out.append(float((c ** 4).mean()) / s2 ** 2 - 3.0)               # prefix kurtosis
    # KS without length factors: sup difference of empirical CDFs, bounded in [0,1]
    pos = np.searchsorted(hist_sorted, np.sort(p), side="right") / len(hist_sorted)
    ks = float(np.max(np.abs(pos - (np.arange(1, n + 1) / n))))
    out.append(ks)
    pmad = float(np.median(np.abs(p - np.median(p)))) + 1e-9
    out.append(float(np.log(pmad / (hmad + 1e-9))))                  # log MAD ratio
    out.append(acf(p, 1) - hacf1)                                    # autocorrelation shifts
    out.append(acf(p, 2) - hacf2)
    out.append(acf(p, 5) - hacf5)
    ap = np.abs(p)
    out.append(float(ap.mean() - habs))                              # shift of mean |z|
    # Gaussian ranks of the prefix within the history
    u = (np.searchsorted(hist_sorted, p, side="left")
         + np.searchsorted(hist_sorted, p, side="right")) / 2.0
    u = np.clip((u + 0.5) / (len(hist_sorted) + 1.0), 1e-6, 1 - 1e-6)
    from math import sqrt
    g = np.sqrt(2) * erfinv_vec(2 * u - 1)
    out.append(float(g.mean()))                                      # mean rank
    out.append(float(np.log(g.std() + 1e-9)))                        # rank spread
    # fresh window: last quarter of the prefix against the history
    tail = p[-max(n // 4, 5):]
    out.append(float(tail.mean() - hmean))
    out.append(float(np.log((tail.std() + 1e-9) / (hstd + 1e-9))))
    # internal contrast: second half of the prefix against the first
    half = n // 2
    if half >= 3:
        out.append(float(p[half:].mean() - p[:half].mean()))
        out.append(float(np.log((p[half:].std() + 1e-9) / (p[:half].std() + 1e-9))))
    else:
        out.extend([0.0, 0.0])
    return out  # 20 features per view

from scipy.special import erfinv as _erfinv
def erfinv_vec(v):
    return _erfinv(v)

def hist_summary(z):
    zs = np.sort(z)
    c = z - z.mean()
    return (zs, np.quantile(z, QS), float(z.mean()), float(z.std()),
            float(np.median(np.abs(z - np.median(z)))),
            acf(z, 1), acf(z, 2), acf(z, 5), float(np.abs(z).mean()))

def pack(z):
    s = hist_summary(z)
    return s[0], (s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8])

rows, count = [], 0
for sid, hist, online, labels in iter_series(x, y):
    feats_views = []
    for transform in (lambda v: v, np.arcsinh):
        h = transform(np.asarray(hist, dtype="float64"))
        o = transform(np.asarray(online, dtype="float64"))
        norm = Normalisation.fit(h)
        n_h = len(h)
        z_h = np.asarray([norm.standardise(float(v), i - n_h) for i, v in enumerate(h)])
        z_o = np.asarray([norm.standardise(float(v), i) for i, v in enumerate(o)])
        feats_views.append((pack(z_h), z_o))
    next_scan, current = 1, [0.0] * 40
    for step in range(len(online)):
        if step + 1 >= next_scan:
            next_scan = max(next_scan + 1, int(next_scan * 1.12))
            current = []
            for (hist_sorted, hstats), z_o in feats_views:
                current.extend(two_sample(hist_sorted, hstats, z_o[: step + 1]))
        rows.append(list(current))
    count += 1
    if count % 1000 == 0:
        print(f"  {count} series, {time.time()-t0:.0f}s", flush=True)

a = np.asarray(rows, dtype="float32")
np.save("B40.npy", a)
print(f"done {time.time()-t0:.0f}s: {a.shape}", flush=True)
